// Native backend for bigWigOperationsR (scale, merge, ...). bigWig-only, so
// it links libBigWig + libdeflate + zlib but NOT htslib. Shares the streaming
// writer machinery with the coverage tool via streaming.hpp.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "libBigWig/bigWig.h"
#include "parallel.hpp"
#include "numeric.hpp"
#include "bigwig_reader.hpp"
#include <memory>
#include "streaming.hpp"
#include "writer_common.hpp"
#include "handles.hpp"

namespace py = pybind11;

namespace {
using dtp::OutFmt;
using dtp::StreamWriter;
using dtp::Window;
using dtp::run_windows;
using dtp::validate_writer_options;
using dtp::checked_float;
using dtp::BigWigPtr;

// Copy a bigWig, multiplying every value by `factor`. Used e.g. to flip a
// reverse-strand track (factor = -1). Reads variable-span intervals and writes
// them back scaled, so it preserves the input's resolution exactly.
void bigwig_scale(const std::string &in_path, const std::string &out_path,
                  double factor, int max_zooms, int compression_level) {
    if (!std::isfinite(factor))
        throw std::runtime_error("factor must be finite");
    validate_writer_options(max_zooms, compression_level);
    dtp::BigWigLibraryScope library;
    BigWigPtr ifp(bwOpen(in_path.c_str(), nullptr, "r"));
    if (!ifp) throw std::runtime_error("could not open bigWig: " + in_path);

    const int64_t n = ifp->cl->nKeys;
    std::vector<const char *> names(n);
    std::vector<uint32_t> lens(n);
    for (int64_t i = 0; i < n; ++i) { names[i] = ifp->cl->chrom[i]; lens[i] = ifp->cl->len[i]; }

    BigWigPtr ofp(bwOpen(out_path.c_str(), nullptr, "w"));
    if (!ofp) throw std::runtime_error("could not open bigWig: " + out_path);

    bool ok = (bwCreateHdr(ofp.get(), max_zooms) == 0);
    if (ok) ofp->writeBuffer->compressLevel = compression_level;
    if (ok) {
        chromList_t *cl = bwCreateChromList(names.data(), lens.data(), n);
        if (!cl) ok = false;
        else { ofp->cl = cl; if (bwWriteHdr(ofp.get()) != 0) ok = false; }
    }
    // Read each chromosome in bounded chunks so memory stays O(chunk) rather than
    // O(chromosome) (a dense binSize-1 track would otherwise load ~1 interval/base).
    // Each interval is written once, by the chunk containing its start, so the output
    // intervals are byte-identical to a whole-chromosome pass.
    const uint32_t CHUNK = 8u << 20;  // 8 Mbp
    std::vector<float> sv;
    std::vector<const char *> chr;
    std::vector<uint32_t> st, en;
    std::string value_error;
    bool omitted_nonfinite = false;
    for (int64_t i = 0; ok && i < n; ++i) {
        for (uint64_t cs64 = 0; ok && cs64 < lens[i]; cs64 += CHUNK) {
            dtp::check_python_signals();
            const uint32_t cs = static_cast<uint32_t>(cs64);
            const uint32_t ce = static_cast<uint32_t>(
                std::min<uint64_t>(cs64 + CHUNK, lens[i]));
            auto iv = dtp::read_bigwig_intervals(ifp.get(), in_path, names[i], cs, ce);
            if (iv && iv->l) {
                sv.clear(); st.clear(); en.clear();
                for (uint32_t j = 0; j < iv->l; ++j) {
                    if (iv->start[j] < cs) continue;  // owned by the previous chunk
                    if (!std::isfinite(iv->value[j])) {
                        omitted_nonfinite = true;
                        continue;
                    }
                    st.push_back(iv->start[j]);
                    en.push_back(iv->end[j]);
                    try {
                        sv.push_back(checked_float(
                            static_cast<double>(iv->value[j]) * factor,
                            "scaled bigWig value"));
                    } catch (const std::exception &error) {
                        value_error = error.what();
                        ok = false;
                        break;
                    }
                }
                if (ok && !st.empty()) {
                    chr.assign(st.size(), names[i]);
                    if (bwAddIntervals(ofp.get(), chr.data(), st.data(), en.data(),
                                       sv.data(), static_cast<uint32_t>(st.size())) != 0)
                        ok = false;
                }
            }
        }
    }

    if (bwCloseChecked(ofp.release())) ok = false;
    if (omitted_nonfinite)
        std::fprintf(stderr,
            "Warning: omitted non-finite values from input bigWig '%s'.\n",
            in_path.c_str());
    if (!value_error.empty()) throw std::runtime_error(value_error);
    if (!ok) throw std::runtime_error("failed writing bigWig: " + out_path);
}

// Per-thread merge worker: its own read handle per input bigWig.
struct MergeWorker {
    std::vector<BigWigPtr> h;
};

// Compute one window's merged per-bin value. For each input, read its intervals
// over the window ONCE (bwGetOverlappingIntervals) and distribute them across the
// output bins in a single linear pass -> O(intervals + nbins), not the O(nbins ×
// intervals) of per-bin bwStats. Zero-valued (and NaN) intervals are skipped, so
// sparse tracks with big zero-runs are cheap. Each input's per-bin mean (weighted
// sum / actual bin width) is accumulated; a chrom absent from an input yields NULL
// -> contributes 0.
inline void merge_window(const std::vector<BigWigPtr> &hs, const Window &w,
                         const std::vector<std::string> &paths,
                         const std::vector<const char *> &names, uint32_t chrom_len,
                         uint32_t bin_size, bool do_mean, int nin, double scale,
                         std::vector<std::atomic<bool>> &omitted_nonfinite,
                         std::vector<std::vector<float>> &out) {
    const uint32_t nbins = w.bin1 - w.bin0;
    const int64_t ws_bp = static_cast<int64_t>(w.bin0) * bin_size;
    int64_t we_bp = static_cast<int64_t>(w.bin1) * bin_size;
    if (we_bp > chrom_len) we_bp = chrom_len;
    const char *nm = names[w.tid];
    const int64_t bs = bin_size, b0 = w.bin0;

    std::vector<dtp::CompensatedSum> acc(nbins);
    std::vector<dtp::CompensatedSum> bin_sum(nbins);
    for (std::size_t input = 0; input < hs.size(); ++input) {
        bigWigFile_t *h = hs[input].get();
        std::fill(bin_sum.begin(), bin_sum.end(), dtp::CompensatedSum{});
        constexpr int64_t max_read_bases = 262144;
        for (int64_t chunk_start = ws_bp; chunk_start < we_bp; chunk_start += max_read_bases) {
        const int64_t chunk_end = std::min(we_bp, chunk_start + max_read_bases);
        auto iv = dtp::read_bigwig_intervals(h, paths[input], nm,
            static_cast<uint32_t>(chunk_start), static_cast<uint32_t>(chunk_end));
        if (!iv) continue;
        for (uint32_t j = 0; j < iv->l; ++j) {
            const float v = iv->value[j];
            if (!std::isfinite(v)) {
                omitted_nonfinite[input].store(true, std::memory_order_relaxed);
                continue;
            }
            if (v == 0.0f) continue;
            int64_t s = iv->start[j], e = iv->end[j];
            if (s < chunk_start) s = chunk_start;
            if (e > chunk_end) e = chunk_end;
            if (e <= s) continue;
            const int64_t fb = s / bs - b0, lb = (e - 1) / bs - b0;
            if (fb == lb) {
                bin_sum[fb].add(static_cast<double>(v) * (e - s));
            } else {
                bin_sum[fb].add(static_cast<double>(v) * ((fb + b0 + 1) * bs - s));
                bin_sum[lb].add(static_cast<double>(v) * (e - (lb + b0) * bs));
                for (int64_t b = fb + 1; b < lb; ++b)
                    bin_sum[b].add(static_cast<double>(v) * bs);
            }
        }
        }
        for (uint32_t i = 0; i < nbins; ++i) {
            if (bin_sum[i].value() != 0.0) {
                const int64_t abin = b0 + i;
                int64_t be = (abin + 1) * bs;
                if (be > chrom_len) be = chrom_len;
                acc[i].add(bin_sum[i].value() / (be - abin * bs));
            }
        }
    }
    const double denom = do_mean ? static_cast<double>(nin) : 1.0;
    out.assign(1, std::vector<float>(nbins));
    for (uint32_t i = 0; i < nbins; ++i)
        out[0][i] = checked_float(acc[i].value() / denom * scale,
                                  "merged bigWig value");
}

// Merge N bigWigs by summing/averaging their per-bin signal. Output chroms are the
// UNION of input chroms in first-appearance order (natural, not lexical -- avoids
// the reader chr10-inaccessibility bug); shared chroms must agree on length.
void bigwig_merge(py::list in_paths, const std::string &out_path,
                  const std::string &out_format, const std::string &combine,
                  uint32_t bin_size, double scale, int threads, int max_zooms,
                  int compression_level) {
    if (bin_size == 0) throw std::runtime_error("binSize must be > 0");
    if (!std::isfinite(scale)) throw std::runtime_error("scale must be finite");
    if (threads < 0) throw std::runtime_error("threads must be >= 0");
    validate_writer_options(max_zooms, compression_level);
    bool do_mean;
    if (combine == "mean") do_mean = true;
    else if (combine == "sum") do_mean = false;
    else throw std::runtime_error("combine must be sum|mean");
    OutFmt fmt;
    if (out_format == "bigwig") fmt = OutFmt::BigWig;
    else if (out_format == "bedgraph") fmt = OutFmt::BedGraph;
    else if (out_format == "bedgraph.gz") fmt = OutFmt::BedGraphGz;
    else throw std::runtime_error("out_format must be bigwig|bedgraph|bedgraph.gz");

    std::vector<std::string> paths;
    for (auto p : in_paths) paths.push_back(p.cast<std::string>());
    if (paths.empty()) throw std::runtime_error("merge needs at least one input");
    const int nin = static_cast<int>(paths.size());

    const bool is_bw = (fmt == OutFmt::BigWig);
    dtp::BigWigLibraryScope library;  // also for reads

    // Build the union chrom list (first-appearance order), validating lengths.
    std::vector<std::string> names_s;
    std::vector<uint32_t> lengths;
    std::unordered_map<std::string, int> idx;
    for (const auto &p : paths) {
        BigWigPtr f(bwOpen(p.c_str(), nullptr, "r"));
        if (!f) throw std::runtime_error("could not open bigWig: " + p);
        for (int64_t i = 0; i < f->cl->nKeys; ++i) {
            const std::string nm = f->cl->chrom[i];
            const uint32_t ln = f->cl->len[i];
            auto it = idx.find(nm);
            if (it == idx.end()) {
                idx[nm] = static_cast<int>(names_s.size());
                names_s.push_back(nm); lengths.push_back(ln);
            } else if (lengths[it->second] != ln) {
                throw std::runtime_error("chromosome '" + nm + "' has different lengths "
                                         "across inputs (" +
                                         std::to_string(lengths[it->second]) + " vs " +
                                         std::to_string(ln) + ")");
            }
        }
    }
    const int n_targets = static_cast<int>(names_s.size());
    std::vector<const char *> names(n_targets);
    for (int t = 0; t < n_targets; ++t) names[t] = names_s[t].c_str();

    // Open the output writer.
    BigWigPtr bw;
    dtp::FilePtr fh;
    auto finalize = [&]() {
        bool failed = false;
        if (bw) failed |= bwCloseChecked(bw.release()) != 0;
        if (fh) failed |= std::fclose(fh.release()) != 0;
        if (failed) throw std::runtime_error("failed finalizing output: " + out_path);
    };
    if (is_bw) {
        bw.reset(bwOpen(out_path.c_str(), nullptr, "w"));
        if (!bw) throw std::runtime_error("could not open bigWig: " + out_path);
        bool ok = (bwCreateHdr(bw.get(), max_zooms) == 0);
        if (ok) bw->writeBuffer->compressLevel = compression_level;
        if (ok) {
            chromList_t *cl = bwCreateChromList(names.data(), lengths.data(), n_targets);
            if (!cl) ok = false;
            else { bw->cl = cl; if (bwWriteHdr(bw.get()) != 0) ok = false; }
        }
        if (!ok) throw std::runtime_error("bigWig header write failed");
    } else {
        fh.reset(std::fopen(out_path.c_str(), "wb"));
        if (!fh) throw std::runtime_error("could not open output: " + out_path);
    }
    std::vector<StreamWriter> writers;
    writers.push_back(StreamWriter{fmt, bw.get(), fh.get(), compression_level, names, lengths, bin_size});

    // Tile the union chroms into windows.
    std::vector<Window> windows;
    for (int t = 0; t < n_targets; ++t) {
        const uint32_t nb = static_cast<uint32_t>(
            (static_cast<uint64_t>(lengths[t]) + bin_size - 1) / bin_size);
        for (uint32_t b = 0; b < nb;) {
            const uint32_t end = static_cast<uint32_t>(
                std::min<uint64_t>(static_cast<uint64_t>(b) + (1u << 18), nb));
            windows.push_back({t, b, end});
            b = end;
        }
    }
    const std::size_t nthreads =
        dtp::thread_count(threads, std::max<std::size_t>(windows.size(), 1));
    std::vector<std::atomic<bool>> omitted_nonfinite(paths.size());
    for (auto &flag : omitted_nonfinite)
        flag.store(false, std::memory_order_relaxed);

    run_windows(
            windows, nthreads, 1, writers,
            [&]() {
                MergeWorker w;
                for (const auto &p : paths) {
                    BigWigPtr f(bwOpen(p.c_str(), nullptr, "r"));
                    if (!f) throw std::runtime_error("worker could not open " + p);
                    w.h.push_back(std::move(f));
                }
                return w;
            },
            [&](MergeWorker &w, const Window &win, std::vector<std::vector<float>> &o) {
                merge_window(w.h, win, paths, names, lengths[win.tid], bin_size, do_mean,
                             nin, scale, omitted_nonfinite, o);
            });
    writers[0].finish();
    for (std::size_t input = 0; input < paths.size(); ++input)
        if (omitted_nonfinite[input].load(std::memory_order_relaxed))
            std::fprintf(stderr,
                "Warning: omitted non-finite values from input bigWig '%s'.\n",
                paths[input].c_str());
    finalize();
}

py::dict bigwig_info(const std::string &path, bool statistics) {
    dtp::BigWigLibraryScope library;
    BigWigPtr file(bwOpen(path.c_str(), nullptr, "r"));
    if (!file || !file->cl || !file->hdr) throw std::runtime_error("could not open bigWig: " + path);
    py::dict chroms, result;
    for (int64_t c = 0; c < file->cl->nKeys; ++c) chroms[py::str(file->cl->chrom[c])] = file->cl->len[c];
    result["chroms"] = chroms;
    result["version"] = file->hdr->version; result["nLevels"] = file->hdr->nLevels;
    if (!statistics) return result;
    std::uint64_t count = 0;
    double low = std::numeric_limits<double>::infinity(), high = -low;
    double mean = dtp::missing_value(), std = dtp::missing_value();
    {
        py::gil_scoped_release release;
        dtp::CompensatedSum total, squares;
        for (int pass = 0; pass < 2; ++pass) {
            for (int64_t c = 0; c < file->cl->nKeys; ++c) {
                const auto length = file->cl->len[c];
                for (std::uint64_t start = 0; start < length; start += 262144) {
                    dtp::check_python_signals();
                    const auto end = std::min<std::uint64_t>(length, start + 262144);
                    auto iv = dtp::read_bigwig_intervals(
                        file.get(), path, file->cl->chrom[c], start, end);
                    if (!iv) continue;
                    for (std::uint32_t i = 0; i < iv->l; ++i) {
                        const double value = iv->value[i];
                        if (!std::isfinite(value)) continue;
                        const auto first = std::max<std::uint64_t>(start, iv->start[i]);
                        const auto last = std::min<std::uint64_t>(end, iv->end[i]);
                        if (last <= first) continue;
                        const auto width = last - first;
                        if (pass == 0) {
                            count += width; total.add(value * width);
                            low = std::min(low, value); high = std::max(high, value);
                        } else { const double delta = value - mean; squares.add(delta * delta * width); }
                    }
                }
            }
            if (pass == 0 && count) mean = total.value() / count;
        }
        if (count) std = std::sqrt(squares.value() / count);
    }
    result["nBasesCovered"] = count; result["mean"] = mean; result["std"] = std;
    result["minVal"] = count ? low : dtp::missing_value();
    result["maxVal"] = count ? high : dtp::missing_value();
    return result;
}

}  // namespace

PYBIND11_MODULE(_bigwig, m) {
    m.def("bigwig_info", &bigwig_info, py::arg("path"), py::arg("statistics") = true);
    m.doc() = "Native bigWigOperationsR backend (scale, merge).";
    m.def("bigwig_scale", &bigwig_scale, py::arg("in_path"), py::arg("out_path"),
          py::arg("factor"), py::arg("max_zooms") = 10,
          py::arg("compression_level") = -1,
          "Copy a bigWig scaling every value by `factor` (preserves resolution).");
    m.def("bigwig_merge", &bigwig_merge, py::arg("in_paths"), py::arg("out_path"),
          py::arg("out_format") = "bigwig", py::arg("combine") = "sum",
          py::arg("bin_size") = 1, py::arg("scale") = 1.0, py::arg("threads") = 1,
          py::arg("max_zooms") = 10, py::arg("compression_level") = -1,
          "Merge N bigWigs (sum|mean) over the union of chroms (missing = 0).");
}
