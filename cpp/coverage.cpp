// Native BAM coverage, filtering, normalization and bounded-window output.
// The production writer is shared by all coverage operations.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "parallel.hpp"
#include "numeric.hpp"

#include <cstdio>

#include <htslib/hts.h>
#include <htslib/sam.h>
#include <libdeflate.h>

#include "libBigWig/bigWig.h"
#include "streaming.hpp"
#include "writer_common.hpp"
#define DTP_HTSLIB_HANDLES
#include "handles.hpp"

namespace py = pybind11;

namespace {
inline int checked_bam_status(int status) {
    if (status < -1) throw std::runtime_error("error reading BAM data (truncated or corrupt input)");
    return status;
}

using dtp::OutFmt;
using dtp::StreamWriter;
using dtp::Window;
using dtp::run_windows;
using dtp::validate_writer_options;
using dtp::checked_float;
using dtp::HtsFilePtr;
using dtp::HdrPtr;
using dtp::IdxPtr;
using dtp::BamRecordPtr;
using dtp::IteratorPtr;
using dtp::BigWigPtr;
using dtp::FilePtr;

constexpr uint32_t MAX_WINDOW_BINS = 1u << 18;
constexpr uint64_t TARGET_WINDOW_BP = 1u << 20;

inline void poll_coverage_work(uint64_t count) {
    if ((count & 4095u) != 0) return;
    if (dtp::running_in_parallel_worker()) {
        if (dtp::interruption_requested())
            throw std::runtime_error("coverage computation cancelled");
    } else {
        dtp::check_python_signals();
    }
}

uint32_t coverage_window_bins(uint32_t bin_size, int requested = 0) {
    if (bin_size == 0)
        throw std::runtime_error("binSize must be > 0");
    if (requested < 0)
        throw std::runtime_error("window_size must be >= 0");
    const uint64_t bp_limited = std::max<uint64_t>(
        1, TARGET_WINDOW_BP / static_cast<uint64_t>(bin_size));
    const uint64_t requested_limit = requested > 0
        ? static_cast<uint64_t>(requested) : MAX_WINDOW_BINS;
    return static_cast<uint32_t>(std::max<uint64_t>(
        1, std::min<uint64_t>({MAX_WINDOW_BINS, bp_limited,
                               requested_limit})));
}

// Version string of the vendored htslib -- a trivial proof the library is
// actually linked and callable.
std::string htslib_version() {
    return std::string(hts_version());
}

inline bool readable_file(const std::string &path) {
    std::FILE *handle = std::fopen(path.c_str(), "rb");
    if (!handle) return false;
    std::fclose(handle);
    return true;
}

inline std::string replace_extension(const std::string &path,
                                     const std::string &extension) {
    const std::size_t slash = path.find_last_of("/\\");
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos ||
        (slash != std::string::npos && dot < slash))
        return path + extension;
    return path.substr(0, dot) + extension;
}

// Resolve the local sidecar using the same candidate order as the selected
// backend. The CLI includes this exact file in its input/output alias check
// before any writer is opened.
std::string resolve_bam_index_path(const std::string &bam_path) {
    std::vector<std::string> candidates;
    if (bam_path.size() >= 5 &&
        bam_path.compare(bam_path.size() - 5, 5, ".cram") == 0) {
        candidates = {bam_path + ".crai",
                      replace_extension(bam_path, ".crai")};
    } else {
        candidates = {bam_path + ".csi", replace_extension(bam_path, ".csi"),
                      bam_path + ".bai", replace_extension(bam_path, ".bai")};
    }
    for (const auto &candidate : candidates)
        if (readable_file(candidate)) return candidate;
    return "";
}

// Per-reference index statistics, read straight from the .bai/.csi with no
// alignment scan. Returns one dict per target plus the count of placed-but-
// unmapped-at-no-coordinate reads. This is the data the corrected
// normalization denominator is built from (mapped-alignment count per target,
// honoring --ignoreForNormalization by dropping chromosomes here).
py::dict bam_index_stats(const std::string &path) {
    HtsFilePtr file;
    file.reset(hts_open(path.c_str(), "r"));
    if (!file.get()) {
        throw std::runtime_error("could not open BAM: " + path);
    }
    if (file.get()->format.category != sequence_data) {
        throw std::runtime_error("not a sequence (BAM/SAM/CRAM) file: " + path);
    }

    HdrPtr hdr;
    hdr.reset(sam_hdr_read(file.get()));
    if (!hdr.get()) {
        throw std::runtime_error("could not read header: " + path);
    }

    IdxPtr idx;
    idx.reset(sam_index_load(file.get(), path.c_str()));
    if (!idx.get()) {
        throw std::runtime_error(
            "no usable index for '" + path +
            "'. The BAM must be coordinate-sorted and indexed "
            "(samtools index).");
    }

    const int n = sam_hdr_nref(hdr.get());
    py::list targets;
    uint64_t total_mapped = 0;
    uint64_t total_unmapped = 0;
    for (int tid = 0; tid < n; ++tid) {
        uint64_t mapped = 0, unmapped = 0;
        // hts_idx_get_stat returns -1 when a target has no index metadata
        // (e.g. no reads); treat that as zeros rather than an error.
        if (hts_idx_get_stat(idx.get(), tid, &mapped, &unmapped) < 0) {
            mapped = 0;
            unmapped = 0;
        }
        total_mapped += mapped;
        total_unmapped += unmapped;

        py::dict t;
        t["name"] = std::string(sam_hdr_tid2name(hdr.get(), tid));
        t["length"] = static_cast<uint64_t>(sam_hdr_tid2len(hdr.get(), tid));
        t["mapped"] = mapped;
        t["unmapped"] = unmapped;
        targets.append(std::move(t));
    }

    py::dict out;
    out["targets"] = targets;
    out["n_targets"] = n;
    out["total_mapped"] = total_mapped;      // NB: includes secondary/supplementary
    out["total_unmapped"] = total_unmapped;
    out["n_no_coor"] = static_cast<uint64_t>(hts_idx_get_n_no_coor(idx.get()));
    return out;
}

// --- P1: single-thread binned coverage -------------------------------------
//
// Coverage is built genomecov-style: a difference-array sweep over each read's
// *aligned blocks* (CIGAR M/=/X; D and N gap the block, matching pysam
// get_blocks and Bioconductor coverage()), giving true per-base depth. Bins
// aggregate that depth:
//   mean  = sum(depth over bin) / bin_width   (Bioconductor binnedAverage /
//           tileGenome; == deeptools at binSize 1)   [default]
//   sum   = sum(depth over bin)               (total coverage in the bin)
//   count = #reads overlapping the bin, deduped per read (bamCoverage parity)
//
// mean/sum use the O(1)-per-block bin-resolution accumulator below (no per-base
// array): edge bins get their partial overlap, fully-covered interior bins get
// binSize via a bin-level range delta. count uses a per-read bin-dedup delta.

enum class Agg { Mean, Sum, Count };

// SAM FLAG bits, named per the SAM spec / Picard "explain flags"
// (broadinstitute.github.io/picard/explain-flags.html). Own constants rather
// than htslib's BAM_F* so the filtering code reads self-documenting.
constexpr uint16_t SAM_READ_PAIRED    = 0x001;  // read paired
constexpr uint16_t SAM_PROPER_PAIR    = 0x002;  // read mapped in proper pair
constexpr uint16_t SAM_READ_UNMAPPED  = 0x004;  // read unmapped
constexpr uint16_t SAM_MATE_UNMAPPED  = 0x008;  // mate unmapped
constexpr uint16_t SAM_READ_REVERSE   = 0x010;  // read reverse strand
constexpr uint16_t SAM_MATE_REVERSE   = 0x020;  // mate reverse strand
constexpr uint16_t SAM_FIRST_IN_PAIR  = 0x040;  // first in pair (read1)
constexpr uint16_t SAM_SECOND_IN_PAIR = 0x080;  // second in pair (read2)
constexpr uint16_t SAM_SECONDARY      = 0x100;  // not primary alignment
constexpr uint16_t SAM_QC_FAIL        = 0x200;  // fails platform/vendor QC
constexpr uint16_t SAM_DUPLICATE      = 0x400;  // PCR or optical duplicate
constexpr uint16_t SAM_SUPPLEMENTARY  = 0x800;  // supplementary alignment

// Library strandedness (which mate is sense) -- see BAMCOVERAGE_ROADMAP.md.
//   Forward = second/fr/secondstrand: read1 sense.
//   Reverse = first/rf/firststrand (dUTP): read2 sense.
enum class Strand { None, Forward, Reverse };
// Which transcript strand to output.
enum class StrandFilter { None, Forward, Reverse };
// Default flag policy: Primary excludes secondary+supplementary; Deeptools keeps
// stock behaviour (skip only unmapped).
enum class FilterMode { Primary, Deeptools };
// Collapse a fragment to one end / its centre (single base by default).
enum class Collapse { None, FivePrime, ThreePrime, Center };
// Orient the fragment-touch test (P8) by strand (needs strandedness).
enum class TouchStrand { Ignore, Sense, Antisense };

// Transcript strand of a read (+1 forward / -1 reverse) given library type.
inline int transcript_strand(const bam1_t *b, Strand s) {
    const uint16_t f = b->core.flag;
    const int read_sign = (f & SAM_READ_REVERSE) ? -1 : +1;
    const bool is_read2 = (f & SAM_READ_PAIRED) && (f & SAM_SECOND_IN_PAIR);
    if (s == Strand::Forward)  // read1 sense
        return is_read2 ? -read_sign : read_sign;
    // Strand::Reverse -- read2 sense (dUTP)
    return is_read2 ? read_sign : -read_sign;
}

// Extract aligned blocks [start,end) from a BAM record's CIGAR. D/N gap the
// block; query-only ops (I/S/H/P) do not advance the reference.
inline void extract_blocks(const bam1_t *b,
                           std::vector<std::pair<int64_t, int64_t>> &out) {
    out.clear();
    const uint32_t *cig = bam_get_cigar(b);
    const int n = b->core.n_cigar;
    int64_t pos = b->core.pos;
    int64_t bstart = 0, bend = 0;
    bool open = false;
    for (int k = 0; k < n; ++k) {
        const int op = bam_cigar_op(cig[k]);
        const int64_t len = bam_cigar_oplen(cig[k]);
        if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
            if (!open) { bstart = pos; open = true; }
            bend = pos + len;
            pos += len;
        } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
            if (open) { out.emplace_back(bstart, bend); open = false; }
            pos += len;
        }
        // BAM_CINS / BAM_CSOFT_CLIP / BAM_CHARD_CLIP / BAM_CPAD: no ref advance.
    }
    if (open) out.emplace_back(bstart, bend);
}

// All parsed coverage options, shared by the serial and parallel drivers so the
// per-read logic has a single source of truth.
struct FetchBounds {
    int64_t begin = std::numeric_limits<int64_t>::max();
    int64_t end = 0;
    void include(const FetchBounds &other) {
        begin = std::min(begin, other.begin);
        end = std::max(end, other.end);
    }
};

struct CovParams {
    uint32_t bin_size;
    Agg agg;
    int min_mapq;
    uint16_t include;
    uint16_t exclude;
    int min_frag;
    int max_frag;
    bool extend;
    int extend_len;
    Strand strand;
    StrandFilter sfilter;
    Collapse collapse;
    int collapsed_len;
    int64_t halo;  // optional extra query padding; never required for correctness
    int64_t mate_prefetch_padding;  // automatic padding only for bulk mate reads
    uint32_t window_bins;
    const std::vector<std::vector<FetchBounds>> *fetch_bounds = nullptr;
    double scale;        // normalization x scaleFactor (stream 0 / plus)
    double scale_minus;  // normalization x scaleFactor x scaleFactorMinus (minus)
    bool rpkm;           // divide read-count output by each bin's actual width
    int nstreams;  // 1, or 2 for --filterRNAstrand split
    bool split;    // route reads to stream 0 (+)/1 (-) by transcript strand
    bool overlap_filter;  // whitelist/blacklist are fragment-touch filters, not masks
    bool has_blacklist;
    bool has_whitelist;
    TouchStrand touch_strand;  // orient the touch test by strand
};

// Upstream getTLen uses reference-aligned length (including D, excluding N)
// when a single-end/untemplated alignment has TLEN zero.
inline uint64_t fragment_length_for_filter(const bam1_t *rec) {
    const int64_t tlen = rec->core.isize;
    if (tlen != 0)
        return tlen < 0 ? static_cast<uint64_t>(-(tlen + 1)) + 1
                        : static_cast<uint64_t>(tlen);
    uint64_t length = 0;
    const uint32_t *cigar = bam_get_cigar(rec);
    for (uint32_t i = 0; i < rec->core.n_cigar; ++i) {
        const int op = bam_cigar_op(cigar[i]);
        if (op == BAM_CMATCH || op == BAM_CDEL || op == BAM_CEQUAL || op == BAM_CDIFF)
            length += bam_cigar_oplen(cigar[i]);
    }
    return length;
}

inline bool is_proper_fragment(const bam1_t *rec) {
    const uint16_t flag = rec->core.flag;
    if (!(flag & SAM_READ_PAIRED) || !(flag & SAM_PROPER_PAIR) ||
        (flag & SAM_MATE_UNMAPPED) || rec->core.mtid != rec->core.tid ||
        rec->core.mpos < 0 || rec->core.isize == 0)
        return false;
    const bool reverse = flag & SAM_READ_REVERSE;
    const bool mate_reverse = flag & SAM_MATE_REVERSE;
    if (reverse == mate_reverse) return false;
    // Do not trust the aligner's proper-pair bit on its own: a proper fragment
    // must be inward-facing as in stock deepTools' is_proper_pair().  Long
    // genuine fragments remain supported; R deliberately has no hidden
    // maximum insert-size cut-off here.
    return reverse ? rec->core.pos >= rec->core.mpos
                   : rec->core.pos <= rec->core.mpos;
}

// pysam.AlignedSegment.infer_query_length(always=False): query-consuming CIGAR
// operations, excluding hard clips.  This is the relevant lower bound when an
// explicit --extendReads value must not shorten an alignment.
inline int64_t inferred_query_length(const bam1_t *rec) {
    if (rec->core.l_qseq > 0) return rec->core.l_qseq;
    int64_t length = 0;
    const uint32_t *cigar = bam_get_cigar(rec);
    for (uint32_t i = 0; i < rec->core.n_cigar; ++i) {
        const int op = bam_cigar_op(cigar[i]);
        if (op == BAM_CMATCH || op == BAM_CINS || op == BAM_CSOFT_CLIP ||
            op == BAM_CEQUAL || op == BAM_CDIFF)
            length += bam_cigar_oplen(cigar[i]);
    }
    return length;
}

// Strand-split region set: `all` (strand-agnostic, for masking / touch_strand=ignore)
// plus per-strand merged sets for strand-aware touch. An unstranded feature (strand 0)
// goes into all three, so it matches either fragment orientation.
struct RegionSet {
    std::vector<std::pair<uint32_t, uint32_t>> all, plus, minus;
    bool empty() const { return all.empty(); }
};

// Single-record numerator filters (shared by shape_read and the `filtered`
// normalization-denominator counter).
inline bool passes_filters(const bam1_t *rec, const CovParams &P) {
    const uint16_t flag = rec->core.flag;
    if (flag & P.exclude) return false;
    if (P.include && (flag & P.include) != P.include) return false;
    if (rec->core.qual < P.min_mapq) return false;
    if (P.min_frag > 0 || P.max_frag > 0) {
        const uint64_t length = fragment_length_for_filter(rec);
        if (P.min_frag > 0 && length < static_cast<uint64_t>(P.min_frag)) return false;
        if (P.max_frag > 0 && length > static_cast<uint64_t>(P.max_frag)) return false;
    }
    if (P.sfilter != StrandFilter::None) {
        const int ts = transcript_strand(rec, P.strand);
        if (P.sfilter == StrandFilter::Forward && ts != +1) return false;
        if (P.sfilter == StrandFilter::Reverse && ts != -1) return false;
    }
    return true;
}

// Apply filters + build coverage intervals for one record. Returns the output
// stream index (0 for now; strand-split is P5) or -1 to skip the read. `chrom_len`
// is used to clamp fragment/collapse intervals to the reference.
inline int shape_read(const bam1_t *rec, const CovParams &P, uint32_t chrom_len,
                      std::vector<std::pair<int64_t, int64_t>> &iv,
                      int &read_weight, bool already_eligible = false) {
    const uint16_t flag = rec->core.flag;
    if (!already_eligible && !passes_filters(rec, P)) return -1;

    iv.clear();
    read_weight = 1;
    if (P.collapse != Collapse::None) {
        int64_t fs, fe;
        const bool proper = is_proper_fragment(rec);
        if (proper) {
            fs = rec->core.isize > 0 ? rec->core.pos : rec->core.mpos;
            if (fs < 0 || fs >= chrom_len) return -1;
            fe = fs + std::min<uint64_t>(fragment_length_for_filter(rec), chrom_len - fs);
        } else {
            fs = rec->core.pos;
            fe = bam_endpos(rec);
        }
        if (fe <= fs) return -1;
        const bool reverse = proper && rec->core.isize < 0
            ? (flag & SAM_MATE_REVERSE) : (flag & SAM_READ_REVERSE);
        const int sign = (P.strand != Strand::None)
                             ? transcript_strand(rec, P.strand)
                             : (reverse ? -1 : +1);
        const int64_t L = P.collapsed_len;
        int64_t a, b;
        if (P.collapse == Collapse::Center) {
            const int64_t c = (fs + fe) / 2;
            a = c - L / 2;
            b = a + L;
        } else {
            const bool five_at_left = (sign >= 0);
            const bool anchor_left =
                (P.collapse == Collapse::FivePrime) ? five_at_left : !five_at_left;
            if (anchor_left) { a = fs; b = fs + L; }
            else { a = fe - L; b = fe; }
        }
        if (a < fs) a = fs;
        if (b > fe) b = fe;
        if (a < 0) a = 0;
        if (b > chrom_len) b = chrom_len;
        if (b > a) iv.emplace_back(a, b);
    } else if (P.extend) {
        int64_t fs, fe;
        const bool proper = is_proper_fragment(rec);
        if (proper) {
            fs = rec->core.isize > 0 ? rec->core.pos : rec->core.mpos;
            if (fs < 0 || fs >= chrom_len) return -1;
            fe = fs + std::min<uint64_t>(fragment_length_for_filter(rec), chrom_len - fs);
        } else if (P.extend_len > inferred_query_length(rec)) {
            if (flag & SAM_READ_REVERSE) { fe = bam_endpos(rec); fs = fe - P.extend_len; }
            else { fs = rec->core.pos; fe = fs + P.extend_len; }
        } else {
            // An extension shorter than the read is a no-op.  In particular,
            // preserve a spliced alignment's blocks instead of replacing them
            // by a shorter, solid interval.
            extract_blocks(rec, iv);
            if (iv.empty()) return -1;
            if (P.split) return (transcript_strand(rec, P.strand) == +1) ? 0 : 1;
            return 0;
        }
        if (fs < 0) fs = 0;
        if (fe > chrom_len) fe = chrom_len;
        if (fe > fs) iv.emplace_back(fs, fe);
    } else {
        extract_blocks(rec, iv);
    }
    if (iv.empty()) return -1;
    // Route to a strand stream when splitting; single stream otherwise.
    if (P.split) return (transcript_strand(rec, P.strand) == +1) ? 0 : 1;
    return 0;
}

// Accumulate a read's intervals into a window's local bins [win_bin0, win_bin0+nbins),
// clamping to [ws_bp, we_bp). Mirrors the P1/P2 math, window-local.
inline void accumulate_read(const std::vector<std::pair<int64_t, int64_t>> &iv,
                            const CovParams &P, uint32_t win_bin0, uint32_t nbins,
                            uint32_t ws_bp, uint32_t we_bp,
                            std::vector<int64_t> &acc, std::vector<int64_t> &rng,
                            std::vector<int64_t> &cdiff, int read_weight) {
    const int64_t bs_bin = static_cast<int64_t>(win_bin0);
    if (P.agg == Agg::Count) {
        int64_t last_be = -1;  // exclusive local bin already counted for this read
        for (const auto &blk : iv) {
            int64_t s = blk.first, e = blk.second;
            if (s < ws_bp) s = ws_bp;
            if (e > we_bp) e = we_bp;
            if (e <= s) continue;
            int64_t bs = s / P.bin_size - bs_bin;
            int64_t be = (e - 1) / P.bin_size + 1 - bs_bin;
            if (bs < last_be) bs = last_be;
            if (bs >= be) continue;
            cdiff[bs] += read_weight;
            cdiff[be] -= read_weight;
            last_be = be;
        }
    } else {
        for (const auto &blk : iv) {
            int64_t s = blk.first, e = blk.second;
            if (s < ws_bp) s = ws_bp;
            if (e > we_bp) e = we_bp;
            if (e <= s) continue;
            const int64_t fs = s / P.bin_size - bs_bin;
            const int64_t fe = (e - 1) / P.bin_size - bs_bin;
            if (fs == fe) {
                acc[fs] += (e - s);
            } else {
                acc[fs] += ((fs + bs_bin + 1) * static_cast<int64_t>(P.bin_size) - s);
                acc[fe] += (e - (fe + bs_bin) * static_cast<int64_t>(P.bin_size));
                if (fe - 1 >= fs + 1) {
                    rng[fs + 1] += P.bin_size;
                    rng[fe] -= P.bin_size;
                }
            }
        }
    }
    (void)nbins;
}

// Does any aligned block overlap the (merged, disjoint, start-sorted) region set?
// O(log n) per block. Splice-aware: caller passes CIGAR-derived blocks.
inline bool touches(const std::vector<std::pair<int64_t, int64_t>> &blocks,
                    const std::vector<std::pair<uint32_t, uint32_t>> &regions) {
    if (regions.empty()) return false;
    for (const auto &b : blocks) {
        // First region with start >= block.end; the candidate (strictly start <
        // block.end) is the one before it. Half-open, so a region touching at the
        // block boundary does not overlap.
        auto it = std::lower_bound(
            regions.begin(), regions.end(), static_cast<uint32_t>(b.second),
            [](const std::pair<uint32_t, uint32_t> &r, uint32_t v) { return r.first < v; });
        if (it != regions.begin()) {
            --it;
            if (static_cast<int64_t>(it->second) > b.first) return true;  // overlap
        }
    }
    return false;
}

// Merge overlapping/adjacent intervals into a disjoint, start-sorted set.
inline void merge_intervals(std::vector<std::pair<uint32_t, uint32_t>> &v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    std::vector<std::pair<uint32_t, uint32_t>> out;
    out.push_back(v[0]);
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i].first <= out.back().second) {
            if (v[i].second > out.back().second) out.back().second = v[i].second;
        } else {
            out.push_back(v[i]);
        }
    }
    v.swap(out);
}

int resolve_header_tid(sam_hdr_t *header, const std::string &chromosome) {
    const int exact = sam_hdr_name2tid(header, chromosome.c_str());
    if (exact >= 0) return exact;
    std::string alternate;
    if (chromosome.compare(0, 3, "chr") == 0) {
        alternate = chromosome.substr(3);
        if (alternate == "M") alternate = "MT";
    } else {
        alternate = chromosome == "MT" ? "chrM" : "chr" + chromosome;
    }
    return sam_hdr_name2tid(header, alternate.c_str());
}

// Parse a {chrom: [(start,end[,strand]), ...]} dict into per-tid strand-split,
// merged RegionSets. strand: +1 -> {all,plus}; -1 -> {all,minus}; 0/absent ->
// {all,plus,minus} (unstranded feature matches either orientation).
inline std::vector<RegionSet> parse_region_sets(py::object obj, sam_hdr_t *hdr,
                                                int n_targets) {
    std::vector<RegionSet> sets(n_targets);
    if (obj.is_none()) return sets;
    for (auto item : obj.cast<py::dict>()) {
        const std::string chrom = py::str(item.first).cast<std::string>();
        const int tid = resolve_header_tid(hdr, chrom);
        if (tid < 0) continue;
        for (auto iv : item.second.cast<py::list>()) {
            auto t = iv.cast<py::tuple>();
            const uint32_t s = t[0].cast<uint32_t>(), e = t[1].cast<uint32_t>();
            const int strand = t.size() >= 3 ? t[2].cast<int>() : 0;
            if (e <= s) continue;
            sets[tid].all.emplace_back(s, e);
            if (strand >= 0) sets[tid].plus.emplace_back(s, e);
            if (strand <= 0) sets[tid].minus.emplace_back(s, e);
        }
    }
    for (auto &rs : sets) {
        merge_intervals(rs.all); merge_intervals(rs.plus); merge_intervals(rs.minus);
    }
    return sets;
}

// Choose which region vector to test a fragment against, given the touch-strand
// mode and the fragment's transcript strand (+1/-1).
inline const std::vector<std::pair<uint32_t, uint32_t>> &
pick_regions(const RegionSet &rs, TouchStrand ts, int fstrand) {
    if (ts == TouchStrand::Ignore) return rs.all;
    const bool sense = (ts == TouchStrand::Sense);
    const bool want_plus = sense ? (fstrand > 0) : (fstrand <= 0);
    return want_plus ? rs.plus : rs.minus;
}

// Range min/max updates followed by one materialization. A long fragment updates
// O(log windows) nodes, not every covered bin/window. Memory depends on the output
// window count, never on the read count or the largest template span.
class FetchRangeTree {
    std::size_t n_;
    std::vector<FetchBounds> tree_;
public:
    explicit FetchRangeTree(std::size_t n) : n_(n), tree_(2 * n) {}
    void include(std::size_t first, std::size_t end, const FetchBounds &bounds) {
        for (first += n_, end += n_; first < end; first /= 2, end /= 2) {
            if (first & 1) tree_[first++].include(bounds);
            if (end & 1) tree_[--end].include(bounds);
        }
    }
    std::vector<FetchBounds> finish() {
        for (std::size_t i = 1; i < n_; ++i) {
            tree_[2 * i].include(tree_[i]);
            tree_[2 * i + 1].include(tree_[i]);
        }
        return {tree_.begin() + n_, tree_.end()};
    }
};

template<class Observe>
std::vector<std::vector<FetchBounds>> prepare_coverage(
        const std::string &path, const CovParams &P,
        const std::vector<uint32_t> &lengths, bool plan_fetches, Observe observe) {
    std::vector<FetchRangeTree> trees;
    if (plan_fetches) for (uint32_t length : lengths) {
        const uint64_t bins = (static_cast<uint64_t>(length) + P.bin_size - 1) / P.bin_size;
        trees.emplace_back((bins + P.window_bins - 1) / P.window_bins);
    }
    HtsFilePtr file;
    file.reset(hts_open(path.c_str(), "r"));
    if (!file.get()) throw std::runtime_error("could not open BAM for coverage preparation");
    HdrPtr header;
    header.reset(sam_hdr_read(file.get()));
    if (!header.get()) throw std::runtime_error("could not read BAM header for coverage preparation");
    BamRecordPtr record(bam_init1());
    if (!record.get()) throw std::runtime_error("could not allocate a BAM record");
    std::vector<std::pair<int64_t, int64_t>> intervals;
    uint64_t scanned = 0;
    while (checked_bam_status(sam_read1(file.get(), header.get(), record.get())) >= 0) {
        poll_coverage_work(++scanned);
        const int tid = record.get()->core.tid;
        if (tid < 0) continue;
        if (static_cast<std::size_t>(tid) >= lengths.size())
            throw std::runtime_error("BAM record has an invalid reference id");
        observe(record.get());
        if (!plan_fetches) continue;
        int weight = 1;
        if (shape_read(record.get(), P, lengths[tid], intervals, weight) < 0) continue;
        // Fetching one base at an alignment's start is sufficient to retrieve it.
        const FetchBounds bounds{record.get()->core.pos, record.get()->core.pos + 1};
        for (const auto &interval : intervals) {
            const uint64_t first = interval.first / P.bin_size / P.window_bins;
            const uint64_t last = (interval.second - 1) / P.bin_size / P.window_bins;
            trees[tid].include(first, last + 1, bounds);
        }
    }
    std::vector<std::vector<FetchBounds>> result;
    for (auto &tree : trees) result.push_back(tree.finish());
    return result;
}

struct MateInfo {
    bool eligible = false;
    uint64_t positive_span = 0;
    uint8_t touch = 0;  // bit 0 whitelist; bit 1 blacklist
};

inline bool same_mate_info(const MateInfo &a, const MateInfo &b) {
    return a.eligible == b.eligible && a.positive_span == b.positive_span &&
           a.touch == b.touch;
}

struct CachedMate {
    MateInfo info;
    bool ambiguous = false;
    bool verify = false;
};

std::string same_end_key(const bam1_t *rec) {
    std::string result(bam_get_qname(rec));
    result.push_back('\0');
    const uint8_t *rg = bam_aux_get(rec, "RG");
    const char *group = rg ? bam_aux2Z(rg) : nullptr;
    if (group) result.append(group);
    result.push_back('\0');
    const int64_t mate_coordinates[] = {rec->core.mtid, rec->core.mpos};
    result.append(reinterpret_cast<const char *>(mate_coordinates),
                  sizeof(mate_coordinates));
    const uint16_t end = rec->core.flag &
        (SAM_FIRST_IN_PAIR | SAM_SECOND_IN_PAIR);
    result.append(reinterpret_cast<const char *>(&end), sizeof(end));
    return result;
}

uint8_t record_touch(const bam1_t *rec, const CovParams &P,
                     const std::vector<RegionSet> &black,
                     const std::vector<RegionSet> &white,
                     std::vector<std::pair<int64_t, int64_t>> &blocks) {
    const int tid = rec->core.tid;
    if (tid < 0 || static_cast<std::size_t>(tid) >= white.size())
        throw std::runtime_error("BAM record has an invalid reference id");
    extract_blocks(rec, blocks);
    const int strand = transcript_strand(rec, P.strand);
    uint8_t touch = 0;
    if (P.has_whitelist &&
        touches(blocks, pick_regions(white[tid], P.touch_strand, strand)))
        touch |= 1;
    if (P.has_blacklist &&
        touches(blocks, pick_regions(black[tid], P.touch_strand, strand)))
        touch |= 2;
    return touch;
}

struct PendingNonprimaryTouch {
    uint8_t touch = 0;
    int32_t primary_tid = -1;
    int64_t primary_pos = -1;
    bool has_primary = false;
    bool primary_eligible = false;
    bool ambiguous = false;
};

// Nonprimary alignments can occur after their primary pair in coordinate order.
// A small preliminary index (only names of nonprimary alignments that actually
// touch a filter region) lets the primary pair receive their touch decision in
// either direction.  A second scan resolves each loose same-end identity to one
// primary coordinate; collisions are rejected rather than mixed.
std::unordered_map<std::string, uint8_t> build_nonprimary_touch_index(
        const std::string &path, const CovParams &P,
        const std::vector<RegionSet> &black,
        const std::vector<RegionSet> &white) {
    std::unordered_map<std::string, PendingNonprimaryTouch> pending;
    std::vector<std::pair<int64_t, int64_t>> blocks;

    auto scan = [&](bool nonprimary_pass) {
        HtsFilePtr file;
        file.reset(hts_open(path.c_str(), "r"));
        if (!file.get()) throw std::runtime_error("could not reopen BAM: " + path);
        HdrPtr header;
        header.reset(sam_hdr_read(file.get()));
        if (!header.get()) throw std::runtime_error("could not read BAM header: " + path);
        BamRecordPtr record(bam_init1());
        if (!record.get()) throw std::runtime_error("could not allocate a BAM record");
        uint64_t scanned = 0;
        while (checked_bam_status(sam_read1(file.get(), header.get(), record.get())) >= 0) {
            poll_coverage_work(++scanned);
            const uint16_t flag = record.get()->core.flag;
            if (!(flag & SAM_READ_PAIRED) || (flag & SAM_MATE_UNMAPPED) ||
                record.get()->core.mtid < 0 || record.get()->core.mpos < 0)
                continue;
            const bool nonprimary = flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY);
            if (nonprimary_pass) {
                if (!nonprimary || !passes_filters(record.get(), P)) continue;
                const uint8_t touch = record_touch(record.get(), P, black, white, blocks);
                if (!touch) continue;
                std::string key = same_end_key(record.get());
                if (key.size() <= 4096) pending[std::move(key)].touch |= touch;
            } else {
                if (nonprimary) continue;
                const std::string key = same_end_key(record.get());
                auto found = pending.find(key);
                if (found == pending.end()) continue;
                PendingNonprimaryTouch &entry = found->second;
                const bool eligible = passes_filters(record.get(), P);
                if (!entry.has_primary) {
                    entry.primary_tid = record.get()->core.tid;
                    entry.primary_pos = record.get()->core.pos;
                    entry.primary_eligible = eligible;
                    entry.has_primary = true;
                } else if (entry.primary_tid != record.get()->core.tid ||
                           entry.primary_pos != record.get()->core.pos ||
                           entry.primary_eligible != eligible) {
                    entry.ambiguous = true;
                }
            }
        }
    };
    scan(true);
    if (pending.empty()) return {};
    scan(false);

    std::unordered_map<std::string, uint8_t> result;
    result.reserve(pending.size());
    for (auto &item : pending) {
        const PendingNonprimaryTouch &entry = item.second;
        if (entry.has_primary && entry.primary_eligible && !entry.ambiguous)
            result.emplace(std::move(item.first), entry.touch);
    }
    return result;
}

// Each worker has its own lazy htslib handle. Most nearby mates are served from
// a bounded window prefetch. Cache misses use the indexed mate coordinate,
// including other references. Genomic padding affects speed, never correctness.
class MateLookup {
    const std::string &path_;
    hts_idx_t *idx_;
    const CovParams &P_;
    const std::vector<RegionSet> &black_, &white_;
    const std::unordered_map<std::string, uint8_t> &nonprimary_touch_;
    HtsFilePtr file_;
    HdrPtr header_;
    BamRecordPtr record_{bam_init1()};
    std::vector<std::pair<int64_t, int64_t>> blocks_;
    std::unordered_map<std::string, CachedMate> full_cache_, loose_cache_;
    static constexpr std::size_t CACHE_ENTRIES = 32768;
    static constexpr std::size_t CACHE_BYTES = 8u << 20;
    static constexpr std::size_t MAX_CACHE_KEY_BYTES = 4096;
    std::size_t cache_bytes_ = 0;
    std::string observation_key_;
    std::vector<uint64_t> positive_full_, positive_loose_, zero_full_, zero_loose_, positive_positions_;
    static void bloom_add(std::vector<uint64_t> &bits, const std::string &key) {
        const uint64_t hash = std::hash<std::string>{}(key);
        for (unsigned shift : {0u, 21u, 42u}) {
            const auto bit = (hash >> shift) & ((1u << 22) - 1);
            bits[bit >> 6] |= uint64_t(1) << (bit & 63);
        }
    }
    static bool bloom_contains(const std::vector<uint64_t> &bits, const std::string &key) {
        if (bits.empty()) return true;
        const uint64_t hash = std::hash<std::string>{}(key);
        for (unsigned shift : {0u, 21u, 42u}) {
            const auto bit = (hash >> shift) & ((1u << 22) - 1);
            if (!(bits[bit >> 6] & (uint64_t(1) << (bit & 63)))) return false;
        }
        return true;
    }
    bool prefetched_ = false;
    int prefetch_tid_ = -1;
    int64_t prefetch_begin_ = 0, prefetch_end_ = 0;

    void open_lookup() {
        if (file_.get()) return;
        file_.reset(hts_open(path_.c_str(), "r"));
        if (!file_.get()) throw std::runtime_error("could not open BAM for mate lookup");
        header_.reset(sam_hdr_read(file_.get()));
        if (!header_.get()) throw std::runtime_error("could not read BAM header for mate lookup");
    }
    void clear_cache() {
        full_cache_.clear(); loose_cache_.clear(); cache_bytes_ = 0;
    }

    static uint16_t read_end(const bam1_t *rec, bool mate) {
        uint16_t end = rec->core.flag & (SAM_FIRST_IN_PAIR | SAM_SECOND_IN_PAIR);
        if (mate) end = ((end & SAM_FIRST_IN_PAIR) << 1) |
                        ((end & SAM_SECOND_IN_PAIR) >> 1);
        return end;
    }
    // Loose identity is required for a supplementary/secondary request because
    // RNEXT/PNEXT identifies its primary mate, whose reciprocal PNEXT normally
    // points to the requesting read's *primary* alignment rather than this one.
    static void build_key(std::string &result, const bam1_t *rec, bool mate, bool loose) {
        result.clear();
        result.append(bam_get_qname(rec)); result.push_back('\0');
        const uint8_t *rg = bam_aux_get(rec, "RG");
        const char *group = rg ? bam_aux2Z(rg) : nullptr;
        if (group) result.append(group);
        result.push_back('\0');
        const int64_t coords[] = {mate ? rec->core.mtid : rec->core.tid,
                                  mate ? rec->core.mpos : rec->core.pos};
        result.append(reinterpret_cast<const char *>(coords), sizeof(coords));
        const uint16_t end = read_end(rec, mate);
        result.append(reinterpret_cast<const char *>(&end), sizeof(end));
        if (!loose) {
            const int64_t reciprocal[] = {mate ? rec->core.tid : rec->core.mtid,
                                          mate ? rec->core.pos : rec->core.mpos};
            result.append(reinterpret_cast<const char *>(reciprocal), sizeof(reciprocal));
        }
    }
    static std::string loose_key(const bam1_t *rec, bool mate) {
        std::string key; build_key(key, rec, mate, true); return key;
    }
    static std::string full_key(const bam1_t *rec, bool mate) {
        std::string key; build_key(key, rec, mate, false); return key;
    }
    void make_room() {
        if (prefetched_) return;
        if (full_cache_.size() + loose_cache_.size() < CACHE_ENTRIES && cache_bytes_ < CACHE_BYTES) return;
        clear_cache();
    }
    void remember_in(std::unordered_map<std::string, CachedMate> &cache,
                            const std::string &name, const MateInfo &info) {
        if (name.size() > MAX_CACHE_KEY_BYTES) return;
        auto found = cache.find(name);
        if (found == cache.end()) {
            const auto bytes = name.size() + sizeof(CachedMate) + sizeof(std::string) + 64;
            if (full_cache_.size() + loose_cache_.size() >= CACHE_ENTRIES ||
                bytes > CACHE_BYTES - cache_bytes_) return;
            cache_bytes_ += bytes;
            cache.emplace(name, CachedMate{info, false});
        } else if (!same_mate_info(found->second.info, info)) {
            found->second.ambiguous = true;
        }
    }
    void remember_primary(const bam1_t *rec, const MateInfo &info) {
        if (rec->core.flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) return;
        // Overhanging alignments before the query start are incomplete evidence
        // for a coordinate: another same-key record may end before this window.
        if (prefetched_ && (rec->core.tid != prefetch_tid_ ||
            rec->core.pos < prefetch_begin_ || rec->core.pos >= prefetch_end_)) return;
        make_room();
        build_key(observation_key_, rec, false, false);
        const bool useful = info.touch || (info.positive_span && P_.agg != Agg::Count &&
                              (P_.extend || P_.collapse != Collapse::None));
        auto remember = [&](auto &cache, auto &positive, auto &zero, const std::string &key) {
            if (!prefetched_) { remember_in(cache, key, info); return; }
            if (!useful) {
                bloom_add(zero, key);
                const auto existing = cache.find(key);
                if (existing != cache.end()) existing->second.verify = true;
                return;
            }
            bloom_add(positive, key);
            const bool verify = bloom_contains(zero, key);
            remember_in(cache, key, info);
            const auto existing = cache.find(key);
            if (existing != cache.end()) existing->second.verify |= verify;
        };
        if (prefetched_ && useful) {
            const uint64_t bit = static_cast<uint64_t>(rec->core.pos - prefetch_begin_) % (positive_positions_.size() * 64);
            positive_positions_[bit >> 6] |= uint64_t(1) << (bit & 63);
        }
        remember(full_cache_, positive_full_, zero_full_, observation_key_);
        if ((P_.exclude & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) !=
            (SAM_SECONDARY | SAM_SUPPLEMENTARY)) {
            build_key(observation_key_, rec, false, true);
            remember(loose_cache_, positive_loose_, zero_loose_, observation_key_);
        }
    }
    static MateInfo cached_value(const CachedMate &cached) {
        return cached.ambiguous ? MateInfo{} : cached.info;
    }
    MateInfo summarize(const bam1_t *rec) {
        MateInfo info;
        info.eligible = passes_filters(rec, P_);
        if (!info.eligible) return info;
        if (is_proper_fragment(rec) && rec->core.isize > 0)
            info.positive_span = fragment_length_for_filter(rec);
        if (P_.overlap_filter && (P_.has_whitelist || P_.has_blacklist)) {
            info.touch = record_touch(rec, P_, black_, white_, blocks_);
            if (!nonprimary_touch_.empty()) {
                const auto inherited = nonprimary_touch_.find(same_end_key(rec));
                if (inherited != nonprimary_touch_.end()) info.touch |= inherited->second;
            }
        }
        return info;
    }
public:
    MateLookup(const std::string &path, hts_idx_t *idx, const CovParams &P,
               const std::vector<RegionSet> &black, const std::vector<RegionSet> &white,
               const std::unordered_map<std::string, uint8_t> &nonprimary_touch)
        : path_(path), idx_(idx), P_(P), black_(black), white_(white),
          nonprimary_touch_(nonprimary_touch) {
        if (!record_.get()) throw std::runtime_error("could not allocate a mate record");
    }
    // One sequential query supplies nearby mate summaries before any output
    // decisions. A fixed entry/byte budget bounds each worker's cache. Keys
    // omitted under pressure and distant/cross-chromosome mates retain exact
    // indexed fallback. Prefetched entries are never evicted mid-window.
    void prefetch(int tid, int64_t begin, int64_t end) {
        if (prefetched_ && tid == prefetch_tid_ &&
            begin >= prefetch_begin_ && end <= prefetch_end_) return;
        // Tiny output windows must not clear megabytes of membership storage
        // and decode the same nearby reads for every few bases. Coalesce only
        // mate queries; output windows and exact fallback remain independent.
        constexpr int64_t minimum_span = 1 << 16;
        if (end - begin < minimum_span) {
            begin = begin / minimum_span * minimum_span;
            end = ((end + minimum_span - 1) / minimum_span) * minimum_span;
        }
        open_lookup();
        clear_cache(); prefetched_ = true;
        positive_full_.assign(65536, 0); zero_full_.assign(65536, 0);
        positive_loose_.assign(65536, 0); zero_loose_.assign(65536, 0);
        positive_positions_.assign((std::min<int64_t>(end - begin, 1u << 22) + 63) / 64, 0);
        prefetch_tid_ = tid; prefetch_begin_ = begin; prefetch_end_ = end;
        IteratorPtr iterator;
        iterator.reset(sam_itr_queryi(idx_, tid, begin, end));
        if (!iterator.get()) throw std::runtime_error("could not prefetch BAM mate context");
        uint64_t scanned = 0;
        while (checked_bam_status(sam_itr_next(file_.get(), iterator.get(), record_.get())) >= 0) {
            poll_coverage_work(++scanned);
            if (!(record_.get()->core.flag & SAM_READ_PAIRED) ||
                (record_.get()->core.flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY))) continue;
            remember_primary(record_.get(), summarize(record_.get()));
        }
    }
    void prefetch_for_record(const bam1_t *rec) {
        const int tid = rec->core.tid;
        const int64_t pos = rec->core.pos;
        if (tid == prefetch_tid_ && pos >= prefetch_begin_ && pos < prefetch_end_) return;
        const int64_t width = static_cast<int64_t>(P_.window_bins) * P_.bin_size;
        const int64_t begin = pos / width * width;
        const int64_t padding = P_.halo + P_.mate_prefetch_padding;
        prefetch(tid, std::max<int64_t>(0, begin - padding), begin + width + padding);
    }
    MateInfo observe(const bam1_t *rec) {
        MateInfo info = summarize(rec);
        if (!prefetched_ && (rec->core.flag & SAM_READ_PAIRED)) remember_primary(rec, info);
        return info;
    }
    MateInfo mate(const bam1_t *rec) {
        if (!(rec->core.flag & SAM_READ_PAIRED) || (rec->core.flag & SAM_MATE_UNMAPPED) ||
            rec->core.mtid < 0 || rec->core.mpos < 0) return {};
        const bool nonprimary = rec->core.flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY);
        const bool in_prefetch = prefetched_ && rec->core.mtid == prefetch_tid_ &&
            rec->core.mpos >= prefetch_begin_ && rec->core.mpos < prefetch_end_;
        if (in_prefetch) {
            const uint64_t bit = static_cast<uint64_t>(rec->core.mpos - prefetch_begin_) % (positive_positions_.size() * 64);
            if (!(positive_positions_[bit >> 6] & (uint64_t(1) << (bit & 63)))) return {};
        }
        std::string wanted = nonprimary ? loose_key(rec, true) : full_key(rec, true);
        // Bloom negatives prove absence. Every positive is verified exactly.
        if (in_prefetch && !bloom_contains(nonprimary ? positive_loose_ : positive_full_, wanted)) return {};
        auto &cache = nonprimary ? loose_cache_ : full_cache_;
        const auto cached = cache.find(wanted);
        if (cached != cache.end() && !cached->second.verify) return cached_value(cached->second);
        open_lookup();
        IteratorPtr iterator;
        iterator.reset(sam_itr_queryi(idx_, rec->core.mtid, rec->core.mpos, rec->core.mpos + 1));
        if (!iterator.get()) throw std::runtime_error("could not query BAM mate coordinate");
        MateInfo info;
        bool found = false, ambiguous = false;
        uint64_t scanned = 0;
        while (checked_bam_status(sam_itr_next(file_.get(), iterator.get(), record_.get())) >= 0) {
            poll_coverage_work(++scanned);
            if (record_.get()->core.pos != rec->core.mpos ||
                !(record_.get()->core.flag & SAM_READ_PAIRED) ||
                (record_.get()->core.flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) ||
                std::strcmp(bam_get_qname(record_.get()), bam_get_qname(rec)) != 0 ||
                (nonprimary ? loose_key(record_.get(), false) : full_key(record_.get(), false))
                    != wanted)
                continue;
            const MateInfo candidate = summarize(record_.get());
            if (!found) {
                info = candidate;
                found = true;
            } else if (!same_mate_info(info, candidate)) {
                ambiguous = true;
            }
        }
        make_room();
        remember_in(cache, wanted, info);
        const auto stored = cache.find(wanted);
        if (stored != cache.end()) stored->second = CachedMate{info, ambiguous, false};
        return ambiguous ? MateInfo{} : info;
    }
    bool retained(const bam1_t *rec, const MateInfo &own) {
        if (!own.eligible) return false;
        if (!P_.overlap_filter || !(P_.has_whitelist || P_.has_blacklist)) return true;
        if (P_.has_blacklist && (own.touch & 2)) return false;
        if (!P_.has_blacklist && (!P_.has_whitelist || (own.touch & 1))) return true;
        const uint8_t touch = own.touch | mate(rec).touch;
        return (!P_.has_whitelist || (touch & 1)) && (!P_.has_blacklist || !(touch & 2));
    }
};

// Each worker keeps its bounded accumulators between windows. Reusing these
// buffers avoids allocating/zero-initializing a second set for every query.
struct CoverageScratch {
    std::vector<std::vector<int64_t>> acc, rng;
    // A difference array necessarily contains negative end-boundary deltas.
    // Keep it signed and wide enough that the boundary stage cannot overflow
    // before materialization into the int64 running total below.
    std::vector<std::vector<int64_t>> cdiff;
};

// Compute one window's per-bin values (single output stream). `fp` and `rec` are
// per-thread; `idx` is shared read-only.
inline void compute_window(htsFile *fp, hts_idx_t *idx, const Window &w,
                           const CovParams &P, uint32_t chrom_len,
                           const RegionSet &blackset,
                           const RegionSet &whiteset,
                           MateLookup &mates,
                           bam1_t *rec,
                           std::vector<std::pair<int64_t, int64_t>> &iv,
                           std::vector<std::vector<float>> &values,
                           CoverageScratch &scratch) {
    const uint32_t nbins = w.bin1 - w.bin0;
    const uint32_t ws_bp = w.bin0 * P.bin_size;
    const uint32_t we_bp = std::min<uint64_t>(
        static_cast<uint64_t>(w.bin1) * P.bin_size, chrom_len);
    int64_t qstart = static_cast<int64_t>(ws_bp) - P.halo;
    if (qstart < 0) qstart = 0;
    int64_t qend = static_cast<int64_t>(we_bp) + P.halo;
    if (qend > chrom_len) qend = chrom_len;
    if (P.fetch_bounds) {
        const auto &bounds = (*P.fetch_bounds)[w.tid][w.bin0 / P.window_bins];
        qstart = std::min(qstart, bounds.begin);
        qend = std::max(qend, bounds.end);
    }

    const int ns = P.nstreams;
    auto &acc = scratch.acc; auto &rng = scratch.rng; auto &cdiff = scratch.cdiff;
    acc.resize(ns); rng.resize(ns); cdiff.resize(ns);
    for (int s = 0; s < ns; ++s) {
        if (P.agg == Agg::Count) cdiff[s].assign(nbins + 1, 0);
        else { acc[s].assign(nbins, 0); rng[s].assign(nbins + 1, 0); }
    }

    const bool do_overlap = P.overlap_filter &&
                            (P.has_whitelist || P.has_blacklist);
    const bool fragment_depth = P.agg != Agg::Count && (P.extend || P.collapse != Collapse::None);
    if (do_overlap || fragment_depth)
        mates.prefetch(w.tid, std::max<int64_t>(0, qstart - P.mate_prefetch_padding),
                       std::min<int64_t>(chrom_len, qend + P.mate_prefetch_padding));
    uint64_t scanned = 0;
    IteratorPtr itr;
    itr.reset(sam_itr_queryi(idx, w.tid, qstart, qend));
    if (itr.get()) {
        while (checked_bam_status(sam_itr_next(fp, itr.get(), rec)) >= 0) {
            poll_coverage_work(++scanned);
            if (do_overlap || fragment_depth) {
                const MateInfo own = mates.observe(rec);
                if (!mates.retained(rec, own)) continue;
                // Depth counts a retained fragment once; read-count instead
                // places each eligible alignment with unit weight.
                if (fragment_depth && is_proper_fragment(rec) && rec->core.isize < 0 &&
                    mates.mate(rec).positive_span == fragment_length_for_filter(rec)) continue;
            }
            int read_weight = 1;
            const int s = shape_read(rec, P, chrom_len, iv, read_weight,
                                     do_overlap || fragment_depth);
            if (s < 0) continue;
            accumulate_read(iv, P, w.bin0, nbins, ws_bp, we_bp,
                            acc[s], rng[s], cdiff[s], read_weight);
        }
    }

    values.resize(ns);
    for (int s = 0; s < ns; ++s) {
        const double sc = (s == 1) ? P.scale_minus : P.scale;  // per-strand scale
        std::vector<float> &vals = values[s];
        vals.resize(nbins);
        if (P.agg == Agg::Count) {
            int64_t run = 0;
            for (uint32_t i = 0; i < nbins; ++i) {
                run += cdiff[s][i];
                double bin_scale = sc;
                if (P.rpkm) {
                    const uint32_t abin = w.bin0 + i;
                    uint64_t be = (static_cast<uint64_t>(abin) + 1) * P.bin_size;
                    if (be > chrom_len) be = chrom_len;
                    const uint64_t begin = static_cast<uint64_t>(abin) * P.bin_size;
                    bin_scale /= static_cast<double>(be - begin);
                }
                vals[i] = checked_float(run * bin_scale, "coverage value");
            }
        } else {
            int64_t run = 0;
            for (uint32_t i = 0; i < nbins; ++i) {
                run += rng[s][i];
                const int64_t a = acc[s][i] + run;
                if (P.agg == Agg::Sum)
                    vals[i] = checked_float(a * sc, "coverage value");
                else {
                    const uint32_t abin = w.bin0 + i;
                    const uint64_t be = std::min<uint64_t>(
                        (static_cast<uint64_t>(abin) + 1) * P.bin_size, chrom_len);
                    vals[i] = checked_float(
                        static_cast<double>(a) /
                            (be - abin * P.bin_size) * sc,
                        "coverage value");
                }
            }
        }
        // Whitelist: keep only bins overlapping a whitelist region (zero the rest).
        if (P.has_whitelist && !P.overlap_filter) {
            std::vector<char> keep(nbins, 0);
            for (const auto &wl : whiteset.all) {
                int64_t b0 = static_cast<int64_t>(wl.first) / P.bin_size - w.bin0;
                int64_t b1 = static_cast<int64_t>(wl.second - 1) / P.bin_size - w.bin0;
                if (b0 < 0) b0 = 0;
                if (b1 >= static_cast<int64_t>(nbins)) b1 = nbins - 1;
                for (int64_t i = b0; i <= b1; ++i) keep[i] = 1;
            }
            for (uint32_t i = 0; i < nbins; ++i) if (!keep[i]) vals[i] = 0.0f;
        }
        // Blacklist: force blacklisted bins to zero (mask mode only).
        if (!P.overlap_filter) for (const auto &bl : blackset.all) {
            int64_t b0 = static_cast<int64_t>(bl.first) / P.bin_size - w.bin0;
            int64_t b1 = static_cast<int64_t>(bl.second - 1) / P.bin_size - w.bin0;
            if (b0 < 0) b0 = 0;
            if (b1 >= static_cast<int64_t>(nbins)) b1 = nbins - 1;
            for (int64_t i = b0; i <= b1; ++i) vals[i] = 0.0f;
        }
    }
}


void bam_coverage_bigwig(const std::string &bam_path,
                         const std::string &out_path,
                         const std::string &out_path_reverse,
                         const std::string &out_format,
                         uint32_t bin_size,
                         const std::string &aggregation,
                         int min_mapping_quality,
                         uint16_t sam_flag_include,
                         uint16_t sam_flag_exclude,
                         const std::string &filter_mode,
                         bool ignore_duplicates,
                         int min_fragment_length,
                         int max_fragment_length,
                         int extend_reads,
                         const std::string &strandedness,
                         const std::string &filter_rna_strand,
                         const std::string &collapse_str,
                         int collapsed_length,
                         py::object blacklist,
                         py::object whitelist,
                         int threads,
                         int window_size,
                         const std::string &normalization,
                         const std::string &normalization_denominator,
                         bool exact_scaling,
                         double scale_factor,
                         double scale_factor_minus,
                         py::object ignore_for_normalization,
                         int max_zooms,
                         int compression_level,
                         bool filter_by_overlap,
                         const std::string &feature_touch_strand,
                         int filter_by_overlap_halo) {
    if (bin_size == 0) throw std::runtime_error("binSize must be > 0");
    if (min_mapping_quality < 0 || min_mapping_quality > 255)
        throw std::runtime_error("min_mapping_quality must be in 0..255");
    if (min_fragment_length < 0 || max_fragment_length < 0)
        throw std::runtime_error("fragment lengths must be >= 0");
    if (max_fragment_length > 0 && min_fragment_length > max_fragment_length)
        throw std::runtime_error(
            "min_fragment_length must be <= max_fragment_length");
    if (extend_reads < 0)
        throw std::runtime_error("extend_reads must be >= 0");
    if (collapsed_length < 1)
        throw std::runtime_error("collapsed_length must be >= 1");
    if (threads < 0) throw std::runtime_error("threads must be >= 0");
    if (window_size < 0) throw std::runtime_error("window_size must be >= 0");
    if (!std::isfinite(scale_factor) || !std::isfinite(scale_factor_minus))
        throw std::runtime_error("scale factors must be finite");
    validate_writer_options(max_zooms, compression_level);
    Agg agg;
    if (aggregation == "mean") agg = Agg::Mean;
    else if (aggregation == "sum") agg = Agg::Sum;
    else if (aggregation == "count") agg = Agg::Count;
    else throw std::runtime_error("aggregation must be mean|sum|count");
    if (normalization != "none" && normalization != "cpm" &&
        normalization != "rpkm" && normalization != "bpm")
        throw std::runtime_error(
            "normalization must be none|cpm|rpkm|bpm");
    if (normalization != "none" && agg != Agg::Count)
        throw std::runtime_error(
            "normalized metrics require read-count aggregation");

    FilterMode fmode;
    if (filter_mode == "primary") fmode = FilterMode::Primary;
    else if (filter_mode == "deeptools") fmode = FilterMode::Deeptools;
    else throw std::runtime_error("filter_mode must be primary|deeptools");

    Strand strand;
    if (strandedness == "none") strand = Strand::None;
    else if (strandedness == "forward") strand = Strand::Forward;
    else if (strandedness == "reverse") strand = Strand::Reverse;
    else throw std::runtime_error("strandedness must be none|forward|reverse");

    const bool split = (filter_rna_strand == "split");
    StrandFilter sfilter = StrandFilter::None;
    if (!split) {
        if (filter_rna_strand == "none") sfilter = StrandFilter::None;
        else if (filter_rna_strand == "forward") sfilter = StrandFilter::Forward;
        else if (filter_rna_strand == "reverse") sfilter = StrandFilter::Reverse;
        else throw std::runtime_error(
            "filter_rna_strand must be none|forward|reverse|split");
    }
    if ((sfilter != StrandFilter::None || split) && strand == Strand::None)
        throw std::runtime_error(
            "filter_rna_strand needs strandedness (forward|reverse) to be set");
    if (split && out_path_reverse.empty())
        throw std::runtime_error(
            "filter_rna_strand split needs a second output (out_path_reverse)");

    TouchStrand touch_strand;
    if (feature_touch_strand == "ignore") touch_strand = TouchStrand::Ignore;
    else if (feature_touch_strand == "sense") touch_strand = TouchStrand::Sense;
    else if (feature_touch_strand == "antisense") touch_strand = TouchStrand::Antisense;
    else throw std::runtime_error("feature_touch_strand must be ignore|sense|antisense");
    if (touch_strand != TouchStrand::Ignore && strand == Strand::None)
        throw std::runtime_error(
            "feature_touch_strand needs strandedness (forward|reverse) to be set");
    if (filter_by_overlap_halo < -1)
        throw std::runtime_error("filter_by_overlap_halo must be >= 0 or -1 for auto");

    Collapse collapse;
    if (collapse_str == "none") collapse = Collapse::None;
    else if (collapse_str == "5prime") collapse = Collapse::FivePrime;
    else if (collapse_str == "3prime") collapse = Collapse::ThreePrime;
    else if (collapse_str == "center") collapse = Collapse::Center;
    else throw std::runtime_error("collapse must be none|5prime|3prime|center");
    if (collapse != Collapse::None && extend_reads > 0)
        throw std::runtime_error(
            "extend_reads and collapse cannot be used together");
    if (collapse != Collapse::None && collapsed_length < 1)
        throw std::runtime_error("collapsed_length must be >= 1");

    // Effective exclude mask: always drop unmapped; primary mode drops
    // secondary+supplementary; add the caller's exclude bits and, optionally,
    // the duplicate bit.
    uint16_t exclude = sam_flag_exclude | SAM_READ_UNMAPPED;
    if (fmode == FilterMode::Primary) exclude |= (SAM_SECONDARY | SAM_SUPPLEMENTARY);
    if (ignore_duplicates) exclude |= SAM_DUPLICATE;
    const bool extend = extend_reads > 0 && collapse == Collapse::None;

    HtsFilePtr file;
    file.reset(hts_open(bam_path.c_str(), "r"));
    if (!file.get()) throw std::runtime_error("could not open BAM: " + bam_path);
    HdrPtr hdr;
    hdr.reset(sam_hdr_read(file.get()));
    if (!hdr.get()) throw std::runtime_error("could not read header: " + bam_path);
    const int n_targets = sam_hdr_nref(hdr.get());
    if (n_targets <= 0) throw std::runtime_error("no references in header");

    std::vector<uint32_t> lengths(n_targets);
    std::vector<const char *> names(n_targets);
    for (int t = 0; t < n_targets; ++t) {
        lengths[t] = sam_hdr_tid2len(hdr.get(), t);
        names[t] = sam_hdr_tid2name(hdr.get(), t);
    }

    // Blacklist / whitelist: {chrom: [(start,end[,strand]), ...]}. Strand-split,
    // merged RegionSets used either as bin-masks (default) or fragment-touch filters
    // (--filterByOverlap), optionally strand-oriented (--featureTouchStrand).
    std::vector<RegionSet> blacklist_by_tid = parse_region_sets(blacklist, hdr.get(), n_targets);
    std::vector<RegionSet> whitelist_by_tid = parse_region_sets(whitelist, hdr.get(), n_targets);

    OutFmt fmt;
    if (out_format == "bigwig") fmt = OutFmt::BigWig;
    else if (out_format == "bedgraph") fmt = OutFmt::BedGraph;
    else if (out_format == "bedgraph.gz") fmt = OutFmt::BedGraphGz;
    else throw std::runtime_error("out_format must be bigwig|bedgraph|bedgraph.gz");

    // Pack parsed options.
    CovParams P;
    P.bin_size = bin_size; P.agg = agg; P.min_mapq = min_mapping_quality;
    P.include = sam_flag_include; P.exclude = exclude;
    P.min_frag = min_fragment_length; P.max_frag = max_fragment_length;
    P.extend = extend; P.extend_len = extend_reads;
    P.strand = strand; P.sfilter = sfilter;
    P.split = split; P.nstreams = split ? 2 : 1;
    P.overlap_filter = filter_by_overlap;
    P.has_blacklist = !blacklist.is_none();
    P.has_whitelist = !whitelist.is_none();
    P.touch_strand = touch_strand;
    P.collapse = collapse; P.collapsed_len = collapsed_length;
    P.scale = scale_factor;         // refined below (× normalization factor)
    P.scale_minus = scale_factor * scale_factor_minus;
    P.rpkm = false;
    // A configured fragment-length ceiling is also a guaranteed bound on how
    // far a retained alignment can influence this output window. Query at
    // least that far so boundary-crossing fragments are never missed even
    // when the user requests a smaller overlap-prefetch halo.
    P.halo = std::max<int64_t>(
        std::max(0, filter_by_overlap_halo), max_fragment_length);
    P.mate_prefetch_padding = filter_by_overlap_halo < 0 ? 10000 : 0;
    P.window_bins = coverage_window_bins(bin_size, window_size);

    // Index required for random-access windows (shared read-only by producers).
    hts_idx_t *shared_idx = sam_index_load(file.get(), bam_path.c_str());
    if (!shared_idx) {
        throw std::runtime_error("no usable index for '" + bam_path +
            "'. Coordinate-sort and index the BAM (samtools index).");
    }
    IdxPtr idxg(shared_idx);

    const bool admits_nonprimary =
        (P.exclude & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) !=
        (SAM_SECONDARY | SAM_SUPPLEMENTARY);
    const std::unordered_map<std::string, uint8_t> nonprimary_touch =
        (P.overlap_filter && (P.has_whitelist || P.has_blacklist) &&
         admits_nonprimary)
            ? build_nonprimary_touch_index(
                  bam_path, P, blacklist_by_tid, whitelist_by_tid)
            : std::unordered_map<std::string, uint8_t>{};

    // Chromosome exclusions have one meaning in read-count and BPM denominators.
    std::vector<char> ignore(n_targets, 0);
    if (normalization != "none" && !ignore_for_normalization.is_none()) {
        for (auto o : ignore_for_normalization.cast<py::list>()) {
            const int tid = resolve_header_tid(hdr.get(), py::str(o).cast<std::string>());
            if (tid >= 0) ignore[tid] = 1;
        }
    }
    const bool read_normalization = normalization != "none" && normalization != "bpm";
    const bool filtered = normalization_denominator == "filtered";
    if (read_normalization && !filtered && normalization_denominator != "library")
        throw std::runtime_error("normalization_denominator must be library|filtered");
    const bool count_reads = read_normalization && (filtered || exact_scaling);
    const bool plan_fetches = extend || collapse != Collapse::None;
    double counted_denominator = 0;
    std::vector<std::vector<FetchBounds>> fetch_bounds;
    if (plan_fetches || count_reads) {
        std::unique_ptr<MateLookup> count_mates;
        if (count_reads && filtered)
            count_mates = std::make_unique<MateLookup>(
                bam_path, shared_idx, P, blacklist_by_tid, whitelist_by_tid, nonprimary_touch);
        // Both tasks inspect the same records. Keep their distinct eligibility
        // rules, but perform the BAM decode only once when both are requested.
        fetch_bounds = prepare_coverage(bam_path, P, lengths, plan_fetches,
            [&](const bam1_t *record) {
                if (!count_reads || ignore[record->core.tid] ||
                    (record->core.flag & SAM_READ_UNMAPPED)) return;
                if (filtered) {
                    if (P.overlap_filter && (P.has_whitelist || P.has_blacklist))
                        count_mates->prefetch_for_record(record);
                    if (count_mates->retained(record, count_mates->observe(record)))
                        counted_denominator += 1;
                } else if (!(record->core.flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY)))
                    counted_denominator += 1;
            });
        if (plan_fetches) P.fetch_bounds = &fetch_bounds;
    }

    // Open one output per stream (2 for --filterRNAstrand split).
    std::vector<std::string> out_paths = {out_path};
    if (P.split) out_paths.push_back(out_path_reverse);
    const bool is_bw = (fmt == OutFmt::BigWig);
    dtp::BigWigLibraryScope library(is_bw);
    std::vector<BigWigPtr> bws(P.nstreams);
    std::vector<FilePtr> fhs(P.nstreams);
    auto finalize = [&]() {
        bool failed = false;
        for (auto &b : bws) if (b) failed |= bwCloseChecked(b.release()) != 0;
        for (auto &f : fhs) if (f) failed |= std::fclose(f.release()) != 0;
        if (failed) throw std::runtime_error("failed finalizing coverage output");
    };
    for (int s = 0; s < P.nstreams; ++s) {
        if (is_bw) {
            bws[s].reset(bwOpen(out_paths[s].c_str(), nullptr, "w"));
            if (!bws[s]) throw std::runtime_error("could not open bigWig: " + out_paths[s]);
            bool ok = (bwCreateHdr(bws[s].get(), max_zooms) == 0);
            if (ok) bws[s]->writeBuffer->compressLevel = compression_level;
            if (ok) {
                chromList_t *cl = bwCreateChromList(names.data(), lengths.data(), n_targets);
                if (!cl) ok = false;
                else { bws[s]->cl = cl; if (bwWriteHdr(bws[s].get()) != 0) ok = false; }
            }
            if (!ok) throw std::runtime_error("bigWig header write failed");
        } else {
            fhs[s].reset(std::fopen(out_paths[s].c_str(), "wb"));
            if (!fhs[s]) throw std::runtime_error("could not open output: " + out_paths[s]);
        }
    }
    std::vector<StreamWriter> writers;
    writers.reserve(P.nstreams);
    for (int s = 0; s < P.nstreams; ++s)
        writers.push_back(StreamWriter{fmt, bws[s].get(), fhs[s].get(), compression_level,
                                       names, lengths, bin_size});

    // Tile every chromosome into windows (bounded memory per window). The default
    // (256K bins) keeps a window's accumulators small even at binSize 1 (~1-4 MB),
    // so total in-flight memory stays low regardless of thread count.
    const uint32_t window_bins = P.window_bins;
    std::vector<Window> windows;
    for (int t = 0; t < n_targets; ++t) {
        const uint32_t nb = static_cast<uint32_t>(
            (static_cast<uint64_t>(lengths[t]) + bin_size - 1) / bin_size);
        for (uint32_t b = 0; b < nb;) {
            const uint32_t end = static_cast<uint32_t>(
                std::min<uint64_t>(static_cast<uint64_t>(b) + window_bins, nb));
            windows.push_back({t, b, end});
            b = end;
        }
    }
    const std::size_t nwin = windows.size();
    const std::size_t nthreads = dtp::thread_count(threads, std::max<std::size_t>(nwin, 1));

    {
        // --- normalization: compute the scale factor (the P4 correctness fix) ---
        // The denominator D is the LIBRARY SIZE by default, counted independently of
        // the numerator filters and shared across outputs; `filtered` reproduces the
        // old post-filter behaviour.
        enum { N_NONE, N_CPM, N_RPKM, N_BPM } norm;
        if (normalization == "none") norm = N_NONE;
        else if (normalization == "cpm") norm = N_CPM;
        else if (normalization == "rpkm") norm = N_RPKM;
        else if (normalization == "bpm") norm = N_BPM;
        else throw std::runtime_error("normalization must be none|cpm|rpkm|bpm");

        if (norm == N_BPM) {
            // BPM (TPM-like): scale = 1e6 / (sum of all placed bin values). The total
            // is self-referential, so a first parallel pass computes it (unscaled)
            // before the write pass. For split, the pooled forward+reverse sum is the
            // shared denominator (both tracks comparable).
            CovParams Pt = P; Pt.scale = 1.0; Pt.scale_minus = 1.0;
            std::atomic<std::size_t> next{0};
            std::vector<double> partial(nthreads, 0.0);
            dtp::parallel_rows(nthreads, static_cast<int>(nthreads),
                [&](std::size_t ti) {
                    HtsFilePtr worker_file;
                    worker_file.reset(hts_open(bam_path.c_str(), "r"));
                    if (!worker_file.get())
                        throw std::runtime_error(
                            "BPM worker could not open BAM: " + bam_path);
                    BamRecordPtr record(bam_init1());
                    if (!record.get())
                        throw std::runtime_error(
                            "BPM worker could not allocate a BAM record");
                    std::vector<std::pair<int64_t, int64_t>> iv;
                    std::vector<std::vector<float>> vals;
                    CoverageScratch scratch;
                    MateLookup mates(bam_path, shared_idx, Pt, blacklist_by_tid,
                                     whitelist_by_tid, nonprimary_touch);
                    dtp::CompensatedSum sum;
                    while (!dtp::interruption_requested()) {
                        if (!dtp::running_in_parallel_worker())
                            dtp::check_python_signals();
                        const std::size_t w = next.fetch_add(
                            1, std::memory_order_relaxed);
                        if (w >= nwin) break;
                        if (ignore[windows[w].tid]) continue;
                        compute_window(worker_file.get(), shared_idx, windows[w], Pt,
                                       lengths[windows[w].tid],
                                       blacklist_by_tid[windows[w].tid],
                                       whitelist_by_tid[windows[w].tid],
                                       mates,
                                       record.get(), iv, vals, scratch);
                        for (const auto &v : vals)
                            for (float x : v) sum.add(x);
                    }
                    partial[ti] = sum.value();
                });
            dtp::CompensatedSum total_sum;
            for (double p : partial) total_sum.add(p);
            const double total = total_sum.value();
            if (total <= 0.0)
                throw std::runtime_error("BPM denominator is zero "
                                         "(no signal after --ignoreForNormalization)");
            const double f = 1e6 / total;
            P.scale = scale_factor * f;
            P.scale_minus = scale_factor * scale_factor_minus * f;
        } else if (norm != N_NONE) {
            double D = counted_denominator;
            if (!filtered && !exact_scaling) {
                // Library size straight from the index (mapped alignments/target).
                for (int t = 0; t < n_targets; ++t) {
                    if (ignore[t]) continue;
                    uint64_t mp = 0, un = 0;
                    if (hts_idx_get_stat(shared_idx, t, &mp, &un) >= 0)
                        D += static_cast<double>(mp);
                }
            }
            if (D <= 0.0)
                throw std::runtime_error("normalization denominator is zero "
                                         "(no reads after --ignoreForNormalization)");

            double f = 1.0;
            if (norm == N_CPM) f = 1e6 / D;
            else if (norm == N_RPKM) {
                f = 1e9 / D;
                P.rpkm = true;
            }
            P.scale = scale_factor * f;
            P.scale_minus = scale_factor * scale_factor_minus * f;
        }
        if (!std::isfinite(P.scale) || !std::isfinite(P.scale_minus))
            throw std::runtime_error(
                "normalization produced a non-finite scale factor");

        // Per-thread worker: its own BAM handle + record buffer (index shared).
        struct CovWorker {
            HtsFilePtr fp;
            BamRecordPtr rec;
            std::unique_ptr<MateLookup> mates;
            std::vector<std::pair<int64_t, int64_t>> iv;
            CoverageScratch scratch;
        };
        run_windows(
            windows, nthreads, P.nstreams, writers,
            [&]() {
                CovWorker w;
                w.fp.reset(hts_open(bam_path.c_str(), "r"));
                if (!w.fp) throw std::runtime_error("worker could not open BAM");
                w.rec.reset(bam_init1());
                if (!w.rec) throw std::runtime_error("could not allocate a BAM record");
                w.mates = std::make_unique<MateLookup>(
                    bam_path, shared_idx, P, blacklist_by_tid, whitelist_by_tid,
                    nonprimary_touch);
                return w;
            },
            [&](CovWorker &w, const Window &win, std::vector<std::vector<float>> &out) {
                compute_window(w.fp.get(), shared_idx, win, P, lengths[win.tid],
                               blacklist_by_tid[win.tid], whitelist_by_tid[win.tid],
                               *w.mates, w.rec.get(), w.iv, out, w.scratch);
            });

        for (auto &wr : writers) wr.finish();
    }

    finalize();
}

// --- P9: library-strandedness inference (RSeQC infer_experiment analogue) ------
//
// For reads over each user-supplied known-strand probe gene, tally how often the
// read's read1-implied transcript strand (under the "forward" = read1-sense
// hypothesis) matches the gene strand. A high fraction => forward/secondstrand
// library; low => reverse/firststrand (dUTP); ~0.5 => unstranded. Index-
// accelerated: only the probe regions are read.
std::string strand_record_key(const bam1_t *rec, bool canonical_template) {
    std::string result(bam_get_qname(rec));
    result.push_back('\0');
    const uint8_t *rg = bam_aux_get(rec, "RG");
    const char *group = rg ? bam_aux2Z(rg) : nullptr;
    if (group) result.append(group);
    result.push_back('\0');

    int64_t coordinates[4];
    if (canonical_template && (rec->core.flag & SAM_SECOND_IN_PAIR) &&
        !(rec->core.flag & SAM_FIRST_IN_PAIR)) {
        coordinates[0] = rec->core.mtid;
        coordinates[1] = rec->core.mpos;
        coordinates[2] = rec->core.tid;
        coordinates[3] = rec->core.pos;
    } else {
        coordinates[0] = rec->core.tid;
        coordinates[1] = rec->core.pos;
        coordinates[2] = rec->core.mtid;
        coordinates[3] = rec->core.mpos;
    }
    result.append(reinterpret_cast<const char *>(coordinates), sizeof(coordinates));
    if (!canonical_template) {
        const uint16_t identity_flags = rec->core.flag &
            (SAM_READ_PAIRED | SAM_FIRST_IN_PAIR | SAM_SECOND_IN_PAIR |
             SAM_READ_REVERSE | SAM_MATE_REVERSE);
        result.append(reinterpret_cast<const char *>(&identity_flags),
                      sizeof(identity_flags));
        const uint32_t *cigar = bam_get_cigar(rec);
        result.append(reinterpret_cast<const char *>(cigar),
                      rec->core.n_cigar * sizeof(uint32_t));
    }
    return result;
}

void record_strand_evidence(std::unordered_map<std::string, int8_t> &evidence,
                            std::string key, int8_t observation) {
    auto found = evidence.find(key);
    if (found == evidence.end()) {
        evidence.emplace(std::move(key), observation);
    } else if (found->second != observation) {
        found->second = 0;  // one template/read supports conflicting probe strands
    }
}

py::dict infer_strandedness(const std::string &bam_path, py::object regions,
                            int min_mapping_quality, bool require_proper_pair,
                            bool ignore_duplicates, uint16_t sam_flag_include,
                            uint16_t sam_flag_exclude, std::size_t sample_size) {
    if (!sample_size || sample_size > 1000000)
        throw std::invalid_argument("strandedness sample_size must be in 1..1000000");
    if (min_mapping_quality < 0 || min_mapping_quality > 255)
        throw std::runtime_error("min_mapping_quality must be in 0..255");
    HtsFilePtr file;
    file.reset(hts_open(bam_path.c_str(), "r"));
    if (!file.get()) throw std::runtime_error("could not open BAM: " + bam_path);
    HdrPtr hdr;
    hdr.reset(sam_hdr_read(file.get()));
    if (!hdr.get()) throw std::runtime_error("could not read header: " + bam_path);
    IdxPtr idx;
    idx.reset(sam_index_load(file.get(), bam_path.c_str()));
    if (!idx.get())
        throw std::runtime_error("no usable index for '" + bam_path +
                                 "'. Coordinate-sort and index the BAM.");

    struct Probe { int tid; int64_t start, end; int gene; };
    std::vector<Probe> probes;
    for (auto r : regions.cast<py::list>()) {
        auto t = r.cast<py::tuple>();
        const int tid = resolve_header_tid(hdr.get(), t[0].cast<std::string>());
        const int64_t start = t[1].cast<int64_t>(), end = t[2].cast<int64_t>();
        const int gene = t[3].cast<int>();
        if (tid >= 0 && start >= 0 && end > start && (gene == 1 || gene == -1))
            probes.push_back({tid, start, end, gene});
    }
    auto order = [](const Probe &a, const Probe &b) {
        return std::tie(a.tid, a.start, a.end, a.gene) < std::tie(b.tid, b.start, b.end, b.gene);
    };
    std::sort(probes.begin(), probes.end(), order);
    probes.erase(std::unique(probes.begin(), probes.end(), [&](const Probe &a, const Probe &b) {
        return !order(a, b) && !order(b, a);
    }), probes.end());

    // This is library-type inference, not a population census. Spread a fixed
    // observation budget over at most 128 index-weighted genomic strata instead of consuming
    // every alignment under every gene. Both accepted identities and examined
    // records are capped; sparse/repetitive/filtered inputs cannot make it scan
    // the whole BAM. Small inputs still visit all probes and all observations.
    const std::size_t strata = std::min<std::size_t>({probes.size(), sample_size, 128});
    const std::size_t base_quota = strata ? (sample_size + strata - 1) / strata : 0;
    std::vector<std::pair<std::size_t, std::size_t>> selection;
    if (probes.size() <= strata) {
        for (std::size_t i = 0; i < probes.size(); ++i) selection.emplace_back(i, base_quota);
    } else if (strata) {
        // Equal quotas per gene overweight low-expression genes. Compressed
        // index spans are a cheap proxy for read mass, used only to select
        // sampling strata; orientation counts still count each sampled template
        // once. No full-file scan is needed to construct these weights.
        std::vector<double> cumulative;
        double mass = 0;
        for (const auto &probe : probes) {
            IteratorPtr it;
            it.reset(sam_itr_queryi(idx.get(), probe.tid, probe.start, probe.end));
            double weight = 0;
            if (it.get()) for (int j = 0; j < it.get()->n_off; ++j) {
                const auto &chunk = it.get()->off[j];
                if (chunk.v > chunk.u) weight += static_cast<double>(chunk.v - chunk.u) / 65536;
            }
            mass += std::max(1.0, weight);
            cumulative.push_back(mass);
        }
        for (std::size_t q = 0; q < strata; ++q) {
            const double target = (q + .5) * mass / strata;
            const auto i = static_cast<std::size_t>(
                std::lower_bound(cumulative.begin(), cumulative.end(), target) - cumulative.begin());
            if (!selection.empty() && selection.back().first == i) selection.back().second += base_quota;
            else selection.emplace_back(i, base_quota);
        }
    }
    std::unordered_map<std::string, int8_t> proper_evidence, single_evidence, alignment_evidence;
    bool saw_proper = false;
    BamRecordPtr record(bam_init1());
    if (!record.get()) throw std::runtime_error("could not allocate a BAM record");
    uint64_t scanned = 0;
    std::size_t stored = 0, sampled_probes = 0;
    std::vector<std::pair<int64_t, int64_t>> aligned_blocks;
    {
    py::gil_scoped_release release;
    for (const auto &stratum : selection) {
        if (stored >= sample_size) break;
        dtp::check_python_signals();
        const auto &probe = probes[stratum.first];
        const auto quota = stratum.second;
        const std::size_t scan_quota = std::max<std::size_t>(quota * 50, 256);
        IteratorPtr iterator;
        iterator.reset(sam_itr_queryi(idx.get(), probe.tid, probe.start, probe.end));
        if (!iterator.get()) continue;
        ++sampled_probes;
        std::size_t accepted = 0, examined = 0;
        while (accepted < quota && examined < scan_quota && stored < sample_size &&
               checked_bam_status(sam_itr_next(file.get(), iterator.get(), record.get())) >= 0) {
            ++examined;
            if ((++scanned & 4095u) == 0) dtp::check_python_signals();
            const uint16_t flag = record.get()->core.flag;
            if (flag & (SAM_READ_UNMAPPED | SAM_SECONDARY | SAM_SUPPLEMENTARY | SAM_QC_FAIL)) continue;
            if (record.get()->core.qual < min_mapping_quality) continue;
            if (ignore_duplicates && (flag & SAM_DUPLICATE)) continue;
            if (flag & sam_flag_exclude) continue;
            if (sam_flag_include && (flag & sam_flag_include) != sam_flag_include) continue;
            // The BAM iterator retrieves reference spans, including skipped
            // introns. Only aligned bases can supply probe-strand evidence.
            extract_blocks(record.get(), aligned_blocks);
            if (std::none_of(aligned_blocks.begin(), aligned_blocks.end(),
                    [&](const auto &block) {
                        return block.first < probe.end && probe.start < block.second;
                    })) continue;
            const int8_t observation = transcript_strand(record.get(), Strand::Forward) == probe.gene ? 1 : -1;
            auto *evidence = &alignment_evidence;
            bool canonical = false;
            if (require_proper_pair) {
                if (flag & SAM_READ_PAIRED) {
                    if (!is_proper_fragment(record.get())) continue;
                    saw_proper = true;
                    evidence = &proper_evidence;
                    canonical = true;
                } else evidence = &single_evidence;
            }
            std::string key = strand_record_key(record.get(), canonical);
            if (key.size() > 4096) continue;
            const auto before = evidence->size();
            record_strand_evidence(*evidence, std::move(key), observation);
            const auto added = evidence->size() - before;
            stored += added; accepted += added;
        }
    }
    }

    const auto &chosen = require_proper_pair
        ? (saw_proper ? proper_evidence : single_evidence)
        : alignment_evidence;
    uint64_t fwd = 0, rev = 0;
    for (const auto &item : chosen) {
        if (item.second > 0) ++fwd;
        else if (item.second < 0) ++rev;
    }
    const uint64_t total = fwd + rev;

    py::dict out;
    out["forward"] = fwd;
    out["reverse"] = rev;
    out["total"] = total;
    out["sample_strategy"] = "index_weighted_probes";
    out["sample_limit"] = sample_size;
    out["sampled_probes"] = sampled_probes;
    out["alignments_examined"] = scanned;
    out["forward_fraction"] = total ? static_cast<double>(fwd) / total : 0.0;
    return out;
}

inline uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

uint64_t sampling_priority(const bam1_t *rec, uint64_t ordinal, uint64_t salt) {
    uint64_t hash = 1469598103934665603ULL;
    const unsigned char *name = reinterpret_cast<const unsigned char *>(bam_get_qname(rec));
    for (; *name; ++name) {
        hash ^= *name;
        hash *= 1099511628211ULL;
    }
    const uint8_t *rg_tag = bam_aux_get(rec, "RG");
    const unsigned char *group = reinterpret_cast<const unsigned char *>(
        rg_tag ? bam_aux2Z(rg_tag) : nullptr);
    if (group) {
        for (; *group; ++group) {
            hash ^= *group;
            hash *= 1099511628211ULL;
        }
    }
    const uint64_t fields[] = {
        static_cast<uint32_t>(rec->core.tid),
        static_cast<uint32_t>(rec->core.pos),
        static_cast<uint32_t>(rec->core.mtid),
        static_cast<uint32_t>(rec->core.mpos),
        static_cast<uint16_t>(rec->core.flag), ordinal, salt,
    };
    for (const uint64_t field : fields)
        hash = splitmix64(hash ^ splitmix64(field));
    return hash;
}

using LengthSample = std::priority_queue<std::pair<uint64_t, int64_t>>;

void offer_length(LengthSample &values, std::size_t capacity,
                  uint64_t priority, int64_t value) {
    const std::pair<uint64_t, int64_t> candidate{priority, value};
    if (values.size() < capacity) {
        values.push(candidate);
    } else if (candidate < values.top()) {
        values.pop();
        values.push(candidate);
    }
}

// Median fragment/read length from a deterministic bounded sample distributed
// over the complete BAM.  Coordinate-sorted prefixes can be chromosome- or
// library-component-biased, so stopping after the first N alignments is not a
// representative estimator.  A stable hash-priority reservoir remains O(sample)
// memory and produces byte-identical choices across supported platforms.
double estimate_length(const std::string &bam_path, int sample, bool prefer_fragment) {
    if (sample <= 0) throw std::runtime_error("sample must be > 0");
    HtsFilePtr file;
    file.reset(hts_open(bam_path.c_str(), "r"));
    if (!file.get()) throw std::runtime_error("could not open BAM: " + bam_path);
    HdrPtr hdr;
    hdr.reset(sam_hdr_read(file.get()));
    if (!hdr.get()) throw std::runtime_error("could not read header: " + bam_path);

    LengthSample tlens, rlens;
    BamRecordPtr record(bam_init1());
    if (!record.get()) throw std::runtime_error("could not allocate a BAM record");
    uint64_t scanned = 0;
    while (checked_bam_status(sam_read1(file.get(), hdr.get(), record.get())) >= 0) {
        if ((++scanned & 4095u) == 0)
            dtp::check_python_signals();
        if (record.get()->core.flag & SAM_READ_UNMAPPED) continue;
        if (record.get()->core.flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) continue;
        const int64_t rl = inferred_query_length(record.get());
        if (rl > 0) {
            offer_length(rlens, static_cast<std::size_t>(sample),
                         sampling_priority(record.get(), scanned, 0x72656164ULL), rl);
        }
        if (prefer_fragment && is_proper_fragment(record.get()) && record.get()->core.isize > 0)
            offer_length(tlens, static_cast<std::size_t>(sample),
                         sampling_priority(record.get(), scanned, 0x66726167ULL),
                         record.get()->core.isize);
    }

    auto median = [](LengthSample &sampled) -> double {
        std::vector<int64_t> v;
        v.reserve(sampled.size());
        while (!sampled.empty()) {
            v.push_back(sampled.top().second);
            sampled.pop();
        }
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const size_t n = v.size(), m = n / 2;
        return (n & 1) ? static_cast<double>(v[m]) :
            static_cast<double>(v[m - 1]) / 2 + static_cast<double>(v[m]) / 2;
    };
    if (!tlens.empty()) return median(tlens);
    if (!rlens.empty()) return median(rlens);
    return 0;
}

}  // namespace

PYBIND11_MODULE(_coverage, m) {
    m.doc() = "Native bamCoverage backend.";
    m.def("htslib_version", &htslib_version,
          "Version string of the vendored htslib.");
    m.def("bam_index_stats", &bam_index_stats, py::arg("path"),
          "Per-reference mapped/unmapped counts from the BAM index (no scan).");
    m.def("resolve_bam_index_path", &resolve_bam_index_path, py::arg("path"),
          "Path of the local BAM index the selected backend will load, or ''.");
    m.def("coverage_window_bins", &coverage_window_bins,
          py::arg("bin_size"), py::arg("requested") = 0,
          "Production-window bin count after the 262144-bin and 1-Mbp caps.");
    m.def("estimate_fragment_length", [](const std::string &path, int sample) {
              return estimate_length(path, sample, true);
          },
          py::arg("bam_path"), py::arg("sample") = 10000,
          "Median fragment length (proper-pair |TLEN|, else read length) from "
          "a representative bounded sample across the BAM.");
    m.def("infer_strandedness", &infer_strandedness, py::arg("bam_path"),
          py::arg("regions"), py::arg("min_mapping_quality") = 0,
          py::arg("require_proper_pair") = false,
          py::arg("ignore_duplicates") = false,
          py::arg("sam_flag_include") = 0,
          py::arg("sam_flag_exclude") = 0,
          py::arg("sample_size") = 20000,
          "Sample independent read/template sense vs antisense evidence over "
          "known-strand probe genes (regions: list of (chrom, start, end, "
          "strand +1/-1)). Proper paired templates are counted once; if none "
          "are sampled, unpaired reads are used. At most sample_size identities "
          "are retained across up to 128 index-weighted genomic probe strata.");
    m.def("bam_coverage_bigwig", &bam_coverage_bigwig,
          py::arg("bam_path"), py::arg("out_path"),
          py::arg("out_path_reverse") = "",
          py::arg("out_format") = "bigwig",
          py::arg("bin_size") = 50,
          py::arg("aggregation") = "mean",
          py::arg("min_mapping_quality") = 0,
          py::arg("sam_flag_include") = 0,
          py::arg("sam_flag_exclude") = 0,
          py::arg("filter_mode") = "primary",
          py::arg("ignore_duplicates") = false,
          py::arg("min_fragment_length") = 0,
          py::arg("max_fragment_length") = 0,
          py::arg("extend_reads") = 0,
          py::arg("strandedness") = "none",
          py::arg("filter_rna_strand") = "none",
          py::arg("collapse") = "none",
          py::arg("collapsed_length") = 1,
          py::arg("blacklist") = py::none(),
          py::arg("whitelist") = py::none(),
          py::arg("threads") = 1,
          py::arg("window_size") = 0,
          py::arg("normalization") = "none",
          py::arg("normalization_denominator") = "library",
          py::arg("exact_scaling") = false,
          py::arg("scale_factor") = 1.0,
          py::arg("scale_factor_minus") = 1.0,
          py::arg("ignore_for_normalization") = py::none(),
          py::arg("max_zooms") = 10,
          py::arg("compression_level") = -1,
          py::arg("filter_by_overlap") = false,
          py::arg("feature_touch_strand") = "ignore",
          py::arg("filter_by_overlap_halo") = -1,
          "Binned coverage -> bigWig with inline filters, read extension/collapse, "
          "and strand selection. Multicore via a producer/single-writer streaming "
          "model (threads<=0 = all cores). aggregation: mean|sum|count. "
          "scale_factor_minus is an additional multiplier on scale_factor for the "
          "minus stream.");
}
