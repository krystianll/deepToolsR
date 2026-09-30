#pragma once

// Shared streaming-output machinery for the bigWig-writing tools (bamCoverage,
// bigWig merge, ...): genomic-window tiling, the three output backends behind a
// single run-length-merging StreamWriter, and a parallel producer / single-writer
// driver with a bounded ring reorder buffer. Nothing here depends on htslib, so
// bigWig-only tools can use it without pulling in the BAM reader.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <libdeflate.h>

#include "libBigWig/bigWig.h"
#include "parallel.hpp"

namespace dtp {

enum class OutFmt { BigWig, BedGraph, BedGraphGz };

// One-shot gzip of a buffer via libdeflate. Concatenated members form a valid
// multi-member gzip stream, so callers can flush chunk-by-chunk (bounded memory).
inline std::string gzip_chunk(const std::string &in, int level) {
    const int lvl = level < 0 ? 6 : (level > 12 ? 12 : level);
    libdeflate_compressor *c = libdeflate_alloc_compressor(lvl);
    if (!c) throw std::runtime_error("libdeflate_alloc_compressor failed");
    std::string out;
    out.resize(libdeflate_gzip_compress_bound(c, in.size()));
    const size_t n = libdeflate_gzip_compress(c, in.data(), in.size(),
                                              &out[0], out.size());
    libdeflate_free_compressor(c);
    if (n == 0) throw std::runtime_error("gzip compression failed");
    out.resize(n);
    return out;
}

// A tile of bins [bin0, bin1) on reference `tid`.
struct Window { int tid; uint32_t bin0; uint32_t bin1; };

// Emits windows in genomic order, run-length-merging across window boundaries so
// the output matches whole-chromosome RLE. Backends: bigWig (libBigWig), plain
// bedGraph, or gzipped bedGraph (chunked libdeflate members).
struct StreamWriter {
    OutFmt fmt;
    bigWigFile_t *bw = nullptr;   // BigWig
    std::FILE *fh = nullptr;      // BedGraph / BedGraphGz
    int gz_level = -1;            // BedGraphGz
    const std::vector<const char *> &names;
    const std::vector<uint32_t> &lengths;
    uint32_t bin_size;
    int cur_tid = -1;
    bool have = false;
    uint32_t ps = 0, pe = 0;
    float pv = 0.0f;
    std::vector<uint32_t> starts, ends;   // bigWig batch
    std::vector<float> vals;
    std::string textbuf;                  // bedGraph line buffer

    void bw_flush(int tid) {
        if (starts.empty()) return;
        std::vector<const char *> chr(starts.size(), names[tid]);
        if (bwAddIntervals(bw, chr.data(), starts.data(), ends.data(), vals.data(),
                           static_cast<uint32_t>(starts.size())) != 0)
            throw std::runtime_error("bwAddIntervals failed");
        starts.clear(); ends.clear(); vals.clear();
    }
    void text_flush(bool force) {
        if (textbuf.empty()) return;
        if (!force && textbuf.size() < (1u << 20)) return;  // ~1 MB chunks
        if (fmt == OutFmt::BedGraphGz) {
            const std::string z = gzip_chunk(textbuf, gz_level);
            if (std::fwrite(z.data(), 1, z.size(), fh) != z.size())
                throw std::runtime_error("bedGraph.gz write failed");
        } else {
            if (std::fwrite(textbuf.data(), 1, textbuf.size(), fh) != textbuf.size())
                throw std::runtime_error("bedGraph write failed");
        }
        textbuf.clear();
    }
    void emit_run(uint32_t s, uint32_t e, float v) {
        if (fmt == OutFmt::BigWig) {
            starts.push_back(s); ends.push_back(e); vals.push_back(v);
        } else {
            char buf[64];
            const int n = std::snprintf(buf, sizeof(buf), "%s\t%u\t%u\t%g\n",
                                        names[cur_tid], s, e, static_cast<double>(v));
            if (n < 0)
                throw std::runtime_error("bedGraph row formatting failed");
            if (static_cast<std::size_t>(n) < sizeof(buf)) {
                textbuf.append(buf, static_cast<std::size_t>(n));
            } else {
                // snprintf() reports the required payload length when the stack
                // buffer is too small. Retry with room for that payload and its
                // terminating NUL; append only the payload to the output buffer.
                std::vector<char> line(static_cast<std::size_t>(n) + 1);
                const int written = std::snprintf(
                    line.data(), line.size(), "%s\t%u\t%u\t%g\n",
                    names[cur_tid], s, e, static_cast<double>(v));
                if (written != n)
                    throw std::runtime_error("bedGraph row formatting failed");
                textbuf.append(line.data(), static_cast<std::size_t>(written));
            }
        }
    }

    void emit_window(int tid, uint32_t bin0, const std::vector<float> &v) {
        if (tid != cur_tid) {
            if (have) { emit_run(ps, pe, pv); have = false; }
            if (fmt == OutFmt::BigWig) bw_flush(cur_tid);
            cur_tid = tid;
        }
        const uint32_t nb = static_cast<uint32_t>(v.size());
        uint32_t i = 0;
        while (i < nb) {
            uint32_t j = i + 1;
            while (j < nb && v[j] == v[i]) ++j;
            const uint64_t s64 = static_cast<uint64_t>(bin0 + i) * bin_size;
            const uint64_t e64 = static_cast<uint64_t>(bin0 + j) * bin_size;
            const uint32_t s = static_cast<uint32_t>(
                std::min<uint64_t>(s64, lengths[tid]));
            const uint32_t e = static_cast<uint32_t>(
                std::min<uint64_t>(e64, lengths[tid]));
            if (have && pe == s && pv == v[i]) { pe = e; }
            else { if (have) emit_run(ps, pe, pv); ps = s; pe = e; pv = v[i]; have = true; }
            i = j;
        }
        if (fmt == OutFmt::BigWig) bw_flush(tid);
        else text_flush(false);
    }
    void finish() {
        if (have) { emit_run(ps, pe, pv); have = false; }
        if (fmt == OutFmt::BigWig) bw_flush(cur_tid);
        else text_flush(true);
    }
};

// Run `compute` over `windows` and emit results to `writers` in genomic order.
// `make_worker()` builds a per-thread state (its destructor releases resources);
// `compute(worker, window, out)` fills `out[stream][bin]`. Serial when
// nthreads<=1; otherwise a producer pool feeds a single writer through a bounded
// ring reorder buffer (memory O(C windows), deterministic output).
template <class MakeWorker, class Compute>
inline void run_windows(const std::vector<Window> &windows, std::size_t nthreads,
                        int nstreams, std::vector<StreamWriter> &writers,
                        MakeWorker make_worker, Compute compute) {
    const std::size_t nwin = windows.size();
    if (nthreads <= 1) {
        auto ws = make_worker();
        std::vector<std::vector<float>> vals;
        for (const auto &w : windows) {
            check_python_signals();
            compute(ws, w, vals);
            for (int s = 0; s < nstreams; ++s)
                writers[s].emit_window(w.tid, w.bin0, vals[s]);
        }
        return;
    }

    const std::size_t C =
        std::min<std::size_t>(std::max<std::size_t>(nthreads + 4, 4), 16);
    struct Slot { bool ready = false; std::size_t gidx = SIZE_MAX;
                  std::vector<std::vector<float>> values; };
    std::vector<Slot> ring(C);
    std::mutex m;
    std::condition_variable cv_ready, cv_free;
    std::atomic<std::size_t> next_win{0};
    std::atomic<bool> cancelled{false};
    std::size_t writer_cursor = 0;  // guarded by m
    std::exception_ptr err;

    auto producer = [&]() {
        parallel_cancel_flag = &cancelled;
        try {
            auto ws = make_worker();
            std::vector<std::vector<float>> vals;
            while (!cancelled.load(std::memory_order_relaxed)) {
                const std::size_t w = next_win.fetch_add(1, std::memory_order_relaxed);
                if (w >= nwin) break;
                compute(ws, windows[w], vals);
                std::unique_lock<std::mutex> lk(m);
                cv_free.wait(lk, [&] { return err || w < writer_cursor + C; });
                if (err) break;
                Slot &s = ring[w % C];
                s.gidx = w; s.values.swap(vals); s.ready = true;
                cv_ready.notify_all();
            }
        } catch (...) {
            std::unique_lock<std::mutex> lk(m);
            if (!err) err = std::current_exception();
            cancelled.store(true, std::memory_order_relaxed);
            cv_ready.notify_all(); cv_free.notify_all();
        }
        parallel_cancel_flag = nullptr;
    };

    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    try {
        for (std::size_t i = 0; i < nthreads; ++i) pool.emplace_back(producer);
    } catch (...) {
        // Do not unwind across joinable threads if the OS refuses to create the
        // complete pool. Wake any producers already waiting on the ring buffer.
        {
            std::lock_guard<std::mutex> lk(m);
            if (!err) err = std::current_exception();
            cancelled.store(true, std::memory_order_relaxed);
        }
        cv_ready.notify_all();
        cv_free.notify_all();
        for (auto &th : pool) th.join();
        std::rethrow_exception(err);
    }

    // Return the preceding window's storage through the ring to a producer.
    // This retains bounded ownership while avoiding fresh output allocations.
    std::vector<std::vector<float>> vals;
    for (std::size_t wc = 0; wc < nwin; ++wc) {
        {
            std::unique_lock<std::mutex> lk(m);
            while (!err) {
                Slot &s = ring[wc % C];
                if (s.ready && s.gidx == wc) break;
                cv_ready.wait_for(lk, std::chrono::milliseconds(50));
                if (err) break;
                lk.unlock();
                try {
                    check_python_signals();
                } catch (...) {
                    lk.lock();
                    if (!err) err = std::current_exception();
                    cancelled.store(true, std::memory_order_relaxed);
                    cv_free.notify_all();
                    break;
                }
                lk.lock();
            }
            if (err) break;
            Slot &s = ring[wc % C];
            vals.swap(s.values);
            s.ready = false; s.gidx = SIZE_MAX;
            writer_cursor = wc + 1;
        }
        cv_free.notify_all();
        try {
            for (int s = 0; s < nstreams; ++s)
                writers[s].emit_window(windows[wc].tid, windows[wc].bin0, vals[s]);
        } catch (...) {
            std::unique_lock<std::mutex> lk(m);
            if (!err) err = std::current_exception();
            cancelled.store(true, std::memory_order_relaxed);
            cv_free.notify_all();
            break;
        }
    }
    for (auto &th : pool) th.join();
    if (err) std::rethrow_exception(err);
}

}  // namespace dtp
