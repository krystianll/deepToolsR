#pragma once

// The one bigWig writer: opens the file, writes the header and the data
// intervals, and builds the zoom levels from the intervals it was given instead
// of letting libBigWig re-read and decompress the whole file at close.
//
// Every interval is also appended (16 bytes) to a bounded buffer that spills
// sequentially to a temporary file beside the output. At close, libBigWig picks
// the zoom levels as before (bwSetZoomBuilder hook, see
// libBigWig/CHANGES_deepToolsR.md) and build_level() turns the captured runs
// into each level's records in parallel, one task per level. The records match
// libBigWig 0.4.8 field for field, except that every record's sum and sum of
// squares is written: libBigWig leaves them 0 in the last record of each block
// and of each level, so zoom-level means read low there.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <libdeflate.h>

#include "handles.hpp"
#include "parallel.hpp"

namespace dtp {

// A scratch file beside the output's own temporary, removed on every exit path.
struct TempFile {
    std::string path;
    TempFile() = default;
    explicit TempFile(std::string p) : path(std::move(p)) {}
    TempFile(TempFile &&other) noexcept : path(std::move(other.path)) { other.path.clear(); }
    TempFile &operator=(TempFile &&other) noexcept { std::swap(path, other.path); return *this; }
    TempFile(const TempFile &) = delete;
    TempFile &operator=(const TempFile &) = delete;
    ~TempFile() { if (!path.empty()) std::remove(path.c_str()); }
};

class BigWigOutput {
public:
    BigWigOutput(const std::string &path, const std::vector<const char *> &names,
                 const std::vector<uint32_t> &lengths, int max_zooms,
                 int compression_level, int threads)
        : path_(path), spill_path_(path + ".zoomruns"), compression_level_(compression_level),
          threads_(threads) {
        // libBigWig cannot write an empty chromosome list (it divides by its size).
        if (names.empty())
            throw std::runtime_error("cannot write a bigWig: the input has no chromosomes");
        errno = 0;
        fp_.reset(bwOpen(path.c_str(), nullptr, "w"));
        if (!fp_) throw std::runtime_error("could not open bigWig: " + dtp::display_path(path));
        bool ok = bwCreateHdr(fp_.get(), max_zooms) == 0;
        if (ok) fp_->writeBuffer->compressLevel = compression_level;
        if (ok) {
            chromList_t *cl = bwCreateChromList(names.data(), lengths.data(),
                                                static_cast<int64_t>(names.size()));
            ok = cl != nullptr;
            if (ok) { fp_->cl = cl; ok = bwWriteHdr(fp_.get()) == 0; }
        }
        if (!ok) throw write_error("failed writing bigWig header", path);
    }
    BigWigOutput(BigWigOutput &&) = default;
    BigWigOutput &operator=(BigWigOutput &&) = default;
    // Abandoned without close(): no zoom levels, the caller discards the file.
    ~BigWigOutput() { if (fp_) fp_->hdr->nLevels = 0; }

    // Intervals of chromosome `tid`, in genomic order across calls.
    void add(uint32_t tid, const uint32_t *start, const uint32_t *end,
             const float *value, std::size_t n) {
        if (!n) return;
        const uint32_t length = fp_->cl->len[tid];
        for (std::size_t i = 0; i < n; ++i)
            if (start[i] >= end[i] || end[i] > length)
                throw std::runtime_error(
                    "invalid bigWig interval " + std::string(fp_->cl->chrom[tid]) + ":" +
                    std::to_string(start[i]) + "-" + std::to_string(end[i]) + " (chromosome length " +
                    std::to_string(length) + ")");
        chrom_.assign(n, fp_->cl->chrom[tid]);
        errno = 0;
        if (bwAddIntervals(fp_.get(), chrom_.data(), start, end, value,
                           static_cast<uint32_t>(n)) != 0)
            throw write_error("failed writing bigWig", path_);
        if (!fp_->hdr->nLevels) return;
        if (tid < last_tid_ || (tid == last_tid_ && start[0] < last_end_))
            throw std::logic_error("bigWig intervals must be added in genomic order");
        for (std::size_t i = 0; i < n; ++i) {
            runs_.push_back({tid, start[i], end[i], value[i]});
            if (runs_.size() == kBufferRuns) spill();
        }
        last_tid_ = tid;
        last_end_ = end[n - 1];
    }

    // Finalize (zoom levels, indices) and close; true when that failed.
    bool close() {
        errno = 0;
        if (out_ && std::fclose(out_.release()) != 0)
            throw write_error("could not write the zoom-level spill file", spill_.path);
        bwSetZoomBuilder(fp_.get(), &BigWigOutput::build_levels, this);
        const bool failed = bwCloseChecked(fp_.release()) != 0;
        levels_.clear();
        if (error_) std::rethrow_exception(error_);
        return failed;
    }

private:
    struct Run { uint32_t tid, start, end; float value; };
    struct Record { uint32_t tid, start, end, count; float min, max, sum, sumsq; };
    static_assert(sizeof(Run) == 16 && sizeof(Record) == 32, "packed file layouts");
    struct Level {
        std::vector<bwZoomBlock_t> blocks;
        std::vector<std::vector<uint8_t>> data;
    };
    static constexpr std::size_t kBufferRuns = (8u << 20) / sizeof(Run);
    static constexpr std::size_t kReadRuns = (1u << 20) / sizeof(Run);

    void spill() {
        if (!out_) {
            spill_ = TempFile(spill_path_);
            out_.reset(std::fopen(spill_.path.c_str(), "wb"));
            if (!out_) throw std::runtime_error("could not create the zoom-level spill file " +
                                                display_path(spill_.path));
        }
        errno = 0;
        if (std::fwrite(runs_.data(), sizeof(Run), runs_.size(), out_.get()) != runs_.size())
            throw write_error("could not write the zoom-level spill file", spill_.path);
        runs_.clear();
    }

    static void poll() {
        if (!running_in_parallel_worker()) check_python_signals();
        else if (interruption_requested())
            throw std::runtime_error("zoom-level construction cancelled");
    }

    // Every captured run in order: the spill file, then the in-memory tail.
    template <class Visit>
    void for_each_run(Visit visit) const {
        if (!spill_.path.empty()) {
            FilePtr in(std::fopen(spill_.path.c_str(), "rb"));
            if (!in) throw std::runtime_error("could not open the zoom-level spill file");
            std::vector<Run> chunk(kReadRuns);
            std::size_t got;
            while ((got = std::fread(chunk.data(), sizeof(Run), chunk.size(), in.get())) > 0) {
                poll();
                for (std::size_t i = 0; i < got; ++i) visit(chunk[i]);
            }
            if (std::ferror(in.get()))
                throw std::runtime_error("could not read the zoom-level spill file");
        }
        poll();
        for (const Run &run : runs_) visit(run);
    }

    static uint32_t overlap(uint32_t tid0, uint32_t start0, uint32_t end0,
                            uint32_t tid1, uint32_t start1, uint32_t end1) {
        if (tid0 != tid1 || end0 <= start1 || end1 <= start0) return 0;
        return (end0 <= end1 ? end0 : end1) - (start1 > start0 ? start1 : start0);
    }

    // libBigWig's updateInterval/addIntervalValue with the sums always written.
    // A block holds `capacity` records (libBigWig stops one short of
    // itemsPerBlock); the first run that finds a block full opens the next one.
    void build_level(uint32_t zoom, std::size_t capacity, Level &level) const {
        const int lvl = compression_level_ < 0 ? 6 : (compression_level_ > 12 ? 12 : compression_level_);
        std::unique_ptr<libdeflate_compressor, void (*)(libdeflate_compressor *)> compressor(
            libdeflate_alloc_compressor(lvl), libdeflate_free_compressor);
        if (!compressor) throw std::runtime_error("libdeflate_alloc_compressor failed");
        std::vector<Record> block;
        block.reserve(capacity);
        double sum = 0.0, sumsq = 0.0;
        auto close_record = [&] {
            block.back().sum = static_cast<float>(sum);
            block.back().sumsq = static_cast<float>(sumsq);
            sum = sumsq = 0.0;
        };
        auto flush = [&] {
            close_record();
            const std::size_t bytes = block.size() * sizeof(Record);
            std::vector<uint8_t> out(libdeflate_zlib_compress_bound(compressor.get(), bytes));
            const std::size_t n = libdeflate_zlib_compress(compressor.get(), block.data(), bytes,
                                                           out.data(), out.size());
            if (!n) throw std::runtime_error("zoom block compression failed");
            out.resize(n);
            out.shrink_to_fit();  // keep only the compressed bytes until written
            level.blocks.push_back({block.front().tid, block.front().start, block.back().tid,
                                    block.back().end, static_cast<uint32_t>(n),
                                    static_cast<uint32_t>(block.size()), out.data()});
            level.data.push_back(std::move(out));
            block.clear();
        };
        for_each_run([&](const Run &run) {
            const uint32_t tid = run.tid, end = run.end;
            const float value = run.value;
            uint32_t start = run.start;
            while (start < end) {
                uint32_t size = zoom;
                if (start + size < start) size = UINT32_MAX - start;
                if (block.size() == capacity) {
                    flush();
                } else if (!block.empty()) {
                    Record &last = block.back();
                    const uint32_t rv = overlap(last.tid, last.start, last.start + size,
                                                tid, start, end);
                    if (rv) {
                        last.end = start + rv;
                        last.count += rv;
                        if (last.min > value) last.min = value;
                        if (last.max < value) last.max = value;
                        sum += static_cast<double>(rv) * value;
                        sumsq += rv * std::pow(static_cast<double>(value), 2.0);
                        start += rv;
                        continue;
                    }
                    close_record();
                }
                const uint32_t rv = overlap(tid, start, start + size, tid, start, end);
                block.push_back({tid, start, start + size < end ? start + size : end, rv,
                                 value, value, 0.0f, 0.0f});
                sum += static_cast<double>(rv) * value;
                sumsq += rv * std::pow(static_cast<double>(value), 2.0);
                start += rv;
            }
        });
        if (!block.empty()) flush();
    }

    static int build_levels(void *ctx, const uint32_t *zooms, uint16_t n_levels,
                            uint32_t items_per_block, const bwZoomBlock_t **blocks,
                            uint64_t *n_blocks) {
        auto *self = static_cast<BigWigOutput *>(ctx);
        try {
            self->levels_.assign(n_levels, Level{});
            parallel_rows(n_levels, self->threads_, [&](std::size_t i) {
                self->build_level(zooms[i], items_per_block - 1, self->levels_[i]);
            });
            for (uint16_t i = 0; i < n_levels; ++i) {
                blocks[i] = self->levels_[i].blocks.data();
                n_blocks[i] = self->levels_[i].blocks.size();
            }
            return 0;
        } catch (...) {
            self->error_ = std::current_exception();
            return 1;
        }
    }

    // Declared so that destruction closes the bigWig, then the spill stream,
    // then removes the spill file.
    std::string path_, spill_path_;
    TempFile spill_;
    FilePtr out_;
    BigWigPtr fp_;
    int compression_level_;
    int threads_;
    std::vector<const char *> chrom_;
    std::vector<Run> runs_;
    uint32_t last_tid_ = 0, last_end_ = 0;
    std::vector<Level> levels_;
    std::exception_ptr error_;
};

}  // namespace dtp
