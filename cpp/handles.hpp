#pragma once

#include <memory>
#include <stdexcept>
#include <cstdio>

#include "libBigWig/bigWig.h"

#ifdef DTP_HTSLIB_HANDLES
#include <htslib/hts.h>
#include <htslib/sam.h>
#endif

namespace dtp {

struct BigWigDeleter {
    void operator()(bigWigFile_t *handle) const noexcept {
        if (handle) bwClose(handle);
    }
};
using BigWigPtr = std::unique_ptr<bigWigFile_t, BigWigDeleter>;

struct FileDeleter {
    void operator()(std::FILE *handle) const noexcept {
        if (handle) std::fclose(handle);
    }
};
using FilePtr = std::unique_ptr<std::FILE, FileDeleter>;

// Declare before handles: their destructors then run before library teardown.
class BigWigLibraryScope {
public:
    explicit BigWigLibraryScope(bool active = true) : active_(active) {
        if (active_ && bwInit(1 << 17) != 0)
            throw std::runtime_error("bwInit failed");
    }
    ~BigWigLibraryScope() noexcept { if (active_) bwCleanup(); }
    BigWigLibraryScope(const BigWigLibraryScope &) = delete;
    BigWigLibraryScope &operator=(const BigWigLibraryScope &) = delete;
private:
    bool active_;
};

#ifdef DTP_HTSLIB_HANDLES
struct HtsFileDeleter {
    void operator()(htsFile *handle) const noexcept {
        if (handle) hts_close(handle);
    }
};
struct HdrDeleter {
    void operator()(sam_hdr_t *handle) const noexcept {
        if (handle) sam_hdr_destroy(handle);
    }
};
struct IdxDeleter {
    void operator()(hts_idx_t *handle) const noexcept {
        if (handle) hts_idx_destroy(handle);
    }
};
struct BamRecordDeleter {
    void operator()(bam1_t *handle) const noexcept {
        if (handle) bam_destroy1(handle);
    }
};
struct IteratorDeleter {
    void operator()(hts_itr_t *handle) const noexcept {
        if (handle) hts_itr_destroy(handle);
    }
};
using HtsFilePtr = std::unique_ptr<htsFile, HtsFileDeleter>;
using HdrPtr = std::unique_ptr<sam_hdr_t, HdrDeleter>;
using IdxPtr = std::unique_ptr<hts_idx_t, IdxDeleter>;
using BamRecordPtr = std::unique_ptr<bam1_t, BamRecordDeleter>;
using IteratorPtr = std::unique_ptr<hts_itr_t, IteratorDeleter>;
#endif

}  // namespace dtp
