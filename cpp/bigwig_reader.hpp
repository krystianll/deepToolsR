#pragma once

#include "libBigWig/bigWig.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace dtp {
using BigWigIntervals = std::unique_ptr<bwOverlappingIntervals_t,
                                      decltype(&bwDestroyOverlappingIntervals)>;

inline BigWigIntervals read_bigwig_intervals(bigWigFile_t *file,
                                            const std::string &path,
                                            const char *chrom,
                                            uint32_t start, uint32_t end) {
    bwOverlappingIntervals_t *raw = nullptr;
    const int status = bwGetOverlappingIntervalsChecked(file, chrom, start, end, &raw);
    BigWigIntervals result(raw, bwDestroyOverlappingIntervals);
    if (status)
        throw std::runtime_error("Failed reading bigWig '" + path + "' at " +
            chrom + ":" + std::to_string(start) + "-" + std::to_string(end) +
            ": I/O, truncated/corrupt data, or allocation error");
    return result;
}
} // namespace dtp
