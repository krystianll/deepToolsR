#pragma once

#include "libBigWig/bigWig.h"
#include "display_path.hpp"
#include "handles.hpp"
#include <memory>
#include <stdexcept>
#include <string>

namespace dtp {
// Open a bigWig for reading. A damaged chromosome table (missing, or a name
// that is not valid UTF-8) is an error rather than a broken output.
inline BigWigPtr open_bigwig(const std::string &path) {
    BigWigPtr file(bwOpen(path.c_str(), nullptr, "r"));
    if (!file || !file->cl || !file->hdr)
        throw std::runtime_error("could not open bigWig: " + display_path(path));
    for (int64_t c = 0; c < file->cl->nKeys; ++c) {
        const std::string name = file->cl->chrom[c];
        if (valid_utf8(name) != name)
            throw std::runtime_error("bigWig '" + display_path(path) + "' has a chromosome name that "
                                     "is not valid UTF-8 (" + valid_utf8(name) + "); the file is damaged");
    }
    return file;
}

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
        throw std::runtime_error("Failed reading bigWig '" + dtp::display_path(path) + "' at " +
            chrom + ":" + std::to_string(start) + "-" + std::to_string(end) +
            ": I/O, truncated/corrupt data, or allocation error");
    return result;
}
} // namespace dtp
