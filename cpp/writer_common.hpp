#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace dtp {

inline void validate_writer_options(int max_zooms, int compression_level) {
    if (max_zooms < 0)
        throw std::runtime_error("max_zooms must be >= 0");
    if (compression_level != -1 &&
        (compression_level < 1 || compression_level > 12))
        throw std::runtime_error("compression_level must be -1 or 1..12");
}

inline float checked_float(double value, const char *context) {
    if (!std::isfinite(value) ||
        std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))
        throw std::runtime_error(std::string(context) +
                                 " is non-finite or outside float32 range");
    return static_cast<float>(value);
}

}  // namespace dtp
