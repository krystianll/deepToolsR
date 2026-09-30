#pragma once

// Shared text primitives for the computeMatrix file format: field parsing,
// numeric formatting, single-member gzip, and a chunked gzip reader that
// yields the JSON header once and then data-line spans. Both the whole-matrix
// reader/writer (compute_matrix_io.cpp) and the chunked-streaming filter engine
// (compute_matrix_stream.cpp) sit on these, so the boundary/carry handling and
// the on-disk number formatting live in exactly one place.

#include <zlib.h>
#include <libdeflate.h>
#include <fast_float/fast_float.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace dtp {

// Destructors cannot report a buffered write or close failure. Finish output
// explicitly before the caller commits its temporary file.
inline void close_matrix_output(std::ofstream &output, const std::string &path) {
    output.flush();
    output.close();
    if (!output)
        throw std::runtime_error("failed to finish output matrix: " + path);
}

// ---- scalar field parsing -------------------------------------------------

template <class Integer>
inline bool parse_integer(const char *begin, const char *end, Integer &value) {
    const auto result = std::from_chars(begin, end, value);
    return result.ec == std::errc() && result.ptr == end;
}

inline bool parse_double(const char *begin, const char *end, double &value) {
    const auto result = fast_float::from_chars(begin, end, value);
    return result.ec == std::errc() && result.ptr == end;
}

inline std::vector<std::int64_t> parse_coordinates(const char *begin,
                                                   const char *end) {
    std::vector<std::int64_t> output;
    const char *field = begin;
    while (field <= end) {
        const char *comma = std::find(field, end, ',');
        std::int64_t value = 0;
        if (field == comma || !parse_integer(field, comma, value))
            throw std::runtime_error("invalid comma-separated coordinate");
        output.push_back(value);
        if (comma == end) break;
        field = comma + 1;
    }
    return output;
}

// Split a tab-separated row into [begin,end) field spans, stripping a trailing
// '\r'. Returns the number of fields appended (fields is cleared first).
inline std::size_t split_fields(
    const char *begin, const char *end,
    std::vector<std::pair<const char *, const char *>> &fields) {
    if (end > begin && end[-1] == '\r') --end;
    fields.clear();
    const char *field = begin;
    for (const char *cursor = begin;; ++cursor) {
        if (cursor == end || *cursor == '\t') {
            fields.emplace_back(field, cursor);
            if (cursor == end) break;
            field = cursor + 1;
        }
    }
    return fields.size();
}

using MatrixFields = std::vector<std::pair<const char *, const char *>>;
using GenomicBlocks = std::vector<std::pair<std::int64_t, std::int64_t>>;

// One schema/numeric contract for resident reads, native filters, and the
// bounded batches used by Python's text-copy operations. A caller needing the
// numbers receives float32 values directly, with no second parse for filtering.
inline void validate_matrix_fields(const MatrixFields &fields, std::size_t bins,
                                    GenomicBlocks *blocks = nullptr,
                                    float *values = nullptr) {
    if (bins > std::numeric_limits<std::size_t>::max() - 6 || fields.size() != bins + 6)
        throw std::runtime_error("matrix row width does not match sample boundaries");
    if (blocks) blocks->clear();
    auto start = fields[1].first, end = fields[2].first;
    for (;;) {
        const auto next_start = std::find(start, fields[1].second, ',');
        const auto next_end = std::find(end, fields[2].second, ',');
        std::int64_t first, last;
        if (!parse_integer(start, next_start, first) ||
            !parse_integer(end, next_end, last))
            throw std::runtime_error("invalid comma-separated coordinate");
        if (first < 0 || last < first)
            throw std::runtime_error("invalid genomic block boundaries");
        if (blocks) blocks->emplace_back(first, last);
        const bool starts_done = next_start == fields[1].second;
        const bool ends_done = next_end == fields[2].second;
        if (starts_done != ends_done)
            throw std::runtime_error("start/end block counts differ");
        if (starts_done) break;
        start = next_start + 1;
        end = next_end + 1;
    }
    for (std::size_t column = 0; column < bins; ++column) {
        double value;
        const auto &field = fields[column + 6];
        if (!parse_double(field.first, field.second, value))
            throw std::runtime_error("invalid numeric value in bin " + std::to_string(column + 1));
        if (std::isfinite(value) && std::abs(value) > std::numeric_limits<float>::max())
            throw std::runtime_error("numeric value exceeds float32 range in bin " + std::to_string(column + 1));
        if (values) values[column] = std::isfinite(value) ? static_cast<float>(value)
            : std::numeric_limits<float>::quiet_NaN();
    }
}

// ---- numeric formatting (must match the on-disk save_matrix format) -------

inline void append_integer(std::string &output, std::int64_t value) {
    char buffer[32];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (result.ec != std::errc())
        throw std::runtime_error("integer formatting failed");
    output.append(buffer, result.ptr);
}

inline void append_double(std::string &output, double value) {
    // NaN is the sole missing-value sentinel; both infinities format as "nan".
    if (!std::isfinite(value)) {
        output += "nan";
        return;
    }
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                      std::chars_format::fixed, 6);
    if (result.ec != std::errc())
        throw std::runtime_error("float formatting failed");
    output.append(buffer, result.ptr);
}

// ---- single-member gzip ---------------------------------------------------

// deepToolsR: libdeflate gzip compression (2-3x faster than zlib). Output is
// standard gzip, byte-different from zlib but decompressing to identical content.
inline std::string gzip_member(const std::string &input, int compression_level) {
    const int lvl = compression_level < 0 ? 6
                    : (compression_level > 12 ? 12 : compression_level);
    struct libdeflate_compressor *c = libdeflate_alloc_compressor(lvl);
    if (!c) throw std::runtime_error("could not initialize gzip compressor");
    std::string output;
    output.resize(libdeflate_gzip_compress_bound(c, input.size()));
    const size_t n = libdeflate_gzip_compress(c, input.data(), input.size(),
                                              output.data(), output.size());
    libdeflate_free_compressor(c);
    if (n == 0) throw std::runtime_error("gzip compression failed");
    output.resize(n);
    return output;
}

// ---- chunked gzip/plain reader --------------------------------------------

// Drive a gzip-compressed or plain computeMatrix file. zlib transparently
// handles either representation. `on_header(begin, end)` is called exactly
// once with the header line contents (leading '@' and trailing newline
// excluded). Then `on_lines(base, spans)` is called for each buffered group of
// complete data lines: `base` points into a buffer valid only for that call and
// each span is a [start, end) byte offset of one data line (newline excluded,
// empty lines skipped). The final partial line (no trailing newline) is
// delivered as its own one-line group. Runs the gzip read loop directly; the
// caller may release the GIL around it.
template <class HeaderFn, class LinesFn>
void read_gzip_matrix(const std::string &path, std::size_t buffer_size,
                      HeaderFn on_header, LinesFn on_lines) {
    if (buffer_size < 65536) buffer_size = 65536;
    if (buffer_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("matrix read buffer exceeds zlib's size limit");

    std::vector<char> buffer(buffer_size);
    gzFile input = gzopen(path.c_str(), "rb");
    if (!input) throw std::runtime_error("cannot open matrix file: " + path);

    std::string carry;
    bool header_seen = false;
    int bytes = 0;

    try {
        while ((bytes = gzread(input, buffer.data(),
                               static_cast<unsigned>(buffer.size()))) > 0) {
            // zlib >= 1.3.2 reports a truncated stream ("unexpected end of
            // file") only alongside the read that hit it and clears the error
            // on the next gzread, so check after every read, not just the last.
            int read_code = Z_OK;
            const char *read_message = gzerror(input, &read_code);
            if (read_code != Z_OK)
                throw std::runtime_error(read_message ? read_message
                                                      : "gzip read error");
            std::string chunk;
            chunk.reserve(carry.size() + static_cast<std::size_t>(bytes));
            chunk.append(carry);
            chunk.append(buffer.data(), static_cast<std::size_t>(bytes));
            carry.clear();

            const auto final_newline = chunk.find_last_of('\n');
            if (final_newline == std::string::npos) {
                carry = std::move(chunk);
                continue;
            }
            carry.assign(chunk.data() + final_newline + 1,
                         chunk.size() - final_newline - 1);
            chunk.resize(final_newline + 1);

            std::size_t data_start = 0;
            if (!header_seen) {
                const auto newline = chunk.find('\n');
                if (newline == std::string::npos || chunk.empty() ||
                    chunk[0] != '@')
                    throw std::runtime_error("missing computeMatrix JSON header");
                on_header(chunk.data() + 1, chunk.data() + newline);
                data_start = newline + 1;
                header_seen = true;
            }

            std::vector<std::pair<std::size_t, std::size_t>> lines;
            for (std::size_t start = data_start; start < chunk.size();) {
                const auto newline = chunk.find('\n', start);
                if (newline == std::string::npos) break;
                if (newline > start && !(newline == start + 1 && chunk[start] == '\r'))
                    lines.emplace_back(start, newline);
                start = newline + 1;
            }
            if (!lines.empty()) on_lines(chunk.data(), lines);
        }
        int code = Z_OK;
        const char *message = gzerror(input, &code);
        if (bytes < 0 || code != Z_OK) {
            throw std::runtime_error(message ? message : "gzip read error");
        }
        if (!header_seen)
            throw std::runtime_error("missing computeMatrix JSON header");
        if (!carry.empty() && carry != "\r") {
            std::vector<std::pair<std::size_t, std::size_t>> lines{
                {0, carry.size()}};
            on_lines(carry.data(), lines);
        }
    } catch (...) {
        gzclose(input);
        throw;
    }
    if (gzclose(input) != Z_OK)
        throw std::runtime_error("failed closing matrix input: " + path);
}

}  // namespace dtp
