#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "matrix_read.hpp"
#include "parallel.hpp"
#include "numeric.hpp"
#include "filter_policy.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using dtp::parallel_rows;
using dtp::thread_count;

void validate_stream_options(std::size_t buffer_size, int compression_level) {
    if (buffer_size == 0)
        throw std::runtime_error("buffer size must be positive");
    if (compression_level < 0 || compression_level > 9)
        throw std::runtime_error("compression level must be between 0 and 9");
}

void validate_boundaries(const std::vector<std::size_t> &boundaries,
                         const char *name, bool strictly_increasing) {
    if (boundaries.size() < 2 || boundaries.front() != 0)
        throw std::runtime_error(std::string(name) +
                                 " must contain at least [0, end]");
    for (std::size_t index = 1; index < boundaries.size(); ++index) {
        if (boundaries[index] < boundaries[index - 1] ||
            (strictly_increasing && boundaries[index] == boundaries[index - 1]))
            throw std::runtime_error(std::string(name) +
                                     " are not valid monotonic boundaries");
    }
}

// Write one gzip member or the same raw text. Empty chunks write nothing.
void write_block(std::ofstream &out, const std::string &text, int level,
                 bool compressed) {
    if (text.empty()) return;
    if (compressed) {
        const auto bytes = dtp::gzip_member(text, level);
        out.write(bytes.data(), bytes.size());
    } else {
        out.write(text.data(), text.size());
    }
    if (!out) throw std::runtime_error("failed writing matrix block");
}

// One row builder for both modes. The serial path reuses its scratch and
// string; parallel rows own independent scratch and are joined in input order.
template <class MakeScratch, class Build>
std::string build_rows(const char *base,
                       const std::vector<std::pair<std::size_t, std::size_t>> &lines,
                       int threads, MakeScratch make_scratch, Build build) {
    std::string block;
    if (lines.empty()) return block;
    if (thread_count(threads, lines.size()) <= 1) {
        block.reserve((lines.back().second - lines.front().first) + 64);
        auto scratch = make_scratch();
        std::string row;
        for (const auto &span : lines) {
            build(base + span.first, base + span.second, row, scratch);
            block += row;
        }
    } else {
        std::vector<std::string> rows(lines.size());
        parallel_rows(lines.size(), threads, [&](std::size_t local) {
            auto scratch = make_scratch();
            const auto span = lines[local];
            build(base + span.first, base + span.second, rows[local], scratch);
        });
        for (const auto &row : rows) block += row;
    }
    return block;
}

// Shared driver for the two row-dropping filters. Streams survivor rows
// verbatim to `temp_path` (body only, no header) as per-chunk gzip members or
// raw text and returns rebuilt group boundaries. `keep(begin, end)` decides one row.
template <class Keep>
std::vector<std::size_t> filter_body(
    const std::string &input_path, const std::string &temp_path,
    const std::vector<std::size_t> &group_boundaries, int num_threads,
    std::size_t buffer_size, int compression_level, bool compressed, Keep keep) {
    validate_stream_options(buffer_size, compression_level);
    validate_boundaries(group_boundaries, "group boundaries", false);
    if (input_path == temp_path)
        throw std::runtime_error("temporary output must differ from input");
    const std::size_t groups = group_boundaries.size() - 1;

    std::ofstream temp(temp_path, std::ios::binary);
    if (!temp) throw std::runtime_error("cannot open temporary body: " + temp_path);

    std::vector<std::size_t> kept_per_group(groups, 0);
    std::size_t global_row = 0;
    std::size_t group_cursor = 0;

    {
        py::gil_scoped_release release;
        dtp::read_matrix_chunks(
            input_path, buffer_size,
            [&](const char *, const char *, const dtp::MatrixSchema &schema) {
                if (schema.rows != group_boundaries.back())
                    throw std::runtime_error("group boundaries do not match input rows");
            },
            [&](const char *base,
                const std::vector<std::pair<std::size_t, std::size_t>> &lines,
                const dtp::MatrixSchema &schema) {
                if (global_row + lines.size() > group_boundaries.back())
                    throw std::runtime_error(
                        "matrix contains more regions than its header");
                std::vector<char> survive(lines.size());
                parallel_rows(lines.size(), num_threads, [&](std::size_t local) {
                    try {
                        const auto b = lines[local];
                        survive[local] =
                            keep(base + b.first, base + b.second, schema.bins) ? 1 : 0;
                    } catch (const std::exception &error) {
                        throw std::runtime_error("row " +
                            std::to_string(global_row + local + 1) + ": " +
                            error.what());
                    }
                });
                std::string block;
                for (std::size_t local = 0; local < lines.size(); ++local) {
                    const std::size_t row = global_row + local;
                    while (group_cursor + 1 < group_boundaries.size() &&
                           row >= group_boundaries[group_cursor + 1])
                        ++group_cursor;
                    if (!survive[local]) continue;
                    const auto b = lines[local];
                    block.append(base + b.first, base + b.second);
                    block += '\n';
                    ++kept_per_group[group_cursor];
                }
                write_block(temp, block, compression_level, compressed);
                global_row += lines.size();
            });
        if (global_row != group_boundaries.back())
            throw std::runtime_error(
                "expected " + std::to_string(group_boundaries.back()) +
                " regions but read " + std::to_string(global_row));
    }
    dtp::close_matrix_output(temp, temp_path);

    std::vector<std::size_t> bounds(group_boundaries.size());
    bounds[0] = 0;
    for (std::size_t g = 0; g < groups; ++g)
        bounds[g + 1] = bounds[g] + kept_per_group[g];
    return bounds;
}

// filterStrand: keep rows whose strand field (index 5) matches `strand`.
std::vector<std::size_t> stream_filter_strand(
    const std::string &input_path, const std::string &temp_path,
    const std::vector<std::size_t> &group_boundaries, const std::string &strand,
    int num_threads, std::size_t buffer_size, int compression_level,
    bool compressed) {
    return filter_body(
        input_path, temp_path, group_boundaries, num_threads, buffer_size,
        compression_level, compressed, [&](const char *begin, const char *end, std::size_t bins) {
            dtp::MatrixFields fields;
            dtp::split_fields(begin, end, fields);
            dtp::validate_matrix_fields(fields, bins);
            const auto &field = fields[5];
            return static_cast<std::size_t>(field.second - field.first) == strand.size() &&
                   std::equal(field.first, field.second, strand.begin());
        });
}

// How a sample's bins in one row are reduced before comparison to [lo, hi].
using Stat = dtp::Statistic;
using SampleEval = dtp::SampleEval;

// Reuse the values parsed by row validation, with the resident reader's
// float32 semantics. Median uses its own reusable ordering scratch buffer.
SampleEval eval_sample(const std::vector<float> &values,
                       std::size_t col0, std::size_t col1, int stat, double lo,
                       double hi, std::vector<double> &scratch) {
    return dtp::filter_sample(col1 - col0, [&](std::size_t i) {
        return static_cast<double>(values[col0 + i]);
    }, static_cast<Stat>(stat), lo, hi, false, true, &scratch);
}

// filterValues (drop mode): drop a row when any gating sample fails the value
// or per-sample NaN filter, or (AllBins) when every gated bin is NaN. With
// statistic=PerBin, NaN mode Keep, and every sample gating, this reduces to the
// classic "drop a region if any bin is out of [min, max]" (all-missing kept).
std::vector<std::size_t> stream_filter_values(
    const std::string &input_path, const std::string &temp_path,
    const std::vector<std::size_t> &group_boundaries,
    const std::vector<std::size_t> &sample_boundaries,
    const std::vector<std::size_t> &filter_samples, int statistic, int nan_mode,
    bool has_min, double min_value, bool has_max, double max_value,
    int num_threads, std::size_t buffer_size, int compression_level,
    bool compressed) {
    const double lo = has_min ? min_value : -std::numeric_limits<double>::infinity();
    const double hi = has_max ? max_value : std::numeric_limits<double>::infinity();
    const dtp::FilterPolicy policy(sample_boundaries, filter_samples,
        static_cast<Stat>(statistic), nan_mode, dtp::FilterAction::RemoveRegion, lo, hi);
    return filter_body(
        input_path, temp_path, group_boundaries, num_threads, buffer_size,
        compression_level, compressed, [&](const char *begin, const char *end, std::size_t bins) {
            std::vector<std::pair<const char *, const char *>> fields;
            dtp::split_fields(begin, end, fields);
            if (bins != sample_boundaries.back())
                throw std::runtime_error("sample boundaries do not match input bins");
            std::vector<float> values(bins);
            dtp::validate_matrix_fields(fields, bins, nullptr, values.data());
            std::vector<double> scratch;
            return policy.keep_row([&](std::size_t sample) {
                return eval_sample(values, sample_boundaries[sample],
                    sample_boundaries[sample + 1], statistic, lo, hi, scratch);
            }, [](std::size_t) {});
        });
}

// filterValues (mask mode): never removes a row. For each gating sample that
// fails, replace that sample's bins in the row with "nan"; other samples and
// all region metadata pass through verbatim. Order-preserving on rows, so the
// header (unchanged) is written up front and the body streamed one chunk at a
// time (single-buffer fast path, as in subset_columns).
void stream_filter_values_mask(
    const std::string &input_path, const std::string &output_path,
    const std::string &header_line,
    const std::vector<std::size_t> &sample_boundaries,
    const std::vector<std::size_t> &filter_samples, int statistic, int nan_mode,
    bool has_min, double min_value, bool has_max, double max_value,
    int num_threads, std::size_t buffer_size, int compression_level,
    bool compressed) {
    validate_stream_options(buffer_size, compression_level);
    if (input_path == output_path)
        throw std::runtime_error("output must differ from input");
    const double lo = has_min ? min_value : -std::numeric_limits<double>::infinity();
    const double hi = has_max ? max_value : std::numeric_limits<double>::infinity();
    const dtp::FilterPolicy policy(sample_boundaries, filter_samples,
        static_cast<Stat>(statistic), nan_mode, dtp::FilterAction::MaskSample, lo, hi);
    const std::size_t num_samples =
        sample_boundaries.empty() ? 0 : sample_boundaries.size() - 1;

    std::ofstream out(output_path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output matrix: " + output_path);
    write_block(out, "@" + header_line + "\n", compression_level, compressed);

    struct MaskScratch {
        std::vector<std::pair<const char *, const char *>> fields;
        std::vector<char> sample_masked;
        std::vector<double> scratch;
        std::vector<float> values;
    };
    auto build_row = [&](const char *begin, const char *end, std::string &row,
                         MaskScratch &state) {
        auto &fields = state.fields;
        auto &sample_masked = state.sample_masked;
        auto &scratch = state.scratch;
        auto &values = state.values;
        dtp::split_fields(begin, end, fields);
        const std::size_t bins = sample_boundaries.back();
        values.resize(bins);
        dtp::validate_matrix_fields(fields, bins, nullptr, values.data());
        std::fill(sample_masked.begin(), sample_masked.end(), 0);
        policy.keep_row([&](std::size_t sample) {
            return eval_sample(values, sample_boundaries[sample],
                sample_boundaries[sample + 1], statistic, lo, hi, scratch);
        }, [&](std::size_t sample) { sample_masked[sample] = 1; });
        row.clear();
        for (std::size_t f = 0; f < 6; ++f) {
            if (f) row += '\t';
            row.append(fields[f].first, fields[f].second);
        }
        std::size_t sample = 0;
        for (std::size_t c = 0; c < bins; ++c) {
            while (sample + 1 < sample_boundaries.size() &&
                   c >= sample_boundaries[sample + 1])
                ++sample;
            row += '\t';
            if (sample < num_samples && sample_masked[sample])
                row += "nan";
            else
                row.append(fields[c + 6].first, fields[c + 6].second);
        }
        row += '\n';
    };

    py::gil_scoped_release release;
    dtp::read_matrix_chunks(
        input_path, buffer_size,
        [&](const char *, const char *, const dtp::MatrixSchema &schema) {
            if (schema.bins != sample_boundaries.back())
                throw std::runtime_error("sample boundaries do not match input bins");
        },
        [&](const char *base,
            const std::vector<std::pair<std::size_t, std::size_t>> &lines,
            const dtp::MatrixSchema &) {
            const auto block = build_rows(base, lines, num_threads,
                [&]() { return MaskScratch{{}, std::vector<char>(num_samples), {}, {}}; },
                build_row);
            write_block(out, block, compression_level, compressed);
        });
    dtp::close_matrix_output(out, output_path);
}

// Column subset/reorder: order-preserving on rows. Writes the full output file
// (pre-serialised header line first, then the transformed body streamed one
// chunk at a time). `column_order` holds 0-based bin indices in output order.
void stream_subset_columns(const std::string &input_path,
                           const std::string &output_path,
                           const std::string &header_line,
                           const std::vector<std::size_t> &column_order,
                           int num_threads, std::size_t buffer_size,
                           int compression_level, bool compressed) {
    validate_stream_options(buffer_size, compression_level);
    if (input_path == output_path)
        throw std::runtime_error("output must differ from input");
    if (column_order.empty())
        throw std::runtime_error("column order must select at least one column");
    std::vector<std::size_t> sorted_columns(column_order);
    std::sort(sorted_columns.begin(), sorted_columns.end());
    if (std::adjacent_find(sorted_columns.begin(), sorted_columns.end()) !=
        sorted_columns.end())
        throw std::runtime_error("column order cannot contain duplicates");
    std::ofstream out(output_path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output matrix: " + output_path);
    write_block(out, "@" + header_line + "\n", compression_level, compressed);

    std::size_t input_bins = 0;
    struct SubsetScratch {
        std::vector<std::pair<const char *, const char *>> fields;
    };
    // Transform one row's fields into `row` (its previous contents are cleared).
    auto build_row = [&](const char *begin, const char *end, std::string &row,
                         SubsetScratch &state) {
        auto &fields = state.fields;
        dtp::split_fields(begin, end, fields);
        dtp::validate_matrix_fields(fields, input_bins);
        const std::size_t bins = input_bins;
        row.clear();
        for (std::size_t f = 0; f < 6; ++f) {
            if (f) row += '\t';
            row.append(fields[f].first, fields[f].second);
        }
        for (const std::size_t column : column_order) {
            if (column >= bins)
                throw std::runtime_error(
                    "column index out of range for this matrix");
            row += '\t';
            row.append(fields[column + 6].first, fields[column + 6].second);
        }
        row += '\n';
    };

    py::gil_scoped_release release;
    dtp::read_matrix_chunks(
        input_path, buffer_size,
        [&](const char *, const char *, const dtp::MatrixSchema &schema) {
            input_bins = schema.bins;
            if (sorted_columns.back() >= input_bins)
                throw std::runtime_error("column index out of range for this matrix");
        },
        [&](const char *base,
            const std::vector<std::pair<std::size_t, std::size_t>> &lines,
            const dtp::MatrixSchema &) {
            const auto block = build_rows(base, lines, num_threads,
                []() { return SubsetScratch{}; }, build_row);
            write_block(out, block, compression_level, compressed);
        });
    dtp::close_matrix_output(out, output_path);
}

}  // namespace

PYBIND11_MODULE(_compute_matrix_stream, module) {
    module.doc() = "Chunked-streaming computeMatrix filters (order-preserving ops)";
    module.attr("supports_multicore") = true;
    module.def("filter_strand", &stream_filter_strand, py::arg("input_path"),
               py::arg("temp_path"), py::arg("group_boundaries"),
               py::arg("strand"), py::arg("num_threads") = 1,
               py::arg("buffer_size") = 1024 * 1024,
               py::arg("compression_level") = 6,
               py::arg("compressed") = true);
    module.def("filter_values", &stream_filter_values, py::arg("input_path"),
               py::arg("temp_path"), py::arg("group_boundaries"),
               py::arg("sample_boundaries"), py::arg("filter_samples"),
               py::arg("statistic"), py::arg("nan_mode"), py::arg("has_min"),
               py::arg("min_value"), py::arg("has_max"), py::arg("max_value"),
               py::arg("num_threads") = 1, py::arg("buffer_size") = 1024 * 1024,
               py::arg("compression_level") = 6,
               py::arg("compressed") = true);
    module.def("filter_values_mask", &stream_filter_values_mask,
               py::arg("input_path"), py::arg("output_path"),
               py::arg("header_line"), py::arg("sample_boundaries"),
               py::arg("filter_samples"), py::arg("statistic"), py::arg("nan_mode"),
               py::arg("has_min"), py::arg("min_value"), py::arg("has_max"),
               py::arg("max_value"), py::arg("num_threads") = 1,
               py::arg("buffer_size") = 1024 * 1024,
               py::arg("compression_level") = 6,
               py::arg("compressed") = true);
    module.def("subset_columns", &stream_subset_columns, py::arg("input_path"),
               py::arg("output_path"), py::arg("header_line"),
               py::arg("column_order"), py::arg("num_threads") = 1,
               py::arg("buffer_size") = 1024 * 1024,
               py::arg("compression_level") = 6,
               py::arg("compressed") = true);
}
