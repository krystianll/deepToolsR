#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "libBigWig/bigWig.h"
#include "native_matrix.hpp"
#include "parallel.hpp"
#include "numeric.hpp"
#include "bigwig_reader.hpp"
#include "handles.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double NAN_VALUE = std::numeric_limits<double>::quiet_NaN();

inline void poll_native_work(std::size_t index) {
    if ((index & 1023u) != 0) return;
    if (dtp::running_in_parallel_worker()) {
        if (dtp::interruption_requested())
            throw std::runtime_error("native matrix computation cancelled");
    } else {
        dtp::check_python_signals();
    }
}

using dtp::BigWigPtr;

struct Segment {
    // Signed because requested flanks may extend before chromosome position 0;
    // wide because bigWig chromosome lengths span the complete uint32 range.
    int64_t start;
    int64_t end;
};

struct Zone {
    std::vector<Segment> segments;
    int32_t bins;
};

struct OutputSource {
    std::vector<int32_t> path_indices;
    std::vector<double> scales;
};

struct RowSpec {
    std::string chrom;
    std::vector<Zone> zones;
    std::vector<OutputSource> outputs;
    int32_t pad_left = 0;
    int32_t pad_right = 0;
    bool reverse = false;
    int32_t unpadded_bins = 0;
};

// Owns one contiguous native matrix.  Exposing the standard Python buffer
// protocol lets NumPy and the native writer borrow the same allocation without
// coupling the two extension modules to a shared pybind11 class registration.
using dtp::NativeMatrix;

using BinStatistic = dtp::Statistic;
BinStatistic parse_statistic(const std::string& value) {
    return dtp::parse_statistic(value, "bin");
}

std::string chromosome_alias(const std::string& chrom) {
    if (chrom.compare(0, 3, "chr") == 0) {
        const auto suffix = chrom.substr(3);
        return suffix == "M" ? "MT" : suffix;
    }
    return chrom == "MT" ? "chrM" : "chr" + chrom;
}

bool chromosome_size(bigWigFile_t* file, std::string& chrom,
                     uint32_t& size) {
    if (!file || !file->cl) return false;
    for (int32_t i = 0; i < file->cl->nKeys; ++i) {
        if (chrom == file->cl->chrom[i]) {
            size = file->cl->len[i];
            return true;
        }
    }
    // Resolve aliases separately for every source, as stock deepTools does.
    const auto alternate = chromosome_alias(chrom);
    for (int32_t i = 0; i < file->cl->nKeys; ++i) {
        if (alternate == file->cl->chrom[i]) {
            chrom = alternate;
            size = file->cl->len[i];
            return true;
        }
    }
    return false;
}

std::vector<double> read_segment(bigWigFile_t* file,
                                 const std::string& path,
                                 const std::string& chrom,
                                 const Segment& segment,
                                 bool missing_as_zero,
                                 std::atomic<bool>* omitted_nonfinite) {
    if (segment.end < segment.start) {
        throw std::invalid_argument("A genomic segment has negative length");
    }
    const int64_t signed_length = segment.end - segment.start;
    // The current exact per-base reduction materializes one double per queried
    // base and its bin partition uses signed 32-bit indices. Refuse an
    // impractically large interval before either the allocation or an index
    // narrowing can wrap; large chromosome headers and ordinary local queries
    // remain fully supported.
    if (signed_length > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error(
            "A single matrix segment exceeds the supported per-base query length");
    }
    const auto length = static_cast<std::size_t>(signed_length);
    std::vector<double> result(length, missing_as_zero ? 0.0 : NAN_VALUE);
    if (length == 0) return result;

    uint32_t chrom_size = 0;
    std::string resolved_chrom = chrom;
    if (!chromosome_size(file, resolved_chrom, chrom_size)) return result;
    const int64_t clipped_start = std::max<int64_t>(0, segment.start);
    const int64_t clipped_end = std::min<int64_t>(chrom_size, segment.end);
    if (clipped_end <= clipped_start) return result;

    auto values = dtp::read_bigwig_intervals(file, path, resolved_chrom.c_str(),
        static_cast<uint32_t>(clipped_start), static_cast<uint32_t>(clipped_end));
    if (!values) return result;
    // Gaps retain the initialized missing value. Inspect only stored intervals,
    // avoiding both false non-finite warnings and bwGetValues' per-base copy.
    for (std::size_t i = 0; i < values->l; ++i) {
        poll_native_work(i);
        const double value = static_cast<double>(values->value[i]);
        const int64_t begin = std::max<int64_t>(clipped_start, values->start[i]);
        const int64_t end = std::min<int64_t>(clipped_end, values->end[i]);
        if (end <= begin) continue;
        double stored = value;
        if (!std::isfinite(value)) {
            if (omitted_nonfinite)
                omitted_nonfinite->store(true, std::memory_order_relaxed);
            stored = missing_as_zero ? 0.0 : NAN_VALUE;
        }
        std::fill(result.begin() + (begin - segment.start),
                  result.begin() + (end - segment.start), stored);
    }
    return result;
}

std::vector<double> bin_values(const std::vector<double>& values,
                               int32_t bins,
                               BinStatistic statistic) {
    if (bins <= 0) return {};
    if (values.empty()) return std::vector<double>(bins, NAN_VALUE);
    const int32_t length = static_cast<int32_t>(values.size());
    std::vector<double> output;
    output.reserve(static_cast<std::size_t>(bins));

    // Match coverage_from_array() exactly: NumPy's integer linspace truncates
    // each non-negative cumulative boundary toward zero (floor), and empty
    // intervals caused by length < bins reuse the value at `begin`.
    std::vector<double> scratch;
    for (int32_t i = 0; i < bins; ++i) {
        poll_native_work(static_cast<std::size_t>(i));
        const int32_t begin = static_cast<int32_t>(
            static_cast<int64_t>(i) * length / bins);
        const int32_t boundary = static_cast<int32_t>(
            static_cast<int64_t>(i + 1) * length / bins);
        const int32_t end = std::min(length, std::max(begin + 1, boundary));
        const auto reduction = dtp::finite_reduction(end - begin,
            [&](std::size_t j) { return values[begin + j]; }, statistic, .05, &scratch, true);
        output.push_back(reduction.finite.count ? reduction.value : NAN_VALUE);
    }
    if (static_cast<int32_t>(output.size()) != bins) {
        throw std::runtime_error("Binning did not produce the requested bin count");
    }
    return output;
}

std::vector<double> calculate_source(bigWigFile_t* file,
                                     const std::string& path,
                                     const RowSpec& row,
                                     BinStatistic statistic,
                                     bool missing_as_zero,
                                     std::atomic<bool>* omitted_nonfinite) {
    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(row.unpadded_bins));
    for (const Zone& zone : row.zones) {
        std::vector<double> flattened;
        for (const Segment& segment : zone.segments) {
            auto part = read_segment(file, path, row.chrom, segment, missing_as_zero,
                                     omitted_nonfinite);
            flattened.insert(flattened.end(), part.begin(), part.end());
        }
        auto binned = bin_values(flattened, zone.bins, statistic);
        result.insert(result.end(), binned.begin(), binned.end());
    }
    return result;
}

std::vector<BigWigPtr> open_files(const std::vector<std::string>& paths) {
    std::vector<BigWigPtr> files;
    files.reserve(paths.size());
    for (const auto& path : paths) {
        if (bwIsBigWig(path.c_str(), nullptr) == 0) {
            throw std::runtime_error("Not a valid bigWig file: " + path);
        }
        BigWigPtr file(bwOpen(path.c_str(), nullptr, "r"));
        if (!file) throw std::runtime_error("Failed to open bigWig file: " + path);
        files.push_back(std::move(file));
    }
    return files;
}

std::vector<std::pair<std::string, uint32_t>>
bigwig_chrom_sizes(const std::string& path) {
    auto files = open_files({path});
    bigWigFile_t* file = files.front().get();
    if (!file->cl) {
        throw std::runtime_error("bigWig has no chromosome table: " + path);
    }
    std::vector<std::pair<std::string, uint32_t>> output;
    output.reserve(static_cast<std::size_t>(file->cl->nKeys));
    for (int64_t i = 0; i < file->cl->nKeys; ++i) {
        output.emplace_back(file->cl->chrom[i], file->cl->len[i]);
    }
    return output;
}

std::vector<RowSpec> parse_rows(const py::list& chroms,
                                const py::list& zones,
                                const py::list& source_indices,
                                const py::list& source_scales,
                                const py::list& reverse,
                                const py::list& pad_left,
                                const py::list& pad_right,
                                std::size_t path_count,
                                int32_t& output_count,
                                int32_t& bins_per_output) {
    const py::ssize_t rows = chroms.size();
    if (rows <= 0)
        throw std::invalid_argument("At least one row is required for native compute");
    for (const auto* item : {&zones, &source_indices, &source_scales,
                             &reverse, &pad_left, &pad_right}) {
        if (item->size() != rows) {
            throw std::invalid_argument("All per-row native compute inputs must have equal lengths");
        }
    }

    std::vector<RowSpec> parsed;
    parsed.reserve(static_cast<std::size_t>(rows));
    output_count = -1;
    bins_per_output = -1;
    for (py::ssize_t i = 0; i < rows; ++i) {
        RowSpec row;
        row.chrom = py::cast<std::string>(chroms[i]);
        row.reverse = py::cast<bool>(reverse[i]);
        row.pad_left = py::cast<int32_t>(pad_left[i]);
        row.pad_right = py::cast<int32_t>(pad_right[i]);
        if (row.pad_left < 0 || row.pad_right < 0)
            throw std::invalid_argument("Native compute padding cannot be negative");
        std::int64_t unpadded_bins = 0;
        for (const py::handle zone_handle : py::cast<py::list>(zones[i])) {
            const py::tuple zone_tuple = py::cast<py::tuple>(zone_handle);
            if (zone_tuple.size() != 2)
                throw std::invalid_argument("Each native compute zone needs segments and a bin count");
            Zone zone;
            zone.bins = py::cast<int32_t>(zone_tuple[1]);
            if (zone.bins < 0)
                throw std::invalid_argument("Native compute zone bin counts cannot be negative");
            for (const py::handle segment_handle : py::cast<py::list>(zone_tuple[0])) {
                const py::tuple segment = py::cast<py::tuple>(segment_handle);
                if (segment.size() != 2)
                    throw std::invalid_argument("Each genomic segment needs start and end");
                const auto start = py::cast<int64_t>(segment[0]);
                const auto end = py::cast<int64_t>(segment[1]);
                if (end < start)
                    throw std::invalid_argument("A genomic segment has negative length");
                zone.segments.push_back({start, end});
            }
            unpadded_bins += zone.bins;
            if (unpadded_bins > std::numeric_limits<int32_t>::max())
                throw std::overflow_error("Native compute row has too many bins");
            row.zones.push_back(std::move(zone));
        }
        row.unpadded_bins = static_cast<int32_t>(unpadded_bins);

        const py::list row_indices = py::cast<py::list>(source_indices[i]);
        const py::list row_scales = py::cast<py::list>(source_scales[i]);
        if (row_indices.size() != row_scales.size()) {
            throw std::invalid_argument("Source index and scale output counts differ");
        }
        if (output_count < 0) output_count = static_cast<int32_t>(row_indices.size());
        if (output_count != row_indices.size()) {
            throw std::invalid_argument("Every row must have the same number of outputs");
        }
        if (output_count <= 0)
            throw std::invalid_argument("Every native compute row needs at least one output");
        for (py::ssize_t output = 0; output < row_indices.size(); ++output) {
            OutputSource parsed_output;
            parsed_output.path_indices = py::cast<std::vector<int32_t>>(row_indices[output]);
            parsed_output.scales = py::cast<std::vector<double>>(row_scales[output]);
            if (parsed_output.path_indices.empty() ||
                parsed_output.path_indices.size() != parsed_output.scales.size()) {
                throw std::invalid_argument("Each output needs equally sized, non-empty sources and scales");
            }
            for (int32_t index : parsed_output.path_indices) {
                if (index < 0 || static_cast<std::size_t>(index) >= path_count) {
                    throw std::out_of_range("A native compute source index is invalid");
                }
            }
            for (double scale : parsed_output.scales) {
                if (!std::isfinite(scale))
                    throw std::invalid_argument("Native compute source scales must be finite");
            }
            row.outputs.push_back(std::move(parsed_output));
        }
        const std::int64_t row_bins_wide = unpadded_bins + row.pad_left + row.pad_right;
        if (row_bins_wide <= 0 || row_bins_wide > std::numeric_limits<int32_t>::max())
            throw std::overflow_error("Native compute row has an invalid total bin count");
        const int32_t row_bins = static_cast<int32_t>(row_bins_wide);
        if (bins_per_output < 0) bins_per_output = row_bins;
        if (row_bins != bins_per_output) {
            throw std::invalid_argument("Every row must produce the same number of bins");
        }
        parsed.push_back(std::move(row));
    }
    return parsed;
}

template <typename T>
void calculate_row(const std::vector<BigWigPtr>& files,
                   const std::vector<std::string>& paths,
                   const RowSpec& row,
                   BinStatistic statistic,
                   bool missing_as_zero,
                   int32_t bins_per_output,
                   std::vector<std::atomic<bool>>& omitted_nonfinite,
                   T* destination) {
    for (std::size_t output_index = 0; output_index < row.outputs.size(); ++output_index) {
        poll_native_work(output_index);
        const auto& sources = row.outputs[output_index];
        std::vector<double> combined;
        for (std::size_t source = 0; source < sources.path_indices.size(); ++source) {
            auto current = calculate_source(
                files[static_cast<std::size_t>(sources.path_indices[source])].get(),
                paths[static_cast<std::size_t>(sources.path_indices[source])],
                row, statistic, missing_as_zero,
                &omitted_nonfinite[static_cast<std::size_t>(
                    sources.path_indices[source])]);
            const double scale = sources.scales[source];
            for (double& value : current) {
                if (std::isnan(value)) continue;
                value *= scale;
                if (!std::isfinite(value))
                    throw std::overflow_error("Native compute scaling exceeds float64 range");
            }
            if (combined.empty()) combined = std::move(current);
            else {
                for (std::size_t j = 0; j < combined.size(); ++j) {
                    combined[j] += current[j];
                    if (std::isinf(combined[j]))
                        throw std::overflow_error("Native compute combination exceeds float64 range");
                }
            }
        }
        std::vector<double> padded(static_cast<std::size_t>(bins_per_output), NAN_VALUE);
        std::copy(combined.begin(), combined.end(), padded.begin() + row.pad_left);
        if (row.reverse) std::reverse(padded.begin(), padded.end());
        // Missing input remains NaN; arithmetic overflow must abort the output
        // transaction instead of silently manufacturing a missing observation.
        T* out = destination +
                 output_index * static_cast<std::size_t>(bins_per_output);
        for (std::size_t j = 0; j < padded.size(); ++j) {
            poll_native_work(j);
            if (std::isnan(padded[j])) {
                out[j] = std::numeric_limits<T>::quiet_NaN();
            } else {
                if (!std::isfinite(padded[j]) ||
                    std::abs(padded[j]) > std::numeric_limits<T>::max())
                    throw std::overflow_error("Native compute result exceeds output numeric range");
                out[j] = static_cast<T>(padded[j]);
            }
        }
    }
}

class NativeBigWigReader {
public:
    explicit NativeBigWigReader(const std::vector<std::string>& paths)
        : paths_(paths), files_(open_files(paths)) {
        if (paths.empty()) {
            throw std::invalid_argument("At least one bigWig path is required");
        }
    }

    py::array_t<double> bin_row(const std::string& chrom,
                                const py::list& zones,
                                const py::list& source_indices,
                                const py::list& source_scales,
                                bool reverse,
                                int32_t pad_left,
                                int32_t pad_right,
                                const std::string& statistic_name,
                                bool missing_as_zero) const {
        py::list chroms;
        chroms.append(chrom);
        py::list all_zones;
        all_zones.append(zones);
        py::list all_indices;
        all_indices.append(source_indices);
        py::list all_scales;
        all_scales.append(source_scales);
        py::list reverses;
        reverses.append(reverse);
        py::list left;
        left.append(pad_left);
        py::list right;
        right.append(pad_right);
        int32_t output_count = 0;
        int32_t bins_per_output = 0;
        auto rows = parse_rows(chroms, all_zones, all_indices, all_scales,
                               reverses, left, right, paths_.size(),
                               output_count, bins_per_output);
        py::array_t<double> output(output_count * bins_per_output);
        double* output_data = output.mutable_data();
        std::vector<std::atomic<bool>> omitted_nonfinite(paths_.size());
        for (auto& flag : omitted_nonfinite)
            flag.store(false, std::memory_order_relaxed);
        {
            py::gil_scoped_release release;
            calculate_row(files_, paths_, rows.front(), parse_statistic(statistic_name),
                          missing_as_zero, bins_per_output,
                          omitted_nonfinite,
                          output_data);
        }
        for (std::size_t input = 0; input < paths_.size(); ++input)
            if (omitted_nonfinite[input].load(std::memory_order_relaxed))
                std::fprintf(stderr,
                    "Warning: omitted non-finite values from input bigWig '%s'.\n",
                    paths_[input].c_str());
        return output;
    }

private:
    std::vector<std::string> paths_;
    std::vector<BigWigPtr> files_;
};

std::unique_ptr<NativeMatrix> bin_bigwig_batch_owned(
        const std::vector<std::string>& paths,
        const py::list& chroms,
        const py::list& zones,
        const py::list& source_indices,
        const py::list& source_scales,
        const py::list& reverse,
        const py::list& pad_left,
        const py::list& pad_right,
        const std::string& statistic_name,
        bool missing_as_zero,
        int threads) {
    if (paths.empty()) throw std::invalid_argument("At least one bigWig path is required");
    int32_t output_count = 0;
    int32_t bins_per_output = 0;
    auto rows = parse_rows(chroms, zones, source_indices, source_scales,
                           reverse, pad_left, pad_right, paths.size(),
                           output_count, bins_per_output);
    const BinStatistic statistic = parse_statistic(statistic_name);
    const auto output_columns = static_cast<std::uint64_t>(output_count) *
                                static_cast<std::uint64_t>(bins_per_output);
    if (output_columns > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Native compute output dimensions overflow");
    auto output = std::make_unique<NativeMatrix>(
        rows.size(), static_cast<std::size_t>(output_columns));
    auto* output_data = output->data();
    if (rows.empty()) return output;
    std::vector<std::atomic<bool>> omitted_nonfinite(paths.size());
    for (auto& flag : omitted_nonfinite)
        flag.store(false, std::memory_order_relaxed);

    threads = std::max(1, std::min<int>(threads,
        static_cast<int>(rows.size())));
    const std::size_t workers = dtp::thread_count(threads, rows.size());
    std::atomic<std::size_t> next_row{0};

    {
        py::gil_scoped_release release;
        dtp::parallel_rows(workers, static_cast<int>(workers),
            [&](std::size_t) {
                auto files = open_files(paths);
                while (!dtp::interruption_requested()) {
                    const std::size_t row_index = next_row.fetch_add(
                        1, std::memory_order_relaxed);
                    if (row_index >= rows.size()) break;
                    const RowSpec& row = rows[row_index];
                    const std::size_t offset = row_index *
                        static_cast<std::size_t>(output_columns);
                    calculate_row(files, paths, row, statistic, missing_as_zero,
                                  bins_per_output, omitted_nonfinite,
                                  output_data + offset);
                }
            });
    }
    for (std::size_t input = 0; input < paths.size(); ++input)
        if (omitted_nonfinite[input].load(std::memory_order_relaxed))
            std::fprintf(stderr,
                "Warning: omitted non-finite values from input bigWig '%s'.\n",
                paths[input].c_str());
    return output;
}

py::array bin_bigwig_batch(
        const std::vector<std::string>& paths,
        const py::list& chroms,
        const py::list& zones,
        const py::list& source_indices,
        const py::list& source_scales,
        const py::list& reverse,
        const py::list& pad_left,
        const py::list& pad_right,
        const std::string& statistic_name,
        bool missing_as_zero,
        int threads) {
    auto owner = bin_bigwig_batch_owned(
        paths, chroms, zones, source_indices, source_scales, reverse,
        pad_left, pad_right, statistic_name, missing_as_zero, threads);
    NativeMatrix* raw = owner.release();
    py::capsule capsule(raw, [](void* pointer) {
        delete static_cast<NativeMatrix*>(pointer);
    });
    return py::array(
        py::dtype::of<float>(),
        {static_cast<py::ssize_t>(raw->rows()),
         static_cast<py::ssize_t>(raw->columns())},
        {static_cast<py::ssize_t>(raw->columns() * sizeof(float)),
         static_cast<py::ssize_t>(sizeof(float))},
        raw->data(), capsule);
}

}  // namespace

PYBIND11_MODULE(_compute_matrix_native, module) {
    module.doc() = "Native libBigWig-backed matrix binning for deepToolsR";
    module.attr("supports_multicore") = true;
    py::class_<NativeMatrix>(module, "NativeMatrix", py::buffer_protocol())
        .def(py::init<std::size_t, std::size_t>())
        .def_buffer(&NativeMatrix::buffer_info)
        .def_property_readonly("shape", [](const NativeMatrix& matrix) {
            return py::make_tuple(matrix.rows(), matrix.columns());
        });
    py::class_<NativeBigWigReader>(module, "NativeBigWigReader")
        .def(py::init<const std::vector<std::string>&>())
        .def("bin_row", &NativeBigWigReader::bin_row,
             py::arg("chrom"), py::arg("zones"),
             py::arg("source_indices"), py::arg("source_scales"),
             py::arg("reverse"), py::arg("pad_left"), py::arg("pad_right"),
             py::arg("statistic") = "mean",
             py::arg("missing_as_zero") = false);
    module.def("bin_bigwig_batch", &bin_bigwig_batch,
               py::arg("paths"), py::arg("chroms"), py::arg("zones"),
               py::arg("source_indices"), py::arg("source_scales"),
               py::arg("reverse"), py::arg("pad_left"), py::arg("pad_right"),
               py::arg("statistic") = "mean", py::arg("missing_as_zero") = false,
               py::arg("threads") = 1);
    module.def("bin_bigwig_batch_owned", &bin_bigwig_batch_owned,
               py::arg("paths"), py::arg("chroms"), py::arg("zones"),
               py::arg("source_indices"), py::arg("source_scales"),
               py::arg("reverse"), py::arg("pad_left"), py::arg("pad_right"),
               py::arg("statistic") = "mean", py::arg("missing_as_zero") = false,
               py::arg("threads") = 1);
    module.def("bigwig_chrom_sizes", &bigwig_chrom_sizes, py::arg("path"),
               "Return the chromosome names and lengths stored in a bigWig.");
    module.def("chromosome_alias", &chromosome_alias, py::arg("chrom"),
               "Return the alternate chromosome name used after an exact lookup fails.");
}
