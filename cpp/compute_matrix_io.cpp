#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "matrix_read.hpp"
#include "matrix_view.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <fstream>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct RegionData {
    std::string chrom;
    std::vector<std::pair<std::int64_t, std::int64_t>> blocks;
    std::string name;
    std::string score;
    std::string strand;
};

struct OutputBlock {
    std::string data;
    std::size_t index = std::numeric_limits<std::size_t>::max();
    bool ready = false;
};

using dtp::append_double;
using dtp::append_integer;
using dtp::gzip_member;
using dtp::parallel_rows;
using dtp::thread_count;

void parse_row(const char *begin, const char *end, std::size_t row,
               std::size_t bins, float *matrix, RegionData &region) {
    std::vector<std::pair<const char *, const char *>> fields;
    fields.reserve(bins + 6);
    dtp::split_fields(begin, end, fields);
    if (fields.size() != bins + 6)
        throw std::runtime_error("expected " + std::to_string(bins + 6) +
                                 " fields but found " +
                                 std::to_string(fields.size()));

    region.chrom.assign(fields[0].first, fields[0].second);
    region.name.assign(fields[3].first, fields[3].second);
    region.score.assign(fields[4].first, fields[4].second);
    region.strand.assign(fields[5].first, fields[5].second);
    dtp::validate_matrix_fields(fields, bins, &region.blocks, matrix + row * bins);
}

void validate_rows(const std::vector<std::string> &lines, std::size_t bins,
                   std::size_t first_row, int num_threads) {
    if (!bins) throw std::runtime_error("matrix must contain at least one bin");
    py::gil_scoped_release release;
    parallel_rows(lines.size(), num_threads, [&](std::size_t row) {
        const auto &line = lines[row];
        const char *end = line.data() + line.size();
        if (end > line.data() && end[-1] == '\n') --end;
        dtp::MatrixFields fields;
        dtp::split_fields(line.data(), end, fields);
        try {
            dtp::validate_matrix_fields(fields, bins);
        } catch (const std::exception &error) {
            throw std::runtime_error("row " + std::to_string(first_row + row + 1) +
                                     ": " + error.what());
        }
    });
}

py::tuple read_matrix(const std::string &path, std::size_t regions,
                      std::size_t bins,
                      const std::vector<std::size_t> &group_boundaries,
                      int num_threads, std::size_t buffer_size) {
    if (buffer_size == 0)
        throw std::runtime_error("buffer size must be positive");
    if (bins == 0)
        throw std::runtime_error("matrix must contain at least one bin");
    if (regions != 0 && bins > std::numeric_limits<std::size_t>::max() / regions)
        throw std::runtime_error("matrix dimensions overflow addressable memory");
    if (group_boundaries.size() < 2 || group_boundaries.front() != 0 ||
        group_boundaries.back() != regions ||
        !std::is_sorted(group_boundaries.begin(), group_boundaries.end()))
        throw std::runtime_error("group boundaries do not match region count");

    py::array_t<float> matrix({regions, bins});
    float *matrix_data = matrix.mutable_data();
    std::vector<RegionData> metadata(regions);
    std::string header;

    {
        py::gil_scoped_release release;
        std::size_t global_row = 0;
        dtp::read_matrix_chunks(
            path, buffer_size,
            [&](const char *begin, const char *end, const dtp::MatrixSchema &schema) {
                if (schema.rows != regions || schema.bins != bins)
                    throw std::runtime_error("matrix header does not match requested dimensions");
                header.assign(begin, end);
            },
            [&](const char *base,
                const std::vector<std::pair<std::size_t, std::size_t>> &lines,
                const dtp::MatrixSchema &) {
                if (global_row + lines.size() > regions)
                    throw std::runtime_error(
                        "matrix contains more regions than its header");
                parallel_rows(lines.size(), num_threads, [&](std::size_t local) {
                    try {
                        const auto bounds = lines[local];
                        parse_row(base + bounds.first, base + bounds.second,
                                  global_row + local, bins, matrix_data,
                                  metadata[global_row + local]);
                    } catch (const std::exception &error) {
                        throw std::runtime_error("row " +
                            std::to_string(global_row + local + 1) + ": " +
                            error.what());
                    }
                });
                global_row += lines.size();
            });
        if (global_row != regions)
            throw std::runtime_error("expected " + std::to_string(regions) +
                " regions but read " + std::to_string(global_row));
    }

    py::list python_regions;
    std::size_t group = 0;
    for (std::size_t row = 0; row < regions; ++row) {
        while (group + 1 < group_boundaries.size() &&
               row >= group_boundaries[group + 1]) ++group;
        py::list blocks;
        for (const auto &block : metadata[row].blocks)
            blocks.append(py::make_tuple(block.first, block.second));
        py::list region;
        region.append(metadata[row].chrom);
        region.append(std::move(blocks));
        region.append(metadata[row].name);
        region.append(group_boundaries[group + 1]);
        region.append(metadata[row].strand);
        region.append(metadata[row].score);
        python_regions.append(std::move(region));
    }
    return py::make_tuple(header, std::move(python_regions), std::move(matrix));
}

void write_matrix(const std::string &path, const std::string &header,
                  const py::list &regions, const py::buffer &matrix,
                  int num_threads, int compression_level,
                  std::size_t block_rows,
                  py::object row_order_object, py::object column_order_object,
                  bool compressed, py::object row_range, py::object col_range) {
    // Resolve the Python/native storage abstraction once. The hot formatting
    // loop below operates on raw pointers and does not invoke Python or virtual
    // methods for individual matrix cells.
    const auto shape = matrix.request();
    const dtp::MatrixView view(matrix, row_order_object, row_range,
                               column_order_object, col_range);
    const auto rows = static_cast<std::size_t>(shape.shape[0]);
    const auto bins = static_cast<std::size_t>(shape.shape[1]);
    if (!view.single_precision())
        throw std::invalid_argument("matrix output requires float32 values");
    if (!row_order_object.is_none()) {
        std::vector<std::uint8_t> seen(rows, 0);
        for (std::size_t i = 0; i < view.rows; ++i) {
            const auto index = view.source_row(i);
            if (seen[index]++) throw std::invalid_argument("row order requires distinct indices");
        }
    }
    if (!column_order_object.is_none()) {
        std::vector<std::uint8_t> seen(bins, 0);
        for (std::size_t i = 0; i < view.cols; ++i) {
            const auto index = view.source_col(i);
            if (seen[index]++) throw std::invalid_argument("column order requires distinct indices");
        }
    }
    if (static_cast<std::size_t>(py::len(regions)) != rows)
        throw std::runtime_error("region metadata and matrix row counts differ");
    if (compression_level < 0 || compression_level > 9)
        throw std::runtime_error("compression level must be between 0 and 9");
    if (block_rows == 0) {
        // Bound the uncompressed text held by each producer. A fixed 256-row
        // block becomes very large for matrices with thousands of bins.
        constexpr std::size_t target_block_bytes = 1 * 1024 * 1024;
        const std::size_t estimated_row_bytes = bins * 12 + 96;
        block_rows = std::max<std::size_t>(
            1, std::min<std::size_t>(256,
                target_block_bytes / std::max<std::size_t>(1, estimated_row_bytes)));
    }

    // Write-time reordering/selection: emit rows and columns straight from the
    // resident buffer in these orders, so reorder/sort/subset never build a
    // physically permuted copy of the matrix. None means identity; [] is empty.
    const std::size_t out_rows = view.rows;
    const std::size_t out_bins = view.cols;
    if (!out_bins) throw std::invalid_argument("matrix output must contain at least one column");

    std::vector<RegionData> metadata(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        const py::sequence region = regions[row].cast<py::sequence>();
        if (py::len(region) < 6) throw std::runtime_error("invalid region metadata");
        metadata[row].chrom = py::str(region[0]);
        metadata[row].name = py::str(region[2]);
        metadata[row].strand = py::str(region[4]);
        metadata[row].score = py::str(region[5]);
        const py::sequence blocks = region[1].cast<py::sequence>();
        for (const auto item : blocks) {
            const py::sequence block = item.cast<py::sequence>();
            if (py::len(block) != 2)
                throw std::runtime_error("genomic blocks must contain start and end");
            const auto start = py::cast<std::int64_t>(block[0]);
            const auto end = py::cast<std::int64_t>(block[1]);
            if (start < 0 || end < start)
                throw std::runtime_error("invalid genomic block boundaries");
            metadata[row].blocks.emplace_back(start, end);
        }
    }
    py::gil_scoped_release release;
    const std::size_t block_count = (out_rows + block_rows - 1) / block_rows;
    const std::size_t workers = thread_count(
        std::max(1, num_threads - 1), block_count);

    auto produce_block = [&](std::size_t block_index) {
        const std::size_t first = block_index * block_rows;
        const auto last = std::min(out_rows, first + block_rows);
        std::string block;
        block.reserve((last - first) * (out_bins * 12 + 96));
        for (std::size_t output_row = first; output_row < last; ++output_row) {
                if (((output_row - first) & 7u) == 0) {
                    if (dtp::running_in_parallel_worker()) {
                        if (dtp::interruption_requested())
                            throw std::runtime_error(
                                "matrix compression cancelled");
                    } else {
                        dtp::check_python_signals();
                    }
                }
                const std::size_t row = view.source_row(output_row);
                const auto &region = metadata[row];
                block += region.chrom;
                block += '\t';
                for (std::size_t index = 0; index < region.blocks.size(); ++index) {
                    if (index) block += ',';
                    append_integer(block, region.blocks[index].first);
                }
                block += '\t';
                for (std::size_t index = 0; index < region.blocks.size(); ++index) {
                    if (index) block += ',';
                    append_integer(block, region.blocks[index].second);
                }
                block += '\t'; block += region.name;
                block += '\t'; block += region.score;
                block += '\t'; block += region.strand;
                for (std::size_t out_col = 0; out_col < out_bins; ++out_col) {
                    block += '\t';
                    append_double(block, view.at(output_row, out_col));
                }
                block += '\n';
        }
        return compressed ? gzip_member(block, compression_level) : block;
    };
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open output matrix: " + path);
    const std::string header_block = compressed
        ? gzip_member("@" + header + "\n", compression_level)
        : "@" + header + "\n";
    output.write(header_block.data(), header_block.size());
    if (!output) throw std::runtime_error("failed to write matrix header");

    if (num_threads <= 1 || block_count <= 1) {
        for (std::size_t block = 0; block < block_count; ++block) {
            const auto compressed = produce_block(block);
            output.write(compressed.data(), compressed.size());
        }
    } else {
        // Bounded, ordered producer/consumer ring adapted from Metaplotter.
        const std::size_t ring_size = std::min<std::size_t>(
            std::max<std::size_t>(workers + 2, 4), 8);
        std::vector<OutputBlock> ring(ring_size);
        std::mutex mutex;
        std::condition_variable writer_ready;
        std::condition_variable slot_ready;
        std::atomic<std::size_t> next_block{0};
        std::atomic<bool> cancelled{false};
        std::exception_ptr failure;
        std::size_t next_write = 0;

        auto set_error = [&](std::exception_ptr error) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!failure) failure = std::move(error);
                cancelled.store(true, std::memory_order_relaxed);
            }
            writer_ready.notify_all();
            slot_ready.notify_all();
        };

        std::vector<std::thread> producers;
        producers.reserve(workers);
        auto producer = [&]() {
            dtp::parallel_cancel_flag = &cancelled;
            try {
                    while (!cancelled.load(std::memory_order_relaxed)) {
                        const auto block = next_block.fetch_add(1);
                        if (block >= block_count) break;
                        auto compressed = produce_block(block);
                        const auto slot = block % ring_size;
                        std::unique_lock<std::mutex> lock(mutex);
                        slot_ready.wait(lock, [&]() {
                            return cancelled.load(std::memory_order_relaxed) ||
                                (block < next_write + ring_size &&
                                 !ring[slot].ready);
                        });
                        if (cancelled.load(std::memory_order_relaxed)) break;
                        ring[slot].data = std::move(compressed);
                        ring[slot].index = block;
                        ring[slot].ready = true;
                        lock.unlock();
                        writer_ready.notify_one();
                    }
            } catch (...) {
                set_error(std::current_exception());
            }
            dtp::parallel_cancel_flag = nullptr;
        };
        try {
            for (std::size_t worker = 0; worker < workers; ++worker)
                producers.emplace_back(producer);
        } catch (...) {
            set_error(std::current_exception());
        }

        // Keep ordered I/O on the coordinating thread. Timed waits let it poll
        // Python signals while producers compress blocks in parallel.
        while (next_write < block_count &&
               !cancelled.load(std::memory_order_relaxed)) {
            const auto slot = next_write % ring_size;
            std::unique_lock<std::mutex> lock(mutex);
            while (!failure &&
                   !(ring[slot].ready && ring[slot].index == next_write)) {
                writer_ready.wait_for(lock, std::chrono::milliseconds(50));
                if (failure) break;
                lock.unlock();
                try {
                    dtp::check_python_signals();
                } catch (...) {
                    set_error(std::current_exception());
                }
                lock.lock();
            }
            if (failure) break;
            output.write(ring[slot].data.data(), ring[slot].data.size());
            if (!output) {
                lock.unlock();
                set_error(std::make_exception_ptr(std::runtime_error(
                    "failed while writing compressed block")));
                break;
            }
            ring[slot].data.clear();
            ring[slot].ready = false;
            ring[slot].index = std::numeric_limits<std::size_t>::max();
            ++next_write;
            lock.unlock();
            slot_ready.notify_all();
        }
        cancelled.store(true, std::memory_order_relaxed);
        slot_ready.notify_all();
        for (auto &producer : producers) producer.join();
        if (failure) std::rethrow_exception(failure);
    }
    dtp::close_matrix_output(output, path);
}

}  // namespace

PYBIND11_MODULE(_compute_matrix_io, module) {
    module.def("validate_rows", &validate_rows, py::arg("lines"), py::arg("bins"),
               py::arg("first_row") = 0, py::arg("num_threads") = 1);
    module.doc() = "Fast computeMatrix reader and writer";
    module.attr("supports_multicore") = true;
    module.def("read_matrix", &read_matrix, py::arg("path"),
               py::arg("regions"), py::arg("bins"),
               py::arg("group_boundaries"), py::arg("num_threads") = 1,
               py::arg("buffer_size") = 8 * 1024 * 1024);
    module.def("write_matrix", &write_matrix, py::arg("path"),
               py::arg("header"), py::arg("regions"), py::arg("matrix"),
               py::arg("num_threads") = 1,
               py::arg("compression_level") = 6,
               py::arg("block_rows") = 256,
               py::arg("row_order") = py::none(),
               py::arg("column_order") = py::none(),
               py::arg("compressed") = true, DTP_PROJECTION_RANGE_ARGS);
}
