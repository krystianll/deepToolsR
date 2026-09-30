#pragma once

#include <pybind11/numpy.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dtp {

namespace py = pybind11;

// Row-major float32 matrix owned in C++ and exposed to NumPy through the buffer
// protocol (a non-owning view). Missing values are stored as NaN. Defined in a
// header so it can be shared between the binning code and other translation
// units (e.g. the heatmap rasteriser and the future per-sample layout).
class NativeMatrix {
public:
    NativeMatrix(std::size_t rows, std::size_t columns)
        : rows_(rows), columns_(columns),
          values_(checked_elements(rows, columns),
                  std::numeric_limits<float>::quiet_NaN()) {}

    float* data() noexcept { return values_.data(); }
    const float* data() const noexcept { return values_.data(); }
    std::size_t rows() const noexcept { return rows_; }
    std::size_t columns() const noexcept { return columns_; }

    py::buffer_info buffer_info() {
        return py::buffer_info(
            values_.data(), sizeof(float),
            py::format_descriptor<float>::format(), 2,
            {static_cast<py::ssize_t>(rows_),
             static_cast<py::ssize_t>(columns_)},
            {static_cast<py::ssize_t>(columns_ * sizeof(float)),
             static_cast<py::ssize_t>(sizeof(float))});
    }

private:
    static std::size_t checked_elements(std::size_t rows, std::size_t columns) {
        if (rows != 0 && columns > std::numeric_limits<std::size_t>::max() / rows)
            throw std::overflow_error("native matrix dimensions overflow");
        const std::size_t elements = rows * columns;
        if (elements > std::vector<float>().max_size())
            throw std::length_error("native matrix exceeds maximum allocation size");
        return elements;
    }

    std::size_t rows_;
    std::size_t columns_;
    std::vector<float> values_;
};

}  // namespace dtp
