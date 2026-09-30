#pragma once
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#define DTP_PROJECTION_ARGS \
    pybind11::kw_only(), pybind11::arg("rows") = pybind11::none(), \
    pybind11::arg("row_range") = pybind11::none(), \
    pybind11::arg("cols") = pybind11::none(), \
    pybind11::arg("col_range") = pybind11::none()

#define DTP_PROJECTION_RANGE_ARGS \
    pybind11::kw_only(), pybind11::arg("row_range") = pybind11::none(), \
    pybind11::arg("col_range") = pybind11::none()

namespace dtp {
// Borrow buffer exports for the entire call. memcpy supports unaligned and
// signed-stride values; index vectors are validated only at construction.
class MatrixView {
    pybind11::buffer_info values_;
    pybind11::buffer_info row_info_, col_info_;
    pybind11::object row_owner_, col_owner_;
    const std::int64_t *row_list_ = nullptr, *col_list_ = nullptr;
    std::size_t row_offset_ = 0, col_offset_ = 0;
    bool has_rows_ = false, has_cols_ = false;
    bool single_ = false;

    static void check_offsets(const pybind11::buffer_info &info) {
        std::size_t span = 0;
        const auto limit = static_cast<std::size_t>(std::numeric_limits<pybind11::ssize_t>::max());
        for (int i = 0; i < info.ndim; ++i) {
            if (info.shape[i] < 0) throw std::invalid_argument("negative buffer dimension");
            const auto stride = info.strides[i];
            const std::size_t step = stride < 0 ? static_cast<std::size_t>(-(stride + 1)) + 1
                                               : static_cast<std::size_t>(stride);
            const auto n = info.shape[i] > 0 ? static_cast<std::size_t>(info.shape[i] - 1) : 0;
            if (n && step > (limit - span) / n)
                throw std::overflow_error("buffer stride extent exceeds addressable range");
            span += n * step;
        }
    }

    static std::size_t configure_axis(const pybind11::object &indices,
                                      const pybind11::object &range,
                                      std::size_t extent, const char *name,
                                      pybind11::object &owner,
                                      pybind11::buffer_info &info,
                                      const std::int64_t *&list,
                                      std::size_t &offset) {
        if (!indices.is_none() && !range.is_none())
            throw std::invalid_argument(std::string(name) + " list and range conflict");
        if (!indices.is_none()) {
            if (!pybind11::isinstance<pybind11::buffer>(indices))
                throw pybind11::type_error(std::string(name) + " must be an int64 buffer");
            owner = indices;
            info = pybind11::cast<pybind11::buffer>(owner).request();
            if (info.ndim != 1 || info.itemsize != sizeof(std::int64_t) ||
                (info.format != "l" && info.format != "q") ||
                (info.shape[0] > 1 &&
                 info.strides[0] != static_cast<pybind11::ssize_t>(sizeof(std::int64_t))))
                throw std::invalid_argument(std::string(name) + " must be a 1-D C-contiguous int64 buffer");
            list = static_cast<const std::int64_t *>(info.ptr);
            for (pybind11::ssize_t i = 0; i < info.shape[0]; ++i)
                if (list[i] < 0 || static_cast<std::uint64_t>(list[i]) >= extent)
                    throw std::invalid_argument(std::string(name) + " index out of bounds");
            return static_cast<std::size_t>(info.shape[0]);
        }
        if (range.is_none()) return extent;
        if (!pybind11::isinstance<pybind11::tuple>(range) || pybind11::len(range) != 2)
            throw pybind11::type_error(std::string(name) + " range must be a (start, stop) tuple");
        const auto pair = pybind11::reinterpret_borrow<pybind11::tuple>(range);
        const auto start = pybind11::cast<pybind11::ssize_t>(pair[0]);
        const auto stop = pybind11::cast<pybind11::ssize_t>(pair[1]);
        if (start < 0 || stop < start || static_cast<std::uint64_t>(stop) > extent)
            throw std::invalid_argument(std::string(name) + " range out of bounds");
        offset = static_cast<std::size_t>(start);
        return static_cast<std::size_t>(stop - start);
    }

public:
    std::size_t rows, cols;
    MatrixView(const pybind11::buffer &values,
               const pybind11::object &row_indices = pybind11::none(),
               const pybind11::object &row_range = pybind11::none(),
               const pybind11::object &col_indices = pybind11::none(),
               const pybind11::object &col_range = pybind11::none())
        : values_(values.request()) {
        if (values_.ndim != 2) throw std::invalid_argument("matrix must be two-dimensional");
        std::string format = values_.format;
        if (!format.empty() && (format[0] == '=' || format[0] == '@')) format.erase(0, 1);
        single_ = values_.itemsize == sizeof(float) && format == pybind11::format_descriptor<float>::format();
        if (!single_ && !(values_.itemsize == sizeof(double) && format == pybind11::format_descriptor<double>::format()))
            throw std::invalid_argument("matrix must contain native-endian float32 or float64 values");
        check_offsets(values_);
        const auto source_rows = static_cast<std::size_t>(values_.shape[0]);
        const auto source_cols = static_cast<std::size_t>(values_.shape[1]);
        if (source_cols && source_rows > static_cast<std::size_t>(std::numeric_limits<pybind11::ssize_t>::max()) / source_cols)
            throw std::overflow_error("matrix size exceeds addressable range");
        rows = configure_axis(row_indices, row_range, source_rows, "rows",
                              row_owner_, row_info_, row_list_, row_offset_);
        cols = configure_axis(col_indices, col_range, source_cols, "cols",
                              col_owner_, col_info_, col_list_, col_offset_);
        has_rows_ = !row_indices.is_none();
        has_cols_ = !col_indices.is_none();
    }
    bool single_precision() const { return single_; }
    std::size_t source_row(std::size_t row) const {
        return has_rows_ ? static_cast<std::size_t>(row_list_[row]) : row_offset_ + row;
    }
    std::size_t source_col(std::size_t col) const {
        return has_cols_ ? static_cast<std::size_t>(col_list_[col]) : col_offset_ + col;
    }
    double at(std::size_t row, std::size_t col) const {
        const auto r = static_cast<pybind11::ssize_t>(source_row(row));
        const auto c = static_cast<pybind11::ssize_t>(source_col(col));
        const char *address = static_cast<const char *>(values_.ptr) + r * values_.strides[0] + c * values_.strides[1];
        if (single_) { float x; std::memcpy(&x, address, sizeof(x)); return x; }
        double x; std::memcpy(&x, address, sizeof(x)); return x;
    }
    bool present(std::size_t r, std::size_t c, double &x) const { x = at(r, c); return std::isfinite(x); }
};
inline void validate_axis(int axis) {
    if (axis != 0 && axis != 1) throw std::invalid_argument("axis must be 0 or 1");
}
} // namespace dtp
