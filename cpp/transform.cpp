#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "parallel.hpp"
#include "numeric.hpp"
#include <cstdint>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using dtp::parallel_rows;

constexpr float NAN_F = std::numeric_limits<float>::quiet_NaN();

float checked_scalar(double value, const char *name) {
    if (!std::isfinite(value) ||
        std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))
        throw std::runtime_error(std::string(name) +
                                 " must be finite and within float32 range");
    return static_cast<float>(value);
}

float checked_result(double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        throw std::overflow_error("transform result exceeds float32 range");
    return static_cast<float>(value);
}

struct MatrixRef {
    float *data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
};

void validate_sample_bounds(const std::vector<std::size_t> &bounds,
                            std::size_t columns) {
    if (bounds.size() < 2 || bounds.front() != 0 || bounds.back() != columns)
        throw std::runtime_error("sample boundaries must span all matrix columns");
    for (std::size_t index = 1; index < bounds.size(); ++index)
        if (bounds[index] <= bounds[index - 1])
            throw std::runtime_error("sample boundaries must be strictly increasing");
}

// Borrow a mutable, C-contiguous float32 [rows, cols] buffer for in-place work.
MatrixRef as_matrix(const py::array_t<float, 0> &matrix, bool writable = true) {
    const auto info = matrix.request(writable);
    if (reinterpret_cast<std::uintptr_t>(info.ptr) % alignof(float))
        throw std::invalid_argument("matrix must be aligned");
    if (info.ndim != 2)
        throw std::runtime_error("matrix must be two-dimensional");
    if (info.strides[1] != static_cast<py::ssize_t>(sizeof(float)) ||
        info.strides[0] !=
            static_cast<py::ssize_t>(info.shape[1] * sizeof(float)))
        throw std::runtime_error("matrix must be C-contiguous float32");
    MatrixRef ref;
    ref.data = static_cast<float *>(info.ptr);
    ref.rows = static_cast<std::size_t>(info.shape[0]);
    ref.cols = static_cast<std::size_t>(info.shape[1]);
    return ref;
}

void reject_partial_overlap(const MatrixRef &a, const MatrixRef &b) {
    const auto first = reinterpret_cast<std::uintptr_t>(a.data);
    const auto second = reinterpret_cast<std::uintptr_t>(b.data);
    const auto bytes_a = a.rows * a.cols * sizeof(float);
    const auto bytes_b = b.rows * b.cols * sizeof(float);
    if (first != second && first < second + bytes_b && second < first + bytes_a)
        throw std::invalid_argument("partially overlapping matrix buffers are not supported");
}

// ---- elementwise, whole-matrix, in place ----------------------------------

void add_scalar(py::array_t<float, 0> matrix, double value, int num_threads) {
    const MatrixRef m = as_matrix(matrix);
    checked_scalar(value, "value");
    const double v = value;
    py::gil_scoped_release release;
    parallel_rows(m.rows, num_threads, [&](std::size_t row) {
        float *r = m.data + row * m.cols;
        for (std::size_t c = 0; c < m.cols; ++c) r[c] = std::isfinite(r[c]) ? checked_result(static_cast<double>(r[c]) + v) : NAN_F;
    });
}

void scale_scalar(py::array_t<float, 0> matrix, double factor, int num_threads) {
    const MatrixRef m = as_matrix(matrix);
    checked_scalar(factor, "factor");
    const double f = factor;
    py::gil_scoped_release release;
    parallel_rows(m.rows, num_threads, [&](std::size_t row) {
        float *r = m.data + row * m.cols;
        for (std::size_t c = 0; c < m.cols; ++c) r[c] = std::isfinite(r[c]) ? checked_result(static_cast<double>(r[c]) * f) : NAN_F;
    });
}

void log2_inplace(py::array_t<float, 0> matrix, int num_threads) {
    const MatrixRef m = as_matrix(matrix);
    py::gil_scoped_release release;
    parallel_rows(m.rows, num_threads, [&](std::size_t row) {
        float *r = m.data + row * m.cols;
        for (std::size_t c = 0; c < m.cols; ++c) {
            // log2(0) -> -inf, log2(<0) -> NaN; both are missing values.
            const float v = std::log2(r[c]);
            r[c] = std::isfinite(v) ? v : NAN_F;
        }
    });
}

void nonfinite_to_zero(py::array_t<float, 0> matrix, int num_threads) {
    const MatrixRef m = as_matrix(matrix);
    py::gil_scoped_release release;
    parallel_rows(m.rows, num_threads, [&](std::size_t row) {
        float *r = m.data + row * m.cols;
        for (std::size_t c = 0; c < m.cols; ++c)
            if (!std::isfinite(r[c])) r[c] = 0.0f;
    });
}

void nonfinite_to_nan(py::array_t<float, 0> matrix, int num_threads) {
    const auto m = as_matrix(matrix);
    py::gil_scoped_release release;
    parallel_rows(m.rows, num_threads, [&](std::size_t r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            auto &v = m.data[r * m.cols + c];
            if (!std::isfinite(v)) v = NAN_F;
        }
    });
}

// ---- per-sample row scaling (scale row-sum/max/min/mean) -------------------

enum class RowStat { Sum = 0, Max = 1, Min = 2, Mean = 3 };

// Divide every sample block of a row by a NaN-aware statistic taken over that
// sample's `selected` columns. Matches np.nansum/nanmax/nanmin/nanmean: an
// all-missing selection yields sum=0 (-> inf -> NaN on divide) or NaN.
void scale_rows(py::array_t<float, 0> matrix,
                const std::vector<std::size_t> &sample_bounds,
                const std::vector<std::size_t> &selected_starts,
                const std::vector<std::size_t> &selected_ends, int statistic,
                int num_threads) {
    const MatrixRef m = as_matrix(matrix);
    validate_sample_bounds(sample_bounds, m.cols);
    const std::size_t samples = sample_bounds.size() - 1;
    if (selected_starts.size() != samples || selected_ends.size() != samples)
        throw std::runtime_error("selection arrays must match sample count");
    if (statistic < static_cast<int>(RowStat::Sum) ||
        statistic > static_cast<int>(RowStat::Mean))
        throw std::runtime_error("unknown row statistic");
    for (std::size_t sample = 0; sample < samples; ++sample)
        if (selected_starts[sample] < sample_bounds[sample] ||
            selected_ends[sample] > sample_bounds[sample + 1] ||
            selected_starts[sample] >= selected_ends[sample])
            throw std::runtime_error(
                "selected columns must form a non-empty range within their sample");
    const dtp::Statistic operations[] = {dtp::Statistic::Sum, dtp::Statistic::Max,
                                        dtp::Statistic::Min, dtp::Statistic::Mean};
    const auto operation = operations[statistic];
    py::gil_scoped_release release;
    parallel_rows(m.rows, num_threads, [&](std::size_t row) {
        float *r = m.data + row * m.cols;
        for (std::size_t s = 0; s < samples; ++s) {
            const double divisor = dtp::finite_reduction(selected_ends[s] - selected_starts[s],
                [&](std::size_t c) { return r[selected_starts[s] + c]; }, operation, .05, nullptr, true).value;
            for (std::size_t c = sample_bounds[s]; c < sample_bounds[s + 1]; ++c) {
                r[c] = !std::isfinite(r[c]) || !std::isfinite(divisor) || divisor == 0
                    ? NAN_F : checked_result(static_cast<double>(r[c]) / divisor);
            }
        }
    });
}

// ---- multi-matrix combine (sum / mean), NaN-skipping ----------------------

// Accumulate matrices[1..] into matrices[0] in place: each cell sums the finite
// contributors; a cell no matrix covered becomes NaN; mean divides by the
// finite count. Mirrors the old masked stack without materialising N copies.
void combine(py::list matrices, int op /*0 sum, 1 mean*/, int num_threads) {
    const auto n = static_cast<std::size_t>(py::len(matrices));
    if (n < 2) throw std::runtime_error("combine needs at least two matrices");
    if (op != 0 && op != 1)
        throw std::runtime_error("combine operation must be sum or mean");
    std::vector<MatrixRef> refs;
    refs.reserve(n);
    for (const auto item : matrices) {
        if (!py::isinstance<py::array_t<float, 0>>(item)) throw std::invalid_argument("combine requires float32 arrays");
        refs.push_back(as_matrix(py::reinterpret_borrow<py::array_t<float, 0>>(item), refs.empty()));
    }
    const std::size_t rows = refs[0].rows;
    const std::size_t cols = refs[0].cols;
    for (const auto &ref : refs)
        if (ref.rows != rows || ref.cols != cols)
            throw std::runtime_error("all matrices must share a shape");
    for (const auto &ref : refs) reject_partial_overlap(refs[0], ref);
    const bool mean = op == 1;
    py::gil_scoped_release release;
    parallel_rows(rows, num_threads, [&](std::size_t row) {
        float *acc = refs[0].data + row * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            const auto reduced = dtp::finite_reduction(n,
                [&](std::size_t k) { return refs[k].data[row * cols + c]; },
                mean ? dtp::Statistic::Mean : dtp::Statistic::Sum, .05, nullptr, true);
            acc[c] = reduced.finite.count ? checked_result(reduced.value) : NAN_F;
        }
    });
}

// ---- binary combine (difference / ratio / log2FC) -------------------------

enum class Binary { Difference = 0, Ratio = 1, Log2FC = 2 };

// In place on `numerator`. Zero handling is judged from the pre-pseudocount
// inputs (so no separate mask array is allocated); the per-sample pseudocount
// is added only for ratio/log2FC. Non-finite results and zero-masked cells
// become NaN.
void binary_combine(py::array_t<float, 0> numerator,
                    py::array_t<float, 0> denominator,
                    const std::vector<std::size_t> &sample_bounds,
                    const std::vector<double> &pseudocounts, int op,
                    int zero_mode /*0 none, 1 any, 2 both*/, int num_threads) {
    const MatrixRef num = as_matrix(numerator);
    const MatrixRef den = as_matrix(denominator, false);
    if (num.rows != den.rows || num.cols != den.cols)
        throw std::runtime_error("numerator and denominator shapes differ");
    reject_partial_overlap(num, den);
    validate_sample_bounds(sample_bounds, num.cols);
    const std::size_t samples = sample_bounds.size() - 1;
    if (op < static_cast<int>(Binary::Difference) ||
        op > static_cast<int>(Binary::Log2FC))
        throw std::runtime_error("unknown binary operation");
    if (zero_mode < 0 || zero_mode > 2)
        throw std::runtime_error("unknown zero-handling mode");
    const Binary kind = static_cast<Binary>(op);
    if (kind != Binary::Difference && pseudocounts.size() != samples)
        throw std::runtime_error("one pseudocount per sample is required");
    for (double pseudocount : pseudocounts)
        checked_scalar(pseudocount, "pseudocount");
    py::gil_scoped_release release;
    parallel_rows(num.rows, num_threads, [&](std::size_t row) {
        float *nr = num.data + row * num.cols;
        const float *dr = den.data + row * den.cols;
        for (std::size_t s = 0; s < samples; ++s) {
            const double pc = (kind == Binary::Difference) ? 0.0 : pseudocounts[s];
            for (std::size_t c = sample_bounds[s]; c < sample_bounds[s + 1]; ++c) {
                const float n0 = nr[c];
                const float d0 = dr[c];
                bool zero = false;
                if (zero_mode == 1) zero = (n0 == 0.0f) || (d0 == 0.0f);
                else if (zero_mode == 2) zero = (n0 == 0.0f) && (d0 == 0.0f);
                if (zero || !std::isfinite(n0) || !std::isfinite(d0)) { nr[c] = NAN_F; continue; }
                double result;
                if (kind == Binary::Difference) {
                    result = static_cast<double>(n0) - static_cast<double>(d0);
                } else {
                    const double numerator = static_cast<double>(n0) + pc;
                    const double denominator = static_cast<double>(d0) + pc;
                    if (denominator == 0) { nr[c] = NAN_F; continue; }
                    if (kind == Binary::Log2FC) {
                        if (numerator == 0 || std::signbit(numerator) != std::signbit(denominator)) {
                            nr[c] = NAN_F;
                            continue;
                        }
                        // Computing the ratio first rounds tiny, representable
                        // fold changes to exactly one when the operands are
                        // large. log1p retains that signal; the logarithmic
                        // fallback covers ratios outside log1p's domain.
                        const double relative = (static_cast<double>(n0) -
                                                 static_cast<double>(d0)) / denominator;
                        result = std::isfinite(relative) && relative > -1
                            ? std::log1p(relative) / std::log(2.0)
                            : (std::log(std::abs(numerator)) -
                               std::log(std::abs(denominator))) / std::log(2.0);
                    } else {
                        result = numerator / denominator;
                    }
                }
                nr[c] = checked_result(result);
            }
        }
    });
}

}  // namespace

PYBIND11_MODULE(_transform, module) {
    module.def("nonfinite_to_nan", &nonfinite_to_nan, py::arg("matrix").noconvert(), py::arg("num_threads") = 1);
    module.doc() = "In-place NaN-aware transform kernels for computeMatrixOperations";
    module.attr("supports_multicore") = true;
    module.def("add_scalar", &add_scalar, py::arg("matrix").noconvert(), py::arg("value"),
               py::arg("num_threads") = 1);
    module.def("scale_scalar", &scale_scalar, py::arg("matrix").noconvert(),
               py::arg("factor"), py::arg("num_threads") = 1);
    module.def("log2_inplace", &log2_inplace, py::arg("matrix").noconvert(),
               py::arg("num_threads") = 1);
    module.def("nonfinite_to_zero", &nonfinite_to_zero, py::arg("matrix").noconvert(),
               py::arg("num_threads") = 1);
    module.def("scale_rows", &scale_rows, py::arg("matrix").noconvert(),
               py::arg("sample_bounds"), py::arg("selected_starts"),
               py::arg("selected_ends"), py::arg("statistic"),
               py::arg("num_threads") = 1);
    module.def("combine", &combine, py::arg("matrices"), py::arg("op"),
               py::arg("num_threads") = 1);
    module.def("binary_combine", &binary_combine, py::arg("numerator").noconvert(),
               py::arg("denominator").noconvert(), py::arg("sample_bounds"),
               py::arg("pseudocounts"), py::arg("op"), py::arg("zero_mode"),
               py::arg("num_threads") = 1);
}
