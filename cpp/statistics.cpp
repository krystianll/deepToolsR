#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <map>
#include <array>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "parallel.hpp"
#include "numeric.hpp"
#include "filter_policy.hpp"
#include "matrix_view.hpp"
#include "digestible/digestible.h"
#include "include/cephes/xsf/cephes/incbi.h"

namespace py = pybind11;

using dtp::parallel_rows;

namespace {

// Cephes stdtri's two inverse-beta branches, extended to positive real df.
double student_t_quantile(double df, double p) {
    if (std::isnan(df) || std::isnan(p) || df <= 0 || p < 0 || p > 1)
        return std::numeric_limits<double>::quiet_NaN();
    if (p == 0) return -std::numeric_limits<double>::infinity();
    if (p == 1) return std::numeric_limits<double>::infinity();
    if (p == 0.5) return 0;
    // At large df inverse-beta loses significant digits. The normal-limit
    // expansion retains double precision for the probabilities used by CI.
    if (df >= 10000) {
        const double z = xsf::cephes::ndtri(p);
        const double z2 = z * z;
        return z + z * (z2 + 1) / (4 * df)
            + z * (5 * z2 * z2 + 16 * z2 + 3) / (96 * df * df)
            + z * (3 * z2 * z2 * z2 + 19 * z2 * z2 + 17 * z2 - 15)
                / (384 * df * df * df);
    }
    double value;
    if (p > 0.25 && p < 0.75) {
        const double z = xsf::cephes::incbi(0.5, 0.5 * df, std::abs(1 - 2 * p));
        value = std::copysign(std::sqrt(df * z / (1 - z)), p - 0.5);
    } else {
        const double z = xsf::cephes::incbi(0.5 * df, 0.5, 2 * std::min(p, 1 - p));
        if (z == 0 || std::numeric_limits<double>::max() * z < df)
            return std::copysign(std::numeric_limits<double>::infinity(), p - 0.5);
        value = std::copysign(std::sqrt(df / z - df), p - 0.5);
    }
    return value;
}

// Exact type-7 quantile (matching numpy's default 'linear' interpolation) of a
// mutable buffer; p is a percentile in [0, 100]. Uses nth_element, no full sort.
double quantile_type7(std::vector<double> &x, double p) {
    const std::size_t n = x.size();
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    const double q = p / 100.0;
    if (q <= 0.0) return *std::min_element(x.begin(), x.end());
    if (q >= 1.0) return *std::max_element(x.begin(), x.end());
    const double h = (n - 1) * q;
    const std::size_t j = static_cast<std::size_t>(std::floor(h));
    const double g = h - j;
    std::nth_element(x.begin(), x.begin() + j, x.end());
    const double xj = x[j];
    if (g == 0.0) return xj;
    const double xj1 = *std::min_element(x.begin() + j + 1, x.end());
    return std::signbit(xj) == std::signbit(xj1)
        ? xj + g * (xj1 - xj) : (1.0 - g) * xj + g * xj1;
}

// Per-column moments of a projected (possibly strided) float32/float64
// matrix, read-only, with compensated means and
// centered second-pass variance. Returns
// (mean, std_pop, count) per column plus, for geom, the per-column sign and the
// pseudocount used. `geom` reduces log1p(sign*x/pseudocount), or log(sign*x)
// for zero pseudocount (zero/nonzero mixtures require a positive shift),
// so the caller can back-transform the geometric mean and its CI. From
// (mean, std_pop, count): std ddof1 = std_pop*sqrt(n/(n-1)); sem = std_pop/sqrt(n-1).
py::tuple summarize_columns(const py::buffer &values,
                            bool geom, double pseudocount, int num_threads,
                            const py::object &rows_object, const py::object &row_range,
                            const py::object &cols_object, const py::object &col_range) {
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);
    if (!std::isfinite(pseudocount)) throw std::invalid_argument("pseudocount must be finite");
    const std::size_t ncol = view.cols;
    py::array_t<double> mean(ncol), std(ncol), count(ncol);
    double *mean_d = mean.mutable_data();
    double *std_d = std.mutable_data();
    double *count_d = count.mutable_data();
    py::array_t<double> sign(geom ? ncol : 0);
    double *sign_d = geom ? sign.mutable_data() : nullptr;

    {
        py::gil_scoped_release release;
        dtp::GeometricContext context;
        if (geom) {
            context = dtp::prepare_geometric(ncol, view.rows,
                [&](std::size_t k, std::size_t i) { return view.at(i, k); },
                pseudocount);
            pseudocount = context.pseudocount;
        }
        parallel_rows(ncol, num_threads, [&](std::size_t k) {
            const std::size_t c = k;
            double sk = 0;
            if (geom) {
                const auto &domain = context.domains[k];
                sk = sign_d[k] = domain.sign;
                if (pseudocount == 0 && domain.zeros && domain.sign)
                    throw std::invalid_argument("geometric analytic bands require a positive pseudocount when zero and nonzero observations share a bin");
                if (domain.count && !domain.sign) {
                    mean_d[k] = std_d[k] = 0;
                    count_d[k] = static_cast<double>(domain.count);
                    return;
                }
            }
            auto get = [&](std::size_t i) {
                const double v = view.at(i, c);
                if (!std::isfinite(v)) return dtp::missing_value();
                if (!geom) return v;
                return dtp::geometric_log(v, sk, pseudocount);
            };
            const auto size = view.rows;
            const auto summary = dtp::finite_moments(size, get);
            mean_d[k] = summary.mean;
            std_d[k] = summary.stddev;
            count_d[k] = static_cast<double>(summary.finite.count);
        });
    }
    return py::make_tuple(std::move(mean), std::move(std), std::move(count),
                          std::move(sign), pseudocount);
}

// nan-aware percentiles of a (possibly strided) float32 matrix. Missing values
// non-finite values are excluded. For at most `max_exact` elements the
// result is exact (gather + type-7); larger inputs stream through a t-digest
// (approximate, O(1) memory -- no full flatten/sort). `probs` are percentiles in
// [0, 100]. Returns one value per prob (NaN when no finite values are present).
py::array_t<double> nan_quantiles(const py::buffer &values,
                                  const std::vector<double> &probs,
                                  std::size_t max_exact,
                                  unsigned tdigest_compression,
                                  int num_threads, bool exact,
                                  const py::object &rows_object, const py::object &row_range,
                                  const py::object &cols_object, const py::object &col_range) {
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);
    const auto rows = view.rows, cols = view.cols;
    auto present = [&](std::size_t r, std::size_t c, double &x) { return view.present(r, c, x); };
    for (double p : probs)
        if (!std::isfinite(p) || p < 0 || p > 100)
            throw std::invalid_argument("percentiles must be finite and between 0 and 100");
    if (tdigest_compression > 1000000) throw std::invalid_argument("tdigest compression is too large");
    py::array_t<double> result(probs.size());
    double *out = result.mutable_data();
    const std::size_t total = rows * cols;

    if (total <= std::min<std::size_t>(max_exact, 1048576)) {
        std::vector<double> buffer;
        buffer.reserve(total);
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < cols; ++c) {
                double v;
                if (present(r, c, v)) buffer.push_back(static_cast<double>(v));
            }
        for (std::size_t k = 0; k < probs.size(); ++k)
            out[k] = quantile_type7(buffer, probs[k]);
        return result;
    }

    if (!exact) {
    if (tdigest_compression == 0) tdigest_compression = 1000;
    digestible::tdigest<double, std::uint64_t> digest(tdigest_compression);
    constexpr std::size_t window_chunks = 16;
    const std::size_t chunk_rows = std::max<std::size_t>(1, 65536 / cols);
    const std::size_t chunks = (rows - 1) / chunk_rows + 1;
    std::vector<digestible::tdigest<double, std::uint64_t>> partials;
    partials.reserve(std::min(window_chunks, chunks));
    for (std::size_t i = 0; i < std::min(window_chunks, chunks); ++i)
        partials.emplace_back(tdigest_compression);
    {
        py::gil_scoped_release release;
        for (std::size_t window = 0; window < chunks; window += window_chunks) {
            const std::size_t count = std::min(window_chunks, chunks - window);
            parallel_rows(count, num_threads, [&](std::size_t i) {
                auto &local = partials[i];
                local.reset();
                const std::size_t r0 = (window + i) * chunk_rows;
                const std::size_t r1 = std::min(rows, r0 + chunk_rows);
                for (std::size_t r = r0; r < r1; ++r)
                    for (std::size_t c = 0; c < cols; ++c) {
                        double v;
                        if (present(r, c, v)) local.insert(v);
                    }
                local.merge();
            });
            for (std::size_t i = 0; i < count; ++i)
                digest.insert(partials[i]);
            digest.merge();
        }
    }
    const bool empty = digest.size() == 0;
    for (std::size_t k = 0; k < probs.size(); ++k) {
        if (empty) {
            out[k] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const double low = digest.min(), high = digest.max();
        if (low == high) {
            out[k] = low;
            continue;
        }
        const double estimate = digest.quantile(probs[k]);
        if (std::isnan(estimate))
            throw std::runtime_error("t-digest produced a NaN quantile for finite input");
        out[k] = std::max(low, std::min(high, estimate));
    }
    return result;
    }
    // Exact order statistics with bounded memory. IEEE float64 bit keys are
    // ordered by flipping the sign bit for positive values and all bits for
    // negative values. Eight radix passes locate every required rank together.
    // Memory is O(number of requested ranks * 256), independent of matrix size.
    auto key_of = [](double x) {
        std::uint64_t bits; std::memcpy(&bits, &x, sizeof(bits));
        return (bits >> 63) ? ~bits : bits ^ (std::uint64_t(1) << 63);
    };
    auto value_of = [](std::uint64_t key) {
        std::uint64_t bits = (key >> 63) ? key ^ (std::uint64_t(1) << 63) : ~key;
        double value; std::memcpy(&value, &bits, sizeof(value)); return value;
    };
    std::uint64_t count = 0;
    {
        py::gil_scoped_release release;
        for (std::size_t r = 0; r < rows; ++r) {
            if ((r & 4095) == 0) dtp::check_python_signals();
            for (std::size_t c = 0; c < cols; ++c) { double x; count += present(r, c, x); }
        }
    }
    if (!count) {
        std::fill(out, out + probs.size(), dtp::missing_value());
        return result;
    }
    std::vector<std::uint64_t> ranks, prefixes(probs.size() * 2, 0);
    std::vector<double> fractions;
    for (double p : probs) {
        const double h = (count - 1) * (p / 100);
        const auto lo = static_cast<std::uint64_t>(std::floor(h));
        ranks.push_back(lo); ranks.push_back(std::min(count - 1, lo + 1));
        fractions.push_back(h - lo);
    }
    {
        py::gil_scoped_release release;
        for (int byte = 7; byte >= 0; --byte) {
            std::map<std::uint64_t, std::array<std::uint64_t, 256>> histograms;
            for (auto prefix : prefixes) histograms.emplace(prefix, std::array<std::uint64_t, 256>{});
            const unsigned shift = byte * 8;
            for (std::size_t r = 0; r < rows; ++r) {
                if ((r & 4095) == 0) dtp::check_python_signals();
                for (std::size_t c = 0; c < cols; ++c) {
                    double x; if (!present(r, c, x)) continue;
                    const auto key = key_of(x);
                    const auto prefix = byte == 7 ? 0 : key >> (shift + 8);
                    auto it = histograms.find(prefix);
                    if (it != histograms.end()) ++it->second[(key >> shift) & 255];
                }
            }
            for (std::size_t i = 0; i < ranks.size(); ++i) {
                const auto &histogram = histograms.at(prefixes[i]);
                unsigned bucket = 0;
                while (bucket < 255 && ranks[i] >= histogram[bucket]) ranks[i] -= histogram[bucket++];
                prefixes[i] = (prefixes[i] << 8) | bucket;
            }
        }
    }
    for (std::size_t i = 0; i < probs.size(); ++i) {
        const double lo = value_of(prefixes[2 * i]), hi = value_of(prefixes[2 * i + 1]);
        const double fraction = fractions[i];
        out[i] = fraction == 0 ? lo : std::signbit(lo) == std::signbit(hi)
            ? lo + fraction * (hi - lo) : (1 - fraction) * lo + fraction * hi;
    }
    return result;
}

// Permute rows [row_start, row_end) of a C-contiguous 2D buffer in place so that
// afterwards row i (relative to row_start) holds the data previously at row
// order[i] -- i.e. equivalent to matrix[row_start:row_end][order]. Works on any
// itemsize, moving whole rows with a single
// row-sized scratch buffer plus an O(n) index array, so no full-matrix copy is
// allocated. Uses the swap-cycle algorithm: applying the inverse of `order` as a
// swap permutation leaves A[i] == A_old[order[i]].
void permute_rows_inplace(py::buffer matrix, std::size_t row_start,
                          std::size_t row_end,
                          const std::vector<std::size_t> &order) {
    const auto info = matrix.request(true);  // writable
    if (info.ndim != 2) throw std::runtime_error("matrix must be two-dimensional");
    const auto rows = static_cast<std::size_t>(info.shape[0]);
    const auto cols = static_cast<std::size_t>(info.shape[1]);
    const auto itemsize = static_cast<std::size_t>(info.itemsize);
    const std::size_t row_bytes = cols * itemsize;
    if (info.strides[1] != info.itemsize ||
        info.strides[0] != static_cast<py::ssize_t>(row_bytes))
        throw std::runtime_error("matrix must be C-contiguous");
    if (row_end > rows || row_start > row_end)
        throw std::runtime_error("row range out of bounds");
    const std::size_t n = row_end - row_start;
    if (order.size() != n)
        throw std::runtime_error("order length differs from row range");
    // P = inverse of the requested gather order.
    std::vector<std::size_t> perm(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        if (order[i] >= n) throw std::runtime_error("order index out of range");
        if (perm[order[i]] != n) throw std::invalid_argument("order must be a permutation");
        perm[order[i]] = i;
    }

    char *base = static_cast<char *>(info.ptr) + row_start * row_bytes;
    std::vector<char> scratch(row_bytes);
    auto swap_rows = [&](std::size_t a, std::size_t b) {
        char *ra = base + a * row_bytes;
        char *rb = base + b * row_bytes;
        std::memcpy(scratch.data(), ra, row_bytes);
        std::memcpy(ra, rb, row_bytes);
        std::memcpy(rb, scratch.data(), row_bytes);
    };
    for (std::size_t i = 0; i < n; ++i) {
        while (perm[i] != i) {
            const std::size_t j = perm[i];
            swap_rows(i, j);
            std::swap(perm[i], perm[j]);
        }
    }
}

// nan-aware reduction of a 2D (possibly strided) float32 buffer along one axis,
// with a float64 accumulator. axis=0 reduces over rows -> one value per column;
// axis=1 reduces over columns -> one value per row. A value is "missing" when a
// value is not finite. Parallelised over the
// output length. ops: mean, sum, min, max, std (ddof=0), median (type-7).
py::array_t<double> reduce_axis(const py::buffer &values,
                                int axis,
                                const std::string &op, int num_threads,
                                double trim_perc, bool empty_sum_missing,
                                const py::object &rows_object, const py::object &row_range,
                                const py::object &cols_object, const py::object &col_range) {
    dtp::validate_axis(axis);
    dtp::validate_statistic(op);
    const auto operation = dtp::parse_statistic(op);
    if (op == "trim_mean") dtp::trim_count(0, trim_perc);
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);
    const auto output_size = axis == 0 ? view.cols : view.rows;
    const auto scan_size = axis == 0 ? view.rows : view.cols;
    py::array_t<double> result(output_size);
    double *out = result.mutable_data();
    py::gil_scoped_release release;
    parallel_rows(output_size, num_threads, [&](std::size_t k) {
        const auto reduced = dtp::finite_reduction(scan_size, [&](std::size_t i) {
            return view.at(axis == 0 ? i : k, axis == 0 ? k : i);
        }, operation, trim_perc, nullptr, view.single_precision());
        out[k] = empty_sum_missing && operation == dtp::Statistic::Sum && !reduced.finite.count
            ? dtp::missing_value() : reduced.value;
    });
    return result;
}

// Finite signed geometric means, with one shared automatically resolved
// pseudocount across the block. Mixed signs within a reduction are invalid;
// actual zeros and missing values have distinct semantics. No matrix copies.
// Returns (result, nonnegative_pseudocount_used).
py::tuple geom_mean_axis(const py::buffer &values,
                         int axis, double pseudocount, int num_threads,
                         const py::object &rows_object, const py::object &row_range,
                         const py::object &cols_object, const py::object &col_range) {
    dtp::validate_axis(axis);
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);
    const bool along_rows = axis == 0;
    const std::size_t out_len = along_rows ? view.cols : view.rows;
    const std::size_t scan_len = along_rows ? view.rows : view.cols;
    py::array_t<double> result(out_len);
    double *out = result.mutable_data();
    {
        py::gil_scoped_release release;
        auto get = [&](std::size_t k, std::size_t i) {
            return view.at(along_rows ? i : k, along_rows ? k : i);
        };
        const auto context = dtp::prepare_geometric(out_len, scan_len, get, pseudocount);
        pseudocount = context.pseudocount;
        parallel_rows(out_len, num_threads, [&](std::size_t k) {
            out[k] = dtp::geometric_mean(scan_len,
                [&](std::size_t i) { return get(k, i); }, pseudocount, context.domains[k]);
        });
    }
    return py::make_tuple(std::move(result), pseudocount);
}


// --------------------------------------------------------------------------- #
// bootstrap confidence intervals (fully native)
// --------------------------------------------------------------------------- #

using BootStat = dtp::Statistic;
BootStat parse_boot_stat(const std::string &s) {
    return dtp::parse_statistic(s, "profile");
}

// Statistic of one resampled bin (finite values). `sample` may be reordered.
// Geometric bootstrap recomputes the estimator sign in each replicate while
// holding the pseudocount fixed across the complete series.
double resample_statistic(std::vector<double> &sample, BootStat stat,
                          double trim_perc, double pseudocount) {
    const std::size_t m = sample.size();
    if (m == 0) return std::numeric_limits<double>::quiet_NaN();
    auto get = [&](std::size_t i) { return sample[i]; };
    if (stat == BootStat::GeomMean) return dtp::geometric_mean(m, get, pseudocount);
    if (stat == BootStat::Median || stat == BootStat::TrimMean)
        return dtp::order_statistic(sample, stat, trim_perc);
    return dtp::reduce_finite(m, get, stat, trim_perc);
}

// Percentile-bootstrap CI per column. For each bin: gather its finite values
// once (cache-hot), then draw R resamples with the same row draws in every bin,
// compute the statistic, and take the alpha/2 and 1-alpha/2 percentiles of the
// bootstrap distribution. Parallel over bins; nothing but per-bin scratch is
// allocated (no R x ncol estimate matrix). Optional stderr progress bar.
py::tuple bootstrap_ci(const py::buffer &values,
                       const std::string &average_type, double pseudocount,
                       double trim_perc, std::size_t resamples,
                       double ci_level, std::uint64_t seed,
                       int num_threads, bool show_progress, bool include_center,
                       const py::object &rows_object, const py::object &row_range,
                       const py::object &cols_object, const py::object &col_range) {
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);
    auto present = [&](std::size_t r, std::size_t c, double &x) { return view.present(r, c, x); };
    if (!std::isfinite(pseudocount)) throw std::invalid_argument("pseudocount must be finite");
    const std::size_t ncol = view.cols;
    const BootStat stat = parse_boot_stat(average_type);
    if (!std::isfinite(ci_level) || ci_level <= 0 || ci_level >= 1)
        throw std::invalid_argument("ci_level must be between 0 and 1");
    dtp::trim_count(0, trim_perc);
    if (!resamples || !view.rows) throw std::invalid_argument("bootstrap requires rows and resamples");
    dtp::GeometricContext context;
    if (stat == BootStat::GeomMean) {
        py::gil_scoped_release release;
        context = dtp::prepare_geometric(ncol, view.rows,
            [&](std::size_t k, std::size_t i) { return view.at(i, k); },
            pseudocount);
        pseudocount = context.pseudocount;
    }
    const double alpha = (1.0 - ci_level) / 2.0;
    py::array_t<double> lower(ncol), upper(ncol), center(include_center ? ncol : 0);
    double *center_data = include_center ? center.mutable_data() : nullptr;
    double *lo = lower.mutable_data();
    double *up = upper.mutable_data();

    std::atomic<std::size_t> done{0};
    std::atomic<int> last_pct{-1};
    std::mutex progress_mutex;
    auto report = [&](std::size_t d) {
        const int pct = static_cast<int>(d * 100 / ncol);
        if (pct == last_pct.load(std::memory_order_relaxed)) return;
        std::lock_guard<std::mutex> lock(progress_mutex);
        if (pct <= last_pct.load(std::memory_order_relaxed)) return;
        last_pct.store(pct, std::memory_order_relaxed);
        const int width = 30, filled = width * pct / 100;
        std::fprintf(stderr, "\rBootstrap [%.*s%.*s] %3d%% (%zu/%zu)",
                     filled, "##############################",
                     width - filled, "------------------------------",
                     pct, d, ncol);
        if (d == ncol) std::fprintf(stderr, "\n");
        std::fflush(stderr);
    };

    {
        py::gil_scoped_release release;
        const std::size_t nrow = view.rows;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        parallel_rows(ncol, num_threads, [&](std::size_t k) {
            const std::size_t c = k;
            // Full, row-aligned column (NaN where missing) so a drawn index maps
            // to the same gene in every bin -- combined with the shared seed
            // below this makes the resampling gene-stable across bins.
            std::vector<double> col(nrow);
            std::size_t finite_count = 0;
            for (std::size_t i = 0; i < nrow; ++i) {
                double v;
                if (present(i, c, v)) { col[i] = v; ++finite_count; }
                else col[i] = nan;
            }
            double lo_k = nan, up_k = nan;
            if (include_center) {
                auto get = [&](std::size_t i) { return col[i]; };
                center_data[k] = !finite_count ? nan : stat == BootStat::GeomMean
                    ? dtp::geometric_mean(nrow, get, pseudocount, context.domains[k])
                    : dtp::finite_reduction(nrow, get, stat, trim_perc, nullptr,
                                            view.single_precision()).value;
            }
            if (finite_count > 0) {
                const double pc = pseudocount;
                // Same seed for every bin -> identical draw sequence -> the same
                // genes are resampled across all bins within a replicate.
                std::mt19937_64 rng(seed);
                std::uniform_int_distribution<std::size_t> pick(0, nrow - 1);
                std::vector<double> boot;
                boot.reserve(resamples);
                std::vector<double> sample;
                sample.reserve(nrow);
                for (std::size_t r = 0; r < resamples; ++r) {
                    if ((r & 63u) == 0) {
                        if (dtp::running_in_parallel_worker()) {
                            if (dtp::interruption_requested()) return;
                        } else {
                            dtp::check_python_signals();
                        }
                    }
                    // Resample nrow genes; drop the ones missing in this bin.
                    sample.clear();
                    for (std::size_t i = 0; i < nrow; ++i) {
                        const double v = col[pick(rng)];
                        if (std::isfinite(v)) sample.push_back(v);
                    }
                    const double est =
                        resample_statistic(sample, stat, trim_perc, pc);
                    if (std::isfinite(est)) boot.push_back(est);
                }
                if (!boot.empty()) {
                    lo_k = quantile_type7(boot, alpha * 100.0);
                    up_k = quantile_type7(boot, (1.0 - alpha) * 100.0);
                }
            }
            lo[k] = lo_k;
            up[k] = up_k;
            if (show_progress) report(++done);
        });
    }
    if (include_center) return py::make_tuple(std::move(center), std::move(lower), std::move(upper));
    return py::make_tuple(std::move(lower), std::move(upper));
}


py::array_t<double> geometric_inverse(const py::array_t<double> &logs,
                                     const py::array_t<double> &signs, double pc) {
    if (!std::isfinite(pc) || pc < 0 || logs.ndim() != 1 || signs.ndim() != 1 || logs.size() != signs.size())
        throw std::invalid_argument("invalid geometric inverse arguments");
    const auto values = logs.request(), signs_info = signs.request();
    py::array_t<double> result(logs.size()); auto *out = result.mutable_data();
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < values.shape[0]; ++i) {
        double value, sign;
        std::memcpy(&value, static_cast<const char *>(values.ptr) + i * values.strides[0], sizeof(value));
        std::memcpy(&sign, static_cast<const char *>(signs_info.ptr) + i * signs_info.strides[0], sizeof(sign));
        if (sign != -1 && sign != 0 && sign != 1)
            throw std::invalid_argument("geometric sign must be -1, 0 or 1");
        if (std::isnan(value)) out[i] = dtp::missing_value();
        else out[i] = dtp::geometric_inverse(value, sign, pc);
    }
    return result;
}

py::array finite_values(const py::buffer &values,
                        const py::object &rows_object, const py::object &row_range,
                        const py::object &cols_object, const py::object &col_range) {
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);
    std::size_t count = 0;
    {
        py::gil_scoped_release release;
        for (std::size_t r = 0; r < view.rows; ++r)
            for (std::size_t c = 0; c < view.cols; ++c) count += std::isfinite(view.at(r, c));
    }
    if (!count) throw std::invalid_argument("matrix only contains missing values");
    py::array result(view.single_precision() ? py::dtype::of<float>() : py::dtype::of<double>(),
                     std::vector<py::ssize_t>{static_cast<py::ssize_t>(count)});
    auto *out = static_cast<char *>(result.mutable_data());
    const auto itemsize = result.itemsize();
    py::gil_scoped_release release;
    std::size_t i = 0;
    for (std::size_t r = 0; r < view.rows; ++r)
        for (std::size_t c = 0; c < view.cols; ++c) {
            const double x = view.at(r, c);
            if (!std::isfinite(x)) continue;
            if (view.single_precision()) { const float y = x; std::memcpy(out + i * itemsize, &y, itemsize); }
            else std::memcpy(out + i * itemsize, &x, itemsize);
            ++i;
        }
    return result;
}

// Resident adapter for the same policy used by streaming matrix filters.
// Removal needs O(rows) flags. Masking allocates only the output matrix already
// required by the public copy-on-filter contract, never rows x samples flags.
py::tuple filter_matrix(const py::buffer &values,
                        const std::vector<std::size_t> &bounds,
                        const std::vector<std::size_t> &samples,
                        const std::string &statistic, double low, double high,
                        int nan_mode, const std::string &on_fail, int threads,
                        const py::object &rows_object, const py::object &row_range,
                        const py::object &cols_object, const py::object &col_range) {
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);
    const auto stat = dtp::parse_statistic(statistic, "filter");
    const auto action = dtp::parse_filter_action(on_fail);
    const dtp::FilterPolicy policy(bounds, samples, stat, nan_mode, action, low, high);
    if (bounds.back() != view.cols)
        throw std::runtime_error("sample boundaries do not match matrix width");
    const bool masking = action == dtp::FilterAction::MaskSample;
    py::array_t<bool> keep(view.rows);
    auto *keep_data = keep.mutable_data();
    py::array_t<float> result({masking ? view.rows : std::size_t(0), view.cols});
    auto *out = result.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_rows(view.rows, threads, [&](std::size_t r) {
            if (masking) for (std::size_t c = 0; c < view.cols; ++c) {
                const double x = view.at(r, c);
                if (std::isfinite(x) && std::abs(x) > std::numeric_limits<float>::max())
                    throw std::overflow_error("matrix value exceeds float32 range");
                out[r * view.cols + c] = std::isfinite(x) ? static_cast<float>(x)
                    : std::numeric_limits<float>::quiet_NaN();
            }
            std::vector<double> scratch;
            keep_data[r] = policy.keep_row([&](std::size_t s) {
                return dtp::filter_sample(bounds[s + 1] - bounds[s],
                    [&](std::size_t i) { return view.at(r, bounds[s] + i); },
                    stat, low, high, false, view.single_precision(), &scratch);
            }, [&](std::size_t s) {
                std::fill(out + r * view.cols + bounds[s],
                          out + r * view.cols + bounds[s + 1],
                          std::numeric_limits<float>::quiet_NaN());
            });
        });
    }
    if (masking) return py::make_tuple(std::move(keep), std::move(result));
    return py::make_tuple(std::move(keep), py::none());
}

py::tuple filter_rows(const py::buffer &values,
                      const std::string &stat, double low, double high,
                      bool inclusive, int threads, int nan_mode,
                      const py::object &rows_object, const py::object &row_range,
                      const py::object &cols_object, const py::object &col_range) {
    if (stat != "perBin") dtp::validate_statistic(stat);
    if (std::isnan(low) || std::isnan(high) || low > high)
        throw std::invalid_argument("invalid filter thresholds");
    if (nan_mode < 0 || nan_mode > 3) throw std::invalid_argument("unknown missing-value filter mode");
    const auto operation = dtp::parse_statistic(stat);
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);
    py::array_t<bool> failed(view.rows);
    py::array_t<std::int64_t> counts(view.rows);
    auto *out = failed.mutable_data(); auto *count = counts.mutable_data();
    {
    py::gil_scoped_release release;
    parallel_rows(view.rows, threads, [&](std::size_t r) {
        auto get = [&](std::size_t c) { return view.at(r, c); };
        const auto evaluation = dtp::filter_sample(view.cols, get, operation, low, high,
                                                   inclusive, view.single_precision());
        count[r] = static_cast<std::int64_t>(evaluation.finite);
        out[r] = evaluation.fails(static_cast<dtp::NanMode>(nan_mode));
    });
    }
    return py::make_tuple(std::move(failed), std::move(counts));
}

// Exact extrema without compressed matrix copies.
double nonzero_extreme(const py::buffer &values, bool maximum,
                       const py::object &rows_object, const py::object &row_range,
                       const py::object &cols_object, const py::object &col_range) {
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);
    double result = maximum ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    py::gil_scoped_release release;
    for (std::size_t r = 0; r < view.rows; ++r)
        for (std::size_t c = 0; c < view.cols; ++c) {
            const double x = view.at(r, c);
            if (std::isfinite(x) && x != 0) result = maximum ? std::max(result, x) : std::min(result, x);
        }
    if (!std::isfinite(result)) throw std::invalid_argument("Cannot derive a pseudocount from a matrix with no finite, nonzero values");
    return result;
}

py::array_t<float> copy_values(const py::buffer &values, bool zero_missing, int threads,
                               const py::object &rows_object, const py::object &row_range,
                               const py::object &cols_object, const py::object &col_range) {
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);
    py::array_t<float> result({view.rows, view.cols});
    auto *out = result.mutable_data();
    py::gil_scoped_release release;
    parallel_rows(view.rows, threads, [&](std::size_t r) {
        for (std::size_t c = 0; c < view.cols; ++c) {
            const double x = view.at(r, c);
            if (std::isfinite(x) && std::abs(x) > std::numeric_limits<float>::max())
                throw std::overflow_error("matrix value exceeds float32 range");
            out[r * view.cols + c] = std::isfinite(x) ? static_cast<float>(x)
                : zero_missing ? 0 : std::numeric_limits<float>::quiet_NaN();
        }
    });
    return result;
}

template<class Distance>
double silhouette_row(std::size_t row, const std::vector<std::size_t> &labels, Distance distance) {
    const auto n = labels.size();
    std::vector<dtp::CompensatedSum> totals(n);
    std::vector<std::size_t> counts(n, 0);
    for (std::size_t j = 0; j < n; ++j) {
        if (row == j) continue;
        const double d = distance(j);
        if (!std::isfinite(d) || d < 0) throw std::invalid_argument("silhouette requires finite nonnegative distances");
        totals[labels[j]].add(d); ++counts[labels[j]];
    }
    const auto own = labels[row];
    if (!counts[own]) return 0; // true singleton, after excluding itself
    const double a = totals[own].value() / counts[own];
    double b = std::numeric_limits<double>::infinity();
    for (std::size_t group = 0; group < n; ++group)
        if (group != own && counts[group]) b = std::min(b, totals[group].value() / counts[group]);
    if (!std::isfinite(b)) throw std::invalid_argument("silhouette requires at least two clusters");
    const double denom = std::max(a, b);
    return denom == 0 ? 0 : (b - a) / denom;
}

void validate_labels(const std::vector<std::size_t> &labels, std::size_t rows) {
    if (labels.size() != rows) throw std::invalid_argument("labels must match rows");
    for (auto label : labels) if (label >= rows) throw std::invalid_argument("cluster label out of range");
}

double silhouette_score(const py::buffer &distances, std::size_t row, const std::vector<std::size_t> &labels,
                        const py::object &rows_object, const py::object &row_range,
                        const py::object &cols_object, const py::object &col_range) {
    const dtp::MatrixView view(distances, rows_object, row_range, cols_object, col_range);
    validate_labels(labels, view.rows);
    if (view.cols != view.rows || row >= view.rows) throw std::invalid_argument("invalid distance matrix or row");
    py::gil_scoped_release release;
    return silhouette_row(row, labels, [&](std::size_t j) { return view.at(row, j); });
}

py::array_t<double> silhouette_scores(const py::buffer &values,
                                      const std::vector<std::size_t> &labels,
                                      const py::object &rows_object, const py::object &row_range,
                                      const py::object &cols_object, const py::object &col_range,
                                      bool zero_missing, int threads) {
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);
    validate_labels(labels, view.rows);
    py::array_t<double> result(view.rows); auto *out = result.mutable_data();
    py::gil_scoped_release release;
    // Compute distances on demand: O(rows*columns) input plus O(rows) per
    // worker, avoiding scipy.squareform's quadratic resident matrix.
    parallel_rows(view.rows, threads, [&](std::size_t r) {
        out[r] = silhouette_row(r, labels, [&](std::size_t j) {
            double norm = 0;
            for (std::size_t c = 0; c < view.cols; ++c) {
                double a = view.at(r, c), b = view.at(j, c);
                if (zero_missing) {
                    if (!std::isfinite(a)) a = 0;
                    if (!std::isfinite(b)) b = 0;
                }
                norm = std::hypot(norm, a - b);
            }
            return norm;
        });
    });
    return result;
}


}  // namespace

PYBIND11_MODULE(_statistics, module) {
    module.def("student_t_quantile", py::vectorize(&student_t_quantile),
               py::arg("df"), py::arg("p"));
    module.def("pool_workers", []() { return dtp::global_pool().size(); });
    module.def("parallel_worker_count", [](int threads) {
        std::mutex lock;
        std::set<std::thread::id> participants;
        py::gil_scoped_release release;
        parallel_rows(256, threads, [&](std::size_t) {
            {
                std::lock_guard<std::mutex> guard(lock);
                participants.insert(std::this_thread::get_id());
            }
            // Spin rather than sleep: Windows sleep_for rounds sub-millisecond
            // waits to zero or to a whole timer tick, so one thread could drain
            // every row before the others woke.
            const auto until =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
            while (std::chrono::steady_clock::now() < until) {
            }
        });
        return participants.size();
    });
    py::dict filter_statistics;
    for (const auto &name : dtp::statistic_choices("filter"))
        filter_statistics[py::str(name)] = static_cast<int>(dtp::parse_statistic(name));
    module.attr("filter_statistics") = filter_statistics;
    py::dict nan_modes;
    nan_modes["keep"] = static_cast<int>(dtp::NanMode::Keep);
    nan_modes["any_bin"] = static_cast<int>(dtp::NanMode::AnyBin);
    nan_modes["any_sample"] = static_cast<int>(dtp::NanMode::AnySample);
    nan_modes["all_bins"] = static_cast<int>(dtp::NanMode::AllBins);
    module.attr("filter_nan_modes") = nan_modes;
    module.def("filter_matrix", &filter_matrix, py::arg("matrix"),
               py::arg("sample_boundaries"), py::arg("filter_samples"),
               py::arg("statistic"), py::arg("low"), py::arg("high"),
               py::arg("nan_mode") = 0, py::arg("on_fail") = "removeRegion",
               py::arg("num_threads") = 1, DTP_PROJECTION_ARGS);
    py::dict choices;
    for (const char *use : {"bin", "filter", "profile"}) choices[py::str(use)] = dtp::statistic_choices(use);
    module.attr("statistic_choices") = choices;
    module.def("filter_rows", &filter_rows, py::arg("matrix"), py::arg("statistic"),
               py::arg("low"), py::arg("high"), py::arg("inclusive") = false,
               py::arg("num_threads") = 1, py::arg("nan_mode") = 0, DTP_PROJECTION_ARGS);
    module.def("nonzero_extreme", &nonzero_extreme, py::arg("matrix"), py::arg("maximum"),
               DTP_PROJECTION_ARGS);
    module.def("copy_values", &copy_values, py::arg("matrix"),
               py::arg("zero_missing") = false, py::arg("num_threads") = 1,
               DTP_PROJECTION_ARGS);
    module.def("silhouette_score", &silhouette_score, py::arg("matrix"), py::arg("row"),
               py::arg("labels"), DTP_PROJECTION_ARGS);
    module.def("silhouette_scores", &silhouette_scores, py::arg("matrix"), py::arg("labels"),
               DTP_PROJECTION_ARGS,
               py::arg("zero_missing") = true, py::arg("num_threads") = 1);
    module.def("finite_values", &finite_values, py::arg("matrix"), DTP_PROJECTION_ARGS);
    module.def("geometric_inverse", &geometric_inverse);
    module.doc() = "Finite-aware float32/float64 reductions, sorting and permutation";
    module.attr("supports_multicore") = true;
    module.def("permute_rows_inplace", &permute_rows_inplace, py::arg("matrix"),
               py::arg("row_start"), py::arg("row_end"), py::arg("order"));
    module.def("reduce_axis", &reduce_axis, py::arg("matrix"),
               py::arg("axis") = 0,
               py::arg("op") = "mean", py::arg("num_threads") = 1,
               py::arg("trim_perc") = 0.05, py::arg("empty_sum_missing") = false,
               DTP_PROJECTION_ARGS);
    module.def("geom_mean_axis", &geom_mean_axis, py::arg("matrix"),
               py::arg("axis") = 0,
               py::arg("pseudocount") = -1.0, py::arg("num_threads") = 1,
               DTP_PROJECTION_ARGS);
    module.def("nan_quantiles", &nan_quantiles, py::arg("matrix"),
               py::arg("probs"),
               py::arg("max_exact") = 1048576, py::arg("tdigest_compression") = 0,
               py::arg("num_threads") = 1, py::arg("exact") = false,
               DTP_PROJECTION_ARGS);
    module.def("summarize_columns", &summarize_columns, py::arg("matrix"),
               py::arg("geom") = false,
               py::arg("pseudocount") = -1.0, py::arg("num_threads") = 1,
               DTP_PROJECTION_ARGS);
    module.def("bootstrap_ci", &bootstrap_ci, py::arg("matrix"),
               py::arg("average_type") = "mean", py::arg("pseudocount") = -1.0,
               py::arg("trim_perc") = 0.0, py::arg("resamples") = 300,
               py::arg("ci_level") = 0.95, py::arg("seed") = 0,
               py::arg("num_threads") = 1, py::arg("show_progress") = false,
               py::arg("include_center") = false, DTP_PROJECTION_ARGS);
}
