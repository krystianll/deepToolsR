#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtp {
inline double missing_value() { return std::numeric_limits<double>::quiet_NaN(); }

// Neumaier compensation retains small terms through cancellation. Inputs to
// this accumulator must be bounded so their sum cannot overflow; callers use
// a scaled pass when reducing values near the float64 range limit.
struct CompensatedSum {
    double sum = 0, correction = 0;
    void add(double x) {
        const double next = sum + x;
        correction += std::abs(sum) >= std::abs(x)
            ? (sum - next) + x : (x - next) + sum;
        sum = next;
    }
    bool add_checked(double x) {
        CompensatedSum next = *this;
        next.add(x);
        if (!std::isfinite(next.sum) || !std::isfinite(next.correction) ||
            !std::isfinite(next.value()))
            return false;
        *this = next;
        return true;
    }
    double value() const { return sum + correction; }
};

// Relative shifted logs avoid cancellation when the pseudocount dominates
// the signal. The matching inverse uses expm1, with a log-domain fallback
// when an intermediate exponential would overflow despite a small pc.
inline double geometric_log(double value, double sign, double pc) {
    if (!std::isfinite(value)) return missing_value();
    const double x = sign * value;
    if (pc == 0) return x > 0 ? std::log(x) : missing_value();
    if (x <= -pc) return missing_value();
    const double ratio = x / pc;
    return std::isfinite(ratio) ? std::log1p(ratio) : std::log(x) - std::log(pc);
}
inline double geometric_inverse(double value, double sign, double pc) {
    if (std::isnan(value)) return missing_value();
    if (sign == 0) return 0;
    const double result = pc == 0 ? std::exp(value)
        : value < 700 ? pc * std::expm1(value) : std::exp(value + std::log(pc)) - pc;
    if (!std::isfinite(result)) throw std::overflow_error("geometric result exceeds float64 range");
    return sign * result;
}

// Stable IDs are also used by the streaming Python adapter. Each caller
// restricts this vocabulary to its supported subset (e.g. no per-bin std).
enum class Statistic {
    PerBin = 0, Mean = 1, Median = 2, Sum = 3, Min = 4, Max = 5,
    Std = 6, TrimMean = 7, Count = 8, Nonzero = 9, GeomMean = 10
};
inline Statistic parse_statistic(const std::string &op) {
    if (op == "perBin") return Statistic::PerBin;
    if (op == "mean") return Statistic::Mean;
    if (op == "median") return Statistic::Median;
    if (op == "sum") return Statistic::Sum;
    if (op == "min") return Statistic::Min;
    if (op == "max") return Statistic::Max;
    if (op == "std") return Statistic::Std;
    if (op == "trim_mean") return Statistic::TrimMean;
    if (op == "count") return Statistic::Count;
    if (op == "nonzero") return Statistic::Nonzero;
    if (op == "geom_mean") return Statistic::GeomMean;
    throw std::invalid_argument("unsupported statistic: " + op);
}
// Capabilities are exposed to Python so advertised options and native dispatch
// cannot drift. Retain deliberately different subsets for bins and profiles.
inline std::vector<std::string> statistic_choices(const std::string &use) {
    if (use == "bin") return {"mean", "median", "min", "max", "sum"};
    if (use == "filter") return {"perBin", "mean", "median", "sum", "min", "max"};
    if (use == "profile") return {"mean", "geom_mean", "trim_mean", "median", "min", "max", "std", "sum"};
    throw std::invalid_argument("unknown statistic use: " + use);
}
inline Statistic parse_statistic(const std::string &op, const std::string &use) {
    const auto choices = statistic_choices(use);
    if (std::find(choices.begin(), choices.end(), op) == choices.end())
        throw std::invalid_argument("unsupported " + use + " statistic: " + op);
    return parse_statistic(op);
}
inline void validate_statistic(const std::string &op) {
    const auto stat = parse_statistic(op);
    if (stat == Statistic::PerBin || stat == Statistic::GeomMean)
        throw std::invalid_argument("unsupported statistic: " + op);
}

// Trim floor(n*p) from each finite tail, as in scipy.stats.trim_mean.
inline std::size_t trim_count(std::size_t n, double proportion) {
    if (!std::isfinite(proportion) || proportion < 0 || proportion >= 0.5)
        throw std::invalid_argument("trim_perc must be at least 0 and less than 0.5");
    return static_cast<std::size_t>(std::floor(n * proportion));
}

struct FiniteSummary {
    std::size_t count = 0, nonzero = 0;
    double low = std::numeric_limits<double>::infinity(), high = -low;
    void add(double x) {
        ++count;
        nonzero += x != 0;
        low = std::min(low, x); high = std::max(high, x);
    }
    double magnitude() const { return std::max(std::abs(low), std::abs(high)); }
    double sum_scale() const {
        const double m = magnitude();
        return count && m > std::numeric_limits<double>::max() / count / 2 ? m : 1;
    }
};

template<class Get>
FiniteSummary scan_finite(std::size_t size, Get get, std::vector<double> *scratch = nullptr) {
    FiniteSummary result;
    if (scratch) scratch->clear();
    for (std::size_t i = 0; i < size; ++i) {
        const double x = get(i);
        if (!std::isfinite(x)) continue;
        result.add(x);
        if (scratch) scratch->push_back(x);
    }
    return result;
}

// Only finite values; the caller owns this scratch and permits reordering.
inline double order_statistic(std::vector<double> &scratch, Statistic op, double trim = .05) {
    const auto count = scratch.size();
    if (op == Statistic::TrimMean) trim_count(0, trim);
    if (!count) return missing_value();
    if (op == Statistic::Median) {
        const auto middle = count / 2;
        std::nth_element(scratch.begin(), scratch.begin() + middle, scratch.end());
        const double upper = scratch[middle];
        if (count & 1) return upper;
        const double lower = *std::max_element(scratch.begin(), scratch.begin() + middle);
        return std::signbit(lower) == std::signbit(upper)
            ? lower + (upper - lower) * .5 : lower * .5 + upper * .5;
    }
    if (op != Statistic::TrimMean) throw std::invalid_argument("not an order statistic");
    const auto tail = trim_count(count, trim);
    std::sort(scratch.begin(), scratch.end());
    const auto kept = count - 2 * tail;
    const double magnitude = std::max(std::abs(scratch[tail]), std::abs(scratch[count - tail - 1]));
    const double scale = magnitude > std::numeric_limits<double>::max() / kept / 2 ? magnitude : 1;
    if (scale != 1) {
        CompensatedSum unscaled;
        bool safe = true;
        for (std::size_t i = tail; i < count - tail && safe; ++i)
            safe = unscaled.add_checked(scratch[i]);
        if (safe) return unscaled.value() / kept;
    }
    CompensatedSum total;
    for (std::size_t i = tail; i < count - tail; ++i) total.add(scratch[i] / scale);
    return (total.value() / kept) * scale;
}

struct Moments {
    double mean = missing_value(), stddev = missing_value();
    FiniteSummary finite;
};

// Three scans in total, without copying the input. The second scan computes
// both the compensated mean and the center of normalized deviations. Keeping
// those deviations relative to an observed value retains adjacent-float spread.
template<class Get>
Moments moments_from_summary(std::size_t size, Get get, FiniteSummary finite,
                             bool need_mean = true) {
    Moments result; result.finite = finite;
    if (!finite.count) return result;
    const double magnitude = finite.magnitude();
    const double scale = magnitude > 0 ? magnitude : 1;
    const double mean_scale = finite.sum_scale();
    auto deviation = [&](double x) {
        return scale == 1 ? x - finite.low : std::signbit(x) == std::signbit(finite.low)
            ? (x - finite.low) / scale : x / scale - finite.low / scale;
    };
    CompensatedSum differences, total;
    for (std::size_t i = 0; i < size; ++i) {
        const double x = get(i);
        if (!std::isfinite(x)) continue;
        differences.add(deviation(x));
        if (need_mean) total.add(x / mean_scale);
    }
    if (need_mean) result.mean = (total.value() / finite.count) * mean_scale;
    const double center = differences.value() / finite.count;
    CompensatedSum squares;
    for (std::size_t i = 0; i < size; ++i) {
        const double x = get(i);
        if (!std::isfinite(x)) continue;
        const double d = deviation(x) - center;
        squares.add(d * d);
    }
    result.stddev = std::sqrt(squares.value() / finite.count) * scale;
    return result;
}
template<class Get>
Moments finite_moments(std::size_t size, Get get) {
    return moments_from_summary(size, get, scan_finite(size, get));
}

struct Reduction { double value; FiniteSummary finite; };
inline double finish_sum(double total, double scale, std::size_t count, Statistic op) {
    if (op == Statistic::Mean) return (total / count) * scale;
    const double result = total * scale;
    if (!std::isfinite(result)) throw std::overflow_error("sum exceeds float64 range");
    return result;
}

template<class Get>
Reduction finite_reduction(std::size_t size, Get get, Statistic op,
                           double trim = .05, std::vector<double> *work = nullptr,
                           bool float32_values = false) {
    if (op == Statistic::GeomMean || op == Statistic::PerBin)
        throw std::invalid_argument("statistic requires a domain-specific reduction");
    if (op == Statistic::TrimMean) trim_count(0, trim);
    std::vector<double> local;
    auto &scratch = work ? *work : local;
    const bool gather = op == Statistic::Median || op == Statistic::TrimMean;
    FiniteSummary finite;
    CompensatedSum total;
    // A float32 sum cannot overflow double for any addressable input size.
    // Streaming text uses this path after narrowing to the matrix storage type,
    // so it never needs to parse a row again or materialize it for mean/sum.
    const bool one_pass_sum = float32_values && (op == Statistic::Mean || op == Statistic::Sum);
    if (one_pass_sum) {
        for (std::size_t i = 0; i < size; ++i) {
            const double x = get(i);
            if (!std::isfinite(x)) continue;
            finite.add(x); total.add(x);
        }
    } else finite = scan_finite(size, get, gather ? &scratch : nullptr);
    const auto count = finite.count;
    if (op == Statistic::Count) return {static_cast<double>(count), finite};
    if (op == Statistic::Nonzero) return {static_cast<double>(finite.nonzero), finite};
    if (!count) return {op == Statistic::Sum ? 0 : missing_value(), finite};
    if (op == Statistic::Min) return {finite.low, finite};
    if (op == Statistic::Max) return {finite.high, finite};
    if (gather) return {order_statistic(scratch, op, trim), finite};
    if (op == Statistic::Std)
        return {moments_from_summary(size, get, finite, false).stddev, finite};
    const double scale = one_pass_sum ? 1 : finite.sum_scale();
    if (!one_pass_sum) {
        if (scale != 1) {
            CompensatedSum unscaled;
            bool safe = true;
            for (std::size_t i = 0; i < size && safe; ++i) {
                const double x = get(i);
                if (std::isfinite(x)) safe = unscaled.add_checked(x);
            }
            if (safe) return {finish_sum(unscaled.value(), 1, count, op), finite};
        }
        for (std::size_t i = 0; i < size; ++i) {
            const double x = get(i);
            if (std::isfinite(x)) total.add(x / scale);
        }
    }
    return {finish_sum(total.value(), scale, count, op), finite};
}
template<class Get>
double reduce_finite(std::size_t size, Get get, Statistic op, double trim = .05) {
    return finite_reduction(size, get, op, trim).value;
}
template<class Get>
double reduce_finite(std::size_t size, Get get, const std::string &op, double trim = .05) {
    validate_statistic(op);
    return reduce_finite(size, get, parse_statistic(op), trim);
}

// Identical threshold and missing-sample rules for resident and text filters.
enum class NanMode { Keep = 0, AnyBin = 1, AnySample = 2, AllBins = 3 };
struct SampleEval {
    bool value_fail = false;
    std::size_t finite = 0, total = 0;
    bool fails(NanMode mode) const {
        return value_fail || (mode == NanMode::AnyBin && finite < total) ||
            (mode == NanMode::AnySample && finite == 0);
    }
};
template<class Get>
SampleEval filter_sample(std::size_t size, Get get, Statistic op, double low,
                         double high, bool inclusive, bool float32_values,
                         std::vector<double> *scratch = nullptr) {
    auto reduction = finite_reduction(size, get,
        op == Statistic::PerBin ? Statistic::Min : op, .05, scratch, float32_values);
    SampleEval result{false, reduction.finite.count, size};
    if (result.finite) {
        const double lo = op == Statistic::PerBin ? reduction.finite.low : reduction.value;
        const double hi = op == Statistic::PerBin ? reduction.finite.high : reduction.value;
        result.value_fail = inclusive ? lo <= low || hi >= high : lo < low || hi > high;
    }
    return result;
}

struct GeometricDomain {
    std::size_t count = 0, zeros = 0;
    double sign = 0, minimum = std::numeric_limits<double>::infinity();
    double low = std::numeric_limits<double>::infinity(), high = -low;
};

// A signed geometric mean reflects a nonnegative sample onto its common
// sign. Opposite signs within one reduction are not a valid domain.
template<class Get>
GeometricDomain geometric_domain(std::size_t size, Get get) {
    GeometricDomain domain;
    for (std::size_t i = 0; i < size; ++i) {
        const double x = get(i);
        if (!std::isfinite(x)) continue;
        ++domain.count;
        domain.low = std::min(domain.low, x);
        domain.high = std::max(domain.high, x);
        if (x == 0) { ++domain.zeros; continue; }
        const double sign = x > 0 ? 1 : -1;
        if (domain.sign && domain.sign != sign)
            throw std::invalid_argument("geometric mean requires a common sign within each bin or row; mixed positive and negative values are invalid");
        domain.sign = sign;
        domain.minimum = std::min(domain.minimum, std::abs(x));
    }
    return domain;
}

struct GeometricContext {
    double pseudocount = 0;
    std::vector<GeometricDomain> domains;
};
template<class Get>
GeometricContext prepare_geometric(std::size_t lines, std::size_t size, Get get, double pc) {
    if (!std::isfinite(pc)) throw std::invalid_argument("pseudocount must be finite");
    GeometricContext result;
    result.domains.reserve(lines);
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < lines; ++k) {
        const auto domain = geometric_domain(size, [&](std::size_t i) { return get(k, i); });
        minimum = std::min(minimum, domain.minimum);
        result.domains.push_back(domain);
    }
    // Auto ignores zeros; only all-zero/all-missing blocks resolve to zero.
    const double half = minimum / 2;
    result.pseudocount = pc >= 0 ? pc : !std::isfinite(minimum) ? 0 : half > 0 ? half : minimum;
    return result;
}

template<class Get>
double geometric_mean(std::size_t size, Get get, double pc, const GeometricDomain &domain) {
    if (!domain.count) return missing_value();
    if (domain.low == domain.high) return domain.low;
    if (!domain.sign || (pc == 0 && domain.zeros)) return 0;
    const double mean = reduce_finite(size, [&](std::size_t i) {
        return geometric_log(get(i), domain.sign, pc);
    }, Statistic::Mean);
    return geometric_inverse(mean, domain.sign, pc);
}
template<class Get>
double geometric_mean(std::size_t size, Get get, double pc) {
    return geometric_mean(size, get, pc, geometric_domain(size, get));
}
} // namespace dtp
