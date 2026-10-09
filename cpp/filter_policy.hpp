#pragma once

#include "numeric.hpp"

namespace dtp {

enum class FilterAction { RemoveRegion, MaskSample };

inline FilterAction parse_filter_action(const std::string &name) {
    if (name == "removeRegion") return FilterAction::RemoveRegion;
    if (name == "maskSample") return FilterAction::MaskSample;
    throw std::invalid_argument("unknown filter action: " + name);
}

// Validated once per operation. Input adapters supply sample evaluations and
// apply masks; the selected-sample and whole-row decisions live only here.
class FilterPolicy {
    const std::vector<std::size_t> &samples_;
    NanMode nan_mode_;
    FilterAction action_;
public:
    FilterPolicy(const std::vector<std::size_t> &bounds,
                 const std::vector<std::size_t> &samples, Statistic statistic,
                 int nan_mode, FilterAction action, double low, double high)
        : samples_(samples), nan_mode_(static_cast<NanMode>(nan_mode)), action_(action) {
        if (bounds.size() < 2 || bounds.front() != 0)
            throw std::runtime_error("sample boundaries must contain at least [0, end]");
        for (std::size_t i = 1; i < bounds.size(); ++i)
            if (bounds[i] <= bounds[i - 1])
                throw std::runtime_error("sample boundaries are not valid monotonic boundaries");
        if (statistic < Statistic::PerBin || statistic > Statistic::Max)
            throw std::runtime_error("unknown filter statistic");
        if (nan_mode < static_cast<int>(NanMode::Keep) || nan_mode > static_cast<int>(NanMode::AllBins))
            throw std::runtime_error("unknown missing-value filter mode");
        std::vector<bool> seen(bounds.size() - 1, false);
        for (const auto sample : samples) {
            if (sample >= seen.size() || seen[sample])
                throw std::runtime_error("filter samples must be distinct valid indices");
            seen[sample] = true;
        }
        if (std::isnan(low) || low == std::numeric_limits<double>::infinity())
            throw std::runtime_error("minimum must be finite or -inf");
        if (std::isnan(high) || high == -std::numeric_limits<double>::infinity())
            throw std::runtime_error("maximum must be finite or +inf");
        if (low > high) throw std::runtime_error("minimum cannot exceed maximum");
    }

    // Returns whether the row survives. Remove mode may stop at the first
    // failure; mask mode evaluates every selected sample and always keeps rows.
    // No selected samples means no filtering, including under AllBins.
    template<class Evaluate, class Mask>
    bool keep_row(Evaluate evaluate, Mask mask) const {
        bool all_missing = !samples_.empty();
        for (const auto sample : samples_) {
            const SampleEval result = evaluate(sample);
            all_missing &= result.finite == 0;
            if (result.fails(nan_mode_)) {
                if (action_ == FilterAction::RemoveRegion) return false;
                mask(sample);
            }
        }
        if (nan_mode_ == NanMode::AllBins && all_missing) {
            if (action_ == FilterAction::RemoveRegion) return false;
            for (const auto sample : samples_) mask(sample);
        }
        return true;
    }
};
} // namespace dtp
