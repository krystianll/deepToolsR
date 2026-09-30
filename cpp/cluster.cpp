#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "parallel.hpp"

// fastcluster's core has no Python dependency. Keep the upstream source intact.
#define fc_isnan(X) std::isnan(X)
#include "include/fastcluster/fastcluster.cpp"

namespace py = pybind11;

namespace {

struct Features {
    py::array_t<float, py::array::c_style> array;
    py::buffer_info buffer;
    const float *data;
    std::size_t rows, cols;

    explicit Features(py::array_t<float, py::array::c_style> input)
        : array(std::move(input)), buffer(array.request()),
          data(static_cast<const float *>(buffer.ptr)),
          rows(static_cast<std::size_t>(buffer.shape[0])),
          cols(static_cast<std::size_t>(buffer.shape[1])) {
        if (buffer.ndim != 2 || rows == 0 || cols == 0)
            throw py::value_error("features must be a nonempty two-dimensional float32 matrix");
    }
};

std::uint64_t splitmix64(std::uint64_t &state) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Xoshiro {
    std::uint64_t state[4];
    explicit Xoshiro(std::uint64_t seed) {
        for (auto &word : state) word = splitmix64(seed);
    }
    std::uint64_t next() {
        const auto rotate = [](std::uint64_t value, int bits) {
            return (value << bits) | (value >> (64 - bits));
        };
        const auto result = rotate(state[1] * 5, 7) * 9;
        const auto t = state[1] << 17;
        state[2] ^= state[0]; state[3] ^= state[1];
        state[1] ^= state[2]; state[0] ^= state[3];
        state[2] ^= t; state[3] = rotate(state[3], 45);
        return result;
    }
};

double distance(const float *point, const double *centre, std::size_t cols) {
    double sums[4] = {0, 0, 0, 0};
    std::size_t c = 0;
    for (; c + 3 < cols; c += 4) {
        for (std::size_t lane = 0; lane < 4; ++lane) {
            const double delta = static_cast<double>(point[c + lane]) -
                                 centre[c + lane];
            sums[lane] += delta * delta;
        }
    }
    for (; c < cols; ++c) {
        const double delta = static_cast<double>(point[c]) - centre[c];
        sums[c & 3] += delta * delta;
    }
    return std::sqrt((sums[0] + sums[1]) + (sums[2] + sums[3]));
}

struct Assignment {
    std::vector<std::int64_t> labels;
    std::vector<double> sums;
    std::vector<std::size_t> counts;
    double distortion = 0;
};

// fastcluster's vector Ward needs mutable centroids. Keep merged centroids in
// O(n*d) scratch, while the original float32 workspace stays intact for the
// caller's cluster-mean ordering and silhouette calculation.
struct WardVector {
    const Features &features;
    std::vector<float> centroids;
    std::vector<std::size_t> representative;
    std::vector<std::size_t> members;
    std::size_t next = 0;

    explicit WardVector(const Features &input)
        : features(input), centroids((input.rows - 1) * input.cols),
          representative(input.rows), members(input.rows, 1) {
        std::iota(representative.begin(), representative.end(), 0);
    }

    const float *point(t_index row) const {
        const auto index = representative[static_cast<std::size_t>(row)];
        return index < features.rows
            ? features.data + index * features.cols
            : centroids.data() + (index - features.rows) * features.cols;
    }

    template <bool check_nan>
    double sqeuclidean(t_index left, t_index right) const {
        const float *a = point(left);
        const float *b = point(right);
        double sum = 0;
        for (std::size_t c = 0; c < features.cols; ++c) {
            const double delta = static_cast<double>(a[c]) - b[c];
            sum += delta * delta;
        }
        if (check_nan && std::isnan(sum)) throw nan_error();
        return sum;
    }

    double ward_initial(t_index left, t_index right) const {
        return sqeuclidean<true>(left, right);
    }

    static double ward_initial_conversion(double distance) {
        return distance * 0.5;
    }

    double ward(t_index left, t_index right) const {
        const auto a = members[static_cast<std::size_t>(left)];
        const auto b = members[static_cast<std::size_t>(right)];
        return sqeuclidean<true>(left, right) *
            static_cast<double>(a) * static_cast<double>(b) /
            static_cast<double>(a + b);
    }

    void merge_inplace(t_index left, t_index right) {
        const float *a = point(left);
        const float *b = point(right);
        const auto count_a = members[static_cast<std::size_t>(left)];
        const auto count_b = members[static_cast<std::size_t>(right)];
        float *merged = centroids.data() + next * features.cols;
        for (std::size_t c = 0; c < features.cols; ++c)
            merged[c] = static_cast<float>(
                (static_cast<double>(a[c]) * count_a +
                 static_cast<double>(b[c]) * count_b) /
                static_cast<double>(count_a + count_b));
        representative[static_cast<std::size_t>(right)] = features.rows + next;
        members[static_cast<std::size_t>(right)] += count_a;
        ++next;
    }

    void merge_inplace_weighted(t_index left, t_index right) {
        const float *a = point(left);
        const float *b = point(right);
        float *merged = centroids.data() + next * features.cols;
        for (std::size_t c = 0; c < features.cols; ++c)
            merged[c] = static_cast<float>((static_cast<double>(a[c]) + b[c]) * 0.5);
        representative[static_cast<std::size_t>(right)] = features.rows + next++;
    }
};

Assignment assign(const Features &features, const std::vector<double> &centres,
                  std::size_t k, int threads) {
    constexpr std::size_t chunk_rows = 64;
    const auto chunks = (features.rows + chunk_rows - 1) / chunk_rows;
    std::vector<Assignment> partial(chunks);
    dtp::parallel_rows(chunks, threads, [&](std::size_t chunk) {
        auto &part = partial[chunk];
        const auto begin = chunk * chunk_rows;
        const auto end = std::min(features.rows, begin + chunk_rows);
        part.labels.resize(end - begin);
        part.sums.assign(k * features.cols, 0);
        part.counts.assign(k, 0);
        for (auto row = begin; row < end; ++row) {
            const float *point = features.data + row * features.cols;
            std::size_t best = 0;
            double nearest = distance(point, centres.data(), features.cols);
            for (std::size_t centre = 1; centre < k; ++centre) {
                const double candidate = distance(
                    point, centres.data() + centre * features.cols, features.cols);
                if (candidate < nearest) { nearest = candidate; best = centre; }
            }
            part.labels[row - begin] = static_cast<std::int64_t>(best);
            ++part.counts[best];
            for (std::size_t c = 0; c < features.cols; ++c)
                part.sums[best * features.cols + c] += point[c];
            part.distortion += nearest;
        }
    });
    Assignment result;
    result.labels.resize(features.rows);
    result.sums.assign(k * features.cols, 0);
    result.counts.assign(k, 0);
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        const auto &part = partial[chunk];
        std::copy(part.labels.begin(), part.labels.end(),
                  result.labels.begin() + chunk * chunk_rows);
        for (std::size_t i = 0; i < k * features.cols; ++i)
            result.sums[i] += part.sums[i];
        for (std::size_t i = 0; i < k; ++i) result.counts[i] += part.counts[i];
        result.distortion += part.distortion;
    }
    result.distortion /= static_cast<double>(features.rows);
    return result;
}

py::tuple kmeans(const Features &features, std::size_t k, std::uint64_t seed,
                 int restarts, double thresh, int threads) {
    if (k < 1 || k > features.rows || restarts < 1 || thresh < 0)
        throw py::value_error("invalid k-means parameters");
    Xoshiro rng(seed);
    std::vector<double> best_centres;
    double best_distortion = std::numeric_limits<double>::infinity();
    std::size_t best_k = 0;
    py::gil_scoped_release release;
    for (int restart = 0; restart < restarts; ++restart) {
        std::vector<std::size_t> choices(features.rows);
        std::iota(choices.begin(), choices.end(), 0);
        std::vector<double> centres(k * features.cols);
        for (std::size_t i = 0; i < k; ++i) {
            const std::size_t pick = i + rng.next() % (features.rows - i);
            std::swap(choices[i], choices[pick]);
            for (std::size_t c = 0; c < features.cols; ++c)
                centres[i * features.cols + c] =
                    features.data[choices[i] * features.cols + c];
        }
        std::size_t active = k;
        double previous = std::numeric_limits<double>::infinity();
        for (int iteration = 0; iteration < 300; ++iteration) {
            const auto assigned = assign(features, centres, active, threads);
            std::vector<double> next;
            next.reserve(centres.size());
            for (std::size_t j = 0; j < active; ++j) {
                if (assigned.counts[j] == 0) continue;
                for (std::size_t c = 0; c < features.cols; ++c)
                    next.push_back(assigned.sums[j * features.cols + c] /
                                   static_cast<double>(assigned.counts[j]));
            }
            active = next.size() / features.cols;
            centres.swap(next);
            if (std::abs(previous - assigned.distortion) <= thresh) break;
            previous = assigned.distortion;
        }
        const double distortion = assign(features, centres, active, threads).distortion;
        if (distortion < best_distortion) {
            best_distortion = distortion;
            best_centres = std::move(centres);
            best_k = active;
        }
    }
    const auto final = assign(features, best_centres, best_k, threads);
    py::gil_scoped_acquire acquire;
    py::array_t<std::int64_t> labels(features.rows);
    std::copy(final.labels.begin(), final.labels.end(), labels.mutable_data());
    py::array_t<double> centroids({best_k, features.cols});
    std::copy(best_centres.begin(), best_centres.end(), centroids.mutable_data());
    return py::make_tuple(labels, centroids, best_distortion);
}

py::array_t<double> ward_linkage(const Features &features, int threads,
                                 std::uint64_t distance_budget_bytes) {
    const auto n = features.rows;
    if (n < 2) throw py::value_error("Ward linkage requires at least two rows");
    if (n > static_cast<std::size_t>(MAX_INDEX / 4))
        throw py::value_error("too many rows for Ward linkage");
    const auto distance_bytes = static_cast<std::uint64_t>(n) * (n - 1) / 2 * 8;
    const bool fallback = distance_bytes > distance_budget_bytes;
    cluster_result result(static_cast<t_index>(n - 1));
    py::gil_scoped_release release;
    if (fallback) {
        WardVector metric(features);
        generic_linkage_vector<METHOD_VECTOR_WARD>(static_cast<t_index>(n),
                                                   metric, result);
        result.sqrtdouble(0);
    } else {
        std::vector<double> condensed(n * (n - 1) / 2);
        dtp::parallel_rows(n - 1, threads, [&](std::size_t i) {
            const float *left = features.data + i * features.cols;
            const std::size_t offset = i * (2 * n - i - 1) / 2;
            for (std::size_t j = i + 1; j < n; ++j) {
                const float *right = features.data + j * features.cols;
                // Four fixed lanes give one schedule at every thread count.
                double sums[4] = {0, 0, 0, 0};
                std::size_t c = 0;
                for (; c + 3 < features.cols; c += 4) {
                    for (std::size_t lane = 0; lane < 4; ++lane) {
                        const double delta = static_cast<double>(left[c + lane]) -
                                             right[c + lane];
                        sums[lane] += delta * delta;
                    }
                }
                for (; c < features.cols; ++c) {
                    const double delta = static_cast<double>(left[c]) - right[c];
                    sums[c & 3] += delta * delta;
                }
                condensed[offset + j - i - 1] =
                    (sums[0] + sums[1]) + (sums[2] + sums[3]);
            }
        });
        auto_array_ptr<t_index> members(static_cast<t_index>(n), 1);
        NN_chain_core<METHOD_METR_WARD, t_index>(static_cast<t_index>(n),
                                                  condensed.data(), members, result);
        result.sqrt();
        std::stable_sort(result[0], result[static_cast<t_index>(n - 1)]);
    }
    union_find nodes(static_cast<t_index>(n));
    py::gil_scoped_acquire acquire;
    py::array_t<double> output({n - 1, std::size_t(4)});
    auto z = output.mutable_unchecked<2>();
    for (std::size_t row = 0; row < n - 1; ++row) {
        const auto &merge = result[static_cast<t_index>(row)][0];
        const auto left = fallback ? merge.node1 : nodes.Find(merge.node1);
        const auto right = fallback ? merge.node2 : nodes.Find(merge.node2);
        if (!fallback) nodes.Union(left, right);
        z(row, 0) = std::min(left, right);
        z(row, 1) = std::max(left, right);
        z(row, 2) = merge.dist;
        z(row, 3) = (left < static_cast<t_index>(n) ? 1 : z(left - n, 3)) +
                    (right < static_cast<t_index>(n) ? 1 : z(right - n, 3));
    }
    return output;
}

py::array_t<std::int64_t> cut_maxclust(
    py::array_t<double, py::array::c_style> linkage, std::size_t k) {
    const auto info = linkage.request();
    if (info.ndim != 2 || info.shape[1] != 4 || info.shape[0] < 1)
        throw py::value_error("linkage must have shape (n - 1, 4)");
    const auto n = static_cast<std::size_t>(info.shape[0]) + 1;
    if (k < 1) throw py::value_error("cluster count must be positive");
    const auto z = linkage.unchecked<2>();
    py::array_t<std::int64_t> output(n);
    auto *labels = output.mutable_data();
    if (k >= n) {
        for (std::size_t i = 0; i < n; ++i) labels[i] = i + 1;
        return output;
    }
    std::vector<double> max_height(n - 1);
    for (std::size_t i = 0; i < n - 1; ++i) {
        max_height[i] = z(i, 2);
        for (int child = 0; child < 2; ++child) {
            const auto id = static_cast<std::size_t>(z(i, child));
            if (id >= n) max_height[i] = std::max(max_height[i], max_height[id - n]);
        }
    }
    // Match fcluster's threshold search over monotonic subtree heights.
    std::ptrdiff_t lower = -1;
    std::ptrdiff_t upper = static_cast<std::ptrdiff_t>(n - 1);
    while (upper - lower > 1) {
        const auto middle = (lower + upper) >> 1;
        const double threshold = max_height[static_cast<std::size_t>(middle)];
        std::vector<std::size_t> pending = {2 * n - 2};
        std::size_t clusters = 0;
        while (!pending.empty() && clusters <= k) {
            const auto id = pending.back();
            pending.pop_back();
            if (id < n || max_height[id - n] <= threshold) {
                ++clusters;
            } else {
                pending.push_back(static_cast<std::size_t>(z(id - n, 1)));
                pending.push_back(static_cast<std::size_t>(z(id - n, 0)));
            }
        }
        if (clusters > k) lower = middle;
        else upper = middle;
    }
    const double cutoff = max_height[static_cast<std::size_t>(upper)];
    // Match SciPy's depth-first cluster numbering, including when a child
    // subtree is visited before its parent assigns leaf labels.
    std::vector<unsigned char> visited(2 * n - 1, 0);
    std::vector<std::size_t> stack(n);
    std::size_t depth = 0;
    stack[0] = 2 * n - 2;
    std::int64_t next_label = 0;
    std::size_t leader = 2 * n;
    while (true) {
        const auto root = stack[depth] - n;
        const auto left = static_cast<std::size_t>(z(root, 0));
        const auto right = static_cast<std::size_t>(z(root, 1));
        if (leader == 2 * n && max_height[root] <= cutoff) {
            leader = root;
            ++next_label;
        }
        if (left >= n && !visited[left]) {
            visited[left] = 1;
            stack[++depth] = left;
            continue;
        }
        if (right >= n && !visited[right]) {
            visited[right] = 1;
            stack[++depth] = right;
            continue;
        }
        if (left < n) {
            if (leader == 2 * n) ++next_label;
            labels[left] = next_label;
        }
        if (right < n) {
            if (leader == 2 * n) ++next_label;
            labels[right] = next_label;
        }
        if (leader == root) leader = 2 * n;
        if (depth == 0) break;
        --depth;
    }
    return output;
}

}  // namespace

PYBIND11_MODULE(_cluster, module) {
    module.def("kmeans", [](py::array_t<float, py::array::c_style> features,
                            std::size_t k, std::uint64_t seed, int restarts,
                            double thresh, int threads) {
        Features input(std::move(features));
        return kmeans(input, k, seed, restarts, thresh, threads);
    }, py::arg("features"), py::arg("k"), py::kw_only(), py::arg("seed"),
       py::arg("restarts") = 20, py::arg("thresh") = 1e-5,
       py::arg("threads"));
    module.def("ward_linkage", [](py::array_t<float, py::array::c_style> features,
                                  int threads, std::uint64_t distance_budget_bytes) {
        Features input(std::move(features));
        return ward_linkage(input, threads, distance_budget_bytes);
    }, py::arg("features"), py::kw_only(), py::arg("threads"),
       py::arg("distance_budget_bytes") = std::uint64_t(2) << 30);
    module.def("cut_maxclust", &cut_maxclust, py::arg("linkage"), py::arg("k"));
}
