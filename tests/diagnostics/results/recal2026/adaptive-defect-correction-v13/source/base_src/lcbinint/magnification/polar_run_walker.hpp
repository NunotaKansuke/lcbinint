#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace lcbinint::magnification::detail {

// One maximal inside interval in a polar angular column.  `run_index` is an
// opaque caller-owned index; keeping it here lets the walker return segments
// without copying the radial integral or other per-run data.
struct PolarRunInterval {
    int iphi = 0;
    int left = 0;
    int right = -1;
    std::size_t run_index = 0;
};

// A one-to-one chain of polar runs as the angular coordinate advances.  The
// chain is cut at a split or merge, in the same way that Bennett's
// check_bktrack rescans a strip when a radial boundary turns around.
struct PolarOuterSegment {
    std::vector<std::size_t> run_indices;
    bool lower_boundary = false;
    bool upper_boundary = false;
    bool lower_junction = false;
    bool upper_junction = false;
    bool closed = false;
};

struct PolarOuterSegmentWalk {
    std::vector<PolarOuterSegment> segments;
    std::size_t adjacency_edges = 0;
    std::size_t junction_nodes = 0;
    bool complete = true;
};

inline bool polar_runs_overlap_or_touch(
    const PolarRunInterval& first,
    const PolarRunInterval& second)
{
    return static_cast<std::int64_t>(first.left) <=
            static_cast<std::int64_t>(second.right) + 1 &&
        static_cast<std::int64_t>(second.left) <=
            static_cast<std::int64_t>(first.right) + 1;
}

// Decompose one polar image component into disjoint chains.  The graph is
// directed from column iphi to (iphi + 1) mod phi_bins.  A clean finite image
// arc is therefore a path; an image that covers the complete angular ring is
// a cycle and is returned as a closed segment with no usable outer endpoint.
//
// The caller supplies only one connected component at a time.  Invalid
// intervals or column indices fail closed through `complete=false` rather
// than allowing a partial outer correction.
inline PolarOuterSegmentWalk walk_polar_outer_segments(
    const std::vector<PolarRunInterval>& runs,
    int phi_bins)
{
    PolarOuterSegmentWalk result;
    if (runs.empty()) {
        return result;
    }
    if (phi_bins < 2) {
        result.complete = false;
        return result;
    }

    for (const auto& run : runs) {
        if (run.iphi < 0 || run.iphi >= phi_bins ||
            run.left > run.right) {
            result.complete = false;
            return result;
        }
    }

    const std::size_t node_count = runs.size();
    const std::size_t no_neighbor =
        std::numeric_limits<std::size_t>::max();
    std::vector<std::vector<std::size_t>> by_phi(
        static_cast<std::size_t>(phi_bins));
    for (std::size_t node = 0; node < node_count; ++node) {
        by_phi[static_cast<std::size_t>(runs[node].iphi)].push_back(node);
    }
    for (auto& column : by_phi) {
        std::sort(
            column.begin(), column.end(),
            [&](std::size_t first, std::size_t second) {
                if (runs[first].left != runs[second].left) {
                    return runs[first].left < runs[second].left;
                }
                if (runs[first].right != runs[second].right) {
                    return runs[first].right < runs[second].right;
                }
                return first < second;
            });
    }

    // Only one-to-one edges are followed.  Degrees are capped at two: the
    // exact degree is not needed to distinguish a clean chain from a
    // split/merge junction, and this avoids large adjacency allocations for
    // the usual narrow image arcs.
    std::vector<std::size_t> single_predecessor(node_count, no_neighbor);
    std::vector<std::size_t> single_successor(node_count, no_neighbor);
    std::vector<unsigned char> predecessor_degree(node_count, 0);
    std::vector<unsigned char> successor_degree(node_count, 0);
    for (int iphi = 0; iphi < phi_bins; ++iphi) {
        const int next_iphi = iphi + 1 == phi_bins ? 0 : iphi + 1;
        const auto& first_column = by_phi[static_cast<std::size_t>(iphi)];
        const auto& second_column = by_phi[static_cast<std::size_t>(next_iphi)];
        for (const std::size_t first : first_column) {
            for (const std::size_t second : second_column) {
                if (!polar_runs_overlap_or_touch(runs[first], runs[second])) {
                    continue;
                }
                auto& successor_count = successor_degree[first];
                if (successor_count < 2U) {
                    ++successor_count;
                }
                if (successor_count == 1U) {
                    single_successor[first] = second;
                } else {
                    single_successor[first] = no_neighbor;
                }
                auto& predecessor_count = predecessor_degree[second];
                if (predecessor_count < 2U) {
                    ++predecessor_count;
                }
                if (predecessor_count == 1U) {
                    single_predecessor[second] = first;
                } else {
                    single_predecessor[second] = no_neighbor;
                }
                ++result.adjacency_edges;
            }
        }
    }

    for (std::size_t node = 0; node < node_count; ++node) {
        if (predecessor_degree[node] > 1U ||
            successor_degree[node] > 1U) {
            ++result.junction_nodes;
        }
    }

    const auto has_clean_predecessor = [&](std::size_t node) {
        return predecessor_degree[node] == 1U &&
            single_predecessor[node] != no_neighbor &&
            successor_degree[single_predecessor[node]] == 1U;
    };
    const auto has_clean_successor = [&](std::size_t node) {
        return successor_degree[node] == 1U &&
            single_successor[node] != no_neighbor &&
            predecessor_degree[single_successor[node]] == 1U;
    };
    const auto append_segment = [&](const std::vector<std::size_t>& chain,
                                    bool closed) {
        if (chain.empty()) {
            return;
        }
        PolarOuterSegment segment;
        segment.closed = closed;
        segment.run_indices.reserve(chain.size());
        for (const std::size_t node : chain) {
            segment.run_indices.push_back(runs[node].run_index);
        }
        const std::size_t first = chain.front();
        const std::size_t last = chain.back();
        segment.lower_boundary = !closed && predecessor_degree[first] == 0U;
        segment.upper_boundary = !closed && successor_degree[last] == 0U;
        segment.lower_junction = !closed &&
            predecessor_degree[first] != 0U && !has_clean_predecessor(first);
        segment.upper_junction = !closed &&
            successor_degree[last] != 0U && !has_clean_successor(last);
        result.segments.push_back(std::move(segment));
    };

    std::vector<bool> assigned(node_count, false);
    // Start at genuine boundaries and at the child side of a split/merge.
    // A chain never crosses a non-one-to-one edge, so each node is assigned
    // exactly once even when one junction is shared by multiple chains.
    for (std::size_t seed = 0; seed < node_count; ++seed) {
        if (assigned[seed] || has_clean_predecessor(seed)) {
            continue;
        }
        std::vector<std::size_t> chain;
        std::size_t current = seed;
        while (!assigned[current]) {
            chain.push_back(current);
            assigned[current] = true;
            if (!has_clean_successor(current)) {
                break;
            }
            current = single_successor[current];
        }
        append_segment(chain, false);
    }

    // A clean component that spans the full angular ring has no boundary
    // seed.  Retain it as a closed cycle for diagnostics, but never expose an
    // endpoint correction to the caller.
    for (std::size_t seed = 0; seed < node_count; ++seed) {
        if (assigned[seed]) {
            continue;
        }
        std::vector<std::size_t> chain;
        std::size_t current = seed;
        bool closed = false;
        while (!assigned[current]) {
            chain.push_back(current);
            assigned[current] = true;
            if (successor_degree[current] != 1U ||
                single_successor[current] == no_neighbor) {
                break;
            }
            current = single_successor[current];
            if (current == seed) {
                closed = true;
                break;
            }
        }
        append_segment(chain, closed);
    }

    std::size_t accounted = 0;
    for (const auto& segment : result.segments) {
        accounted += segment.run_indices.size();
    }
    result.complete = accounted == node_count;
    return result;
}

} // namespace lcbinint::magnification::detail
