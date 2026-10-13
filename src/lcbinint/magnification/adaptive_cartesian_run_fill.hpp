#pragma once
#include "adaptive_ld_saddle_error.hpp"
#include "adaptive_retained_row_interpolation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <functional>
#include <limits>
#include <memory>
#include "adaptive_outer_endpoint_model.hpp"
#include "adaptive_same_grid_flux_error.hpp"
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace lcbinint::magnification::adaptive_detail {

struct CartesianLatticeSeed {
    std::int64_t ix = 0;
    std::int64_t iy = 0;
};

struct CartesianRun {
    std::int64_t iy = 0;
    std::int64_t lo = 0;
    std::int64_t hi = -1;
};
struct CartesianCoverageDomain {
    std::int64_t extent=0;
    std::size_t maximum_boxes=65536;
    std::function<bool(std::int64_t,std::int64_t,std::int64_t,std::int64_t)> outside;
    // A certified fence may restrict the global domain. Component extents
    // are proposals only; the callback must prove that no physical image
    // can leave their union. An empty return retains the global domain.
    std::function<std::vector<std::array<std::int64_t,4>>(const std::vector<std::array<std::int64_t,4>>&)> fenced_regions;
};

struct CartesianRunBoundaryDiagnostic {
    double left_fraction=std::numeric_limits<double>::quiet_NaN();
    double right_fraction=std::numeric_limits<double>::quiet_NaN();
    double left_bracket_lower=0,left_bracket_upper=1;
    double right_bracket_lower=0,right_bracket_upper=1;
    int rounded_zero_edges=0;
    double signed_model_flux_residual=0.0;
    double absolute_model_flux_residual=0.0;
    int unresolved_edges=0;
};

struct CartesianRunBoundaryTraceEvent {
    CartesianRun run;
    CartesianRunBoundaryDiagnostic diagnostic;
    double cell_spacing=0;
    int fill_level=0;
    std::size_t run_index=0;
    double midpoint_area=0;
    double outer_weight=1;
    int outer_pair=-1;
};

enum class CartesianRunFillTraceEventKind {
    run_discovered,
    frontier_popped,
    components_merged,
};

struct CartesianRunFillTraceEvent {
    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

    CartesianRunFillTraceEventKind kind =
        CartesianRunFillTraceEventKind::run_discovered;
    CartesianRun run;
    std::size_t run_index = npos;
    // For a discovered run this is its parent frontier run.  For a merge it
    // is the other run whose component was joined.
    std::size_t related_run_index = npos;
    std::size_t component = npos;
    int fill_level = 0;
    double cell_spacing = 0.0;
};

// Optional sink for the exact run-fill order.  The ordinary fill path uses the
// zero-state no-op sink below, so it pays neither for event storage nor for
// trace event work. A trace-enabled caller can retain the events.
struct CartesianRunFillTrace {
    // A trace is diagnostic output, not part of the numerical result. Keep a
    // hard ceiling so an accidentally requested high-resolution trace cannot
    // turn an otherwise bounded fill into an unbounded event log.
    static constexpr std::size_t default_maximum_events = 1U << 18;

    std::vector<CartesianRunFillTraceEvent> events;
    std::vector<CartesianRunBoundaryTraceEvent> boundary_diagnostics;
    int fill_level = 0;
    double cell_spacing = 0.0;
    std::size_t maximum_events = default_maximum_events;
    bool truncated = false;
    bool coverage_requested=false,coverage_certified=false;
    std::int64_t coverage_boxes=0,coverage_unresolved=0,coverage_components=0;

    template <typename... Args>
    void append(Args&&... args)
    {
        if (events.size() >= maximum_events) {
            truncated = true;
            return;
        }
        events.emplace_back(std::forward<Args>(args)...);
    }

    void run_discovered(
        std::size_t run_index,
        const CartesianRun& run,
        std::size_t parent_run_index,
        std::size_t component)
    {
        append(CartesianRunFillTraceEvent{
            CartesianRunFillTraceEventKind::run_discovered,
            run,
            run_index,
            parent_run_index,
            component,
            fill_level,
            cell_spacing,
        });
    }

    void frontier_popped(
        std::size_t run_index,
        const CartesianRun& run,
        std::size_t component)
    {
        append(CartesianRunFillTraceEvent{
            CartesianRunFillTraceEventKind::frontier_popped,
            run,
            run_index,
            CartesianRunFillTraceEvent::npos,
            component,
            fill_level,
            cell_spacing,
        });
    }

    void components_merged(
        std::size_t run_index,
        std::size_t related_run_index,
        std::size_t component)
    {
        append(CartesianRunFillTraceEvent{
            CartesianRunFillTraceEventKind::components_merged,
            {},
            run_index,
            related_run_index,
            component,
            fill_level,
            cell_spacing,
        });
    }
};

struct CartesianRunFillNoTrace {
    void run_discovered(
        std::size_t,
        const CartesianRun&,
        std::size_t,
        std::size_t) const noexcept {}

    void frontier_popped(
        std::size_t,
        const CartesianRun&,
        std::size_t) const noexcept {}

    void components_merged(
        std::size_t,
        std::size_t,
        std::size_t) const noexcept {}
};

inline CartesianRunFillNoTrace& default_cartesian_run_fill_trace() noexcept
{
    static CartesianRunFillNoTrace sink;
    return sink;
}


struct CartesianBoundaryContribution {
    double area = 0.0;
    int edges = 0;
    bool valid = true;
    std::optional<CartesianRunBoundaryDiagnostic> diagnostic;
    bool error_recorded=false;
    double error_signed=0,error_absolute=0,error_remainder=0,error_limb_density=0;
    std::array<double,2> error_fractions{0,0};
    int error_unresolved=0;
};

struct CartesianRetainedRun {CartesianRun run;double area=0;std::array<double,5> errors{};std::array<double,2> fractions{};std::optional<RetainedRowCore> core;};

struct CartesianRunFillLimits {
    std::int64_t maximum_evaluations =
        std::numeric_limits<std::int64_t>::max();
    std::size_t maximum_runs = 1U << 20;
    std::size_t expected_rows = 0;
};

struct CartesianRunFillCounters {
    std::int64_t mapped_cell_evaluations = 0;
    std::int64_t frontier_intervals_popped = 0;
    std::int64_t maximal_runs_discovered = 0;
    std::int64_t row_intervals_stored = 0;
    // Rows whose intervals contain two or more runs from the same connected
    // component.  Runs belonging to separate image components on one y row
    // are not a within-component topology event.
    std::int64_t rows_with_multiple_runs = 0;
    std::size_t maximum_runs_in_row = 0;
    std::int64_t seeds_offered = 0;
    std::int64_t unique_seed_cells = 0;
    std::int64_t duplicate_seeds_avoided = 0;
    std::int64_t coverage_probe_cells = 0;
    std::int64_t coverage_components_added = 0;
    std::int64_t coverage_boxes=0,coverage_unresolved=0;
    std::int64_t provisional_components = 0;
    std::int64_t merged_components = 0;
};

enum class CartesianRunFillStatus {
    ok,
    evaluation_budget_exhausted,
    run_budget_exhausted,
    coordinate_overflow,
    invalid_contribution,
};

struct CartesianRunRecord {
    CartesianRun run;
    std::size_t component = 0;
};

// Zero row errors have a logical entry without a forty-byte payload.
// Nonzero records use the ordinary dense layout; signed zeros are preserved.
class CartesianRunErrorStorage {
    using Value = std::array<double, 5>;
    std::vector<Value> values;
    std::size_t count = 0;
    bool compress = std::getenv("LCBININT_DIAGNOSTIC_DISABLE_ZERO_ROW_ERRORS") == nullptr;
    inline static const Value zero{};

public:
    std::size_t size() const noexcept { return count; }
    bool empty() const noexcept { return count == 0; }
    const Value& operator[](std::size_t i) const
    {
        return values.empty() ? zero : values[i];
    }

    struct Iterator {
        const CartesianRunErrorStorage* owner;
        std::size_t index;
        const Value& operator*() const { return (*owner)[index]; }
        Iterator& operator++() { ++index; return *this; }
        bool operator!=(const Iterator& other) const { return index != other.index; }
    };
    Iterator begin() const { return {this, 0}; }
    Iterator end() const { return {this, count}; }

    void push_back(const Value& value)
    {
        if (values.empty() && compress) {
            if (std::memcmp(value.data(), zero.data(), sizeof(double) * 5) == 0) {
                ++count;
                return;
            }
            // Materialize the zero prefix only when a nonzero error appears.
            values.resize(count);
        }
        values.push_back(value);
        ++count;
    }
};

struct CartesianRunFillResult {
    CartesianRunFillStatus status = CartesianRunFillStatus::ok;
    std::vector<CartesianRunRecord> runs;
    // Horizontal integral for each run, including its Bennett boundary terms.
    // Kept aligned with `runs` so the outer-coordinate rule can reconstruct
    // F(y) for each connected image component.
    std::vector<double> run_areas;
    std::vector<std::optional<RetainedRowCore>> run_cores;
    // Compact error metadata only; aligned with runs in the shared estimator.
    CartesianRunErrorStorage run_flux_errors;
    std::vector<std::array<double,2>> run_error_fractions;
    // Empty unless explicitly requested by a richer boundary callback. Keep
    // per-run witnesses for local repair without repeating the global fill.
    std::vector<CartesianRunBoundaryDiagnostic> run_boundary_diagnostics;
    std::vector<std::size_t> component_roots;
    std::vector<int> component_rows;
    std::vector<long double> component_areas;
    std::vector<std::int64_t> component_cells;
    std::vector<int> component_boundary_edges;
    CartesianRunFillCounters counters;
    bool coverage_certified=false;
    double coverage_seconds=0;

    bool ok() const noexcept { return status == CartesianRunFillStatus::ok; }
};

// A vertical image segment is a one-to-one chain of horizontal runs in
// adjacent y rows.  The chain is deliberately separated at a split or merge
// in the run graph.  Bennett's check_bktrack routine does the analogous job
// while walking polar rows: it notices that a boundary has turned around and
// rescans the skipped strip as a separate image segment.  Here the scanline
// fill has already retained every maximal run, so the strip scan is equivalent
// to constructing the adjacent-row graph and walking its one-to-one chains.
struct CartesianOuterSegment {
    std::vector<std::size_t> run_indices;
    bool lower_boundary = false;
    bool upper_boundary = false;
    bool lower_junction = false;
    bool upper_junction = false;
};

struct CartesianOuterSegmentWalk {
    std::vector<CartesianOuterSegment> segments;
    std::size_t adjacency_edges = 0;
    std::size_t junction_nodes = 0;
    bool complete = true;
};

inline bool cartesian_runs_overlap_or_touch(
    const CartesianRun& first,
    const CartesianRun& second)
{
    const bool first_starts_before_second_end =
        second.hi == std::numeric_limits<std::int64_t>::max() ||
        first.lo <= second.hi + 1;
    const bool second_starts_before_first_end =
        first.hi == std::numeric_limits<std::int64_t>::max() ||
        second.lo <= first.hi + 1;
    return first_starts_before_second_end &&
        second_starts_before_first_end;
}

// Build a deterministic, disjoint decomposition of one connected run-fill
// component.  Runs in a segment always advance by one lattice row.  A node
// with more than one predecessor or successor is a topology junction, so no
// outer endpoint rule is inferred through it.  This is the conservative part
// of the walker: a later caller may use a segment endpoint only after also
// checking that the adjacent physical row is outside the image.
inline CartesianOuterSegmentWalk walk_cartesian_outer_segments(
    const std::vector<CartesianRunRecord>& runs,
    std::size_t component)
{
    CartesianOuterSegmentWalk result;
    std::vector<std::size_t> nodes;
    nodes.reserve(runs.size());
    for (std::size_t index = 0; index < runs.size(); ++index) {
        if (runs[index].component == component) {
            nodes.push_back(index);
        }
    }
    std::sort(
        nodes.begin(), nodes.end(),
        [&](std::size_t first, std::size_t second) {
            const auto& lhs = runs[first].run;
            const auto& rhs = runs[second].run;
            if (lhs.iy != rhs.iy) {
                return lhs.iy < rhs.iy;
            }
            if (lhs.lo != rhs.lo) {
                return lhs.lo < rhs.lo;
            }
            if (lhs.hi != rhs.hi) {
                return lhs.hi < rhs.hi;
            }
            return first < second;
        });
    if (nodes.empty()) {
        return result;
    }

    constexpr std::size_t no_neighbor =
        std::numeric_limits<std::size_t>::max();
    // Only one-to-one edges are followed.  Retaining the first neighbour and
    // a degree capped at two avoids one heap allocation per run on the common
    // clean-arc path while still certifying every split/merge.
    std::vector<std::size_t> single_predecessor(
        nodes.size(), no_neighbor);
    std::vector<std::size_t> single_successor(nodes.size(), no_neighbor);
    std::vector<unsigned char> predecessor_degree(nodes.size(), 0);
    std::vector<unsigned char> successor_degree(nodes.size(), 0);

    for (std::size_t first_row = 0; first_row < nodes.size();) {
        std::size_t first_end = first_row + 1;
        const std::int64_t row = runs[nodes[first_row]].run.iy;
        while (first_end < nodes.size() &&
               runs[nodes[first_end]].run.iy == row) {
            ++first_end;
        }
        std::size_t second_row = first_end;
        if (row != std::numeric_limits<std::int64_t>::max() &&
            second_row < nodes.size() &&
            runs[nodes[second_row]].run.iy == row + 1) {
            std::size_t second_end = second_row + 1;
            while (second_end < nodes.size() &&
                   runs[nodes[second_end]].run.iy == row + 1) {
                ++second_end;
            }
            for (std::size_t first = first_row; first < first_end; ++first) {
                for (std::size_t second = second_row;
                     second < second_end; ++second) {
                    if (!cartesian_runs_overlap_or_touch(
                            runs[nodes[first]].run,
                            runs[nodes[second]].run)) {
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
        first_row = first_end;
    }

    for (std::size_t local = 0; local < nodes.size(); ++local) {
        if (predecessor_degree[local] > 1U ||
            successor_degree[local] > 1U) {
            ++result.junction_nodes;
        }
    }

    std::vector<bool> assigned(nodes.size(), false);
    for (std::size_t seed = 0; seed < nodes.size(); ++seed) {
        if (assigned[seed]) {
            continue;
        }

        // Walk backwards through only one-to-one edges, then forwards from
        // the seed.  This gives a disjoint chain even when a parent run fans
        // out: the parent belongs to one chain and each child starts another
        // chain at the junction.
        std::vector<std::size_t> reversed;
        reversed.push_back(seed);
        assigned[seed] = true;
        std::size_t current = seed;
        while (predecessor_degree[current] == 1U) {
            const std::size_t previous = single_predecessor[current];
            if (previous == no_neighbor || assigned[previous] ||
                successor_degree[previous] != 1U) {
                break;
            }
            reversed.push_back(previous);
            assigned[previous] = true;
            current = previous;
        }
        std::reverse(reversed.begin(), reversed.end());

        std::vector<std::size_t> chain = std::move(reversed);
        current = seed;
        while (successor_degree[current] == 1U) {
            const std::size_t next = single_successor[current];
            if (next == no_neighbor || assigned[next] ||
                predecessor_degree[next] != 1U) {
                break;
            }
            chain.push_back(next);
            assigned[next] = true;
            current = next;
        }

        CartesianOuterSegment segment;
        segment.run_indices.reserve(chain.size());
        for (const std::size_t local : chain) {
            segment.run_indices.push_back(nodes[local]);
        }
        segment.lower_boundary = predecessor_degree[chain.front()] == 0U;
        segment.upper_boundary = successor_degree[chain.back()] == 0U;
        segment.lower_junction = predecessor_degree[chain.front()] != 0U;
        segment.upper_junction = successor_degree[chain.back()] != 0U;
        result.segments.push_back(std::move(segment));
    }

    // The decomposition must account for every run exactly once.  Keep this
    // as a certificate rather than silently applying a partial correction if
    // a future change to the graph walk violates that invariant.
    std::size_t accounted = 0;
    for (const auto& segment : result.segments) {
        accounted += segment.run_indices.size();
    }
    result.complete = accounted == nodes.size();
    return result;
}

// Lift one exact coarse-lattice point from every run in a component.  A
// component that is 8-connected on the coarse lattice may split into several
// center-sampled components after refinement; seeding only its first run would
// then retain just one of those pieces.  The exact lift keeps every selected
// point at the same physical coordinate on an integer refinement factor.
inline std::optional<std::vector<CartesianLatticeSeed>>
lift_cartesian_component_run_seeds(
    const CartesianRunFillResult& coarse,
    std::size_t component,
    std::int64_t factor,
    std::int64_t row_factor=0)
{
    if(row_factor==0)row_factor=factor;
    if (factor <= 0 || row_factor <= 0) {
        return std::nullopt;
    }
    const auto scale = [](std::int64_t value,std::int64_t factor)
        -> std::optional<std::int64_t> {
        if (value > std::numeric_limits<std::int64_t>::max() / factor ||
            value < std::numeric_limits<std::int64_t>::min() / factor) {
            return std::nullopt;
        }
        return value * factor;
    };

    std::vector<CartesianLatticeSeed> seeds;
    seeds.reserve(static_cast<std::size_t>(std::count_if(
        coarse.runs.begin(), coarse.runs.end(),
        [component](const CartesianRunRecord& record) {
            return record.component == component;
        })));
    for (const auto& record : coarse.runs) {
        if (record.component != component) {
            continue;
        }
        const auto ix = scale(record.run.lo,factor);
        const auto iy = scale(record.run.iy,row_factor);
        if (!ix.has_value() || !iy.has_value()) {
            return std::nullopt;
        }
        seeds.push_back({*ix, *iy});
    }
    return seeds;
}

namespace cartesian_run_fill_detail {

struct InsideInterval {
    std::int64_t lo = 0;
    std::int64_t hi = -1;
    std::size_t run_index = 0;
};

template <typename Interval>
class SmallIntervalRow {
public:
    std::size_t size() const noexcept
    {
        if (!has_first_) {
            return 0;
        }
        return spilled_ ? intervals_->size() : 1;
    }

    const Interval* find(std::int64_t ix) const
    {
        if (!has_first_) {
            return nullptr;
        }
        if (!spilled_) {
            return ix >= first_.lo && ix <= first_.hi ? &first_ : nullptr;
        }
        const auto after = std::upper_bound(
            intervals_->begin(), intervals_->end(), ix,
            [](std::int64_t value, const Interval& interval) {
                return value < interval.lo;
            });
        if (after == intervals_->begin()) {
            return nullptr;
        }
        const auto& candidate = *std::prev(after);
        return ix <= candidate.hi ? &candidate : nullptr;
    }

    template <typename Visitor>
    void visit_overlapping(
        std::int64_t lo, std::int64_t hi, Visitor&& visitor) const
    {
        if (!has_first_) {
            return;
        }
        if (!spilled_) {
            if (first_.hi >= lo && first_.lo <= hi) {
                visitor(first_);
            }
            return;
        }
        for (const auto& interval : *intervals_) {
            if (interval.hi < lo) {
                continue;
            }
            if (interval.lo > hi) {
                break;
            }
            visitor(interval);
        }
    }

    template <typename Visitor>
    void visit_all(Visitor&& visitor) const
    {
        if (!has_first_) {
            return;
        }
        if (!spilled_) {
            visitor(first_);
            return;
        }
        for (const auto& interval : *intervals_) {
            visitor(interval);
        }
    }

    void insert(Interval interval)
    {
        if (!has_first_) {
            first_ = interval;
            has_first_ = true;
            return;
        }
        if (!spilled_) {
            intervals_ = std::make_unique<std::vector<Interval>>();
            intervals_->reserve(4);
            intervals_->push_back(first_);
            intervals_->push_back(interval);
            if ((*intervals_)[1].lo < (*intervals_)[0].lo) {
                std::swap((*intervals_)[0], (*intervals_)[1]);
            }
            spilled_ = true;
            return;
        }
        const auto position = std::lower_bound(
            intervals_->begin(), intervals_->end(), interval.lo,
            [](const Interval& item, std::int64_t value) {
                return item.lo < value;
            });
        intervals_->insert(position, interval);
    }

private:
    bool has_first_ = false;
    bool spilled_ = false;
    Interval first_ {};
    std::unique_ptr<std::vector<Interval>> intervals_;
};

struct RowState {
    SmallIntervalRow<InsideInterval> inside;
};

class RowRegistry {
public:
    explicit RowRegistry(
        std::size_t expected_rows = 0,
        std::size_t expected_entries = 0)
    {
        reserve(expected_rows, std::max(expected_rows, expected_entries));
    }

    const RowState* find(std::int64_t iy) const
    {
        if (buckets_.empty()) {
            return nullptr;
        }
        std::size_t index = bucket(iy, buckets_.size());
        while (buckets_[index] != missing) {
            const auto& entry = entries_[buckets_[index]];
            if (entry.key == iy) {
                return &entry.value;
            }
            index = (index + 1) & (buckets_.size() - 1);
        }
        return nullptr;
    }

    RowState& get(std::int64_t iy)
    {
        if (buckets_.empty() || (entries_.size() + 1) * 10 >= buckets_.size() * 7) {
            rehash(buckets_.empty() ? 16 : buckets_.size() * 2);
        }
        std::size_t index = bucket(iy, buckets_.size());
        while (buckets_[index] != missing &&
               entries_[buckets_[index]].key != iy) {
            index = (index + 1) & (buckets_.size() - 1);
        }
        if (buckets_[index] == missing) {
            buckets_[index] = entries_.size();
            entries_.push_back({iy, {}});
        }
        return entries_[buckets_[index]].value;
    }

    const InsideInterval* find_inside(
        std::int64_t iy, std::int64_t ix) const
    {
        const auto* row = find(iy);
        return row == nullptr ? nullptr : row->inside.find(ix);
    }

    template <typename Visitor>
    void visit_rows(Visitor&& visitor) const
    {
        for (const auto& entry : entries_) {
            visitor(entry.key, entry.value);
        }
    }

private:
    struct Entry {
        std::int64_t key = 0;
        RowState value;
    };
    static constexpr std::size_t missing =
        std::numeric_limits<std::size_t>::max();

    static std::size_t bucket(std::int64_t key, std::size_t capacity)
    {
        return static_cast<std::size_t>(key) & (capacity - 1);
    }

    void reserve(std::size_t expected_rows, std::size_t expected_entries)
    {
        std::size_t capacity = 16;
        while (capacity * 7 / 10 < expected_rows) {
            capacity *= 2;
        }
        entries_.reserve(expected_entries);
        rehash(capacity);
    }

    void rehash(std::size_t capacity)
    {
        std::vector<std::size_t> replacement(capacity, missing);
        for (std::size_t entry_index = 0;
             entry_index < entries_.size(); ++entry_index) {
            std::size_t index = bucket(entries_[entry_index].key, capacity);
            while (replacement[index] != missing) {
                index = (index + 1) & (capacity - 1);
            }
            replacement[index] = entry_index;
        }
        buckets_ = std::move(replacement);
    }

    std::vector<Entry> entries_;
    std::vector<std::size_t> buckets_;
};

class Components {
public:
    explicit Components(std::size_t expected = 0)
    {
        parent_.reserve(expected);
        area_.reserve(expected);
        cells_.reserve(expected);
        boundary_edges_.reserve(expected);
    }

    std::size_t add()
    {
        const std::size_t index = parent_.size();
        parent_.push_back(index);
        area_.push_back(0.0L);
        cells_.push_back(0);
        boundary_edges_.push_back(0);
        return index;
    }

    std::size_t find(std::size_t index)
    {
        std::size_t root = index;
        while (parent_[root] != root) {
            root = parent_[root];
        }
        while (parent_[index] != index) {
            const std::size_t next = parent_[index];
            parent_[index] = root;
            index = next;
        }
        return root;
    }

    bool merge(std::size_t lhs, std::size_t rhs)
    {
        lhs = find(lhs);
        rhs = find(rhs);
        if (lhs == rhs) {
            return false;
        }
        // The smaller root decides identity so final component labels do not
        // depend on frontier order.
        const std::size_t root = std::min(lhs, rhs);
        const std::size_t child = std::max(lhs, rhs);
        parent_[child] = root;
        area_[root] += area_[child];
        cells_[root] += cells_[child];
        boundary_edges_[root] += boundary_edges_[child];
        return true;
    }

    void accumulate(
        std::size_t index, double area, std::int64_t cells, int boundary_edges)
    {
        const std::size_t root = find(index);
        area_[root] += static_cast<long double>(area);
        cells_[root] += cells;
        boundary_edges_[root] += boundary_edges;
    }

    long double area(std::size_t index) { return area_[find(index)]; }
    std::int64_t cells(std::size_t index) { return cells_[find(index)]; }
    int boundary_edges(std::size_t index)
    {
        return boundary_edges_[find(index)];
    }

    std::size_t size() const noexcept { return parent_.size(); }

private:
    std::vector<std::size_t> parent_;
    std::vector<long double> area_;
    std::vector<std::int64_t> cells_;
    std::vector<int> boundary_edges_;
};

inline bool decrement(std::int64_t value, std::int64_t& result)
{
    if (value == std::numeric_limits<std::int64_t>::min()) {
        return false;
    }
    result = value - 1;
    return true;
}

inline bool increment(std::int64_t value, std::int64_t& result)
{
    if (value == std::numeric_limits<std::int64_t>::max()) {
        return false;
    }
    result = value + 1;
    return true;
}

} // namespace cartesian_run_fill_detail

// Deterministic 8-connected scanline fill over an implicit integer lattice.
//
// `classify(ix, iy)` returns a cell state, `is_inside(state)` is the physical
// membership predicate, `cell_weight(...)` returns the cell's contribution,
// and `boundary_weight(...)` returns the two physical run-edge corrections.
// Every newly discovered inside cell is classified once.  Empty projection
// gaps may be rechecked from another neighbouring run, but only boundary/gap
// cells are revisited; no storage or allocation grows with an unbounded empty
// plane.  Global accounting is intentionally absent: callers may suppress an
// already-accounted cell in `cell_weight`, but that decision can never stop
// topology traversal or create a boundary.
template <typename CellState,
          typename Classify,
          typename IsInside,
          typename CellWeight,
          typename BoundaryWeight,
          typename Trace = CartesianRunFillNoTrace>
CartesianRunFillResult fill_cartesian_runs(
    std::vector<CartesianLatticeSeed> seeds,
    Classify&& classify,
    IsInside&& is_inside,
    CellWeight&& cell_weight,
    BoundaryWeight&& boundary_weight,
    CartesianRunFillLimits limits = {},
    Trace& trace = default_cartesian_run_fill_trace(),
    const std::vector<CartesianLatticeSeed>* coverage_probes = nullptr,
    const CartesianCoverageDomain* coverage_domain = nullptr,
    const std::function<std::optional<std::array<double,2>>(std::int64_t,std::int64_t,std::int64_t)>* interior_block = nullptr,
    int interior_block_size=32,
    const std::function<std::optional<CartesianRetainedRun>(std::int64_t,std::int64_t)>* retained_row=nullptr,
    bool retain_row_cores=false)
{
    using namespace cartesian_run_fill_detail;
    const bool lazy_row_cores=std::getenv("LCBININT_DIAGNOSTIC_DISABLE_LAZY_CORES")==nullptr;

    CartesianRunFillResult result;
    result.counters.seeds_offered = static_cast<std::int64_t>(seeds.size());
    std::sort(
        seeds.begin(), seeds.end(),
        [](const CartesianLatticeSeed& lhs, const CartesianLatticeSeed& rhs) {
            if (lhs.iy != rhs.iy) {
                return lhs.iy < rhs.iy;
            }
            return lhs.ix < rhs.ix;
        });
    seeds.erase(
        std::unique(
            seeds.begin(), seeds.end(),
            [](const CartesianLatticeSeed& lhs, const CartesianLatticeSeed& rhs) {
                return lhs.ix == rhs.ix && lhs.iy == rhs.iy;
            }),
        seeds.end());
    result.counters.unique_seed_cells = static_cast<std::int64_t>(seeds.size());
    result.counters.duplicate_seeds_avoided =
        result.counters.seeds_offered - result.counters.unique_seed_cells;

    const std::size_t expected_row_count = std::min(
        limits.maximum_runs,
        std::max(limits.expected_rows, seeds.size() * 4));
    const std::size_t expected_runs = std::min(
        limits.maximum_runs,
        expected_row_count + expected_row_count / 2);
    RowRegistry rows(expected_row_count, expected_runs);
    Components components(seeds.size());
    std::vector<std::size_t> frontier;
    frontier.reserve(expected_runs);
    std::size_t frontier_head = 0;
    result.runs.reserve(expected_runs);
    if(std::getenv("LCBININT_DIAGNOSTIC_DISABLE_ROW_RESERVE")==nullptr) {
        result.run_areas.reserve(expected_runs);
        result.run_error_fractions.reserve(expected_runs);
    }

    const auto fail = [&](CartesianRunFillStatus status) {
        result.status = status;
        return false;
    };
    const auto evaluate = [&](std::int64_t ix, std::int64_t iy)
        -> std::optional<CellState> {
        if (result.counters.mapped_cell_evaluations >=
            limits.maximum_evaluations) {
            result.status = CartesianRunFillStatus::evaluation_budget_exhausted;
            return std::nullopt;
        }
        ++result.counters.mapped_cell_evaluations;
        return classify(ix, iy);
    };

    const auto covered_hi = [&](const RowState* row,
                                std::int64_t ix)
        -> std::optional<std::int64_t> {
        if (row == nullptr) {
            return std::nullopt;
        }
        if (const auto* interval = row->inside.find(ix)) {
            return interval->hi;
        }
        return std::nullopt;
    };

    const auto discover_run = [&](std::int64_t seed_ix,
                                  std::int64_t iy,
                                  const CellState& seed_state,
                                  std::size_t component,
                                  std::size_t parent_run_index,
                                  std::optional<CellState> left_hint)
        -> std::optional<std::size_t> {
        if (result.runs.size() >= limits.maximum_runs) {
            fail(CartesianRunFillStatus::run_budget_exhausted);
            return std::nullopt;
        }

        std::int64_t lo=seed_ix,hi=seed_ix,cells=0;double area=0;
        CartesianBoundaryContribution boundary;
        const auto saved=retained_row ? (*retained_row)(seed_ix,iy) : std::nullopt;
        if(saved) {
            lo=saved->run.lo;hi=saved->run.hi;cells=hi-lo+1;area=saved->area;
            boundary.edges=2;boundary.error_recorded=true;
            boundary.error_signed=saved->errors[0];boundary.error_absolute=saved->errors[1];boundary.error_unresolved=int(saved->errors[2]);boundary.error_remainder=saved->errors[3];boundary.error_limb_density=saved->errors[4];boundary.error_fractions=saved->fractions;
        } else {

        CellState left_inside = seed_state;
        CellState right_inside = seed_state;
        std::optional<CellState> left_outside;
        std::optional<CellState> right_outside;
        // Optional richer boundary callbacks reuse neighbours that the scan
        // has already classified. No cell cache or additional map evaluations.
        std::optional<CellState> left_second,right_second,left_first,right_first;
        area = cell_weight(seed_ix, iy, seed_state);
        if (!std::isfinite(area)) {
            fail(CartesianRunFillStatus::invalid_contribution);
            return std::nullopt;
        }
        cells = 1;
        double interior_remainder=0;

        std::int64_t ix = 0;
        if (!decrement(lo, ix)) {
            fail(CartesianRunFillStatus::coordinate_overflow);
            return std::nullopt;
        }
        if (left_hint.has_value()) {
            left_outside = std::move(left_hint);
        } else {
            while (true) {
                if (interior_block && ix%interior_block_size==0 && ix >= std::numeric_limits<std::int64_t>::min()+32 && ix <= std::numeric_limits<std::int64_t>::max()-32) {
                    const std::int64_t count=interior_block_size;
                    const std::int64_t end=ix-(count-1);
                    if (const auto sum=(*interior_block)(std::min(ix,end),std::max(ix,end),iy)) {
                        // Keep the two endpoint states required by Bennett's correction.
                        const auto last=evaluate(end,iy),second=evaluate(end+1,iy);
                        if (!last || !second) return std::nullopt;
                        if (!is_inside(*last) || !is_inside(*second) || !std::isfinite((*sum)[0]) || !std::isfinite((*sum)[1])) {
                            fail(CartesianRunFillStatus::invalid_contribution);return std::nullopt;
                        }
                        if (!left_first) left_first=*second;
                        left_second=*second;left_inside=*last;
                        lo=end;area+=(*sum)[0];interior_remainder+=(*sum)[1];cells+=count;
                        if (!decrement(end,ix)) {fail(CartesianRunFillStatus::coordinate_overflow);return std::nullopt;}
                        continue;
                    }
                }
                const auto state = evaluate(ix, iy);
                if (!state.has_value()) {
                    return std::nullopt;
                }
                if (!is_inside(*state)) {
                    left_outside = *state;
                    break;
                }
                lo = ix;
                if constexpr (std::is_invocable_v<BoundaryWeight,const CartesianRun&,
                    const CellState&,const CellState&,const CellState&,const CellState&,
                    const std::optional<CellState>&,const std::optional<CellState>&>) {
                    left_second=left_inside;
                    if (!left_first) left_first=*state;
                }
                left_inside = *state;
                const double contribution = cell_weight(ix, iy, *state);
                if (!std::isfinite(contribution)) {
                    fail(CartesianRunFillStatus::invalid_contribution);
                    return std::nullopt;
                }
                area += contribution;
                ++cells;
                if (!decrement(ix, ix)) {
                    fail(CartesianRunFillStatus::coordinate_overflow);
                    return std::nullopt;
                }
            }
        }

        if (!increment(hi, ix)) {
            fail(CartesianRunFillStatus::coordinate_overflow);
            return std::nullopt;
        }
        while (true) {
            if (interior_block && ix%interior_block_size==0 && ix >= std::numeric_limits<std::int64_t>::min()+32 && ix <= std::numeric_limits<std::int64_t>::max()-32) {
                const std::int64_t count=interior_block_size;
                const std::int64_t end=ix+(count-1);
                if (const auto sum=(*interior_block)(std::min(ix,end),std::max(ix,end),iy)) {
                    // Keep the two endpoint states required by Bennett's correction.
                    const auto last=evaluate(end,iy),second=evaluate(end-1,iy);
                    if (!last || !second) return std::nullopt;
                    if (!is_inside(*last) || !is_inside(*second) || !std::isfinite((*sum)[0]) || !std::isfinite((*sum)[1])) {
                        fail(CartesianRunFillStatus::invalid_contribution);return std::nullopt;
                    }
                    if (!right_first) right_first=*second;
                    right_second=*second;right_inside=*last;
                    hi=end;area+=(*sum)[0];interior_remainder+=(*sum)[1];cells+=count;
                    if (!increment(end,ix)) {fail(CartesianRunFillStatus::coordinate_overflow);return std::nullopt;}
                    continue;
                }
            }
            const auto state = evaluate(ix, iy);
            if (!state.has_value()) {
                return std::nullopt;
            }
            if (!is_inside(*state)) {
                right_outside = *state;
                break;
            }
            hi = ix;
            if constexpr (std::is_invocable_v<BoundaryWeight,const CartesianRun&,
                const CellState&,const CellState&,const CellState&,const CellState&,
                const std::optional<CellState>&,const std::optional<CellState>&>) {
                right_second=right_inside;
                if (!right_first) right_first=*state;
            }
            right_inside = *state;
            const double contribution = cell_weight(ix, iy, *state);
            if (!std::isfinite(contribution)) {
                fail(CartesianRunFillStatus::invalid_contribution);
                return std::nullopt;
            }
            area += contribution;
            ++cells;
            if (!increment(ix, ix)) {
                fail(CartesianRunFillStatus::coordinate_overflow);
                return std::nullopt;
            }
        }

        if (!left_outside.has_value() || !right_outside.has_value()) {
            fail(CartesianRunFillStatus::coordinate_overflow);
            return std::nullopt;
        }
        const CartesianRun run {iy, lo, hi};

        if constexpr (std::is_invocable_v<BoundaryWeight,const CartesianRun&,
            const CellState&,const CellState&,const CellState&,const CellState&,
            const std::optional<CellState>&,const std::optional<CellState>&>) {
            if (!left_second) left_second=right_first;
            if (!right_second) right_second=left_first;
            boundary=boundary_weight(run,left_inside,*left_outside,right_inside,*right_outside,
                                     left_second,right_second);
        } else boundary=boundary_weight(run,left_inside,*left_outside,right_inside,*right_outside);
        boundary.error_remainder+=interior_remainder;
        if (!boundary.valid || !std::isfinite(boundary.area)) {
            fail(CartesianRunFillStatus::invalid_contribution);
            return std::nullopt;
        }
        area += boundary.area;

        }
        const CartesianRun run{iy,lo,hi};
        const std::size_t run_index = result.runs.size();
        result.runs.push_back({run, component});
        result.run_areas.push_back(area);
        if(retain_row_cores && !lazy_row_cores)result.run_cores.push_back(saved?saved->core:std::nullopt);
        else if(retain_row_cores && (!result.run_cores.empty() || (saved && saved->core))) {
            // An empty vector already means that no retained core exists.
            // Allocate aligned optional metadata only on the first real core.
            if(result.run_cores.empty())result.run_cores.resize(run_index);
            result.run_cores.push_back(saved?saved->core:std::nullopt);
        }
        if(boundary.error_recorded) {
            result.run_flux_errors.push_back({boundary.error_signed,boundary.error_absolute,static_cast<double>(boundary.error_unresolved),boundary.error_remainder,boundary.error_limb_density});
            result.run_error_fractions.push_back(boundary.error_fractions);
        }
        if (boundary.diagnostic) {
            if (result.run_boundary_diagnostics.empty())
                result.run_boundary_diagnostics.resize(run_index);
            result.run_boundary_diagnostics.push_back(*boundary.diagnostic);
            if constexpr (std::is_same_v<Trace,CartesianRunFillTrace>) {
                if (trace.boundary_diagnostics.size()<trace.maximum_events)
                    trace.boundary_diagnostics.push_back({run,*boundary.diagnostic,trace.cell_spacing,trace.fill_level,run_index,area,1.0});
                else trace.truncated=true;
            }
        } else if (!result.run_boundary_diagnostics.empty()) {
            result.run_boundary_diagnostics.emplace_back();
        }
        trace.run_discovered(
            run_index, run, parent_run_index, component);
        components.accumulate(component, area, cells, boundary.edges);
        auto& inside_row = rows.get(iy).inside;
        inside_row.insert({lo, hi, run_index});
        result.counters.maximum_runs_in_row = std::max(
            result.counters.maximum_runs_in_row, inside_row.size());
        frontier.push_back(run_index);
        ++result.counters.maximal_runs_discovered;
        ++result.counters.row_intervals_stored;
        return run_index;
    };

    const auto drain_frontier = [&]() {
      while (frontier_head < frontier.size()) {
        const std::size_t run_index = frontier[frontier_head++];
        ++result.counters.frontier_intervals_popped;
        const CartesianRun run = result.runs[run_index].run;
        const std::size_t component = result.runs[run_index].component;
        trace.frontier_popped(run_index, run, component);

        std::int64_t candidate_lo = 0;
        std::int64_t candidate_hi = 0;
        if (!decrement(run.lo, candidate_lo) ||
            !increment(run.hi, candidate_hi)) {
            fail(CartesianRunFillStatus::coordinate_overflow);
            return false;
        }
        for (int direction : {-1, 1}) {
            std::int64_t next_iy = 0;
            const bool row_ok = direction < 0
                ? decrement(run.iy, next_iy)
                : increment(run.iy, next_iy);
            if (!row_ok) {
                fail(CartesianRunFillStatus::coordinate_overflow);
                return false;
            }

            if (const auto* existing_row = rows.find(next_iy)) {
                existing_row->inside.visit_overlapping(
                    candidate_lo, candidate_hi,
                    [&](const InsideInterval& interval) {
                    if (components.merge(
                            component, result.runs[interval.run_index].component)) {
                        ++result.counters.merged_components;
                        trace.components_merged(
                            run_index, interval.run_index, component);
                    }
                });
            }

            std::int64_t ix = candidate_lo;
            std::optional<CellState> previous_outside;
            const RowState* row = rows.find(next_iy);
            while (ix <= candidate_hi) {
                if (const auto covered = covered_hi(row, ix)) {
                    if (*covered >= candidate_hi) {
                        break;
                    }
                    if (!increment(*covered, ix)) {
                        fail(CartesianRunFillStatus::coordinate_overflow);
                        return false;
                    }
                    previous_outside.reset();
                    continue;
                }

                const auto state = evaluate(ix, next_iy);
                if (!state.has_value()) {
                    return false;
                }
                if (!is_inside(*state)) {
                    previous_outside = *state;
                    if (ix == candidate_hi) {
                        break;
                    }
                    if (!increment(ix, ix)) {
                        fail(CartesianRunFillStatus::coordinate_overflow);
                        return false;
                    }
                    continue;
                }

                const std::size_t root = components.find(component);
                const std::int64_t seed_ix = ix;
                const auto discovered = discover_run(
                    seed_ix, next_iy, *state, root, run_index, previous_outside);
                if (!discovered.has_value()) {
                    return false;
                }
                row = rows.find(next_iy);
                const auto& child = result.runs[*discovered].run;
                if (child.hi >= candidate_hi) {
                    break;
                }
                if (!increment(child.hi, ix) || !increment(ix, ix)) {
                    fail(CartesianRunFillStatus::coordinate_overflow);
                    return false;
                }
                previous_outside.reset();
            }
        }
      }
      return true;
    };

    // Preserve master scheduling for quadrature diagnostics. Sequential seed
    // processing remains available explicitly for the archived seed experiment.
    const bool sequential_seeds = std::getenv("LCBININT_SEQUENTIAL_SEED_FILL") != nullptr;
    for (const auto& seed : seeds) {
        if (rows.find_inside(seed.iy, seed.ix) != nullptr) {
            ++result.counters.duplicate_seeds_avoided;
            continue;
        }
        const auto state = evaluate(seed.ix, seed.iy);
        if (!state.has_value()) {
            return result;
        }
        if (!is_inside(*state)) {
            continue;
        }
        const std::size_t component = components.add();
        ++result.counters.provisional_components;
        if (!discover_run(
                seed.ix, seed.iy, *state, component,
                CartesianRunFillTraceEvent::npos,
                std::nullopt).has_value()) {
            return result;
        }
        if (sequential_seeds && !drain_frontier()) return result;
    }
    if (!drain_frontier()) return result;

    // Incremental coverage repair: the normal frontier is exhausted first.
    // Only an unreached inside cell starts more work. The row registry, run
    // graph, cell weights, and existing frontier are retained throughout.
    if (coverage_probes != nullptr || coverage_domain != nullptr) {
        const auto add_cell = [&](std::int64_t ix,std::int64_t iy) {
            ++result.counters.coverage_probe_cells;
            if (rows.find_inside(iy,ix)!=nullptr) return true;
            const auto state=evaluate(ix,iy);
            if (!state.has_value()) return false;
            if (!is_inside(*state)) return true;
            const auto component=components.add();
            ++result.counters.provisional_components;
            ++result.counters.coverage_components_added;
            if (!discover_run(ix,iy,*state,component,
                CartesianRunFillTraceEvent::npos,std::nullopt).has_value()) return false;
            return drain_frontier();
        };
        if (coverage_probes) for (const auto& probe:*coverage_probes) {
            if (!add_cell(probe.ix,probe.iy)) return result;
            // A claimed centre already reaches every inside 8-neighbour.
            // Only an outside centre needs the same local snapping fallback
            // as a regular seed; search all neighbours to retain split pieces.
            if (rows.find_inside(probe.iy,probe.ix)!=nullptr) continue;
            for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx) {
                if (dx==0 && dy==0) continue;
                std::int64_t ix=probe.ix,iy=probe.iy;
                if ((dx<0 && !decrement(ix,ix)) || (dx>0 && !increment(ix,ix))
                    || (dy<0 && !decrement(iy,iy)) || (dy>0 && !increment(iy,iy))) {
                    fail(CartesianRunFillStatus::coordinate_overflow);return result;
                }
                if (!add_cell(ix,iy)) return result;
            }
        }
        if (coverage_domain && coverage_domain->extent>=0 && coverage_domain->outside) {
            const auto coverage_started=std::chrono::steady_clock::now();
            using Box=std::array<std::int64_t,4>;
            const auto n=coverage_domain->extent;
            std::vector<Box> pending;
            if (coverage_domain->fenced_regions) {
                std::vector<Box> bounds(components.size(),{n,-n,n,-n});
                for (const auto& r:result.runs) {
                    auto& b=bounds[components.find(r.component)];
                    b[0]=std::min(b[0],r.run.lo);b[1]=std::max(b[1],r.run.hi);
                    b[2]=std::min(b[2],r.run.iy);b[3]=std::max(b[3],r.run.iy);
                }
                bounds.erase(std::remove_if(bounds.begin(),bounds.end(),[](const Box& b){return b[0]>b[1];}),bounds.end());
                pending=coverage_domain->fenced_regions(bounds);
            }
            if (pending.empty()) pending.push_back({-n,n,-n,n});
            while (!pending.empty()) {
                if (static_cast<std::size_t>(result.counters.coverage_boxes)>=coverage_domain->maximum_boxes) {
                    result.counters.coverage_unresolved+=pending.size();break;
                }
                const Box b=pending.back();pending.pop_back();++result.counters.coverage_boxes;
                bool covered=true;
                for (auto y=b[2];y<=b[3];++y) {
                    const auto* interval=rows.find_inside(y,b[0]);
                    if (!interval || interval->hi<b[1]) {covered=false;break;}
                }
                if (covered || coverage_domain->outside(b[0],b[1],b[2],b[3])) continue;
                if (b[0]==b[1] && b[2]==b[3]) {
                    if (!add_cell(b[0],b[2])) return result;
                    if (!rows.find_inside(b[2],b[0])) ++result.counters.coverage_unresolved;
                    continue;
                }
                if (b[1]-b[0]>=b[3]-b[2]) {
                    const auto m=b[0]+(b[1]-b[0])/2;
                    pending.push_back({b[0],m,b[2],b[3]});pending.push_back({m+1,b[1],b[2],b[3]});
                } else {
                    const auto m=b[2]+(b[3]-b[2])/2;
                    pending.push_back({b[0],b[1],b[2],m});pending.push_back({b[0],b[1],m+1,b[3]});
                }
            }
            result.coverage_certified=result.counters.coverage_unresolved==0;
            result.coverage_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-coverage_started).count();
        }
    }

    result.component_roots.resize(components.size());
    result.component_areas.assign(components.size(), 0.0L);
    result.component_cells.assign(components.size(), 0);
    result.component_boundary_edges.assign(components.size(), 0);
    for (std::size_t index = 0; index < components.size(); ++index) {
        const std::size_t root = components.find(index);
        result.component_roots[index] = root;
        if (root == index) {
            result.component_areas[root] = components.area(root);
            result.component_cells[root] = components.cells(root);
            result.component_boundary_edges[root] =
                components.boundary_edges(root);
        }
    }
    for (auto& run : result.runs) {
        run.component = components.find(run.component);
    }
    result.component_rows.assign(components.size(), 0);
    rows.visit_rows([&](std::int64_t, const RowState& row) {
        if (row.inside.size() == 1) {
            row.inside.visit_all([&](const InsideInterval& interval) {
                ++result.component_rows[
                    result.runs[interval.run_index].component];
            });
            return;
        }
        std::vector<std::size_t> row_components;
        row_components.reserve(row.inside.size());
        bool multiple_runs_in_component = false;
        row.inside.visit_all([&](const InsideInterval& interval) {
            const std::size_t root =
                result.runs[interval.run_index].component;
            if (std::find(
                    row_components.begin(), row_components.end(), root) ==
                row_components.end()) {
                row_components.push_back(root);
                ++result.component_rows[root];
            } else {
                multiple_runs_in_component = true;
            }
        });
        if (multiple_runs_in_component) {
            ++result.counters.rows_with_multiple_runs;
        }
    });
    return result;
}

// Same local endpoint and smooth-remainder model for uniform/LD and for
// coarse/accepted refined support. Metadata and moments only; no new rays.
struct SameGridOuterError {
    double signed_residual=0,uncertainty=0;bool resolved=false;
    double limb_density_correction=0,limb_density_uncertainty=0;
    // Second differences explained by successful local singular models.
    // Only populated by the internal endpoint pass, not a new ray cache.
    std::vector<std::pair<std::int64_t,double>> modeled_second;
    // Diagnostic decomposition; existing acceptance arithmetic is unchanged.
    double cap_signed=0,cap_remainder=0,smooth_remainder=0;
    bool missing_cap_rows=false;
};
struct SameGridErrorNode {std::int64_t y;double lo,hi,value,w,width,limb_density;};
inline std::vector<SameGridErrorNode> same_grid_error_nodes(const CartesianRunFillResult& fill,
 std::size_t component,const std::vector<double>* weights,const std::vector<std::size_t>* ordered_runs=nullptr) {
 std::vector<SameGridErrorNode> nodes;nodes.reserve(ordered_runs?ordered_runs->size():fill.runs.size());
 const bool fractions=fill.run_error_fractions.size()==fill.runs.size();
 for(std::size_t k=0;k<(ordered_runs?ordered_runs->size():fill.runs.size());++k) {
  const auto j=ordered_runs?(*ordered_runs)[k]:k;
  if(!ordered_runs&&component!=std::numeric_limits<std::size_t>::max()&&fill.component_roots[fill.runs[j].component]!=component)continue;
  const auto& r=fill.runs[j].run;const auto f=fractions?fill.run_error_fractions[j]:std::array<double,2>{0,0};
  nodes.push_back({r.iy,double(r.lo)-f[0],double(r.hi)+f[1],fill.run_areas[j],weights?(*weights)[j]:1,double(r.hi-r.lo)+f[0]+f[1],fill.run_flux_errors.size()==fill.runs.size()?fill.run_flux_errors[j][4]:0});
 }
 if(!ordered_runs)std::sort(nodes.begin(),nodes.end(),[](const SameGridErrorNode& a,const SameGridErrorNode& b){return a.y!=b.y?a.y<b.y:a.lo<b.lo;});
 return nodes;
}
// Local topology terms from retained runs and crossing fractions only.
// No lens maps or image solves; these remain asymptotic error models.
inline SameGridOuterError same_grid_internal_error(const std::vector<SameGridErrorNode>& image,double limb_brightness,bool held_cap_checks=false) {
 using Node=SameGridErrorNode;
 if(image.empty())return {};
 std::vector<Node> gaps;
 for(std::size_t j=1;j<image.size();++j)if(image[j-1].y==image[j].y&&image[j-1].hi<image[j].lo)
  gaps.push_back({image[j].y,image[j-1].hi,image[j].lo,image[j].lo-image[j-1].hi,1,image[j].lo-image[j-1].hi,0});
 SameGridOuterError result{0,0,true};
 const auto process=[&](const std::vector<Node>& ns,bool gap) {
  std::vector<std::array<std::size_t,2>> links(ns.size());
  std::vector<std::array<int,2>> degrees(ns.size());
  std::size_t row=0;
  while(row<ns.size()) {
   std::size_t end=row+1;while(end<ns.size()&&ns[end].y==ns[row].y)++end;
   if(end<ns.size()&&ns[row].y!=std::numeric_limits<std::int64_t>::max()&&ns[end].y==ns[row].y+1) {
    std::size_t next_end=end+1;while(next_end<ns.size()&&ns[next_end].y==ns[end].y)++next_end;
    for(std::size_t i=row;i<end;++i)for(std::size_t j=end;j<next_end;++j) {
     const bool overlap=gap?(ns[i].lo<ns[j].hi&&ns[j].lo<ns[i].hi):(ns[i].lo<=ns[j].hi+1&&ns[j].lo<=ns[i].hi+1);
     if(overlap){links[i][1]=j;++degrees[i][1];links[j][0]=i;++degrees[j][0];}
    }
   }
   row=end;
  }
  std::vector<bool> closed(gap?ns.size():0,false);
  if(gap&&!std::getenv("LCBININT_DIAGNOSTIC_LEGACY_CLOSED_GAP")) {
   for(std::size_t i=0;i<ns.size();++i) {
    if(degrees[i][0]!=0||degrees[i][1]!=1)continue;
    std::array<std::size_t,7> panel{};std::array<double,7> widths{};
    panel[0]=i;std::size_t count=1;
    while(count<7&&degrees[panel[count-1]][1]==1) {
     const auto j=links[panel[count-1]][1];if(degrees[j][0]!=1)break;
     panel[count++]=j;
    }
    if(count<3||degrees[panel[count-1]][1]!=0)continue;
    const auto enclosed=[&](std::size_t n,int direction) {
     if((direction<0&&ns[n].y==std::numeric_limits<std::int64_t>::min())||(direction>0&&ns[n].y==std::numeric_limits<std::int64_t>::max()))return false;
     const auto y=ns[n].y+direction;
     auto it=std::lower_bound(image.begin(),image.end(),y,[](const Node& a,std::int64_t yy){return a.y<yy;});
     for(;it!=image.end()&&it->y==y;++it)if(it->lo<=ns[n].lo&&it->hi>=ns[n].hi)return true;
     return false;
    };
    if(!enclosed(i,-1)||!enclosed(panel[count-1],1))continue;
    double used=0;for(std::size_t k=0;k<count;++k){widths[k]=ns[panel[k]].value;used+=widths[k];}
    const auto model=closed_short_row_cap(widths,count);
    if(!model.resolved) {
     if(held_cap_checks) {
      // Two four-row endpoint expansions overlap on this closed footprint.
      // An invalid finite-cap fit cannot license both infinite-tail models.
      // Retain the existing zero-halo curvature fallback on these rows only.
      double curvature=0;
      for(std::size_t k=0;k<count;++k)curvature+=std::abs((k?widths[k-1]:0)-2*widths[k]+(k+1<count?widths[k+1]:0));
      result.uncertainty+=limb_brightness*curvature/24;result.resolved=false;
      for(std::size_t k=0;k<count;++k)closed[panel[k]]=true;
     }
     continue;
    }
    const auto alternative=closed_short_row_cap(widths,count,static_cast<int>(count/2)-1);
    const double integral=alternative.resolved?.5*(model.integral+alternative.integral):model.integral;
    const auto shape=[&](double x){return alternative.resolved?.5*(model.value(x)+alternative.value(x)):model.value(x);};
    double raw=0;
    const auto residual=[&](int k){return k>=0&&k<static_cast<int>(count)?widths[k]-shape(k):0.;};
    for(int k=0;k<static_cast<int>(count);++k)raw+=std::abs(residual(k-1)-2*residual(k)+residual(k+1));
    double sensitivity=alternative.resolved?.5*std::abs(model.integral-alternative.integral):0;
    if(held_cap_checks) {
     // Every interior sample defines a three-node closed-cap fit. An
     // inconsistent root footprint leaves the modeled defect uncertain;
     // no extra lens evaluation or whole-component restart is needed.
     for(int middle=1;middle+1<static_cast<int>(count);++middle) {
      const auto held=closed_short_row_cap(widths,count,middle);
      sensitivity=std::max(sensitivity,held.resolved?std::abs(held.integral-integral):std::abs(integral-used));
     }
     // On a footprint shorter than both four-row endpoint stencils, a
     // three-point fit cannot independently resolve the two limbs. Keep
     // the local zero-halo curvature allowance until more rows exist.
     double curvature=0;
     for(std::size_t k=0;k<count;++k)curvature+=std::abs((k?widths[k-1]:0)-2*widths[k]+(k+1<count?widths[k+1]:0));
     sensitivity=std::max(sensitivity,curvature/24);
    }
    result.uncertainty+=limb_brightness*(raw/24+sensitivity);
    // Closed short-gap model: approximation plus retained-row sensitivity;
    // the three-row case has no independent held-out width sample.
    // A quadrature defect is Integral(model)-Sum(model), not
    // Integral(model)-Sum(data). The latter leaves the retained fit residual
    // sum in the signed estimate while raw/24 only covers its quadrature
    // defect. Preserve the numerical sum and estimate its integration error.
    double model_sum=used;
    if(held_cap_checks){model_sum=0;for(std::size_t k=0;k<count;++k)model_sum+=shape(k);}
    result.signed_residual-=limb_brightness*(integral-model_sum);
    for(int offset=-2;offset<=static_cast<int>(count)+1;++offset) {
     const long double y=static_cast<long double>(ns[i].y)+offset;
     if(y<std::numeric_limits<std::int64_t>::min()||y>std::numeric_limits<std::int64_t>::max())continue;
     result.modeled_second.emplace_back(static_cast<std::int64_t>(y),-limb_brightness*(shape(offset-1)-2*shape(offset)+shape(offset+1)));
    }
    for(std::size_t k=0;k<count;++k)closed[panel[k]]=true;
   }
  }
  for(std::size_t i=0;i<ns.size();++i)for(int d:{0,1}) {
   if((gap&&closed[i])||degrees[i][1-d])continue;
   if(!gap&&(ns[i].y==image.front().y||ns[i].y==image.back().y))continue;
   if(gap) {
    const auto py=ns[i].y+(d? -1:1);
    auto it=std::lower_bound(image.begin(),image.end(),py,[](const Node& a,std::int64_t y){return a.y<y;});bool enclosed=false;
    for(;it!=image.end()&&it->y==py;++it)if(it->lo<=ns[i].lo&&it->hi>=ns[i].hi)enclosed=true;
    if(!enclosed)continue;
   }
   std::array<double,4> values;std::array<std::size_t,4> panel{i,0,0,0};int count=1;
   while(count<4&&degrees[panel[count-1]][d]==1) {
    const auto j=links[panel[count-1]][d];if(degrees[j][1-d]!=1)break;panel[count++]=j;
   }
   const auto local_fallback=[&]() {
    // Reuse the existing zero-halo second-difference indicator on this
    // retained cap footprint. No arbitrary global safety factor or refill.
    double raw=std::abs(ns[panel[0]].value)+std::abs(ns[panel[count-1]].value);
    for(int k=0;k<count;++k)raw+=std::abs((k?ns[panel[k-1]].value:0)-2*ns[panel[k]].value+(k+1<count?ns[panel[k+1]].value:0));
    result.uncertainty+=raw*(gap?limb_brightness:1)/24;result.resolved=false;
   };
   if(count<4) {local_fallback();continue;}
   for(int k=0;k<4;++k)values[k]=ns[panel[k]].value;
   const auto model=outer_endpoint_quadratic_model(values,1,1,1,false);
   if(!model.resolved||model.eta>1) {local_fallback();continue;}
   const double w0=gap?1:ns[panel[0]].w,w1=gap?1:ns[panel[1]].w;
   const auto moments=power_endpoint_quadrature_defects(model.eta,w0,w1);
   const double defect=model.leading_coefficient*moments[0]+model.next_coefficient*moments[1];
   result.signed_residual+=(gap?-limb_brightness:1)*defect;
   const double expansion=2*std::abs(model.next_coefficient)*(model.eta+4)/model.leading_coefficient;
   const double held=model.fourth_relative_defect*(*std::max_element(values.begin(),values.end()))*(gap?limb_brightness:1);
   const double t=model.eta+3;
   result.uncertainty+=std::getenv("LCBININT_DIAGNOSTIC_LEGACY_INTERNAL_REMAINDER")||!std::isfinite(expansion)||expansion>.5?held:
       4*held/(t*t*std::sqrt(t))*std::abs(moments[2]);
   if(std::isfinite(expansion)&&expansion<=.5) {
    const double amplitude=gap?-limb_brightness:1;
    const auto shape=[&](double t) {
        return t>0?amplitude*std::sqrt(t)*model.leading_coefficient*std::sqrt(1+2*model.next_coefficient/model.leading_coefficient*t):0.;
    };
    const int direction=d?1:-1;
    for(int offset=-2;offset<=3;++offset) {
     const long double y=static_cast<long double>(ns[i].y)+direction*offset;
     if(y<std::numeric_limits<std::int64_t>::min()||y>std::numeric_limits<std::int64_t>::max())continue;
     const double t=model.eta+offset;
     result.modeled_second.emplace_back(static_cast<std::int64_t>(y),shape(t-1)-2*shape(t)+shape(t+1));
    }
   }

  }
 };
 process(image,false);if(limb_brightness!=0)process(gaps,true);return result;
}
inline SameGridOuterError same_grid_internal_error(const CartesianRunFillResult& fill,
 std::size_t component,double limb_brightness,const std::vector<double>* weights) {
 return same_grid_internal_error(same_grid_error_nodes(fill,component,weights),limb_brightness);
}
inline SameGridOuterError same_grid_outer_error(const CartesianRunFillResult& fill,
    std::size_t component=std::numeric_limits<std::size_t>::max(),const std::vector<double>* weights=nullptr,double limb_brightness=1,const std::vector<std::size_t>* ordered_runs=nullptr,bool collect_table_density=true,unsigned endpoint_mask=3,bool include_interior=true,double row_flux_correction_scale=0,bool guard_missing_caps=false,bool coarse_shape_check=false) {
    // Keep only row aggregates on the ordinary path. Full interval nodes
    // are materialized from the same ordered index only for a junction.
    std::vector<std::size_t> owned_indices;
    if(!ordered_runs) {
        owned_indices.reserve(fill.runs.size());
        for(std::size_t j=0;j<fill.runs.size();++j)
            if(component==std::numeric_limits<std::size_t>::max()||fill.component_roots[fill.runs[j].component]==component)owned_indices.push_back(j);
        std::sort(owned_indices.begin(),owned_indices.end(),[&](std::size_t a,std::size_t b) {
            const auto& x=fill.runs[a].run;const auto& y=fill.runs[b].run;
            return x.iy!=y.iy?x.iy<y.iy:x.lo<y.lo;
        });
        ordered_runs=&owned_indices;
    }
    struct Row {std::int64_t first;double second,width,weighted,limb_density;int count;};
    std::vector<Row> rows;rows.reserve(ordered_runs->size());
    const bool split_ld=limb_brightness<1&&fill.run_error_fractions.size()==fill.runs.size();
    for(const auto j:*ordered_runs) {
        const auto& r=fill.runs[j].run;
        // The outer rule must see the horizontally corrected row function;
        // otherwise a separate outer correction retains a mixed tensor defect.
        const double area=fill.run_areas[j]+(row_flux_correction_scale!=0?fill.run_flux_errors[j][0]*row_flux_correction_scale:0);
        const double width=split_ld?double(r.hi-r.lo)+fill.run_error_fractions[j][0]+fill.run_error_fractions[j][1]:0;
        const double weighted=area*(weights?(*weights)[j]:1);
        const double density=collect_table_density&&split_ld&&fill.run_flux_errors.size()==fill.runs.size()?fill.run_flux_errors[j][4]:0;
        if(!rows.empty()&&rows.back().first==r.iy) {
            auto& row=rows.back();row.second+=area;row.width+=width;row.weighted+=weighted;row.limb_density+=density;++row.count;
        } else rows.push_back({r.iy,area,width,weighted,density,1});
    }
    if(rows.empty())return {};
    const auto shifted_cap_defect=[&](int end,double w0,double w1)->std::optional<double> {
        if(rows.size()<8)return std::nullopt;
        std::array<double,4> widths;
        for(int k=0;k<4;++k){const auto& row=rows[end?rows.size()-2-k:k+1];widths[k]=split_ld?row.width:row.second;}
        const auto cubic=outer_endpoint_cubic_model(widths,2.5);
        const auto model=cubic.resolved?cubic:outer_endpoint_quadratic_model(widths,1,1,1,false,2.5);
        const double eta=model.eta-1;
        if(!model.resolved || eta<0 || eta>1.5 || !(model.leading_coefficient>0))return std::nullopt;
        const auto moments=power_endpoint_quadrature_defects(eta,w0,w1);
        const double next2=model.squared_cubic_coefficient/(2*model.leading_coefficient)-
            model.next_coefficient*model.next_coefficient/(2*model.leading_coefficient);
        const double amplitude=split_ld?limb_brightness:1;
        double defect=amplitude*(model.leading_coefficient*moments[0]+model.next_coefficient*moments[1]+next2*moments[2]);
        if(split_ld && eta>1e-12) {
            std::array<double,3> ld;
            for(int k=0;k<3;++k){const auto& row=rows[end?rows.size()-1-k:k];ld[k]=row.second-limb_brightness*row.width;}
            const double u=ld[0]/eta,v=ld[1]/(eta+1),b=v-u,a=u-b*eta;
            const double q=(ld[2]/(eta+2)-a-b*(eta+2))/2;
            defect+=a*integer_endpoint_quadrature_defect(eta,w0,w1,1)+b*integer_endpoint_quadrature_defect(eta,w0,w1,2)+
                q*(integer_endpoint_quadrature_defect(eta,w0,w1,3)-(2*eta+1)*integer_endpoint_quadrature_defect(eta,w0,w1,2)+
                   eta*(eta+1)*integer_endpoint_quadrature_defect(eta,w0,w1,1));
        }
        return std::isfinite(defect)?std::optional<double>(defect):std::nullopt;
    };
    const auto fallback=[&]() {
        if(endpoint_mask!=3 || !include_interior)return SameGridOuterError{0,std::numeric_limits<double>::infinity(),false};
        long double raw=0;
        for(std::size_t i=0;i<rows.size();++i) {
            const auto iy=rows[i].first;
            if(iy==std::numeric_limits<std::int64_t>::min()||iy==std::numeric_limits<std::int64_t>::max())return SameGridOuterError{0,std::numeric_limits<double>::infinity(),false};
            const bool prev=i>0&&rows[i-1].first==iy-1,next=i+1<rows.size()&&rows[i+1].first==iy+1;
            raw+=std::abs((prev?rows[i-1].second:0)-2*rows[i].second+(next?rows[i+1].second:0));
            if(!prev)raw+=std::abs(rows[i].second);if(!next)raw+=std::abs(rows[i].second);
        }
        double uncertainty=static_cast<double>(raw/24);
        if(coarse_shape_check)for(int end:{0,1})if(const auto alternative=shifted_cap_defect(end,1,1))
            uncertainty+=std::abs(*alternative);
        return SameGridOuterError{0,uncertainty,false};
    };
    if(rows.size()<8 || static_cast<long double>(rows.back().first)-rows.front().first+1!=rows.size())return fallback();
    const auto weight=[&](std::size_t row){return weights&&rows[row].second>0?rows[row].weighted/rows[row].second:1.;};
    SameGridOuterError result{0,0,true};
    const double phase_limit=std::getenv("LCBININT_DIAGNOSTIC_LEGACY_PHASE_RANGE")?1.:1.5;
    for(int end:{0,1}) {
        if(!(endpoint_mask&(1U<<end)))continue;
        std::array<double,4> values,ld_values{};
        for(int k=0;k<4;++k)values[k]=rows[end?rows.size()-1-k:k].second;
        if(split_ld)for(int k=0;k<4;++k) {
            const double width=rows[end?rows.size()-1-k:k].width;
            ld_values[k]=values[k]-limb_brightness*width;values[k]=width;
        }
        const auto quadratic=outer_endpoint_quadratic_model(values,1,1,1,false,
            guard_missing_caps?std::numeric_limits<double>::infinity():1.5);
        const auto cubic=std::getenv("LCBININT_DIAGNOSTIC_LEGACY_CAP_MODEL")?OuterEndpointModel{}:outer_endpoint_cubic_model(values,phase_limit);
        const auto model=!cubic.resolved?quadratic:cubic;
        const double endpoint_signed_before=result.signed_residual;
        if(guard_missing_caps && model.resolved && model.eta>phase_limit && model.leading_coefficient>0) {
            auto incomplete=fallback();
            // A cap phase beyond one row can mean that no lattice centre
            // enters its narrow tip. Account for the fitted missing segment,
            // rather than treating represented-row curvature as its error.
            const double extent=model.eta-1;
            const double amplitude=split_ld?limb_brightness:1.;
            incomplete.uncertainty+=amplitude*(2./3*std::abs(model.leading_coefficient)*std::pow(extent,1.5)+
                2./5*std::abs(model.next_coefficient)*std::pow(extent,2.5));
            if(split_ld) {
                const double e=model.eta,u=ld_values[0]/e,v=ld_values[1]/(e+1),b=v-u,a=u-b*e;
                incomplete.uncertainty+=std::abs(a)*extent*extent/2+std::abs(b)*extent*extent*extent/3;
            }
            incomplete.missing_cap_rows=true;
            return incomplete;
        }
        if(!model.resolved||model.eta<0||model.eta>phase_limit)return fallback();
        const auto first=end?rows.size()-1:0,second=end?rows.size()-2:1;
        const double w0=weight(first),w1=weight(second);
        if(collect_table_density&&split_ld&&model.eta>1e-12) {
            const double e=model.eta,u=rows[first].limb_density*std::sqrt(e),v=rows[second].limb_density*std::sqrt(e+1),b=v-u,a=u-b*e;
            const double correction=a*inverse_sqrt_endpoint_quadrature_defect(e)+b*power_endpoint_quadrature_defect(e,1,1,0);
            if(std::isfinite(correction)) {
                result.limb_density_correction+=correction;
                // Sensitivity to this locally modeled endpoint correction.
                // No assertion that radial caustic singularities are bounded.
                result.limb_density_uncertainty+=std::abs(correction);
            }
        }
        const double extent=model.eta+3;
        // The fitted sqrt(a*t+b*t^2) expansion must remain inside its
        // convergence range over the retained four-row footprint.
        const double expansion=2*std::abs(model.next_coefficient)*extent/model.leading_coefficient;
        if(!std::isfinite(expansion)||expansion>.5)return fallback();
        const double cubic_term=cubic.resolved?model.squared_cubic_coefficient/(2*model.leading_coefficient):0;
        const double next2=cubic_term-model.next_coefficient*model.next_coefficient/(2*model.leading_coefficient);
        const double amplitude=split_ld?limb_brightness:1;
        const auto moments=power_endpoint_quadrature_defects(model.eta,w0,w1);
        const double next2_defect=amplitude*next2*moments[2];
        result.signed_residual+=next2_defect;
        result.uncertainty+=std::abs(next2_defect)*expansion;
        result.signed_residual+=amplitude*(model.leading_coefficient*moments[0]+model.next_coefficient*moments[1]);
        if(row_flux_correction_scale!=0 && amplitude>0 && cubic.resolved) {
            // Four rows determine the cubic squared-width model. The fifth
            // and sixth are held out; measure the next squared-width term.
            double held_coefficient=0;
            for(int k:{4,5}) {
                const auto& row=rows[end?rows.size()-1-k:k];
                const double actual=split_ld?row.width:row.second,t=model.eta+k;
                const double predicted=model.leading_coefficient*model.leading_coefficient*t+
                    2*model.leading_coefficient*model.next_coefficient*t*t+
                    model.squared_cubic_coefficient*t*t*t;
                held_coefficient=std::max(held_coefficient,std::abs(actual*actual-predicted)/(t*t*t*t));
            }
            result.uncertainty+=amplitude*held_coefficient/(2*model.leading_coefficient)*
                std::abs(power_endpoint_quadrature_defect(model.eta,w0,w1,3));
        }
        if(split_ld&&model.eta>1e-12) {
            // Integrating sqrt(source-depth) across a regular parabolic cap
            // gives a leading t term, not sqrt(t). Fit t+t^2 at known eta.
            const double e=model.eta,u=ld_values[0]/e,v=ld_values[1]/(e+1),b=v-u,a=u-b*e;
            result.signed_residual+=a*integer_endpoint_quadrature_defect(e,w0,w1,1)+b*integer_endpoint_quadrature_defect(e,w0,w1,2);
            const double t=e+3;
            if(row_flux_correction_scale!=0) {
                // Third row fits the next term; fourth remains held out.
                const double quadratic=(ld_values[2]/(e+2)-a-b*(e+2))/2;
                const double correction=quadratic*(integer_endpoint_quadrature_defect(e,w0,w1,3)-
                    (2*e+1)*integer_endpoint_quadrature_defect(e,w0,w1,2)+
                    e*(e+1)*integer_endpoint_quadrature_defect(e,w0,w1,1));
                result.signed_residual+=correction;
                result.uncertainty+=std::abs(correction);
                const double defect=ld_values[3]-a*t-b*t*t-quadratic*t*(t-e)*(t-e-1);
                const double next=defect/(t*(t-e)*(t-e-1)*(t-e-2));
                const double moment=integer_endpoint_quadrature_defect(e,w0,w1,4)-
                    (3*e+3)*integer_endpoint_quadrature_defect(e,w0,w1,3)+
                    (3*e*e+6*e+2)*integer_endpoint_quadrature_defect(e,w0,w1,2)-
                    e*(e+1)*(e+2)*integer_endpoint_quadrature_defect(e,w0,w1,1);
                result.uncertainty+=std::abs(next*moment);
            } else {
                const double defect=ld_values[3]-a*t-b*t*t;
                result.uncertainty+=std::abs(defect)/(t*t*t)*std::abs((std::pow(e,4)-2*std::pow(e,3)+e*e-1./30)/4);
            }
        }
        const double scale=*std::max_element(values.begin(),values.end()),t=model.eta+3;
        // The held-out fourth row estimates the next t^(5/2) term. Heuristic
        // remainder factor; never a proof or calibration coverage constant.
        result.uncertainty+=4*amplitude*quadratic.fourth_relative_defect*scale/(t*t*std::sqrt(t))*
            std::abs(power_endpoint_quadrature_defect(model.eta,1,1,2));
        if(cubic.resolved&&quadratic.eta>=0&&quadratic.eta<=phase_limit) {
            const auto qm=power_endpoint_quadrature_defects(quadratic.eta,w0,w1);
            double alternative=amplitude*(quadratic.leading_coefficient*qm[0]+quadratic.next_coefficient*qm[1]-quadratic.next_coefficient*quadratic.next_coefficient/(2*quadratic.leading_coefficient)*qm[2]);
            if(split_ld&&quadratic.eta>1e-12) {
                const double e=quadratic.eta,u=ld_values[0]/e,v=ld_values[1]/(e+1),b=v-u,a=u-b*e;
                alternative+=a*integer_endpoint_quadrature_defect(e,w0,w1,1)+b*integer_endpoint_quadrature_defect(e,w0,w1,2);
            }
            const double change=alternative-(result.signed_residual-endpoint_signed_before);
            // Center the two retained-row models and keep their half spread.
            // This represents shape sensitivity, not a certified error bound.
            result.signed_residual+=.5*change;result.uncertainty+=.5*std::abs(change);
        }
        if(coarse_shape_check) {
            // Shift the retained stencil by one row; transport its root back
            // to the same physical cap. This detects coarse-fit sensitivity
            // without another image-plane evaluation or a convergence ladder.
            const double current=result.signed_residual-endpoint_signed_before;
            const auto alternative=shifted_cap_defect(end,w0,w1);
            result.uncertainty+=alternative?std::abs(*alternative-current):std::abs(current);
        }
    }
    result.cap_signed=result.signed_residual;result.cap_remainder=result.uncertainty;
    if(!include_interior)return result;
    SameGridOuterError internal{0,0,true};
    if(fill.run_error_fractions.size()==fill.runs.size()&&std::any_of(rows.begin(),rows.end(),[](const Row& row){return row.count>1;})) {
        internal=same_grid_internal_error(same_grid_error_nodes(fill,component,weights,ordered_runs),limb_brightness,row_flux_correction_scale!=0);
        std::sort(internal.modeled_second.begin(),internal.modeled_second.end());
    }
    std::vector<double> saddle_second;
    if(split_ld && std::getenv("LCBININT_DIAGNOSTIC_DISABLE_SADDLE")==nullptr) {
        const bool paired_enabled=std::getenv("LCBININT_DIAGNOSTIC_DISABLE_PAIRED_SADDLE")==nullptr;
        struct SaddlePiece {std::size_t junction;double signed_residual,remainder;bool active;};
        std::vector<SaddlePiece> old_pieces;
        for(std::size_t i=4;i+5<rows.size();++i) {
            if(!((rows[i].count==1&&rows[i+1].count==2)||(rows[i].count==2&&rows[i+1].count==1)))continue;
            bool isolated=true;for(std::size_t j=i-3;j<=i+3;++j)if(j!=i&&rows[j].count!=rows[j+1].count)isolated=false;
            if(row_flux_correction_scale!=0) {
                const auto lo=i>7?i-7:0,hi=std::min(rows.size()-2,i+7);
                for(auto j=lo;j<=hi;++j)if(j!=i&&rows[j].count!=rows[j+1].count)isolated=false;
            }
            if(!isolated)continue;
            const int direction=rows[i+1].count==2?1:-1;
            const std::size_t gap_start=direction==1?i+1:i;
            std::array<double,4> gaps{};bool found=true;
            for(int k=0;k<4;++k) {
                const auto yy=rows[gap_start+direction*k].first;
                auto it=std::lower_bound(ordered_runs->begin(),ordered_runs->end(),yy,[&](std::size_t j,std::int64_t y){return fill.runs[j].run.iy<y;});
                if(it==ordered_runs->end()){found=false;break;}
                const auto first=*it++;if(it==ordered_runs->end()||fill.runs[*it].run.iy!=yy){found=false;break;}
                const auto second=*it;
                gaps[k]=(double(fill.runs[second].run.lo)-fill.run_error_fractions[second][0])-(double(fill.runs[first].run.hi)+fill.run_error_fractions[first][1]);
                if(!(gaps[k]>0)){found=false;break;}
            }
            if(!found)continue;
            const auto geometry=outer_endpoint_quadratic_model(gaps,1,1,1,false);
            if(!geometry.resolved||geometry.eta<=0||geometry.eta>=1)continue;
            const double phase=direction==1?1-geometry.eta:geometry.eta;
            std::array<double,8> values{};
            for(int k=0;k<8;++k)values[k]=rows[i+k-3].second-limb_brightness*rows[i+k-3].width;
            const auto model=fit_saddle_rows(values,phase);if(!model.resolved)continue;
            if(saddle_second.empty())saddle_second.resize(rows.size());
            // The four junction rows use the same fitted LD row function;
            // uniform-width singular terms remain in the existing model.
            for(int k=2;k<=5;++k)saddle_second[i+k-3]=model.predicted[k-1]-2*model.predicted[k]+model.predicted[k+1];
            result.signed_residual+=model.signed_residual;
            // The junction residual already counts second differences of
            // fit disagreement. Retain its quadrature remainder and
            // coefficient sensitivity instead of the full row-value error.
            const double remainder=std::getenv("LCBININT_DIAGNOSTIC_LEGACY_SADDLE_REMAINDER")?model.held_defect:model.quadrature_remainder;
            result.uncertainty+=remainder;
            if(paired_enabled)old_pieces.push_back({i,model.signed_residual,remainder,true});
        }        std::vector<bool> paired_covered;
        if(paired_enabled)for(std::size_t start=6;start+9<rows.size();++start) {
            if(rows[start-1].count!=1||rows[start].count!=2)continue;
            std::size_t count=0;while(count<8&&start+count<rows.size()&&rows[start+count].count==2)++count;
            // Closed-gap fits remain outside the outer cap panels.
            if(count<3||count>7||start+count+5>=rows.size())continue;
            bool isolated=true;for(int j=1;j<=3;++j)if(rows[start-j].count!=1||rows[start+count+j-1].count!=1)isolated=false;
            if(!isolated)continue;
            std::array<double,7> gaps{},gap_low{},gap_high{};bool found=true;
            for(std::size_t k=0;k<count;++k) {
                const auto yy=rows[start+k].first;
                auto it=std::lower_bound(ordered_runs->begin(),ordered_runs->end(),yy,[&](std::size_t j,std::int64_t y){return fill.runs[j].run.iy<y;});
                if(it==ordered_runs->end()){found=false;break;}
                const auto first=*it++;if(it==ordered_runs->end()||fill.runs[*it].run.iy!=yy){found=false;break;}
                const auto second=*it;
                gap_low[k]=double(fill.runs[first].run.hi)+fill.run_error_fractions[first][1];
                gap_high[k]=double(fill.runs[second].run.lo)-fill.run_error_fractions[second][0];
                gaps[k]=gap_high[k]-gap_low[k];
            }
            if(!found)continue;
            const auto enclosed=[&](std::size_t row,double lo,double hi) {
                const auto yy=rows[row].first;
                auto it=std::lower_bound(ordered_runs->begin(),ordered_runs->end(),yy,[&](std::size_t j,std::int64_t y){return fill.runs[j].run.iy<y;});
                if(it==ordered_runs->end())return false;
                const auto j=*it;const auto& run=fill.runs[j].run;
                return double(run.lo)-fill.run_error_fractions[j][0]<=lo&&double(run.hi)+fill.run_error_fractions[j][1]>=hi;
            };
            if(!enclosed(start-1,gap_low[0],gap_high[0])||!enclosed(start+count,gap_low[count-1],gap_high[count-1]))continue;
            const auto geometry=closed_short_row_cap(gaps,count);if(!geometry.resolved)continue;
            std::array<double,13> values{};const int footprint=static_cast<int>(count)+6;
            for(int k=0;k<footprint;++k)values[k]=rows[start+k-3].second-limb_brightness*rows[start+k-3].width;
            const auto model=fit_paired_saddle_rows(values,footprint,3+geometry.centre-geometry.radius,3+geometry.centre+geometry.radius);
            if(!model.resolved)continue;
            double old_signed=0,old_remainder=0;
            for(const auto& piece:old_pieces)if(piece.active&&(piece.junction==start-1||piece.junction==start+count-1)) {
                old_signed+=piece.signed_residual;old_remainder+=piece.remainder;
            }
            double baseline=old_remainder,candidate=model.quadrature_remainder,smooth_remainder=0;bool overlap=false;
            for(int k=1;k+1<footprint;++k) {
                const auto j=start+k-3;
                if(!paired_covered.empty()&&paired_covered[j])overlap=true;
                bool junction=false;for(std::size_t h=j-2;h<j+2;++h)if(rows[h].count!=rows[h+1].count)junction=true;
                if(!junction)continue;
                smooth_remainder+=7*std::abs(model.regular_fourth)/5760;
                double geometric=0;
                auto it=std::lower_bound(internal.modeled_second.begin(),internal.modeled_second.end(),rows[j].first,[](const auto& a,std::int64_t y){return a.first<y;});
                for(;it!=internal.modeled_second.end()&&it->first==rows[j].first;++it)geometric+=it->second;
                const double residual=rows[j-1].second-2*rows[j].second+rows[j+1].second-geometric;
                baseline+=std::abs(residual-(saddle_second.empty()?0:saddle_second[j]))/24;
                candidate+=std::abs(residual-(model.predicted[k-1]-2*model.predicted[k]+model.predicted[k+1]))/24;
            }
            // Compare indicators from retained rows, never actual/reference
            // errors. An enrichment that increases uncertainty adds no value.
            candidate+=smooth_remainder;
            if(overlap||!std::isfinite(candidate)||candidate>=baseline)continue;
            if(saddle_second.empty())saddle_second.resize(rows.size());
            if(paired_covered.empty())paired_covered.resize(rows.size());
            for(int k=1;k+1<footprint;++k) {
                const auto j=start+k-3;
                saddle_second[j]=model.predicted[k-1]-2*model.predicted[k]+model.predicted[k+1];paired_covered[j]=true;
            }
            result.signed_residual+=model.signed_residual-old_signed;result.uncertainty+=model.quadrature_remainder+smooth_remainder-old_remainder;
            for(auto& piece:old_pieces)if(piece.junction==start-1||piece.junction==start+count-1)piece.active=false;
        }

    }
    std::size_t modeled_position=0;
    long double smooth_defect=0;
    for(std::size_t i=4;i+4<rows.size();++i) {
        const double second=rows[i-1].second-2*rows[i].second+rows[i+1].second;
        const double fourth=rows[i-2].second-4*rows[i-1].second+6*rows[i].second-4*rows[i+1].second+rows[i+2].second;
        // Changes of run multiplicity flag an internal gap/fold endpoint.
        // Smooth signed fourth defects telescope; taking abs at every row
        // instead would mistake cancelling boundary-location noise for bias.
        bool junction=false;
        for(std::size_t j=i-2;j<i+2;++j)if(rows[j].count!=rows[j+1].count)junction=true;
        if(junction) {
            double modeled=0;
            while(modeled_position<internal.modeled_second.size()&&internal.modeled_second[modeled_position].first<rows[i].first)++modeled_position;
            while(modeled_position<internal.modeled_second.size()&&internal.modeled_second[modeled_position].first==rows[i].first)modeled+=internal.modeled_second[modeled_position++].second;
            result.uncertainty+=std::abs(second-modeled-(saddle_second.empty()?0:saddle_second[i]))/24;
        }
        else smooth_defect+=fourth;
    }
    result.smooth_remainder=7*std::abs(static_cast<double>(smooth_defect))/5760;
    result.uncertainty+=7*std::abs(static_cast<double>(smooth_defect))/5760;
    result.signed_residual+=internal.signed_residual;result.uncertainty+=internal.uncertainty;result.resolved=result.resolved&&internal.resolved;
    if(row_flux_correction_scale!=0 && include_interior && (limb_brightness>0 || guard_missing_caps)) {
        // Gap graphs need not form a closed cap: one of the two image arms
        // can end before the gap closes. Row multiplicity still identifies
        // the short critical footprint where the endpoint stencils overlap.
        // Keep its retained zero-halo allowance; replace rather than add to
        // the already accounted junction uncertainty.
        double critical_allowance=0;
        for(std::size_t start=1;start<rows.size();++start) {
            if(rows[start-1].count!=1 || rows[start].count!=2)continue;
            std::size_t count=0;while(start+count<rows.size() && rows[start+count].count==2)++count;
            if(count<1 || (!guard_missing_caps && count<3) || count>7 || start+count>=rows.size() || rows[start+count].count!=1)continue;
            std::array<double,7> gaps{};bool complete=true;
            for(std::size_t k=0;k<count;++k) {
                const auto y=rows[start+k].first;
                auto it=std::lower_bound(ordered_runs->begin(),ordered_runs->end(),y,[&](std::size_t j,std::int64_t yy){return fill.runs[j].run.iy<yy;});
                if(it==ordered_runs->end()){complete=false;break;}
                const auto first=*it++;
                if(it==ordered_runs->end() || fill.runs[*it].run.iy!=y){complete=false;break;}
                const auto second=*it;
                gaps[k]=double(fill.runs[second].run.lo)-fill.run_error_fractions[second][0]-double(fill.runs[first].run.hi)-fill.run_error_fractions[first][1];
                if(!(std::isfinite(gaps[k])&&gaps[k]>0)){complete=false;break;}
            }
            if(!complete)continue;
            double curvature=0;
            for(std::size_t k=0;k<count;++k)curvature+=std::abs((k?gaps[k-1]:0)-2*gaps[k]+(k+1<count?gaps[k+1]:0));
            critical_allowance+=limb_brightness*curvature/24;
            if(guard_missing_caps && count<3) {
                // One or two samples cannot identify the two gap caps and
                // their amplitude. Retain the represented gap-area scale
                // until refinement supplies a usable local shape stencil.
                for(std::size_t k=0;k<count;++k)critical_allowance+=gaps[k];
            }
        }
        result.uncertainty=std::max(result.uncertainty,result.cap_remainder+result.smooth_remainder+critical_allowance);
    }
    return result;
}

} // namespace lcbinint::magnification::detail
