#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace lcbinint::magnification::detail {

struct CartesianLatticeSeed {
    std::int64_t ix = 0;
    std::int64_t iy = 0;
};

struct CartesianRun {
    std::int64_t iy = 0;
    std::int64_t lo = 0;
    std::int64_t hi = -1;
};

struct CartesianBoundaryContribution {
    double area = 0.0;
    int edges = 0;
    bool valid = true;
};

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
    std::int64_t rows_with_multiple_runs = 0;
    std::size_t maximum_runs_in_row = 0;
    std::int64_t seeds_offered = 0;
    std::int64_t unique_seed_cells = 0;
    std::int64_t duplicate_seeds_avoided = 0;
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

struct CartesianRunFillResult {
    CartesianRunFillStatus status = CartesianRunFillStatus::ok;
    std::vector<CartesianRunRecord> runs;
    std::vector<std::size_t> component_roots;
    std::vector<int> component_rows;
    std::vector<long double> component_areas;
    std::vector<std::int64_t> component_cells;
    std::vector<int> component_boundary_edges;
    CartesianRunFillCounters counters;

    bool ok() const noexcept { return status == CartesianRunFillStatus::ok; }
};

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
          typename BoundaryWeight>
CartesianRunFillResult fill_cartesian_runs(
    std::vector<CartesianLatticeSeed> seeds,
    Classify&& classify,
    IsInside&& is_inside,
    CellWeight&& cell_weight,
    BoundaryWeight&& boundary_weight,
    CartesianRunFillLimits limits = {})
{
    using namespace cartesian_run_fill_detail;

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
                                  std::optional<CellState> left_hint)
        -> std::optional<std::size_t> {
        if (result.runs.size() >= limits.maximum_runs) {
            fail(CartesianRunFillStatus::run_budget_exhausted);
            return std::nullopt;
        }

        std::int64_t lo = seed_ix;
        std::int64_t hi = seed_ix;
        CellState left_inside = seed_state;
        CellState right_inside = seed_state;
        std::optional<CellState> left_outside;
        std::optional<CellState> right_outside;
        double area = cell_weight(seed_ix, iy, seed_state);
        if (!std::isfinite(area)) {
            fail(CartesianRunFillStatus::invalid_contribution);
            return std::nullopt;
        }
        std::int64_t cells = 1;

        std::int64_t ix = 0;
        if (!decrement(lo, ix)) {
            fail(CartesianRunFillStatus::coordinate_overflow);
            return std::nullopt;
        }
        if (left_hint.has_value()) {
            left_outside = std::move(left_hint);
        } else {
            while (true) {
                const auto state = evaluate(ix, iy);
                if (!state.has_value()) {
                    return std::nullopt;
                }
                if (!is_inside(*state)) {
                    left_outside = *state;
                    break;
                }
                lo = ix;
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
            const auto state = evaluate(ix, iy);
            if (!state.has_value()) {
                return std::nullopt;
            }
            if (!is_inside(*state)) {
                right_outside = *state;
                break;
            }
            hi = ix;
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
        const CartesianBoundaryContribution boundary = boundary_weight(
            run, left_inside, *left_outside, right_inside, *right_outside);
        if (!boundary.valid || !std::isfinite(boundary.area)) {
            fail(CartesianRunFillStatus::invalid_contribution);
            return std::nullopt;
        }
        area += boundary.area;

        const std::size_t run_index = result.runs.size();
        result.runs.push_back({run, component});
        components.accumulate(component, area, cells, boundary.edges);
        auto& inside_row = rows.get(iy).inside;
        const std::size_t prior_intervals = inside_row.size();
        inside_row.insert({lo, hi, run_index});
        if (prior_intervals == 1) {
            ++result.counters.rows_with_multiple_runs;
        }
        result.counters.maximum_runs_in_row = std::max(
            result.counters.maximum_runs_in_row, inside_row.size());
        frontier.push_back(run_index);
        ++result.counters.maximal_runs_discovered;
        ++result.counters.row_intervals_stored;
        return run_index;
    };

    // Register every seed run before consuming any frontier.  Traversal
    // therefore cannot decide whether a later certified seed exists.
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
                std::nullopt).has_value()) {
            return result;
        }
    }

    const auto drain_frontier = [&]() {
      while (frontier_head < frontier.size()) {
        const std::size_t run_index = frontier[frontier_head++];
        ++result.counters.frontier_intervals_popped;
        const CartesianRun run = result.runs[run_index].run;
        const std::size_t component = result.runs[run_index].component;

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
                    seed_ix, next_iy, *state, root, previous_outside);
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

    if (!drain_frontier()) {
        return result;
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
        row.inside.visit_all([&](const InsideInterval& interval) {
            const std::size_t root =
                result.runs[interval.run_index].component;
            if (std::find(
                    row_components.begin(), row_components.end(), root) ==
                row_components.end()) {
                row_components.push_back(root);
                ++result.component_rows[root];
            }
        });
    });
    return result;
}

} // namespace lcbinint::magnification::detail
