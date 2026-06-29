# Overlap & Component Union Redesign

## Current State (after c49afa1 fix)

**Short-term fix applied:**
- Prevent double-subtraction of same component
- Quick check: `subtracted_previous_overlap[seed_index]`
- Max error: 9.5844% → 0.0252% ✓

**But design is still incomplete:**
- Component tracking is seed-index based, not component-ID based
- Inclusion-exclusion is approximate, not rigorous
- Fold parity and overlap handling mixed together
- No proper component union datastructure

## Problem Cases Still Vulnerable

1. **Complex fold overlaps**: When multiple fold images heavily overlap near caustic cusps
2. **Grid phase effects**: Different bin counts cause components to split/merge differently
3. **Parity transitions**: Fold regions near caustic branches where parity flips

Current `subtracted_previous_overlap[seed_index]` is a band-aid. Real solution needs proper component tracking.

## Proposed Design

### Core Change: Component-Based Union Tracking

**Instead of:**
```cpp
std::vector<double> areaimage(images.size(), 0.0);  // per seed
std::vector<int> subtracted_previous_overlap(images.size(), 0);  // per seed ← band-aid
```

**Do:**
```cpp
struct ProcessedComponent {
    int component_id;                    // unique per connected region
    std::vector<size_t> seed_indices;    // which seeds make up this component
    double area;                          // total area of this component
    std::vector<std::pair<int,int>> rows_bounds;  // which y-rows are covered
    int fold_parity;                     // +1 or -1 for fold disambiguation
};

std::vector<ProcessedComponent> processed_components;
std::map<size_t, int> seed_to_component_id;  // which component owns each seed
```

### Union-Based Area Calculation

**Current (buggy):**
```cpp
area = 0.0;
for (each seed) {
    area += areaimage[seed];
    if (overlaps) {
        area -= areaimage[other];  // ← what if other was already subtracted?
    }
}
```

**Proposed (correct):**
```cpp
// Track processed components, not individual seeds
std::set<int> already_counted_components;

for (each new seed) {
    int component = assign_to_or_create_component(seed);
    
    // Union with overlapping past components
    for (each overlapping past_component) {
        if (already_counted_components.count(past_component)) {
            // Already in union, skip
            continue;
        }
        // Use inclusion-exclusion properly
        compute_component_union(component, past_component);
        already_counted_components.insert(past_component);
    }
    
    already_counted_components.insert(component);
}

area = sum of all component areas in union
```

### Separation of Concerns

**1. Component identification** (new responsibility):
```cpp
int assign_to_or_create_component(
    size_t seed_index,
    const std::vector<ProcessedComponent>& existing_components,
    bool fold_parity_sign)
{
    // Check if seed overlaps with existing components
    // If yes: merge/expand component
    // If no: create new component
    // Return component_id
}
```

**2. Overlap detection** (clarified):
```cpp
bool components_overlap(
    const ProcessedComponent& comp_a,
    const ProcessedComponent& comp_b)
{
    // Check row_bounds, then detailed flood-fill overlap
    // Pure geometry, no bookkeeping
}
```

**3. Fold parity guard** (isolated):
```cpp
bool is_valid_fold_member(
    size_t image_index,
    const ProcessedComponent& fold_component)
{
    // Only check: does this image have correct parity for this fold?
    // Pure Jacobian/parity logic
}
```

**4. Area union** (systematic):
```cpp
void add_component_to_union(
    ProcessedComponent& union_component,
    const ProcessedComponent& new_component)
{
    // Proper inclusion-exclusion for areas
    // Merge row_bounds
    // Update component_id mapping
}
```

## Implementation Roadmap

### Phase 1: Component Datastructure (Safety First)
- Add `ProcessedComponent` struct
- Add `seed_to_component_id` mapping
- Refactor to track components in parallel with current code
- **Keep current logic working** (dual-track)
- Tests should still pass

### Phase 2: Component-Based Skip Logic  
- Replace: `if (overlap[image_index] == 1) skip;`
- With: check against `seed_to_component_id`
- Verify same behavior

### Phase 3: Component-Based Subtraction
- Replace: `area -= areaimage[other]`
- With: `union_with(processed_components[other_component_id])`
- Remove `subtracted_previous_overlap` hack

### Phase 4: Parity Separation
- Move parity logic to `is_valid_fold_member()`
- Remove parity checks from overlap logic
- Cleaner responsibility flow

### Phase 5: Rigorous Union
- Implement proper inclusion-exclusion
- Handle complex multi-component overlaps
- Test edge cases: cusps, wide caustics, grid phase variations

## Validation Strategy

1. **Regression**: All current tests must pass at each phase
2. **Edge cases**: Add tests for:
   - Wide caustics (s < 1.0, rho >= 0.01)
   - Cusps (high curvature caustics)  
   - Grid phase sensitivity (fixed bins 50, 125, 200, 350)
3. **Robustness**: Run adaptive sweep, check for new regressions

## Expected Benefits

- **Correctness**: Proper inclusion-exclusion instead of heuristics
- **Maintainability**: Clear separation: component ID management, overlap detection, parity logic, area union
- **Robustness**: No more grid-phase dependent bugs
- **Future-proofing**: Can handle complex caustic structures safely

## Timeline

- **Phase 1-2**: ~4-6 hours (safe refactor, parallel structure)
- **Phase 3-4**: ~3-4 hours (logic replacement)
- **Phase 5**: ~2-3 hours (rigorous union, edge cases)
- **Testing**: ~2 hours (validation at each phase)

Total: ~2-3 working days for complete redesign

## Known Risks

- **Phase 1-2 duplication**: Code bloat temporarily, must clean up in Phase 3
- **Performance**: New datastructures might be slower, optimize if needed
- **Subtle regressions**: Grid-phase variations might hide bugs, need thorough testing

Mitigation: **Incremental phases with full testing at each step**, not a big bang rewrite.
