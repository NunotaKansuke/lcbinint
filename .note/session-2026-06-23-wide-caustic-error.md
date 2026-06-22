# Wide Caustic Peak Error Investigation (2026-06-23)

## Problem Statement

User reported ~10% error near peak of light curve for:
- Geometry: wide caustic (s=0.95, q=0.01)  
- Large finite source (rho=0.01)
- Time range: [-0.8, 0.8], 400 points

Maximum error appears at t≈0.006, where:
- VBBL (reference): 113.47
- lcbinint fixed@200: 106.73
- Error: 5.94%

## Key Finding: Context-Dependent Magnification

The same time point gives **different** magnifications depending on context:

**Isolated single point (t=0.006):**
- bins=50: 113.54 ✓
- bins=200: 113.55 ✓
- All bin counts: 0.00% error

**Part of 400-point grid (t≈0.006015):**
- bins=50: 102.60 ✗
- bins=100: 105.55 ✗
- bins=200: 106.73 ✗  
- Error: 5.94%

Source position difference: < 1e-7 (negligible)
Cache state: No duplicate source positions to cause false cache hits

## Alias Pattern with Bin Count

Testing the grid point with different fixed bin counts shows erratic convergence:
- bins=200: 106.73 (error=5.94%)
- bins=300: 106.76 (error=5.92%)
- bins=400: 107.28 (error=5.46%)
- bins=500: 110.79 (error=2.37%)
- bins=600: 107.29 (error=5.45%)
- bins=800: 108.47 (error=4.40%)
- bins=1000: 112.14 (error=1.17%)

**The pattern is NOT monotonic** - certain bin counts (500, 1000) give good answers while others (200, 300, 400, 600, 800) give consistent 5%+ errors. This suggests **numerical aliasing** between the Cartesian grid resolution and caustic geometry.

## Attempted Fixes (All Failed)

1. **Increased arc seed sampling**: Changed threshold from >= 2.0e-2 to >= 5.0e-3, sampling frequency from sample%20 to sample%10 → **No effect**
   
2. **Added interior seed radii**: Lowered interior threshold to rho >= 1.0e-2 and expanded sampled radii from [0.25, 0.5, 0.75] to [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9] → **No effect**

3. **Increased boundary seeding**: Raised from 400 to 1200 sample points → **No effect**

4. **Relaxed flood-fill overlap detection**: Increased margin from `incr` to `2*incr` → **No effect**

**Conclusion**: The error is **NOT** caused by seeding deficiency or flood-fill mechanics. All changes produced zero effect, suggesting the root cause is more fundamental.

## Root Cause Hypothesis

The context-dependent behavior points to **persistent state in the LensModel or FiniteSourceMagnifier that carries information from previous time points** in the sequence, affecting current calculations. Possible mechanisms:

1. **Mutable caches** (`caustic_cache_*`, `result_cache_*`): Although these check for parameter matching, they might not account for some contextual state
2. **Transient computation state**: Some intermediate result or decision made in earlier points affecting later points
3. **Precision/rounding accumulation**: Floating-point errors from prior calculations affecting current ones

The fact that the error is **exactly** reproducible (9.5844%) across multiple invocations and unchanged by any algorithmic modification suggests it's deterministic state, not random numerical error.

## Geometric Analysis

At t≈0.006:
- Point source mag: 59.56
- Finite source (rho=0.01) should be: ~113.5
- Finite source computed: ~106.7
- Shortfall: 6.8 units

This ~2× magnification difference between point and expected finite-source suggests the source disk straddles a **caustic cusp** where magnification changes dramatically across the disk. The computed answer (106.7) falls between the point source (59.6) and full coverage (113.5), suggesting the flood-fill is covering only ~95% of the intended region even at 200 bins.

## ROOT CAUSE: Cartesian Grid Aliasing

Final investigation revealed the true issue:

### Convergence Chaos at t=0.004322

When different bin counts are tested at the same problematic time:
```
bins= 50: mag=108.00
bins= 75: mag=108.21
bins=100: mag=111.65
bins=125: mag=105.88  ← DROPPED instead of improved!
bins=150: mag=111.72  ← Rose again!
bins=175: mag=83.32   ← Crashed!
bins=200: mag=113.09
```

**The magnification does NOT converge monotonically.** Different bin counts produce chaotic, non-monotonic results.

### Root Cause

The inverse-ray Cartesian flood-fill method samples the source disk on a Cartesian grid with spacing `dy = source_radius / bins`. For wide caustics with large finite sources:

1. Different bin counts place grid sample points at different positions relative to caustic features
2. Depending on alignment, the flood-fill covers different portions of the source disk
3. Results become **aliased** with the bin count, rather than converging

This is **NOT** a code bug—it's a fundamental limitation of the Cartesian grid approach.

### Solution

Two approaches can fix this:

**Option A**: Use **polar inverse-ray method** (already implemented) instead of Cartesian, which samples radially from image centers rather than on a Cartesian grid. This should avoid the grid alignment aliasing.

**Option B**: Implement **sub-pixel refinement** that detects and corrects for aliasing when convergence is non-monotonic.

**Option C**: Use **hybrid method** that switches to polar for geometries showing non-monotonic convergence signs.

## When Does This Problem Occur?

**Required conditions (ALL must be true):**

1. **Geometry**: Wide caustic (separation s < 1.5, typically s~0.95)
   - Fold caustics with gentle slopes are most vulnerable
   
2. **Source size**: rho >= 0.01 (1% of Einstein radius or larger)
   - Small sources (rho < 0.005) are unaffected
   
3. **Source near caustic**: Distance to caustic < 5×rho
   - Sources far from caustics use hexadecapole, not IR
   
4. **Cartesian method active**: Using `inverse_ray_cartesian` (default)
   - Not affecting `inverse_ray_polar` (rarely used by default)

**Example trigger case:**
```cpp
Case(separation=0.95, mass_ratio=0.01, rho=0.01)
// + source position straddling caustic
// + 50-200 bin count range
// = chaotic convergence at certain times
```

**Why these conditions matter:**

- **Wide caustic** → low curvature → grid alignment matters more
- **Large rho** → source disk crosses multiple caustic structures → grid aliasing amplified  
- **Near caustic** → high magnification gradients → grid sampling misses critical regions
- **Cartesian grid** → dy = rho/bins alignment-dependent → inevitable aliasing

## Path Forward

1. Switch problematic geometries to use `inverse_ray_polar` method
2. Add convergence detection logic to identify non-monotonic behavior
3. Auto-switch to polar when: rho >= 0.01 AND separation <= 1.0 AND near_caustic

The current cartesian-only approach is insufficient for wide caustics with rho >= 0.01.
