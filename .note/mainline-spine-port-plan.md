# Mainline Spine Port Plan

## Status

Stable branch commit:

```text
eb2c08e Add guarded mode7 spine scan
```

This commit is the current source of truth for the guarded image-spine
implementation. The mainline worktree should not be changed until the port is
explicitly started.

Current stable worktree still has unrelated notebook changes under
`example/compare-vbbl/`. Do not include those in the spine port.

## Why Not Merge The Stable Branch Directly

The stable branch contains earlier mode7/frontier/timing work that does not map
cleanly onto current mainline. A direct merge conflicts in:

```text
include/lcbinint/lcbinint.h
python/lcbinint_pybind.cpp
src/lcbinint/magnification/finite_source_magnifier.cpp
src/lcbinint/magnification/finite_source_magnifier.hpp
src/lcbinint/model/lens_model.cpp
src/lcbinint/model/lens_parameters.cpp
src/lcbinint/model/lens_parameters.hpp
tests/regression/test_vbm_consistency.py
```

Mainline already has a cleaner finite-source API:

```text
finite_mode = 1: Cartesian inverse ray
finite_mode = 2: polar inverse ray with cache
```

The spine scan should therefore be ported as a new finite-source kernel rather
than by importing the whole legacy `FINITE=7` history.

## Minimal Mainline Target

Add a third explicit finite-source kernel:

```text
finite_mode = 3: experimental image-spine guided scan
```

Keep the existing behavior unchanged for:

```text
finite_mode <= 0
finite_mode = 1
finite_mode = 2
point-source shortcut
hexadecapole shortcut
adaptive hex guard
```

Mainline naming can be:

```cpp
FiniteSourceMethod::inverse_ray_spine
```

and method-name output:

```text
inverse_ray_spine
```

## Code To Port

From stable `src/lcbinint/magnification/finite_source_magnifier.cpp`, port only
the guarded spine pieces:

```text
kLocal7Spine*
Local7Frame
Local7SpinePoint
Local7SpineEligibility
local7_derivatives_binary
local7_make_frame
local7_apply_inverse_jacobian
local7_spine_eligibility
local7_spine_step
local7_spine_frame_safe
local7_spine_try_step
local7_build_spine_direction
local7_spine_integrate_normals
local7_spine_area_binary
```

Do not port the old square-ring code. Do not port the unsafe non-caustic
high-magnification broadening attempt.

Mainline already has these equivalents and should reuse them:

```text
BinaryLensMapper
make_binary_lens_mapper
map_binary_lens_real
mapped_binary_lens_distance2
binary_jacobian
binary_jacobian_sign
legacy_augmented_image_seeds
legacy_imagearea4_binary
source_flux
legacy_limb_brightness
```

## Selection Policy

For the first mainline patch, spine should stay conservative:

1. Run the normal mainline seed generation.
2. Count caustic-born seeds from the augmented seed path when available.
3. Use spine only when caustic-born fold-pair eligibility passes:
   - caustic-born branch exists,
   - seed has high area Jacobian,
   - seed determinant is small but finite,
   - opposite-parity nearest partner exists,
   - the mutual/ambiguity guard passes.
4. If eligibility or spine construction fails, use the whole-call Cartesian
   kernel for that call.

Do not mix arbitrary mode4 branch areas with spine branch areas until branch
overlap accounting is made explicit. The stable patch only treats the accepted
fold pair as a local union and marks the paired seed as overlapped.

## Tests To Port

Add regression coverage using mainline public API rather than legacy direct
helpers:

```python
lcbinint.Options(center_of_mass=1, mode=3, source_bins=300, ...)
```

Required cases:

1. Wide caustic-born fold pair:
   - `s = 1.4`
   - `q = 0.4`
   - source near `(-0.24854045037531268, -0.15)`
   - `rho = 1e-4`
   - `source_bins = 300`
   - compare mode3 against mode1, tolerance about `2e-4`.
2. Close equal-mass high-magnification non-caustic guard:
   - `s = 0.6`
   - `q = 1.0`
   - source near `(-0.09201927708355606, 0.029966615534330332)`
   - `rho = 0.003`
   - `source_bins = 60`
   - mode3 should match mode1 because spine should not be enabled.

If a direct finite-source helper is added later, use it for point-level tests.
For the minimal mainline patch, avoid adding new public debug helpers unless the
port needs them.

## Validation Before Mainline Merge

Use a clean worktree or throwaway branch from mainline. Do not use the dirty
notebook worktree for the actual port.

```bash
cmake --build build
ctest --test-dir build --output-on-failure

LD_LIBRARY_PATH=/rogue1_8/nunota/local/gsl/lib python -S - <<'PY'
import sys
sys.path.insert(0, "build")
sys.path.append("/home/nunota/.miniconda3/envs/myenv/lib/python3.10/site-packages")
import pytest
raise SystemExit(pytest.main(["-q", "tests/regression/test_vbm_consistency.py"]))
PY
```

Also run a small timing check on the caustic-born point and confirm mode3 is
materially faster than mode1 while staying within the tolerance above.

## Out Of Scope For The First Port

- Universal replacement of Cartesian inverse ray.
- Non-caustic high-magnification spine broadening.
- Removing Cartesian fallback.
- Adding an auto dispatcher.
- Per-branch mixing beyond the guarded fold-pair union.
- Notebook updates.

Those should be separate patches after the guarded spine kernel is present in
mainline and covered by tests.
