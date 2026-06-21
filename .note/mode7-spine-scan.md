# Mode7 Image Spine Guided Scan

## Purpose

`FINITE=7` / legacy mode7 is an experimental local-coordinate finite-source
kernel. The current useful spine path targets caustic-born fold-pair images
where the image is a long, thin, curved strip.

The goal is not to replace mode4 everywhere. The goal is to reduce exact binary
lens-equation evaluations in the cases where mode4 spends most of its time
scanning a wide 2D patch around a thin image.

## Current Stable Scope

The stable spine path is enabled only when:

- mode7 is explicitly selected,
- finite-source seeds include caustic-born branches,
- a high-Jacobian, low-determinant seed has an opposite-parity partner,
- the partner passes a nearest-neighbor / mutual-nearest ambiguity check,
- the spine frame construction and normal scans remain finite.

For this caustic-born fold-pair case, one spine is treated as a local union
scan for the fold pair. The paired seed is marked as overlapped so the same
branch area is not counted again.

The non-caustic high-magnification extension was tested but is not currently
enabled. It can be fast, but branch overlap accounting is not robust enough yet.

## Algorithm Summary

For an accepted seed:

1. Build a local `Local7Frame`.
2. Grow a spine in both directions along the local long-axis eigenvector.
3. At each proposed spine point, evaluate the exact binary lens equation and
   apply damped Newton correction toward the predicted source-plane offset.
4. Stop the spine when the source disk is left, curvature is too high, or the
   correction no longer converges.
5. For each spine point, scan only the local short-axis direction with exact
   lens-equation inside/outside tests.
6. Accumulate image-plane area using local spine spacing times normal spacing.

This keeps exact inside/outside decisions. The speedup comes from generating
far fewer candidate image-plane samples.

## Current Measured Behavior

Representative direct finite-source point:

```text
separation = 1.4
mass_ratio = 0.4
y1 = -0.24854045037531268
y2 = -0.15
rho = 1e-4
source_bins = 300
limb_darkening_c = 0.5
```

Typical result:

```text
mode4 time: about 180-260 ms
mode7 time: about 10-15 ms
relative difference mode7 vs mode4: about 6e-6
mode7 exact_lens_evals: about 1.0M
```

Caustic-neighborhood sweeps:

```text
wide crossing A, 41 points: max relative difference about 1e-4
wide crossing B, 41 points: max relative difference about 8e-5
```

## Known Non-Goals For This Patch

Do not enable the spine kernel as a universal mode4 replacement.

Do not enable non-caustic high-magnification spine by default yet. Tests found
cases where adding two spine branches directly causes overcounting, and simple
image-cell duplicate hashing removes valid distinct images.

Do not remove the existing mode7 frontier scanner. It remains the fallback
inside mode7 when the spine eligibility or spine construction is not trusted.

## Validation Commands

Use the local build module, not an installed `lcbinint` from site-packages:

```bash
cmake --build build
ctest --test-dir build --output-on-failure

LD_LIBRARY_PATH=/rogue1_8/nunota/local/gsl/lib python -S - <<'PY'
import sys
sys.path.insert(0, "/rogue1_8/nunota/lcbinint-idea-stable/build")
sys.path.append("/home/nunota/.miniconda3/envs/myenv/lib/python3.10/site-packages")
import pytest
raise SystemExit(pytest.main(["-q", "tests/regression/test_vbm_consistency.py"]))
PY
```

## Next Work

The next robust extension is branch-overlap accounting for non-caustic
high-magnification images. The right direction is to keep spine sampling but
make branch union/separation explicit. The simple attempts tried so far were:

- adding both paired spines directly: fast but can overcount,
- image-plane cell hashing: avoids some duplicates but can delete valid images.

Both are insufficient as a final solution.
