# Certified component tile kernel: handoff

## Scope and branch

- Worktree: `/rogue1_8/nunota/lcbinint-certified-component`
- Branch: `codex/certified-component-tile-kernel`
- Base: `feature/jax-cpu-inverse-ray-mvp`
- The earlier experimental worktree/branch is intentionally separate and was
  not merged into this branch.

The target defect is the binary finite-source cusp case

```text
source_x       = 0.653
source_y       = 0.020
separation     = 1.2
mass_ratio     = 0.1
source_radius  = 0.020
```

The historical Cartesian inverse ray converged near `3.94855` even though the
uniform reference is `3.960889170813`.  The missed contribution is a thin
component born in a cap of the source disk.  A fixed source-limb probe set can
miss that cap at every refinement level, so a coarse/fine value difference is
not a valid completeness test.

Reference values used here:

| Profile | Reference |
| --- | ---: |
| Uniform | `3.960889170813` |
| Linear limb darkening, c=0.5 | `3.836256795465` |

## Changes currently present

### Common component certificate

Files:

- `src/lcbinint/magnification/binary_component_certificate.hpp`
- `src/lcbinint/magnification/binary_component_certificate.cpp`

`certify_binary_source_components()` walks cached caustic polylines, finds
segments entering the source disk, and generates local probes whose radius is
proportional to the measured cap depth.  It retains probes with five physical
images.  This is intended as a geometry-derived component witness, not a
limb-sample-density heuristic.

### Common double tile kernel

Files:

- `src/lcbinint/magnification/binary_tile_kernel.hpp`
- `src/lcbinint/magnification/binary_tile_kernel.cpp`

`integrate_binary_component_tiles()` deduplicates tiles rooted at a certified
image set and integrates cell moments.  Its source-disk classification was
changed from four corner remaps to an analytic local phi-gradient bound.

### Current integration points

- `python/bind_jax_ir.cpp`: fused binary Cartesian discovery calls the common
  component certificate and uses its probes to populate tile support.
- `src/lcbinint/magnification/finite_source_magnifier.cpp`: if a certificate
  finds more images than legacy seeds, the native contact path calls the
  common double tile kernel.

## Measured state

Native, Cartesian, component tile path:

| bins | Uniform | Linear c=0.5 |
| ---: | ---: | ---: |
| 64 | 3.9610483014 | 3.8368531231 |
| 128 | 3.9609268758 | 3.8363978155 |
| 256 | 3.9609016801 | 3.8362463532 |
| 512 | 3.9608868052 | 3.8361883636 |

The 512-bin errors are approximately `2.4e-6` (uniform) and `6.8e-5`
(linear).  These figures show the support omission is addressed for this
case, but are not proof that the implementation is complete.

JAX fused CPU path, limb samples fixed at 16:

| resolution | Uniform |
| ---: | ---: |
| 32 | 3.9626660609 |
| 64 | 3.9613737930 |
| 128 | 3.9610933482 |

At 128 bins, rough measured wall times were approximately:

| Case | Native | JAX fused CPU |
| --- | ---: | ---: |
| Ordinary non-contact | 1.35 ms | 5.3 ms |
| Cusp reproduction | 13.2 ms | 6.0 ms |

The native contact overhead is not acceptable for the requested performance
condition.

## Verification performed

Build:

```bash
cmake -S . -B build-certified -DGSL_ROOT=/home/nunota/.miniconda3/envs/myenv
cmake --build build-certified -j4
ctest --test-dir build-certified --output-on-failure
```

`unit_core` passed.

The following were invoked with the build-side native module explicitly
preloaded to avoid the editable installation redirect:

```text
tests/jax_ir/test_discovery.py
tests/jax_ir/test_public_api.py
tests/regression/test_binary_cusp_component.py
```

The progress output reached the end, but the final pytest summary was not
captured.  Re-run and record the exact exit status before claiming the suite
passes.

The old test named
`test_sixteen_limb_seeds_recover_component_missed_by_eight` was changed to
assert limb-count independence, since a certificate should make 8, 16, and
32 limb samples agree for that case.

## Known incomplete or unsafe aspects

1. **Not one kernel yet.**  The shared library kernel is double-only.  JAX
   still has the template/Jet cell-moment implementation in
   `python/bind_jax_ir.cpp`.  The native and JAX value implementations are
   similar but not identical.

2. **No refinement support certificate.**  There is no persistent component
   identity, no support fingerprint, and no coarse/fine comparison that
   forces `converged=False` when support changes.  Do not treat the current
   error estimate as sufficient for the requested fail-closed contract.

3. **JAX gradients are not settled.**  The fused JAX value path discovers
   certified support, but its JVP uses the existing Jet path.  Source-x and
   separation derivatives were observed to move substantially across
   resolutions.  Do not add a gradient-stability acceptance test until the
   value and JVP consume an identical stopped-gradient support descriptor.

4. **Pure JAX discovery remains old.**
   `python/lcbinint_jax/discovery.py` still generates centre plus fixed limb
   roots.  The common certificate is connected only to the fused C++ path.

5. **Native speed is inadequate.**  The common double kernel integrates too
   many cells and uses simple subcell boundary quadrature.  It is accurate in
   the reproduction but roughly 10x slower than the ordinary native path.

## Recommended next implementation order

1. Move the binary moment machinery from `python/bind_jax_ir.cpp` into a
   templated header/library boundary usable by both the double native kernel
   and the Jet JAX kernel.  Do not maintain two independently evolving cell
   classifiers.

2. Introduce a `BinaryTileSupport` descriptor containing:
   - certified component witnesses;
   - component IDs;
   - deduplicated tile coordinates;
   - overflow/root-failure status; and
   - a stable support fingerprint.

3. Generate this descriptor once per evaluation, pass it to both value and
   JVP paths, and stop gradients through the descriptor itself.

4. Make native refinement explicitly compare consecutive support fingerprints.
   If they differ, report `converged=false` regardless of the value delta.

5. Replace the current native subcell-only boundary work with the shared
   affine moment implementation before attempting performance conclusions.

6. Add acceptance tests covering:
   - uniform and linear references above;
   - no false convergence when a component changes;
   - JAX value/JVP resolution stability;
   - native/JAX normal and cusp benchmarks; and
   - pure JAX behavior or an explicit contract that CPU uses the certified
     fused support path.

## Git state

No commit has been created.  The branch contains the files described above
and should be reviewed as an incomplete foundation, not merged as a final
fix.
