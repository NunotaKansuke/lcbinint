# Uniform triple-lens speed and accuracy

Measured on rogue1, Release build of experiment/algebraic_boundary at 7038d6c2 plus working-tree changes. Baseline native SHA-256 b07282fd6a0e7d7be8bfaafd416ab6e04961de74c38fbb8b6c832a466acbbd09, adopted native SHA-256 77f1243c40cb953a8924fdfaed6b4a491411c01137ef55a736537049bb1b7e17. Each driver verifies its imported extension path.

The contact planner previously formed regular-image events and then certified the remaining image plane. For uniform values it now tries a single complete event plan first. The direct search has a bounded share of the remaining budget. On refusal the planner reuses the projected radial cover and can still construct the regular components. Uniform gradients retain the image-specific domains. Applying the direct strategy to gradients moved one archived epoch's normalized gradient by 0.0111 and was rejected. A separate Gauss–Patterson value-panel trial did not improve timing. Six lens permutations did not give a consistently better frame.

Certified monotone angular intervals can now use scalar Newton proposals. Endpoint signs from the enclosed polynomial establish the bracket before acceptance. Indeterminate proposals fall back to the existing interval solver. For a dominant primary, the polar search also inverts the primary's point-lens map on both parity branches to narrow possible stationary limb positions. Near caustics the routing code shares a previously proved intersection across competing clearance checks for the same lens geometry. An exhausted search budget alone supplies no intersection witness.

## Timing

The public-API benchmark uses the 16 triple fixtures in ../eight_mode_speed_20261127/bench.py with offsets 0 and 1. Four two-epoch calls per geometry, relative tolerance 1e-4 and absolute tolerance zero. CPU 2 for the fixed-lens test and CPU 6 for the varying-lens test, one calling thread, three paired repetitions each. Run 2 reverses implementation order. Each timing includes the public light-curve call and excludes model construction. In the varying test, call j scales s by (1+0.0001j) and q by (1-0.0001j). This exercises lens-cache invalidation in a fit. It does not simulate an orbit over the two epochs. All timed epochs returned status zero, with matching support and validity flags. timing.tsv contains every paired run.

| Geometry | Mode | Before, ms/epoch | After, ms/epoch | Speed ratio |
|---|---|---:|---:|---:|
| Fixed | Value | 2.552 | 1.992 | 1.28 |
| Fixed | Value and gradient | 1.904 | 1.772 | 1.07 |
| Vary s and q | Value | 3.004 | 2.226 | 1.35 |
| Vary s and q | Value and gradient | 2.577 | 2.212 | 1.16 |

The independent public value-only comparison uses original fixtures 568 and 889, three epochs each, absolute and relative tolerances 1e-4, five warm calls per curve, then the median of three repeat totals. Baseline took 21.54 ms, adopted build 20.22 ms and VBM MultiPoly 6.95 ms. The VBM comparison uses elapsed time and a different workload from timing.tsv.

## Numerical checks

All 1,768 archived triple geometries were rerun at the central epoch with the limb coefficient forced to zero in both modes, yielding 3,536 successful evaluations. Support flags, gradient validity and status all matched the baseline. The maximum relative value change was 4.37e-5. The maximum gradient change divided by max(1, the per-epoch baseline gradient infinity norm) was 6.82e-5. A second pass with each archived limb coefficient retained identical flags for another 3,536 evaluations.

An independent VBM MultiPoly run used absolute tolerance 1e-8 and relative tolerance 1e-6 at 64 distinct timed epoch/geometry inputs. Maximum relative error was 4.22e-5 for values and 2.66e-7 for values returned with gradients. Some individual outputs lost accuracy within the requested tolerance. At archived epoch 1333 the baseline and adopted relative value errors against stricter VBM were 2.48e-7 and 4.39e-5 respectively. For 1493 they were 1.41e-6 and 4.13e-5.

The largest uniform gradient change was in dmu/ds at 1299. Baseline gave -137394.78 and the adopted sequence gave -137385.41. With VBM relative tolerance 1e-8, centred differences at steps 1.25e-7, 6.25e-8 and 3.125e-8 were -137414.72, -137393.52 and -137388.23. Extrapolating the last two steps to zero with a quadratic truncation error gives -137386.47. This tests a single derivative at a particularly sensitive epoch.

All seven targeted C++ suites passed, including point signs, angular isolation, event enclosures and regular support. The full triple Python regression set returned 259 passes and the same nine existing point-source test failures. Raw builds, rejected candidates, timing drivers, strict reference checks, finite-difference tables and test logs are in the parent checkout under build-accuracy-20261127/uniform-search. run.sh runs the fixed geometry workload, vary.sh runs its varying counterpart, uniform.py runs the archived uniform cases, ref.py produces the strict timing references and steps.py reproduces the derivative step study.
