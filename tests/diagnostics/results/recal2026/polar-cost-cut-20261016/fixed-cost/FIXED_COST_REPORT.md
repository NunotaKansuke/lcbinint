# Resolution-independent cost audit

Date: 2026-10-16

Branch: `feature/polar-cost-cut`

Frozen comparison base: `e27f2663b3f2cebc3f3a6bf1196f7b2268f7e197`

## Change

The binary-image seed cache is used by both Cartesian and Polar before the resolution-dependent integration work. Its old lookup scanned up to 1,024 entries and shifted the vector on FIFO eviction. The candidate keeps entries in a ring and uses a fixed-size open-addressing index only after a cache with at least 64 entries records a real hit. It populates the index as keys recur. A cold stream of distinct positions stays on the allocation-free linear path.

Cache keys still use exact equality across lens parameters, source position, source radius, caustic distance, and center-seed presence. NaN keys remain uncached, matching the old non-reflexive equality behavior. Seed generation, support results, flood fill, quadrature, and error estimation are unchanged.

## Comparison

The vector reference was rebuilt from the frozen `HEAD` versions of the two native files, with the current Polar two-bin/ratio-six discovery change applied. Both probe executables use the same runner source, compiler flags, static libraries, and CPU affinity. The five paired runs alternate executable order. The benchmark source, input, binaries, and hashes are preserved in this directory.

Warm results measure 300,000 cyclic lookups per phase, after cache population and identity verification. They include the returned seed-vector copy and checksum, but exclude seed generation and integration.

| Cached positions | Vector (ns/hit) | Lazy index (ns/hit) | Vector / lazy speedup |
|---:|---:|---:|---:|
| 16 | 22.40 | 23.29 | 0.96x |
| 32 | 33.84 | 33.69 | 1.00x |
| 48 | 47.44 | 46.66 | 1.02x |
| 64 | 61.18 | 42.32 | 1.45x |
| 256 | 195.27 | 43.74 | 4.47x |
| 1,024 | 693.16 | 48.84 | 14.19x |

A fresh process was also timed while constructing the distinct seeds and then requesting every retained entry for exact identity verification. The latter is the first repeated pass and activates/populates the candidate index. It includes process startup and seed construction.

| Cached positions | Vector first pass (ms) | Lazy-index first pass (ms) | Relative difference |
|---:|---:|---:|---:|
| 64 | 4.653 | 4.708 | +1.2% |
| 128 | 4.861 | 4.904 | +0.9% |
| 256 | 5.251 | 5.307 | +1.1% |
| 1,024 | 8.386 | 8.490 | +1.2% |

At 1,024 positions, a later fully warm pass saves about 0.66 ms in this microbenchmark. That is cache-lookup time only; it does not establish a 0.66 ms whole-LightCurve improvement. The measured gain requires the exact same keys to recur. At 64 positions, the per-pass saving is about 1.2 microseconds, so index activation is less useful for short histories.

## Checks and limits

- Adaptive Polar API regression: 38 checks passed, 0 failed.
- Cache identity/eviction probe: after inserting 2,048 distinct positions into the 1,024-entry cache, all 1,024 retained seed vectors and support flags matched their saved values exactly.
- This change does not alter numerical results. It changes only reuse lookup and FIFO storage.
- No whole-LightCurve benchmark was run after this cache change. The earlier `auto` cohort also overlaps all 768 route-selector selected geometries, so it cannot establish selector generalization.

## Reproduction

Run from the repository root:

```bash
python tests/diagnostics/results/recal2026/polar-cost-cut-20261016/fixed-cost/run_seed_cache_cost_bench.py
```

The raw five-run samples, scopes, source/header/input/probe hashes, and executable hashes are in `timings-final-lazy-index.json`. `seed_cache_probe_timing.cpp` is the shared timing runner. The paired executables are `seed-cache-probe-fair-vector` and `seed-cache-probe-lazy-index`. The separate `seed-cache-probe-lazy-index-eviction` runner uses `seed_cache_probe.cpp` to verify indexed FIFO deletion and exact retained values after 2,048 positions.
