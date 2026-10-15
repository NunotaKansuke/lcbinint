# Default gradient adoption

The compiled forward-gradient API now enables the validated close-root geometry screen and regularized replay by default. Both build_api_overlay.py callers (build.sh and build_algebraic.sh) use the same overlay. build_geometry.sh remains a comparison build and avoids applying an existing overlay twice. No new user tolerance or gradient-error-driven adaptation is introduced.

The default algebraic build completed successfully with:

```bash
bash benchmarks/holonomic/fused_gradient/build_algebraic.sh
```

Module: build-algebraic-boundary/fused-gradient/python/_lcbinint_holonomic_fused*.so. Restart an existing Python/Jupyter kernel to load it.

All generated C++ header contents match the previously validated geometry build, apart from two documentation-comment changes (header_diff.json). Numerical results and performance limits are those documented in ../gradient_refine_20261015/README.md. No additional numerical suite was run for this build-wiring change.

The commit includes the preceding value-mesh tangent scheduling and algebraic-path removal of the empirical automatic-nbin tolerance restriction, which this work uses. It preserves the generic finite-source backend selector and the uncertified gradient status. The existing comparison notebook changes and older exploratory evidence are left outside the commit.

provenance.json records the built module and generated manifest hashes. commit_paths.json records the explicit staged path list.
