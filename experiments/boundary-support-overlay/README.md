# Isolated boundary/support experiment kernels

Normal builds use the original master (`b51c00c`) kernels and bindings. Enable
this overlay explicitly with `-DLCBININT_BOUNDARY_SUPPORT_EXPERIMENTS=ON`.
`enable.cmake` selects copied experimental sources and puts experimental headers
first in the include path; the research-only helpers retained under root `src/`
are compiled/referenced only in that build.

`build-boundary-support` is configured ON. `build-master-main-20261012` is OFF.
The pre-isolation source manifest and full tracked patch are retained here.
The Python experiment bindings and subsequent research edits live in this
overlay, rather than modifying default master kernels. The manifest describes
the pre-isolation snapshot; later edits have separate experiment artifact hashes.

Aggressive seed variant for algebraic integration is preserved separately in
`docs/design/algebraic-discriminant-handoff.md`. Neither the newer endpoint
selector nor its lattice coverage failures are promoted to the ordinary path.

The current resolution prototype certifies a local boundary graph's slope and
curvature. It supplies a midpoint geometric area remainder for uniform brightness,
not an error certificate for the complete Bennett-corrected magnification.

For quadrature diagnostics the default Cartesian seed scheduling now matches master (offer all seeds before draining the frontier). Set `LCBININT_SEQUENTIAL_SEED_FILL=1` explicitly to reproduce the archived sequential seed experiment. This switch can affect run partition and quadrature values.

Current single-grid residual fixes and regression evidence are documented in
`docs/design/nbin-global-residual-root-cause-report.md`. The residual diagnostic
remains opt-in and uncertified. Its local missing-image repair reuses centre
images already solved during seed construction. The overlay boundary kernel
now also handles exact lens poles and polishes crossings once within their
existing brackets; those kernel fixes apply with the residual diagnostic off.

The full uniform-source adaptive area prototype, retained-state contract,
accuracy/cost measurements and uncertified limits are documented in
`docs/design/nbin-adaptive-image-area-report.md`. It partitions the complete
represented area; it does not add its result to the Bennett magnification.
