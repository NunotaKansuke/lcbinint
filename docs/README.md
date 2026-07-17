# Documentation

Start with the repository `README.md` for installation and the callable API.
The focused documents here describe numerical behavior and its validation:

- [`numerical-methods.md`](numerical-methods.md): runtime method selection,
  fourth-order finite-source correction, automatic `nbin`, external-contour
  recommendations, and parallax conventions.
- [`finite-source-auto-calibration.md`](finite-source-auto-calibration.md):
  calibration campaign and frozen automatic-resolution/backend rules.
- [`finite-source-safety-validation.md`](finite-source-safety-validation.md):
  independent safety checks around point-source and finite-source switching.

The reproducible scripts and frozen machine-readable calibration artifacts live
under `tests/diagnostics/`. User-facing workflows live under `example/`.
