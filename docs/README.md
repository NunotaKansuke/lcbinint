# Documentation

`lcbinint` keeps its documentation as Markdown alongside its source. Start
with the guide that matches the task:

- [`user-guide.md`](user-guide.md): install the package, evaluate a light
  curve, choose physical and numerical settings, inspect diagnostics, use
  parallax, and plot lens-plane geometry.
- [`effects-and-examples.md`](effects-and-examples.md): short runnable recipes
  for individual high-order effects, each paired with observed output.
- [`python-api.md`](python-api.md): public Python API reference, including
  parameter names, return values, coordinate conventions, and limitations.
- [`c-api.md`](c-api.md): C/C++ integration reference for the public header.
- [`development.md`](development.md): project layout, CMake/Python builds,
  test commands, and the validation workflow for contributors.

The focused documents below describe numerical behavior and its validation:

- [`numerical-methods.md`](numerical-methods.md): runtime method selection,
  fourth-order finite-source correction, automatic `nbin`, external-contour
  recommendations, and parallax conventions.
- [`finite-source-auto-calibration.md`](finite-source-auto-calibration.md):
  calibration campaign and frozen automatic-resolution/backend rules.
- [`finite-source-safety-validation.md`](finite-source-safety-validation.md):
  independent safety checks around point-source and finite-source switching.

The reproducible scripts and frozen machine-readable calibration artifacts live
under `tests/diagnostics/`. User-facing workflows live under `example/`.
