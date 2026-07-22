# lcbinint documentation

Welcome. The recommended path is a short visual course: begin with a binary
lens, then add one physical effect at a time. Every numbered tutorial contains
copyable code, an executable script, and a generated light-curve / source
trajectory / caustic figure.

## Start here: the tutorial course

Follow these pages in order. Each one introduces exactly one new modelling
idea, so it is clear which choice changed the light curve.

| Step | Tutorial | You will learn |
| --- | --- | --- |
| 1 | [Binary lens](tutorials/01-binary-lens.md) | Build a reusable light curve and read its source trajectory against caustics. |
| 2 | [Parallax](tutorials/02-parallax.md) | Add annual, terrestrial, and space-observatory motion correctly. |
| 3 | [Lens orbital motion](tutorials/03-orbital-motion.md) | Evolve the projected lens geometry with time. |
| 4 | [Xallarap](tutorials/04-xallarap.md) | Add source orbital motion. |
| 5 | [Binary source](tutorials/05-binary-source.md) | Combine two source trajectories and their fluxes. |
| 6 | [Triple lens](tutorials/06-triple-lens.md) | Add a second companion and inspect triple caustics. |
| 7 | [Putting it together](tutorials/07-putting-it-together.md) | Choose the right model/options, validate it, and navigate the rest of the docs. |

Run all six plotted examples from the repository root:

```sh
for tutorial in example/tutorials/binary_lens.py example/tutorials/parallax.py \
  example/tutorials/orbital_motion.py example/tutorials/xallarap.py \
  example/tutorials/binary_source.py example/tutorials/triple_lens.py; do
  PYTHONPATH=build python "$tutorial"
done
```

The resulting figures are checked in under `docs/assets/tutorials/`; their
source files are together in [`example/tutorials/`](../example/tutorials/).
The [tutorial gallery](effects-and-examples.md) is a compact visual index of
the same course.

## Reference and validation

Once the required effects are clear, use these documents for exact details:

- [User guide](user-guide.md): installation, normal evaluation workflow,
  diagnostics, parallax setup, batching, and image-plane work.
- [Python API reference](python-api.md): every public class, option, parameter,
  output type, alias, and supported mode.
- [C/C++ API reference](c-api.md): public header, structures, native calls, and
  linking guidance.
- [Numerical methods](numerical-methods.md): finite-source method selection,
  tolerance semantics, automatic `nbin`, and parallax conventions.

The following are evidence and maintainer-facing validation notes:

- [Finite-source auto calibration](finite-source-auto-calibration.md)
- [Finite-source safety validation](finite-source-safety-validation.md)
- [Developer guide](development.md)

## At a glance

Start with `LightCurve()` and a binary parameter mapping. Keep physical effects
on `Model`; keep coordinate convention and numerical accuracy controls on
`Options`. Use `LightCurve.info()` whenever finite-source accuracy matters.
Then move from tutorial 1 upward, enabling only the effect required by the
event. This separation is the simplest way to keep a complex fit explainable.
