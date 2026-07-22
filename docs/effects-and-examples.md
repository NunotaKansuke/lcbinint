# Tutorial gallery

This is the visual, effect-by-effect entry point for reproducing
VBMicrolensing-style examples with `lcbinint`. Each tutorial is a small
notebook-style page: a focused model choice, a runnable Python script, and
separate generated figures for the light curve and the corresponding source
trajectory/caustics. Start with the baseline and add only the physical effects
your event requires.

## Reading order

| Tutorial | What changes | Figure |
| --- | --- | --- |
| [1. Binary lens](tutorials/01-binary-lens.md) | Base binary-lens trajectory and caustics | [source](../example/tutorials/binary_lens.py) |
| [2. Parallax](tutorials/02-parallax.md) | Annual and terrestrial observer motion | [source](../example/tutorials/parallax.py) |
| [3. Lens orbital motion](tutorials/03-orbital-motion.md) | Time-evolving binary-lens geometry | [source](../example/tutorials/orbital_motion.py) |
| [4. Xallarap](tutorials/04-xallarap.md) | Source orbital motion | [source](../example/tutorials/xallarap.py) |
| [5. Binary source](tutorials/05-binary-source.md) | Flux-weighted pair of source trajectories | [source](../example/tutorials/binary_source.py) |
| [6. Triple lens](tutorials/06-triple-lens.md) | Second lens companion and triple caustics | [source](../example/tutorials/triple_lens.py) |
| [7. Putting it together](tutorials/07-putting-it-together.md) | Model-selection checklist and validation handoff | — |

## Run every tutorial

Build the extension, then run any script independently from the repository
root. Each script writes its figure below `docs/assets/tutorials/`, making the
rendered documentation and executable source stay visually aligned.

```sh
PYTHONPATH=build python example/tutorials/binary_lens.py
PYTHONPATH=build python example/tutorials/parallax.py
PYTHONPATH=build python example/tutorials/orbital_motion.py
PYTHONPATH=build python example/tutorials/xallarap.py
PYTHONPATH=build python example/tutorials/binary_source.py
PYTHONPATH=build python example/tutorials/triple_lens.py
```

The scripts use Matplotlib's non-interactive backend, so they also work in CI
or a remote shell. The figures are generated from the checked-in code—not
hand-drawn documentation assets. Their exact values are illustrative; use
`LightCurve.info()` and the [numerical methods](numerical-methods.md) guide
when setting a finite-source accuracy budget.

## More detail

- [User guide](user-guide.md): installation and full workflow context.
- [Python API reference](python-api.md): complete parameters and return types.
- [Image-plane example](../example/image-plane/): image regions and plotting.
- [VBMicrolensing comparisons](../example/compare-vbm/): optional external
  accuracy/timing comparison.
