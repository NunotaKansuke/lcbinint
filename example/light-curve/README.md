# Light-curve callables

`light_curve_modes.py` shows the intended public API:

- construct reusable `LightCurve` callables
- declare physical effects with `Model`
- keep numerical choices in `Options`
- pass lens parameters on every evaluation
- inspect the selected finite-source methods and convergence diagnostics
- enable annual and terrestrial parallax explicitly with `sky`, `site`, and
  `t_ref`

Run against an in-tree build:

```bash
PYTHONPATH=build python example/light-curve/light_curve_modes.py
```

The site alone does not activate terrestrial parallax. Both `parallax=True`
and `terrestrial=True` are explicit `Model` choices.
