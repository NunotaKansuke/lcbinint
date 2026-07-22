# 5. Binary source: combine two source trajectories

A binary source is selected on `Model` and combines two magnifications using a
source-flux ratio. This tutorial uses the coupled-xallarap representation, in
which `q_mass` determines the internally derived secondary xallarap amplitude.

```python
binary = lcbinint.LightCurve(
    source="binary", xallarap="circular_elements"
)
params.update(q_source=0.5, q_mass=2.0)
```

![Binary-source light curve](../assets/tutorials/05-binary-source-light-curve.png)

![Binary-source primary trajectory and caustics](../assets/tutorials/05-binary-source-geometry.png)

```sh
PYTHONPATH=build python example/tutorials/binary_source.py
```

The geometry panel shows the primary trajectory used for orientation. For a
non-coupled binary source, supply `q_source`, `t0_2`, and `u0_2` instead of a
positive `q_mass`.

---

**Course navigation:** [← Previous: Xallarap](04-xallarap.md) · [Tutorial home](../README.md) · [Next: Triple lens →](06-triple-lens.md)
