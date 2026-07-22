[Previous: Binary sources](BinarySources.md) · [Documentation home](readme.md) · [Next: Binary source + xallarap](BinarySourceXallarap.md)

# Xallarap

Xallarap adds orbital motion to a single source. This page gives one complete
example for each circular parameterization. Their parameters are the source-1
parameters used by the corresponding examples in
[Binary source + xallarap](BinarySourceXallarap.md).

## Circular velocity

`circular_velocity` uses the sky-plane position `xi_1`, `xi_2` at `t_ref` and
the velocity-state parameters `w1`, `w2`, `w3`.

```python
import numpy as np
import matplotlib.pyplot as plt
import lcbinint

times = np.linspace(7470.0, 7530.0, 300)
parameters = {
    "s": 0.9, "q": 0.1, "alpha": 0.7, "tE": 30.0,
    "t0": 7500.0, "u0": 0.20, "rho": 0.004,
    "xi_1": 0.02, "xi_2": -0.01,
    "w1": 0.004, "w2": 0.35, "w3": 0.08,
}
static_curve = lcbinint.LightCurve()
curve = lcbinint.LightCurve(xallarap="circular_velocity", t_ref=7500.0)
rectilinear_parameters = dict(parameters)
for key in ("xi_1", "xi_2", "w1", "w2", "w3"):
    rectilinear_parameters.pop(key)

static_magnification = static_curve(times, rectilinear_parameters)
magnification = curve(times, parameters)
```

```python
plt.figure(figsize=(4.2, 2.7))
plt.plot(times, static_magnification, color="0.55", ls="--", label="rectilinear")
plt.plot(times, magnification, color="#0173B2", label="xallarap")
plt.xlabel("Time")
plt.ylabel("Magnification")
plt.legend(loc="upper left", fontsize=8)
plt.show()
```

![Velocity xallarap light curve](figures/Xallarap_velocity_lightcurve.png)

```python
trajectory = curve.source_trajectory(times, parameters)
static_trajectory = static_curve.source_trajectory(times, rectilinear_parameters)
caustics = curve.caustics(parameters)

plt.figure(figsize=(3.4, 3.2))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(static_trajectory.x, static_trajectory.y, color="0.55", ls="--", label="rectilinear")
plt.plot(trajectory.x, trajectory.y, color="#0173B2", label="xallarap")
plt.xlabel("Trajectory coordinate 1")
plt.ylabel("Trajectory coordinate 2")
plt.axis("equal")
plt.legend(fontsize=7)
plt.show()
```

![Velocity xallarap trajectory and caustics](figures/Xallarap_velocity_geometry.png)

### Kepler velocity

`kepler_velocity` uses the same position--velocity inputs and adds
`xa_szs` and `xa_ar`. The result is evaluated in the same way.

```python
kepler_parameters = dict(parameters, xa_szs=0.2, xa_ar=1.4)
kepler_curve = lcbinint.LightCurve(xallarap="kepler_velocity", t_ref=7500.0)
kepler_magnification = kepler_curve(times, kepler_parameters)
```

## Circular elements

`circular_elements` uses `xi_1`, `xi_2`, `period_xa`, and `inc_xa`.
The narrow finite-source feature here is resolved with a denser time grid.

```python
times = np.linspace(7470.0, 7530.0, 1200)
parameters = {
    "s": 0.9, "q": 0.1, "alpha": 0.7, "tE": 30.0,
    "t0": 7500.0, "u0": 0.20, "rho": 0.004,
    "xi_1": 0.02, "xi_2": -0.01,
    "period_xa": 90.0, "inc_xa": 0.6,
}
static_curve = lcbinint.LightCurve()
curve = lcbinint.LightCurve(xallarap="circular_elements", t_ref=7500.0)
rectilinear_parameters = dict(parameters)
for key in ("xi_1", "xi_2", "period_xa", "inc_xa"):
    rectilinear_parameters.pop(key)

static_magnification = static_curve(times, rectilinear_parameters)
magnification = curve(times, parameters)
```

```python
plt.figure(figsize=(4.2, 2.7))
plt.plot(times, static_magnification, color="0.55", ls="--", label="rectilinear")
plt.plot(times, magnification, color="#0173B2", label="xallarap")
plt.xlabel("Time")
plt.ylabel("Magnification")
plt.legend(loc="upper left", fontsize=8)
plt.show()
```

![Elements xallarap light curve](figures/Xallarap_elements_lightcurve.png)

```python
trajectory = curve.source_trajectory(times, parameters)
static_trajectory = static_curve.source_trajectory(times, rectilinear_parameters)
caustics = curve.caustics(parameters)

plt.figure(figsize=(3.4, 3.2))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(static_trajectory.x, static_trajectory.y, color="0.55", ls="--", label="rectilinear")
plt.plot(trajectory.x, trajectory.y, color="#0173B2", label="xallarap")
plt.xlabel("Trajectory coordinate 1")
plt.ylabel("Trajectory coordinate 2")
plt.axis("equal")
plt.legend(fontsize=7)
plt.show()
```

![Elements xallarap trajectory and caustics](figures/Xallarap_elements_geometry.png)

### Keplerian elements

The API name for the eccentric-elements form is `orbital_elements`. Add
`ecc_xa` and `peri_xa` to the circular-elements state, then evaluate it.

```python
kepler_parameters = dict(parameters, ecc_xa=0.2, peri_xa=0.4)
kepler_curve = lcbinint.LightCurve(xallarap="orbital_elements", t_ref=7500.0)
kepler_magnification = kepler_curve(times, kepler_parameters)
```

[Previous: Binary sources](BinarySources.md) · [Documentation home](readme.md) · [Next: Binary source + xallarap](BinarySourceXallarap.md)
