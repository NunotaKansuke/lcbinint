[Back to Light Curve Functions](LightCurves.md)

# Parallax

## Target coordinates

```python
sky = lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2")
```

## Light curve functions with parallax

```python
import numpy as np
import matplotlib.pyplot as plt
import lcbinint

s = 0.9
q = 0.1
u0 = 0.0
alpha = 1.0
rho = 0.01
tE = 30.0
t0 = 7500
piEN = 0.3
piEE = -0.2

params = {
    "s": s, "q": q, "u0": u0, "alpha": alpha,
    "rho": rho, "tE": tE, "t0": t0,
    "piEN": piEN, "piEE": piEE,
}
t = np.linspace(t0 - tE, t0 + tE, 300)

options = lcbinint.Options(tol=1e-3, reltol=1e-3)
static_curve = lcbinint.LightCurve(options=options)
parallax_curve = lcbinint.LightCurve(
    model=lcbinint.Model(parallax=True, sky=sky, t_ref=t0),
    options=options,
)

magnifications = static_curve(t, params)
magnifications_parallax = parallax_curve(t, params)
trajectory = static_curve.source_trajectory(t, params)
trajectory_parallax = parallax_curve.source_trajectory(t, params)
```

Plot the static and parallax light curves:

```python
plt.figure()
plt.plot(t, magnifications, "g")
plt.plot(t, magnifications_parallax, "m")
plt.xlabel("Time")
plt.ylabel("Magnification")
plt.show()
```

![Binary-lens light curve with parallax](figures/BinaryLens_lightcurve_parallax.png)

Plot the caustics and trajectories in a separate block:

```python
caustics = static_curve.caustics(params)

plt.figure(figsize=(5, 5))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(-np.asarray(x), -np.asarray(y))
plt.plot(-np.asarray(trajectory.x), -np.asarray(trajectory.y), "g")
plt.plot(-np.asarray(trajectory_parallax.x), -np.asarray(trajectory_parallax.y), "m")
plt.axis("equal")
plt.show()
```

![Source trajectory with parallax](figures/BinaryLens_lightcurve_parallax_caustics.png)

## Satellite Parallax

For a space observatory, pass a table with `(JD, RA_deg, Dec_deg, distance_AU)`
columns to its own site while reusing the same physical model:

```python
satellite_table = np.loadtxt("satellite1.txt")
space_curve = lcbinint.LightCurve(
    model=parallax_curve.model,
    site=lcbinint.obs.Site("space", satellite_table),
    options=options,
)
magnifications_satellite = space_curve(t, params)
```

## Terrestrial parallax

```python
terrestrial_model = lcbinint.Model(
    parallax=True,
    terrestrial=True,
    sky=sky,
    t_ref=t0,
)
africa = lcbinint.LightCurve(
    model=terrestrial_model,
    site=lcbinint.obs.Site("ground", -29.0, 20.0),
    options=options,
)
chile = lcbinint.LightCurve(
    model=terrestrial_model,
    site=lcbinint.obs.Site("ground", -29.0, -70.7),
    options=options,
)
```

[Go to Orbital Motion](OrbitalMotion.md)
