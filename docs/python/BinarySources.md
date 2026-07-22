[Previous: Orbital motion](OrbitalMotion.md) · [Documentation home](readme.md) · [Next: Combining physical effects](CombinedEffects.md)

# Binary sources

> VBMicrolensing correspondence: [BinarySources.md](https://github.com/valboz/VBMicrolensing/blob/main/docs/python/BinarySources.md). The same physical effects are demonstrated, but the parameters below deliberately use `lcbinint`'s barycentric state-vector convention rather than VBMicrolensing's function-specific transformation.

`lcbinint` uses one convention for every binary-source lens model:

| Parameter | Meaning at `t_ref` |
| --- | --- |
| `t0`, `u0`, `tE`, `alpha` | Rectilinear trajectory of the source barycenter. |
| `xi_1`, `xi_2` | Projected position of source 1 relative to the barycenter, in the trajectory basis. |
| `q_mass` | Source mass ratio (M_2/M_1); source 2 is placed at `(-xi_1/q_mass, -xi_2/q_mass)`. |
| `q_source` | Flux ratio (F_2/F_1), used only for the magnification blend. |
| `w1` | Fractional radial velocity of the relative source orbit. |
| `w2` | Angular velocity of the relative source orbit. |
| `w3` | Fractional line-of-sight velocity; the circular constraint determines the unobserved depth. |

This keeps source dynamics independent of luminosity. In particular,
`q_source` never silently determines `q_mass`.

## Xallarap

```python
import numpy as np
import matplotlib.pyplot as plt
import lcbinint

tE = 37.3
t0 = 7550.4
u0 = 0.075
rho = 0.004
FR = 0.4
q_mass = 1.0
xi_1 = 0.04
xi_2 = -0.025
w1 = 0.021
w2 = -0.02
w3 = 0.03

params = {
    "s": 1.0, "q": 1.0, "alpha": 0.0,
    "tE": tE, "t0": t0, "u0": u0, "rho": rho,
    "q_source": FR, "q_mass": q_mass,
    "xi_1": xi_1, "xi_2": xi_2,
    "w1": w1, "w2": w2, "w3": w3,
}
t = np.linspace(t0 - tE, t0 + tE, 300)

options = lcbinint.Options(tol=1e-3, reltol=1e-3)
xallarap_curve = lcbinint.LightCurve(
    model=lcbinint.Model(
        lens="binary",
        source="binary",
        xallarap="circular_velocity",
    ),
    options=options,
)

static_params = {**params, "w1": 0.0, "w2": 0.0, "w3": 0.0}
magnifications_static = xallarap_curve(t, static_params)
magnifications_xallarap = xallarap_curve(t, params)
```

Plot the two light curves in their own block:

```python
plt.figure(figsize=(3.6, 2.4))
plt.plot(t, magnifications_static)
plt.plot(t, magnifications_xallarap, "y")
plt.xlabel("Time")
plt.ylabel("Magnification")
plt.show()
```

![Binary-source xallarap light curve](figures/BinarySource_lightcurve_xallarap_2.png)

## Binary sources and binary lenses

```python
s = 0.9
q = 0.1
u0 = 0.1
alpha = 1.0
rho = 0.01
tE = 30.0
t0 = 7500
piEN = 0.3
piEE = -0.2
gamma1 = 0.011
gamma2 = -0.005
gamma3 = 0.005
FR = 1.0
q_mass = 1.0
xi_1 = 0.0
xi_2 = -0.1
source_w1 = 0.01
source_w2 = 0.02
source_w3 = 0.015

params = {
    "s": s, "q": q, "u0": u0, "alpha": alpha,
    "rho": rho, "tE": tE, "t0": t0,
    "piEN": piEN, "piEE": piEE,
    "g1": gamma1, "g2": gamma2, "g3": gamma3,
    "q_source": FR, "q_mass": q_mass,
    "xi_1": xi_1, "xi_2": xi_2,
    "w1": source_w1, "w2": source_w2, "w3": source_w3,
}
t = np.linspace(t0 - tE, t0 + tE, 300)
sky = lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2")
combined_options = lcbinint.Options(
    coordinates="vbm", tol=1e-3, reltol=1e-3
)

single_source_model = lcbinint.Model(
    parallax=True,
    orbital_motion="circular",
    sky=sky,
    t_ref=t0,
)
single_source_curve = lcbinint.LightCurve(
    model=single_source_model,
    options=combined_options,
)
binary_source_model = lcbinint.Model(
    lens="binary",
    source="binary",
    parallax=True,
    orbital_motion="circular",
    xallarap="circular_velocity",
    sky=sky,
    t_ref=t0,
)
binary_source_curve = lcbinint.LightCurve(
    model=binary_source_model,
    options=combined_options,
)
single_source_params = {
    key: value for key, value in params.items()
    if key not in {"q_source", "q_mass", "xi_1", "xi_2", "w1", "w2", "w3"}
}
magnifications_single_source = single_source_curve(t, single_source_params)
magnifications_binary_source = binary_source_curve(t, params)
```

The figure makes the extra binary-source and xallarap terms visible instead of
leaving the combined calculation as a single printed array:

```python
plt.figure(figsize=(5.6, 3.2))
plt.plot(t, magnifications_single_source, "y", label="single source + lens orbit")
plt.plot(t, magnifications_binary_source, "g", label="binary source + xallarap")
plt.xlabel("Time")
plt.ylabel("Magnification")
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
plt.show()
```

![Binary-source binary-lens light curve](figures/BinarySourceBinaryLens_lightcurve.png)

[Previous: Orbital motion](OrbitalMotion.md) · [Documentation home](readme.md) · [Next: Combining physical effects](CombinedEffects.md)
