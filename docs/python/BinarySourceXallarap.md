[Previous: Xallarap](Xallarap.md) · [Documentation home](readme.md) · [Next: Combining higher-order effects](CombinedEffects.md)

# Binary source + xallarap

This page combines `source="binary"` with xallarap. `flux_ratio` weights the
two light curves; `source_mass_ratio = M2 / M1` fixes the second orbital state
from the first one: `r2 = -r1 / source_mass_ratio`.

All integrated models require `rho1`, `rho2`, `flux_ratio`, and
`source_mass_ratio`.

| Xallarap mode | Additional inputs |
| --- | --- |
| `"circular_elements"` | `xi_1`, `xi_2`, `period_xa`, `inc_xa` |
| `"orbital_elements"` | `xi_1`, `xi_2`, `period_xa`, `ecc_xa`, `peri_xa`, `inc_xa` |
| `"circular_velocity"` | `xi_1`, `xi_2`, `w1`, `w2`, `w3` |
| `"kepler_velocity"` | `xi_1`, `xi_2`, `w1`, `w2`, `w3`, `xa_szs`, `xa_ar` |

## Direct xallarap coordinates

With `source_orbit_coordinates="xallarap"`, `t0` and `u0` specify the CoM
track and `xi_1`, `xi_2` specify source 1 relative to that track.

```python
import numpy as np
import matplotlib.pyplot as plt
import lcbinint

times = np.linspace(7470.0, 7530.0, 300)
parameters = {
    "s": 0.9, "q": 0.1, "alpha": 0.7, "tE": 30.0,
    "t0": 7500.0, "u0": 0.35,
    "rho1": 0.004, "rho2": 0.002, "flux_ratio": 0.4,
    "source_mass_ratio": 0.7,
    "xi_1": 0.006, "xi_2": -0.003,
    "w1": 0.001, "w2": 0.12, "w3": 0.02,
}
binary_curve = lcbinint.LightCurve(
    source="binary", xallarap="circular_velocity",
    source_orbit_coordinates="xallarap", t_ref=7500.0,
)
component_curve = lcbinint.LightCurve(xallarap="circular_velocity", t_ref=7500.0)

source1_params = dict(parameters, rho=parameters["rho1"])
for key in ("rho1", "rho2", "flux_ratio", "source_mass_ratio"):
    source1_params.pop(key)
source2_params = dict(
    source1_params,
    rho=parameters["rho2"],
    xi_1=-source1_params["xi_1"] / parameters["source_mass_ratio"],
    xi_2=-source1_params["xi_2"] / parameters["source_mass_ratio"],
)
source1_magnification = component_curve(times, source1_params)
source2_magnification = component_curve(times, source2_params)
binary_magnification = binary_curve(times, parameters)
```

```python
plt.figure(figsize=(4.2, 2.7))
plt.plot(times, source1_magnification, color="#0173B2", alpha=0.45, lw=1.0, label="source 1")
plt.plot(times, source2_magnification, color="#029E73", alpha=0.45, lw=1.0, label="source 2")
plt.plot(times, binary_magnification, color="black", lw=1.5, label="total")
plt.xlabel("Time")
plt.ylabel("Magnification")
plt.legend(loc="upper left", fontsize=8)
plt.show()
```

![Direct binary-source xallarap light curve](figures/BinarySource_xallarap_lightcurve.png)

```python
source1_trajectory = component_curve.source_trajectory(times, source1_params)
source2_trajectory = component_curve.source_trajectory(times, source2_params)
caustics = lcbinint.LightCurve().caustics(source1_params)

plt.figure(figsize=(3.4, 3.2))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(source1_trajectory.x, source1_trajectory.y, color="#0173B2", label="source 1")
plt.plot(source2_trajectory.x, source2_trajectory.y, color="#029E73", label="source 2")
plt.xlabel("Trajectory coordinate 1")
plt.ylabel("Trajectory coordinate 2")
plt.axis("equal")
plt.legend(fontsize=7)
plt.show()
```

![Direct binary-source xallarap trajectories and caustics](figures/BinarySource_xallarap_trajectories.png)

## Trajectory-offset coordinates

`source_orbit_coordinates="trajectory_offset"` accepts the two tangent tracks
at `t_ref`: `t0`, `u0`, `t0_2`, and `u0_2`. Use the same binary-source inputs
above. The values below represent the same CoM state and source-1 offset as
the direct example at `t_ref`.

```python
offset_parameters = dict(parameters)
offset_parameters.pop("xi_1")
offset_parameters.pop("xi_2")
offset_parameters.update({
    "t0": 7499.82, "u0": 0.347,
    "t0_2": 7500.257142857, "u0_2": 0.354285714,
})
binary_curve = lcbinint.LightCurve(
    source="binary", xallarap="circular_velocity",
    source_orbit_coordinates="trajectory_offset", t_ref=7500.0,
)
binary_magnification = binary_curve(times, offset_parameters)
```

The component curves use the CoM-converted states. The full copy/paste version
is shown by the calculation that generated the plot:

```python
q_s = offset_parameters["source_mass_ratio"]
relative_tau = (offset_parameters["t0"] - offset_parameters["t0_2"]) / offset_parameters["tE"]
relative_beta = offset_parameters["u0_2"] - offset_parameters["u0"]
source1_params = {
    "s": offset_parameters["s"], "q": offset_parameters["q"], "alpha": offset_parameters["alpha"],
    "tE": offset_parameters["tE"],
    "t0": (offset_parameters["t0"] + q_s * offset_parameters["t0_2"]) / (1.0 + q_s),
    "u0": (offset_parameters["u0"] + q_s * offset_parameters["u0_2"]) / (1.0 + q_s),
    "rho": offset_parameters["rho1"],
    "xi_1": -q_s * relative_tau / (1.0 + q_s),
    "xi_2": -q_s * relative_beta / (1.0 + q_s),
    "w1": offset_parameters["w1"], "w2": offset_parameters["w2"], "w3": offset_parameters["w3"],
}
source2_params = dict(
    source1_params, rho=offset_parameters["rho2"],
    xi_1=-source1_params["xi_1"] / q_s,
    xi_2=-source1_params["xi_2"] / q_s,
)
source1_magnification = component_curve(times, source1_params)
source2_magnification = component_curve(times, source2_params)
```

```python
plt.figure(figsize=(4.2, 2.7))
plt.plot(times, source1_magnification, color="#0173B2", alpha=0.45, lw=1.0, label="source 1")
plt.plot(times, source2_magnification, color="#029E73", alpha=0.45, lw=1.0, label="source 2")
plt.plot(times, binary_magnification, color="black", lw=1.5, label="total")
plt.xlabel("Time")
plt.ylabel("Magnification")
plt.legend(loc="upper left", fontsize=8)
plt.show()
```

![Trajectory-offset binary-source xallarap light curve](figures/BinarySource_xallarap_offset_lightcurve.png)

```python
source1_trajectory = component_curve.source_trajectory(times, source1_params)
source2_trajectory = component_curve.source_trajectory(times, source2_params)
caustics = lcbinint.LightCurve().caustics(source1_params)

plt.figure(figsize=(3.4, 3.2))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(source1_trajectory.x, source1_trajectory.y, color="#0173B2", label="source 1")
plt.plot(source2_trajectory.x, source2_trajectory.y, color="#029E73", label="source 2")
plt.xlabel("Trajectory coordinate 1")
plt.ylabel("Trajectory coordinate 2")
plt.axis("equal")
plt.legend(fontsize=7)
plt.show()
```

![Trajectory-offset source trajectories and caustics](figures/BinarySource_xallarap_offset_geometry.png)

[Previous: Xallarap](Xallarap.md) · [Documentation home](readme.md) · [Next: Combining higher-order effects](CombinedEffects.md)
