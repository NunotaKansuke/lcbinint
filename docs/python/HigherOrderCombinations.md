[← Higher-order effects](CombinedEffects.md)

# Higher-order combination catalogue

The catalogue is grouped first by lens and source multiplicity. Within every group, the examples progress through three levels: source-size baselines, annual parallax, and parallax with additional higher-order effects. Lens orbit is therefore never shown without parallax, and triple lenses are limited to their supported static-lens geometry.

`source` selects source multiplicity (`"single"` or `"binary"`). `finite_source=False` selects point-source evaluation and sets every source radius to zero during evaluation.

```python
import numpy as np
import matplotlib.pyplot as plt
import lcbinint

times = np.linspace(7470.0, 7530.0, 160)
sky = lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2")
options = lcbinint.Options(coordinates="vbm", tol=1e-3, reltol=1e-3)

phase = np.linspace(-1.0, 1.0, len(times))
satellite_ephemeris = {
    "jd": 2450000.0 + times,
    "ra_deg": 270.0 + 12.0 * phase,
    "dec_deg": -20.0 + 4.0 * np.sin(np.pi * phase),
    "distance_au": 0.55 + 0.05 * phase,
}
satellite_table = np.column_stack(tuple(satellite_ephemeris.values()))
parallax_sites = {
    "terrestrial": lcbinint.obs.Site("ground", -29.0, -70.7),
    "space": lcbinint.obs.Site("space", satellite_table),
}
```

## Binary lens, single source

### 1. Source-size baselines

#### Point source

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho': 0.0}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=False, parallax=False, t_ref=7500.0))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_point_source_lightcurve.png" alt="Point source light curve" width="56%"> <img src="figures/HigherOrder_binary_single_point_source_geometry.png" alt="Point source caustics and trajectories" width="40%"></p>

#### Finite source only

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=False, t_ref=7500.0))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_finite_source_only_lightcurve.png" alt="Finite source only light curve" width="56%"> <img src="figures/HigherOrder_binary_single_finite_source_only_geometry.png" alt="Finite source only caustics and trajectories" width="40%"></p>

### 2. Parallax

#### Annual parallax

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax_lightcurve.png" alt="Annual parallax light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax_geometry.png" alt="Annual parallax caustics and trajectories" width="40%"></p>

#### Terrestrial parallax

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, terrestrial=True), site=parallax_sites["terrestrial"])
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_terrestrial_parallax_lightcurve.png" alt="Terrestrial parallax light curve" width="56%"> <img src="figures/HigherOrder_binary_single_terrestrial_parallax_geometry.png" alt="Terrestrial parallax caustics and trajectories" width="40%"></p>

#### Space parallax

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, terrestrial=True), site=parallax_sites["space"])
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_space_parallax_lightcurve.png" alt="Space parallax light curve" width="56%"> <img src="figures/HigherOrder_binary_single_space_parallax_geometry.png" alt="Space parallax caustics and trajectories" width="40%"></p>

### 3. Parallax with additional higher-order effects

#### Annual parallax + circular-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='circular_elements'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__circular_elements_xallarap_lightcurve.png" alt="Annual parallax + circular-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__circular_elements_xallarap_geometry.png" alt="Annual parallax + circular-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'ecc_xa': 0.2, 'peri_xa': 0.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='orbital_elements'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__kepler_elements_xallarap_lightcurve.png" alt="Annual parallax + Kepler-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__kepler_elements_xallarap_geometry.png" alt="Annual parallax + Kepler-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + direct circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='circular_velocity'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__direct_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + direct circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__direct_circular_velocity_xallarap_geometry.png" alt="Annual parallax + direct circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + direct Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='kepler_velocity'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__direct_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + direct Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__direct_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + direct Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__circular_lens_orbit_lightcurve.png" alt="Annual parallax + circular lens orbit light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__circular_lens_orbit_geometry.png" alt="Annual parallax + circular lens orbit caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit + circular-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular', xallarap='circular_elements'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__circular_lens_orbit__circular_elements_xallarap_lightcurve.png" alt="Annual parallax + circular lens orbit + circular-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__circular_lens_orbit__circular_elements_xallarap_geometry.png" alt="Annual parallax + circular lens orbit + circular-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit + Kepler-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'ecc_xa': 0.2, 'peri_xa': 0.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular', xallarap='orbital_elements'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__circular_lens_orbit__kepler_elements_xallarap_lightcurve.png" alt="Annual parallax + circular lens orbit + Kepler-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__circular_lens_orbit__kepler_elements_xallarap_geometry.png" alt="Annual parallax + circular lens orbit + Kepler-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit + direct circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular', xallarap='circular_velocity'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__circular_lens_orbit__direct_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + circular lens orbit + direct circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__circular_lens_orbit__direct_circular_velocity_xallarap_geometry.png" alt="Annual parallax + circular lens orbit + direct circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit + direct Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular', xallarap='kepler_velocity'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__circular_lens_orbit__direct_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + circular lens orbit + direct Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__circular_lens_orbit__direct_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + circular lens orbit + direct Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__kepler_lens_orbit_lightcurve.png" alt="Annual parallax + Kepler lens orbit light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__kepler_lens_orbit_geometry.png" alt="Annual parallax + Kepler lens orbit caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit + circular-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler', xallarap='circular_elements'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__kepler_lens_orbit__circular_elements_xallarap_lightcurve.png" alt="Annual parallax + Kepler lens orbit + circular-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__kepler_lens_orbit__circular_elements_xallarap_geometry.png" alt="Annual parallax + Kepler lens orbit + circular-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit + Kepler-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'ecc_xa': 0.2, 'peri_xa': 0.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler', xallarap='orbital_elements'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__kepler_lens_orbit__kepler_elements_xallarap_lightcurve.png" alt="Annual parallax + Kepler lens orbit + Kepler-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__kepler_lens_orbit__kepler_elements_xallarap_geometry.png" alt="Annual parallax + Kepler lens orbit + Kepler-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit + direct circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler', xallarap='circular_velocity'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__kepler_lens_orbit__direct_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + Kepler lens orbit + direct circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__kepler_lens_orbit__direct_circular_velocity_xallarap_geometry.png" alt="Annual parallax + Kepler lens orbit + direct circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit + direct Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler', xallarap='kepler_velocity'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_single_annual_parallax__kepler_lens_orbit__direct_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + Kepler lens orbit + direct Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_single_annual_parallax__kepler_lens_orbit__direct_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + Kepler lens orbit + direct Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

## Binary lens, binary source

### 1. Source-size baselines

#### Point source

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho1': 0.0, 'rho2': 0.0, 'flux_ratio': 0.4, 't0_2': 7501.2, 'u0_2': -0.06}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=False, parallax=False, t_ref=7500.0))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_point_source_lightcurve.png" alt="Point source light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_point_source_geometry.png" alt="Point source caustics and trajectories" width="40%"></p>

#### Finite source only

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 't0_2': 7501.2, 'u0_2': -0.06}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=False, t_ref=7500.0))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_finite_source_only_lightcurve.png" alt="Finite source only light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_finite_source_only_geometry.png" alt="Finite source only caustics and trajectories" width="40%"></p>

### 2. Parallax

#### Annual parallax

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 't0_2': 7501.2, 'u0_2': -0.06}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax_lightcurve.png" alt="Annual parallax light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax_geometry.png" alt="Annual parallax caustics and trajectories" width="40%"></p>

### 3. Parallax with additional higher-order effects

#### Annual parallax + circular-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='circular_elements'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__circular_elements_xallarap_lightcurve.png" alt="Annual parallax + circular-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__circular_elements_xallarap_geometry.png" alt="Annual parallax + circular-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'ecc_xa': 0.2, 'peri_xa': 0.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='orbital_elements'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_elements_xallarap_lightcurve.png" alt="Annual parallax + Kepler-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_elements_xallarap_geometry.png" alt="Annual parallax + Kepler-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + direct circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='circular_velocity', source_orbit_coordinates='xallarap'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__direct_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + direct circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__direct_circular_velocity_xallarap_geometry.png" alt="Annual parallax + direct circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + direct Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='kepler_velocity', source_orbit_coordinates='xallarap'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__direct_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + direct Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__direct_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + direct Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + trajectory-offset circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7499.4, 'u0': 0.19, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7, 't0_2': 7500.857142857, 'u0_2': 0.214285714}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='circular_velocity', source_orbit_coordinates='trajectory_offset'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__trajectory_offset_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + trajectory-offset circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__trajectory_offset_circular_velocity_xallarap_geometry.png" alt="Annual parallax + trajectory-offset circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + trajectory-offset Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7499.4, 'u0': 0.19, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7, 't0_2': 7500.857142857, 'u0_2': 0.214285714}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='kepler_velocity', source_orbit_coordinates='trajectory_offset'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__trajectory_offset_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + trajectory-offset Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__trajectory_offset_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + trajectory-offset Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 't0_2': 7501.2, 'u0_2': -0.06}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit_lightcurve.png" alt="Annual parallax + circular lens orbit light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit_geometry.png" alt="Annual parallax + circular lens orbit caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit + circular-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular', xallarap='circular_elements'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__circular_elements_xallarap_lightcurve.png" alt="Annual parallax + circular lens orbit + circular-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__circular_elements_xallarap_geometry.png" alt="Annual parallax + circular lens orbit + circular-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit + Kepler-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'ecc_xa': 0.2, 'peri_xa': 0.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular', xallarap='orbital_elements'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__kepler_elements_xallarap_lightcurve.png" alt="Annual parallax + circular lens orbit + Kepler-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__kepler_elements_xallarap_geometry.png" alt="Annual parallax + circular lens orbit + Kepler-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit + direct circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular', xallarap='circular_velocity', source_orbit_coordinates='xallarap'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__direct_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + circular lens orbit + direct circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__direct_circular_velocity_xallarap_geometry.png" alt="Annual parallax + circular lens orbit + direct circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit + direct Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular', xallarap='kepler_velocity', source_orbit_coordinates='xallarap'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__direct_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + circular lens orbit + direct Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__direct_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + circular lens orbit + direct Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit + trajectory-offset circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7499.4, 'u0': 0.19, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7, 't0_2': 7500.857142857, 'u0_2': 0.214285714}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular', xallarap='circular_velocity', source_orbit_coordinates='trajectory_offset'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__trajectory_offset_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + circular lens orbit + trajectory-offset circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__trajectory_offset_circular_velocity_xallarap_geometry.png" alt="Annual parallax + circular lens orbit + trajectory-offset circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + circular lens orbit + trajectory-offset Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7499.4, 'u0': 0.19, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7, 't0_2': 7500.857142857, 'u0_2': 0.214285714}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='circular', xallarap='kepler_velocity', source_orbit_coordinates='trajectory_offset'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__trajectory_offset_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + circular lens orbit + trajectory-offset Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__circular_lens_orbit__trajectory_offset_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + circular lens orbit + trajectory-offset Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 't0_2': 7501.2, 'u0_2': -0.06}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit_lightcurve.png" alt="Annual parallax + Kepler lens orbit light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit_geometry.png" alt="Annual parallax + Kepler lens orbit caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit + circular-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler', xallarap='circular_elements'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__circular_elements_xallarap_lightcurve.png" alt="Annual parallax + Kepler lens orbit + circular-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__circular_elements_xallarap_geometry.png" alt="Annual parallax + Kepler lens orbit + circular-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit + Kepler-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'ecc_xa': 0.2, 'peri_xa': 0.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler', xallarap='orbital_elements'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__kepler_elements_xallarap_lightcurve.png" alt="Annual parallax + Kepler lens orbit + Kepler-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__kepler_elements_xallarap_geometry.png" alt="Annual parallax + Kepler lens orbit + Kepler-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit + direct circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler', xallarap='circular_velocity', source_orbit_coordinates='xallarap'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__direct_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + Kepler lens orbit + direct circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__direct_circular_velocity_xallarap_geometry.png" alt="Annual parallax + Kepler lens orbit + direct circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit + direct Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler', xallarap='kepler_velocity', source_orbit_coordinates='xallarap'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__direct_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + Kepler lens orbit + direct Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__direct_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + Kepler lens orbit + direct Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit + trajectory-offset circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7499.4, 'u0': 0.19, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7, 't0_2': 7500.857142857, 'u0_2': 0.214285714}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler', xallarap='circular_velocity', source_orbit_coordinates='trajectory_offset'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__trajectory_offset_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + Kepler lens orbit + trajectory-offset circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__trajectory_offset_circular_velocity_xallarap_geometry.png" alt="Annual parallax + Kepler lens orbit + trajectory-offset circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler lens orbit + trajectory-offset Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7499.4, 'u0': 0.19, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7, 't0_2': 7500.857142857, 'u0_2': 0.214285714}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='binary', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, orbital_motion='kepler', xallarap='kepler_velocity', source_orbit_coordinates='trajectory_offset'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(7500.0, parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__trajectory_offset_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + Kepler lens orbit + trajectory-offset Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_binary_binary_annual_parallax__kepler_lens_orbit__trajectory_offset_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + Kepler lens orbit + trajectory-offset Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

## Triple lens, single source

### 1. Source-size baselines

#### Point source

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'rho': 0.0}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='single', finite_source=False, parallax=False, t_ref=7500.0))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_single_point_source_lightcurve.png" alt="Point source light curve" width="56%"> <img src="figures/HigherOrder_triple_single_point_source_geometry.png" alt="Point source caustics and trajectories" width="40%"></p>

#### Finite source only

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='single', finite_source=True, parallax=False, t_ref=7500.0))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_single_finite_source_only_lightcurve.png" alt="Finite source only light curve" width="56%"> <img src="figures/HigherOrder_triple_single_finite_source_only_geometry.png" alt="Finite source only caustics and trajectories" width="40%"></p>

### 2. Parallax

#### Annual parallax

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_single_annual_parallax_lightcurve.png" alt="Annual parallax light curve" width="56%"> <img src="figures/HigherOrder_triple_single_annual_parallax_geometry.png" alt="Annual parallax caustics and trajectories" width="40%"></p>

### 3. Parallax with additional higher-order effects

#### Annual parallax + circular-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='circular_elements'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_single_annual_parallax__circular_elements_xallarap_lightcurve.png" alt="Annual parallax + circular-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_triple_single_annual_parallax__circular_elements_xallarap_geometry.png" alt="Annual parallax + circular-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'ecc_xa': 0.2, 'peri_xa': 0.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='orbital_elements'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_single_annual_parallax__kepler_elements_xallarap_lightcurve.png" alt="Annual parallax + Kepler-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_triple_single_annual_parallax__kepler_elements_xallarap_geometry.png" alt="Annual parallax + Kepler-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + direct circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='circular_velocity'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_single_annual_parallax__direct_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + direct circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_triple_single_annual_parallax__direct_circular_velocity_xallarap_geometry.png" alt="Annual parallax + direct circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + direct Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho': 0.004}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='single', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='kepler_velocity'))
magnification = curve(times, parameters)
trajectory = curve.source_trajectory(times, parameters)
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, magnification, color="#0173B2")
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory.x, trajectory.y, color="#0173B2")
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_single_annual_parallax__direct_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + direct Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_triple_single_annual_parallax__direct_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + direct Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

## Triple lens, binary source

### 1. Source-size baselines

#### Point source

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'rho1': 0.0, 'rho2': 0.0, 'flux_ratio': 0.4, 't0_2': 7501.2, 'u0_2': -0.06}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='binary', finite_source=False, parallax=False, t_ref=7500.0))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_binary_point_source_lightcurve.png" alt="Point source light curve" width="56%"> <img src="figures/HigherOrder_triple_binary_point_source_geometry.png" alt="Point source caustics and trajectories" width="40%"></p>

#### Finite source only

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 't0_2': 7501.2, 'u0_2': -0.06}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='binary', finite_source=True, parallax=False, t_ref=7500.0))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_binary_finite_source_only_lightcurve.png" alt="Finite source only light curve" width="56%"> <img src="figures/HigherOrder_triple_binary_finite_source_only_geometry.png" alt="Finite source only caustics and trajectories" width="40%"></p>

### 2. Parallax

#### Annual parallax

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 't0_2': 7501.2, 'u0_2': -0.06}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_binary_annual_parallax_lightcurve.png" alt="Annual parallax light curve" width="56%"> <img src="figures/HigherOrder_triple_binary_annual_parallax_geometry.png" alt="Annual parallax caustics and trajectories" width="40%"></p>

### 3. Parallax with additional higher-order effects

#### Annual parallax + circular-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='circular_elements'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_binary_annual_parallax__circular_elements_xallarap_lightcurve.png" alt="Annual parallax + circular-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_triple_binary_annual_parallax__circular_elements_xallarap_geometry.png" alt="Annual parallax + circular-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + Kepler-elements xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'xi_1': 0.006, 'xi_2': -0.003, 'period_xa': 12.0, 'inc_xa': 0.6, 'ecc_xa': 0.2, 'peri_xa': 0.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='orbital_elements'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_binary_annual_parallax__kepler_elements_xallarap_lightcurve.png" alt="Annual parallax + Kepler-elements xallarap light curve" width="56%"> <img src="figures/HigherOrder_triple_binary_annual_parallax__kepler_elements_xallarap_geometry.png" alt="Annual parallax + Kepler-elements xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + direct circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='circular_velocity', source_orbit_coordinates='xallarap'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_binary_annual_parallax__direct_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + direct circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_triple_binary_annual_parallax__direct_circular_velocity_xallarap_geometry.png" alt="Annual parallax + direct circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + direct Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7500.0, 'u0': 0.2, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='kepler_velocity', source_orbit_coordinates='xallarap'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_binary_annual_parallax__direct_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + direct Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_triple_binary_annual_parallax__direct_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + direct Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + trajectory-offset circular-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7499.4, 'u0': 0.19, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7, 't0_2': 7500.857142857, 'u0_2': 0.214285714}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='circular_velocity', source_orbit_coordinates='trajectory_offset'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_binary_annual_parallax__trajectory_offset_circular_velocity_xallarap_lightcurve.png" alt="Annual parallax + trajectory-offset circular-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_triple_binary_annual_parallax__trajectory_offset_circular_velocity_xallarap_geometry.png" alt="Annual parallax + trajectory-offset circular-velocity xallarap caustics and trajectories" width="40%"></p>

#### Annual parallax + trajectory-offset Kepler-velocity xallarap

```python
parameters = {'s': 0.9, 'q': 0.1, 't0': 7499.4, 'u0': 0.19, 'tE': 30.0, 'alpha': 0.7, 'piEN': 0.03, 'piEE': -0.02, 'g1': 0.011, 'g2': -0.005, 'g3': 0.005, 'lom_szs': 0.2, 'lom_ar': 1.4, 'sep2': 1.3, 'q2': 0.01, 'ang': 0.5, 'xi_1': 0.006, 'xi_2': -0.003, 'w1': 0.004, 'w2': 0.35, 'w3': 0.08, 'xa_szs': 0.2, 'xa_ar': 1.4, 'rho1': 0.004, 'rho2': 0.003, 'flux_ratio': 0.4, 'source_mass_ratio': 0.7, 't0_2': 7500.857142857, 'u0_2': 0.214285714}
curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(lens='triple', source='binary', finite_source=True, parallax=True, t_ref=7500.0, sky=sky, xallarap='kepler_velocity', source_orbit_coordinates='trajectory_offset'))
components = curve.binary_source_components(times, parameters)
magnification = components.total
trajectory1 = components.source1.trajectory
trajectory2 = components.source2.trajectory
caustics = curve.caustics(parameters)
```

```python
plt.figure(figsize=(3.8, 2.4))
plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, label="source 1")
plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, label="source 2")
plt.plot(times, magnification, color="black", label="total")
plt.legend(loc="upper left", fontsize=7)
plt.xlabel("Time"); plt.ylabel("Magnification")
plt.show()
```

```python
plt.figure(figsize=(2.8, 2.7))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(x, y, color="#6C6C6C", lw=1.1)
plt.plot(trajectory1.x, trajectory1.y, color="#0173B2", label="source 1")
plt.plot(trajectory2.x, trajectory2.y, color="#029E73", label="source 2")
plt.legend(fontsize=7)
plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
plt.axis("equal"); plt.show()
```

<p><img src="figures/HigherOrder_triple_binary_annual_parallax__trajectory_offset_kepler_velocity_xallarap_lightcurve.png" alt="Annual parallax + trajectory-offset Kepler-velocity xallarap light curve" width="56%"> <img src="figures/HigherOrder_triple_binary_annual_parallax__trajectory_offset_kepler_velocity_xallarap_geometry.png" alt="Annual parallax + trajectory-offset Kepler-velocity xallarap caustics and trajectories" width="40%"></p>

[← Higher-order effects](CombinedEffects.md)
