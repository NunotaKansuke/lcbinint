# 3. Lens orbital motion: evolve the caustics

Lens orbital motion makes the projected separation and orientation time
dependent. This reproduces VBM's circular-orbit example with `s=0.9`, `q=0.1`,
`u0=0`, `alpha=1`, `rho=0.01`, `tE=30`, `t0=7500`, and velocity components
`g1=0.011`, `g2=-0.005`, `g3=0.005`. The example compares static and orbiting
light curves, then overlays caustics at three epochs using matching source
position markers.

```python
import numpy as np
import matplotlib.pyplot as plt
import lcbinint

times = np.linspace(7470.0, 7530.0, 300)
options = lcbinint.Options(coordinates="vbm", nbin="auto", caustic_bins=600)
static_model = lcbinint.Model(lens="binary", source="single")
orbit_model = lcbinint.Model(
    lens="binary", source="single", orbital_motion="circular", t_ref=7500.0,
)
static = lcbinint.LightCurve(options=options, model=static_model)
orbit = lcbinint.LightCurve(options=options, model=orbit_model)

params = dict(
    t0=7500.0, tE=30.0, u0=0.0, alpha=1.0, s=0.9, q=0.1,
    rho=0.01, g1=0.011, g2=-0.005, g3=0.005,
)

static_mag = static(times, params)
orbit_mag = orbit(times, params)
trajectory = orbit.source_trajectory(times, params)
epochs = (7480.0, 7500.0, 7520.0)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(epochs)))

# Notebook cell 1: static vs circular-orbit light curve.
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(times, static_mag, "--", color="0.45", label="static lens")
ax.plot(times, orbit_mag, color="C0", label="circular lens orbit")
ax.set(xlabel="time", ylabel="magnification", title="Orbital-motion light curve")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("orbital-motion-light-curve.png", dpi=170)

# Notebook cell 2: caustics at three epochs and their source positions.
fig, ax = plt.subplots(figsize=(7, 5))
for epoch, color in zip(epochs, colors):
    caustics = orbit.caustics(epoch, params)
    for branch_index, (x, y) in enumerate(zip(caustics.x, caustics.y)):
        ax.plot(x, y, color=color, lw=1.2,
                label=f"t = {epoch:g}" if branch_index == 0 else None)
    index = np.argmin(np.abs(times - epoch))
    ax.scatter(trajectory.x[index], trajectory.y[index], color=color, s=24)
ax.plot(trajectory.x, trajectory.y, color="C0", lw=1.5)
ax.set(xlabel="lens-frame x", ylabel="lens-frame y",
       title="Source trajectory and evolving caustics")
ax.set_aspect("equal", adjustable="box")
ax.legend(title="caustic epoch", frameon=False)
fig.tight_layout()
fig.savefig("orbital-motion-geometry.png", dpi=170)
```

![Circular lens-orbit light-curve comparison](../assets/tutorials/03-orbital-motion-light-curve.png)

![Circular lens orbit: source trajectory with caustics at three epochs](../assets/tutorials/03-orbital-motion-geometry.png)

```sh
PYTHONPATH=build python example/tutorials/orbital_motion.py
```

The complete script explicitly creates `options`, `static_model`, and
`orbit_model` before either `LightCurve`, then uses
`curve.caustics(epoch, params)` for the three caustic snapshots. Copy it as a whole from
[`example/tutorials/orbital_motion.py`](../../example/tutorials/orbital_motion.py).

For Keplerian motion, use `orbital_motion="kepler"` and include `lom_szs` and
`lom_ar`.

---

**Course navigation:** [← Previous: Parallax](02-parallax.md) · [Tutorial home](../README.md) · [Next: Xallarap →](04-xallarap.md)
