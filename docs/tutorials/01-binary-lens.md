# 1. Binary lens: the baseline event

This reproduces the binary-lens quick-start configuration in the
VBMicrolensing documentation: `s=0.9`, `q=0.1`, `u0=0`, `alpha=1`,
`rho=0.01`, `tE=30`, and `t0=7500`. lcbinint takes direct named values rather
than VBM's positional array containing logarithms.

```python
import numpy as np
import matplotlib.pyplot as plt
import lcbinint

times = np.linspace(7470.0, 7530.0, 300)
params = dict(t0=7500.0, tE=30.0, u0=0.0, alpha=1.0,
              s=0.9, q=0.1, rho=0.01)
options = lcbinint.Options(coordinates="vbm", nbin="auto", caustic_bins=600)
model = lcbinint.Model(lens="binary", source="single")
curve = lcbinint.LightCurve(options=options, model=model)

magnification = curve(times, params)
trajectory = curve.source_trajectory(times, params)
caustics = curve.caustics(params)

# Notebook cell 1: light curve.
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(times, magnification, color="C0")
ax.set(xlabel="time", ylabel="magnification", title="Binary-lens light curve")
fig.tight_layout()
fig.savefig("binary-lens-light-curve.png", dpi=170)

# Notebook cell 2: source trajectory and caustics.
fig, ax = plt.subplots(figsize=(7, 5))
for x, y in zip(caustics.x, caustics.y):
    ax.plot(x, y, color="C3", lw=1.2)
ax.plot(trajectory.x, trajectory.y, color="C0", lw=1.5)
ax.scatter(trajectory.x[0], trajectory.y[0], color="C0", s=20)
ax.set(xlabel="lens-frame x", ylabel="lens-frame y",
       title="Source trajectory and caustics")
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
fig.savefig("binary-lens-geometry.png", dpi=170)
```

![Binary-lens light curve](../assets/tutorials/01-binary-lens-light-curve.png)

![Binary-lens source trajectory and caustics](../assets/tutorials/01-binary-lens-geometry.png)

The block is self-contained: paste it into a notebook cell or save it as a
Python file, and it writes both figures. The repository copy is
[`example/tutorials/binary_lens.py`](../../example/tutorials/binary_lens.py).

The script uses the VBM example's finite source (`rho=0.01`) so its
caustic-crossing peak remains visible in a sampled plot. Select limb darkening if needed, and
inspect `curve.info(...).finite_source_converged` when setting an explicit
accuracy budget.

---

**Course navigation:** [Tutorial home](../README.md) · [Next: Parallax →](02-parallax.md)
