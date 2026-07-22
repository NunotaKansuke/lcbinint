[Back to documentation](readme.md)

# Critical curves and caustics

> VBMicrolensing correspondence: [CriticalCurvesAndCaustics.md](https://github.com/valboz/VBMicrolensing/blob/main/docs/python/CriticalCurvesAndCaustics.md). The binary example retains `s=0.6` and `q=0.1`.

The two geometries are deliberately calculated and plotted in different code
blocks.

## Binary Lens

Calculate and plot the caustics:

```python
import numpy as np
from matplotlib import pyplot as plt
import lcbinint

s = 0.6
q = 0.1
params = {"s": s, "q": q}

curve = lcbinint.LightCurve(options=lcbinint.Options(caustic_bins=200))
caustics = curve.caustics(params)

fig = plt.figure(figsize=(5, 5))
for x, y in zip(caustics.x, caustics.y):
    plt.scatter(-np.asarray(x), -np.asarray(y), s=3, color="k")
plt.axis("equal")
plt.show()
```

![Binary-lens caustics](figures/Caustics_binary.png)

Calculate and plot the critical curves in a new block:

```python
critical_curves = curve.critical_curves(params)

fig = plt.figure(figsize=(5, 5))
for x, y in zip(critical_curves.x, critical_curves.y):
    plt.plot(-np.asarray(x), -np.asarray(y), "k")
plt.axis("equal")
plt.show()
```

![Binary-lens critical curves](figures/Criticalcurves_binary.png)

The arbitrary four-lens geometry example is not included because the current
`lcbinint` model selector supports binary and triple lenses only.
