[Back to documentation](readme.md)

# Limb darkening

## Linear limb darkening

```python
import lcbinint

s = 0.8
q = 0.1
y1 = 0.01
y2 = 0.01
rho = 0.01
a1 = 0.51

mag = lcbinint.binary_ray_shooting(
    y1, y2, s=s, q=q, rho=rho,
    limb_darkening=lcbinint.LimbDarkening.linear(a1),
)
print("Magnification with limb-darkened source =", mag)
```

## Quadratic limb darkening

```python
a1 = 0.51
a2 = 0.3

mag = lcbinint.binary_ray_shooting(
    y1, y2, s=s, q=q, rho=rho,
    limb_darkening=lcbinint.LimbDarkening.quadratic(a1, a2),
)
print("Magnification with quadratic limb-darkened source =", mag)
```

Square-root, logarithmic, user-defined, and simultaneous multi-band profiles
are not exposed by the current LCBinInt Python API.
