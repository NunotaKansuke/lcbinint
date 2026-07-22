[Back to documentation](readme.md)

# Accuracy control

Use the same lens and source values while changing only the requested absolute
accuracy:

```python
import lcbinint

s, q, y1, y2, rho = 0.8, 0.1, 0.01, 0.01, 0.01

mag_1e3 = lcbinint.binary_ray_shooting(
    y1, y2, s=s, q=q, rho=rho,
    options=lcbinint.Options(tol=1e-3),
)
print("Magnification (accuracy at 1.e-3) =", mag_1e3)

mag_1e4 = lcbinint.binary_ray_shooting(
    y1, y2, s=s, q=q, rho=rho,
    options=lcbinint.Options(tol=1e-4),
)
print("Magnification (accuracy at 1.e-4) =", mag_1e4)
```

## Precision control

```python
mag_rel_1e1 = lcbinint.binary_ray_shooting(
    y1, y2, s=s, q=q, rho=rho,
    options=lcbinint.Options(reltol=1e-1),
)
print("Magnification (relative precision at 1.e-1) =", mag_rel_1e1)
```
