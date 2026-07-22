[Back to documentation](readme.md)

# Binary lenses

## Binary lensing with point sources

```python
import lcbinint

s = 0.8
q = 0.1
y1 = 0.01
y2 = 0.01

image_plane = lcbinint.image.ImagePlane(
    q=q, s=s, x=y1, y=y2, coordinates="lcbinint"
)
images = image_plane.image_table()
magnification = images["magnification"].sum()
print("Magnification of a point source =", magnification)
```

## Binary lensing with extended sources

```python
rho = 0.01

magnification = lcbinint.binary_ray_shooting(
    y1,
    y2,
    s=s,
    q=q,
    rho=rho,
    options=lcbinint.Options(tol=1e-3),
)
print("Binary-lens magnification =", magnification)
```

The current finite-source result exposes magnification diagnostics but not a
finite-source astrometric centroid, so the astrometry example is not replaced
with a point-source approximation.

[Go to Critical curves and caustics](CriticalCurvesAndCaustics.md)
