import math

import pytest


def test_binary_image_positions_match_light_curve_point_magnification():
    lcbinint = pytest.importorskip("lcbinint")

    q = 0.1
    s = 1.0
    x = 0.2
    y = 0.1

    plane = lcbinint.image.binary(q=q, s=s, x=x, y=y)
    images = plane.images()
    table = plane.image_table()

    assert images.shape[1] == 2
    assert len(images) in (3, 5)
    assert len(table) == len(images)
    assert set(table.dtype.names) == {
        "x", "y", "magnification", "jacobian_determinant", "parity",
    }

    curve = lcbinint.LightCurve(options=lcbinint.Options(coordinates="vbm"))
    info = curve.info(
        [x],
        t0=0.0,
        tE=1.0,
        u0=y,
        alpha=0.0,
        s=s,
        q=q,
        rho=0.0,
    )

    assert table["magnification"].sum() == pytest.approx(
        info.point_source_magnifications[0]
    )
    assert all(math.isfinite(value) for value in table["jacobian_determinant"])


def test_image_plane_geometry_helpers_return_branches():
    lcbinint = pytest.importorskip("lcbinint")

    plane = lcbinint.ImagePlane(q=1.0e-3, s=1.0, x=0.01, y=-0.02, n_points=64)

    caustics = plane.caustics()
    critical_curves = plane.critical_curves()

    assert [len(branch) for branch in caustics.x] == [64, 64, 64, 64]
    assert [len(branch) for branch in critical_curves.x] == [64, 64, 64, 64]


def test_ray_shooting_images_return_finite_source_regions_with_seeds():
    lcbinint = pytest.importorskip("lcbinint")

    plane = lcbinint.ImagePlane(
        q=1.0e-3, s=1.0, x=0.01, y=-0.02, rho=2.0e-3, n_points=64
    )

    seeds = plane.seeds()
    regions = plane.ray_shooting_images(resolution=180)

    assert len(regions) == len(seeds)
    assert any(len(region.points) > 0 for region in regions)
    for region, seed in zip(regions, seeds):
        assert region.seed.tolist() == pytest.approx(seed.tolist())
        assert region.points.shape[1] == 2
        assert region.parity in (-1, 1)


def test_image_plane_plot_smoke():
    pytest.importorskip("matplotlib")
    lcbinint = pytest.importorskip("lcbinint")

    import matplotlib

    matplotlib.use("Agg")
    plane = lcbinint.image.binary(
        q=1.0e-3, s=1.0, x=0.01, y=-0.02, rho=2.0e-3, n_points=32
    )
    ax = plane.plot(image_resolution=120)

    assert ax.get_aspect() == 1.0
    assert ax.get_legend() is None
