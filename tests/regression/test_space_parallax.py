"""VBMicrolensing-compatible geocentric spacecraft parallax."""

import math

import numpy as np
import pytest


def test_space_site_requires_vbm_table_shape():
    lcbinint = pytest.importorskip("lcbinint")
    with pytest.raises(ValueError, match="shape"):
        lcbinint.obs.Site("space", np.zeros((2, 3)))
    with pytest.raises(TypeError, match="requires 'ground' or 'space'"):
        lcbinint.obs.Site()


def test_space_site_source_trajectory_matches_vbm(tmp_path):
    lcbinint = pytest.importorskip("lcbinint")
    vbm_module = pytest.importorskip("VBBinaryLensing")
    if not hasattr(vbm_module, "VBBinaryLensing"):
        pytest.skip("requires the VBBinaryLensing Python binding")

    # VBM takes reduced HJD times, whereas its satellite table stores full JD.
    tref = 9000.0
    table = np.array([
        [2458998.0, 0.0, 0.0, 0.010],
        [2458999.0, 0.0, 0.0, 0.011],
        [2459000.0, 0.0, 0.0, 0.012],
        [2459001.0, 0.0, 0.0, 0.013],
        [2459002.0, 0.0, 0.0, 0.014],
    ])
    satellite_dir = tmp_path / "satellites"
    satellite_dir.mkdir()
    satellite_path = satellite_dir / "satellite1.txt"
    satellite_path.write_text(
        "$$SOE\n" + "\n".join(" ".join(map(str, row)) + " 0.0" for row in table) + "\n$$EOE\n"
    )
    coordinate_path = tmp_path / "event.coordinates"
    coordinate_path.write_text("18:00:00\n-30:00:00\n")

    vbm = vbm_module.VBBinaryLensing()
    vbm.SetObjectCoordinates(str(coordinate_path), str(satellite_dir))
    vbm.parallaxsystem = 1
    vbm.t0_par_fixed = 1
    vbm.t0_par = tref
    vbm.satellite = 1

    times = np.array([8998.2, 9000.0, 9001.6])
    pi_north, pi_east = 0.1, 0.05
    _, y1, y2 = vbm.PSPLLightCurveParallax(
        [0.1, math.log(20.0), tref, pi_north, pi_east], times.tolist()
    )
    vbm.satellite = 0
    _, earth_y1, earth_y2 = vbm.PSPLLightCurveParallax(
        [0.1, math.log(20.0), tref, pi_north, pi_east], times.tolist()
    )

    curve = lcbinint.LightCurve(
        parallax=True,
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        site=lcbinint.obs.Site("space", table),
        t_ref=tref,
    )
    trajectory = curve.source_trajectory(
        times,
        t0=tref,
        tE=20.0,
        u0=0.1,
        s=1.1,
        q=0.2,
        rho=0.0,
        piEN=pi_north,
        piEE=pi_east,
    )
    earth_trajectory = lcbinint.LightCurve(
        parallax=True,
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        t_ref=tref,
    ).source_trajectory(
        times,
        t0=tref,
        tE=20.0,
        u0=0.1,
        s=1.1,
        q=0.2,
        rho=0.0,
        piEN=pi_north,
        piEE=pi_east,
    )

    # Compare the satellite-only source-trajectory displacement. This cancels
    # the intentionally different Earth ephemerides while directly checking
    # VBM's geocentric satellite-table convention.
    np.testing.assert_allclose(
        np.asarray(trajectory.x) - np.asarray(earth_trajectory.x),
        -np.asarray(y1) + np.asarray(earth_y1), rtol=0.0, atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(trajectory.y) - np.asarray(earth_trajectory.y),
        -np.asarray(y2) + np.asarray(earth_y2), rtol=0.0, atol=1.0e-12,
    )


def test_light_curves_share_one_model_but_each_keeps_its_site():
    lcbinint = pytest.importorskip("lcbinint")
    model = lcbinint.Model(
        parallax=True,
        terrestrial=True,
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        t_ref=2459000.0,
    )
    ground = lcbinint.LightCurve(
        model=model, site=lcbinint.obs.Site("ground", -29.0, 70.0)
    )
    space = lcbinint.LightCurve(
        model=model,
        site=lcbinint.obs.Site("space", [[2458999.0, 270.0, -30.0, 0.01],
                                            [2459001.0, 270.0, -30.0, 0.01]]),
    )

    assert ground.model is model
    assert space.model is model
    assert ground.site.kind == "ground"
    assert space.site.kind == "space"

    # The physical model is not copied at LightCurve construction time.
    model.parallax = False
    assert not ground.parallax
    assert not space.parallax
