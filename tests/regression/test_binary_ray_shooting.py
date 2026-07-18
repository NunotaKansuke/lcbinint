"""Direct finite-source binary-lens integration API."""

import pytest


def test_binary_ray_shooting_matches_vbm_binary_mag2():
    lcbinint = pytest.importorskip("lcbinint")
    vbm_module = pytest.importorskip("VBBinaryLensing")

    s, q, x, y, rho = 1.2, 0.05, 0.1, -0.05, 0.01
    vbm = vbm_module.VBBinaryLensing()
    expected = vbm.BinaryMag2(s, q, x, y, rho)
    actual = lcbinint.binary_ray_shooting(x, y, s=s, q=q, rho=rho)

    assert actual == pytest.approx(expected, rel=2.0e-5)


def test_binary_ray_shooting_requires_a_finite_source():
    lcbinint = pytest.importorskip("lcbinint")
    with pytest.raises(ValueError, match="positive rho"):
        lcbinint.binary_ray_shooting(0.0, 0.0, s=1.0, q=0.1, rho=0.0)
