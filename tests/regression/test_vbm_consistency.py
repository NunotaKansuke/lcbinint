import math

import pytest


BINARY_POINT_CASES = [
    pytest.param(1.0, 0.1, 0.2, 0.1, 5.871444912771214, id="resonant_low_q"),
    pytest.param(0.7, 0.3, -0.4, 0.2, 2.116643550532278, id="close_binary"),
    pytest.param(1.5, 1.0, 0.05, -0.2, 1.5493462433112466, id="wide_equal_mass"),
    pytest.param(1.0, 1.0e-3, 0.3, 0.4, 2.1789388609029046, id="planetary"),
]


def _vbm_binary_mag0(separation, mass_ratio, y1, y2):
    module = pytest.importorskip("VBBinaryLensing")
    vbbinary_lensing = module.VBBinaryLensing()
    return vbbinary_lensing.BinaryMag0(separation, mass_ratio, y1, y2)


def _lcbinint_binary_mag0(separation, mass_ratio, y1, y2):
    lcbinint = pytest.importorskip("lcbinint")

    if hasattr(lcbinint, "binary_mag0"):
        return lcbinint.binary_mag0(separation, mass_ratio, y1, y2)

    raise NotImplementedError(
        "lcbinint binary point-source Python API is not implemented yet"
    )


def _lcbinint_lens_model_mag0(separation, mass_ratio, y1, y2):
    lcbinint = pytest.importorskip("lcbinint")

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=y2,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
    )
    options = lcbinint.Options(center_of_mass=1)

    return lcbinint.LensModel(params, options).magnification(y1)


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,expected", BINARY_POINT_CASES)
def test_vbm_binary_reference_values_are_stable(
    separation, mass_ratio, y1, y2, expected
):
    actual = _vbm_binary_mag0(separation, mass_ratio, y1, y2)

    assert math.isfinite(actual)
    assert math.isclose(actual, expected, rel_tol=5.0e-13, abs_tol=5.0e-13)


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,expected", BINARY_POINT_CASES)
def test_lcbinint_binary_point_source_matches_vbm(
    separation, mass_ratio, y1, y2, expected
):
    del expected

    reference = _vbm_binary_mag0(separation, mass_ratio, y1, y2)
    actual = _lcbinint_binary_mag0(separation, mass_ratio, y1, y2)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-10, abs_tol=1.0e-11)


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,expected", BINARY_POINT_CASES)
def test_lcbinint_lens_model_binary_point_source_matches_vbm(
    separation, mass_ratio, y1, y2, expected
):
    del expected

    reference = _vbm_binary_mag0(separation, mass_ratio, y1, y2)
    actual = _lcbinint_lens_model_mag0(separation, mass_ratio, y1, y2)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-10, abs_tol=1.0e-11)


def test_lcbinint_lens_model_wide_binary_legacy_offset_matches_vbm():
    separation = 1.5
    mass_ratio = 1.0
    y1 = 0.2
    y2 = 0.1
    m2 = mass_ratio / (1.0 + mass_ratio)
    legacy_offset = m2 * separation - m2 / separation

    reference = _vbm_binary_mag0(separation, mass_ratio, y1 - legacy_offset, y2)

    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=y2,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
    )
    options = lcbinint.Options(center_of_mass=0)
    actual = lcbinint.LensModel(params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-10, abs_tol=1.0e-11)
