"""Regression checks for the Bennett partial-element boundary rule."""

import pytest


def _single_lens_limit(
    lcbinint, delta_c, limb_darkening_c, grid="cartesian", source_bins=256
):
    options = lcbinint.Options(
        coordinates="center_of_mass",
        inverse_ray_grid=grid,
        source_bins=source_bins,
        max_source_bins=source_bins,
        bennett_delta_c=delta_c,
    )
    curve = lcbinint.LightCurve(
        lens="binary",
        options=options,
        limb_darkening_c=limb_darkening_c,
    )
    # q=1e-8 is an independent single-lens limit for this integration check.
    return curve.magnification(
        0.1,
        t0=0.0,
        tE=1.0,
        u0=0.0,
        alpha=0.0,
        s=1.0,
        q=1.0e-8,
        rho=0.02,
    ).item()


def test_bennett_delta_c_is_public_and_reaches_the_native_options():
    lcbinint = pytest.importorskip("lcbinint")
    assert lcbinint.Options().bennett_delta_c == pytest.approx(0.15)
    assert lcbinint.Options(bennett_delta_c=0.037).bennett_delta_c == pytest.approx(0.037)


def test_guarded_partial_element_rule_improves_a_limb_darkened_limit():
    lcbinint = pytest.importorskip("lcbinint")
    uniform_default = _single_lens_limit(lcbinint, 0.15, 0.0)
    uniform_formal = _single_lens_limit(lcbinint, 0.0, 0.0)
    # For a uniform source A+B=1/2+delta in both branches, so delta_c changes
    # no result apart from roundoff.
    assert uniform_default == pytest.approx(uniform_formal, abs=1.0e-12)

    reference = 10.0843013911
    guarded = _single_lens_limit(lcbinint, 0.15, 0.5)
    unguarded = _single_lens_limit(lcbinint, 1.0, 0.5)
    assert abs(guarded - reference) < 5.0e-5
    assert abs(guarded - reference) < 0.2 * abs(unguarded - reference)


def test_guarded_partial_element_rule_is_used_by_the_polar_radial_integrator():
    lcbinint = pytest.importorskip("lcbinint")
    reference = 10.0843013911
    guarded = _single_lens_limit(lcbinint, 0.15, 0.5, grid="polar")
    formal = _single_lens_limit(lcbinint, 0.0, 0.5, grid="polar")
    # The polar path integrates the same Bennett rule in its radial runs.  Its
    # coarser angular geometry has a different absolute error, so pin the
    # direction of the independent single-lens improvement rather than reuse
    # the Cartesian absolute threshold above.
    assert abs(guarded - reference) < abs(formal - reference)


def test_outer_equation_15_is_active_on_a_clean_cartesian_arc():
    lcbinint = pytest.importorskip("lcbinint")
    reference = 10.0843013911
    value = _single_lens_limit(
        lcbinint, 0.15, 0.5, grid="cartesian", source_bins=32
    )
    # The coarse clean arc is the controlled case in which the Eq. (15)
    # endpoint replacement is expected to be active.  Without it the same
    # 32-bin calculation is about 4.5e-4 below the reference; with it the
    # error is below 1e-5.
    assert abs(value - reference) < 2.0e-5
