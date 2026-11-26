"""Public adaptive Cartesian checks for unresolved image tips."""
import lcbinint
import pytest


def evaluate(limb, absolute_tolerance, max_bins):
    curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            coordinates="center_of_mass",
            finite_source_tol=absolute_tolerance,
            finite_source_reltol=1e-4,
            max_source_bins=max_bins,
        ),
    )
    result = curve.info(
        [0.7360277971408414], t0=0., tE=1., alpha=0.,
        u0=1.071502221810499, s=0.4907165561511101,
        q=1 / 0.2765108317331833, rho=0.31787435464773617,
        limb_darkening_c=limb,
    )
    assert result.finite_source_method_names[0] == "inverse_ray_cartesian"
    return result


@pytest.mark.parametrize("limb, reference", [
    (0., 1.2958200054997626),    # saved six-panel reference, geom_id 5026
    (0.5, 1.2966894833090212),  # same geometry, geom_id 5030
])
def test_cartesian_short_image_precision(limb, reference):
    result = evaluate(limb, absolute_tolerance=0., max_bins=512)
    assert result.finite_source_converged[0]
    assert result.finite_source_magnifications[0] == pytest.approx(reference, rel=1e-4)


def test_cartesian_mixed_tolerance_with_limited_grid():
    relative_only = evaluate(0., absolute_tolerance=0., max_bins=16)
    mixed = evaluate(0., absolute_tolerance=0.02, max_bins=16)
    assert not relative_only.finite_source_converged[0]
    assert mixed.finite_source_converged[0]
    assert 1e-4 * mixed.finite_source_magnifications[0] < mixed.finite_source_error_estimates[0] <= 0.02


@pytest.mark.parametrize("limb, reference", [
    (0., 9.293930294789611),
    (0.5, 8.664080145886523),
])
def test_cartesian_unresolved_tip_precision(limb, reference):
    # Independent VBM RelTol=1e-8 references for geometries 2978 and 2982.
    curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            coordinates="center_of_mass",
            finite_source_tol=0.,
            finite_source_reltol=1e-4,
            # The uniform image is resolved on the initial grid once
            # both physical endpoints enter the one-missing-row rule.
            max_source_bins=16 if limb == 0. else 512,
        ),
    )
    result = curve.info(
        [-0.144430535716425], t0=0., tE=1., alpha=0.,
        u0=0.02245630760623187, s=1.0453856112914581,
        q=1 / 0.0026130078448480553, rho=0.00010827507294018582,
        limb_darkening_c=limb,
    )
    assert result.finite_source_method_names[0] == "inverse_ray_cartesian"
    assert result.finite_source_converged[0]
    assert result.finite_source_magnifications[0] == pytest.approx(reference, rel=1e-4)


def test_cartesian_unresolved_endpoint_remainder():
    # Geometry 2498 still needs targeted refinement after tip relocation.
    curve = lcbinint.LightCurve(lens="binary", options=lcbinint.Options(
        coordinates="center_of_mass", finite_source_tol=0., finite_source_reltol=1e-4,
    ))
    info = curve.info(
        [3.645634812663749], t0=0., tE=1., alpha=0., u0=0.056926348374955194,
        s=0.2413441410894618, q=1 / 0.000367760840816759, rho=0.4319678301129469,
    )
    assert info.finite_source_method_names[0] == "inverse_ray_cartesian"
    assert info.finite_source_converged[0]
    assert info.finite_source_magnifications[0] == pytest.approx(1.0088242358890631, rel=1e-4)
