import math

import numpy as np
import pytest


def test_parameter_batch_matches_scalar_light_curve_exactly():
    lcbinint = pytest.importorskip("lcbinint")
    curve = lcbinint.LightCurve()
    times = np.linspace(-1.0, 1.0, 41)
    rows = [
        {
            "t0": 0.01 * index,
            "tE": 10.0,
            "u0": 0.08 + 0.01 * index,
            "s": 1.2,
            "q": 0.05,
            "alpha": 0.3,
            "rho": 0.0,
        }
        for index in range(8)
    ]

    expected = np.asarray([curve(times, row) for row in rows])
    actual = curve.magnification_batch(times, rows)

    assert actual.shape == (len(rows), len(times))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(curve.magnification_batch(times, rows), actual)


def test_parameter_batch_accepts_parameters_and_empty_input():
    lcbinint = pytest.importorskip("lcbinint")
    curve = lcbinint.LightCurve()
    times = np.asarray([-0.5, 0.0, 0.5])
    parameter = lcbinint.Parameters(
        t0=0.0, tE=10.0, u0=0.1, s=1.2, q=0.05, alpha=0.3, rho=0.0
    )

    batch = curve.magnification_batch(times, [parameter, parameter])
    np.testing.assert_array_equal(batch[0], curve(times, parameter))
    np.testing.assert_array_equal(batch[1], batch[0])
    assert curve.magnification_batch(times, []).shape == (0, len(times))


def test_binary_source_native_batch_preserves_scalar_semantics():
    lcbinint = pytest.importorskip("lcbinint")
    curve = lcbinint.LightCurve(source="binary")
    times = np.linspace(-0.5, 0.5, 7)
    rows = [
        {
            "t0": 0.0,
            "tE": 10.0,
            "u0": 0.1,
            "s": 1.2,
            "q": 0.05,
            "alpha": 0.3,
            "rho": 0.0,
            "q_source": 0.2,
            "t0_2": 0.1 + 0.01 * index,
            "u0_2": 0.2,
        }
        for index in range(3)
    ]

    expected = np.asarray([curve(times, row) for row in rows])
    np.testing.assert_array_equal(curve.magnification_batch(times, rows), expected)

    flux = 120.0 * expected[0] + 4.0
    flux += 0.01 * np.cos(np.arange(len(times)))
    error = np.full(len(times), 0.1)
    fused = curve.light_curve_log_likelihood_batch(
        times, flux, error, rows, "gaussian", "fit"
    )
    weights = 1.0 / error**2
    for index, magnification in enumerate(expected):
        design = np.column_stack((magnification, np.ones(len(times))))
        normal = design.T @ (weights[:, None] * design)
        source, blend = np.linalg.solve(normal, design.T @ (weights * flux))
        residual = (flux - source * magnification - blend) / error
        np.testing.assert_allclose(
            fused["log_likelihood"][index], -0.5 * residual @ residual,
            rtol=2e-9,
        )


@pytest.mark.parametrize("flux_mode", ["fit", "marginalize"])
def test_fused_gaussian_likelihood_matches_numpy_reference(flux_mode):
    lcbinint = pytest.importorskip("lcbinint")
    curve = lcbinint.LightCurve()
    times = np.linspace(-0.8, 0.8, 23)
    rows = [
        {
            "t0": 0.01 * index,
            "tE": 10.0,
            "u0": 0.08 + 0.01 * index,
            "s": 1.2,
            "q": 0.05,
            "alpha": 0.3,
            "rho": 0.0,
        }
        for index in range(4)
    ]
    magnifications = curve.magnification_batch(times, rows)
    flux = 200.0 * magnifications[0] + 7.0
    flux += 0.02 * np.sin(np.arange(len(times)))
    error = np.full(len(times), 0.2)
    actual = curve.light_curve_log_likelihood_batch(
        times, flux, error, rows, "gaussian", flux_mode
    )

    weights = 1.0 / error**2
    for index, magnification in enumerate(magnifications):
        design = np.column_stack((magnification, np.ones(len(times))))
        normal = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * flux)
        source, blend = np.linalg.solve(normal, rhs)
        residual = (flux - source * magnification - blend) / error
        chi2 = residual @ residual
        np.testing.assert_allclose(actual["source_flux"][index], source, rtol=2e-11)
        np.testing.assert_allclose(actual["blend_flux"][index], blend, rtol=2e-11)
        if flux_mode == "fit":
            expected = -0.5 * chi2
        else:
            expected = (
                -0.5 * (len(times) - 2) * np.log(chi2)
                - 0.5 * np.log(np.linalg.det(normal))
            )
        np.testing.assert_allclose(actual["log_likelihood"][index], expected, rtol=2e-9)


def test_fused_sampled_flux_ignores_inference_flux_parameter_names():
    lcbinint = pytest.importorskip("lcbinint")
    curve = lcbinint.LightCurve()
    times = np.linspace(-0.4, 0.4, 9)
    rows = [
        {
            "t0": 0.0,
            "tE": 10.0,
            "u0": 0.1,
            "s": 1.2,
            "q": 0.05,
            "alpha": 0.3,
            "rho": 0.0,
            "Fs_survey": 100.0,
            "Fb_survey": 5.0,
        }
    ]
    lens_row = {
        key: value
        for key, value in rows[0].items()
        if not key.startswith(("Fs_", "Fb_"))
    }
    magnification = curve(times, lens_row)
    flux = 100.0 * magnification + 5.0
    result = curve.light_curve_log_likelihood_batch(
        times,
        flux,
        np.ones(len(times)),
        rows,
        "student_t",
        "sample",
        4.0,
        np.asarray([100.0]),
        np.asarray([5.0]),
    )
    assert result["log_likelihood"][0] == pytest.approx(
        len(times)
        * (
            math.lgamma(2.5)
            - math.lgamma(2.0)
            - 0.5 * np.log(4.0 * np.pi)
        )
    )
