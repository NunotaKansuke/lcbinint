import pytest


def test_light_curve_data_k_emin_define_effective_errors_and_weights():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    flux_err = np.array([3.0, 4.0])
    data = lcbinint.obs.LightCurveData(
        np.array([0.0, 1.0]),
        np.array([10.0, 11.0]),
        flux_err,
        k=2.0,
        emin=1.0,
    )

    expected_sigma = 2.0 * np.sqrt(flux_err * flux_err + 1.0)
    assert data.k == pytest.approx(2.0)
    assert data.emin == pytest.approx(1.0)
    assert data.flux_err.tolist() == pytest.approx(flux_err.tolist())
    assert data.effective_flux_err.tolist() == pytest.approx(expected_sigma.tolist())
    assert data.weight.tolist() == pytest.approx((1.0 / expected_sigma**2).tolist())


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"k": 0.0}, "k must be finite and positive"),
        ({"k": -1.0}, "k must be finite and positive"),
        ({"emin": -0.1}, "emin must be finite and non-negative"),
    ],
)
def test_light_curve_data_rejects_invalid_error_model(kwargs, match):
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    with pytest.raises(ValueError, match=match):
        lcbinint.obs.LightCurveData(
            np.array([0.0]),
            np.array([10.0]),
            np.array([1.0]),
            **kwargs,
        )


def test_model_residuals_use_effective_errors():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    times = np.array([0.0, 1.0, 2.0])
    flux = np.array([5.0, 8.0, 11.0])
    flux_err = np.array([1.0, 1.0, 1.0])
    data = lcbinint.obs.LightCurveData(times, flux, flux_err, k=2.0, emin=0.0)
    light_curve = lcbinint.lc.LightCurve()

    model = lcbinint.bayes.Model(light_curve=light_curve, data=data)
    model.param("t0", lcbinint.bayes.Uniform(-1.0, 1.0))
    model.param("tE", lcbinint.bayes.Uniform(1.0, 10.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.likelihood()

    theta = [0.0, 5.0, 0.2]
    residuals_with_k = np.asarray(model.residuals(theta))

    data_no_k = lcbinint.obs.LightCurveData(times, flux, flux_err)
    model_no_k = lcbinint.bayes.Model(light_curve=light_curve, data=data_no_k)
    model_no_k.param("t0", lcbinint.bayes.Uniform(-1.0, 1.0))
    model_no_k.param("tE", lcbinint.bayes.Uniform(1.0, 10.0))
    model_no_k.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model_no_k.likelihood()
    residuals_no_k = np.asarray(model_no_k.residuals(theta))

    assert residuals_with_k.tolist() == pytest.approx((0.5 * residuals_no_k).tolist())
