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
