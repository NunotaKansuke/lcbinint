import math

import pytest


def _make_model(lcbinint, np, likelihood="gaussian", **likelihood_kwargs):
    times = np.array([7998.0, 7999.0, 8000.0, 8001.0, 8002.0])
    true = {
        "t0": 8000.0,
        "tE": 20.0,
        "u0": 0.15,
        "s": 1.2,
        "q": 0.08,
        "alpha": 0.4,
    }
    light_curve = lcbinint.lc.LightCurve()
    flux = 1000.0 * light_curve(times, true) + 25.0
    flux[2] += 40.0
    flux_err = np.full(len(times), 20.0)
    data = lcbinint.obs.LightCurveData(times, flux, flux_err, k=1.1, emin=3.0)

    model = lcbinint.bayes.Model(light_curve=light_curve, data=data)
    model.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    model.param("tE", lcbinint.bayes.LogUniform(1.0, 100.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.param("s", lcbinint.bayes.Uniform(0.5, 2.0))
    model.param("q", lcbinint.bayes.LogUniform(1.0e-3, 0.5))
    model.param("alpha", lcbinint.bayes.Uniform(0.0, math.pi))
    model.likelihood(likelihood, **likelihood_kwargs)
    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
    ]
    return model, data, theta


def test_student_t_likelihood_matches_wls_residual_approximation():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    nu = 4.5
    model, data, theta = _make_model(lcbinint, np, "student_t", nu=nu)
    residuals = np.asarray(model.residuals(theta))
    sigma = np.asarray(data.effective_flux_err)
    norm = (
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * math.log(nu * math.pi)
    )
    expected = np.sum(
        norm - np.log(sigma) - 0.5 * (nu + 1.0) * np.log1p(residuals**2 / nu)
    )

    assert model.log_likelihood(theta) == pytest.approx(expected)


def test_student_t_log_prob_and_fluxes_fast_path_matches_log_likelihood():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, theta = _make_model(lcbinint, np, "student_t", nu=3.0)

    assert model.log_prob(theta) == pytest.approx(
        model.log_prior(theta) + model.log_likelihood(theta)
    )

    chain = lcbinint.run_sampler(
        model,
        nsteps=2,
        burnin=0,
        options=lcbinint.SamplerOptions(
            nwalkers=8,
            seed=11,
            log_path="",
            auto_stop=False,
        ),
    )
    assert chain.flat_samples.shape == (16, 6)


def test_student_t_likelihood_rejects_invalid_nu():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    with pytest.raises(ValueError, match="nu > 0"):
        _make_model(lcbinint, np, "student_t", nu=0.0)


def test_likelihood_rejects_unknown_keyword_options():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    with pytest.raises(TypeError, match="unknown likelihood option"):
        _make_model(lcbinint, np, "gaussian", scale=2.0)
