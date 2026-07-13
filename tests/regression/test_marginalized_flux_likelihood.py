import math

import pytest


def _make_model(lcbinint, np, likelihood="gaussian", **kwargs):
    true = {
        "t0": 8000.0,
        "tE": 20.0,
        "u0": 0.15,
        "s": 1.2,
        "q": 0.08,
        "alpha": 0.4,
    }
    times = np.array([7998.0, 7999.0, 8000.0, 8001.0, 8002.0])
    light_curve = lcbinint.lc.LightCurve()
    magnification = light_curve(times, true)
    flux = 900.0 * magnification + 35.0
    flux[2] += 12.0
    data = lcbinint.obs.LightCurveData(
        times,
        flux,
        np.full(len(times), 20.0),
        name="tiny",
    )

    model = lcbinint.bayes.Model(light_curve=light_curve, data=data)
    model.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    model.param("tE", lcbinint.bayes.LogUniform(1.0, 100.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.param("s", lcbinint.bayes.Uniform(0.5, 2.0))
    model.param("q", lcbinint.bayes.LogUniform(1.0e-3, 0.5))
    model.param("alpha", lcbinint.bayes.Uniform(0.0, math.pi))
    model.likelihood(likelihood, flux="marginalize", **kwargs)
    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
    ]
    return model, data, theta, magnification


def test_gaussian_likelihood_can_marginalize_flux_parameters():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, data, theta, magnification = _make_model(lcbinint, np)
    flux = model.fluxes(theta)["tiny"]
    residual = data.flux - flux["Fs"] * magnification - flux["Fb"]
    weights = 1.0 / np.asarray(data.effective_flux_err) ** 2
    chi2 = np.sum(weights * residual**2)
    s_w = np.sum(weights)
    s_w_a = np.sum(weights * magnification)
    s_w_a2 = np.sum(weights * magnification * magnification)
    det = s_w_a2 * s_w - s_w_a * s_w_a
    expected = -0.5 * (len(data.time) - 2) * math.log(chi2) - 0.5 * math.log(det)

    assert model.log_likelihood(theta) == pytest.approx(expected)

    base_prob, one_pass_fluxes, conditionals = model._log_prob_and_fluxes(theta)
    conditional = conditionals["tiny"]
    expected_scale = math.sqrt(chi2 / (len(data.time) - 2) * s_w / det)
    assert base_prob == pytest.approx(model.log_prob(theta))
    assert one_pass_fluxes["tiny"] == pytest.approx(flux)
    assert conditional["mean"] == pytest.approx(flux["Fs"])
    assert conditional["scale"] == pytest.approx(expected_scale)
    assert conditional["df"] == pytest.approx(len(data.time) - 2)


def test_theta_star_marginalizes_conditional_student_t_flux():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, theta, _ = _make_model(lcbinint, np)
    n_flux = 32
    seed = 17

    @model.theta_star(n_flux=n_flux, n_theta=1, seed=seed)
    def _(fluxes):
        return math.log(abs(fluxes["tiny"]["Fs"]) / 1000.0), 0.0

    @model.prior
    def _(thetaS, **_):
        return -0.5 * ((thetaS - 0.9) / 0.1) ** 2

    base_prob, _, conditionals = model._log_prob_and_fluxes(theta)
    conditional = conditionals["tiny"]
    rng = np.random.default_rng(seed)
    fs_draws = (
        conditional["mean"]
        + conditional["scale"] * rng.standard_t(conditional["df"], n_flux)
    )
    weights = -0.5 * ((np.abs(fs_draws) / 1000.0 - 0.9) / 0.1) ** 2
    peak = np.max(weights)
    expected_extra = peak + math.log(np.mean(np.exp(weights - peak)))

    assert model.log_prob(theta) == pytest.approx(base_prob + expected_extra)


def test_marginalized_flux_sampler_records_conditional_best_fit_fluxes():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, _, _ = _make_model(lcbinint, np)
    chain = lcbinint.run_sampler(
        model,
        nsteps=2,
        burnin=0,
        options=lcbinint.SamplerOptions(
            nwalkers=8,
            seed=21,
            log_path="",
            auto_stop=False,
        ),
    )

    assert set(chain.fluxes["tiny"]) == {"Fs", "Fb"}
    assert chain.fluxes["tiny"]["Fs"].shape == (16,)
    assert chain.get_fluxes(flat=False)["tiny"]["Fb"].shape == (2, 8)


def test_marginalized_flux_rejects_student_t_for_now():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    with pytest.raises(ValueError, match="gaussian"):
        _make_model(lcbinint, np, "student_t", nu=4.0)
