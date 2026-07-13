import math

import pytest


def _make_model(lcbinint, np, reparam=False):
    true = {
        "t0": 8000.0,
        "tE": 20.0,
        "u0": 0.15,
        "s": 1.2,
        "q": 0.08,
        "alpha": 0.4,
        "piEN": 0.04,
        "piEE": 0.03,
    }
    times = np.array([7998.0, 7999.0, 8000.0, 8001.0, 8002.0])
    light_curve = lcbinint.lc.LightCurve()
    flux = 1000.0 * light_curve(times, true) + 25.0
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
    if reparam:
        rp = model.reparam(["piEN", "piEE"])
        rp.param("piE", lcbinint.bayes.LogUniform(0.01, 1.0))
        rp.param("phi_piE", lcbinint.bayes.Uniform(0.0, 2.0 * math.pi))

        @rp.transform
        def _(piE, phi_piE):
            return {
                "piEN": piE * math.cos(phi_piE),
                "piEE": piE * math.sin(phi_piE),
            }
    else:
        model.param("piEN", lcbinint.bayes.Uniform(-1.0, 1.0))
        model.param("piEE", lcbinint.bayes.Uniform(-1.0, 1.0))
    model.likelihood("gaussian")
    return model, true


def test_custom_likelihood_can_use_flux_context():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, true = _make_model(lcbinint, np)
    seen = {}

    @model.likelihood(context=True)
    def _(context, **_):
        seen["Fs"] = context.fluxes["tiny"]["Fs"]
        seen["Fb"] = context.fluxes["tiny"]["Fb"]
        return -0.5 * ((context.fluxes["tiny"]["Fs"] - 1000.0) / 10.0) ** 2

    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
        true["piEN"],
        true["piEE"],
    ]
    flux = model.fluxes(theta)["tiny"]

    expected_extra = -0.5 * ((flux["Fs"] - 1000.0) / 10.0) ** 2
    expected = model.log_prior(theta) + model.log_likelihood(theta) + expected_extra
    assert model.log_prob(theta) == pytest.approx(expected)
    assert seen["Fs"] == pytest.approx(flux["Fs"])
    assert seen["Fb"] == pytest.approx(flux["Fb"])


def test_custom_likelihood_context_works_with_reparam_sampler():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _ = _make_model(lcbinint, np, reparam=True)

    @model.likelihood(context=True)
    def _(context, **_):
        return -0.5 * ((context.fluxes["tiny"]["Fb"] - 25.0) / 10.0) ** 2

    chain = lcbinint.run_sampler(
        model,
        nsteps=2,
        burnin=0,
        options=lcbinint.SamplerOptions(
            nwalkers=8,
            seed=31,
            log_path="",
            auto_stop=False,
        ),
    )

    assert chain.fluxes["tiny"]["Fs"].shape == (16,)


def test_theta_star_adds_log_space_gaussian_from_fitted_flux():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, true = _make_model(lcbinint, np)
    model.param("thetaS", lcbinint.bayes.LogUniform(0.01, 10.0))
    seen = {}

    @model.theta_star
    def _(fluxes):
        seen.update(fluxes)
        return math.log(0.7), 0.2

    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
        true["piEN"],
        true["piEE"],
        math.log(0.7),
    ]
    expected_relation = -math.log(0.2) - 0.5 * math.log(2.0 * math.pi)
    expected = model.log_prior(theta) + model.log_likelihood(theta) + expected_relation

    assert model.log_prob(theta) == pytest.approx(expected)
    assert seen["tiny"]["Fs"] == pytest.approx(1000.0)
    assert seen["tiny"]["Fb"] == pytest.approx(25.0)


def test_theta_star_zero_sigma_derives_value_before_physical_prior():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, true = _make_model(lcbinint, np)
    seen = {}

    @model.theta_star
    def _(fluxes):
        assert fluxes["tiny"]["Fs"] == pytest.approx(1000.0)
        return math.log(0.7), 0.0

    @model.prior
    def _(thetaS, **_):
        seen["thetaS"] = thetaS
        return 0.0

    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
        true["piEN"],
        true["piEE"],
    ]

    expected = model.log_prior(theta) + model.log_likelihood(theta)
    assert model.log_prob(theta) == pytest.approx(expected)
    assert seen["thetaS"] == pytest.approx(0.7)


def test_theta_star_sampled_value_works_with_reparam_adapter():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, true = _make_model(lcbinint, np, reparam=True)
    model.param("thetaS", lcbinint.bayes.LogUniform(0.01, 10.0))

    @model.theta_star
    def _(fluxes):
        return math.log(0.7), 0.2

    piE = math.hypot(true["piEN"], true["piEE"])
    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
        math.log(0.7),
        math.log(piE),
        math.atan2(true["piEE"], true["piEN"]),
    ]

    assert math.isfinite(model._sampling_adapter().log_prob(theta))


def test_theta_star_without_parameter_is_marginalized():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, true = _make_model(lcbinint, np)

    @model.theta_star(samples=32, seed=4)
    def _(fluxes):
        return math.log(0.7), 0.2

    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
        true["piEN"],
        true["piEE"],
    ]
    expected = model.log_prior(theta) + model.log_likelihood(theta)
    assert model.log_prob(theta) == pytest.approx(expected)


def test_theta_star_marginalization_applies_hard_prior_bounds():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from scipy.special import ndtri
    from scipy.stats import qmc

    model, true = _make_model(lcbinint, np)
    samples = 64
    seed = 9

    @model.theta_star(samples=samples, seed=seed)
    def _(fluxes):
        return math.log(0.7), 0.2

    @model.prior
    def _(thetaS, **_):
        return 0.0 if thetaS < 0.7 else float("-inf")

    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
        true["piEN"],
        true["piEE"],
    ]
    uniforms = qmc.Sobol(d=1, scramble=True, seed=seed).random_base2(6)[:, 0]
    accepted = ndtri(uniforms) < 0.0
    expected_extra = math.log(np.count_nonzero(accepted) / samples)
    expected = model.log_prior(theta) + model.log_likelihood(theta) + expected_extra

    assert model.log_prob(theta) == pytest.approx(expected)


def test_theta_star_deterministic_conflicts_with_registered_parameter():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    _, true = _make_model(lcbinint, np)
    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
        true["piEN"],
        true["piEE"],
    ]

    deterministic, _ = _make_model(lcbinint, np)
    deterministic.param("thetaS", lcbinint.bayes.LogUniform(0.01, 10.0))

    @deterministic.theta_star
    def _(fluxes):
        return math.log(0.7), 0.0

    with pytest.raises(RuntimeError, match="thetaS is deterministic"):
        deterministic.log_prob([*theta, math.log(0.7)])
