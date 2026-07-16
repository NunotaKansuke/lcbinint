import math

import pytest


def _make_model(lcbinint, np, reparam=False, flux_mode=None):
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
    observed_flux = 1000.0 * light_curve(times, true) + 25.0
    data = lcbinint.obs.LightCurveData(
        times,
        observed_flux,
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
    if flux_mode == "sample":
        model.param("Fs_tiny", lcbinint.bayes.Uniform(-1.0e5, 1.0e5))
        model.param("Fb_tiny", lcbinint.bayes.Uniform(-1.0e5, 1.0e5))
        model.likelihood("gaussian", flux="sample")
    else:
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


def test_galactic_context_receives_sampled_fluxes():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, true = _make_model(lcbinint, np, flux_mode="sample")
    seen = {}

    class Galaxy:
        names = ("tE",)

        @staticmethod
        def log_density(theta, context=None):
            seen.update(context)
            return 0.0

    model.galactic_prior(
        Galaxy(),
        context=lambda params, likelihood: {
            "Fs": likelihood.fluxes["tiny"]["Fs"],
            "Fb": likelihood.fluxes["tiny"]["Fb"],
            "flux_mode": likelihood.flux_mode,
        },
    )
    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
        true["piEN"],
        true["piEE"],
        950.0,
        30.0,
    ]

    assert math.isfinite(model.log_prob(theta))
    assert seen == {"Fs": 950.0, "Fb": 30.0, "flux_mode": "sample"}


def test_galactic_joint_photometry_receives_fitted_fluxes():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, true = _make_model(lcbinint, np)
    seen = {}

    class Galaxy:
        names = ("tE",)

        @staticmethod
        def log_density(theta, context=None, magnitudes=None):
            raise AssertionError("joint source photometry must use log_joint_density")

        @staticmethod
        def log_joint_density(theta, context=None, magnitudes=None):
            seen.update(magnitudes)
            return -0.5 * magnitudes["Imag"] ** 2

    model.galactic_prior(
        Galaxy(),
        magnitudes=lambda params, likelihood: {
            "Imag": likelihood.fluxes["tiny"]["Fs"] / 1000.0,
        },
    )
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
    fitted = model.fluxes(theta)["tiny"]["Fs"] / 1000.0

    expected = model.log_prior(theta) + model.log_likelihood(theta) - 0.5 * fitted**2
    assert model.log_prob(theta) == pytest.approx(expected)
    assert seen == {"Imag": pytest.approx(fitted)}


def test_theta_star_cannot_be_registered_as_a_sampled_parameter():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _ = _make_model(lcbinint, np)
    with pytest.raises(ValueError, match="derived exclusively"):
        model.param("thetaS", lcbinint.bayes.LogUniform(0.01, 10.0))
    with pytest.raises(ValueError, match="cannot be a reparameterization target"):
        model.reparam(["thetaS"])


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


def test_latent_theta_star_works_with_reparam_adapter():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, true = _make_model(lcbinint, np, reparam=True)
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


def test_theta_star_isochrone_requires_matching_galactic_prior():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, true = _make_model(lcbinint, np)
    isochrone = object()

    @model.theta_star(isochrone=isochrone)
    def _(_fluxes):
        return {"Imag": 18.0, "Vmag": 20.0}

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

    with pytest.raises(RuntimeError, match="exactly one Galactic prior"):
        model.log_prob(theta)


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


def test_theta_star_fixed_value_is_defined_by_zero_log_sigma():
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
    @deterministic.theta_star
    def _(fluxes):
        return math.log(0.7), 0.0

    assert math.isfinite(deterministic.log_prob(theta))
