import math
from types import SimpleNamespace

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


def test_flux_aware_galactic_context_rejects_marginalized_flux():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, theta, _ = _make_model(lcbinint, np)
    galaxy = SimpleNamespace(
        names=("tE",),
        log_density=lambda values, context=None: 0.0,
    )
    model.galactic_prior(
        galaxy,
        context=lambda params, likelihood: {
            "Fs": likelihood.fluxes["tiny"]["Fs"]
        },
    )

    with pytest.raises(RuntimeError, match="requires marginalized thetaS"):
        model.log_prob(theta)


def test_flux_aware_galactic_magnitudes_reject_marginalized_flux():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, theta, _ = _make_model(lcbinint, np)
    galaxy = SimpleNamespace(
        names=("tE",),
        log_density=lambda values, **kwargs: 0.0,
        log_joint_density=lambda values, **kwargs: 0.0,
    )
    model.galactic_prior(
        galaxy,
        magnitudes=lambda params, likelihood: {
            "Imag": likelihood.fluxes["tiny"]["Fs"]
        },
    )

    with pytest.raises(RuntimeError, match="requires marginalized thetaS"):
        model.log_prob(theta)


def test_theta_star_marginalizes_conditional_student_t_flux():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from scipy.special import stdtrit
    from scipy.stats import qmc

    model, _, theta, _ = _make_model(lcbinint, np)
    samples = 32
    seed = 17

    @model.theta_star(samples=samples, seed=seed)
    def _(fluxes):
        return math.log(abs(fluxes["tiny"]["Fs"]) / 1000.0), 0.0

    @model.prior
    def _(thetaS, **_):
        return -0.5 * ((thetaS - 0.9) / 0.1) ** 2

    base_prob, _, conditionals = model._log_prob_and_fluxes(theta)
    conditional = conditionals["tiny"]
    uniforms = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(5)[:, 0]
    fs_draws = (
        conditional["mean"]
        + conditional["scale"] * stdtrit(conditional["df"], uniforms)
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


def test_flux_theta_star_and_gapmoe_distance_marginalize_together():
    lcbinint = pytest.importorskip("lcbinint")
    gapmoe = pytest.importorskip("gapmoe")
    np = pytest.importorskip("numpy")
    from gapmoe.priors.high_level import IsochroneModel
    from gapmoe.source_selection import CmdCoordinates, CmdPriorTable

    model, _, theta, _ = _make_model(lcbinint, np)
    model.param("rho", lcbinint.bayes.LogUniform(1.0e-4, 1.0e-1))
    model.param("piEN", lcbinint.bayes.Uniform(-1.0, 1.0))
    model.param("piEE", lcbinint.bayes.Uniform(-1.0, 1.0))
    theta = [*theta, math.log(0.005), 0.1, 0.05]

    @model.theta_star(samples=4, seed=19)
    def _(fluxes):
        return math.log(0.005 * abs(fluxes["tiny"]["Fs"]) / 900.0), 0.1

    @model.prior
    def ordinary_lc_prior(u0, **_):
        return -0.5 * (u0 / 0.5) ** 2

    reference_edges = np.linspace(-8.0, 20.0, 57)
    color_edges = np.linspace(-2.0, 8.0, 41)
    density = np.full((11, 56, 40), 1.0 / 280.0)
    isochrone = IsochroneModel(
        reference_band="Imag",
        color_bands=("Vmag", "Imag"),
        table=CmdPriorTable(
            coordinates=CmdCoordinates("Imag", "Vmag", "Imag"),
            reference_edges=reference_edges,
            color_edges=color_edges,
            density_by_component=density,
        ),
    )
    galaxy = (
        gapmoe.Model()
        .set(
            l=0.25,
            b=-3.75,
            extinction={"Imag": 1.2, "Vmag": 2.0},
        )
        .set_flow(release="rate-included-v1")
        .galactic_model(isochrone)
    )
    prior = galaxy.parameterize(
        gapmoe.ParamType(parallax=True, distance="marginalize")
    )

    model.galactic_prior(
        prior,
        context={"vEarth": (0.0, 0.0)},
        magnitudes=lambda params, likelihood: {
            "Imag": 18.0
            - 2.5 * math.log10(abs(likelihood.fluxes["tiny"]["Fs"]) / 900.0),
            "Vmag": 20.0
            - 2.5 * math.log10(abs(likelihood.fluxes["tiny"]["Fs"]) / 900.0),
        },
    )

    assert math.isfinite(model.log_prob(theta))


def test_sampler_runs_full_flux_theta_star_galactic_marginalization(tmp_path):
    lcbinint = pytest.importorskip("lcbinint")
    pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    jnp = pytest.importorskip("jax.numpy")

    model, _, _, _ = _make_model(lcbinint, np)

    @model.theta_star(samples=4, seed=23)
    def _(fluxes):
        return math.log(abs(fluxes["tiny"]["Fs"]) / 1000.0), 0.1

    class Galaxy:
        names = ("tE", "thetaS")

        @staticmethod
        def log_density(theta, context=None, magnitudes=None):
            return -0.5 * jnp.sum(jnp.asarray(theta) ** 2)

        @staticmethod
        def log_joint_density(theta, context=None, magnitudes=None):
            return (
                -0.5 * (theta[0] / 100.0) ** 2
                -0.5 * (theta[1] / 10.0) ** 2
                -0.5 * (magnitudes["Imag"] / 100.0) ** 2
            )

    model.galactic_prior(
        Galaxy(),
        magnitudes=lambda params, likelihood: {
            "Imag": abs(likelihood.fluxes["tiny"]["Fs"]) / 1000.0,
        },
    )
    path = tmp_path / "full-marginal.h5"
    chain = lcbinint.run_sampler(
        model,
        nsteps=2,
        burnin=0,
        options=lcbinint.SamplerOptions(
            nwalkers=12,
            seed=29,
            h5_path=str(path),
            log_path="",
            log_every=1,
            auto_stop=False,
        ),
    )

    assert chain.get_samples().shape == (24, 6)
    assert np.isfinite(chain.get_log_prob()).all()
    assert chain._flux_conditional_scales.shape == (2, 12, 1)
    loaded = lcbinint.load_chain(str(path))
    assert loaded._flux_conditional_scales.shape == (2, 12, 1)
    assert loaded.get_fluxes()["tiny"]["Fs"].shape == (24,)


def test_kepler_lom_runs_with_flux_theta_star_marginalization():
    lcbinint = pytest.importorskip("lcbinint")
    gapmoe = pytest.importorskip("gapmoe")
    np = pytest.importorskip("numpy")
    from gapmoe.priors.high_level import IsochroneModel
    from gapmoe.source_selection import CmdCoordinates, CmdPriorTable

    true = {
        "t0": 8000.0,
        "tE": 50.0,
        "u0": 0.1,
        "rho": 0.005,
        "q": 0.1,
        "s": 1.0,
        "alpha": 0.5,
        "piEN": 0.1,
        "piEE": 0.05,
        "g1": -0.0001,
        "g2": -0.0001,
        "g3": -0.01,
        "lom_szs": 0.1,
        "lom_ar": 1.1,
    }
    times = np.linspace(7998.0, 8002.0, 9)
    light_curve = lcbinint.lc.LightCurve(
        lens="binary",
        orbital_motion="kepler",
        t_ref=8000.0,
    )
    flux = 900.0 * light_curve(times, true) + 25.0
    flux[4] += 5.0
    data = lcbinint.obs.LightCurveData(
        times, flux, np.full(len(times), 20.0), name="tiny"
    )
    model = lcbinint.bayes.Model(light_curve=light_curve, data=data)
    priors = {
        "t0": lcbinint.bayes.Uniform(7990.0, 8010.0),
        "tE": lcbinint.bayes.LogUniform(1.0, 100.0),
        "u0": lcbinint.bayes.Uniform(0.0, 1.0),
        "rho": lcbinint.bayes.LogUniform(1.0e-4, 0.1),
        "q": lcbinint.bayes.LogUniform(1.0e-3, 1.0),
        "s": lcbinint.bayes.Uniform(0.5, 2.0),
        "alpha": lcbinint.bayes.Uniform(0.0, math.pi),
        "piEN": lcbinint.bayes.Uniform(-1.0, 1.0),
        "piEE": lcbinint.bayes.Uniform(-1.0, 1.0),
        "g1": lcbinint.bayes.Uniform(-0.1, 0.1),
        "g2": lcbinint.bayes.Uniform(-0.1, 0.1),
        "g3": lcbinint.bayes.Uniform(-0.1, 0.1),
        "lom_szs": lcbinint.bayes.Uniform(-2.0, 2.0),
        "lom_ar": lcbinint.bayes.Uniform(0.51, 2.0),
    }
    for name, prior in priors.items():
        model.param(name, prior)
    model.likelihood("gaussian", flux="marginalize")

    @model.theta_star(samples=4, seed=31)
    def _(fluxes):
        return math.log(0.005 * abs(fluxes["tiny"]["Fs"]) / 900.0), 0.05

    reference_edges = np.linspace(-8.0, 20.0, 57)
    color_edges = np.linspace(-2.0, 8.0, 41)
    isochrone = IsochroneModel(
        reference_band="Imag",
        color_bands=("Vmag", "Imag"),
        table=CmdPriorTable(
            coordinates=CmdCoordinates("Imag", "Vmag", "Imag"),
            reference_edges=reference_edges,
            color_edges=color_edges,
            density_by_component=np.full((11, 56, 40), 1.0 / 280.0),
        ),
    )
    galaxy = (
        gapmoe.Model()
        .set(
            l=0.25,
            b=-3.75,
            extinction={"Imag": 1.2, "Vmag": 2.0},
        )
        .set_flow(release="rate-included-v1")
        .galactic_model(isochrone)
    )
    prior = galaxy.parameterize(
        gapmoe.ParamType(parallax=True, orbital_motion="kepler")
    )
    model.galactic_prior(
        prior,
        context={"vEarth": (0.0, 0.0)},
        magnitudes=lambda params, likelihood: {
            "Imag": 18.0
            - 2.5 * math.log10(abs(likelihood.fluxes["tiny"]["Fs"]) / 900.0),
            "Vmag": 20.0
            - 2.5 * math.log10(abs(likelihood.fluxes["tiny"]["Fs"]) / 900.0),
        },
    )
    theta = [
        math.log(true[name]) if name in {"tE", "rho", "q"} else true[name]
        for name in priors
    ]

    assert prior.names == tuple(priors)
    assert math.isfinite(model.log_prob(theta))
