import math

import pytest


def _make_sampled_flux_model(lcbinint, np, likelihood="gaussian", **kwargs):
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
    data = lcbinint.obs.LightCurveData(
        times,
        1234.0 * magnification - 20.0,
        np.full(len(times), 25.0),
        name="tiny",
    )

    model = lcbinint.bayes.Model(light_curve=light_curve, data=data)
    model.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    model.param("tE", lcbinint.bayes.LogUniform(1.0, 100.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.param("s", lcbinint.bayes.Uniform(0.5, 2.0))
    model.param("q", lcbinint.bayes.LogUniform(1.0e-3, 0.5))
    model.param("alpha", lcbinint.bayes.Uniform(0.0, math.pi))
    model.param("Fs_tiny", lcbinint.bayes.Uniform(0.0, 5000.0))
    model.param("Fb_tiny", lcbinint.bayes.Uniform(-1000.0, 1000.0))
    model.likelihood(likelihood, flux="sample", **kwargs)

    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
        1234.0,
        -20.0,
    ]
    return model, data, theta, magnification


def test_gaussian_likelihood_can_sample_flux_parameters():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, data, theta, magnification = _make_sampled_flux_model(lcbinint, np)
    flux = model.fluxes(theta)

    assert flux["tiny"]["Fs"] == pytest.approx(1234.0)
    assert flux["tiny"]["Fb"] == pytest.approx(-20.0)

    residual = data.flux - 1234.0 * magnification + 20.0
    expected_chi2 = np.sum((residual / data.effective_flux_err) ** 2)
    assert model.chi2(theta) == pytest.approx(expected_chi2)
    assert model.log_likelihood(theta) == pytest.approx(-0.5 * expected_chi2)


def test_sampled_flux_works_with_unified_isochrone_theta_star_api():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")
    model, _, theta, _ = _make_sampled_flux_model(lcbinint, np)
    isochrone = object()
    seen = {}

    class Provider:
        names = ("tE",)
        integration_samples = 8

        @staticmethod
        def log_density(theta, context=None, magnitudes=None):
            return 0.0

        @staticmethod
        def _isochrone_conditional_terms(theta, *, magnitudes, context=None):
            seen.update(magnitudes)
            values = np.arange(8, dtype=float) + 1.0
            return {
                "log_terms": np.zeros(8),
                "physical": {
                    "ML": values,
                    "DL": values,
                    "DS": values + 1.0,
                    "mu_N": values,
                    "mu_E": values,
                    "thetaS": np.full(8, 0.005),
                },
                }

    Provider.isochrone = isochrone

    @model.theta_star(isochrone=isochrone)
    def _(fluxes):
        return {"Imag": fluxes["tiny"]["Fs"] / 1000.0}

    model.galactic_prior(Provider())

    assert math.isfinite(model.log_prob(theta))
    assert seen == {"Imag": pytest.approx(1.234)}


def test_sampler_records_sampled_flux_parameters():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, _, _ = _make_sampled_flux_model(lcbinint, np)
    chain = lcbinint.run_sampler(
        model,
        nsteps=2,
        burnin=0,
        options=lcbinint.SamplerOptions(
            nwalkers=10,
            seed=12,
            log_path="",
            auto_stop=False,
        ),
    )

    names = chain.param_names
    fs_idx = names.index("Fs_tiny")
    fb_idx = names.index("Fb_tiny")
    samples = chain.get_samples()

    assert chain.fluxes["tiny"]["Fs"].tolist() == pytest.approx(
        samples[:, fs_idx].tolist()
    )
    assert chain.fluxes["tiny"]["Fb"].tolist() == pytest.approx(
        samples[:, fb_idx].tolist()
    )


def test_sampled_flux_requires_flux_likelihood_option():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, theta, _ = _make_sampled_flux_model(lcbinint, np)
    model.likelihood("gaussian", flux="fit")

    with pytest.raises(ValueError, match="flux='sample'"):
        model.log_likelihood(theta)


def test_student_t_likelihood_can_sample_flux_parameters():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, data, theta, magnification = _make_sampled_flux_model(
        lcbinint, np, "student_t", nu=3.5
    )
    residuals = (data.flux - 1234.0 * magnification + 20.0) / data.effective_flux_err
    nu = 3.5
    norm = (
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * math.log(nu * math.pi)
    )
    expected = np.sum(
        norm
        - np.log(data.effective_flux_err)
        - 0.5 * (nu + 1.0) * np.log1p((residuals**2) / nu)
    )

    assert model.log_likelihood(theta) == pytest.approx(expected)


def test_flux_all_registers_unbounded_flux_parameters_for_all_datasets():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

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

    event = lcbinint.obs.Event()
    event.add(
        lcbinint.obs.LightCurveData(
            times,
            1000.0 * magnification + 10.0,
            np.full(len(times), 20.0),
            name="A",
        )
    )
    event.add(
        lcbinint.obs.LightCurveData(
            times,
            500.0 * magnification - 5.0,
            np.full(len(times), 30.0),
            name="B",
        )
    )

    model = lcbinint.bayes.Model(light_curve=light_curve, event=event)
    model.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    model.param("tE", lcbinint.bayes.LogUniform(1.0, 100.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.param("s", lcbinint.bayes.Uniform(0.5, 2.0))
    model.param("q", lcbinint.bayes.LogUniform(1.0e-3, 0.5))
    model.param("alpha", lcbinint.bayes.Uniform(0.0, math.pi))
    with pytest.raises(ValueError, match="explicit prior"):
        model.param("flux_all")

    model.param("flux_all", lcbinint.bayes.Uniform(-1.0e6, 1.0e6))
    model.likelihood("gaussian", flux="sample")

    assert model.dataset_names == ["A", "B"]
    assert model.param_names[-4:] == ["Fs_A", "Fb_A", "Fs_B", "Fb_B"]

    theta = [
        true["t0"],
        math.log(true["tE"]),
        true["u0"],
        true["s"],
        math.log(true["q"]),
        true["alpha"],
        1000.0,
        10.0,
        500.0,
        -5.0,
    ]
    fluxes = model.fluxes(theta)

    assert math.isfinite(model.log_prior(theta))
    assert fluxes["A"]["Fs"] == pytest.approx(1000.0)
    assert fluxes["A"]["Fb"] == pytest.approx(10.0)
    assert fluxes["B"]["Fs"] == pytest.approx(500.0)
    assert fluxes["B"]["Fb"] == pytest.approx(-5.0)
    assert model.log_likelihood(theta) == pytest.approx(0.0)


def test_every_sampled_parameter_requires_an_explicit_prior():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, _, _ = _make_sampled_flux_model(lcbinint, np)

    with pytest.raises(ValueError, match="derived exclusively"):
        model.param("thetaS")

    with pytest.raises(ValueError, match="derived exclusively"):
        model.param("thetaS", lcbinint.bayes.LogUniform(0.01, 1.0))
