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
