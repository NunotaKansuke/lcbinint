import math
from types import SimpleNamespace

import pytest


class FakeGalacticModel:
    def __init__(self, names):
        self.param_type = SimpleNamespace(names=tuple(names))
        self.calls = []

    def log_prob(self, theta, context=None):
        self.calls.append((tuple(theta), context))
        scale = 1.0 if context is None else context.get("scale", 1.0)
        return -0.5 * scale * sum(float(x) ** 2 for x in theta)


class FakePhysicalGalacticModel(FakeGalacticModel):
    def __init__(self, names):
        super().__init__(names)
        self.param_type.derived_names = ("e", "cos_i")

    def to_physical(self, theta, context=None):
        return (1.5, 4.0, 8.0, 3.0, -2.0, 0.3, 0.25)


class FakeDeterministicGalacticModel(FakeGalacticModel):
    def to_deterministic_physical(self, theta, context=None):
        return {"ML": 1.5, "thetaE": 0.75}

    def to_physical(self, theta, context=None):
        raise AssertionError("to_physical should not be used when deterministic values exist")


class FakeSamplingGalacticModel:
    def __init__(self, names):
        self.param_type = SimpleNamespace(names=tuple(names))
        self.calls = []

    def sample_physical(self, theta, context=None, rng=None):
        self.calls.append((tuple(theta), context))
        offset = 0.0 if context is None else context.get("offset", 0.0)
        return {
            "ML": sum(float(value) for value in theta) + offset,
            "DS": 8.0 + offset,
        }


class FakeJaxGalacticModel:
    def __init__(self, names):
        self.param_type = SimpleNamespace(names=tuple(names))

    def log_prob(self, theta, context=None):
        import jax.numpy as jnp

        theta_s = jnp.asarray(context["thS"])
        return -0.5 * (theta[0] * theta_s) ** 2


def _make_model(lcbinint, np):
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
    model.param("piEN", lcbinint.bayes.Uniform(-1.0, 1.0))
    model.param("piEE", lcbinint.bayes.Uniform(-1.0, 1.0))
    model.likelihood("gaussian")
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
    return model, theta, true


def test_galactic_prior_uses_param_type_names_and_context_callable():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, theta, true = _make_model(lcbinint, np)
    galaxy = FakeGalacticModel(("tE", "piEN", "piEE"))

    model.galactic_prior(
        galaxy,
        context=lambda params: {"scale": params["u0"] + 2.0},
    )

    expected_extra = -0.5 * (true["u0"] + 2.0) * (
        true["tE"] ** 2 + true["piEN"] ** 2 + true["piEE"] ** 2
    )
    expected = model.log_prior(theta) + model.log_likelihood(theta) + expected_extra

    assert model.log_prob(theta) == pytest.approx(expected)
    theta_seen, context_seen = galaxy.calls[-1]
    assert theta_seen == pytest.approx((true["tE"], true["piEN"], true["piEE"]))
    assert context_seen == {"scale": true["u0"] + 2.0}


def test_latent_theta_star_vmaps_jax_galactic_prior():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")
    pytest.importorskip("jax")

    model, theta, true = _make_model(lcbinint, np)
    n_theta = 32
    seed = 13

    @model.theta_star(n_theta=n_theta, seed=seed)
    def _(fluxes):
        return math.log(0.7), 0.2

    term = model.galactic_prior(
        FakeJaxGalacticModel(("tE",)),
        context=lambda params: {"thS": params["thetaS"]},
    )

    theta_stars = np.exp(
        math.log(0.7)
        + 0.2 * np.random.default_rng(seed).standard_normal(n_theta)
    )
    weights = -0.5 * (true["tE"] * theta_stars) ** 2
    peak = np.max(weights)
    expected_extra = peak + math.log(np.mean(np.exp(weights - peak)))
    expected = model.log_prior(theta) + model.log_likelihood(theta) + expected_extra

    assert model.log_prob(theta) == pytest.approx(expected, rel=1.0e-6)
    assert term._batch_fn is not None
    assert not term._batch_disabled


def test_galactic_prior_can_use_distance_marginalized_no_parallax_names():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, theta, true = _make_model(lcbinint, np)
    model.param("rho", lcbinint.bayes.LogUniform(1.0e-4, 1.0e-1))
    theta = [*theta, math.log(0.005)]
    galaxy = FakeGalacticModel(("t0", "tE", "u0", "rho"))

    model.galactic_prior(galaxy, context={"thS": 0.5})
    assert math.isfinite(model.log_prob(theta))
    theta_seen, context_seen = galaxy.calls[-1]
    assert theta_seen == pytest.approx((true["t0"], true["tE"], true["u0"], 0.005))
    assert context_seen == {"thS": 0.5}


def test_galactic_prior_accepts_explicit_names_without_param_type():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, theta, true = _make_model(lcbinint, np)
    galaxy = SimpleNamespace(
        log_prob=lambda values, context=None: -sum(values) + context["offset"]
    )

    model.galactic_prior(
        galaxy,
        names=("u0", "q"),
        context={"offset": 3.0},
    )

    expected_extra = -true["u0"] - true["q"] + 3.0
    expected = model.log_prior(theta) + model.log_likelihood(theta) + expected_extra
    assert model.log_prob(theta) == pytest.approx(expected)


def test_custom_prior_can_use_galactic_physical_values():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, theta, _ = _make_model(lcbinint, np)
    galaxy = FakePhysicalGalacticModel(("tE", "piEN", "piEE"))
    seen = {}

    model.galactic_prior(galaxy)

    @model.prior
    def _(ML, DL, DS, mu_N, mu_E, e, cos_i, **_):
        seen.update(
            ML=ML,
            DL=DL,
            DS=DS,
            mu_N=mu_N,
            mu_E=mu_E,
            e=e,
            cos_i=cos_i,
        )
        return -0.5 * ML

    expected = model.log_prior(theta) + model.log_likelihood(theta)
    expected += galaxy.log_prob((20.0, 0.04, 0.03))
    expected += -0.5 * 1.5

    assert model.log_prob(theta) == pytest.approx(expected)
    assert seen == {
        "ML": 1.5,
        "DL": 4.0,
        "DS": 8.0,
        "mu_N": 3.0,
        "mu_E": -2.0,
        "e": 0.3,
        "cos_i": 0.25,
    }


def test_custom_prior_uses_only_galactic_deterministic_values_when_available():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, theta, _ = _make_model(lcbinint, np)
    galaxy = FakeDeterministicGalacticModel(("tE", "piEN", "piEE"))
    seen = {}

    model.galactic_prior(galaxy)

    @model.prior
    def _(ML, thetaE, **params):
        seen.update(ML=ML, thetaE=thetaE, params=params)
        return 0.0

    assert math.isfinite(model.log_prob(theta))
    assert seen["ML"] == 1.5
    assert seen["thetaE"] == 0.75
    assert "DL" not in seen["params"]
    assert "DS" not in seen["params"]


def test_custom_prior_missing_marginalized_value_explains_post_sampling_path():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, theta, _ = _make_model(lcbinint, np)
    galaxy = FakeDeterministicGalacticModel(("tE", "piEN", "piEE"))
    model.galactic_prior(galaxy)

    @model.prior
    def _(DS):
        return 0.0

    with pytest.raises(RuntimeError, match="not deterministic|get_galactic_physical|DS"):
        model.log_prob(theta)


def test_galactic_prior_missing_names_fail_clearly():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, theta, _ = _make_model(lcbinint, np)
    model.galactic_prior(FakeGalacticModel(("missing",)))

    with pytest.raises(RuntimeError, match="missing model parameter"):
        model.log_prob(theta)


def test_galactic_prior_works_after_reparam_transform():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, true = _make_model(lcbinint, np)
    model = lcbinint.bayes.Model(light_curve=model._lc, data=model._event_or_data)
    model.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    model.param("tE", lcbinint.bayes.LogUniform(1.0, 100.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.param("s", lcbinint.bayes.Uniform(0.5, 2.0))
    model.param("q", lcbinint.bayes.LogUniform(1.0e-3, 0.5))
    model.param("alpha", lcbinint.bayes.Uniform(0.0, math.pi))
    model.likelihood("gaussian")

    rp = model.reparam(["piEN", "piEE"])
    rp.param("piE", lcbinint.bayes.LogUniform(0.01, 1.0))
    rp.param("phi_piE", lcbinint.bayes.Uniform(0.0, 2.0 * math.pi))

    @rp.transform
    def _(piE, phi_piE):
        return {
            "piEN": piE * math.cos(phi_piE),
            "piEE": piE * math.sin(phi_piE),
        }

    galaxy = FakeGalacticModel(("piEN", "piEE"))
    model.galactic_prior(galaxy)

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
    adapter = model._sampling_adapter()

    assert math.isfinite(adapter.log_prob(theta))
    assert galaxy.calls[-1][0] == pytest.approx((true["piEN"], true["piEE"]))


def test_reparam_extra_prior_rejects_before_likelihood():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, true = _make_model(lcbinint, np)
    model = lcbinint.bayes.Model(light_curve=model._lc, data=model._event_or_data)
    model.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    model.param("tE", lcbinint.bayes.LogUniform(1.0, 100.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.param("s", lcbinint.bayes.Uniform(0.5, 2.0))
    model.param("q", lcbinint.bayes.LogUniform(1.0e-3, 0.5))
    model.param("alpha", lcbinint.bayes.Uniform(0.0, math.pi))
    model.likelihood("gaussian")

    rp = model.reparam(["piEN", "piEE"])
    rp.param("piE", lcbinint.bayes.LogUniform(0.01, 1.0))
    rp.param("phi_piE", lcbinint.bayes.Uniform(0.0, 2.0 * math.pi))

    @rp.transform
    def _(piE, phi_piE):
        return {
            "piEN": piE * math.cos(phi_piE),
            "piEE": piE * math.sin(phi_piE),
        }

    called = {"likelihood": False}

    @model.prior
    def _(**_):
        return float("-inf")

    @model.likelihood
    def _(**_):
        called["likelihood"] = True
        raise AssertionError("custom likelihood should not run after -inf prior")

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

    assert model._sampling_adapter().log_prob(theta) == float("-inf")
    assert not called["likelihood"]


def test_reparam_custom_prior_can_use_galactic_physical_values():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, true = _make_model(lcbinint, np)
    model = lcbinint.bayes.Model(light_curve=model._lc, data=model._event_or_data)
    model.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    model.param("tE", lcbinint.bayes.LogUniform(1.0, 100.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.param("s", lcbinint.bayes.Uniform(0.5, 2.0))
    model.param("q", lcbinint.bayes.LogUniform(1.0e-3, 0.5))
    model.param("alpha", lcbinint.bayes.Uniform(0.0, math.pi))
    model.likelihood("gaussian")

    rp = model.reparam(["piEN", "piEE"])
    rp.param("piE", lcbinint.bayes.LogUniform(0.01, 1.0))
    rp.param("phi_piE", lcbinint.bayes.Uniform(0.0, 2.0 * math.pi))

    @rp.transform
    def _(piE, phi_piE):
        return {
            "piEN": piE * math.cos(phi_piE),
            "piEE": piE * math.sin(phi_piE),
        }

    galaxy = FakePhysicalGalacticModel(("piEN", "piEE"))
    seen = {}
    model.galactic_prior(galaxy)

    @model.prior
    def _(ML, e, cos_i, **_):
        seen.update(ML=ML, e=e, cos_i=cos_i)
        return 0.0

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
    assert seen == {"ML": 1.5, "e": 0.3, "cos_i": 0.25}


def test_get_galactic_physical_samples_chain_values():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, _ = _make_model(lcbinint, np)
    chain = lcbinint.run_sampler(
        model,
        nsteps=2,
        burnin=0,
        options=lcbinint.SamplerOptions(
            nwalkers=8,
            seed=41,
            log_path="",
            auto_stop=False,
        ),
    )
    galaxy = FakeSamplingGalacticModel(("tE", "piEN", "piEE"))

    phys = model.get_galactic_physical(
        chain,
        galaxy,
        context=lambda params: {"offset": params["u0"]},
        flat=False,
    )

    assert set(phys) == {"ML", "DS"}
    assert phys["ML"].shape == (2, 8)
    assert phys["DS"].shape == (2, 8)
    assert len(galaxy.calls) == 16
    assert "offset" in galaxy.calls[0][1]


def test_get_galactic_physical_uses_reparam_transform():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, _ = _make_model(lcbinint, np)
    model = lcbinint.bayes.Model(light_curve=model._lc, data=model._event_or_data)
    model.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    model.param("tE", lcbinint.bayes.LogUniform(1.0, 100.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.param("s", lcbinint.bayes.Uniform(0.5, 2.0))
    model.param("q", lcbinint.bayes.LogUniform(1.0e-3, 0.5))
    model.param("alpha", lcbinint.bayes.Uniform(0.0, math.pi))
    model.likelihood("gaussian")

    rp = model.reparam(["piEN", "piEE"])
    rp.param("piE", lcbinint.bayes.LogUniform(0.01, 1.0))
    rp.param("phi_piE", lcbinint.bayes.Uniform(0.0, 2.0 * math.pi))

    @rp.transform
    def _(piE, phi_piE):
        return {
            "piEN": piE * math.cos(phi_piE),
            "piEE": piE * math.sin(phi_piE),
        }

    chain = lcbinint.run_sampler(
        model,
        nsteps=2,
        burnin=0,
        options=lcbinint.SamplerOptions(
            nwalkers=8,
            seed=43,
            log_path="",
            auto_stop=False,
        ),
    )
    galaxy = FakeSamplingGalacticModel(("piEN", "piEE"))

    phys = model.get_galactic_physical(chain, galaxy)

    assert phys["ML"].shape == (16,)
    assert len(galaxy.calls) == 16
    assert all(len(theta) == 2 for theta, _ in galaxy.calls)


def test_galactic_prior_requires_names_when_prior_has_no_param_type():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, _ = _make_model(lcbinint, np)
    galaxy = SimpleNamespace(log_prob=lambda values, context=None: 0.0)

    with pytest.raises(ValueError, match="requires names"):
        model.galactic_prior(galaxy)


def test_extra_prior_rejects_before_python_likelihood():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, theta, _ = _make_model(lcbinint, np)
    called = {"likelihood": False}

    @model.prior
    def _(**_):
        return float("-inf")

    @model.likelihood
    def _(**_):
        called["likelihood"] = True
        raise AssertionError("custom likelihood should not run after -inf prior")

    assert model.log_prob(theta) == float("-inf")
    assert not called["likelihood"]


def test_guard_rejects_before_galactic_prior_and_likelihood():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, theta, _ = _make_model(lcbinint, np)
    galaxy = FakeGalacticModel(("tE",))
    model.galactic_prior(galaxy)
    called = {"likelihood": False}

    @model.guard
    def _(u0, **_):
        return u0 < 0.1

    @model.likelihood
    def _(**_):
        called["likelihood"] = True
        raise AssertionError("likelihood should not run after a failed guard")

    assert model.has_py_extras()
    assert model.log_prob(theta) == float("-inf")
    assert galaxy.calls == []
    assert not called["likelihood"]


def test_guard_receives_reparameterized_physical_values():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _, true = _make_model(lcbinint, np)
    model = lcbinint.bayes.Model(light_curve=model._lc, data=model._event_or_data)
    model.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    model.param("tE", lcbinint.bayes.LogUniform(1.0, 100.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.param("s", lcbinint.bayes.Uniform(0.5, 2.0))
    model.param("q", lcbinint.bayes.LogUniform(1.0e-3, 0.5))
    model.param("alpha", lcbinint.bayes.Uniform(0.0, math.pi))
    model.likelihood("gaussian")

    rp = model.reparam(["piEN", "piEE"])
    rp.param("piE", lcbinint.bayes.LogUniform(0.01, 1.0))
    rp.param("phi_piE", lcbinint.bayes.Uniform(0.0, 2.0 * math.pi))

    @rp.transform
    def _(piE, phi_piE):
        return {
            "piEN": piE * math.cos(phi_piE),
            "piEE": piE * math.sin(phi_piE),
        }

    seen = []

    @model.guard
    def _(piEN, piEE, **_):
        seen.append((piEN, piEE))
        return False

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

    assert model._sampling_adapter().log_prob(theta) == float("-inf")
    assert seen == [pytest.approx((true["piEN"], true["piEE"]))]


def test_guard_requires_a_boolean_result():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, theta, _ = _make_model(lcbinint, np)

    @model.guard
    def _(**_):
        return 0.0

    with pytest.raises(TypeError, match="guard\\(\\) must return bool"):
        model.log_prob(theta)
