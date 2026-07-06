import math
import os

import pytest


def _make_parallax_model(lcbinint, np, include_cartesian_params=False):
    true = {
        "t0": 8000.0,
        "tE": 20.0,
        "u0": 0.10,
        "s": 1.3,
        "q": 0.10,
        "alpha": 0.5,
        "piEN": 0.05,
        "piEE": 0.02,
    }
    rng = np.random.default_rng(7)
    times = np.sort(rng.uniform(true["t0"] - 2.0, true["t0"] + 2.0, 8))

    light_curve = lcbinint.lc.LightCurve()
    flux = 1000.0 * light_curve(times, true) + 10.0
    flux_err = np.full(len(times), 30.0)
    data = lcbinint.obs.LightCurveData(times, flux, flux_err, name="tiny")

    model = lcbinint.bayes.Model(light_curve=light_curve, data=data)
    model.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    model.param("tE", lcbinint.bayes.LogUniform(1.0, 100.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.param("s", lcbinint.bayes.Uniform(0.5, 2.0))
    model.param("q", lcbinint.bayes.LogUniform(1.0e-3, 0.5))
    model.param("alpha", lcbinint.bayes.Uniform(0.0, math.pi))
    if include_cartesian_params:
        model.param("piEN", lcbinint.bayes.Uniform(-1.0, 1.0))
        model.param("piEE", lcbinint.bayes.Uniform(-1.0, 1.0))
    model.likelihood()

    rp = model.reparam(["piEN", "piEE"])
    rp.param("piE", lcbinint.bayes.LogUniform(0.01, 1.0))
    rp.param("phi_piE", lcbinint.bayes.Uniform(0.0, 2.0 * math.pi))

    @rp.transform
    def _(piE, phi_piE):
        return {
            "piEN": piE * math.cos(phi_piE),
            "piEE": piE * math.sin(phi_piE),
        }

    return model, true


def _make_plain_model(lcbinint, np):
    true = {
        "t0": 8000.0,
        "tE": 20.0,
        "u0": 0.10,
        "s": 1.3,
        "q": 0.10,
        "alpha": 0.5,
    }
    times = np.array([7999.5, 8000.0, 8000.5, 8001.0])

    light_curve = lcbinint.lc.LightCurve()
    flux = 1000.0 * light_curve(times, true) + 10.0
    flux_err = np.full(len(times), 30.0)
    data = lcbinint.obs.LightCurveData(times, flux, flux_err, name="tiny")

    model = lcbinint.bayes.Model(light_curve=light_curve, data=data)
    model.param("t0", lcbinint.bayes.Uniform(7999.0, 8001.0))
    model.param("tE", lcbinint.bayes.LogUniform(1.0, 100.0))
    model.param("u0", lcbinint.bayes.Uniform(0.0, 1.0))
    model.param("s", lcbinint.bayes.Uniform(0.5, 2.0))
    model.param("q", lcbinint.bayes.LogUniform(1.0e-3, 0.5))
    model.param("alpha", lcbinint.bayes.Uniform(0.0, math.pi))
    model.likelihood()
    return model


def test_model_reparam_replaces_targets_without_model_param_registration():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, true = _make_parallax_model(lcbinint, np)
    adapter = model._sampling_adapter()

    assert adapter.param_names == [
        "t0",
        "tE",
        "u0",
        "s",
        "q",
        "alpha",
        "piE",
        "phi_piE",
    ]
    assert adapter._sample_transforms == [
        "identity",
        "log",
        "identity",
        "identity",
        "log",
        "identity",
        "log",
        "identity",
    ]

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
    assert math.isfinite(adapter.log_prob(theta))


def test_run_sampler_returns_sampling_space_chain_for_reparam():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _ = _make_parallax_model(lcbinint, np)
    chain = lcbinint.run_sampler(
        model,
        nsteps=2,
        burnin=0,
        options=lcbinint.SamplerOptions(
            nwalkers=8,
            seed=3,
            log_path="",
            auto_stop=False,
        ),
    )

    assert chain.param_names == [
        "t0",
        "tE",
        "u0",
        "s",
        "q",
        "alpha",
        "piE",
        "phi_piE",
    ]
    assert chain.transforms == [
        "identity",
        "log",
        "identity",
        "identity",
        "log",
        "identity",
        "log",
        "identity",
    ]
    assert chain.flat_samples.shape == (16, 8)
    assert chain.samples.shape == (16, 8)
    assert chain.get_samples().shape == (16, 8)
    assert chain.get_samples(flat=False).shape == (2, 8, 8)
    assert chain.get_samples(discard=1, thin=1).shape == (8, 8)
    assert chain.get_log_prob().shape == (16,)
    assert chain.get_log_prob(flat=False).shape == (2, 8)
    assert set(chain.fluxes["tiny"]) == {"Fs", "Fb"}
    assert chain.fluxes["tiny"]["Fs"].shape == (16,)
    assert chain.fluxes["tiny"]["Fb"].shape == (16,)
    assert chain.get_fluxes()["tiny"]["Fs"].shape == (16,)
    assert chain.get_fluxes(flat=False)["tiny"]["Fs"].shape == (2, 8)
    assert chain.get_fluxes(discard=1)["tiny"]["Fs"].shape == (8,)


def test_model_reparam_targets_are_removed_even_if_registered_as_model_params():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _ = _make_parallax_model(
        lcbinint,
        np,
        include_cartesian_params=True,
    )
    adapter = model._sampling_adapter()

    assert "piEN" not in adapter.param_names
    assert "piEE" not in adapter.param_names
    assert adapter.param_names[-2:] == ["piE", "phi_piE"]


def test_model_reparam_validation_errors():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    model, _ = _make_parallax_model(lcbinint, np)
    with pytest.raises(ValueError, match="already reparameterized"):
        model.reparam(["piEN"])

    light_curve = lcbinint.lc.LightCurve()
    data = lcbinint.obs.LightCurveData(
        np.array([8000.0, 8001.0]),
        np.array([1.0, 1.0]),
        np.array([0.1, 0.1]),
    )
    bad = lcbinint.bayes.Model(light_curve=light_curve, data=data)
    bad.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    bad.likelihood()

    empty = bad.reparam(["piEN"])
    with pytest.raises(RuntimeError, match="register at least one"):
        bad._sampling_adapter()

    empty.param("piE", lcbinint.bayes.LogUniform(0.01, 1.0))
    with pytest.raises(RuntimeError, match="call transform"):
        bad._sampling_adapter()


def test_model_reparam_transform_must_return_targets():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    light_curve = lcbinint.lc.LightCurve()
    data = lcbinint.obs.LightCurveData(
        np.array([8000.0, 8001.0, 8002.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.1, 0.1, 0.1]),
    )
    model = lcbinint.bayes.Model(light_curve=light_curve, data=data)
    model.param("t0", lcbinint.bayes.Uniform(7990.0, 8010.0))
    model.likelihood()

    rp = model.reparam(["piEN", "piEE"])
    rp.param("piE", lcbinint.bayes.LogUniform(0.01, 1.0))

    @rp.transform
    def _(piE):
        return {"piEN": piE}

    adapter = model._sampling_adapter()
    with pytest.raises(RuntimeError, match="piEE"):
        adapter.log_prob([8000.0, math.log(0.1)])


def test_run_sampler_h5_roundtrip_preserves_fluxes(tmp_path):
    lcbinint = pytest.importorskip("lcbinint")
    pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")

    model, _ = _make_parallax_model(lcbinint, np)
    path = tmp_path / "chain.h5"
    chain = lcbinint.run_sampler(
        model,
        nsteps=2,
        burnin=0,
        options=lcbinint.SamplerOptions(
            nwalkers=8,
            seed=4,
            log_path="",
            log_every=1,
            auto_stop=False,
            h5_path=os.fspath(path),
        ),
    )
    loaded = lcbinint.load_chain(os.fspath(path))

    assert loaded.param_names == chain.param_names
    assert set(loaded.fluxes) == set(chain.fluxes)
    assert np.asarray(loaded.fluxes["tiny"]["Fs"]).tolist() == pytest.approx(
        np.asarray(chain.fluxes["tiny"]["Fs"]).tolist()
    )
    assert np.asarray(loaded.fluxes["tiny"]["Fb"]).tolist() == pytest.approx(
        np.asarray(chain.fluxes["tiny"]["Fb"]).tolist()
    )


def test_run_sampler_h5_roundtrip_preserves_fluxes_for_cpp_model(tmp_path):
    lcbinint = pytest.importorskip("lcbinint")
    pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")

    model = _make_plain_model(lcbinint, np)
    path = tmp_path / "chain_cpp.h5"
    chain = lcbinint.run_sampler(
        model,
        nsteps=2,
        burnin=0,
        options=lcbinint.SamplerOptions(
            nwalkers=8,
            seed=5,
            log_path="",
            log_every=1,
            auto_stop=False,
            h5_path=os.fspath(path),
        ),
    )
    loaded = lcbinint.load_chain(os.fspath(path))

    assert set(chain.fluxes["tiny"]) == {"Fs", "Fb"}
    assert chain.fluxes["tiny"]["Fs"].shape == (16,)
    assert set(loaded.fluxes) == {"tiny"}
    assert np.asarray(loaded.fluxes["tiny"]["Fs"]).tolist() == pytest.approx(
        np.asarray(chain.fluxes["tiny"]["Fs"]).tolist()
    )
    assert np.asarray(loaded.fluxes["tiny"]["Fb"]).tolist() == pytest.approx(
        np.asarray(chain.fluxes["tiny"]["Fb"]).tolist()
    )
