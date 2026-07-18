import pytest


def test_model_lens_accepts_short_names():
    lcbinint = pytest.importorskip("lcbinint")

    binary = lcbinint.Model(lens="binary")
    assert binary.lens == "binary"

    triple = lcbinint.Model(lens="triple")
    assert triple.lens == "triple"
    assert "lens=triple" in repr(triple)


def test_light_curve_lens_is_stored_in_model_and_normalized():
    lcbinint = pytest.importorskip("lcbinint")

    binary = lcbinint.LightCurve(lens="binary")
    assert binary.lens == "binary"
    assert binary.model.lens == "binary"

    model = lcbinint.Model(lens="triple")
    triple = lcbinint.LightCurve(model=model)
    assert triple.lens == "triple"
    assert triple.model.lens == "triple"


def test_light_curve_rejects_unknown_lens():
    lcbinint = pytest.importorskip("lcbinint")

    with pytest.raises(ValueError, match="lens must be"):
        lcbinint.LightCurve(lens="quad")


def test_light_curve_lens_matches_q2_parameters():
    lcbinint = pytest.importorskip("lcbinint")
    params = dict(t0=0.0, tE=1.0, u0=0.2, s=1.0, q=0.1, rho=0.0)

    with pytest.raises(RuntimeError, match="requires a positive q2"):
        lcbinint.LightCurve(lens="triple")([0.0], params)
    with pytest.raises(RuntimeError, match="cannot be used with a positive q2"):
        lcbinint.LightCurve(lens="binary")([0.0], {**params, "q2": 0.01, "sep2": 0.8})

    result = lcbinint.LightCurve(lens="triple")(
        [0.0], {**params, "q2": 0.01, "sep2": 0.8})
    assert result.shape == (1,)


def test_legacy_lens_aliases_are_rejected():
    lcbinint = pytest.importorskip("lcbinint")

    with pytest.raises(ValueError, match="lens must be"):
        lcbinint.Model(lens="binary_lens")
    with pytest.raises(ValueError, match="lens must be"):
        lcbinint.LightCurve(lens="triple_lens")


def test_obsolete_model_names_are_not_exported():
    lcbinint = pytest.importorskip("lcbinint")

    assert not hasattr(lcbinint, "Effects")
    assert not hasattr(lcbinint, "ModelSpec")
    with pytest.raises(KeyError, match="unknown option 'spec'"):
        lcbinint.LightCurve(spec=lcbinint.Model())
