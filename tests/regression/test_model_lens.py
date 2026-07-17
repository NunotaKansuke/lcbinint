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
