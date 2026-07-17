import pytest


def test_model_spec_lens_accepts_short_names():
    lcbinint = pytest.importorskip("lcbinint")

    binary = lcbinint.ModelSpec(lens="binary")
    assert binary.lens == "binary"

    triple = lcbinint.ModelSpec(lens="triple")
    assert triple.lens == "triple"
    assert "lens=triple" in repr(triple)


def test_light_curve_lens_is_stored_in_spec_and_normalized():
    lcbinint = pytest.importorskip("lcbinint")

    binary = lcbinint.LightCurve(lens="binary")
    assert binary.lens == "binary"
    assert binary.spec.lens == "binary"

    spec = lcbinint.ModelSpec(lens="triple")
    triple = lcbinint.LightCurve(spec=spec)
    assert triple.lens == "triple"
    assert triple.spec.lens == "triple"


def test_light_curve_rejects_unknown_lens():
    lcbinint = pytest.importorskip("lcbinint")

    with pytest.raises(ValueError, match="lens must be"):
        lcbinint.LightCurve(lens="quad")


def test_legacy_lens_aliases_are_rejected():
    lcbinint = pytest.importorskip("lcbinint")

    with pytest.raises(ValueError, match="lens must be"):
        lcbinint.ModelSpec(lens="binary_lens")
    with pytest.raises(ValueError, match="lens must be"):
        lcbinint.LightCurve(lens="triple_lens")


def test_effects_alias_is_not_exported():
    lcbinint = pytest.importorskip("lcbinint")

    assert not hasattr(lcbinint, "Effects")
    assert not hasattr(lcbinint, "Effects")
