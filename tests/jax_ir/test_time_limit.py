"""Time-windowed parallax and satellite ephemeris constants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcbinint
from lcbinint.jax_backend import _earth_ephemeris
from lcbinint_jax import annual_parallax_offsets


jax.config.update("jax_enable_x64", True)

_PARAMETERS = {
    "t0": 9000.0,
    "tE": 20.0,
    "u0": 0.1,
    "alpha": 0.2,
    "s": 1.1,
    "q": 10.0,
    "rho": 0.0,
    "piEN": 0.1,
    "piEE": 0.05,
}
_TIMES = np.asarray((8992.2, 9000.0, 9008.7))
_LIMIT = (8990.0, 9010.0)


def _model():
    return lcbinint.Model(
        parallax=True,
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        t_ref=9000.0,
    )


def test_options_validate_and_preserve_time_limit():
    options = lcbinint.Options(jax=True, t_lim=[8990, 9010])
    assert options.t_lim == _LIMIT
    assert "t_lim=(8990.0, 9010.0)" in repr(options)

    options.t_lim = (8991, 9009)
    assert options.t_lim == (8991.0, 9009.0)
    curve = lcbinint.LightCurve(model=_model(), options=options)
    with pytest.raises(AttributeError, match="fixed at construction"):
        curve.options.t_lim = (8992, 9008)

    for invalid in ((1,), (1, 1), (2, 1), (np.nan, 2)):
        with pytest.raises((TypeError, ValueError)):
            lcbinint.Options(t_lim=invalid)


def test_annual_ephemeris_window_is_small_and_numerically_identical():
    full_ephemeris = _earth_ephemeris()
    limited_ephemeris = _earth_ephemeris(_LIMIT, 9000.0)
    assert limited_ephemeris.time.size < full_ephemeris.time.size / 100
    assert limited_ephemeris.time.size == 25

    full = lcbinint.LightCurve(
        model=_model(),
        options=lcbinint.Options(jax=True, coordinates="vbm"),
    )
    limited = lcbinint.LightCurve(
        model=_model(),
        options=lcbinint.Options(
            jax=True, coordinates="vbm", t_lim=_LIMIT
        ),
    )
    np.testing.assert_array_equal(
        limited(jnp.asarray(_TIMES), _PARAMETERS),
        full(jnp.asarray(_TIMES), _PARAMETERS),
    )
    full_gradient = jax.grad(
        lambda pi_en: jnp.sum(
            full(jnp.asarray(_TIMES), {**_PARAMETERS, "piEN": pi_en})
        )
    )(_PARAMETERS["piEN"])
    limited_gradient = jax.grad(
        lambda pi_en: jnp.sum(
            limited(jnp.asarray(_TIMES), {**_PARAMETERS, "piEN": pi_en})
        )
    )(_PARAMETERS["piEN"])
    np.testing.assert_array_equal(limited_gradient, full_gradient)


def test_annual_window_retains_a_distant_reference_time_neighborhood():
    reference_time = 8500.0
    full = _earth_ephemeris()
    limited = _earth_ephemeris(_LIMIT, reference_time)
    assert limited.time.size < 40

    arguments = (
        jnp.asarray(_TIMES),
        0.1,
        0.05,
        270.0,
        -30.0,
        reference_time,
    )
    full_offsets = annual_parallax_offsets(*arguments, full)
    limited_offsets = annual_parallax_offsets(*arguments, limited)
    np.testing.assert_array_equal(limited_offsets[0], full_offsets[0])
    np.testing.assert_array_equal(limited_offsets[1], full_offsets[1])


def test_space_ephemeris_is_windowed_for_native_and_jax():
    table_times = np.arange(8950.0, 9051.0)
    table = np.column_stack(
        (
            table_times + 2450000.0,
            0.01 * (table_times - 9000.0),
            0.005 * (table_times - 9000.0),
            0.01 + 1.0e-5 * (table_times - 9000.0),
        )
    )
    site = lcbinint.obs.Site("space", table)
    full_native = lcbinint.LightCurve(
        model=_model(),
        site=site,
        options=lcbinint.Options(coordinates="vbm"),
    )
    limited_native = lcbinint.LightCurve(
        model=_model(),
        site=site,
        options=lcbinint.Options(coordinates="vbm", t_lim=_LIMIT),
    )
    assert limited_native.site.ephemeris_time.size == 24
    assert limited_native.site.ephemeris_time.size < site.ephemeris_time.size
    np.testing.assert_array_equal(
        limited_native.source_trajectory(_TIMES, _PARAMETERS).x,
        full_native.source_trajectory(_TIMES, _PARAMETERS).x,
    )
    np.testing.assert_array_equal(
        limited_native.source_trajectory(_TIMES, _PARAMETERS).y,
        full_native.source_trajectory(_TIMES, _PARAMETERS).y,
    )

    limited_jax = lcbinint.LightCurve(
        model=_model(),
        site=site,
        options=lcbinint.Options(
            jax=True, coordinates="vbm", t_lim=_LIMIT
        ),
    )
    np.testing.assert_allclose(
        limited_jax.source_trajectory(
            jnp.asarray(_TIMES), _PARAMETERS
        ).x,
        limited_native.source_trajectory(_TIMES, _PARAMETERS).x,
        rtol=0.0,
        atol=2.0e-15,
    )


def test_time_limit_rejects_concrete_times_and_masks_traced_times():
    native = lcbinint.LightCurve(
        model=_model(),
        options=lcbinint.Options(coordinates="vbm", t_lim=_LIMIT),
    )
    jax_curve = lcbinint.LightCurve(
        model=_model(),
        options=lcbinint.Options(
            jax=True, coordinates="vbm", t_lim=_LIMIT
        ),
    )
    with pytest.raises(ValueError, match="Options.t_lim"):
        native(np.asarray((8989.0,)), _PARAMETERS)
    with pytest.raises(ValueError, match="Options.t_lim"):
        jax_curve(jnp.asarray((8989.0,)), _PARAMETERS)

    result = jax.jit(lambda times: jax_curve(times, _PARAMETERS))(
        jnp.asarray((8989.0, 9000.0, 9011.0))
    )
    assert jnp.isnan(result[0])
    assert jnp.isfinite(result[1])
    assert jnp.isnan(result[2])
