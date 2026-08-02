import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcbinint


TIMES = jnp.linspace(-0.12, 0.12, 6)
TRUTH = {
    "t0": 0.0,
    "tE": 1.0,
    "u0": 0.2,
    "alpha": 0.3,
    "s": 1.2,
    "q": 0.1,
    "rho": 0.01,
    "limb_darkening_c": 0.4,
}
TRIPLE_TIMES = jnp.linspace(-0.08, 0.08, 4)
TRIPLE_TRUTH = {
    "t0": 0.0,
    "tE": 1.0,
    "u0": 0.25,
    "alpha": 0.3,
    "s": 1.0,
    "q": 0.1,
    "rho": 0.02,
    "q2": 0.03,
    "sep2": 0.7,
    "ang": 0.8,
    "limb_darkening_c": 0.4,
}
TRIPLE_SCALES = jnp.asarray(
    (1.0e-4, 1.0e-4, 2.0e-3, 2.0e-3, 1.0e-3,
     2.0e-3, 2.0e-3, 2.0e-3, 2.0e-3, 2.0e-3)
)


def _curve():
    return lcbinint.LightCurve(
        options=lcbinint.Options(
            jax=True,
            coordinates="center_of_mass",
            tol=1.0e-4,
            reltol=1.0e-4,
        )
    )


def _triple_curve():
    return lcbinint.LightCurve(
        lens="triple",
        options=lcbinint.Options(
            jax=True,
            coordinates="center_of_mass",
            tol=1.0e-4,
            reltol=1.0e-4,
        ),
    )


def _triple_parameters(standardized):
    active = TRIPLE_SCALES * standardized
    return {
        **TRIPLE_TRUTH,
        "t0": TRIPLE_TRUTH["t0"] + active[0],
        "u0": TRIPLE_TRUTH["u0"] + active[1],
        "tE": TRIPLE_TRUTH["tE"] * jnp.exp(active[2]),
        "alpha": TRIPLE_TRUTH["alpha"] + active[3],
        "s": TRIPLE_TRUTH["s"] * jnp.exp(active[4]),
        "q": TRIPLE_TRUTH["q"] * jnp.exp(active[5]),
        "rho": TRIPLE_TRUTH["rho"] * jnp.exp(active[6]),
        "q2": TRIPLE_TRUTH["q2"] * jnp.exp(active[7]),
        "sep2": TRIPLE_TRUTH["sep2"] * jnp.exp(active[8]),
        "ang": TRIPLE_TRUTH["ang"] + active[9],
    }


def _standardized_log_density(curve, observed, scale):
    def log_density(position):
        parameters = {
            **TRUTH,
            "t0": TRUTH["t0"] + scale * position[0],
        }
        residual = (curve(TIMES, parameters) - observed) / 0.02
        return -0.5 * (jnp.sum(residual * residual) + position[0] ** 2)

    return log_density


def _leapfrog(log_density, position, momentum, step_size, steps):
    gradient = jax.grad(log_density)
    momentum = momentum + 0.5 * step_size * gradient(position)
    for index in range(steps):
        position = position + step_size * momentum
        if index + 1 != steps:
            momentum = momentum + step_size * gradient(position)
    momentum = momentum + 0.5 * step_size * gradient(position)
    return position, momentum


def test_public_jax_likelihood_has_reverse_mode_gradient():
    curve = _curve()
    observed = curve(TIMES, TRUTH)
    scale = 1.0e-4
    log_density = _standardized_log_density(curve, observed, scale)
    position = jnp.asarray((0.2,))

    value, gradient = jax.jit(jax.value_and_grad(log_density))(position)
    finite_difference = (
        log_density(position + 1.0e-4)
        - log_density(position - 1.0e-4)
    ) / 2.0e-4

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_allclose(
        gradient[0], finite_difference, rtol=2.0e-3, atol=2.0e-4
    )


def test_public_jax_likelihood_leapfrog_is_reversible():
    curve = _curve()
    observed = curve(TIMES, TRUTH)
    log_density = _standardized_log_density(curve, observed, 1.0e-4)
    position = jnp.asarray((0.15,))
    momentum = jnp.asarray((-0.4,))

    final_position, final_momentum = _leapfrog(
        log_density, position, momentum, 0.03, 3
    )
    recovered_position, recovered_momentum = _leapfrog(
        log_density, final_position, final_momentum, -0.03, 3
    )

    np.testing.assert_allclose(
        recovered_position, position, rtol=0.0, atol=2.0e-10
    )
    np.testing.assert_allclose(
        recovered_momentum, momentum, rtol=0.0, atol=2.0e-10
    )


def test_public_jax_likelihood_runs_in_numpyro_nuts():
    numpyro = pytest.importorskip("numpyro")
    distributions = pytest.importorskip("numpyro.distributions")
    infer = pytest.importorskip("numpyro.infer")

    curve = _curve()
    observed = curve(TIMES, TRUTH)
    scale = 1.0e-4

    def model():
        standardized_t0 = numpyro.sample(
            "standardized_t0", distributions.Normal(0.0, 1.0)
        )
        parameters = {
            **TRUTH,
            "t0": TRUTH["t0"] + scale * standardized_t0,
        }
        numpyro.sample(
            "flux",
            distributions.Normal(curve(TIMES, parameters), 0.02),
            obs=observed,
        )

    kernel = infer.NUTS(
        model,
        init_strategy=infer.init_to_value(
            values={"standardized_t0": jnp.asarray(0.0)}
        ),
    )
    sampler = infer.MCMC(
        kernel,
        num_warmup=12,
        num_samples=12,
        num_chains=1,
        progress_bar=False,
    )
    sampler.run(
        jax.random.PRNGKey(421),
        extra_fields=("diverging", "accept_prob"),
    )
    samples = sampler.get_samples()["standardized_t0"]
    diagnostics = sampler.get_extra_fields()

    assert samples.shape == (12,)
    assert jnp.all(jnp.isfinite(samples))
    assert not jnp.any(diagnostics["diverging"])
    assert jnp.mean(diagnostics["accept_prob"]) > 0.5


def test_public_triple_likelihood_has_ten_parameter_reverse_gradient():
    curve = _triple_curve()
    observed = curve(TRIPLE_TIMES, TRIPLE_TRUTH)

    def log_density(standardized):
        residual = (
            curve(TRIPLE_TIMES, _triple_parameters(standardized)) - observed
        ) / 0.02
        return -0.5 * (
            jnp.sum(residual * residual)
            + jnp.sum(standardized * standardized)
        )

    value, gradient = jax.jit(jax.value_and_grad(log_density))(
        jnp.zeros(10)
    )
    assert jnp.isfinite(value)
    assert gradient.shape == (10,)
    assert jnp.all(jnp.isfinite(gradient))


def test_public_triple_likelihood_runs_as_ten_parameter_numpyro_nuts():
    numpyro = pytest.importorskip("numpyro")
    distributions = pytest.importorskip("numpyro.distributions")
    infer = pytest.importorskip("numpyro.infer")

    curve = _triple_curve()
    observed = curve(TRIPLE_TIMES, TRIPLE_TRUTH)

    def model():
        standardized = numpyro.sample(
            "standardized",
            distributions.Normal(jnp.zeros(10), jnp.ones(10)).to_event(1),
        )
        numpyro.sample(
            "flux",
            distributions.Normal(
                curve(TRIPLE_TIMES, _triple_parameters(standardized)),
                0.02,
            ),
            obs=observed,
        )

    sampler = infer.MCMC(
        infer.NUTS(
            model,
            max_tree_depth=2,
            init_strategy=infer.init_to_value(
                values={"standardized": jnp.zeros(10)}
            ),
        ),
        num_warmup=3,
        num_samples=3,
        num_chains=1,
        progress_bar=False,
    )
    sampler.run(jax.random.PRNGKey(913))
    samples = sampler.get_samples()["standardized"]
    assert samples.shape == (3, 10)
    assert jnp.all(jnp.isfinite(samples))
