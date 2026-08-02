#!/usr/bin/env python3
"""High-dimensional Binary/Triple NumPyro NUTS validation on CPU."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import lcbinint  # noqa: E402


NOISE_SIGMA = 0.02
BINARY_NAMES = (
    "t0",
    "u0",
    "log_tE",
    "alpha",
    "log_s",
    "log_q",
    "log_rho",
)
TRIPLE_NAMES = BINARY_NAMES + ("log_q2", "log_sep2", "ang")
PRIOR_SCALES = {
    "t0": 0.1,
    "u0": 0.1,
    "log_tE": 0.3,
    "alpha": 0.3,
    "log_s": 0.2,
    "log_q": 0.5,
    "log_rho": 0.3,
    "log_q2": 0.5,
    "log_sep2": 0.3,
    "ang": 0.3,
}


def configurations(epochs):
    return {
        ("binary", "smooth"): {
            "times": jnp.linspace(-0.18, 0.18, epochs),
            "truth": {
                "t0": 0.0,
                "tE": 1.0,
                "u0": 0.2,
                "alpha": 0.3,
                "s": 1.2,
                "q": 0.1,
                "rho": 0.01,
                "limb_darkening_c": 0.4,
            },
        },
        ("binary", "caustic"): {
            "times": jnp.linspace(-0.0183, 0.0177, epochs),
            "truth": {
                "t0": 0.16094385335766867,
                "tE": 1.0,
                "u0": -0.04971671365184316,
                "alpha": 2.273727691344453,
                "s": 0.9,
                "q": 0.1,
                "rho": 0.01,
                "limb_darkening_c": 0.4,
            },
        },
        ("triple", "smooth"): {
            "times": jnp.linspace(-0.15, 0.15, epochs),
            "truth": {
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
            },
        },
        ("triple", "caustic"): {
            # At t=0 the source centre is the independently audited crossing
            # point (-0.04665563087, 0.07266183991).
            "times": jnp.linspace(-0.0183, 0.0177, epochs),
            "truth": {
                "t0": 0.04665563087,
                "tE": 1.0,
                "u0": 0.07266183991,
                "alpha": 0.0,
                "s": 0.9,
                "q": 0.1,
                "rho": 0.01,
                "q2": 0.003,
                "sep2": 1.5,
                "ang": 1.0,
                "limb_darkening_c": 0.45,
            },
        },
    }


def transformed_names(lens, parameter_set="all"):
    names = BINARY_NAMES if lens == "binary" else TRIPLE_NAMES
    if parameter_set == "trajectory":
        return names[:4]
    if parameter_set == "lens":
        return tuple(name for name in names if name not in names[:4])
    return names


def encode(parameters, names):
    values = []
    for name in names:
        if name.startswith("log_"):
            values.append(np.log(parameters[name[4:]]))
        else:
            values.append(parameters[name])
    return jnp.asarray(values)


def decode(transformed, template, names):
    parameters = dict(template)
    for index, name in enumerate(names):
        if name.startswith("log_"):
            parameters[name[4:]] = jnp.exp(transformed[index])
        else:
            parameters[name] = transformed[index]
    return parameters


def make_curve(lens, use_jax, tolerance=1.0e-4):
    return lcbinint.LightCurve(
        lens=lens,
        options=lcbinint.Options(
            jax=use_jax,
            coordinates="center_of_mass",
            tol=tolerance,
            reltol=tolerance,
        ),
    )


def block(value):
    return jax.block_until_ready(value)


def fisher_whitener(curve, times, truth, names):
    truth_vector = encode(truth, names)

    def evaluate(transformed):
        return curve(times, decode(transformed, truth, names))

    jacobian = block(jax.jit(jax.jacrev(evaluate))(truth_vector))
    scaled = np.asarray(jacobian) / NOISE_SIGMA
    prior_precision = np.diag(
        [1.0 / PRIOR_SCALES[name] ** 2 for name in names]
    )
    fisher = scaled.T @ scaled + prior_precision
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    whitener = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    condition = float(eigenvalues[-1] / eigenvalues[0])
    return truth_vector, jnp.asarray(whitener), {
        "fisher_eigenvalues": eigenvalues.tolist(),
        "fisher_condition_number": condition,
        "raw_jacobian_norms": np.linalg.norm(
            np.asarray(jacobian), axis=0
        ).tolist(),
    }


def make_problem(lens, configuration, seed, parameter_set="all"):
    names = transformed_names(lens, parameter_set)
    curve = make_curve(lens, True)
    times = configuration["times"]
    truth = configuration["truth"]
    truth_vector, whitener, fisher = fisher_whitener(
        curve, times, truth, names
    )

    def predict(standardized):
        transformed = truth_vector + whitener @ standardized
        return curve(times, decode(transformed, truth, names))

    exact = block(predict(jnp.zeros(len(names))))
    observed = exact + NOISE_SIGMA * jax.random.normal(
        jax.random.PRNGKey(seed), exact.shape
    )

    def log_density(standardized):
        residual = (predict(standardized) - observed) / NOISE_SIGMA
        return -0.5 * (
            jnp.sum(residual * residual)
            + jnp.sum((standardized / 3.0) ** 2)
        )

    return {
        "names": names,
        "curve": curve,
        "times": times,
        "truth": truth,
        "truth_vector": truth_vector,
        "whitener": whitener,
        "predict": predict,
        "exact": exact,
        "observed": observed,
        "log_density": log_density,
        "fisher": fisher,
    }


def timed_jax(function, argument, repeat):
    start = time.perf_counter()
    first = block(function(argument))
    compile_and_first = time.perf_counter() - start
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        block(function(argument))
        samples.append(time.perf_counter() - start)
    return first, {
        "compile_and_first_seconds": compile_and_first,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "samples_seconds": samples,
    }


def gradient_audit(log_density, dimension):
    point = jnp.linspace(-0.15, 0.15, dimension)
    value, gradient = block(
        jax.jit(jax.value_and_grad(log_density))(point)
    )
    step = 1.0e-4
    finite_difference = []
    for index in range(dimension):
        direction = jnp.zeros(dimension).at[index].set(step)
        finite_difference.append(
            float(
                (
                    log_density(point + direction)
                    - log_density(point - direction)
                )
                / (2.0 * step)
            )
        )
    finite_difference = np.asarray(finite_difference)
    error = np.asarray(gradient) - finite_difference
    return {
        "value": float(value),
        "finite": bool(
            np.isfinite(value) and np.all(np.isfinite(np.asarray(gradient)))
        ),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "maximum_relative_error": float(
            np.max(
                np.abs(error)
                / np.maximum(np.abs(finite_difference), 1.0)
            )
        ),
    }


def native_gradient_audit(problem, lens):
    native = make_curve(lens, False, tolerance=1.0e-6)
    names = problem["names"]
    truth_vector = np.asarray(problem["truth_vector"])

    def host_parameters(transformed):
        parameters = decode(
            jnp.asarray(transformed), problem["truth"], names
        )
        return {
            name: float(value) if np.ndim(value) == 0 else value
            for name, value in parameters.items()
        }

    jax_gradient = np.asarray(
        jax.grad(lambda active: jnp.sum(
            problem["curve"](
                problem["times"],
                decode(active, problem["truth"], names),
            )
        ))(problem["truth_vector"])
    )
    native_gradient = []
    for index, name in enumerate(names):
        if name in ("t0", "u0"):
            step = 1.0e-4
        elif name == "log_s":
            step = 3.0e-4
        elif name == "log_rho":
            step = 3.0e-3
        else:
            step = 1.0e-3
        direction = np.zeros(len(names))
        direction[index] = step
        plus = np.sum(
            native(
                np.asarray(problem["times"]),
                host_parameters(truth_vector + direction),
            )
        )
        minus = np.sum(
            native(
                np.asarray(problem["times"]),
                host_parameters(truth_vector - direction),
            )
        )
        native_gradient.append((plus - minus) / (2.0 * step))
    native_gradient = np.asarray(native_gradient)
    relative = np.abs(jax_gradient - native_gradient) / np.maximum(
        np.abs(native_gradient), 1.0
    )
    return {
        "jax_gradient": jax_gradient.tolist(),
        "native_finite_difference": native_gradient.tolist(),
        "relative_error": relative.tolist(),
        "maximum_relative_error": float(np.max(relative)),
    }


def leapfrog_audit(log_density, dimension, seed):
    value_and_gradient = jax.jit(jax.value_and_grad(log_density))

    def integrate(position, momentum, step_size, steps):
        _, gradient = block(value_and_gradient(position))
        momentum = momentum + 0.5 * step_size * gradient
        for index in range(steps):
            position = position + step_size * momentum
            if index + 1 != steps:
                _, gradient = block(value_and_gradient(position))
                momentum = momentum + step_size * gradient
        _, gradient = block(value_and_gradient(position))
        return position, momentum + 0.5 * step_size * gradient

    position = jnp.zeros(dimension)
    momentum = jax.random.normal(jax.random.PRNGKey(seed), position.shape)
    initial_energy = -log_density(position) + 0.5 * jnp.sum(momentum**2)
    final_position, final_momentum = integrate(
        position, momentum, 0.02, 4
    )
    final_energy = -log_density(final_position) + 0.5 * jnp.sum(
        final_momentum**2
    )
    recovered_position, recovered_momentum = integrate(
        final_position, final_momentum, -0.02, 4
    )
    return {
        "step_size": 0.02,
        "steps": 4,
        "energy_error": float(final_energy - initial_energy),
        "reversibility_max_abs": float(
            jnp.max(
                jnp.abs(
                    jnp.concatenate(
                        (
                            recovered_position - position,
                            recovered_momentum - momentum,
                        )
                    )
                )
            )
        ),
    }


def run_nuts(problem, args):
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    from numpyro.infer import MCMC, NUTS, init_to_value

    dimension = len(problem["names"])

    def model():
        standardized = numpyro.sample(
            "standardized",
            dist.Normal(
                jnp.zeros(dimension), 3.0 * jnp.ones(dimension)
            ).to_event(1),
        )
        numpyro.sample(
            "flux",
            dist.Normal(problem["predict"](standardized), NOISE_SIGMA),
            obs=problem["observed"],
        )

    sampler = MCMC(
        NUTS(
            model,
            target_accept_prob=args.target_accept,
            max_tree_depth=args.max_tree_depth,
            dense_mass=args.dense_mass,
            init_strategy=init_to_value(
                values={"standardized": jnp.zeros(dimension)}
            ),
        ),
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method="sequential",
        progress_bar=False,
    )
    start = time.perf_counter()
    sampler.run(
        jax.random.PRNGKey(args.seed + 100),
        extra_fields=(
            "accept_prob",
            "diverging",
            "energy",
            "num_steps",
        ),
    )
    elapsed = time.perf_counter() - start
    samples = np.asarray(
        sampler.get_samples(group_by_chain=True)["standardized"]
    )
    extra = {
        name: np.asarray(value)
        for name, value in sampler.get_extra_fields(
            group_by_chain=True
        ).items()
    }
    ess = np.asarray(effective_sample_size(jnp.asarray(samples)))
    result = {
        "elapsed_seconds_compile_warmup_sampling": elapsed,
        "draws_per_second": args.chains * args.samples / elapsed,
        "effective_samples": ess.tolist(),
        "minimum_effective_samples_per_second": float(np.min(ess) / elapsed),
        "posterior_standardized_mean": np.mean(
            samples, axis=(0, 1)
        ).tolist(),
        "posterior_standardized_sd": np.std(
            samples, axis=(0, 1), ddof=1
        ).tolist(),
        "truth_maximum_posterior_z_score": float(
            np.max(
                np.abs(np.mean(samples, axis=(0, 1)))
                / np.maximum(np.std(samples, axis=(0, 1), ddof=1), 1.0e-12)
            )
        ),
        "divergences": int(np.count_nonzero(extra["diverging"])),
        "mean_accept_probability": float(np.mean(extra["accept_prob"])),
        "median_leapfrog_steps": float(np.median(extra["num_steps"])),
        "maximum_leapfrog_steps": int(np.max(extra["num_steps"])),
    }
    if args.chains >= 2:
        result["split_rhat"] = np.asarray(
            split_gelman_rubin(jnp.asarray(samples))
        ).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", choices=("binary", "triple"), required=True)
    parser.add_argument(
        "--case", choices=("smooth", "caustic"), required=True
    )
    parser.add_argument(
        "--parameter-set",
        choices=("trajectory", "lens", "all"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--dense-mass", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--skip-nuts", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    configuration = configurations(args.epochs)[(args.lens, args.case)]
    problem = make_problem(
        args.lens, configuration, args.seed, args.parameter_set
    )
    zero = jnp.zeros(len(problem["names"]))
    _, forward = timed_jax(
        jax.jit(problem["predict"]), zero, args.repeat
    )
    _, value_and_grad = timed_jax(
        jax.jit(jax.value_and_grad(problem["log_density"])),
        zero,
        args.repeat,
    )

    native = make_curve(args.lens, False)
    native(np.asarray(problem["times"]), problem["truth"])
    native_samples = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        native(np.asarray(problem["times"]), problem["truth"])
        native_samples.append(time.perf_counter() - start)

    output = {
        "configuration": vars(args)
        | {
            "platform": platform.platform(),
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "noise_sigma": NOISE_SIGMA,
        },
        "parameter_names": problem["names"],
        "truth": problem["truth"],
        "magnification_range": [
            float(jnp.min(problem["exact"])),
            float(jnp.max(problem["exact"])),
        ],
        "fisher": problem["fisher"],
        "gradient_audit": gradient_audit(
            problem["log_density"], len(problem["names"])
        ),
        "native_gradient_audit": native_gradient_audit(
            problem, args.lens
        ),
        "leapfrog_audit": leapfrog_audit(
            problem["log_density"], len(problem["names"]), args.seed
        ),
        "timings": {
            "jax_forward": forward,
            "jax_value_and_grad": value_and_grad,
            "native_forward": {
                "median_seconds": statistics.median(native_samples),
                "minimum_seconds": min(native_samples),
                "samples_seconds": native_samples,
            },
        },
        "nuts": None if args.skip_nuts else run_nuts(problem, args),
    }
    text = json.dumps(output, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
