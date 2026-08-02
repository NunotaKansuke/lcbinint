#!/usr/bin/env python3
"""End-to-end CPU HMC and gradient benchmark for binary finite sources.

The benchmark deliberately separates compile time, steady-state log-density
cost, leapfrog quality, dispatcher changes, and NUTS sampling.  Parameters are
standardized with a local Fisher scale so that sampler pathologies are not
caused merely by using days and Einstein-radius units in the same mass matrix.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import lcbinint  # noqa: E402
from lcbinint_jax import binary_magnification_trajectory  # noqa: E402


PARAMETER_NAMES = ("t0", "u0")
NOISE_SIGMA = 0.02


def cases(epochs):
    caustic = np.asarray((0.06611188225495068, 0.1549319030240759))
    normal = np.asarray((-0.64645695, -0.76295047))
    normal /= np.linalg.norm(normal)
    angle = float(np.arctan2(-normal[1], normal[0]))
    perpendicular = np.asarray((np.sin(angle), np.cos(angle)))
    return {
        "smooth": {
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
        "caustic": {
            # This crosses the audited s=0.9, q=0.1 fold along its normal.
            # The epoch grid avoids exact source-limb tangency, where the
            # mathematical light curve itself has unequal one-sided slopes.
            "times": jnp.linspace(-0.0183, 0.0177, epochs),
            "truth": {
                "t0": -float(caustic @ normal),
                "tE": 1.0,
                "u0": float(caustic @ perpendicular),
                "alpha": angle,
                "s": 0.9,
                "q": 0.1,
                "rho": 0.01,
                "limb_darkening_c": 0.4,
            },
        },
    }


def public_curve(jax_enabled):
    return lcbinint.LightCurve(
        options=lcbinint.Options(
            jax=jax_enabled,
            coordinates="center_of_mass",
            tol=1.0e-4,
            reltol=1.0e-4,
        )
    )


def parameters_from_standardized(truth, scales, standardized):
    return {
        **truth,
        **{
            name: truth[name] + scales[index] * standardized[index]
            for index, name in enumerate(PARAMETER_NAMES)
        },
    }


def source_positions(times, parameters):
    tau = (times - parameters["t0"]) / parameters["tE"]
    angle = parameters["alpha"]
    source_x = (
        parameters["u0"] * jnp.sin(angle) + tau * jnp.cos(angle)
    )
    source_y = (
        parameters["u0"] * jnp.cos(angle) - tau * jnp.sin(angle)
    )
    return source_x, source_y


def microlux_magnification(times, parameters):
    from microlux.basic_function import to_lowmass
    from microlux.limb_darkening import LinearLimbDarkening
    from microlux.trajectory_model import extended_light_curve_from_trajectory_l

    source_x, source_y = source_positions(times, parameters)
    trajectory = to_lowmass(
        parameters["s"],
        parameters["q"],
        source_x + 1j * source_y,
    )
    return extended_light_curve_from_trajectory_l(
        trajectory,
        parameters["s"],
        parameters["q"],
        parameters["rho"],
        tol=1.0e-4,
        retol=1.0e-4,
        default_strategy=(30, 30, 60, 120, 240),
        analytic=True,
        limb_darkening=LinearLimbDarkening(
            parameters["limb_darkening_c"]
        ),
        n_annuli=80,
    )


def vbm_magnification(times, parameters):
    import VBMicrolensing

    source_x, source_y = source_positions(
        np.asarray(times), parameters
    )
    engine = VBMicrolensing.VBMicrolensing()
    engine.Tol = 1.0e-4
    engine.RelTol = 1.0e-4
    engine.a1 = parameters["limb_darkening_c"]
    return np.asarray(
        [
            engine.BinaryMagDark(
                parameters["s"],
                parameters["q"],
                float(x),
                float(y),
                parameters["rho"],
                1.0e-4,
            )
            for x, y in zip(source_x, source_y)
        ]
    )


def block(value):
    return jax.block_until_ready(value)


def timed_jax(function, argument, repeat):
    start = time.perf_counter()
    first = block(function(argument))
    compile_seconds = time.perf_counter() - start
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        block(function(argument))
        samples.append(time.perf_counter() - start)
    return first, {
        "compile_and_first_seconds": compile_seconds,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "samples_seconds": samples,
    }


def timed_python(function, repeat):
    function()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "samples_seconds": samples,
    }


def fisher_scales(curve, times, truth):
    def raw_curve(raw):
        parameters = {
            **truth,
            **{
                name: raw[index]
                for index, name in enumerate(PARAMETER_NAMES)
            },
        }
        return curve(times, parameters)

    raw_truth = jnp.asarray([truth[name] for name in PARAMETER_NAMES])
    jacobian = jax.jit(jax.jacrev(raw_curve))(raw_truth)
    block(jacobian)
    norms = np.linalg.norm(np.asarray(jacobian), axis=0)
    scales = NOISE_SIGMA / np.maximum(norms, 1.0e-12)
    # A near-null local direction should not be allowed to jump to another
    # topology solely because of preconditioning.
    scales = np.minimum(scales, np.asarray((0.01, 0.01)))
    return jnp.asarray(scales), norms


def make_log_density(engine, times, truth, scales, observed):
    curve = public_curve(True)

    def log_density(standardized):
        parameters = parameters_from_standardized(
            truth, scales, standardized
        )
        predicted = (
            curve(times, parameters)
            if engine == "lcbinint"
            else microlux_magnification(times, parameters)
        )
        residual = (predicted - observed) / NOISE_SIGMA
        return -0.5 * (
            jnp.sum(residual * residual)
            + jnp.sum((standardized / 3.0) ** 2)
        )

    return log_density


def leapfrog_audit(log_density, seed):
    value_and_gradient = jax.jit(jax.value_and_grad(log_density))

    def leapfrog(position, momentum, step_size, steps):
        _, gradient = block(value_and_gradient(position))
        momentum = momentum + 0.5 * step_size * gradient
        for index in range(steps):
            position = position + step_size * momentum
            if index + 1 != steps:
                _, gradient = block(value_and_gradient(position))
                momentum = momentum + step_size * gradient
        _, gradient = block(value_and_gradient(position))
        momentum = momentum + 0.5 * step_size * gradient
        return position, momentum

    key = jax.random.PRNGKey(seed)
    energy_errors = []
    reversibility = []
    for _ in range(8):
        key, momentum_key = jax.random.split(key)
        position = jnp.zeros(len(PARAMETER_NAMES))
        momentum = jax.random.normal(momentum_key, position.shape)
        initial_energy = -log_density(position) + 0.5 * jnp.sum(momentum**2)
        final_position, final_momentum = leapfrog(
            position, momentum, 0.025, 5
        )
        final_energy = -log_density(final_position) + 0.5 * jnp.sum(
            final_momentum**2
        )
        recovered_position, recovered_momentum = leapfrog(
            final_position, final_momentum, -0.025, 5
        )
        energy_errors.append(float(final_energy - initial_energy))
        reversibility.append(
            float(
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
            )
        )
    return {
        "step_size": 0.025,
        "steps": 5,
        "energy_error_max_abs": max(map(abs, energy_errors)),
        "energy_error_rms": float(
            np.sqrt(np.mean(np.square(energy_errors)))
        ),
        "reversibility_max_abs": max(reversibility),
    }


def dispatcher_audit(case, scales, log_density):
    offsets = jnp.linspace(-4.0, 4.0, 17)
    methods = []
    gradients = []
    gradient_function = jax.jit(jax.grad(log_density))
    for offset in offsets:
        standardized = jnp.asarray((offset, 0.0))
        parameters = parameters_from_standardized(
            case["truth"], scales, standardized
        )
        source_x, source_y = source_positions(case["times"], parameters)
        result = binary_magnification_trajectory(
            source_x,
            source_y,
            parameters["s"],
            parameters["q"],
            parameters["rho"],
            parameters["limb_darkening_c"],
            0.0,
            cartesian_backend="ffi",
            root_backend="ffi",
            expanded_cartesian_fallback=True,
        )
        methods.append(np.asarray(result.method))
        gradients.append(
            float(gradient_function(standardized)[0])
        )
    methods = np.asarray(methods)
    gradients = np.asarray(gradients)
    method_changes = np.any(methods[1:] != methods[:-1], axis=1)
    return {
        "grid_standardized_t0": np.asarray(offsets).tolist(),
        "method_boundary_intervals": np.flatnonzero(method_changes).tolist(),
        "maximum_adjacent_gradient_jump": float(
            np.max(np.abs(np.diff(gradients)))
        ),
        "gradient": gradients.tolist(),
        "method_counts": {
            str(method): int(np.count_nonzero(methods == method))
            for method in np.unique(methods)
        },
    }


def run_nuts(engine, case, scales, observed, args, seed):
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    from numpyro.infer import MCMC, NUTS, init_to_value

    curve = public_curve(True)

    def model():
        standardized = numpyro.sample(
            "standardized",
            dist.Normal(
                jnp.zeros(len(PARAMETER_NAMES)),
                3.0 * jnp.ones(len(PARAMETER_NAMES)),
            ).to_event(1),
        )
        parameters = parameters_from_standardized(
            case["truth"], scales, standardized
        )
        predicted = (
            curve(case["times"], parameters)
            if engine == "lcbinint"
            else microlux_magnification(case["times"], parameters)
        )
        numpyro.sample(
            "flux",
            dist.Normal(predicted, NOISE_SIGMA),
            obs=observed,
        )

    sampler = MCMC(
        NUTS(
            model,
            target_accept_prob=args.target_accept,
            max_tree_depth=args.max_tree_depth,
            dense_mass=not args.diagonal_mass,
            init_strategy=init_to_value(
                values={
                    "standardized": jnp.zeros(len(PARAMETER_NAMES))
                }
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
        jax.random.PRNGKey(seed),
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
        "samples_per_second": args.chains * args.samples / elapsed,
        "effective_samples": ess.tolist(),
        "minimum_effective_samples_per_second": float(np.min(ess) / elapsed),
        "posterior_standardized_mean": np.mean(
            samples, axis=(0, 1)
        ).tolist(),
        "posterior_standardized_sd": np.std(
            samples, axis=(0, 1), ddof=1
        ).tolist(),
        "divergences": int(np.count_nonzero(extra["diverging"])),
        "mean_accept_probability": float(np.mean(extra["accept_prob"])),
        "median_leapfrog_steps": float(np.median(extra["num_steps"])),
        "maximum_leapfrog_steps": int(np.max(extra["num_steps"])),
        "energy_variance": float(np.var(extra["energy"])),
    }
    if args.chains >= 2:
        result["split_rhat"] = np.asarray(
            split_gelman_rubin(jnp.asarray(samples))
        ).tolist()
    return result


def benchmark_case(name, case, args):
    print(f"[{name}] preparing data and Fisher scales", file=sys.stderr)
    curve = public_curve(True)
    native_curve = public_curve(False)
    truth = case["truth"]
    times = case["times"]
    exact = curve(times, truth)
    block(exact)
    scales, raw_jacobian_norms = fisher_scales(curve, times, truth)
    key = jax.random.PRNGKey(args.seed + (0 if name == "smooth" else 1))
    observed = exact + NOISE_SIGMA * jax.random.normal(key, exact.shape)

    benchmark_engines = (
        ("lcbinint", "microlux")
        if args.benchmark_engine == "both"
        else (args.benchmark_engine,)
    )
    log_densities = {
        engine: make_log_density(
            engine, times, truth, scales, observed
        )
        for engine in benchmark_engines
    }
    zero = jnp.zeros(len(PARAMETER_NAMES))
    timings = {}
    values = {}
    for engine, log_density in log_densities.items():
        print(f"[{name}] timing {engine}", file=sys.stderr)
        values[engine], timings[f"{engine}_log_density"] = timed_jax(
            jax.jit(log_density), zero, args.repeat
        )
        _, timings[f"{engine}_value_and_grad"] = timed_jax(
            jax.jit(jax.value_and_grad(log_density)),
            zero,
            args.repeat,
        )

    native_parameters = parameters_from_standardized(
        truth, scales, np.zeros(len(PARAMETER_NAMES))
    )
    timings["native_lcbinint_forward"] = timed_python(
        lambda: native_curve(np.asarray(times), native_parameters),
        args.repeat,
    )
    timings["vbm_forward"] = timed_python(
        lambda: vbm_magnification(times, native_parameters),
        args.repeat,
    )

    native_values = np.asarray(native_curve(np.asarray(times), truth))
    vbm_values = vbm_magnification(times, truth)
    microlux_values = (
        np.asarray(microlux_magnification(times, truth))
        if "microlux" in benchmark_engines
        else None
    )

    nuts = {}
    if args.nuts_engine == "both":
        requested_engines = ("lcbinint", "microlux")
    elif args.nuts_engine == "none":
        requested_engines = ()
    else:
        requested_engines = (args.nuts_engine,)
    for engine in requested_engines:
        print(f"[{name}] running {engine} NUTS", file=sys.stderr)
        nuts[engine] = run_nuts(
            engine,
            case,
            scales,
            observed,
            args,
            args.seed + 10 + (0 if engine == "lcbinint" else 100),
        )

    denominator = np.maximum(np.abs(vbm_values), 1.0)
    return {
        "truth": truth,
        "parameter_names": PARAMETER_NAMES,
        "fisher_scales": np.asarray(scales).tolist(),
        "raw_curve_jacobian_norms": raw_jacobian_norms.tolist(),
        "epochs": len(times),
        "magnification_range": [
            float(jnp.min(exact)),
            float(jnp.max(exact)),
        ],
        "accuracy_against_vbm": {
            "lcbinint_jax_max_relative": float(
                np.max(np.abs(np.asarray(exact) - vbm_values) / denominator)
            ),
            "lcbinint_native_max_relative": float(
                np.max(np.abs(native_values - vbm_values) / denominator)
            ),
            "microlux_max_relative": (
                float(
                    np.max(
                        np.abs(microlux_values - vbm_values) / denominator
                    )
                )
                if microlux_values is not None
                else None
            ),
        },
        "log_density_at_truth": {
            engine: float(value) for engine, value in values.items()
        },
        "timings": timings,
        "leapfrog": {
            engine: leapfrog_audit(
                log_density,
                args.seed + (0 if engine == "lcbinint" else 1),
            )
            for engine, log_density in log_densities.items()
            if args.leapfrog_engine in ("both", engine)
        },
        "dispatcher": (
            dispatcher_audit(
                case, scales, log_densities["lcbinint"]
            )
            if "lcbinint" in log_densities
            else None
        ),
        "nuts": nuts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("smooth", "caustic", "both"), default="both"
    )
    parser.add_argument(
        "--nuts-engine",
        choices=("lcbinint", "microlux", "both", "none"),
        default="lcbinint",
    )
    parser.add_argument(
        "--leapfrog-engine",
        choices=("lcbinint", "microlux", "both", "none"),
        default="lcbinint",
        help="microlux leapfrog audits are intentionally opt-in on CPU",
    )
    parser.add_argument(
        "--benchmark-engine",
        choices=("lcbinint", "microlux", "both"),
        default="both",
        help="engines used for log-density timing and leapfrog audits",
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument(
        "--diagonal-mass",
        action="store_true",
        help="disable the default dense two-parameter mass matrix",
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected = cases(args.epochs)
    if args.case != "both":
        selected = {args.case: selected[args.case]}
    output = {
        "configuration": vars(args)
        | {
            "platform": platform.platform(),
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "noise_sigma": NOISE_SIGMA,
        },
        "cases": {
            name: benchmark_case(name, case, args)
            for name, case in selected.items()
        },
    }
    text = json.dumps(output, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
