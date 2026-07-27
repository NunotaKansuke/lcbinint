#!/usr/bin/env python3
"""Render plain Matplotlib binary/triple lens curves, gradients, and geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import lcbinint
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

jax.config.update("jax_enable_x64", True)

from lcbinint_jax import (  # noqa: E402
    binary_magnification_trajectory,
    triple_magnification_batch,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets"
BINARY_TIME_SAMPLES = 2_000
# The documentation example uses 300 points; retain a similarly compact
# triple-lens gradient calculation because its small tertiary caustics are
# substantially more expensive to differentiate.
TRIPLE_TIME_SAMPLES = 360

BINARY_PARAMETERS = jnp.asarray(
    # t0, u0, tE, alpha, rho, s, q, Gamma
    (0.0, 0.25, 30.0, 0.7, 0.01, 1.2, 0.03, 0.45)
)
TRIPLE_PARAMETERS = jnp.asarray(
    # t0, u0, tE, alpha, rho, s, q, s13, q3, psi, Gamma
    (0.0, 0.0, 30.0, 1.0, 0.01, 0.9, 0.1, 1.5, 0.003, 1.0, 0.45)
)

BINARY_LABELS = (
    r"$\partial A/\partial t_0$",
    r"$\partial A/\partial u_0$",
    r"$\partial A/\partial t_{\rm E}$",
    r"$\partial A/\partial \alpha$",
    r"$\partial A/\partial \rho$",
    r"$\partial A/\partial s$",
    r"$\partial A/\partial q$",
    r"$\partial A/\partial \Gamma$",
)
TRIPLE_LABELS = (
    *BINARY_LABELS[:7],
    r"$\partial A/\partial s_{13}$",
    r"$\partial A/\partial q_3$",
    r"$\partial A/\partial \psi$",
    BINARY_LABELS[7],
)


def source_trajectory(
    time: jax.Array, parameters: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return the native center-of-mass trajectory convention."""

    t0, u0, timescale, alpha = parameters[:4]
    tau = (time - t0) / timescale
    source_x = u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
    source_y = u0 * jnp.cos(alpha) - tau * jnp.sin(alpha)
    return source_x, source_y


def binary_curve(time: jax.Array, parameters: jax.Array) -> jax.Array:
    source_x, source_y = source_trajectory(time, parameters)
    return binary_magnification_trajectory(
        source_x,
        source_y,
        parameters[5],
        parameters[6],
        parameters[4],
        parameters[7],
        expanded_cartesian_fallback=True,
    ).magnification


def triple_curve(time: jax.Array, parameters: jax.Array) -> jax.Array:
    source_x, source_y = source_trajectory(time, parameters)
    return triple_magnification_batch(
        source_x,
        source_y,
        parameters[5],
        parameters[6],
        parameters[8],
        parameters[7],
        parameters[9],
        parameters[4],
        parameters[10],
        # Match the explicitly center-of-mass caustic/trajectory plot.
        convention="center_of_mass",
    ).magnification


def plot_curve_and_gradients(
    time: np.ndarray,
    curve: np.ndarray,
    derivatives: np.ndarray,
    labels: tuple[str, ...],
    title: str,
    parameter_text: str,
    output: Path,
) -> None:
    # Keep every gradient panel as wide as the light curve, so their time axes
    # line up exactly with the main panel.
    derivative_rows = len(labels)
    figure = plt.figure(
        figsize=(12.0, 3.2 + 1.45 * derivative_rows),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        derivative_rows + 1,
        1,
        height_ratios=(2.15, *([1.0] * derivative_rows)),
    )

    light_curve_axis = figure.add_subplot(grid[0])
    light_curve_axis.plot(time, curve, color="tab:blue", lw=1.6)
    light_curve_axis.set_ylabel("Magnification")
    light_curve_axis.set_title(title)
    light_curve_axis.text(
        0.99,
        0.95,
        parameter_text,
        transform=light_curve_axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    light_curve_axis.grid(alpha=0.25)
    light_curve_axis.tick_params(labelbottom=False)

    derivative_axes: list[plt.Axes] = []
    for index, label in enumerate(labels):
        axis = figure.add_subplot(grid[index + 1], sharex=light_curve_axis)
        axis.plot(time, derivatives[:, index], color="tab:orange", lw=1.2)
        axis.axhline(0.0, color="0.5", lw=0.7)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        derivative_axes.append(axis)

    for axis in derivative_axes:
        row = axis.get_subplotspec().rowspan.start
        if row == derivative_rows:
            axis.set_xlabel(r"$t-t_0$ [days]")
        else:
            axis.tick_params(labelbottom=False)

    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def native_parameters(lens: str) -> dict[str, float]:
    values = {
        "t0": 0.0,
        "u0": 0.25,
        "tE": 30.0,
        "alpha": 0.7,
        "rho": 0.01,
        "s": 1.2,
        "q": 0.03,
    }
    if lens == "triple":
        values.update(
            {
                "u0": 0.0,
                "tE": 30.0,
                "alpha": 1.0,
                "rho": 0.01,
                "sep2": 1.5,
                "q2": 0.003,
                "ang": 1.0,
            }
        )
    return values


def plot_caustics_and_trajectory(
    time: np.ndarray,
    lens: str,
    output: Path,
) -> None:
    parameters = native_parameters(lens)
    solver = lcbinint.LightCurve(
        lens=lens,
        options=lcbinint.Options(
            coordinates="center_of_mass",
            caustic_bins=1200,
        ),
    )
    caustics = solver.caustics(parameters)
    trajectory = solver.source_trajectory(time, parameters)
    trajectory_x = np.asarray(trajectory.x)
    trajectory_y = np.asarray(trajectory.y)
    jax_x, jax_y = jax.device_get(
        source_trajectory(
            jnp.asarray(time),
            jnp.asarray(
                (
                    parameters["t0"],
                    parameters["u0"],
                    parameters["tE"],
                    parameters["alpha"],
                )
            ),
        )
    )
    np.testing.assert_allclose(jax_x, trajectory_x, atol=2.0e-15)
    np.testing.assert_allclose(jax_y, trajectory_y, atol=2.0e-15)

    figure, axis = plt.subplots(figsize=(6.2, 6.0), constrained_layout=True)
    for index, (caustic_x, caustic_y) in enumerate(
        zip(caustics.x, caustics.y)
    ):
        axis.plot(
            caustic_x,
            caustic_y,
            color="tab:red",
            lw=1.3,
            label="Caustic" if index == 0 else None,
        )
    axis.plot(
        trajectory_x,
        trajectory_y,
        color="tab:blue",
        lw=1.4,
        label="Source trajectory",
    )

    center = int(np.argmin(np.abs(time)))
    axis.add_patch(
        Circle(
            (trajectory_x[center], trajectory_y[center]),
            radius=parameters["rho"],
            fill=False,
            edgecolor="tab:blue",
            lw=1.2,
            label=r"Source at $t_0$",
        )
    )
    arrow = min(center + 20, len(time) - 1)
    axis.annotate(
        "",
        xy=(trajectory_x[arrow], trajectory_y[arrow]),
        xytext=(trajectory_x[center], trajectory_y[center]),
        arrowprops={"arrowstyle": "->", "color": "tab:blue", "lw": 1.4},
    )

    axis.set_xlabel(r"$x/\theta_{\rm E}$")
    axis.set_ylabel(r"$y/\theta_{\rm E}$")
    axis.set_title(f"{lens.capitalize()} lens caustics and source trajectory")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def render(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = (
        (
            "binary",
            binary_curve,
            BINARY_PARAMETERS,
            BINARY_LABELS,
            BINARY_TIME_SAMPLES,
            (
                r"$s=1.2,\ q=0.03,\ u_0=0.25,\ \alpha=0.7,\ "
                r"\rho=0.01,\ t_{\rm E}=30\,{\rm d},\ \Gamma=0.45$"
            ),
        ),
        (
            "triple",
            triple_curve,
            TRIPLE_PARAMETERS,
            TRIPLE_LABELS,
            TRIPLE_TIME_SAMPLES,
            (
                r"$s=0.9,\ q=0.1,\ s_{13}=1.5,\ q_3=0.003,\ \psi=1.0,\ "
                r"u_0=0,\ \alpha=1.0,\ \rho=0.01,\ "
                r"t_{\rm E}=30\,{\rm d},\ \Gamma=0.45$"
            ),
        ),
    )
    outputs: list[Path] = []
    for lens, function, parameters, labels, time_samples, parameter_text in jobs:
        time = jnp.linspace(
            parameters[0] - parameters[2],
            parameters[0] + parameters[2],
            time_samples,
        )
        curve = function(time, parameters)
        derivatives = jax.jacfwd(function, argnums=1)(time, parameters)
        curve, derivatives = jax.device_get((curve, derivatives))
        if not np.all(np.isfinite(curve)):
            raise RuntimeError(f"{lens} curve contains non-finite values")
        if not np.all(np.isfinite(derivatives)):
            raise RuntimeError(f"{lens} derivatives contain non-finite values")

        gradient_output = output_dir / f"{lens}_lightcurve_gradients.png"
        geometry_output = output_dir / f"{lens}_caustics_trajectory.png"
        plot_curve_and_gradients(
            np.asarray(time),
            np.asarray(curve),
            np.asarray(derivatives),
            labels,
            f"{lens.capitalize()} lens: finite-source light curve and derivatives",
            parameter_text,
            gradient_output,
        )
        plot_caustics_and_trajectory(
            np.asarray(time),
            lens,
            geometry_output,
        )
        outputs.extend(
            (
                gradient_output,
                gradient_output.with_suffix(".pdf"),
                geometry_output,
                geometry_output.with_suffix(".pdf"),
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    for output in render(arguments.output_dir):
        print(output)


if __name__ == "__main__":
    main()
