"""Shared plotting helpers for the notebook-style tutorial scripts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def output_path(name):
    root = Path(__file__).resolve().parents[2]
    return root / "docs" / "assets" / "tutorials" / f"{name}.png"


def _plot_branches(ax, branches, *, label="caustic"):
    for index, (x, y) in enumerate(zip(branches.x, branches.y)):
        ax.plot(x, y, color="#d1495b", lw=1.25, label=label if index == 0 else None)


def render_tutorial(
    name, title, curve, times, params, *, comparisons=(), caustic_time=None,
    caustic_epochs=None, geometry_limits=None,
):
    """Save a light-curve and source-trajectory/caustic figure.

    ``comparisons`` is an iterable of ``(label, curve)`` pairs plotted before
    the primary curve. All curves use the same epochs and parameter mapping.
    """
    times = np.asarray(times, dtype=float)
    magnification = curve(times, params)
    geometry_params = dict(params)
    for key in ("q_source", "fluxratio", "q_mass", "t0_2", "u0_2"):
        geometry_params.pop(key, None)
    trajectory = curve.source_trajectory(times, geometry_params)
    light_curve_figure, light_curve_ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)

    for label, comparison in comparisons:
        comparison_params = dict(geometry_params)
        if comparison.model.xallarap == "none":
            for key in (
                "xi_1", "xi_2", "period_xa", "ecc_xa", "peri_xa", "inc_xa",
                "w1", "w2", "w3", "omega_xa", "phi_xa", "xa_szs", "xa_ar",
            ):
                comparison_params.pop(key, None)
        light_curve_ax.plot(
            times, comparison(times, comparison_params), "--", lw=1.25, alpha=0.85, label=label
        )
    light_curve_ax.plot(times, magnification, color="#1f77b4", lw=1.8, label=title)
    light_curve_ax.set(xlabel="time", ylabel="magnification", title="Light curve")
    light_curve_ax.legend(frameon=False, fontsize=8)

    geometry_figure, geometry_ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)

    if caustic_epochs is not None:
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(caustic_epochs)))
        for epoch, color in zip(caustic_epochs, colors):
            branches = curve.caustics(float(epoch), geometry_params)
            for index, (x, y) in enumerate(zip(branches.x, branches.y)):
                geometry_ax.plot(x, y, color=color, lw=1.2,
                                 label=f"t = {epoch:g}" if index == 0 else None)
            closest = int(np.argmin(np.abs(times - epoch)))
            geometry_ax.scatter(trajectory.x[closest], trajectory.y[closest], color=color,
                                s=24, zorder=4)
    elif caustic_time is None:
        caustics = curve.caustics(geometry_params)
        _plot_branches(geometry_ax, caustics)
    else:
        caustics = curve.caustics(caustic_time, geometry_params)
        _plot_branches(geometry_ax, caustics)
    geometry_ax.plot(trajectory.x, trajectory.y, color="#1f77b4", lw=1.5)
    geometry_ax.scatter(trajectory.x[0], trajectory.y[0], color="#1f77b4", s=18, zorder=3)
    geometry_ax.scatter(trajectory.x[-1], trajectory.y[-1], color="#1f77b4", marker="x", s=28, zorder=3)
    geometry_ax.set(
        xlabel="lens-frame x", ylabel="lens-frame y", title="Source trajectory and caustics"
    )
    geometry_ax.set_aspect("equal", adjustable="box")
    if geometry_limits is not None:
        x_min, x_max, y_min, y_max = geometry_limits
        geometry_ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
    if caustic_epochs is not None:
        geometry_ax.legend(title="caustic epoch", frameon=False, fontsize=8, loc="best")

    light_curve_target = output_path(f"{name}-light-curve")
    geometry_target = output_path(f"{name}-geometry")
    light_curve_target.parent.mkdir(parents=True, exist_ok=True)
    light_curve_figure.savefig(light_curve_target, dpi=170)
    geometry_figure.savefig(geometry_target, dpi=170)
    plt.close(light_curve_figure)
    plt.close(geometry_figure)
    print(f"{name}: min={np.min(magnification):.6g} max={np.max(magnification):.6g}")
    print(f"saved {light_curve_target.relative_to(Path.cwd())}")
    print(f"saved {geometry_target.relative_to(Path.cwd())}")
