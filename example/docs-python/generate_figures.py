"""Generate the figures embedded by docs/python.

Run from the repository root with ``PYTHONPATH=build python
example/docs-python/generate_figures.py``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

import lcbinint


OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "python" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(OUTPUT / name, dpi=150)
    plt.close()


def plot_branches(branches, *args, **kwargs):
    for x, y in zip(branches.x, branches.y):
        plt.plot(-np.asarray(x), -np.asarray(y), *args, **kwargs)


def standard_binary():
    params = dict(s=0.9, q=0.1, u0=0.0, alpha=1.0, rho=0.01, tE=30.0, t0=7500)
    times = np.linspace(7470, 7530, 300)
    curve = lcbinint.LightCurve(options=lcbinint.Options(tol=1e-3, reltol=1e-3))
    magnifications = curve(times, params)
    trajectory = curve.source_trajectory(times, params)

    plt.figure(figsize=(6, 4))
    plt.plot(times, magnifications)
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("BinaryLens_lightcurve.png")

    plt.figure(figsize=(5, 5))
    plot_branches(curve.caustics(params))
    plt.plot(-np.asarray(trajectory.x), -np.asarray(trajectory.y))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    save("BinaryLens_lightcurve_caustics.png")


def standard_triple():
    params = dict(
        s=0.9, q=0.1, u0=0.0, alpha=1.0, rho=0.01, tE=30.0, t0=7500,
        sep2=1.5, q2=0.003, ang=1.0,
    )
    times = np.linspace(7470, 7530, 300)
    curve = lcbinint.LightCurve(
        lens="triple", options=lcbinint.Options(tol=1e-3, reltol=1e-3)
    )
    magnifications = curve(times, params)
    trajectory = curve.source_trajectory(times, params)

    plt.figure(figsize=(6, 4))
    plt.plot(times, magnifications)
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("TripleLens_lightcurve.png")

    plt.figure(figsize=(5, 5))
    plot_branches(curve.caustics(params), "r")
    plt.plot(-np.asarray(trajectory.x), -np.asarray(trajectory.y))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    save("TripleLens_lightcurve_caustics.png")


def critical_curves_and_caustics():
    params = dict(s=0.6, q=0.1)
    curve = lcbinint.LightCurve(options=lcbinint.Options(caustic_bins=200))

    plt.figure(figsize=(5, 5))
    plot_branches(curve.caustics(params), "k")
    plt.axis("equal")
    save("Caustics_binary.png")

    plt.figure(figsize=(5, 5))
    plot_branches(curve.critical_curves(params), "k")
    plt.axis("equal")
    save("Criticalcurves_binary.png")


def parallax():
    params = dict(
        s=0.9, q=0.1, u0=0.0, alpha=1.0, rho=0.01, tE=30.0, t0=7500,
        piEN=0.3, piEE=-0.2,
    )
    times = np.linspace(7470, 7530, 300)
    options = lcbinint.Options(tol=1e-3, reltol=1e-3)
    static = lcbinint.LightCurve(options=options)
    moved = lcbinint.LightCurve(
        model=lcbinint.Model(
            parallax=True,
            sky=lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2"),
            t_ref=7500,
        ),
        options=options,
    )
    static_mag = static(times, params)
    moved_mag = moved(times, params)
    static_trajectory = static.source_trajectory(times, params)
    moved_trajectory = moved.source_trajectory(times, params)

    plt.figure(figsize=(6, 4))
    plt.plot(times, static_mag, "g")
    plt.plot(times, moved_mag, "m")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("BinaryLens_lightcurve_parallax.png")

    plt.figure(figsize=(5, 5))
    plot_branches(static.caustics(params))
    plt.plot(-np.asarray(static_trajectory.x), -np.asarray(static_trajectory.y), "g")
    plt.plot(-np.asarray(moved_trajectory.x), -np.asarray(moved_trajectory.y), "m")
    plt.axis("equal")
    save("BinaryLens_lightcurve_parallax_caustics.png")


def orbital_motion():
    params = dict(
        s=0.9, q=0.1, u0=0.0, alpha=1.0, rho=0.01, tE=30.0, t0=7500,
        piEN=0.3, piEE=-0.2, g1=0.011, g2=-0.005, g3=0.005,
    )
    times = np.linspace(7470, 7530, 300)
    sky = lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2")
    options = lcbinint.Options(tol=1e-3, reltol=1e-3)
    static = lcbinint.LightCurve(options=options)
    moved = lcbinint.LightCurve(
        model=lcbinint.Model(parallax=True, sky=sky, t_ref=7500), options=options
    )
    orbital = lcbinint.LightCurve(
        model=lcbinint.Model(
            parallax=True, orbital_motion="circular", sky=sky, t_ref=7500
        ),
        options=options,
    )
    trajectory = orbital.source_trajectory(times, params)

    plt.figure(figsize=(6, 4))
    plt.plot(times, static(times, params), "g")
    plt.plot(times, moved(times, params), "m")
    plt.plot(times, orbital(times, params), "y")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("BinaryLens_lightcurve_orbital.png")

    indices = [100, 150, 200]
    colors = [(0, 0, 1, 1), (0.4, 0, 0.6, 1), (0.6, 0, 0.4, 1)]
    plt.figure(figsize=(5, 5))
    for index, color in zip(indices, colors):
        plot_branches(orbital.caustics(float(times[index]), params), color=color)
    source_x = -np.asarray(trajectory.x)
    source_y = -np.asarray(trajectory.y)
    plt.plot(source_x, source_y, "y")
    for index, color in zip(indices, colors):
        plt.plot([source_x[index]], [source_y[index]], color=color, marker="o")
    plt.axis("equal")
    save("BinaryLens_lightcurve_orbital_caustics.png")


def binary_source():
    params = dict(
        s=1.0, q=1.0, alpha=0.0, tE=37.3, q_source=0.4,
        u0=0.1, u0_2=0.05, t0=7550.4, t0_2=7555.8, rho=0.004,
        piEN=0.03, piEE=-0.02, w1=0.021, w2=-0.02, w3=0.03,
    )
    times = np.linspace(7550.4 - 37.3, 7550.4 + 37.3, 300)
    options = lcbinint.Options(tol=1e-3, reltol=1e-3)
    static = lcbinint.LightCurve(
        model=lcbinint.Model(lens="binary", source="binary"), options=options
    )
    xallarap = lcbinint.LightCurve(
        model=lcbinint.Model(
            lens="binary", source="binary", xallarap="circular_velocity"
        ),
        options=options,
    )

    plt.figure(figsize=(6, 4))
    plt.plot(times, static(times, params))
    plt.plot(times, xallarap(times, params), "y")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("BinarySource_lightcurve_xallarap_2.png")


def main():
    standard_binary()
    standard_triple()
    critical_curves_and_caustics()
    parallax()
    orbital_motion()
    binary_source()


if __name__ == "__main__":
    main()
