"""Generate the figures embedded by docs/python.

Run from the repository root with ``PYTHONPATH=build python
example/docs-python/generate_figures.py``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import lcbinint


OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "python" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(OUTPUT / name, dpi=220)
    plt.close()


def plot_branches(branches, *args, **kwargs):
    for x, y in zip(branches.x, branches.y):
        plt.plot(-np.asarray(x), -np.asarray(y), *args, **kwargs)


def plot_caustics(branches, *args, **kwargs):
    """Draw the ordered physical caustics as continuous polylines."""
    plot_branches(branches, *args, **kwargs)


def standard_binary():
    params = dict(s=0.9, q=0.1, u0=0.0, alpha=1.0, rho=0.01, tE=30.0, t0=7500)
    times = np.linspace(7470, 7530, 300)
    curve = lcbinint.LightCurve(options=lcbinint.Options(tol=1e-3, reltol=1e-3))
    magnifications = curve(times, params)
    trajectory = curve.source_trajectory(times, params)

    plt.figure(figsize=(3.8, 2.55))
    plt.plot(times, magnifications)
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("BinaryLens_lightcurve.png")

    plt.figure(figsize=(2.8, 2.8))
    plot_caustics(curve.caustics(params), color="tab:red", lw=1.1)
    plt.plot(
        -np.asarray(trajectory.x), -np.asarray(trajectory.y), color="tab:blue"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    save("BinaryLens_lightcurve_caustics.png")


def binary_lens_images():
    q, s, y1, y2, rho = 0.1, 0.8, 0.01, 0.01, 0.01
    image_plane = lcbinint.image.ImagePlane(
        q=q, s=s, x=y1, y=y2, rho=rho, coordinates="lcbinint"
    )
    caustics = image_plane.caustics()
    critical_curves = image_plane.critical_curves()
    image_regions = image_plane.ray_shooting_images(resolution=300)

    fig, (source_ax, image_ax) = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for x, y in zip(caustics.x, caustics.y):
        source_ax.plot(x, y, color="tab:red", lw=1.1)
    source_ax.scatter([y1], [y2], marker="*", color="tab:blue")
    source_ax.add_patch(Circle((y1, y2), rho, fill=False, color="tab:blue"))
    source_ax.set(title="Source plane", xlabel="x", ylabel="y", aspect="equal")

    for x, y in zip(critical_curves.x, critical_curves.y):
        image_ax.plot(x, y, color="tab:blue")
    for region in image_regions:
        if len(region.points):
            image_ax.scatter(region.points[:, 0], region.points[:, 1], s=2)
    image_ax.set(title="Image plane", xlabel="x", ylabel="y", aspect="equal")
    fig.tight_layout()
    plt.savefig(OUTPUT / "BinaryLens_images.png", dpi=220)
    plt.close()


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

    plt.figure(figsize=(3.8, 2.55))
    plt.plot(times, magnifications)
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("TripleLens_lightcurve.png")

    plt.figure(figsize=(2.8, 2.8))
    plot_caustics(curve.caustics(params), color="tab:red", lw=1.1)
    plt.plot(
        -np.asarray(trajectory.x), -np.asarray(trajectory.y), color="tab:blue"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    save("TripleLens_lightcurve_caustics.png")


def critical_curves_and_caustics():
    params = dict(s=0.6, q=0.1)
    curve = lcbinint.LightCurve(options=lcbinint.Options(caustic_bins=200))

    plt.figure(figsize=(2.8, 2.8))
    plot_caustics(curve.caustics(params), color="tab:red", lw=1.1)
    plt.axis("equal")
    save("Caustics_binary.png")

    plt.figure(figsize=(2.8, 2.8))
    plot_branches(curve.critical_curves(params), color="tab:blue")
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

    plt.figure(figsize=(3.8, 2.55))
    plt.plot(times, static_mag, "g")
    plt.plot(times, moved_mag, "m")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("BinaryLens_lightcurve_parallax.png")

    plt.figure(figsize=(2.8, 2.8))
    plot_caustics(static.caustics(params), color="tab:red", lw=1.1)
    plt.plot(
        -np.asarray(static_trajectory.x), -np.asarray(static_trajectory.y),
        color="tab:blue", linestyle="--",
    )
    plt.plot(
        -np.asarray(moved_trajectory.x), -np.asarray(moved_trajectory.y),
        color="tab:blue",
    )
    plt.axis("equal")
    save("BinaryLens_lightcurve_parallax_caustics.png")

    satellite_phase = np.linspace(-1.0, 1.0, len(times))
    satellite_table = np.column_stack([
        2450000.0 + times,
        270.0 + 12.0 * satellite_phase,
        -20.0 + 4.0 * np.sin(np.pi * satellite_phase),
        0.55 + 0.05 * satellite_phase,
    ])
    satellite_model = lcbinint.Model(
        parallax=True,
        terrestrial=True,
        sky=lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2"),
        t_ref=7500,
    )
    ground = lcbinint.LightCurve(
        model=satellite_model,
        site=lcbinint.obs.Site("ground", -29.0, -70.7),
        options=options,
    )
    space = lcbinint.LightCurve(
        model=satellite_model,
        site=lcbinint.obs.Site("space", satellite_table),
        options=options,
    )
    ground_mag = ground(times, params)
    space_mag = space(times, params)
    fig, (curve_ax, difference_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(4.8, 3.9),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    curve_ax.plot(times, ground_mag, label="ground: Chile")
    curve_ax.plot(times, space_mag, label="spacecraft")
    curve_ax.set_ylabel("Magnification")
    curve_ax.legend()
    difference_ax.plot(times, space_mag - ground_mag)
    difference_ax.axhline(0.0, color="0.6", linewidth=1)
    difference_ax.set(xlabel="Time", ylabel="space - ground")
    fig.tight_layout()
    plt.savefig(OUTPUT / "SatelliteParallax_comparison.png", dpi=220)
    plt.close()

    terrestrial_model = lcbinint.Model(
        parallax=True,
        terrestrial=True,
        sky=lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2"),
        t_ref=7500,
    )
    africa = lcbinint.LightCurve(
        model=terrestrial_model,
        site=lcbinint.obs.Site("ground", -29.0, 20.0),
        options=options,
    )
    chile = lcbinint.LightCurve(
        model=terrestrial_model,
        site=lcbinint.obs.Site("ground", -29.0, -70.7),
        options=options,
    )
    africa_mag = africa(times, params)
    chile_mag = chile(times, params)
    fig, (curve_ax, difference_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(4.8, 3.9),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    curve_ax.plot(times, africa_mag, label="Africa: 29 S, 20 E")
    curve_ax.plot(times, chile_mag, label="Chile: 29 S, 70.7 W")
    curve_ax.set_ylabel("Magnification")
    curve_ax.legend()
    difference_ax.plot(times, 1e3 * (africa_mag - chile_mag))
    difference_ax.axhline(0.0, color="0.6", linewidth=1)
    difference_ax.set(
        xlabel="Time", ylabel=r"$10^3\,(A_{Africa}-A_{Chile})$"
    )
    fig.tight_layout()
    plt.savefig(OUTPUT / "TerrestrialParallax_comparison.png", dpi=220)
    plt.close()


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

    plt.figure(figsize=(3.8, 2.55))
    plt.plot(times, static(times, params), "g")
    plt.plot(times, moved(times, params), "m")
    plt.plot(times, orbital(times, params), "y")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("BinaryLens_lightcurve_orbital.png")

    indices = [100, 150, 200]
    colors = [(0, 0, 1, 1), (0.4, 0, 0.6, 1), (0.6, 0, 0.4, 1)]
    plt.figure(figsize=(2.8, 2.8))
    for index, color in zip(indices, colors):
        plot_caustics(orbital.caustics(float(times[index]), params), color=color, lw=1.1)
    source_x = -np.asarray(trajectory.x)
    source_y = -np.asarray(trajectory.y)
    plt.plot(source_x, source_y, "y")
    for index, color in zip(indices, colors):
        plt.plot([source_x[index]], [source_y[index]], color=color, marker="o")
    plt.axis("equal")
    save("BinaryLens_lightcurve_orbital_caustics.png")


def binary_source():
    params = dict(
        s=1.0, q=1.0, alpha=0.0, tE=37.3, t0=7550.4,
        u0=0.075, rho=0.004, q_source=0.4, q_mass=1.0,
        xi_1=0.04, xi_2=-0.025, w1=0.021, w2=-0.02, w3=0.03,
    )
    times = np.linspace(7550.4 - 37.3, 7550.4 + 37.3, 300)
    options = lcbinint.Options(tol=1e-3, reltol=1e-3)
    xallarap = lcbinint.LightCurve(
        model=lcbinint.Model(
            lens="binary", source="binary", xallarap="circular_velocity"
        ),
        options=options,
    )

    plt.figure(figsize=(4.8, 3.0))
    static_params = dict(params, w1=0.0, w2=0.0, w3=0.0)
    plt.plot(times, xallarap(times, static_params))
    plt.plot(times, xallarap(times, params), "y")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("BinarySource_lightcurve_xallarap_2.png")


def binary_source_binary_lens():
    params = dict(
        s=0.9, q=0.1, u0=0.1, alpha=1.0, rho=0.01, tE=30.0, t0=7500,
        piEN=0.3, piEE=-0.2, g1=0.011, g2=-0.005, g3=0.005,
        q_source=1.0, q_mass=1.0, xi_1=0.0, xi_2=-0.1,
        w1=0.01, w2=0.02, w3=0.015,
    )
    times = np.linspace(7470, 7530, 300)
    sky = lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2")
    options = lcbinint.Options(
        coordinates="vbm", tol=1e-3, reltol=1e-3
    )
    single_source = lcbinint.LightCurve(
        model=lcbinint.Model(
            parallax=True, orbital_motion="circular", sky=sky, t_ref=7500
        ),
        options=options,
    )
    binary_source_curve = lcbinint.LightCurve(
        model=lcbinint.Model(
            lens="binary", source="binary", parallax=True,
            orbital_motion="circular", xallarap="circular_velocity",
            sky=sky, t_ref=7500,
        ),
        options=options,
    )
    binary_source_magnifications = binary_source_curve(times, params)
    single_source_params = {
        key: value for key, value in params.items()
        if key not in {"q_source", "q_mass", "xi_1", "xi_2", "w1", "w2", "w3"}
    }
    plt.figure(figsize=(6.6, 3.7))
    plt.plot(
        times, single_source(times, single_source_params), "y",
        label="single source + lens orbit",
    )
    plt.plot(
        times, binary_source_magnifications, "g",
        label="binary source + xallarap",
    )
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    save("BinarySourceBinaryLens_lightcurve.png")


def limb_darkening():
    s, q, y2, rho = 0.8, 0.1, 0.01, 0.01
    x = np.linspace(-0.04, 0.04, 161)
    uniform = np.array([
        lcbinint.binary_ray_shooting(xi, y2, s=s, q=q, rho=rho) for xi in x
    ])
    linear = np.array([
        lcbinint.binary_ray_shooting(
            xi, y2, s=s, q=q, rho=rho,
            limb_darkening=lcbinint.LimbDarkening.linear(0.51),
        )
        for xi in x
    ])
    square_root = np.array([
        lcbinint.binary_ray_shooting(
            xi, y2, s=s, q=q, rho=rho,
            limb_darkening=lcbinint.LimbDarkening.square_root(0.51, 0.3),
        )
        for xi in x
    ])
    plt.figure(figsize=(3.8, 2.55))
    plt.plot(x, uniform, label="uniform")
    plt.plot(x, linear, label="linear: 0.51")
    plt.plot(x, square_root, label="square root: 0.51, 0.3")
    plt.xlabel("Source x")
    plt.ylabel("Magnification")
    plt.legend()
    save("LimbDarkening_comparison.png")


def accuracy_method_selection():
    params = dict(s=0.9, q=0.1, u0=0.0, alpha=1.0, rho=0.01, tE=30.0, t0=7500)
    times = np.linspace(7470, 7530, 300)
    curve = lcbinint.LightCurve(
        options=lcbinint.Options(
            nbin="auto", inverse_ray_grid="auto", tol=1e-4, reltol=1e-3
        )
    )
    info = curve.info(times, params)
    magnifications = np.asarray(info.magnifications)
    methods = np.asarray(info.finite_source_method_names)
    names = list(dict.fromkeys(methods.tolist()))

    fig, (light_ax, method_ax) = plt.subplots(
        2, 1, figsize=(4.8, 3.8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    light_ax.plot(times, magnifications)
    light_ax.set_ylabel("Magnification")
    for level, name in enumerate(names):
        selected = methods == name
        method_ax.scatter(times[selected], np.full(selected.sum(), level), s=9)
    method_ax.set_yticks(range(len(names)), names)
    method_ax.set(xlabel="Time", ylabel="Method")
    fig.tight_layout()
    plt.savefig(OUTPUT / "Accuracy_method_selection.png", dpi=220)
    plt.close()


def coordinates():
    params = dict(s=0.9, q=0.1, u0=0.0, alpha=1.0, rho=0.01, tE=30.0, t0=7500)
    times = np.linspace(7470, 7530, 300)
    curve = lcbinint.LightCurve(
        options=lcbinint.Options(coordinates="vbm", caustic_bins=600)
    )
    trajectory = curve.source_trajectory(times, params)
    plt.figure(figsize=(2.8, 2.8))
    plot_caustics(curve.caustics(params), color="tab:red", lw=1.1)
    display_x = -np.asarray(trajectory.x)
    display_y = -np.asarray(trajectory.y)
    plt.plot(display_x, display_y, color="tab:blue")
    plt.scatter(display_x[[0, -1]], display_y[[0, -1]], color="tab:blue")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.axis("equal")
    save("Coordinates_binary.png")


def combined_effects():
    params = dict(
        s=0.9, q=0.1, u0=0.1, alpha=1.0, rho=0.01, tE=30.0, t0=7500,
        piEN=0.3, piEE=-0.2, g1=0.011, g2=-0.005, g3=0.005,
        q_source=1.0, q_mass=1.0, xi_1=0.0, xi_2=-0.1,
        w1=0.01, w2=0.02, w3=0.015,
    )
    times = np.linspace(7470, 7530, 300)
    sky = lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2")
    options = lcbinint.Options(
        coordinates="vbm", nbin="auto", tol=1e-4, reltol=1e-3
    )
    static = lcbinint.LightCurve(options=options)
    parallax_curve = lcbinint.LightCurve(
        model=lcbinint.Model(parallax=True, sky=sky, t_ref=7500),
        options=options,
    )
    parallax_orbit = lcbinint.LightCurve(
        model=lcbinint.Model(
            parallax=True, orbital_motion="circular", sky=sky, t_ref=7500
        ),
        options=options,
    )
    combined = lcbinint.LightCurve(
        model=lcbinint.Model(
            lens="binary", source="binary", parallax=True,
            orbital_motion="circular", xallarap="circular_velocity",
            sky=sky, t_ref=7500,
        ),
        options=options,
    )
    single_source_params = {
        key: value for key, value in params.items()
        if key not in {"q_source", "q_mass", "xi_1", "xi_2", "w1", "w2", "w3"}
    }
    plt.figure(figsize=(7.4, 3.8))
    plt.plot(times, static(times, single_source_params), label="static 2L1S")
    plt.plot(times, parallax_curve(times, single_source_params), label="+ parallax")
    plt.plot(times, parallax_orbit(times, single_source_params), label="+ lens orbit")
    plt.plot(times, combined(times, params), label="+ binary source + xallarap")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    save("CombinedEffects_lightcurve.png")

    trajectory = combined.source_trajectory(times, params)
    display_x = -np.asarray(trajectory.x)
    display_y = -np.asarray(trajectory.y)
    indices = [75, 150, 225]
    colors = ["tab:blue", "tab:purple", "tab:red"]
    plt.figure(figsize=(3.3, 3.3))
    for index, color in zip(indices, colors):
        plot_caustics(
            combined.caustics(float(times[index]), params), color=color, lw=1.1
        )
        plt.scatter([display_x[index]], [display_y[index]], color=color)
    plt.plot(display_x, display_y, color="0.25")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.axis("equal")
    save("CombinedEffects_geometry.png")


def main():
    standard_binary()
    binary_lens_images()
    standard_triple()
    critical_curves_and_caustics()
    parallax()
    orbital_motion()
    binary_source()
    binary_source_binary_lens()
    limb_darkening()
    accuracy_method_selection()
    coordinates()
    combined_effects()


if __name__ == "__main__":
    main()
