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
    q, s, y1, y2, rho = 0.1, 0.8, 0.015, 0.0, 0.01
    image_plane = lcbinint.image.ImagePlane(
        q=q, s=s, x=y1, y=y2, rho=rho, coordinates="lcbinint"
    )
    caustics = image_plane.caustics()
    critical_curves = image_plane.critical_curves()
    image_regions = image_plane.ray_shooting_images(resolution=300)

    fig, (source_ax, image_ax) = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for x, y in zip(caustics.x, caustics.y):
        source_ax.plot(x, y, color="tab:red", lw=1.1)
    source_ax.add_patch(Circle(
        (y1, y2), rho, fill=False, edgecolor="tab:blue", linewidth=1.4
    ))
    source_ax.set(title="Source plane", xlabel="x", ylabel="y", aspect="equal")

    for x, y in zip(critical_curves.x, critical_curves.y):
        image_ax.plot(x, y, color="black")
    for region in image_regions:
        if len(region.points):
            image_ax.scatter(region.points[:, 0], region.points[:, 1], s=2, color="tab:red")
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
        plt.plot(
            [source_x[index]], [source_y[index]], color=color,
            marker="o", markersize=2.5,
        )
    plt.axis("equal")
    save("BinaryLens_lightcurve_orbital_caustics.png")


def binary_source():
    params = dict(
        s=1.0, q=1.0, alpha=0.0, tE=37.3, t0=7550.4,
        u0=0.075, rho1=0.004, t0_2=7552.0, u0_2=-0.04, rho2=0.002,
        flux_ratio=0.4,
    )
    times = np.linspace(7550.4 - 37.3, 7550.4 + 37.3, 300)
    options = lcbinint.Options(tol=1e-3, reltol=1e-3)
    binary = lcbinint.LightCurve(source="binary", options=options)

    source1 = lcbinint.LightCurve(options=options)
    source1_params = dict(params, rho=params["rho1"])
    source2_params = dict(
        params,
        t0=params["t0_2"], u0=params["u0_2"], rho=params["rho2"],
    )
    for key in ("rho1", "t0_2", "u0_2", "rho2", "flux_ratio"):
        source1_params.pop(key)
        source2_params.pop(key)

    source1_magnification = source1(times, source1_params)
    source2_magnification = source1(times, source2_params)
    binary_magnification = binary(times, params)

    plt.figure(figsize=(4.8, 3.0))
    plt.plot(times, source1_magnification, color="#0173B2", alpha=0.45, lw=1.0, label="source 1")
    plt.plot(times, source2_magnification, color="#029E73", alpha=0.45, lw=1.0, label="source 2")
    plt.plot(times, binary_magnification, color="black", lw=1.5,
             label="total")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="upper left", fontsize=8)
    save("BinarySource_static_lightcurve.png")

    source1_trajectory = source1.source_trajectory(times, source1_params)
    source2_trajectory = source1.source_trajectory(times, source2_params)
    caustics = source1.caustics(source1_params)
    plt.figure(figsize=(2.8, 2.8))
    plot_caustics(caustics, color="#6C6C6C", lw=1.1)
    plt.plot(
        -np.asarray(source1_trajectory.x), -np.asarray(source1_trajectory.y),
        color="#0173B2", label="source 1",
    )
    plt.plot(
        -np.asarray(source2_trajectory.x), -np.asarray(source2_trajectory.y),
        color="#029E73", label="source 2",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend(fontsize=7)
    save("BinarySource_static_geometry.png")


def binary_source_xallarap_trajectories():
    """Projected source paths for the binary velocity-xallarap convention."""
    t_ref = 7500.0
    mass_ratio = 0.7
    times = np.linspace(t_ref - 30.0, t_ref + 30.0, 300)
    common = dict(
        s=0.9, q=0.1, alpha=1.0, tE=30.0, t0=t_ref, u0=0.10, rho=0.0,
        xi_1=0.04, xi_2=-0.02, w1=0.01, w2=0.8, w3=0.2,
    )
    xallarap = lcbinint.LightCurve(
        xallarap="circular_velocity", t_ref=t_ref,
    )
    static = lcbinint.LightCurve()
    first = xallarap.source_trajectory(times, **common)
    second = xallarap.source_trajectory(
        times,
        **dict(
            common,
            xi_1=-common["xi_1"] / mass_ratio,
            xi_2=-common["xi_2"] / mass_ratio,
        ),
    )
    centre = static.source_trajectory(times, **common)
    caustics = static.caustics(**common)
    ref_index = int(np.argmin(np.abs(times - t_ref)))

    plt.figure(figsize=(3.4, 3.2))
    for x, y in zip(caustics.x, caustics.y):
        plt.plot(x, y, color="#6C6C6C", lw=1.1, zorder=1)
    plt.plot(centre.x, centre.y, color="0.55", linestyle="--", label="CoM track")
    plt.plot(first.x, first.y, color="#0173B2", label="source 1")
    plt.plot(second.x, second.y, color="#029E73", label="source 2")
    plt.scatter(
        [first.x[ref_index], second.x[ref_index], centre.x[ref_index]],
        [first.y[ref_index], second.y[ref_index], centre.y[ref_index]],
        color=["#0173B2", "#029E73", "0.35"], s=14, zorder=3,
    )
    plt.xlabel("Trajectory coordinate 1")
    plt.ylabel("Trajectory coordinate 2")
    plt.axis("equal")
    plt.legend(fontsize=8)
    save("BinarySource_xallarap_trajectories.png")


def binary_source_xallarap_lightcurve():
    params = dict(
        s=0.9, q=0.1, alpha=1.0, tE=30.0, t0=7500.0, u0=0.10,
        rho1=0.004, rho2=0.002, flux_ratio=0.4, source_mass_ratio=0.7,
        xi_1=0.04, xi_2=-0.02, w1=0.01, w2=0.8, w3=0.2,
    )
    times = np.linspace(7470.0, 7530.0, 300)
    circular = lcbinint.LightCurve(
        source="binary", xallarap="circular_velocity",
        source_orbit_coordinates="xallarap", t_ref=7500.0,
    )
    component_curve = lcbinint.LightCurve(xallarap="circular_velocity", t_ref=7500.0)
    source1_params = dict(params, rho=params["rho1"])
    for key in ("rho1", "rho2", "flux_ratio", "source_mass_ratio"):
        source1_params.pop(key)
    source2_params = dict(
        source1_params,
        rho=params["rho2"],
        xi_1=-source1_params["xi_1"] / params["source_mass_ratio"],
        xi_2=-source1_params["xi_2"] / params["source_mass_ratio"],
    )
    plt.figure(figsize=(4.2, 2.7))
    plt.plot(times, component_curve(times, source1_params), color="#0173B2", alpha=0.45, lw=1.0, label="source 1")
    plt.plot(times, component_curve(times, source2_params), color="#029E73", alpha=0.45, lw=1.0, label="source 2")
    plt.plot(times, circular(times, params), color="black", lw=1.5, label="total")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="upper left", fontsize=8)
    save("BinarySource_xallarap_lightcurve.png")


def binary_source_xallarap_elements_lightcurve():
    params = dict(
        s=0.9, q=0.1, alpha=1.0, tE=30.0, t0=7500.0, u0=0.10,
        rho1=0.004, rho2=0.002, flux_ratio=0.4, source_mass_ratio=0.7,
        xi_1=0.04, xi_2=-0.02,
    )
    times = np.linspace(7470.0, 7530.0, 300)
    circular = lcbinint.LightCurve(
        source="binary", xallarap="circular_elements", t_ref=7500.0,
    )
    component_curve = lcbinint.LightCurve(xallarap="circular_elements", t_ref=7500.0)
    component_params = dict(params, period_xa=120.0, inc_xa=0.8)
    source1_params = dict(component_params, rho=params["rho1"])
    for key in ("rho1", "rho2", "flux_ratio", "source_mass_ratio"):
        source1_params.pop(key)
    source2_params = dict(
        source1_params,
        rho=params["rho2"],
        xi_1=-source1_params["xi_1"] / params["source_mass_ratio"],
        xi_2=-source1_params["xi_2"] / params["source_mass_ratio"],
    )
    plt.figure(figsize=(4.2, 2.7))
    plt.plot(times, component_curve(times, source1_params), color="#0173B2", alpha=0.45, lw=1.0, label="source 1")
    plt.plot(times, component_curve(times, source2_params), color="#029E73", alpha=0.45, lw=1.0, label="source 2")
    plt.plot(
        times, circular(times, component_params),
        color="black", lw=1.5, label="total",
    )
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="upper left", fontsize=8)
    save("BinarySource_xallarap_elements_lightcurve.png")

    caustics = lcbinint.LightCurve().caustics(source1_params)
    first = component_curve.source_trajectory(times, source1_params)
    second = component_curve.source_trajectory(times, source2_params)
    plt.figure(figsize=(3.4, 3.2))
    plot_caustics(caustics, color="#6C6C6C", lw=1.1)
    plt.plot(first.x, first.y, color="#0173B2", label="source 1")
    plt.plot(second.x, second.y, color="#029E73", label="source 2")
    plt.xlabel("Trajectory coordinate 1")
    plt.ylabel("Trajectory coordinate 2")
    plt.axis("equal")
    plt.legend(fontsize=7)
    save("BinarySource_xallarap_elements_geometry.png")


def binary_source_xallarap_offset_lightcurve():
    params = dict(
        s=0.9, q=0.1, alpha=1.0, tE=30.0, t0=7500.0, u0=0.10,
        t0_2=7501.2, u0_2=-0.06, rho1=0.004, rho2=0.002,
        flux_ratio=0.4, source_mass_ratio=0.7,
        w1=0.01, w2=0.8, w3=0.2,
    )
    times = np.linspace(7470.0, 7530.0, 300)
    curve = lcbinint.LightCurve(
        source="binary", xallarap="circular_velocity",
        source_orbit_coordinates="trajectory_offset", t_ref=7500.0,
    )
    component_curve = lcbinint.LightCurve(xallarap="circular_velocity", t_ref=7500.0)
    source_mass_ratio = params["source_mass_ratio"]
    relative_tau = (params["t0"] - params["t0_2"]) / params["tE"]
    relative_beta = params["u0_2"] - params["u0"]
    source1_params = dict(
        s=params["s"], q=params["q"], alpha=params["alpha"], tE=params["tE"],
        t0=(params["t0"] + source_mass_ratio * params["t0_2"]) / (1.0 + source_mass_ratio),
        u0=(params["u0"] + source_mass_ratio * params["u0_2"]) / (1.0 + source_mass_ratio),
        rho=params["rho1"],
        xi_1=-source_mass_ratio * relative_tau / (1.0 + source_mass_ratio),
        xi_2=-source_mass_ratio * relative_beta / (1.0 + source_mass_ratio),
        w1=params["w1"], w2=params["w2"], w3=params["w3"],
    )
    source2_params = dict(
        source1_params,
        rho=params["rho2"],
        xi_1=-source1_params["xi_1"] / source_mass_ratio,
        xi_2=-source1_params["xi_2"] / source_mass_ratio,
    )
    plt.figure(figsize=(4.2, 2.7))
    plt.plot(times, component_curve(times, source1_params), color="#0173B2", alpha=0.45, lw=1.0, label="source 1")
    plt.plot(times, component_curve(times, source2_params), color="#029E73", alpha=0.45, lw=1.0, label="source 2")
    plt.plot(times, curve(times, params), color="black", lw=1.5, label="total")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="upper left", fontsize=8)
    save("BinarySource_xallarap_offset_lightcurve.png")

    caustics = lcbinint.LightCurve().caustics(source1_params)
    first = component_curve.source_trajectory(times, source1_params)
    second = component_curve.source_trajectory(times, source2_params)
    plt.figure(figsize=(3.4, 3.2))
    plot_caustics(caustics, color="#6C6C6C", lw=1.1)
    plt.plot(first.x, first.y, color="#0173B2", label="source 1")
    plt.plot(second.x, second.y, color="#029E73", label="source 2")
    plt.xlabel("Trajectory coordinate 1")
    plt.ylabel("Trajectory coordinate 2")
    plt.axis("equal")
    plt.legend(fontsize=7)
    save("BinarySource_xallarap_offset_geometry.png")


def xallarap_single_source():
    times = np.linspace(7470.0, 7530.0, 300)
    common = dict(s=0.9, q=0.1, alpha=1.0, tE=30.0, t0=7500.0, u0=0.10, rho=0.004,
                  xi_1=0.04, xi_2=-0.02)
    static = lcbinint.LightCurve()

    elements_params = dict(common, period_xa=120.0, inc_xa=0.8)
    elements = lcbinint.LightCurve(xallarap="circular_elements", t_ref=7500.0)
    plt.figure(figsize=(4.2, 2.7))
    plt.plot(times, elements(times, elements_params), color="#0173B2")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("Xallarap_elements_lightcurve.png")
    trajectory = elements.source_trajectory(times, elements_params)
    caustics = static.caustics(elements_params)
    plt.figure(figsize=(3.4, 3.2))
    plot_caustics(caustics, color="#6C6C6C", lw=1.1)
    plt.plot(trajectory.x, trajectory.y, color="#0173B2")
    plt.xlabel("Trajectory coordinate 1")
    plt.ylabel("Trajectory coordinate 2")
    plt.axis("equal")
    save("Xallarap_elements_geometry.png")

    velocity_params = dict(common, w1=0.01, w2=0.8, w3=0.2)
    velocity = lcbinint.LightCurve(xallarap="circular_velocity", t_ref=7500.0)
    plt.figure(figsize=(4.2, 2.7))
    plt.plot(times, velocity(times, velocity_params), color="#0173B2")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    save("Xallarap_velocity_lightcurve.png")
    trajectory = velocity.source_trajectory(times, velocity_params)
    caustics = static.caustics(velocity_params)
    plt.figure(figsize=(3.4, 3.2))
    plot_caustics(caustics, color="#6C6C6C", lw=1.1)
    plt.plot(trajectory.x, trajectory.y, color="#0173B2")
    plt.xlabel("Trajectory coordinate 1")
    plt.ylabel("Trajectory coordinate 2")
    plt.axis("equal")
    save("Xallarap_velocity_geometry.png")


def binary_source_binary_lens():
    params = dict(
        s=0.9, q=0.1, u0=0.1, alpha=1.0, rho1=0.01, tE=30.0, t0=7500,
        piEN=0.3, piEE=-0.2, g1=0.011, g2=-0.005, g3=0.005,
        t0_2=7501.0, u0_2=-0.05, rho2=0.005, flux_ratio=1.0,
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
            orbital_motion="circular",
            sky=sky, t_ref=7500,
        ),
        options=options,
    )
    binary_source_magnifications = binary_source_curve(times, params)
    single_source_params = {
        key: value for key, value in params.items()
        if key not in {"t0_2", "u0_2", "rho2", "rho1", "flux_ratio"}
    }
    plt.figure(figsize=(6.6, 3.7))
    plt.plot(
        times, single_source(times, single_source_params), "y",
        label="single source + lens orbit",
    )
    plt.plot(
        times, binary_source_magnifications, "g",
        label="binary source",
    )
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    save("BinarySourceBinaryLens_lightcurve.png")


def limb_darkening():
    params = dict(s=0.9, q=0.1, u0=0.0, alpha=1.0, rho=0.01, tE=30.0, t0=7500.0)
    times = np.linspace(7470.0, 7530.0, 300)
    zoom_times = np.linspace(7500.8, 7502.6, 500)
    options = lcbinint.Options(tol=1e-3, reltol=1e-3)
    uniform_curve = lcbinint.LightCurve(
        options=options, limb_darkening=lcbinint.LimbDarkening.none()
    )
    linear_curve = lcbinint.LightCurve(
        options=options, limb_darkening=lcbinint.LimbDarkening.linear(0.51)
    )
    square_root_curve = lcbinint.LightCurve(
        options=options, limb_darkening=lcbinint.LimbDarkening.square_root(0.51, 0.3)
    )

    plt.figure(figsize=(3.8, 2.55))
    plt.plot(times, square_root_curve(times, params), color="tab:green", label="square-root profile")
    plt.axvspan(7500.8, 7502.6, color="0.85", zorder=0, label="zoomed interval")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(fontsize=8)
    save("LimbDarkening_full_event.png")

    plt.figure(figsize=(5.5, 3.2))
    plt.plot(zoom_times, uniform_curve(zoom_times, params), label="uniform")
    plt.plot(zoom_times, linear_curve(zoom_times, params), label="linear: 0.51")
    plt.plot(
        zoom_times, square_root_curve(zoom_times, params),
        label="square root: 0.51, 0.3",
    )
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
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
    method_colors = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#56B4E9"]

    fig, (light_ax, method_ax) = plt.subplots(
        2, 1, figsize=(4.8, 3.8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    light_ax.plot(times, magnifications)
    light_ax.set_ylabel("Magnification")
    for level, name in enumerate(names, start=1):
        selected = methods == name
        method_ax.scatter(
            times[selected], np.full(selected.sum(), level), s=9,
            color=method_colors[level - 1],
        )
    method_ax.set_yticks(range(1, len(names) + 1))
    method_ax.set(xlabel="Time", ylabel="Method")
    fig.tight_layout()
    plt.savefig(OUTPUT / "Accuracy_method_selection.png", dpi=220)
    plt.close()

    trajectory = curve.source_trajectory(times, params)
    caustics = curve.caustics(params)
    plt.figure(figsize=(2.8, 2.8))
    for x, y in zip(caustics.x, caustics.y):
        plt.plot(-np.asarray(x), -np.asarray(y), color="#6C6C6C", lw=1.1)

    display_x = -np.asarray(trajectory.x)
    display_y = -np.asarray(trajectory.y)
    for color_index, name in enumerate(names):
        indices = np.flatnonzero(methods == name)
        breaks = np.where(np.diff(indices) != 1)[0] + 1
        for run in np.split(indices, breaks):
            if len(run) > 1:
                plt.plot(
                    display_x[run], display_y[run],
                    color=method_colors[color_index], lw=1.2,
                )
            elif len(run) == 1:
                plt.plot(
                    display_x[run], display_y[run], color=method_colors[color_index],
                    marker="o", markersize=2.5,
                )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    save("Accuracy_method_geometry.png")


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
        xi_1=0.0, xi_2=-0.1, w1=0.01, w2=0.02, w3=0.015,
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
            lens="binary", parallax=True,
            orbital_motion="circular", xallarap="circular_velocity",
            sky=sky, t_ref=7500,
        ),
        options=options,
    )
    single_source_params = {
        key: value for key, value in params.items()
        if key not in {"xi_1", "xi_2", "w1", "w2", "w3"}
    }
    plt.figure(figsize=(7.4, 3.8))
    plt.plot(times, static(times, single_source_params), label="static 2L1S")
    plt.plot(times, parallax_curve(times, single_source_params), label="+ parallax")
    plt.plot(times, parallax_orbit(times, single_source_params), label="+ lens orbit")
    plt.plot(times, combined(times, params), label="+ xallarap")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    save("CombinedEffects_lightcurve.png")

    trajectory = combined.source_trajectory(times, params)
    display_x = -np.asarray(trajectory.x)
    display_y = -np.asarray(trajectory.y)
    indices = [75, 150, 225]
    colors = ["#0173B2", "#029E73", "#CC78BC"]
    plt.figure(figsize=(3.3, 3.3))
    plt.plot(display_x, display_y, color="0.35", lw=0.9, zorder=1)
    for index, color in zip(indices, colors):
        plot_caustics(
            combined.caustics(float(times[index]), params), color=color, lw=1.1,
            zorder=2,
        )
        plt.scatter(
            [display_x[index]], [display_y[index]], s=16, color=color,
            edgecolor="white", linewidth=0.35, zorder=3,
        )
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
    binary_source_xallarap_elements_lightcurve()
    binary_source_xallarap_lightcurve()
    binary_source_xallarap_offset_lightcurve()
    xallarap_single_source()
    binary_source_xallarap_trajectories()
    binary_source_binary_lens()
    limb_darkening()
    accuracy_method_selection()
    coordinates()
    combined_effects()


if __name__ == "__main__":
    main()
