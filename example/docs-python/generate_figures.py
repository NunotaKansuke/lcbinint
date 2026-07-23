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
        plt.plot(x, y, *args, **kwargs)


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
        trajectory.x, trajectory.y, color="tab:blue"
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
        trajectory.x, trajectory.y, color="tab:blue"
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
        static_trajectory.x, static_trajectory.y,
        color="tab:blue", linestyle="--",
    )
    plt.plot(
        moved_trajectory.x, moved_trajectory.y,
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
    source_x = np.asarray(trajectory.x)
    source_y = np.asarray(trajectory.y)
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

    components = binary.binary_source_components(times, params)

    plt.figure(figsize=(4.8, 3.0))
    plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, lw=1.0, label="source 1")
    plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, lw=1.0, label="source 2")
    plt.plot(times, components.total, color="black", lw=1.5,
             label="total")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="upper left", fontsize=8)
    save("BinarySource_static_lightcurve.png")

    caustics = binary.caustics(params)
    plt.figure(figsize=(2.8, 2.8))
    plot_caustics(caustics, color="#6C6C6C", lw=1.1)
    plt.plot(
        components.source1.trajectory.x, components.source1.trajectory.y,
        color="#0173B2", label="source 1",
    )
    plt.plot(
        components.source2.trajectory.x, components.source2.trajectory.y,
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
    times = np.linspace(t_ref - 30.0, t_ref + 30.0, 300)
    params = dict(
        s=0.9, q=0.1, alpha=0.7, tE=30.0, t0=t_ref, u0=0.20,
        rho1=0.004, rho2=0.002, flux_ratio=0.4, source_mass_ratio=0.7,
        xi_1=0.02, xi_2=-0.01, w1=0.004, w2=0.35, w3=0.08,
    )
    curve = lcbinint.LightCurve(
        source="binary", xallarap="circular_velocity",
        source_orbit_coordinates="xallarap", t_ref=t_ref,
    )
    components = curve.binary_source_components(times, params)
    caustics = curve.caustics(params)

    plt.figure(figsize=(3.4, 3.2))
    for x, y in zip(caustics.x, caustics.y):
        plt.plot(x, y, color="#6C6C6C", lw=1.1, zorder=1)
    plt.plot(components.source1.trajectory.x, components.source1.trajectory.y,
             color="#0173B2", label="source 1")
    plt.plot(components.source2.trajectory.x, components.source2.trajectory.y,
             color="#029E73", label="source 2")
    plt.xlabel("Trajectory coordinate 1")
    plt.ylabel("Trajectory coordinate 2")
    plt.axis("equal")
    plt.legend(fontsize=8)
    save("BinarySource_xallarap_trajectories.png")


def binary_source_xallarap_lightcurve():
    params = dict(
        s=0.9, q=0.1, alpha=0.7, tE=30.0, t0=7500.0, u0=0.20,
        rho1=0.004, rho2=0.002, flux_ratio=0.4, source_mass_ratio=0.7,
        xi_1=0.02, xi_2=-0.01, w1=0.004, w2=0.35, w3=0.08,
    )
    times = np.linspace(7470.0, 7530.0, 300)
    circular = lcbinint.LightCurve(
        source="binary", xallarap="circular_velocity",
        source_orbit_coordinates="xallarap", t_ref=7500.0,
    )
    components = circular.binary_source_components(times, params)
    plt.figure(figsize=(4.2, 2.7))
    plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, lw=1.0, label="source 1")
    plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, lw=1.0, label="source 2")
    plt.plot(times, components.total, color="black", lw=1.5, label="total")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="upper left", fontsize=8)
    save("BinarySource_xallarap_lightcurve.png")


def binary_source_xallarap_elements_lightcurve():
    params = dict(
        s=0.9, q=0.1, alpha=0.7, tE=30.0, t0=7500.0, u0=0.20,
        rho1=0.004, rho2=0.002, flux_ratio=0.4, source_mass_ratio=0.7,
        xi_1=0.02, xi_2=-0.01,
    )
    times = np.linspace(7470.0, 7530.0, 1200)
    circular = lcbinint.LightCurve(
        source="binary", xallarap="circular_elements", t_ref=7500.0,
    )
    component_params = dict(
        params, xi_1=0.006, xi_2=-0.003, period_xa=12.0, inc_xa=0.6,
    )
    components = circular.binary_source_components(times, component_params)
    plt.figure(figsize=(4.2, 2.7))
    plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, lw=1.0, label="source 1")
    plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, lw=1.0, label="source 2")
    plt.plot(
        times, components.total,
        color="black", lw=1.5, label="total",
    )
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="upper left", fontsize=8)
    save("BinarySource_xallarap_elements_lightcurve.png")

    caustics = circular.caustics(component_params)
    plt.figure(figsize=(3.4, 3.2))
    for x, y in zip(caustics.x, caustics.y):
        plt.plot(x, y, color="#6C6C6C", lw=1.1)
    plt.plot(components.source1.trajectory.x, components.source1.trajectory.y, color="#0173B2", label="source 1")
    plt.plot(components.source2.trajectory.x, components.source2.trajectory.y, color="#029E73", label="source 2")
    plt.xlabel("Trajectory coordinate 1")
    plt.ylabel("Trajectory coordinate 2")
    plt.axis("equal")
    plt.legend(fontsize=7)
    save("BinarySource_xallarap_elements_geometry.png")


def binary_source_xallarap_offset_lightcurve():
    params = dict(
        s=0.9, q=0.1, alpha=0.7, tE=30.0, t0=7499.4, u0=0.19,
        t0_2=7500.857142857, u0_2=0.214285714, rho1=0.004, rho2=0.002,
        flux_ratio=0.4, source_mass_ratio=0.7,
        w1=0.004, w2=0.35, w3=0.08,
    )
    times = np.linspace(7470.0, 7530.0, 300)
    curve = lcbinint.LightCurve(
        source="binary", xallarap="circular_velocity",
        source_orbit_coordinates="trajectory_offset", t_ref=7500.0,
    )
    components = curve.binary_source_components(times, params)
    plt.figure(figsize=(4.2, 2.7))
    plt.plot(times, components.source1.magnification, color="#0173B2", alpha=0.45, lw=1.0, label="source 1")
    plt.plot(times, components.source2.magnification, color="#029E73", alpha=0.45, lw=1.0, label="source 2")
    plt.plot(times, components.total, color="black", lw=1.5, label="total")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="upper left", fontsize=8)
    save("BinarySource_xallarap_offset_lightcurve.png")

    caustics = curve.caustics(params)
    plt.figure(figsize=(3.4, 3.2))
    plot_caustics(caustics, color="#6C6C6C", lw=1.1)
    plt.plot(components.source1.trajectory.x, components.source1.trajectory.y, color="#0173B2", label="source 1")
    plt.plot(components.source2.trajectory.x, components.source2.trajectory.y, color="#029E73", label="source 2")
    plt.xlabel("Trajectory coordinate 1")
    plt.ylabel("Trajectory coordinate 2")
    plt.axis("equal")
    plt.legend(fontsize=7)
    save("BinarySource_xallarap_offset_geometry.png")


def xallarap_single_source():
    times = np.linspace(7470.0, 7530.0, 300)
    common = dict(s=0.9, q=0.1, alpha=0.7, tE=30.0, t0=7500.0, u0=0.30, rho=0.004,
                  xi_1=0.02, xi_2=-0.01)
    static = lcbinint.LightCurve()

    elements_params = dict(
        common, u0=0.20, xi_1=0.006, xi_2=-0.003,
        period_xa=12.0, inc_xa=0.6,
    )
    elements = lcbinint.LightCurve(xallarap="circular_elements", t_ref=7500.0)
    elements_times = np.linspace(7470.0, 7530.0, 1200)
    elements_static_params = dict(elements_params)
    for key in ("xi_1", "xi_2", "period_xa", "inc_xa"):
        elements_static_params.pop(key)
    plt.figure(figsize=(4.2, 2.7))
    plt.plot(elements_times, static(elements_times, elements_static_params), color="0.55", ls="--", label="rectilinear")
    plt.plot(elements_times, elements(elements_times, elements_params), color="#0173B2", label="xallarap")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="upper left", fontsize=8)
    save("Xallarap_elements_lightcurve.png")
    trajectory = elements.source_trajectory(elements_times, elements_params)
    caustics = static.caustics(elements_static_params)
    plt.figure(figsize=(3.4, 3.2))
    for x, y in zip(caustics.x, caustics.y):
        plt.plot(x, y, color="#6C6C6C", lw=1.1)
    static_trajectory = static.source_trajectory(elements_times, elements_static_params)
    plt.plot(static_trajectory.x, static_trajectory.y, color="0.55", ls="--", label="rectilinear")
    plt.plot(trajectory.x, trajectory.y, color="#0173B2", label="xallarap")
    plt.xlabel("Trajectory coordinate 1")
    plt.ylabel("Trajectory coordinate 2")
    plt.axis("equal")
    plt.legend(fontsize=7)
    save("Xallarap_elements_geometry.png")

    velocity_params = dict(common, u0=0.20, w1=0.004, w2=0.35, w3=0.08)
    velocity = lcbinint.LightCurve(xallarap="circular_velocity", t_ref=7500.0)
    velocity_static_params = dict(velocity_params)
    for key in ("xi_1", "xi_2", "w1", "w2", "w3"):
        velocity_static_params.pop(key)
    plt.figure(figsize=(4.2, 2.7))
    plt.plot(times, static(times, velocity_static_params), color="0.55", ls="--", label="rectilinear")
    plt.plot(times, velocity(times, velocity_params), color="#0173B2", label="xallarap")
    plt.xlabel("Time")
    plt.ylabel("Magnification")
    plt.legend(loc="upper left", fontsize=8)
    save("Xallarap_velocity_lightcurve.png")
    trajectory = velocity.source_trajectory(times, velocity_params)
    caustics = velocity.caustics(velocity_params)
    plt.figure(figsize=(3.4, 3.2))
    for x, y in zip(caustics.x, caustics.y):
        plt.plot(x, y, color="#6C6C6C", lw=1.1)
    static_trajectory = static.source_trajectory(times, velocity_static_params)
    plt.plot(static_trajectory.x, static_trajectory.y, color="0.55", ls="--", label="rectilinear")
    plt.plot(trajectory.x, trajectory.y, color="#0173B2", label="xallarap")
    plt.xlabel("Trajectory coordinate 1")
    plt.ylabel("Trajectory coordinate 2")
    plt.axis("equal")
    plt.legend(fontsize=7)
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
        plt.plot(x, y, color="#6C6C6C", lw=1.1)

    display_x = np.asarray(trajectory.x)
    display_y = np.asarray(trajectory.y)
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
    display_x = np.asarray(trajectory.x)
    display_y = np.asarray(trajectory.y)
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
    display_x = np.asarray(trajectory.x)
    display_y = np.asarray(trajectory.y)
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


def higher_order_combination_figures():
    """Render the catalogue figures used by the higher-order index pages."""
    sky = lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2")
    options = lcbinint.Options(coordinates="vbm", tol=1e-3, reltol=1e-3)
    times = np.linspace(7470.0, 7530.0, 300)
    common = dict(
        s=0.9, q=0.1, t0=7500.0, u0=0.20, tE=30.0, alpha=0.7,
        piEN=0.03, piEE=-0.02, g1=0.011, g2=-0.005, g3=0.005,
        xi_1=0.02, xi_2=-0.01, w1=0.004, w2=0.35, w3=0.08,
    )
    configurations = [
        ("FiniteSource", "Finite source", False, False, False, False),
        ("ParallaxOrbital", "Parallax + lens orbit", False, True, True, False),
        ("ParallaxXallarap", "Parallax + xallarap", False, True, False, True),
        ("ParallaxOrbitalXallarap", "Parallax + lens orbit + xallarap", False, True, True, True),
        ("BinarySourceParallax", "Binary source + parallax", True, True, False, False),
        ("BinarySourceParallaxOrbital", "Binary source + parallax + lens orbit", True, True, True, False),
        ("BinarySourceParallaxXallarap", "Binary source + parallax + xallarap", True, True, False, True),
        ("BinarySourceParallaxOrbitalXallarap", "Binary source + parallax + lens orbit + xallarap", True, True, True, True),
    ]
    for stem, title, binary_source, parallax, orbital, xallarap in configurations:
        params = dict(common)
        if not xallarap:
            for key in ("xi_1", "xi_2", "w1", "w2", "w3"):
                params.pop(key)
        if binary_source:
            params.update(
                rho1=0.004,
                rho2=0.003,
                flux_ratio=0.4,
            )
            if xallarap:
                params["source_mass_ratio"] = 0.7
            else:
                params.update(t0_2=7501.2, u0_2=-0.06)
        else:
            params["rho"] = 0.004

        model_args = dict(
            source="binary" if binary_source else "single",
            parallax=parallax,
            sky=sky if parallax else None,
            t_ref=7500.0,
        )
        if orbital:
            model_args["orbital_motion"] = "circular"
        if xallarap:
            model_args["xallarap"] = "circular_velocity"
            if binary_source:
                model_args["source_orbit_coordinates"] = "xallarap"
        curve = lcbinint.LightCurve(model=lcbinint.Model(**model_args), options=options)
        magnification = curve(times, params)

        plt.figure(figsize=(3.8, 2.4))
        plt.plot(times, magnification, color="#0173B2")
        plt.xlabel("Time")
        plt.ylabel("Magnification")
        plt.title(title, fontsize=9)
        save(f"{stem}_lightcurve.png")

        caustics = curve.caustics(7500.0, params) if orbital else curve.caustics(params)
        plt.figure(figsize=(2.8, 2.7))
        plot_caustics(caustics, color="#6C6C6C", lw=1.1)
        if binary_source:
            components = curve.binary_source_components(times, params)
            plt.plot(components.source1.trajectory.x, components.source1.trajectory.y,
                     color="#0173B2", label="source 1")
            plt.plot(components.source2.trajectory.x, components.source2.trajectory.y,
                     color="#029E73", label="source 2")
            plt.legend(fontsize=7)
        else:
            trajectory = curve.source_trajectory(times, params)
            plt.plot(trajectory.x, trajectory.y, color="#0173B2")
        plt.xlabel("Trajectory coordinate 1")
        plt.ylabel("Trajectory coordinate 2")
        plt.axis("equal")
        save(f"{stem}_geometry.png")


def higher_order_catalogue():
    """Render and document every hierarchy-respecting high-order configuration.

    This deliberately makes one light-curve figure and one geometry figure per
    configuration.  The catalogue is intended for reading one physical model
    at a time, rather than for comparing a contact sheet of tiny plots.
    """
    sky = lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2")
    options = lcbinint.Options(coordinates="vbm", tol=1e-3, reltol=1e-3)
    times = np.linspace(7470.0, 7530.0, 160)
    space_geometry_times = np.linspace(7470.0, 7530.0, 400)
    space_table = np.column_stack([
        2450000.0 + space_geometry_times,
        np.full(len(space_geometry_times), 180.0),
        np.full(len(space_geometry_times), -30.0),
        np.full(len(space_geometry_times), 1.0),
    ])
    parallax_sites = {
        "chile": lcbinint.obs.Site("ground", -29.0, -70.7),
        "africa": lcbinint.obs.Site("ground", -29.0, 20.0),
        "space": lcbinint.obs.Site("space", space_table),
    }
    base = dict(s=.9, q=.1, t0=7500., u0=.20, tE=30., alpha=.7,
                piEN=.03, piEE=-.02)
    def modes(binary):
        out = [("Parallax", None, None, None)]
        x = [("circular-elements xallarap", "circular_elements", None),
             ("Kepler-elements xallarap", "orbital_elements", None),
             ("direct circular-velocity xallarap", "circular_velocity", "xallarap"),
             ("direct Kepler-velocity xallarap", "kepler_velocity", "xallarap")]
        if binary:
            x += [("trajectory-offset circular-velocity xallarap", "circular_velocity", "trajectory_offset"),
                  ("trajectory-offset Kepler-velocity xallarap", "kepler_velocity", "trajectory_offset")]
        out += [("Parallax + " + name, xm, co, None) for name, xm, co in x]
        for orbit, label in (("circular", "O"), ("kepler", "OK")):
            orbit_name = "circular lens orbit" if orbit == "circular" else "Kepler lens orbit"
            out.append(("Parallax + " + orbit_name, None, None, orbit))
            out += [("Parallax + " + orbit_name + " + " + name, xm, co, orbit)
                    for name, xm, co in x]
        return out

    def slug(text):
        return "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_").replace("__", "_")

    def config_parameters(lens, source, xmode, coordinates, orbit):
        params = dict(base)
        if orbit:
            params.update(g1=.011, g2=-.005, g3=.005)
            if orbit == "kepler":
                params.update(lom_szs=.2, lom_ar=1.4)
        if lens == "triple":
            params.update(sep2=1.3, q2=.01, ang=.5)
        if xmode in ("circular_elements", "orbital_elements"):
            params.update(xi_1=.006, xi_2=-.003, period_xa=12., inc_xa=.6)
            if xmode == "orbital_elements":
                params.update(ecc_xa=.2, peri_xa=.4)
        elif xmode:
            params.update(xi_1=.006, xi_2=-.003, w1=.004, w2=.35, w3=.08)
            if xmode == "kepler_velocity":
                params.update(xa_szs=.2, xa_ar=1.4)
        if source == "binary":
            params.update(rho1=.004, rho2=.003, flux_ratio=.4)
            if xmode:
                params["source_mass_ratio"] = .7
                if coordinates == "trajectory_offset":
                    params.update(t0=7499.4, u0=.19, t0_2=7500.857142857, u0_2=.214285714)
            else:
                params.update(t0_2=7501.2, u0_2=-.06)
        else:
            params["rho"] = .004
        return params

    catalogue = [
        "[← Higher-order effects](CombinedEffects.md)",
        "", "# Higher-order combination catalogue", "",
        "The catalogue is grouped first by lens and source multiplicity. Within every group, the examples progress through three levels: source-size baselines, annual parallax, and parallax with additional higher-order effects. Lens orbit is therefore never shown without parallax, and triple lenses are limited to their supported static-lens geometry.",
        "", '`source` selects source multiplicity (`"single"` or `"binary"`). `finite_source=False` selects point-source evaluation and sets every source radius to zero during evaluation.',
        "", "```python", "import numpy as np", "import matplotlib.pyplot as plt", "import lcbinint", "", "times = np.linspace(7470.0, 7530.0, 160)", "sky = lcbinint.obs.SkyCoord(\"17:59:02.3\", \"-29:04:15.2\")", "options = lcbinint.Options(coordinates=\"vbm\", tol=1e-3, reltol=1e-3)", "", "space_geometry_times = np.linspace(7470.0, 7530.0, 400)", "space_ephemeris = {", "    \"jd\": 2450000.0 + space_geometry_times,", "    \"ra_deg\": np.full(len(space_geometry_times), 180.0),", "    \"dec_deg\": np.full(len(space_geometry_times), -30.0),", "    \"distance_au\": np.full(len(space_geometry_times), 1.0),", "}", "parallax_sites = {", "    \"chile\": lcbinint.obs.Site(\"ground\", -29.0, -70.7),", "    \"africa\": lcbinint.obs.Site(\"ground\", -29.0, 20.0),", "    \"space\": lcbinint.obs.Site(\"space\", np.column_stack(tuple(space_ephemeris.values()))),", "}", "```", "",
    ]
    groups = (("binary", "single", "Binary lens, single source"),
              ("binary", "binary", "Binary lens, binary source"),
              ("triple", "single", "Triple lens, single source"),
              ("triple", "binary", "Triple lens, binary source"))
    for lens, source, heading in groups:
        catalogue += ["## " + heading, ""]
        binary = source == "binary"
        configs = modes(binary)
        if lens == "triple":
            configs = [item for item in configs if item[3] is None]
        configs.insert(0, ("Finite source", None, None, None))
        configs.insert(0, ("Point source", None, None, None))
        observer_configs = []
        for label, xmode, coordinates, orbit in configs:
            if label in ("Point source", "Finite source"):
                observer_configs.append((label, xmode, coordinates, orbit, None))
                continue
            suffix = label.removeprefix("Parallax")
            observers = ("annual",)
            if (label == "Parallax" and lens == "binary" and source == "single"):
                observers = ("annual", "terrestrial", "space")
            for observer in observers:
                observer_configs.append((observer.capitalize() + " parallax" + suffix,
                                         xmode, coordinates, orbit, observer))
        level = None
        for label, xmode, coordinates, orbit, observer in observer_configs:
            if label in ("Point source", "Finite source"):
                category = "### 1. Source-size baselines"
            elif xmode is None and orbit is None:
                category = "### 2. Parallax"
            else:
                category = "### 3. Parallax with additional higher-order effects"
            if category != level:
                catalogue += [category, ""]
                level = category
            params = config_parameters(lens, source, xmode, coordinates, orbit)
            point_source = label == "Point source"
            parallax = observer is not None
            if point_source:
                if binary:
                    params["rho1"] = 0.0
                    params["rho2"] = 0.0
                else:
                    params["rho"] = 0.0
            if not parallax:
                params.pop("piEN")
                params.pop("piEE")
            sample_times = times
            geometry_times = sample_times
            if observer == "terrestrial":
                params.update(u0=0.0003, rho=0.00001)
                sample_times = np.linspace(7501.37, 7501.46, 600)
                geometry_times = np.linspace(7485.0, 7515.0, 300)
            if observer == "space":
                # A transverse observer baseline makes both the full light
                # curve and the two source tracks easy to compare.
                params.update(u0=0.03, rho=0.004, piEN=0.08, piEE=-0.06)
                sample_times = np.linspace(7470.0, 7530.0, 1800)
                geometry_times = space_geometry_times
            args = dict(lens=lens, source=source,
                        finite_source=not point_source,
                        parallax=parallax, t_ref=7500.)
            if parallax:
                args["sky"] = sky
            if observer in ("terrestrial", "space"):
                args["terrestrial"] = True
            if orbit:
                args["orbital_motion"] = orbit
            if xmode:
                args["xallarap"] = xmode
            if binary and coordinates:
                args["source_orbit_coordinates"] = coordinates
            curve_kwargs = dict(options=options, model=lcbinint.Model(**args))
            if observer == "terrestrial":
                curve_kwargs["site"] = parallax_sites["chile"]
            if observer == "space":
                curve_kwargs["site"] = parallax_sites["space"]
            curve = lcbinint.LightCurve(**curve_kwargs)
            comparison_curves = None
            if observer == "space":
                comparison_curves = {
                    "ground": lcbinint.LightCurve(
                        options=options, model=lcbinint.Model(**args),
                        site=parallax_sites["chile"]),
                    "space": curve,
                }
            elif observer == "terrestrial":
                comparison_curves = {
                    "Chile": curve,
                    "Africa": lcbinint.LightCurve(
                        options=options, model=lcbinint.Model(**args),
                        site=parallax_sites["africa"]),
                }
            components = curve.binary_source_components(sample_times, params) if binary else None
            comparison_magnifications = (
                {name: value(sample_times, params) for name, value in comparison_curves.items()}
                if comparison_curves else None
            )
            mag = (comparison_magnifications["space"] if observer == "space" else
                   comparison_magnifications["Chile"] if observer == "terrestrial" else
                   components.total if binary else curve(sample_times, params))
            caustics = curve.caustics(7500., params) if orbit else curve.caustics(params)
            stem = "HigherOrder_" + slug("_".join((lens, source, label)))
            if observer == "terrestrial":
                figure, (curve_axis, difference_axis) = plt.subplots(
                    2, 1, sharex=True, figsize=(3.8, 3.15),
                    gridspec_kw={"height_ratios": [3, 1]},
                )
                curve_axis.plot(sample_times, comparison_magnifications["Chile"], color="#0173B2", lw=1.1, label="Chile")
                curve_axis.plot(sample_times, comparison_magnifications["Africa"], color="#CC79A7", lw=1.1, label="Africa")
                curve_axis.set_ylabel("Magnification")
                curve_axis.legend(loc="upper left", fontsize=7)
                difference_axis.plot(sample_times, comparison_magnifications["Africa"] - comparison_magnifications["Chile"], color="#6C6C6C", lw=1.0)
                difference_axis.axhline(0.0, color="0.75", lw=.8)
                difference_axis.set(xlabel="Time", ylabel="Africa − Chile")
            else:
                plt.figure(figsize=(3.8, 2.4))
            if observer == "terrestrial":
                pass
            elif binary:
                plt.plot(sample_times, components.source1.magnification, color="#0173B2", alpha=.45, lw=.9, label="source 1")
                plt.plot(sample_times, components.source2.magnification, color="#029E73", alpha=.45, lw=.9, label="source 2")
                plt.plot(sample_times, mag, color="black", lw=1.3, label="total")
                plt.legend(loc="upper left", fontsize=7)
            elif comparison_curves:
                colors = ("#0173B2", "#6C6C6C") if observer == "space" else ("#0173B2", "#CC79A7")
                for (name, values), color in zip(comparison_magnifications.items(), colors):
                    plt.plot(sample_times, values, color=color, lw=1.1, label=name)
                plt.legend(loc="upper left", fontsize=7)
            else:
                plt.plot(sample_times, mag, color="#0173B2", lw=1.1)
            if observer != "terrestrial":
                plt.xlabel("Time"); plt.ylabel("Magnification")
            save(f"{stem}_lightcurve.png")
            plt.figure(figsize=(2.8, 2.7))
            plot_caustics(caustics, color="#6C6C6C", lw=1.1)
            if binary:
                plt.plot(components.source1.trajectory.x, components.source1.trajectory.y, color="#0173B2", lw=1.0, label="source 1")
                plt.plot(components.source2.trajectory.x, components.source2.trajectory.y, color="#029E73", lw=1.0, label="source 2")
                plt.legend(fontsize=7)
            elif comparison_curves:
                colors = ("#0173B2", "#6C6C6C") if observer == "space" else ("#0173B2", "#CC79A7")
                for (name, value), color in zip(comparison_curves.items(), colors):
                    tr = value.source_trajectory(geometry_times, params)
                    plt.plot(tr.x, tr.y, color=color, lw=1.0, label=name)
                plt.legend(fontsize=7)
            else:
                tr = curve.source_trajectory(sample_times, params)
                plt.plot(tr.x, tr.y, color="#0173B2", lw=1.0)
            plt.xlabel("Trajectory coordinate 1"); plt.ylabel("Trajectory coordinate 2")
            plt.axis("equal")
            save(f"{stem}_geometry.png")

            model_parts = []
            for key, value in args.items():
                if key == "sky":
                    rendered = "sky"
                else:
                    rendered = repr(value)
                model_parts.append(f"{key}={rendered}")
            model_code = ", ".join(model_parts)
            catalogue += ["#### " + label, "", "```python"]
            if observer == "terrestrial":
                catalogue += ["# A narrow, high-magnification caustic feature makes the site offset visible.", "times = np.linspace(7501.37, 7501.46, 600)", "geometry_times = np.linspace(7485.0, 7515.0, 300)"]
            if observer == "space":
                catalogue += ["# Compare the complete event from ground and space.", "times = np.linspace(7470.0, 7530.0, 1800)", "geometry_times = space_geometry_times"]
            catalogue += ["parameters = " + repr(params),
                          "curve = lcbinint.LightCurve(options=options, model=lcbinint.Model(" + model_code + "))"]
            if observer == "space":
                catalogue[-1] = "space_model = lcbinint.Model(" + model_code + ")"
                catalogue += ["parallax_curves = {", "    \"ground\": lcbinint.LightCurve(model=space_model, site=parallax_sites[\"chile\"], options=options),", "    \"space\": lcbinint.LightCurve(model=space_model, site=parallax_sites[\"space\"], options=options),", "}", "magnifications = {name: value(times, parameters) for name, value in parallax_curves.items()}", "trajectories = {name: value.source_trajectory(geometry_times, parameters) for name, value in parallax_curves.items()}", "caustics = parallax_curves[\"ground\"].caustics(parameters)"]
            elif observer == "terrestrial":
                catalogue[-1] = "site_model = lcbinint.Model(" + model_code + ")"
                catalogue += ["terrestrial_curves = {", "    \"Chile\": lcbinint.LightCurve(model=site_model, site=parallax_sites[\"chile\"], options=options),", "    \"Africa\": lcbinint.LightCurve(model=site_model, site=parallax_sites[\"africa\"], options=options),", "}", "magnifications = {name: value(times, parameters) for name, value in terrestrial_curves.items()}", "trajectories = {name: value.source_trajectory(geometry_times, parameters) for name, value in terrestrial_curves.items()}", "caustics = terrestrial_curves[\"Chile\"].caustics(parameters)"]
            elif binary:
                catalogue += ["components = curve.binary_source_components(times, parameters)", "magnification = components.total", "trajectory1 = components.source1.trajectory", "trajectory2 = components.source2.trajectory"]
            else:
                catalogue += ["magnification = curve(times, parameters)", "trajectory = curve.source_trajectory(times, parameters)"]
            if observer not in ("terrestrial", "space"):
                catalogue += [("caustics = curve.caustics(7500.0, parameters)" if orbit else "caustics = curve.caustics(parameters)")]
            catalogue += ["```", "", "```python"]
            if observer == "terrestrial":
                catalogue += ["fig, (curve_ax, difference_ax) = plt.subplots(", "    2, 1, sharex=True, figsize=(3.8, 3.15),", "    gridspec_kw={\"height_ratios\": [3, 1]},", ")", "curve_ax.plot(times, magnifications[\"Chile\"], color=\"#0173B2\", label=\"Chile\")", "curve_ax.plot(times, magnifications[\"Africa\"], color=\"#CC79A7\", label=\"Africa\")", "curve_ax.set_ylabel(\"Magnification\")", "curve_ax.legend(loc=\"upper left\", fontsize=7)", "difference_ax.plot(times, magnifications[\"Africa\"] - magnifications[\"Chile\"], color=\"#6C6C6C\")", "difference_ax.axhline(0.0, color=\"0.75\", lw=0.8)", "difference_ax.set(xlabel=\"Time\", ylabel=\"Africa − Chile\")"]
            else:
                catalogue += ["plt.figure(figsize=(3.8, 2.4))"]
            if observer == "space":
                catalogue += ["plt.plot(times, magnifications[\"ground\"], color=\"#0173B2\", label=\"ground\")", "plt.plot(times, magnifications[\"space\"], color=\"#6C6C6C\", label=\"space\")", "plt.legend(loc=\"upper left\", fontsize=7)"]
            elif observer == "terrestrial":
                pass
            elif binary:
                catalogue += ["plt.plot(times, components.source1.magnification, color=\"#0173B2\", alpha=0.45, label=\"source 1\")", "plt.plot(times, components.source2.magnification, color=\"#029E73\", alpha=0.45, label=\"source 2\")", "plt.plot(times, magnification, color=\"black\", label=\"total\")", "plt.legend(loc=\"upper left\", fontsize=7)"]
            else:
                catalogue += ["plt.plot(times, magnification, color=\"#0173B2\")"]
            if observer != "terrestrial":
                catalogue += ["plt.xlabel(\"Time\"); plt.ylabel(\"Magnification\")"]
            catalogue += ["plt.show()", "```", "",
                          "```python", "plt.figure(figsize=(2.8, 2.7))", "for x, y in zip(caustics.x, caustics.y):", "    plt.plot(x, y, color=\"#6C6C6C\", lw=1.1)"]
            if observer == "space":
                catalogue += ["plt.plot(trajectories[\"ground\"].x, trajectories[\"ground\"].y, color=\"#0173B2\", label=\"ground\")", "plt.plot(trajectories[\"space\"].x, trajectories[\"space\"].y, color=\"#6C6C6C\", label=\"space\")", "plt.legend(fontsize=7)"]
            elif observer == "terrestrial":
                catalogue += ["plt.plot(trajectories[\"Chile\"].x, trajectories[\"Chile\"].y, color=\"#0173B2\", label=\"Chile\")", "plt.plot(trajectories[\"Africa\"].x, trajectories[\"Africa\"].y, color=\"#CC79A7\", label=\"Africa\")", "plt.legend(fontsize=7)"]
            elif binary:
                catalogue += ["plt.plot(trajectory1.x, trajectory1.y, color=\"#0173B2\", label=\"source 1\")", "plt.plot(trajectory2.x, trajectory2.y, color=\"#029E73\", label=\"source 2\")", "plt.legend(fontsize=7)"]
            else:
                catalogue += ["plt.plot(trajectory.x, trajectory.y, color=\"#0173B2\")"]
            catalogue += ["plt.xlabel(\"Trajectory coordinate 1\"); plt.ylabel(\"Trajectory coordinate 2\")", "plt.axis(\"equal\"); plt.show()", "```", "", f"<p><img src=\"figures/{stem}_lightcurve.png\" alt=\"{label} light curve\" width=\"56%\"> <img src=\"figures/{stem}_geometry.png\" alt=\"{label} caustics and trajectories\" width=\"40%\"></p>", ""]
    catalogue += ["[← Higher-order effects](CombinedEffects.md)", ""]
    (OUTPUT.parent / "HigherOrderCombinations.md").write_text("\n".join(catalogue))


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
    higher_order_combination_figures()
    higher_order_catalogue()


if __name__ == "__main__":
    main()
