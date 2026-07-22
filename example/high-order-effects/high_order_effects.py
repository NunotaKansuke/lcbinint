"""Small, runnable catalogue of lcbinint physical-model effects.

Run from a source checkout after building the extension:
    PYTHONPATH=build python example/high-order-effects/high_order_effects.py
"""

import numpy as np

import lcbinint


T_REF = 2459000.0
TIMES = T_REF + np.array([-2.0, 0.0, 2.0])
PARAMS = dict(
    t0=T_REF, tE=20.0, u0=0.1, alpha=0.3,
    s=0.9, q=0.02, rho=0.0, piEN=0.1, piEE=0.05,
)


def values(array):
    return np.array2string(np.asarray(array), precision=6, separator=", ")


def main():
    # Binary lens: the same reusable callable is evaluated at three epochs.
    binary = lcbinint.LightCurve()
    print("binary lens:", values(binary(TIMES, PARAMS)))

    # Annual parallax needs an enabled model, sky position, reference epoch,
    # and parallax components in PARAMS.
    annual_model = lcbinint.Model(
        parallax=True,
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        t_ref=T_REF,
    )
    annual = lcbinint.LightCurve(model=annual_model)
    print("annual parallax:", values(annual(TIMES, PARAMS)))

    # Add an explicit ground observatory to enable the terrestrial term.
    ground_model = lcbinint.Model(
        parallax=True,
        terrestrial=True,
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        t_ref=T_REF,
    )
    ground = lcbinint.LightCurve(
        model=ground_model,
        site=lcbinint.obs.Site("ground", -29.0, 70.7),
    )
    print("annual + terrestrial:", values(ground(TIMES, PARAMS)))

    # A space site uses [JD, RA_deg, Dec_deg, distance_AU] rows.
    table = np.array([
        [T_REF - 3.0, 0.0, 0.0, 0.010],
        [T_REF,       0.0, 0.0, 0.012],
        [T_REF + 3.0, 0.0, 0.0, 0.014],
    ])
    space = lcbinint.LightCurve(
        model=annual_model,
        site=lcbinint.obs.Site("space", table),
    )
    print("annual + space site:", values(space(TIMES, PARAMS)))

    # Lens orbital motion evolves the lens geometry at each epoch.
    orbit = lcbinint.LightCurve(
        model=lcbinint.Model(orbital_motion="circular", t_ref=T_REF)
    )
    orbit_params = dict(PARAMS, g1=0.004, g2=0.011, g3=0.006)
    print("circular lens orbit:", values(orbit(TIMES, orbit_params)))

    # Xallarap changes the source trajectory. This compact example uses times
    # relative to its reference epoch to make the parameters easy to read.
    relative_times = np.array([-5.0, 0.0, 5.0])
    xallarap = lcbinint.LightCurve(xallarap="circular_elements")
    xallarap_params = dict(
        t0=0.0, tE=20.0, u0=0.3, alpha=0.5, s=1.0, q=0.1, rho=0.0,
        xi_1=0.25, xi_2=-0.1, period_xa=35.0, inc_xa=1.1,
    )
    print("circular xallarap:", values(xallarap(relative_times, xallarap_params)))

    # A binary source combines two source trajectories with q_source as the
    # source-flux ratio. q_mass instead selects the coupled-xallarap form.
    binary_source = lcbinint.LightCurve(
        source="binary", xallarap="circular_elements"
    )
    print("binary source:", values(binary_source(
        relative_times,
        dict(xallarap_params, q_source=0.5, t0_2=1.0, u0_2=-0.2),
    )))

    # Triple mode is selected on the model and requires q2 > 0 per call.
    triple = lcbinint.LightCurve(lens="triple")
    triple_params = dict(
        t0=0.0, tE=1.0, u0=0.01, alpha=0.5, s=1.0, q=1e-3,
        q2=1e-4, sep2=0.5, ang=1.2, rho=0.0,
    )
    print("triple lens:", values(triple(np.array([-0.1, 0.0, 0.2]), triple_params)))


if __name__ == "__main__":
    main()
