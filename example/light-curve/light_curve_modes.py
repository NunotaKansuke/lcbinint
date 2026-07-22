import numpy as np

import lcbinint


T_REF = 2459000.0
TIMES = T_REF + np.linspace(-20.0, 20.0, 401)
PARAMS = {
    "t0": T_REF,
    "tE": 30.0,
    "u0": 0.02,
    "alpha": 0.5,
    "s": 0.95,
    "q": 1.0e-2,
    "rho": 5.0e-3,
    "piEN": 0.15,
    "piEE": -0.08,
}


def summarize(label, curve):
    info = curve.info(TIMES, PARAMS)
    methods, counts = np.unique(info.finite_source_method_names, return_counts=True)
    method_counts = dict(zip(map(str, methods), counts.tolist()))
    print(
        f"{label}: min={min(info.magnifications):.6g} "
        f"max={max(info.magnifications):.6g} methods={method_counts} "
        f"all_converged={info.all_converged}"
    )
    return info


def main():
    options = lcbinint.Options(
        coordinates="vbm",
        nbin="auto",
        inverse_ray_grid="auto",
    )

    static = lcbinint.LightCurve(options=options)
    summarize("static/geocentric", static)

    model = lcbinint.Model(
        parallax=True,
        terrestrial=True,
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        t_ref=T_REF,
    )
    ground = lcbinint.LightCurve(
        model=model,
        options=options,
        site=lcbinint.obs.Site("ground", -29.0, 70.7),
    )
    info = summarize("annual+terrestrial parallax", ground)
    trajectory = ground.source_trajectory(TIMES, PARAMS)
    print(
        "trajectory endpoints: "
        f"({trajectory.x[0]:.6g}, {trajectory.y[0]:.6g}) -> "
        f"({trajectory.x[-1]:.6g}, {trajectory.y[-1]:.6g})"
    )
    assert len(info.magnifications) == TIMES.size


if __name__ == "__main__":
    main()
