"""Run the frozen full Polar speed corpus against the e27 parent runner."""
import argparse
from pathlib import Path
import subprocess
import sys

import make_polar_adaptive_vbm_six_panel_20261014 as bench

ROOT = bench.ROOT
NAME = "polar-cost-cut-full-initial2-v5-20261016"
BASELINE = ROOT / "tests/diagnostics/results/recal2026/polar-cost-cut-20261016/parent-e27f2663-runner"


def prepare_runner_with_quadmath():
    command = [
        "/usr/bin/c++", "-O3", "-DNDEBUG", "-std=gnu++17", "-fopenmp",
        "-I" + str(ROOT / "include"), "-I" + str(ROOT / "src"),
        str(bench.RUNNER_SOURCE), "-o", str(bench.RUN / "runner"),
        "-Wl,--start-group", str(bench.BUILD / "liblcbinint_lightcurve.a"),
        str(bench.BUILD / "liblcbinint_magnification.a"), "-Wl,--end-group",
        "-Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib",
        "/home/nunota/.miniconda3/envs/myenv/lib/libgsl.so",
        "/home/nunota/.miniconda3/envs/myenv/lib/libgslcblas.so", "-lm", "-lquadmath",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=NAME)
    args = parser.parse_args()
    output = ROOT / "tests/diagnostics/results/recal2026/polar-speed-20261014" / args.name
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {output}")
    bench.prepare_runner = prepare_runner_with_quadmath
    import polar_speed_optimization as campaign
    sys.argv = [str(Path(campaign.__file__)), "--name", args.name, "--full",
                "--baseline", str(BASELINE)]
    campaign.main()


if __name__ == "__main__":
    main()
