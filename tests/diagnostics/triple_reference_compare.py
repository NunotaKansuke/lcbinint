from __future__ import annotations

import argparse
import ctypes
import dataclasses
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_LEGACY_C = Path("/moao38_7/nunota/binfit/integral/lcbinint.c")


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    source_x: float
    source_y: float
    s: float
    q: float
    q2: float
    sep2: float
    ang: float
    rho: float = 0.0


CASES = [
    Case("planetary_subsystem_left", -0.09263782795758546, -0.03908195790173323, 1.0, 1.0e-3, 1.0e-4, 0.5, 1.2),
    Case("planetary_subsystem_high_mag", -0.00479425538604203, 0.008775825618903728, 1.0, 1.0e-3, 1.0e-4, 0.5, 1.2),
    Case("planetary_subsystem_right", 0.17067435180044185, 0.10449139266017765, 1.0, 1.0e-3, 1.0e-4, 0.5, 1.2),
    Case("moderate_inner_pair", 0.35, -0.22, 0.8, 0.03, 0.02, 0.35, -0.7),
    Case("wide_primary", -0.45, 0.18, 1.4, 0.2, 0.05, 0.7, 2.1),
    Case("finite_near_caustic", 0.0, 0.0, 1.0, 1.0e-3, 1.0e-4, 0.5, 1.2, 1.0e-3),
    Case("finite_hex_candidate", -0.2, 0.0, 1.0, 1.0e-3, 1.0e-4, 0.5, 1.2, 5.0e-3),
]


def load_lcbinint(extension: Path | None):
    if extension is None:
        import lcbinint as module

        return module
    spec = importlib.util.spec_from_file_location("lcbinint", extension)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load lcbinint extension from {extension}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lcbinint"] = module
    spec.loader.exec_module(module)
    return module


def compile_legacy_reference(source: Path) -> ctypes.CDLL | None:
    if not source.exists():
        return None
    wrapper = f"""
#define main lcbinint_legacy_main
#include "{source}"
#undef main

double lcbinint_legacy_amp_point3_xy(
    double sx, double sy, double s, double q, double q2, double sep2, double ang)
{{
    double eps2 = q / (1.0 + q + q2);
    double eps3 = q2 / (1.0 + q + q2);
    double eps1 = 1.0 - eps2 - eps3;
    double eps4 = eps2 + eps3;
    fcomplex zs;
    fcomplex z1;
    fcomplex z2;
    fcomplex z3;
    zs.r = sx;
    zs.i = sy;
    z1.r = -eps4 * s;
    z1.i = 0.0;
    z2.r = eps1 * s + eps3 / eps4 * sep2 * cos(ang);
    z2.i = eps3 / eps4 * sep2 * sin(ang);
    z3.r = eps1 * s - eps2 / eps4 * sep2 * cos(ang);
    z3.i = -eps2 / eps4 * sep2 * sin(ang);
    return amp_point3(zs, z1, z2, z3, eps1, eps2, eps3);
}}
"""
    tmpdir = tempfile.TemporaryDirectory(prefix="lcbinint_legacy_")
    wrapper_path = Path(tmpdir.name) / "legacy_wrapper.c"
    library_path = Path(tmpdir.name) / "legacy_wrapper.so"
    wrapper_path.write_text(wrapper)
    command = [
        "gcc",
        "-O2",
        "-w",
        "-shared",
        "-fPIC",
        f"-I{source.parent}",
        str(wrapper_path),
    ]
    for dependency in ("zroots.c", "laguer.c", "complex.c", "nrutil.c", "option.c"):
        dependency_path = source.parent / dependency
        if dependency_path.exists():
            command.append(str(dependency_path))
    command += ["-lm", "-o", str(library_path)]
    try:
        subprocess.check_call(
            command
        )
    except subprocess.CalledProcessError as exc:
        print(f"legacy_compile_failed={exc}", file=sys.stderr)
        tmpdir.cleanup()
        return None
    try:
        mode = getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_LAZY", 0)
        lib = ctypes.CDLL(str(library_path), mode=mode)
    except OSError as exc:
        print(f"legacy_load_failed={exc}", file=sys.stderr)
        tmpdir.cleanup()
        return None
    lib._tmpdir = tmpdir  # keep the shared object alive
    lib.lcbinint_legacy_amp_point3_xy.argtypes = [ctypes.c_double] * 7
    lib.lcbinint_legacy_amp_point3_xy.restype = ctypes.c_double
    return lib


def lcbinint_value(lcbinint, case: Case, source_bins: int, caustic_bins: int) -> tuple[float, str]:
    options = lcbinint.Options(source_bins=source_bins, caustic_bins=caustic_bins)
    curve = lcbinint.LightCurve(lens="triple_lens", options=options)
    params = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": case.source_y,
        "alpha": 0.0,
        "s": case.s,
        "q": case.q,
        "q2": case.q2,
        "sep2": case.sep2,
        "ang": case.ang,
        "rho": case.rho,
    }
    info = curve.info([case.source_x], params)
    method = info.finite_source_method_names[0] if case.rho > 0.0 else "point_source"
    return float(info.magnifications[0]), method


def old_lens_positions(case: Case) -> list[complex]:
    eps2 = case.q / (1.0 + case.q + case.q2)
    eps3 = case.q2 / (1.0 + case.q + case.q2)
    eps1 = 1.0 - eps2 - eps3
    eps4 = eps2 + eps3
    z1 = complex(-eps4 * case.s, 0.0)
    z2 = complex(eps1 * case.s + eps3 / eps4 * case.sep2 * math.cos(case.ang), eps3 / eps4 * case.sep2 * math.sin(case.ang))
    z3 = complex(eps1 * case.s - eps2 / eps4 * case.sep2 * math.cos(case.ang), -eps2 / eps4 * case.sep2 * math.sin(case.ang))
    return [z1, z2, z3]


def vbm_geometry_params(case: Case, convention: str) -> tuple[list[float], complex]:
    source = complex(case.source_x, case.source_y)
    if convention == "direct":
        return [
            math.log(case.s),
            math.log(case.q),
            math.log(max(case.rho, 1.0e-12)),
            0.0,
            0.0,
            math.log(case.sep2),
            math.log(case.q2),
            case.ang,
        ], source
    if convention == "legacy_edges":
        z1, z2, z3 = old_lens_positions(case)
        v12 = z2 - z1
        v13 = z3 - z1
        angle12 = math.atan2(v12.imag, v12.real)
        psi = math.atan2(v13.imag, v13.real) - angle12
        eps2 = case.q / (1.0 + case.q + case.q2)
        eps1 = 1.0 - eps2 - case.q2 / (1.0 + case.q + case.q2)
        com12 = (eps1 * z1 + eps2 * z2) / (eps1 + eps2)
        return [
            math.log(abs(v12)),
            math.log(case.q),
            math.log(max(case.rho, 1.0e-12)),
            0.0,
            0.0,
            math.log(abs(v13)),
            math.log(case.q2),
            psi,
        ], (source - com12) * complex(math.cos(-angle12), math.sin(-angle12))
    raise ValueError(f"unknown VBM convention: {convention}")


def vbm_value(case: Case, convention: str, tol: float, include_finite: bool) -> float | None:
    if case.rho > 0.0 and not include_finite:
        return None
    try:
        import VBMicrolensing
    except ImportError:
        return None
    vbb = VBMicrolensing.VBMicrolensing()
    vbb.Tol = tol
    vbb.RelTol = 0.0
    geometry_params, vbm_source = vbm_geometry_params(case, convention)
    # VBM's static light-curve trajectory reports y=(-time,-u0) for alpha=0.
    # Use that sign convention explicitly so the returned source coordinate is
    # the intended point in the selected VBM lens frame.
    params = [
        geometry_params[0],
        geometry_params[1],
        -vbm_source.imag,
        0.0,
        geometry_params[2],
        geometry_params[3],
        geometry_params[4],
        geometry_params[5],
        geometry_params[6],
        geometry_params[7],
    ]
    result = vbb.TripleLightCurve(params, [-vbm_source.real])
    return float(np.asarray(result[0], dtype=float)[0])


def rel_error(value: float | None, reference: float) -> float | None:
    if value is None or reference == 0.0:
        return None
    return value / reference - 1.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare triple-lens lcbinint against the legacy amp_point3 path and "
            "optionally VBMicrolensing. VBM triple-lens parameter conventions are "
            "not treated as authoritative here."
        )
    )
    parser.add_argument("--extension", type=Path, default=None)
    parser.add_argument("--legacy-c", type=Path, default=Path(os.environ.get("LCBININT_LEGACY_C", DEFAULT_LEGACY_C)))
    parser.add_argument("--source-bins", type=int, default=12)
    parser.add_argument("--caustic-bins", type=int, default=1400)
    parser.add_argument("--vbm-convention", choices=["direct", "legacy_edges"], default="legacy_edges")
    parser.add_argument("--vbm-tol", type=float, default=1.0e-5)
    parser.add_argument(
        "--vbm-finite",
        action="store_true",
        help="Also call VBM for finite-source triple cases. This is off by default because some VBM triple finite calls can crash the interpreter.",
    )
    args = parser.parse_args()

    lcbinint = load_lcbinint(args.extension)
    legacy = compile_legacy_reference(args.legacy_c)
    print(f"legacy_c={args.legacy_c if legacy is not None else 'not available'}")
    print(f"vbm_convention={args.vbm_convention}")
    print("case,method,lcbinint,legacy,rel_vs_legacy,vbm,rel_vs_vbm")
    for case in CASES:
        actual, method = lcbinint_value(lcbinint, case, args.source_bins, args.caustic_bins)
        legacy_value = None
        if legacy is not None and case.rho == 0.0:
            legacy_value = legacy.lcbinint_legacy_amp_point3_xy(
                case.source_x, case.source_y, case.s, case.q, case.q2, case.sep2, case.ang
            )
        vbm = vbm_value(case, args.vbm_convention, args.vbm_tol, args.vbm_finite)
        legacy_rel = rel_error(actual, legacy_value) if legacy_value is not None else None
        vbm_rel = rel_error(actual, vbm) if vbm is not None else None
        print(
            f"{case.name},{method},{actual:.16g},"
            f"{'' if legacy_value is None else f'{legacy_value:.16g}'},"
            f"{'' if legacy_rel is None else f'{legacy_rel:.6e}'},"
            f"{'' if vbm is None else f'{vbm:.16g}'},"
            f"{'' if vbm_rel is None else f'{vbm_rel:.6e}'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
