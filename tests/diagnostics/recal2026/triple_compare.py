"""The reduced triple-lens comparison.

The binary study is the campaign's subject; this is the check that its answers
do not stop being true when a third body is added.  It is deliberately small.
The questions it asks are the two that would change how the paper is written:

1. Does lcbinint deliver its requested accuracy on triple lenses, and at what
   cost against VBMicrolensing at a matched accuracy?
2. Does the binary grid-switch finding carry over -- specifically, is the
   Cartesian inverse-ray grid the odd one out near tangency here too?

Everything is driven through the *trajectory* API on both sides.  That is not
laziness: the binary sweeps drive engines positionally and therefore had to
measure a frame convention before they could compare anything, and a mistake
there is invisible in the output.  ``TripleLightCurve`` and ``LightCurve.info``
both take trajectory parameters, so the only convention in play is the
parameter ordering, which :func:`verify_frames` pins against measured values
before any sweep runs.

The reference is the median of three witnesses -- lcbinint's Cartesian grid,
lcbinint's polar grid and VBM's contour integrator -- rather than a nominated
one.  The binary study found the Cartesian grid returning a certified,
resolution-independent 1.6% error near tangency, so a reference defined as
"the finest certified Cartesian value" would inherit that error and report it
as everyone else's.  A median of three survives one dissenter; the spread is
carried alongside so a row where two dissent is reported rather than believed.

VBMicrolensing offers three root-finding methods for more than two lenses and
all three are measured, because they are not a detail: on the smoke case they
agree to 3e-13 but span a factor of 3.5 in cost, and the one VBM selects when
nothing is said -- ``Nopoly`` -- is the slowest.  Reporting only the default
would understate VBM by that factor and make the comparison worthless.

VBMicrolensing 5.5 segfaults on some high-magnification triple geometries, so
each case runs in its own subprocess and a crash is recorded as a result
instead of taking the sweep down with it.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Requested accuracies, matched across the two engines.  The same three the
# binary study calibrates, so the two tables can be read side by side.
TOLERANCES = (1.0e-2, 1.0e-3, 1.0e-4)

PROFILES = {"uniform": 0.0, "linear": 0.5}

# Epochs per light curve, and the span they cover in units of tE.  The span is
# centred on closest approach because that is where the caustic structure is;
# a full-tE span would spend most of its epochs in the point-source wings and
# would measure the point-source exit rather than the finite-source path.
BLOCK_EPOCHS = 48
BLOCK_SPAN = 0.6

# Timed passes per measurement, after one warm-up pass.  Matches the binary
# harness so the two speed tables mean the same thing.
TIMING_REPEAT = 3

# The witness settings.  ``REFERENCE_BINS`` is the top of the binary study's
# resolution ladder, and the contour tolerance is far below the tightest
# accuracy being calibrated so that a dissent at 1e-4 cannot be the witness's
# own truncation.
REFERENCE_BINS = 400
REFERENCE_CONTOUR_RELATIVE = 1.0e-9
CONTOUR_ABSOLUTE_FLOOR = 1.0e-12

# VBM's multi-lens root-finding methods, in the order they are reported.
# ``Nopoly`` is what VBM uses when ``SetMethod`` is never called.
VBM_METHODS = ("Singlepoly", "Multipoly", "Nopoly")

# Which method supplies the contour witness of the reference, in preference
# order; the first one that survives the case is used.  ``Nopoly`` is last
# despite being VBM's default, because measured over this corpus it segfaults
# on 13 of 32 geometries while the two polynomial methods complete all 32.  A
# reference witness that dies on 41% of the corpus is not a reference.  The
# three were measured to agree to 3e-13 and 7e-12 where all three survive, so
# the choice costs nothing in accuracy.
REFERENCE_CONTOUR_PREFERENCE = ("Multipoly", "Singlepoly", "Nopoly")
REFERENCE_CONTOUR_METHOD = REFERENCE_CONTOUR_PREFERENCE[0]

MODULE = "tests.diagnostics.recal2026.triple_compare"

# What ``point_source_threshold`` must be set to in order to force an
# inverse-ray grid on a triple lens.  Zero is the binary harness's answer and is
# actively wrong here: ``triple_mag`` exits to the point source when the caustic
# distance *exceeds* ``point_source_threshold * rho``, so zero sends every epoch
# that is not exactly on the caustic to the point source.  Measured, a forced
# witness at nbin=200 returned ``methods=['point_source']`` in 0.001 s at every
# resolution until this was raised.  A large finite value disables the exit
# without disabling the caustic-distance refinement the routing still wants.
FORCE_POINT_SOURCE_THRESHOLD = 1.0e6


# --------------------------------------------------------------------------
# Parameter conventions, measured
# --------------------------------------------------------------------------

# VBM's triple parameter vector, in its own order:
#   [log s12, log q2, u0, alpha, log rho, log tE, t0, log s13, log q3, psi]
# lcbinint names the same five geometry numbers s, q, sep2, q2, ang.  The
# mapping is measured by :func:`verify_frames`, not trusted.
_FRAME_PROBES = (
    # (s, q, q2, sep2, ang, rho, u0, alpha, t)
    (1.00, 2.0e-1, 1.0e-2, 0.70, 1.20, 0.010, 0.05, 0.60, 0.03),
    (1.30, 5.0e-2, 1.0e-3, 1.10, -0.80, 0.005, -0.02, 2.10, -0.05),
    (0.80, 5.0e-1, 1.0e-1, 0.40, 2.50, 0.020, 0.11, 1.00, 0.07),
)

# How far the two engines were measured to agree on the probes when this module
# was written.  A convention that silently changes shows up here as a hard
# failure rather than as an accuracy result.
FRAME_TOLERANCE = 1.0e-5


def vbm_parameters(s, q, q2, sep2, ang, rho, u0, alpha):
    """lcbinint's triple parameter names, in VBM's vector order."""
    return [math.log(s), math.log(q), u0, alpha, math.log(rho),
            0.0, 0.0, math.log(sep2), math.log(q2), ang]


def lcbinint_curve(profile_c, *, reltol=0.0, grid=None, nbin="auto",
                   force_grid=False, max_source_bins=None):
    import lcbinint

    options = {"coordinates": "vbm", "nbin": nbin}
    if grid is not None:
        options["inverse_ray_grid"] = grid
    if force_grid:
        # Same reasoning as the binary harness: naming a grid does not get one,
        # because the point-source and hexadecapole exits are taken before the
        # grid is chosen.  A witness that silently returns a hexadecapole value
        # is not a witness for the grid it was asked about.  The point-source
        # exit is disabled by raising its threshold rather than zeroing it --
        # see FORCE_POINT_SOURCE_THRESHOLD, where the sense is inverted
        # relative to the binary path.
        options["point_source_threshold"] = FORCE_POINT_SOURCE_THRESHOLD
        options["hexadecapole_threshold"] = 0.0
        options["adaptive_hex_threshold"] = 0.0
    if reltol:
        options["reltol"] = reltol
    if max_source_bins is not None:
        options["max_source_bins"] = max_source_bins
    return lcbinint.LightCurve(lens="triple", options=lcbinint.Options(**options))


def lcbinint_block(curve, case, times, profile_c, repeat=1):
    """One light curve, timed as a block, with per-epoch diagnostics.

    With ``repeat`` above one the first call is a warm-up and the reported time
    is the median of the rest, which is what the binary harness's
    ``time_light_curve_block`` measures and therefore what the triple numbers
    have to measure to sit in the same table.  The distinction is not small
    here: a triple caustic cache costs more to build than a binary one, and on
    the smoke case the cold pass is several times the warm pass.
    """
    times = np.asarray(times, dtype=float)

    def once():
        started = time.perf_counter()
        info = curve.info(
            times, t0=0.0, tE=1.0,
            u0=case["u0"], alpha=case["alpha"],
            s=case["s"], q=case["q"], rho=case["rho"],
            q2=case["q2"], sep2=case["sep2"], ang=case["ang"],
            limb_darkening_c=profile_c)
        return time.perf_counter() - started, info

    elapsed, info = once()
    if repeat > 1:
        samples = [once()[0] for _ in range(repeat)]
        elapsed = float(np.median(samples))
    return {
        "magnifications": np.asarray(
            info.finite_source_magnifications, dtype=float).ravel().tolist(),
        "error_estimates": np.asarray(
            info.finite_source_error_estimates, dtype=float).ravel().tolist(),
        "converged": np.asarray(
            info.finite_source_converged).ravel().astype(bool).tolist(),
        "methods": [str(name) for name in
                    np.asarray(info.finite_source_method_names).ravel()],
        "point_magnifications": np.asarray(
            info.point_source_magnifications, dtype=float).ravel().tolist(),
        "seconds": elapsed,
        "seconds_per_epoch": elapsed / len(times),
    }


def vbm_block(case, times, profile_c, relative_tolerance,
              method=REFERENCE_CONTOUR_METHOD, repeat=1):
    """VBM's triple light curve at a requested relative tolerance.

    ``method`` names one of ``VBM_METHODS``.  It is always passed explicitly,
    including for the default, so that a change in VBM's own default shows up
    as a version difference rather than silently moving what is being timed.
    """
    import VBMicrolensing

    engine = VBMicrolensing.VBMicrolensing()
    engine.SetMethod(getattr(VBMicrolensing.VBMicrolensing, method))
    engine.RelTol = relative_tolerance
    engine.Tol = CONTOUR_ABSOLUTE_FLOOR
    engine.a1 = profile_c
    parameters = vbm_parameters(
        case["s"], case["q"], case["q2"], case["sep2"], case["ang"],
        case["rho"], case["u0"], case["alpha"])
    times = list(times)

    def once():
        started = time.perf_counter()
        values = engine.TripleLightCurve(parameters, times)[0]
        return time.perf_counter() - started, values

    elapsed, values = once()
    if repeat > 1:
        samples = [once()[0] for _ in range(repeat)]
        elapsed = float(np.median(samples))
    return {
        "magnifications": [float(v) for v in values],
        "method": method,
        "seconds": elapsed,
        "seconds_per_epoch": elapsed / len(times),
    }


def vbm_requests(profiles, tolerances):
    """Every block one VBM method has to produce for a case, keyed by name.

    The reference witness and the timed rows are gathered into one request list
    so that a method is started once per case rather than once per block: VBM's
    triple setup is not free, and the timing below is of the light curve, not
    of the import.
    """
    out = []
    for profile_name, profile_c in profiles:
        out.append({"key": f"{profile_name}|reference",
                    "profile_c": profile_c,
                    "relative_tolerance": REFERENCE_CONTOUR_RELATIVE,
                    "repeat": 1})
        for tolerance in tolerances:
            out.append({"key": f"{profile_name}|{tolerance:g}",
                        "profile_c": profile_c,
                        "relative_tolerance": tolerance,
                        "repeat": TIMING_REPEAT})
    return out


def vbm_method_blocks(case, times, requests, *, timeout, core=-1):
    """Every VBM method's blocks for one case, each method in its own process.

    Isolating per *case* is not enough once the crash is known to be
    method-specific: it would discard the two methods that worked, and
    lcbinint's own rows with them, every time VBM's default died.  Isolating
    per method loses only the method that crashed, and a crash then becomes a
    reported property of that method rather than a hole in the corpus.
    """
    table = {}
    with tempfile.TemporaryDirectory() as workdir:
        for method in VBM_METHODS:
            target = Path(workdir) / f"vbm-{method}.json"
            payload = json.dumps({
                "case": case, "times": list(np.asarray(times, dtype=float)),
                "method": method, "requests": requests})
            command = [sys.executable, "-m", MODULE,
                       "--vbm-child", payload,
                       "--child-output", str(target)]
            if core >= 0:
                command += ["--core", str(core)]
            entry = {"method": method, "blocks": {}}
            try:
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=timeout)
                if result.returncode == 0 and target.exists():
                    entry["blocks"] = json.loads(target.read_text())
                else:
                    entry["failure"] = {
                        "returncode": result.returncode,
                        # -11 is SIGSEGV, which is the expected failure here
                        # and is why this runs in a subprocess at all.
                        "signal": -result.returncode
                        if result.returncode < 0 else None,
                        "stderr": result.stderr[-2000:],
                    }
            except subprocess.TimeoutExpired:
                entry["failure"] = {"returncode": None, "timed_out": timeout}
            table[method] = entry
    return table


def reference_contour(table, key):
    """The first preferred VBM method that produced the reference block."""
    for method in REFERENCE_CONTOUR_PREFERENCE:
        block = table.get(method, {}).get("blocks", {}).get(key)
        if block is not None and all(
                math.isfinite(value) for value in block["magnifications"]):
            return block
    return None


def verify_frames(tolerance=FRAME_TOLERANCE):
    """Refuse to run if the two engines stop describing the same geometry."""
    report = []
    for (s, q, q2, sep2, ang, rho, u0, alpha, t) in _FRAME_PROBES:
        case = {"s": s, "q": q, "q2": q2, "sep2": sep2, "ang": ang,
                "rho": rho, "u0": u0, "alpha": alpha}
        native = lcbinint_block(
            lcbinint_curve(0.0, reltol=1.0e-6, force_grid=True),
            case, [t], 0.0)["magnifications"][0]
        contour = vbm_block(case, [t], 0.0, 1.0e-8)["magnifications"][0]
        gap = abs(native - contour) / max(abs(contour), 1.0)
        report.append({"case": case, "t": t, "lcbinint": native,
                       "vbm": contour, "relative_gap": gap})
        if not math.isfinite(gap) or gap > tolerance:
            raise SystemExit(
                f"triple frame convention check failed: {gap:.3e} > "
                f"{tolerance:.0e} at s={s} q={q} q2={q2} sep2={sep2} ang={ang}")
    return report


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

def make_cases(count, seed):
    """Triple geometries with a trajectory across the caustic structure.

    The third body is drawn over four decades in mass and around the second
    body at a separation of its own, so the corpus spans the range from "a
    binary with a perturbation" to three comparable masses.  ``u0`` is drawn
    log-uniformly and small: a linearly sampled impact parameter puts almost
    every trajectory in the point-source wings, which measures the wrong exit.
    """
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        s = float(np.exp(rng.uniform(math.log(0.6), math.log(2.0))))
        q = float(np.exp(rng.uniform(math.log(1.0e-3), math.log(1.0))))
        q2 = float(np.exp(rng.uniform(math.log(1.0e-5), math.log(1.0e-1))))
        sep2 = float(np.exp(rng.uniform(math.log(0.3), math.log(1.6))))
        ang = float(rng.uniform(-math.pi, math.pi))
        rho = float(np.exp(rng.uniform(math.log(1.0e-4), math.log(3.0e-2))))
        u0 = float(np.exp(rng.uniform(math.log(1.0e-3), math.log(3.0e-1))))
        u0 *= 1.0 if rng.random() < 0.5 else -1.0
        alpha = float(rng.uniform(0.0, 2.0 * math.pi))
        cases.append({
            "case_id": index, "s": s, "q": q, "q2": q2, "sep2": sep2,
            "ang": ang, "rho": rho, "u0": u0, "alpha": alpha,
        })
    return cases


# --------------------------------------------------------------------------
# One case
# --------------------------------------------------------------------------

def witness_reference(case, times, profile_c, contour):
    """Three independent values per epoch, and their spread.

    The value is the median rather than a nominated witness.  See the module
    docstring: the binary study found the Cartesian grid capable of a
    certified, resolution-independent error, so a reference that trusts one
    witness by name can be wrong with full confidence.
    """
    cartesian = lcbinint_block(
        lcbinint_curve(profile_c, grid="cartesian", nbin=REFERENCE_BINS,
                       force_grid=True, max_source_bins=REFERENCE_BINS),
        case, times, profile_c)
    polar = lcbinint_block(
        lcbinint_curve(profile_c, grid="polar", nbin=REFERENCE_BINS,
                       force_grid=True, max_source_bins=REFERENCE_BINS),
        case, times, profile_c)
    # ``contour`` is supplied by the caller from the isolated VBM run.  One
    # contour witness is enough: the three VBM methods were measured to agree
    # to 3e-13 and 7e-12 wherever all three survive, so a second one would add
    # a vote without adding independence.

    values, spreads, dissenters, counts = [], [], [], []
    stack = zip(cartesian["magnifications"], polar["magnifications"],
                contour["magnifications"], polar["methods"])
    for cart, pol, cont, polar_method in stack:
        offered = [("cartesian", cart), ("contour", cont)]
        # The polar witness only counts on epochs where it actually ran polar.
        # Inside one source radius of the caustic ``triple_mag`` routes an
        # explicit polar request to the Cartesian grid, and a Cartesian value
        # wearing the polar witness's name would give lcbinint's one grid two
        # of the three votes -- precisely at tangency, where the binary study
        # found that grid to be the party in error.  Dropping it leaves an
        # honest two-witness disagreement, which the spread gate then handles.
        if polar_method == "inverse_ray_polar":
            offered.insert(1, ("polar", pol))
        usable = [(name, value) for name, value in offered
                  if math.isfinite(value)]
        counts.append(len(usable))
        if len(usable) < 2:
            values.append(float("nan"))
            spreads.append(float("inf"))
            dissenters.append("incomplete")
            continue
        numbers = [value for _, value in usable]
        middle = float(np.median(numbers))
        scale = max(abs(middle), 1.0)
        values.append(middle)
        spreads.append((max(numbers) - min(numbers)) / scale)
        gaps = [abs(value - middle) / scale for value in numbers]
        dissenters.append(usable[int(np.argmax(gaps))][0])

    return {
        "values": values,
        "spreads": spreads,
        "dissenters": dissenters,
        "witness_counts": counts,
        "cartesian": cartesian["magnifications"],
        "polar": polar["magnifications"],
        "contour": contour["magnifications"],
        "contour_method": contour["method"],
        # Which route each grid witness actually took.  A witness that fell to
        # the point source is not a witness for its grid, and this is where
        # that shows: the first version of this harness had every witness on
        # 'point_source' and produced a reference nobody could have used.
        "methods": {"cartesian": sorted(set(cartesian["methods"])),
                    "polar": sorted(set(polar["methods"]))},
        "seconds": {"cartesian": cartesian["seconds"],
                    "polar": polar["seconds"],
                    "contour": contour["seconds"]},
    }


# How much of the tolerance being judged the reference is allowed to consume.
# An epoch whose witnesses disagree by more than this fraction of the target
# cannot resolve a pass from a failure at that target.
SPREAD_BUDGET = 0.1


def achieved_error(values, reference, tolerance):
    """Worst and median relative gap from the reference, over usable epochs.

    An epoch whose three witnesses disagree by more than ``SPREAD_BUDGET`` of
    the tolerance being judged cannot judge it, so those epochs are dropped
    rather than counted as either a pass or a failure.  How many were dropped
    is reported: a row that judged four of its forty-eight epochs is a row
    whose verdict is about the reference, not about the engine, and the count
    is what makes that visible instead of hiding it inside a worst-case number.
    """
    gate = SPREAD_BUDGET * tolerance
    gaps = []
    dropped_unusable = 0
    dropped_spread = 0
    for value, truth, spread in zip(values, reference["values"],
                                    reference["spreads"]):
        if not math.isfinite(value) or not math.isfinite(truth):
            dropped_unusable += 1
            continue
        if not math.isfinite(spread):
            dropped_unusable += 1
            continue
        if spread > gate:
            dropped_spread += 1
            continue
        gaps.append((abs(value - truth) / max(abs(truth), 1.0), spread))
    summary = {
        "tolerance": tolerance,
        "spread_gate": gate,
        "dropped_unusable": dropped_unusable,
        "dropped_spread": dropped_spread,
    }
    if not gaps:
        return {**summary, "worst": float("nan"), "median": float("nan"),
                "epochs": 0}
    errors = np.array([gap for gap, _ in gaps])
    return {
        **summary,
        "worst": float(errors.max()),
        "median": float(np.median(errors)),
        "epochs": len(errors),
        "reference_spread_worst": float(max(spread for _, spread in gaps)),
    }


def run_case(case, *, profiles, tolerances, timeout=1800.0, core=-1):
    times = np.linspace(-0.5 * BLOCK_SPAN, 0.5 * BLOCK_SPAN, BLOCK_EPOCHS)
    # The per-method budget has to leave room for lcbinint inside the same
    # outer timeout, or a slow VBM method takes the case down by starving the
    # rows it was being compared against.
    table = vbm_method_blocks(case, times,
                              vbm_requests(profiles, tolerances),
                              timeout=timeout / (len(VBM_METHODS) + 1),
                              core=core)
    rows = []
    for profile_name, profile_c in profiles:
        contour = reference_contour(table, f"{profile_name}|reference")
        if contour is None:
            # No usable contour witness leaves two witnesses that share a
            # codebase, which is not a reference.  Say so rather than publish
            # a number resting on lcbinint agreeing with itself.
            rows.append({
                "profile": profile_name,
                "limb_darkening_c": profile_c,
                "reference_unavailable": {
                    method: entry.get("failure") for method, entry
                    in table.items()},
            })
            continue
        reference = witness_reference(case, times, profile_c, contour)
        row = {
            "profile": profile_name,
            "limb_darkening_c": profile_c,
            "reference": reference,
            "engines": [],
        }
        for tolerance in tolerances:
            native = lcbinint_block(
                lcbinint_curve(profile_c, reltol=tolerance), case, times,
                profile_c, repeat=TIMING_REPEAT)
            row["engines"].append({
                "engine": "lcbinint_auto",
                "knob": tolerance,
                "seconds_per_epoch": native["seconds_per_epoch"],
                "error": achieved_error(native["magnifications"], reference,
                                        tolerance),
                "methods": sorted(set(native["methods"])),
                "converged_epochs": int(sum(native["converged"])),
                "self_reported_worst": float(
                    np.nanmax(native["error_estimates"])),
            })
            for vbm_method in VBM_METHODS:
                block = table[vbm_method]["blocks"].get(
                    f"{profile_name}|{tolerance:g}")
                entry = {
                    "engine": f"vbm_{vbm_method.lower()}",
                    "vbm_method": vbm_method,
                    "knob": tolerance,
                }
                if block is None:
                    entry["failure"] = table[vbm_method].get("failure")
                else:
                    entry["seconds_per_epoch"] = block["seconds_per_epoch"]
                    entry["error"] = achieved_error(
                        block["magnifications"], reference, tolerance)
                row["engines"].append(entry)
        row["magnification_range"] = [
            float(np.nanmin(reference["values"])),
            float(np.nanmax(reference["values"])),
        ]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Driver: one subprocess per case, so a VBM crash is data rather than the end
# --------------------------------------------------------------------------

def _run_one_case_child(case_json, output_path, profiles, tolerances,
                        timeout, core):
    case = json.loads(case_json)
    started = time.perf_counter()
    rows = run_case(case, profiles=profiles, tolerances=tolerances,
                    timeout=timeout, core=core)
    Path(output_path).write_text(json.dumps({
        "case": case, "rows": rows,
        "seconds": time.perf_counter() - started,
    }))


def _run_vbm_child(payload_json, output_path):
    """One VBM method's whole block list, in a process that may die."""
    payload = json.loads(payload_json)
    blocks = {}
    for request in payload["requests"]:
        blocks[request["key"]] = vbm_block(
            payload["case"], payload["times"], request["profile_c"],
            request["relative_tolerance"], method=payload["method"],
            repeat=request["repeat"])
    Path(output_path).write_text(json.dumps(blocks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Not required: the child modes below write to --child-output and never
    # touch the sweep directory.
    parser.add_argument("--output", default="")
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--profiles", default="uniform,linear")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--core", type=int, default=-1)
    parser.add_argument("--cores", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--child", default="")
    parser.add_argument("--vbm-child", default="")
    parser.add_argument("--child-output", default="")
    arguments = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    profiles = [(name.strip(), PROFILES[name.strip()])
                for name in arguments.profiles.split(",") if name.strip()]

    if arguments.core >= 0:
        try:
            os.sched_setaffinity(0, {arguments.core})
        except OSError:
            pass

    if arguments.vbm_child:
        _run_vbm_child(arguments.vbm_child, arguments.child_output)
        return

    if arguments.child:
        _run_one_case_child(arguments.child, arguments.child_output,
                            profiles, TOLERANCES, arguments.timeout,
                            arguments.core)
        return

    if not arguments.output:
        parser.error("--output is required unless running as a child")
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    print("verifying triple parameter conventions", flush=True)
    frames = verify_frames()
    for entry in frames:
        print(f"  gap {entry['relative_gap']:.2e}", flush=True)
    (output / "frames.json").write_text(json.dumps(frames, indent=1))

    cases = make_cases(arguments.cases, arguments.seed)
    (output / "manifest.json").write_text(json.dumps({
        "seed": arguments.seed, "cases": arguments.cases,
        "tolerances": list(TOLERANCES), "block_epochs": BLOCK_EPOCHS,
        "block_span": BLOCK_SPAN, "reference_bins": REFERENCE_BINS,
        "reference_contour_relative": REFERENCE_CONTOUR_RELATIVE,
        "reference_contour_preference": list(REFERENCE_CONTOUR_PREFERENCE),
        "vbm_methods": list(VBM_METHODS),
        "spread_budget": SPREAD_BUDGET,
        "profiles": [name for name, _ in profiles],
    }, indent=1))

    cores = ([int(part) for part in arguments.cores.split(",")]
             if arguments.cores else [])
    pending = [case for case in cases
               if not (output / f"case-{case['case_id']:05d}.json").exists()]

    def run(index_and_case):
        index, case = index_and_case
        target = output / f"case-{case['case_id']:05d}.json"
        command = [
            sys.executable, "-m", "tests.diagnostics.recal2026.triple_compare",
            "--output", str(output), "--child", json.dumps(case),
            "--child-output", str(target),
            "--profiles", arguments.profiles,
            # Forwarded, not defaulted: the child divides this budget among the
            # VBM methods, and without it the per-method budget silently
            # reverted to a quarter of argparse's default.  Two cases were
            # recorded as VBM timeouts that were only the harness's own clock.
            "--timeout", str(arguments.timeout),
        ]
        if cores:
            command += ["--core", str(cores[index % len(cores)])]
        elif arguments.core >= 0:
            command += ["--core", str(arguments.core)]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=arguments.timeout)
            failed = result.returncode != 0 or not target.exists()
            detail = {"returncode": result.returncode,
                      "stderr": result.stderr[-4000:]}
        except subprocess.TimeoutExpired:
            # A timeout is a result too: a geometry VBM cannot finish is a
            # geometry the comparison has to report, not one it may drop.
            failed = True
            detail = {"returncode": None, "timed_out": arguments.timeout}
        if failed:
            target.write_text(json.dumps(
                {"case": case, "crashed": True, **detail}))
        return case["case_id"], failed

    # Threads, not processes: each one only waits on a subprocess, and the
    # subprocess is what carries the affinity and the work.
    started = time.perf_counter()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        for case_id, failed in pool.map(run, enumerate(pending)):
            done += 1
            status = "CRASHED" if failed else "done"
            print(f"[{done}/{len(pending)}] case {case_id} {status} "
                  f"(elapsed {time.perf_counter()-started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
