#!/usr/bin/env python3
"""The JAX and microlux axes, placed on the speed sweep's own Pareto plane.

This does not re-derive anything ``sweep_speed`` already established.  It reads
that sweep's blocks and reuses their references verbatim, so a microlux point
and an lcbinint point in the same row are measured against the same number.
Rebuilding the references here would cost more than the timing does -- each one
is a converged ladder plus a contour evaluation -- and, worse, two independently
built references differing at the 1e-6 level would show up as an engine
disagreement rather than as reference noise.

Timing lives in its own process per worker because JAX is in it: importing JAX
reserves memory and starts its own thread pool, and a traced function compiles
on first call.  Compilation is warmed and reported separately -- a fit evaluates
thousands of light curves against one compilation, so folding it into the
per-epoch cost would describe a use nobody has.

The run is split into one process pass per engine setting, and that split is
what makes it finish at all.  A worker that held every setting at once died in
XLA's JIT with ``Unable to allocate section memory``, at twenty-four workers, at
sixteen, and at one worker on one block -- so it was never the pool and never
the machine's memory, which stood at a hundred and seventy gigabytes free.  It
is ``vm.max_map_count``, 65530 here.  Counted one family at a time, microlux
costs about 4300 mappings and 320 MB for each distinct sampling strategy it
compiles, and the lcbinint JAX path about 550 each; the twenty settings a single
worker used to hold came to roughly 39600 mappings before a row had been timed.
One setting per pass keeps a worker under ten thousand.

The cost of the split is that censoring can no longer look at the previous
setting in the same row, because that setting now runs in a different process
and often on a different day.  ``--censor-from`` reads the previous pass's
output instead, which is the same rule -- stop this row's ladder once the looser
setting already blew the per-epoch budget -- evaluated across passes.

Run this when ``sweep_speed`` is *not* running.  Both are timing measurements
and they would contend for memory bandwidth and shared cache; the numbers would
be a measurement of the machine's load rather than of the engines.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .engines import (DEFAULT_BLOCK, PROFILES, lcbinint_fixed,
                      time_light_curve_block)
from .sweep_speed import BLOCK_SPAN_IN_RADII, TIMED_TOLERANCES, achieved_error

# Two Cartesian buckets that ``sweep_speed`` also timed, re-timed here so the
# two runs can be put on one axis.  JAX will not share a machine with 24 of
# itself -- each worker holds its compiled executables for the life of the
# process, and the pool has to be smaller than the speed sweep's -- so these
# timings are taken under lighter contention than the ones they are compared
# against.  That difference is a number, and this is how it gets measured
# instead of assumed: the control's ratio to its stored counterpart is the
# scale factor between the runs, per row and per core.
CONTROL_BUCKETS = (24, 50)


def _block_times(row):
    """The epochs ``sweep_speed`` timed, rebuilt from what it recorded.

    Reconstruction rather than storage: the blocks are a deterministic function
    of the position and radius the row already carries, and a stored array would
    be one more thing that can drift out of step with the sweep that made it.
    """
    span = BLOCK_SPAN_IN_RADII * row["rho"]
    return np.linspace(row["x"] - 0.5 * span, row["x"] + 0.5 * span,
                       row.get("block_epochs", DEFAULT_BLOCK))


def _references(row):
    """The stored reference epochs, in the shape ``achieved_error`` wants."""
    return {
        int(index): entry
        for index, entry in (row.get("references") or {}).items()
        if entry.get("status") == "ok"
    }


# One engine per (kind, knob, profile) for the life of the worker.  Both
# engines wrap a traced function, and a fresh wrapper retraces: the JAX cache is
# keyed on the callable and its static arguments, so rebuilding an engine per
# row would pay compilation once per row and report it as the first call's cost
# instead of amortising it, which is the opposite of how either library is used.
_ENGINES = {}


def _engine(kind, knob, profile_c, budget=None):
    """The cached engine, and whether this call is the one that builds it.

    The flag is what makes ``first_call_seconds`` readable: on the row that
    first uses a setting it is compilation, and on every row after it is a warm
    call that happens to be untimed.  Without the flag the two are the same
    field holding two different quantities.
    """
    from .engines_ext import LcbinintJaxEngine, MicroluxEngine

    key = (kind, knob, budget, profile_c)
    engine = _ENGINES.get(key)
    if engine is not None:
        return engine, False
    if kind == "microlux":
        engine = MicroluxEngine(tol=knob, strategy=budget, profile_c=profile_c)
    else:
        engine = LcbinintJaxEngine(reltol=knob, profile_c=profile_c)
    _ENGINES[key] = engine
    return engine, True


def row_key(source, row):
    """What identifies one row across passes.

    ``case_id`` alone does not: a case contributes nine rows, one per distance
    factor, and the profile doubles that again.  Keying the censor map on
    anything coarser would carry one row's verdict onto eight others.
    """
    return (source, row.get("case_id"), row.get("profile"), row.get("x"))


def _passes():
    """The (engine, setting) pairs that each get their own process."""
    from .engines_ext import MICROLUX_SETTINGS

    plan = [("microlux", index) for index in range(len(MICROLUX_SETTINGS))]
    plan += [("lcbinint_jax", index) for index in range(len(TIMED_TOLERANCES))]
    # The control is native C++ with no JIT behind it, so both buckets fit in
    # one pass; splitting it would buy nothing and cost a process start.
    plan.append(("control", 0))
    return plan


def evaluate_row(row, *, kind, setting, repeat, seconds_cap=0.25,
                 previous_seconds=None):
    """One engine setting on one block of the speed sweep.

    ``previous_seconds`` is what the next-looser setting cost on this same row
    in the preceding pass.  Once that exceeds ``seconds_cap`` the ladder stops,
    on the same reasoning ``sweep_speed`` uses for its buckets: the tighter
    settings are minutes of wall time for a point already far off the useful end
    of the front.  The stop is recorded, because a missing point must not be
    read as a fast one.
    """
    from .engines_ext import MICROLUX_SETTINGS

    s, q, rho = row["s"], row["q"], row["rho"]
    times = _block_times(row)
    u0, alpha = row["u0"], row["alpha"]
    profile_c = row["limb_darkening_c"]
    references = _references(row)

    out = {
        "case_id": row["case_id"],
        "s": s, "q": q, "rho": rho, "x": row["x"], "y": row["y"],
        "u0": u0, "alpha": alpha,
        "profile": row["profile"],
        "limb_darkening_c": profile_c,
        "block_epochs": len(times),
        "reference_floor": row.get("reference_floor"),
        "reference_epochs": len(references),
        "magnification": row.get("magnification"),
        "engines": [],
    }
    if not references:
        # Without a reference the timing is unplaceable on the accuracy axis.
        # Recorded as a skip so the row is not silently absent from the join.
        out["skipped"] = "no usable reference epoch"
        return out

    if kind == "microlux":
        tolerance, strategy = MICROLUX_SETTINGS[setting]
        if previous_seconds is not None and previous_seconds > seconds_cap:
            out["engines"].append({
                "engine": "microlux", "knob": tolerance,
                "budget": max(strategy),
                "censored_after_seconds_per_epoch": previous_seconds})
            return out
        try:
            engine, compiled_here = _engine(
                "microlux", tolerance, profile_c, strategy)
            measured = engine.time_block(s, q, rho, times, u0,
                                         np.degrees(alpha), repeat=repeat)
        except Exception as error:  # noqa: BLE001
            out["engines"].append({
                "engine": "microlux", "knob": tolerance,
                "budget": max(strategy),
                "failed": f"{type(error).__name__}: {error}"})
            return out
        out["engines"].append({
            "engine": "microlux",
            "knob": tolerance,
            "budget": max(strategy),
            "seconds_per_epoch": measured["seconds_per_epoch"],
            "error": achieved_error(measured["values"], references),
            # A point that ran out of sampling budget is not a point on
            # microlux's accuracy curve; it is the ceiling being hit.
            "budget_exhausted": measured["budget_exhausted"],
            "first_call_seconds": measured["first_call_seconds"],
            "compiled_here": compiled_here,
        })
        return out

    if kind == "lcbinint_jax":
        tolerance = TIMED_TOLERANCES[setting]
        if previous_seconds is not None and previous_seconds > seconds_cap:
            out["engines"].append({
                "engine": "lcbinint_jax", "knob": tolerance,
                "censored_after_seconds_per_epoch": previous_seconds})
            return out
        try:
            engine, compiled_here = _engine(
                "lcbinint_jax", tolerance, profile_c)
            measured = engine.time_block(s, q, rho, times, u0, alpha,
                                         repeat=repeat)
        except Exception as error:  # noqa: BLE001
            out["engines"].append({
                "engine": "lcbinint_jax", "knob": tolerance,
                "failed": f"{type(error).__name__}: {error}"})
            return out
        out["engines"].append({
            "engine": "lcbinint_jax",
            "knob": tolerance,
            "seconds_per_epoch": measured["seconds_per_epoch"],
            "error": achieved_error(measured["values"], references),
            "first_call_seconds": measured["first_call_seconds"],
            "compiled_here": compiled_here,
        })
        return out

    for bucket in CONTROL_BUCKETS:
        # Not cached in ``_ENGINES``: the native engine is rebuilt per call by
        # design, exactly as ``sweep_speed`` built it, because a control that is
        # timed differently from the thing it calibrates measures nothing.
        try:
            seconds, info = time_light_curve_block(
                lambda b=bucket: lcbinint_fixed(
                    "cartesian", b, profile_c)._ensure(),
                s, q, rho, times, u0, alpha, profile_c, repeat=repeat)
        except Exception as error:  # noqa: BLE001
            out["engines"].append({
                "engine": "lcbinint_cartesian_control", "knob": bucket,
                "failed": f"{type(error).__name__}: {error}"})
            continue
        values = np.asarray(info.finite_source_magnifications, dtype=float)
        out["engines"].append({
            "engine": "lcbinint_cartesian_control",
            "knob": bucket,
            "seconds_per_epoch": seconds,
            # Compared against the stored value for the same bucket, so the
            # accuracy is a check that the two runs computed the same thing,
            # not a new point on the front.
            "error": achieved_error(values, references),
        })
    return out


def _claim_core(cores, slots, lock):
    """Take a core nobody else holds, and give it back when this worker dies.

    A monotonic worker counter would be enough for a pool whose workers live as
    long as the run.  This pool's do not -- they are recycled to bound what each
    one accumulates -- and the replacement's index would wrap onto a core that
    is still occupied, putting two timed workers on one core.  Claiming and
    releasing an explicit slot is what keeps one-core-per-worker true across a
    restart, and one-core-per-worker is the whole basis of the timings.
    """
    if not cores:
        return None
    with lock:
        for index, taken in enumerate(slots):
            if not taken:
                slots[index] = 1
                break
        else:
            return None
    core = cores[index]

    def release():
        with lock:
            slots[index] = 0

    atexit.register(release)
    try:
        os.sched_setaffinity(0, {core})
    except OSError:
        return None
    return core


def _initialise(cores, slots, lock, needs_jax):
    _claim_core(cores, slots, lock)
    # XLA picks its own intra-op pool size from the visible CPU count, which is
    # the whole machine even after pinning; left alone it would oversubscribe
    # the one core this worker owns and time a thread fight.
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if needs_jax:
        from .engines_ext import configure_jax
        configure_jax()


def _previous_seconds(path, source):
    """This engine's cost per row in the preceding pass, keyed by ``row_key``.

    A missing file is not an error: the first pass of a ladder has no
    predecessor, and a pass whose predecessor skipped a block has nothing to
    censor that block by.
    """
    if not path:
        return {}
    stored = Path(path) / Path(source).name.replace("block-", "ext-")
    if not stored.exists():
        return {}
    out = {}
    for row in json.loads(stored.read_text()).get("rows", []):
        for entry in row.get("engines", []):
            seconds = entry.get("seconds_per_epoch")
            if seconds is None:
                # A censored predecessor censors its successor too, and it
                # carries the cost that stopped the ladder rather than its own.
                seconds = entry.get("censored_after_seconds_per_epoch")
            if seconds is not None:
                out[row_key(source, row)] = seconds
    return out


def _worker(payload):
    (source, destination, repeat, seconds_cap, kind, setting,
     censor_from) = payload
    target = Path(destination)
    if target.exists():
        return target.name, "skipped", 0, 0.0
    started = time.perf_counter()
    payload_in = json.loads(Path(source).read_text())
    previous = _previous_seconds(censor_from, source)
    rows = []
    for row in payload_in.get("rows", []):
        if "error" in row or not row.get("references"):
            continue
        try:
            rows.append(evaluate_row(
                row, kind=kind, setting=setting, repeat=repeat,
                seconds_cap=seconds_cap,
                previous_seconds=previous.get(row_key(source, row))))
        except Exception as error:  # noqa: BLE001
            rows.append({"case_id": row.get("case_id"), "x": row.get("x"),
                         "profile": row.get("profile"),
                         "error": f"{type(error).__name__}: {error}"})
    result = {
        "case": payload_in.get("case"),
        "source": Path(source).name,
        "engine": kind,
        "setting": setting,
        "rows": rows,
        "seconds": time.perf_counter() - started,
        "core": sorted(os.sched_getaffinity(0)),
    }
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(result))
    temporary.replace(target)
    return target.name, "done", len(rows), result["seconds"]


_MODULE = "tests.diagnostics.recal2026.sweep_ext"


def _drive(arguments):
    """Run every pass in turn, each as its own process, chaining the censor.

    Sequential rather than concurrent: two passes would each want the same
    cores, and these are timings.  Resumption is free because a worker skips a
    destination that already exists, so a pass that died can simply be re-run.
    """
    import subprocess
    import sys

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    plan = _passes()
    (output / "manifest.json").write_text(json.dumps({
        "blocks": str(arguments.blocks),
        "repeat": arguments.repeat,
        "seconds_cap": arguments.seconds_cap,
        "passes": [{"engine": kind, "setting": index,
                    "directory": f"{kind}-{index}"} for kind, index in plan],
        "profiles": {name: value for name, value in PROFILES.items()},
    }, indent=2))

    previous = {}
    failures = []
    for kind, index in plan:
        directory = output / f"{kind}-{index}"
        command = [
            sys.executable, "-m", _MODULE,
            "--blocks", arguments.blocks, "--output", str(directory),
            "--engine", kind, "--setting", str(index),
            "--workers", str(arguments.workers),
            "--repeat", str(arguments.repeat),
            "--cores", arguments.cores,
            "--seconds-cap", str(arguments.seconds_cap),
            "--blocks-per-worker", str(arguments.blocks_per_worker),
        ]
        if arguments.limit:
            command += ["--limit", str(arguments.limit)]
        if kind in previous:
            command += ["--censor-from", str(previous[kind])]
        print(f"=== pass {kind}[{index}] -> {directory}", flush=True)
        code = subprocess.run(command).returncode
        if code:
            failures.append(f"{kind}[{index}] exit {code}")
            print(f"=== pass {kind}[{index}] FAILED with {code}", flush=True)
        # Chained even on failure: the pass wrote whatever blocks it finished,
        # and censoring on a partial predecessor is the same rule on fewer rows.
        previous[kind] = directory
    if failures:
        print("failed passes: " + ", ".join(failures), flush=True)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", required=True,
                        help="sweep_speed output directory to read")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cores", default="",
                        help="comma-separated cores to pin to, e.g. 0-15")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--blocks-per-worker", type=int, default=2,
                        help="recycle a worker after this many blocks; 0 keeps "
                             "it for the whole run")
    parser.add_argument("--seconds-cap", type=float, default=0.25)
    parser.add_argument("--engine", choices=("microlux", "lcbinint_jax",
                                             "control"),
                        help="run one pass for this engine; omit to drive "
                             "every pass in turn as its own subprocess")
    parser.add_argument("--setting", type=int, default=0,
                        help="index into the engine's settings, for --engine")
    parser.add_argument("--censor-from", default="",
                        help="output directory of the next-looser setting's "
                             "pass, whose per-epoch costs stop this one")
    parser.add_argument("--verify", action="store_true",
                        help="run the frame and profile checks and exit")
    arguments = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if arguments.verify:
        from .engines_ext import (configure_jax, verify_microlux_frame,
                                  verify_microlux_limb_darkening)
        configure_jax()
        ok, accepted, rejected = verify_microlux_frame()
        print(f"frame: ok={ok} accepted_gap={accepted:.3e} "
              f"best_rejected_gap={rejected:.3e}")
        ok_ld, worst = verify_microlux_limb_darkening()
        print(f"limb darkening: ok={ok_ld} worst_gap={worst:.3e}")
        raise SystemExit(0 if (ok and ok_ld) else 1)

    if arguments.engine is None:
        raise SystemExit(_drive(arguments))

    cores = []
    for piece in arguments.cores.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, hi = piece.split("-")
            cores.extend(range(int(lo), int(hi) + 1))
        else:
            cores.append(int(piece))
    if not cores:
        cores = sorted(os.sched_getaffinity(0))[:arguments.workers]

    blocks = sorted(Path(arguments.blocks).glob("block-*.json"))
    if arguments.limit:
        blocks = blocks[:arguments.limit]
    if not blocks:
        raise SystemExit(f"no block-*.json under {arguments.blocks}")
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({
        "blocks": str(arguments.blocks),
        "block_files": len(blocks),
        "repeat": arguments.repeat,
        "seconds_cap": arguments.seconds_cap,
        "cores": cores,
        "profiles": {name: value for name, value in PROFILES.items()},
    }, indent=2))

    payloads = [
        (str(path), str(output / path.name.replace("block-", "ext-")),
         arguments.repeat, arguments.seconds_cap, arguments.engine,
         arguments.setting, arguments.censor_from)
        for path in blocks
    ]

    import multiprocessing

    slots = multiprocessing.Array("i", [0] * len(cores))
    lock = multiprocessing.Lock()
    started = time.perf_counter()
    done = 0
    with multiprocessing.Pool(
            arguments.workers, initializer=_initialise,
            initargs=(cores, slots, lock, arguments.engine != "control"),
            maxtasksperchild=arguments.blocks_per_worker or None) as pool:
        for name, status, rows, elapsed in pool.imap_unordered(_worker, payloads):
            done += 1
            rate = done / max(time.perf_counter() - started, 1e-9)
            remaining = (len(payloads) - done) / max(rate, 1e-9)
            print(f"[{done}/{len(payloads)}] {name} {status} rows={rows} "
                  f"{elapsed:.1f}s  eta {remaining / 60:.1f} min", flush=True)
    print(f"finished in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
