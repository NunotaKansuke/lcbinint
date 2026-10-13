"""Rerun and plot the six-panel comparison with the ordinary adaptive build.

Input VBM reference values and direct-kernel timings are frozen in the
publication artifact's inputs.tsv. The native executable measures the
ordinary C++ kernel at relative tolerances 1e-3 and 1e-4, starting at nbin16.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "figures/recal2026/adaptive-vbm-six-panel-20261014"
RUN = ROOT / "tests/diagnostics/results/recal2026/adaptive-vbm-six-panel-20261014"
BUILD = ROOT / "build-adaptive-production"
SOURCE = ROOT / "tests/diagnostics/recal2026/final_adaptive_normal_six_panel_runner.cpp"


def write_status(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def prepare_runner():
    command = [
        "/usr/bin/c++", "-O3", "-DNDEBUG", "-std=gnu++17", "-fopenmp",
        "-DADAPTIVE", "-DRELATIVE_SINGLE_PASS",
        "-I" + str(ROOT / "include"), "-I" + str(ROOT / "src"),
        str(SOURCE), "-o", str(RUN / "runner"), "-Wl,--start-group",
        str(BUILD / "liblcbinint_lightcurve.a"),
        str(BUILD / "liblcbinint_magnification.a"), "-Wl,--end-group",
        "-Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib",
        "/home/nunota/.miniconda3/envs/myenv/lib/libgsl.so",
        "/home/nunota/.miniconda3/envs/myenv/lib/libgslcblas.so", "-lm",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return command


def run_campaign(limit=None):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    for key in list(os.environ):
        if key.startswith("LCBININT_DIAGNOSTIC_"):
            os.environ.pop(key)
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    RUN.mkdir(parents=True, exist_ok=True)
    inputs = list(csv.DictReader((ARTIFACT / "inputs.tsv").open(), delimiter="\t"))
    if limit is not None:
        inputs = inputs[:limit]
    command = prepare_runner()
    shards = RUN / "shards"
    shards.mkdir(exist_ok=True)
    started = time.time()
    all_rows = []
    chunk_size = 32
    total = len(inputs)
    for shard, offset in enumerate(range(0, total, chunk_size)):
        part = inputs[offset:offset + chunk_size]
        inp = shards / f"{shard:04d}-input.tsv"
        raw = shards / f"{shard:04d}-raw.tsv"
        log = shards / f"{shard:04d}.log"
        with inp.open("w") as f:
            for r in part:
                f.write("\t".join(r[k] for k in ("geom_id", "s", "q", "rho", "time", "y", "c")) + "\n")
        valid = False
        if raw.exists():
            rows = list(csv.DictReader(raw.open(), delimiter="\t"))
            expected = {(r["geom_id"], t) for r in part for t in ("0.001", "0.0001")}
            observed = {(r["sample_id"], str(float(r["target"]))) for r in rows}
            valid = len(rows) == 2 * len(part) and observed == expected
        if not valid:
            proc_env = os.environ.copy()
            with log.open("a") as stream:
                proc = subprocess.Popen(
                    [str(RUN / "runner"), str(inp), str(raw), "adaptive"],
                    cwd=ROOT, env=proc_env, stdout=stream, stderr=stream,
                )
                write_status(RUN / "status.json", dict(
                    state="running", pid=os.getpid(), worker_pid=proc.pid,
                    completed=offset, total=total, shard=shard,
                    updated_unix=time.time(), command=command,
                ))
                try:
                    code = proc.wait(timeout=32 * 30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    code = -9
            if code != 0:
                write_status(RUN / "status.json", dict(state="incomplete", shard=shard, exit_code=code))
                raise RuntimeError(f"native shard {shard} failed; its log and output were retained")
            rows = list(csv.DictReader(raw.open(), delimiter="\t"))
            expected = {(r["geom_id"], t) for r in part for t in ("0.001", "0.0001")}
            observed = {(r["sample_id"], str(float(r["target"]))) for r in rows}
            if len(rows) != 2 * len(part) or observed != expected:
                write_status(RUN / "status.json", dict(state="incomplete", shard=shard, rows=len(rows), expected_rows=2 * len(part)))
                raise RuntimeError(f"native shard {shard} returned {len(rows)} rows")
        all_rows.extend(rows)
        write_status(RUN / "status.json", dict(
            state="running", pid=os.getpid(), completed=min(offset + len(part), total),
            total=total, shard=shard, elapsed_seconds=time.time() - started,
            updated_unix=time.time(),
        ))
        print(f"adaptive kernel {min(offset + len(part), total)}/{total}", flush=True)
    raw_path = RUN / "raw.tsv"
    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(all_rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(all_rows)
    write_status(RUN / "status.json", dict(
        state="measurement_complete", pid=os.getpid(), completed=total, total=total,
        elapsed_seconds=time.time() - started, updated_unix=time.time(),
    ))
    provenance = dict(
        source_commit="a68bca9d0e8234ef499dee3d860d23afef6061d1 plus ordinary-build adaptive port and caustic vertex-index seed port",
        source_sha256=hashlib.sha256((ROOT / "src/lcbinint/magnification/finite_source_magnifier.cpp").read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source_files={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                ROOT / "src/lcbinint/magnification/finite_source_magnifier.cpp",
                ROOT / "src/lcbinint/magnification/probe_diagnostics.hpp",
                ROOT / "src/lcbinint/magnification/probe_diagnostics.cpp",
                *sorted((ROOT / "src/lcbinint/magnification").glob("adaptive_*")),
                SOURCE,
            ]
        },
        input_sha256=hashlib.sha256((ARTIFACT / "inputs.tsv").read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256((RUN / "runner").read_bytes()).hexdigest(),
        lightcurve_library_sha256=hashlib.sha256((BUILD / "liblcbinint_lightcurve.a").read_bytes()).hexdigest(),
        magnification_library_sha256=hashlib.sha256((BUILD / "liblcbinint_magnification.a").read_bytes()).hexdigest(),
        plotting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plot_style_sha256=hashlib.sha256((ROOT / "tests/diagnostics/recal2026/paper_six_panel_style.py").read_bytes()).hexdigest(),
        build="build-adaptive-production; LCBININT_BOUNDARY_SUPPORT_EXPERIMENTS=OFF",
        execution="one CPU; nbin16; relative tolerance 1e-3/1e-4; max nbin512; lattice map budget1,000,000",
        timing="native binary_mag kernel; centre roots outside timer; one untimed warm call builds caustic and seed caches; no caustic construction, VBM, Python, or extra scale call in timed region; 3 timed calls, median",
        vbm="saved direct BinaryMag/BinaryMagDark times and RelTol1e-6 magnifications, unchanged from the prior six-panel corpus",
        command=command,
        rows=len(all_rows),
        python=platform.python_version(),
        cpu=platform.processor(),
        affinity=sorted(os.sched_getaffinity(0)),
        certified=False,
    )
    (ARTIFACT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return inputs, all_rows


def plot_and_report(inputs=None, raw_rows=None):
    os.environ.setdefault("MPLCONFIGDIR", str(RUN / "matplotlib-cache"))
    import sys
    sys.path.insert(0, str(ROOT / "tests/diagnostics/recal2026"))
    import numpy as np
    import paper_six_panel_style as style

    def cell_statistics(records, x_field, y_field, x_edges, y_edges, minimum_count):
        shape = (len(y_edges) - 1, len(x_edges) - 1)
        speed_buckets = [[[] for _ in range(shape[1])] for _ in range(shape[0])]
        error_buckets = [[[] for _ in range(shape[1])] for _ in range(shape[0])]
        for record in records:
            x = int(np.searchsorted(x_edges, record[x_field], side="right") - 1)
            y = int(np.searchsorted(y_edges, record[y_field], side="right") - 1)
            if record[x_field] == x_edges[-1]:
                x -= 1
            if record[y_field] == y_edges[-1]:
                y -= 1
            if not (0 <= x < shape[1] and 0 <= y < shape[0]):
                continue
            if math.isfinite(record["ratio"]):
                speed_buckets[y][x].append(record["ratio"])
            if math.isfinite(record["error"]):
                error_buckets[y][x].append(record["error"])
        counts = np.zeros(shape, dtype=int)
        speed = np.full(shape, np.nan)
        error = np.full(shape, np.nan)
        for y in range(shape[0]):
            for x in range(shape[1]):
                counts[y, x] = len(speed_buckets[y][x])
                if counts[y, x] >= minimum_count:
                    speed[y, x] = float(np.median(speed_buckets[y][x]))
                if len(error_buckets[y][x]) >= minimum_count:
                    error[y, x] = float(np.percentile(error_buckets[y][x], 95))
        return counts, speed, error

    style._cell_statistics = cell_statistics

    if inputs is None:
        inputs = list(csv.DictReader((ARTIFACT / "inputs.tsv").open(), delimiter="\t"))
    if raw_rows is None:
        raw_rows = list(csv.DictReader((RUN / "raw.tsv").open(), delimiter="\t"))
    input_by_id = {r["geom_id"]: r for r in inputs}
    rows_by = {(r["sample_id"], str(float(r["target"]))): r for r in raw_rows}
    if len(rows_by) != len(inputs) * 2:
        raise RuntimeError("input/result geometry and target coverage differs")
    joined = []
    for item in inputs:
        for target, vbm_ms in ((0.001, "vbm_ms_1e3"), (0.0001, "vbm_ms_1e4")):
            r = rows_by[item["geom_id"], str(target)]
            mu = float(r["magnification"])
            ref = float(item["reference"])
            current_ms = float(r["ms"])
            err = abs(mu / ref - 1.0) if math.isfinite(mu) and ref > 0 else math.nan
            vbm_time = float(item[vbm_ms])
            joined.append(dict(
                geom_id=int(item["geom_id"]), case_id=int(item["case_id"]),
                profile=item["profile"], d_bin_index=int(item["d_bin_index"]),
                epoch_index=int(item["epoch_index"]), q=float(item["q"]),
                rho=float(item["rho"]), a_finite=ref, target=target,
                magnification=mu, current_ms=current_ms, vbm_ms=vbm_time,
                ratio=vbm_time / current_ms if current_ms > 0 and vbm_time > 0 else math.nan,
                error=err, accepted=r["converged"] == "1",
                actual_pass=err <= target, stop_reason=r["stop_reason"],
                indicator=float(r["indicator"]), maps=int(r["maps"]),
                refinements=int(r["refinements"]),
            ))
    joined_path = ARTIFACT / "joined.tsv"
    with joined_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(joined[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(joined)
    minimum_per_cell = 8

    def bin_counts(records, x_edges, y_edges):
        counts = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=int)
        for record in records:
            x = int(np.searchsorted(x_edges, float(record["q"]), side="right") - 1)
            y = int(np.searchsorted(y_edges, float(record["rho"]), side="right") - 1)
            if float(record["q"]) == x_edges[-1]:
                x -= 1
            if float(record["rho"]) == y_edges[-1]:
                y -= 1
            if 0 <= x < counts.shape[1] and 0 <= y < counts.shape[0]:
                counts[y, x] += 1
        return counts

    groups = []
    for profile in ("uniform", "linear"):
        for target in (0.001, 0.0001):
            selected = [r for r in joined if r["profile"] == profile and r["target"] == target]
            finite = [r for r in selected if math.isfinite(r["error"]) and math.isfinite(r["ratio"])]
            accurate_accepted = [r for r in finite if r["accepted"] and r["actual_pass"]]
            ratios = [r["ratio"] for r in finite]
            errors = [r["error"] for r in selected if math.isfinite(r["error"])]
            plot_records = [
                r for r in selected
                if math.isfinite(r["ratio"]) and r["ratio"] > 0.0
            ]
            q_edges = style._log_edges(plot_records, "q", 1.0e-4, 1.0)
            rho_edges = style._log_edges(plot_records, "rho", 3.0e-5, 1.0)
            input_profile = [r for r in inputs if r["profile"] == profile]
            input_counts = bin_counts(input_profile, q_edges, rho_edges)
            q_rho_counts, q_rho_speed, q_rho_error = style._cell_statistics(
                plot_records, "q", "rho", q_edges, rho_edges, minimum_per_cell
            )
            a_edges = style._log_edges(plot_records, "a_finite", 1.0, 1.0e5)
            _, a_rho_speed, a_rho_error = style._cell_statistics(
                plot_records, "a_finite", "rho", a_edges, rho_edges,
                minimum_per_cell,
            )
            _, a_q_speed, a_q_error = style._cell_statistics(
                plot_records, "a_finite", "q", a_edges, q_edges,
                minimum_per_cell,
            )
            q_rho_grid = dict(
                input_cells_meeting_minimum=int(
                    np.count_nonzero(input_counts >= minimum_per_cell)
                ),
                input_minimum_count=int(input_counts.min()),
                plotted_speed_cells=int(np.count_nonzero(np.isfinite(q_rho_speed))),
                plotted_p95_cells=int(np.count_nonzero(np.isfinite(q_rho_error))),
                p95_cells_over_target=int(
                    np.count_nonzero(
                        np.isfinite(q_rho_error) & (q_rho_error > target)
                    )
                ),
                derived_projection_cells=dict(
                    a_rho_speed=int(np.count_nonzero(np.isfinite(a_rho_speed))),
                    a_rho_p95=int(np.count_nonzero(np.isfinite(a_rho_error))),
                    a_q_speed=int(np.count_nonzero(np.isfinite(a_q_speed))),
                    a_q_p95=int(np.count_nonzero(np.isfinite(a_q_error))),
                ),
                underfilled_cells=[
                    dict(
                        q_low=float(q_edges[i]), q_high=float(q_edges[i + 1]),
                        rho_low=float(rho_edges[j]), rho_high=float(rho_edges[j + 1]),
                        finite_rows=int(q_rho_counts[j, i]),
                    )
                    for j in range(len(rho_edges) - 1)
                    for i in range(len(q_edges) - 1)
                    if not np.isfinite(q_rho_speed[j, i])
                ],
            )
            groups.append(dict(
                profile=profile, target=target, rows=len(selected), finite=len(finite),
                accepted=sum(r["accepted"] for r in selected),
                actual_pass=sum(r["actual_pass"] for r in selected),
                false_accept=sum(r["accepted"] and not r["actual_pass"] for r in selected),
                exceptions=sum(not math.isfinite(r["error"]) for r in selected),
                median_vbm_over_adaptive=float(np.median(ratios)) if ratios else None,
                median_accurate_accepted_ratio=float(np.median([r["ratio"] for r in accurate_accepted])) if accurate_accepted else None,
                adaptive_faster_fraction=float(np.mean(np.asarray(ratios) > 1.0)) if ratios else None,
                p95_actual_relative_error=float(np.percentile(errors, 95)) if errors else None,
                max_actual_relative_error=max(errors) if errors else None,
                median_adaptive_ms=float(np.median([r["current_ms"] for r in finite])) if finite else None,
                median_vbm_ms=float(np.median([r["vbm_ms"] for r in finite])) if finite else None,
                q_rho_grid=q_rho_grid,
                stop_reasons={},
            ))
            groups[-1]["stop_reasons"] = {
                reason: sum(r["stop_reason"] == reason for r in selected)
                for reason in sorted({r["stop_reason"] for r in selected})
            }
            if profile == "uniform":
                boundaries = (0.01, 0.1, 1, 2, 3, 4, 6, 8, 16)
            else:
                boundaries = (0.03, 0.1, 1, 2, 4, 6, 8, 12, 16)
            style.SPEED_BOUNDARIES = np.asarray(boundaries)
            style.SPEED_LABELS = tuple([f"$<{boundaries[0]:g}\\times$"] +
                [f"${x:g}\\times$" for x in boundaries[:-1]] + [f"$\\geq{boundaries[-1]:g}\\times$"])
            eb = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3) if target == 1e-4 else (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
            style.ERROR_BOUNDARIES = np.asarray(eb)
            style.ERROR_LABELS = tuple([f"$\\leq{eb[0]:g}$"] + [f"${x:g}$" for x in eb[1:]] + [f"$\\geq{eb[-1]:g}$"])
            tag = f"{target:.0e}".replace("-0", "-")
            style.make_figure(
                [dict(r, ratio=float(r["ratio"]) if math.isfinite(r["ratio"]) else math.nan,
                      error=float(r["error"]) if math.isfinite(r["error"]) else math.nan)
                 for r in selected],
                ["VBM direct kernel and saved RelTol=1e-6 reference; ordinary adaptive C++ kernel"],
                ARTIFACT / f"{profile}_{tag}", profile, target, minimum_per_cell,
            )
    summary = dict(groups=groups, rows=len(joined), minimum_per_cell=minimum_per_cell,
        speed_ratio="saved VBM direct-kernel time / current native adaptive integration time; values above1 favor lcbinint",
        accuracy="absolute relative difference from saved independent VBM RelTol=1e-6 reference; p95 by cell in figures",
        timing="ordinary production build, initial nbin16, single-call relative target, timed native kernel only; one untimed warm call; centre roots and caustic construction outside the timed region",
        certified=False, complete_error_model=False)
    (ARTIFACT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Ordinary adaptive kernel vs saved VBM: six-panel comparison", "",
        "The four six-panel figures compare the ordinary production build at relative targets 1e-3 and 1e-4, starting from nbin16, against the saved direct VBM kernel timings and independent RelTol=1e-6 values. Each panel is regenerated from the 7,216-row input and 14,432 native results in this directory.", "",
        "The current side uses one relative-tolerance request per timed evaluation. Its initial grid and relative normalization are handled in that request. A separate nbin16 scale call is not timed or added. Center image roots are computed before timing; caustic and seed geometry is built in an untimed warm call; three measured native calls provide each median. VBM times are retained from the previously measured warmed direct BinaryMag/BinaryMagDark calls and were not retimed here. The current C++ timer excludes Python and model setup; any binding-call overhead in the saved VBM times is retained.", "",
        "The lower panels show actual relative difference from the saved VBM RelTol=1e-6 magnification, summarized as cell p95. They are not a mathematical error certificate. Rejected, inaccurate, and exceptional rows remain in joined.tsv and summary counts.", "",
        "| profile | target | finite / rows | estimated accepted | actual pass | false accepts | median VBM / adaptive | p95 relative error | median adaptive ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"|{g['profile']}|{g['target']:g}|{g['finite']}/{g['rows']}|{g['accepted']}|{g['actual_pass']}|{g['false_accept']}|{g['median_vbm_over_adaptive']:.4g}|{g['p95_actual_relative_error']:.4g}|{g['median_adaptive_ms']:.4g}|")
    lines += ["", "The input cohort is a q-rho coverage corpus rather than an iid speed sample. The error estimator remains an empirical estimate, not a general mathematical guarantee.", "", "Reproduce the native build with `cmake -S . -B build-adaptive-production -DCMAKE_BUILD_TYPE=Release -DLCBININT_BOUNDARY_SUPPORT_EXPERIMENTS=OFF -DLCBININT_BUILD_TESTS=OFF -DLCBININT_BUILD_PYTHON=ON -DLCBININT_ENABLE_JAX_FFI=OFF -DCMAKE_PREFIX_PATH=/home/nunota/.miniconda3/envs/myenv`, then `cmake --build build-adaptive-production -j2` and `python tests/diagnostics/recal2026/make_adaptive_vbm_six_panel_20261014.py`. The script resumes complete 32-input shards, reruns incomplete shards, and renders from the committed inputs and native measurements."]
    grid = groups[0]["q_rho_grid"]
    underfilled = grid["underfilled_cells"]
    underfilled_description = "; ".join(
        f"q=[{cell['q_low']:.3g},{cell['q_high']:.3g}), "
        f"rho=[{cell['rho_low']:.3g},{cell['rho_high']:.3g}): "
        f"{cell['finite_rows']}/8 finite rows"
        for cell in underfilled
    ) or "none"
    p95_exceedance_description = "; ".join(
        f"{g['profile']} {g['target']:g}: "
        f"{g['q_rho_grid']['p95_cells_over_target']}/"
        f"{g['q_rho_grid']['plotted_p95_cells']} cells"
        for g in groups
    )
    projection = grid["derived_projection_cells"]
    lines += [
        "",
        f"The input q-rho design covers {grid['input_cells_meeting_minimum']}/144 cells with at least {grid['input_minimum_count']} input rows per cell. The rho axis now keeps its requested 3e-5 lower bound. The previous edge logic floored the observed minimum (~3.14e-5) to 1e-5, adding an unintended low-rho row and shifting the bins.",
        f"After excluding non-finite measurements, {grid['plotted_speed_cells']}/144 q-rho cells meet the 8-row display threshold. The remaining gray q-rho cells are: {underfilled_description}. Gray means masked for insufficient finite data, never zero.",
        f"The A_fs projections are not a fully sampled independent grid because A_fs is derived from the epoch geometry; the speed maps support {projection['a_rho_speed']}/144 A_fs-rho cells and {projection['a_q_speed']}/144 A_fs-q cells.",
        f"q-rho cells whose p95 cross-code difference exceeds the requested target: {p95_exceedance_description}. These cell counts include both accepted and rejected rows; they are distinct from the individual false-accept counts in the table.",
    ]
    (ARTIFACT / "REPORT.md").write_text("\n".join(lines) + "\n")
    provenance_path = ARTIFACT / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    tracked_outputs = [
        ARTIFACT / "inputs.tsv", ARTIFACT / "joined.tsv",
        ARTIFACT / "summary.json", ARTIFACT / "REPORT.md",
        *sorted(ARTIFACT.glob("uniform_*.png")),
        *sorted(ARTIFACT.glob("uniform_*.pdf")),
        *sorted(ARTIFACT.glob("linear_*.png")),
        *sorted(ARTIFACT.glob("linear_*.pdf")),
    ]
    provenance.update(
        plotting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plot_style_sha256=hashlib.sha256((ROOT / "tests/diagnostics/recal2026/paper_six_panel_style.py").read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((ARTIFACT / "inputs.tsv").read_bytes()).hexdigest(),
        artifact_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked_outputs},
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    write_status(RUN / "status.json", dict(state="artifacts_complete_visual_review_pending", rows=len(joined), updated_unix=time.time()))
    print(json.dumps(groups, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="run a leading input subset for a smoke check")
    parser.add_argument("--render-only", action="store_true", help="plot previously completed raw results")
    args = parser.parse_args()
    if args.render_only:
        plot_and_report()
    else:
        inputs, raw = run_campaign(args.limit)
        if args.limit is None:
            plot_and_report(inputs, raw)


if __name__ == "__main__":
    main()
