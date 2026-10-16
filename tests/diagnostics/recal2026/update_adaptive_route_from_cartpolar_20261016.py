#!/usr/bin/env python3
"""Fit and replay auto routing from saved matched Cartesian/Polar timings.

No integrator is called here. Results are selected from the completed
2026-10-16 per-method campaign and evaluated with event-grouped holdout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "tests/diagnostics/results/recal2026/cartesian-polar-sixpanel-current-20261016/full"
JOINED = ROOT / "figures/recal2026/cartesian-polar-vbm-six-panel-20261016-current/joined.tsv"
FEATURES = ROOT / "tests/diagnostics/results/recal2026/predictive-route-20261015/features.tsv"
OLD_MODEL = ROOT / "tests/diagnostics/results/recal2026/predictive-route-20261015/model.json"
ACTIVE_MODEL = ROOT / "src/lcbinint/magnification/adaptive_route_model.hpp"
OUT = ROOT / "tests/diagnostics/results/recal2026/adaptive-route-update-20261016"
FIGURES = ROOT / "figures/recal2026/lightcurve-auto-vbm-six-panel-20261016-integrated"
METHODS = ("cartesian", "polar")
PROFILES = ("uniform", "linear")
TARGETS = (1.0e-3, 1.0e-4)
TRAIN_FAILURE_MULTIPLIER = 5.0
EVALUATION_FAILURE_MULTIPLIER = 10.0


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def truth(value: object) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def tk(value: object) -> str:
    return f"{float(value):.17g}"


def load_helpers():
    path = ROOT / "tests/diagnostics/recal2026/train_adaptive_route.py"
    spec = importlib.util.spec_from_file_location("route_train_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def in_domain(row: dict) -> bool:
    f = row["x"]
    s, q, rho = (10.0 ** f[i] for i in (0, 1, 2))
    return (0.20159474280522882 <= s <= 3.949369483067623
            and 0.0001084095167119894 <= q <= 0.9221703406929264
            and 3.135563210556899e-05 <= rho <= 0.9509862925080927
            and row["profile"] in PROFILES and row["target"] in TARGETS
            and int(round(f[6])) in (3, 5) and all(math.isfinite(v) for v in f))


def assemble_data(helper) -> list[dict]:
    features = {}
    for row in read_tsv(FEATURES):
        key = (int(row["id"]), tk(row["target"]))
        if key in features:
            raise RuntimeError(f"duplicate feature row {key}")
        features[key] = row
    methods = {}
    for row in read_tsv(JOINED):
        key = (int(row["geom_id"]), tk(row["target"]), row["method"])
        if key in methods:
            raise RuntimeError(f"duplicate method result {key}")
        methods[key] = row
    ids = {key[0] for key in methods}
    expected = {(gid, tk(target), method) for gid in ids for target in TARGETS for method in METHODS}
    if len(ids) != 7216 or len(features) != 14432 or set(methods) != expected:
        raise RuntimeError("saved features/method results do not match the 7,216-geometry corpus")

    data = []
    for gid in sorted(ids):
        for target in TARGETS:
            key = (gid, tk(target))
            mrows = [methods[key + (method,)] for method in METHODS]
            feat = features[key]
            cart, polar = mrows
            if cart["case_id"] != polar["case_id"] or cart["profile"] != polar["profile"]:
                raise RuntimeError(f"method metadata differs for geometry {gid}")
            times = [float(r["method_ms"]) if finite(r["method_ms"]) and float(r["method_ms"]) > 0 else math.nan for r in mrows]
            accepted = [truth(r["estimate_accepted"]) for r in mrows]
            errors = [float(r["relative_reference_error"]) if finite(r["relative_reference_error"]) else math.inf for r in mrows]
            mags = [float(r["magnification"]) if finite(r["magnification"]) else math.nan for r in mrows]
            finite_times = [v for v in times if math.isfinite(v)]
            maximum = max(finite_times) if finite_times else math.inf
            evaluation_penalty = maximum * EVALUATION_FAILURE_MULTIPLIER
            training_penalty = maximum * TRAIN_FAILURE_MULTIPLIER
            costs = [times[i] if accepted[i] and math.isfinite(times[i]) else evaluation_penalty for i in range(2)]
            fit_costs = [times[i] if accepted[i] and math.isfinite(times[i]) else training_penalty for i in range(2)]
            row = {
                "id": gid, "case_id": cart["case_id"], "profile": cart["profile"], "target": target,
                "split": helper.split(cart["case_id"]),
                "x": [float(feat[f"f{i}"]) for i in range(15)],
                "feature_us": float(feat["feature_us"]), "overhead_ms": float(feat["feature_us"]) / 1000.0,
                "times": times, "costs": costs, "fit_costs": fit_costs, "accepted": accepted, "errors": errors,
                "mag": mags, "method_rows": mrows,
            }
            row["within_domain"] = in_domain(row)
            data.append(row)
    return data


def count_nodes(tree: dict) -> int:
    if "ensemble" in tree:
        return sum(count_nodes(component) for component in tree["ensemble"])
    if "feature" not in tree:
        return 0
    return 1 + count_nodes(tree["left"]) + count_nodes(tree["right"])


def fit_model(helper, rows: list[dict], config: dict) -> dict:
    kind = config.get("kind", "tree")
    if kind == "forest":
        branches = {}
        for profile in PROFILES:
            for target in TARGETS:
                key = (profile, tk(target))
                group = [row for row in rows if row["profile"] == profile and row["target"] == target]
                branch = config["branches"][profile][tk(target)]
                if not group:
                    branches[key] = {"polar": True}
                else:
                    branches[key] = fit_model(helper, group, {
                        "kind": "tree", "features": list(range(15)),
                        "objective": branch["objective"], "depth": branch["depth"], "min_leaf": 8,
                    })
        return {
            "feature": 3, "threshold": 0.25,
            "left": {"feature": 4, "threshold": -3.5,
                     "left": branches[("uniform", tk(1.0e-4))],
                     "right": branches[("uniform", tk(1.0e-3))]},
            "right": {"feature": 4, "threshold": -3.5,
                      "left": branches[("linear", tk(1.0e-4))],
                      "right": branches[("linear", tk(1.0e-3))]},
        }
    if kind == "boost":
        x = np.asarray([row["x"] for row in rows], dtype=float)
        costs = np.asarray([row["fit_costs"] for row in rows], dtype=float)
        target = np.clip(np.log(costs[:, 0] / costs[:, 1]), -4.0, 4.0)
        base = float(np.mean(target))
        score = np.full(len(rows), base, dtype=float)
        ensemble = []
        for _ in range(config["iterations"]):
            component = helper.fit_regression(x, target - score, config["depth"])
            ensemble.append(component)
            score += 0.12 * np.asarray([helper.regress(component, row) for row in x])
        return {"base": base, "rate": 0.12, "ensemble": ensemble}

    x = np.asarray([row["x"] for row in rows], dtype=float)
    costs = np.asarray([row["fit_costs"] for row in rows], dtype=float)
    winner = (costs[:, 1] < costs[:, 0]).astype(float)
    weights = np.minimum(np.abs(costs[:, 0] - costs[:, 1]),
                         np.quantile(np.abs(costs[:, 0] - costs[:, 1]), 0.95))
    if config["objective"] == "relative":
        weights = np.abs(np.log(costs[:, 0] / costs[:, 1]))
    return helper.fit(x, winner, weights, config["features"], config["depth"], config["min_leaf"])


def choose(trainer, model: dict, row: dict) -> int:
    # Unsupported configurations retain the established Polar fallback.
    return trainer.predict(model, row["x"]) if row["within_domain"] else 1


def model_header(tree: dict) -> str:
    def value_expr(node: dict) -> str:
        if "feature" not in node:
            return f"{float(node['value']):.17g}"
        return (f"(f[{int(node['feature'])}] <= {float(node['threshold']):.17g} ? "
                f"{value_expr(node['left'])} : {value_expr(node['right'])})")

    def emit(node: dict, indent: str) -> str:
        if "feature" not in node:
            return indent + "return " + ("true" if node["polar"] else "false") + ";\n"
        return (indent + f"if (f[{int(node['feature'])}] <= {float(node['threshold']):.17g}) {{\n"
                + emit(node["left"], indent + "    ") + indent + "} else {\n"
                + emit(node["right"], indent + "    ") + indent + "}\n")
    if "ensemble" in tree:
        body = f"    double score={float(tree['base']):.17g};\n"
        for component in tree["ensemble"]:
            body += f"    score += {float(tree['rate']):.17g} * {value_expr(component)};\n"
        body += "    return score > 0;\n"
    else:
        body = emit(tree, "    ")
    return ("#pragma once\n#include <array>\n"
            "namespace lcbinint::magnification::adaptive_detail {\n"
            "// Empirical v2 model fit from matched seed-inclusive Cartesian/Polar timings.\n"
            "// Cases are split by SHA256(route-v1:case_id); test cases are held out.\n"
            "// Fitting penalizes non-acceptance at 5x the slowest finite route; references are excluded.\n"
            "// True selects Polar; configuration gates and fallback remain in the selector.\n"
            "inline bool predict_adaptive_polar(const std::array<double,15>& f) {\n"
            + body + "}\n}\n")


def summarize_actual(rows: list[dict], predictions: list[int]) -> dict:
    timed = [(r, p) for r, p in zip(rows, predictions) if math.isfinite(r["times"][p])]
    ratios = [float(r["method_rows"][p]["vbm_over_method"]) for r, p in timed
              if finite(r["method_rows"][p]["vbm_over_method"]) and float(r["method_rows"][p]["vbm_over_method"]) > 0]
    vbm_values = [float(r["method_rows"][p]["vbm_ms"]) for r, p in timed
                  if finite(r["method_rows"][p]["vbm_ms"]) and float(r["method_rows"][p]["vbm_ms"]) > 0]
    auto_values = [r["times"][p] + r["overhead_ms"] for r, p in timed
                   if finite(r["method_rows"][p]["vbm_ms"]) and float(r["method_rows"][p]["vbm_ms"]) > 0]
    with_feature = [v / a for v, a in zip(vbm_values, auto_values)]
    return {
        "conditions": len(rows), "polar_selected": sum(predictions),
        "cartesian_selected": len(rows) - sum(predictions), "finite_selected_times": len(timed),
        "estimated_accepts": sum(r["accepted"][p] for r, p in zip(rows, predictions)),
        "reference_pass": sum(math.isfinite(r["errors"][p]) and r["errors"][p] <= r["target"] for r, p in zip(rows, predictions)),
        "false_accepts": sum(r["accepted"][p] and math.isfinite(r["errors"][p]) and r["errors"][p] > r["target"] for r, p in zip(rows, predictions)),
        "median_selected_method_ms": statistics.median(r["times"][p] for r, p in timed) if timed else None,
        "median_feature_cost_us": statistics.median(r["feature_us"] for r, _ in timed) if timed else None,
        "median_vbm_over_selected_method": statistics.median(ratios) if ratios else None,
        "median_vbm_over_selected_plus_features": statistics.median(with_feature) if with_feature else None,
        "fraction_vbm_over_selected_plus_features_gt_1": sum(v > 1 for v in with_feature) / len(with_feature) if with_feature else None,
        "total_selected_plus_features_ms": sum(auto_values) if auto_values else None,
        "total_vbm_ms": sum(vbm_values) if vbm_values else None,
        "aggregate_vbm_over_selected_plus_features": sum(vbm_values) / sum(auto_values) if auto_values and sum(auto_values) > 0 else None,
    }


def integrated_rows(trainer, data: list[dict], model: dict) -> list[dict]:
    result = []
    for row in data:
        index = choose(trainer, model, row)
        method = row["method_rows"][index]
        method_ms = row["times"][index]
        auto_ms = method_ms + row["overhead_ms"] if math.isfinite(method_ms) else math.nan
        error = row["errors"][index]
        vbm_ms = float(method["vbm_ms"])
        result.append({
            "geom_id": row["id"], "case_id": row["case_id"], "profile": row["profile"],
            "target": row["target"], "split": row["split"], "selected_method": METHODS[index],
            "route_within_training_domain": row["within_domain"], "q": float(method["q"]),
            "rho": float(method["rho"]), "a_finite": float(method["a_finite"]),
            "magnification": row["mag"][index], "relative_reference_error": error,
            "estimate_accepted": row["accepted"][index],
            "actual_pass": math.isfinite(error) and error <= row["target"],
            "false_accept": row["accepted"][index] and math.isfinite(error) and error > row["target"],
            "status": method["status"], "selected_method_ms": method_ms,
            "route_feature_cost_us": row["feature_us"], "selected_plus_features_ms": auto_ms,
            "vbm_ms": vbm_ms,
            "vbm_over_selected_plus_features": vbm_ms / auto_ms if math.isfinite(auto_ms) and auto_ms > 0 else math.nan,
            "vbm_over_selected_method": vbm_ms / method_ms if math.isfinite(method_ms) and method_ms > 0 else math.nan,
            "maps": int(method["maps"]), "subdivisions": int(method["subdivisions"]),
        })
    return result


def render(rows: list[dict]) -> None:
    os.environ["MPLCONFIGDIR"] = str(FIGURES / "matplotlib-cache")
    FIGURES.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "tests/diagnostics/recal2026"))
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import paper_six_panel_style as style

    style.SPEED_BOUNDARIES = np.asarray([2.0**k for k in range(-6, 7)], dtype=float)
    cmap = mpl.colormaps["RdBu_r"]
    colors = [mpl.colors.to_hex(cmap(v)) for v in np.linspace(0.035, 0.965, len(style.SPEED_BOUNDARIES) + 1)]
    labels = [r"$<1/64\times$"]
    labels += [rf"$1/{2 ** (-k)}\times$" if k < 0 else rf"${2 ** k}\times$" for k in range(-6, 6)]
    labels += [r"$\geq64\times$"]
    colors[labels.index(r"$1\times$")] = "#ffffff"
    style.SPEED_COLOURS, style.SPEED_LABELS = tuple(colors), tuple(labels)
    for profile in PROFILES:
        for target in TARGETS:
            bounds = ((1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3) if target == 1e-4
                      else (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2))
            style.ERROR_BOUNDARIES = np.asarray(bounds, dtype=float)
            style.ERROR_LABELS = tuple([rf"$\leq{bounds[0]:g}$"] + [rf"${v:g}$" for v in bounds[1:]] + [rf"$\geq{bounds[-1]:g}$"])
            group = [r for r in rows if r["profile"] == profile and r["target"] == target
                     and finite(r["vbm_over_selected_plus_features"]) and r["vbm_over_selected_plus_features"] > 0]
            group = [{**r, "ratio": r["vbm_over_selected_plus_features"], "error": r["relative_reference_error"]} for r in group]
            q_edges = style._log_edges(group, "q", 1e-4, 1.0)
            rho_edges = style._log_edges(group, "rho", 3e-5, 1.0)
            a_edges = style._log_edges(group, "a_finite", 1.0, 1e5)
            panels = (("q", "rho", r"$q$", r"$\rho$", r"$q\times\rho$"),
                      ("a_finite", "rho", r"$A_{\rm fs}$", r"$\rho$", r"$A_{\rm fs}\times\rho$"),
                      ("a_finite", "q", r"$A_{\rm fs}$", r"$q$", r"$A_{\rm fs}\times q$"))
            edges = {"q": q_edges, "rho": rho_edges, "a_finite": a_edges}
            fig = plt.figure(figsize=(15.5, 9.2), facecolor="white")
            grid = fig.add_gridspec(4, 3, height_ratios=(1, .15, 1, .15), hspace=.82, wspace=.20,
                                    left=.075, right=.975, top=.88, bottom=.095)
            top = [fig.add_subplot(grid[0, i]) for i in range(3)]
            bottom = [fig.add_subplot(grid[2, i]) for i in range(3)]
            for i, (xf, yf, xl, yl, title) in enumerate(panels):
                style._draw_map(top[i], group, xf, yf, edges[xf], edges[yf], 8, "speed")
                style._draw_map(bottom[i], group, xf, yf, edges[xf], edges[yf], 8, "error")
                top[i].set_title(f"{title} — runtime", fontsize=13, pad=8)
                bottom[i].set_title(f"{title} — p95 reference error", fontsize=13, pad=8)
                style._format_axis(top[i], xl, yl)
                style._format_axis(bottom[i], xl, yl)
            style._add_colourbar(fig, fig.add_subplot(grid[1, :]), style.SPEED_COLOURS, style.SPEED_LABELS,
                                 r"median $R=t_{\rm VBM}/(t_{\rm selected}+t_{\rm route\ features})$ ($R>1$: selected route faster)")
            style._add_colourbar(fig, fig.add_subplot(grid[3, :]), style.ERROR_COLOURS, style.ERROR_LABELS,
                                 "cell p95 relative difference from adjudicated reference")
            profile_name = "linear limb darkening" if profile == "linear" else "uniform source"
            fig.suptitle(rf"{profile_name}, $\epsilon_{{\rm rel}}={style._format_target(target)}$", fontsize=16, y=.965)
            fig.text(.5, .018, "Gray cells: <8 finite speed ratios (top) or reference errors (bottom)",
                     ha="center", va="bottom", fontsize=9, color="#374151")
            stem = FIGURES / f"auto_{profile}_{target:.0e}"
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
            plt.close(fig)
            stem.with_suffix(".md").write_text(
                f"# Event-grouped auto-route replay\n\nProfile `{profile}`, relative target `{target:g}`. Top panels show cell median VBM time divided by the selected saved method time plus saved route-feature extraction time. Bottom panels show cell p95 relative error from the adjudicated reference. This is an offline replay, not a new LightCurve or integral timing.\n",
                encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the fitted model to the active C++ header")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    helper = load_helpers()
    data = assemble_data(helper)
    eligible = [r for r in data if r["within_domain"] and all(math.isfinite(v) for v in r["costs"])]
    splits = {name: [r for r in eligible if r["split"] == name] for name in ("train", "validation", "test")}
    if len(data) != 14432 or any(not v for v in splits.values()):
        raise RuntimeError("paired conditions or event holdout are incomplete")

    candidates = []
    for feature_name, feature_ix in (("point", [3, 4, 5]), ("basic", list(range(7))), ("geometry", list(range(15)))):
        for objective in ("absolute", "relative"):
            for depth in range(1, 7):
                config = {"kind": "tree", "features": feature_ix, "objective": objective, "depth": depth, "min_leaf": 20}
                tree = fit_model(helper, splits["train"], config)
                pred = [helper.predict(tree, r["x"]) for r in splits["validation"]]
                score = helper.summarize(splits["validation"], pred)["geometric_penalized_regret"]
                candidates.append({"name": f"{feature_name}-{objective}-d{depth}", "config": config,
                                   "validation": helper.summarize(splits["validation"], pred),
                                   "nodes": count_nodes(tree)})

    forest_branches = {}
    for profile in PROFILES:
        forest_branches[profile] = {}
        for target in TARGETS:
            train_group = [r for r in splits["train"] if r["profile"] == profile and r["target"] == target]
            valid_group = [r for r in splits["validation"] if r["profile"] == profile and r["target"] == target]
            branch_trials = []
            for objective in ("absolute", "relative"):
                for depth in range(1, 5):
                    branch_config = {"kind": "tree", "features": list(range(15)),
                                     "objective": objective, "depth": depth, "min_leaf": 8}
                    branch_tree = fit_model(helper, train_group, branch_config)
                    branch_prediction = [helper.predict(branch_tree, r["x"]) for r in valid_group]
                    branch_score = helper.summarize(valid_group, branch_prediction)["geometric_penalized_regret"]
                    branch_trials.append({"objective": objective, "depth": depth,
                                          "score": branch_score, "nodes": count_nodes(branch_tree)})
            branch_best = min(trial["score"] for trial in branch_trials)
            branch_chosen = min((trial for trial in branch_trials if trial["score"] <= branch_best * 1.005),
                                key=lambda trial: (trial["nodes"], trial["score"]))
            forest_branches[profile][tk(target)] = {
                "objective": branch_chosen["objective"], "depth": branch_chosen["depth"]}
    forest_config = {"kind": "forest", "branches": forest_branches}
    forest_model = fit_model(helper, splits["train"], forest_config)
    forest_prediction = [helper.predict(forest_model, r["x"]) for r in splits["validation"]]
    candidates.append({"name": "profile-tolerance-forest", "config": forest_config,
                       "validation": helper.summarize(splits["validation"], forest_prediction),
                       "nodes": count_nodes(forest_model)})

    for depth in (2, 3):
        for iterations in (20, 40, 80):
            config = {"kind": "boost", "depth": depth, "iterations": iterations}
            tree = fit_model(helper, splits["train"], config)
            prediction = [helper.predict(tree, r["x"]) for r in splits["validation"]]
            candidates.append({"name": f"boost-d{depth}-n{iterations}", "config": config,
                               "validation": helper.summarize(splits["validation"], prediction),
                               "nodes": count_nodes(tree)})
    best = min(row["validation"]["geometric_penalized_regret"] for row in candidates)
    chosen = min((row for row in candidates if row["validation"]["geometric_penalized_regret"] <= best * 1.005),
                 key=lambda row: (row["nodes"], row["validation"]["geometric_penalized_regret"]))
    config = chosen["config"]
    test_train_model = fit_model(helper, splits["train"], config)
    fit_rows = [r for r in eligible if r["split"] in ("train", "validation")]
    final_model = fit_model(helper, fit_rows, config)
    previous_model = json.loads(OLD_MODEL.read_text(encoding="utf-8"))["tree"]
    test_old = helper.summarize(splits["test"], [choose(helper, previous_model, r) for r in splits["test"]])
    test_new_train_only = helper.summarize(splits["test"], [helper.predict(test_train_model, r["x"]) for r in splits["test"]])
    test_new = helper.summarize(splits["test"], [choose(helper, final_model, r) for r in splits["test"]])
    header = model_header(final_model)
    (OUT / "adaptive_route_model.hpp").write_text(header, encoding="utf-8")
    if args.apply:
        ACTIVE_MODEL.write_text(header, encoding="utf-8")

    result_rows = integrated_rows(helper, data, final_model)
    write_tsv(OUT / "integrated-joined.tsv", result_rows)
    group_results = []
    for profile in PROFILES:
        for target in TARGETS:
            group = [r for r in data if r["profile"] == profile and r["target"] == target]
            gp = [choose(helper, final_model, r) for r in group]
            old_gp = [choose(helper, previous_model, r) for r in group]
            held = [r for r in group if r["split"] == "test"]
            hp = [choose(helper, final_model, r) for r in held]
            old_hp = [choose(helper, previous_model, r) for r in held]
            group_results.append({"profile": profile, "target": target,
                                  "all_data_descriptive": summarize_actual(group, gp),
                                  "heldout_test": summarize_actual(held, hp),
                                  "previous_rule_all_data": summarize_actual(group, old_gp),
                                  "previous_rule_heldout_test": summarize_actual(held, old_hp)})
    render(result_rows)

    split_info = {name: {"conditions": len(rows), "events": len({r["case_id"] for r in rows})}
                  for name, rows in splits.items()}
    report = {
        "integrator_rerun": False, "conditions": len(data), "in_domain_conditions": len(eligible),
        "event_group_split": split_info,
        "selected_model": {"name": chosen["name"], "config": config, "split_nodes": count_nodes(final_model),
                           "best_validation_regret": best, "simplicity_margin": 0.005},
        "candidate_models": candidates,
        "test_comparison": {"previous_rule": test_old, "new_rule_fit_train_only": test_new_train_only,
                            "new_rule_fit_train_validation": test_new},
        "test_blinding": "test case IDs never enter model fitting or the scripted hyperparameter score; the test summaries were inspected during exploratory development, so this is not a sealed blind confirmation",
        "groups": group_results,
        "training_target": f"measured time plus estimated acceptance; non-acceptance receives {TRAIN_FAILURE_MULTIPLIER:g}x the slowest finite route time during fitting; reference errors and VBM timing do not enter fitting",
        "evaluation_loss": f"validation and test regret charge non-acceptance at {EVALUATION_FAILURE_MULTIPLIER:g}x the slowest finite route time",
        "timing_scope": "saved cold method time includes seed-cache miss and adaptive integration; saved feature extraction time is charged; decision-tree evaluation and shared root/caustic preparation are not charged",
        "result_scope": "offline mixed-method replay, not public LightCurve API timing",
        "provenance": {"campaign_manifest_sha256": sha256(CAMPAIGN / "manifest.json"),
                       "raw_sha256": sha256(CAMPAIGN / "raw.tsv"), "method_joined_sha256": sha256(JOINED),
                       "features_sha256": sha256(FEATURES), "new_model_sha256": hashlib.sha256(header.encode()).hexdigest(),
                       "previous_model_sha256": "8b6736ef8a70878a5f3cf6bb04919edb55d67aa36fce140e878d2d530de0e17f"},
        "active_header_updated": args.apply,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(report)
    print(f"selected={chosen['name']} split_nodes={count_nodes(final_model)} test_regret_old/new={test_old['geometric_penalized_regret']:.5f}/{test_new['geometric_penalized_regret']:.5f} applied={args.apply}")
    for row in group_results:
        a, h = row["all_data_descriptive"], row["heldout_test"]
        print(row["profile"], f"{row['target']:.0e}", "all", a["polar_selected"], a["median_vbm_over_selected_plus_features"], a["reference_pass"], a["false_accepts"], "test", h["polar_selected"], h["median_vbm_over_selected_plus_features"])


def write_report(data: dict) -> None:
    test = data["test_comparison"]
    lines = [
        "# Updated Cartesian/Polar auto route and integrated replay", "",
        "No Cartesian or Polar integral was rerun. The rule was refit from the saved matched per-method results, then those results were combined offline.", "",
        f"- Conditions: {data['conditions']:,}; in-domain: {data['in_domain_conditions']:,}.",
        f"- Selected rule: `{data['selected_model']['name']}`, {data['selected_model']['split_nodes']} split nodes.",
        "- Event groups are split by the established SHA256(case_id) rule; test cases do not enter scripted fitting or validation scoring. Their summaries were inspected during exploratory development, so this is not a sealed blind confirmation.",
        "- Candidate selection uses the smallest tree within 0.5% of best validation geometric penalized regret.",
        f"- Training labels use measured time and estimated acceptance; a non-accepted route costs {TRAIN_FAILURE_MULTIPLIER:g}x the slowest finite route time during fitting. Validation and test regret use a fixed {EVALUATION_FAILURE_MULTIPLIER:g}x penalty.",
        "- Reference error and VBM timing are reporting-only.", "",
        "## Held-out route comparison", "",
        "| rule | geometric penalized regret | Polar selected | test conditions |", "|---|---:|---:|---:|",
    ]
    for key, title in (("previous_rule", "previous rule"), ("new_rule_fit_train_only", "updated model, train only"),
                       ("new_rule_fit_train_validation", "updated model, train + validation")):
        row = test[key]
        lines.append(f"| {title} | {row['geometric_penalized_regret']:.4f} | {row['polar_selected']} | {row['n']} |")
    lines.extend(["", "## Integrated results", "",
                  "The four six-panel figures and `integrated-joined.tsv` select one saved adaptive method for every condition. The speed ratio is `VBM / (selected cold method time + saved route-feature extraction time)`; above one favors the selected route. Cold method time includes the indexed seed-cache miss and adaptive integration. The decision-tree evaluation and shared point-root/caustic preparation are outside the ratio, so this is a kernel-level replay rather than a new public `LightCurve` timing.", "",
                  "| profile | target | old Polar | new Polar | old median R | new median R | old aggregate R | new aggregate R | old heldout R | new heldout R | new heldout pass | new heldout false accepts |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in data["groups"]:
        all_row, held = row["all_data_descriptive"], row["heldout_test"]
        old_all, old_held = row["previous_rule_all_data"], row["previous_rule_heldout_test"]
        lines.append(f"| {row['profile']} | {row['target']:.0e} | {old_all['polar_selected']}/{old_all['conditions']} | {all_row['polar_selected']}/{all_row['conditions']} | {old_all['median_vbm_over_selected_plus_features']:.4g} | {all_row['median_vbm_over_selected_plus_features']:.4g} | {old_all['aggregate_vbm_over_selected_plus_features']:.4g} | {all_row['aggregate_vbm_over_selected_plus_features']:.4g} | {old_held['median_vbm_over_selected_plus_features']:.4g} | {held['median_vbm_over_selected_plus_features']:.4g} | {held['reference_pass']}/{held['conditions']} | {held['false_accepts']} |")
    lines.extend(["", "The all-data map includes development rows and is descriptive. Test cases are disjoint from fitting, but the test summaries were viewed during exploration and are not a sealed final check. Error agreement is empirical, not a formal certificate.", "",
                  f"The generated model is saved at `{OUT / 'adaptive_route_model.hpp'}`. Active selector update: `{data['active_header_updated']}`.", ""])
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
