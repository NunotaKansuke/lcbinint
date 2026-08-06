"""An independent third opinion for the tangency band.

The campaign's reference is lcbinint's Cartesian resolution ladder, corroborated
by lcbinint's polar ladder and by VBMicrolensing's contour integrator.  On
10,647 sampled epochs those three witnesses agree to better than 1e-4 relative
everywhere except in a narrow band around ``d/rho = 1``, where eleven epochs
disagree by up to 9.5e-3.  Every one of the eleven sits at ``d/rho`` between
0.8 and 1.35; none sits outside it.  That is a systematic, not a stray row, and
the reference cannot arbitrate it because the reference is one of the parties.

This module supplies a fourth witness that shares no code path with any of
them.  It integrates the *point-source* magnification over the source disk by
brute force:

    A = \\int W(r) A_point(x, y) dA / \\int W(r) dA

on a composite Gauss-Legendre tensor product, row by row in the source plane.
It shares only the polynomial root solver with lcbinint -- not the grid, not
the flood fill, not the certificate, not the ring quadrature the routed answer
uses -- and nothing at all with VBM.  It is far too slow to ship, which is
exactly why it can be trusted here: it has no adaptivity to get wrong.

The convergence claim is measured, not asserted.  Each geometry is integrated
at a rising sequence of panel counts and the arbiter reports the gap between
the last two levels.  A verdict is only issued when that gap is at least ten
times smaller than the disagreement being arbitrated; otherwise the row is
reported as undecided, which is a different statement from "they were equal".

Rows are graded on the horizontal chord of the disk, which is what makes the
brute force affordable: ``LightCurve.info`` with ``alpha=0`` evaluates a whole
row of source positions in one batched native call.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import lcbinint

# Gauss-Legendre order inside each panel.  Eight is high enough that the panel
# count, not the order, is what limits accuracy on a smooth integrand, and low
# enough that doubling the panels really does buy the convergence rate the
# Richardson gap assumes.
PANEL_ORDER = 8

# Panel counts per dimension, coarse to fine.  The gap between the last two is
# the arbiter's own error bar.  The band being arbitrated has a fold touching
# the source limb, so the integrand carries an integrable inverse-square-root
# singularity and the convergence is algebraic rather than spectral: measured
# on the worst disputed epoch the gap falls 8.5e-3 -> 1.9e-5 across these four
# levels, which is the decade of margin the verdict rule below demands.
PANEL_LEVELS = (48, 96, 192, 384)

# The disagreement the arbiter has to resolve is at the 1e-3 level, so its own
# convergence gap has to be at least a decade below that before it may speak.
VERDICT_MARGIN = 0.1


def _limb_weight(normalized_radius2, limb_darkening_c):
    """``I(mu)/I0`` for the linear law, on the same convention as lcbinint.

    ``mu = sqrt(1 - r^2/rho^2)``, ``I/I0 = 1 - c (1 - mu)``.  The ``1 - c/3``
    normalisation that lcbinint applies cancels in the ratio taken below, so it
    is deliberately not applied here: including it would make the arbiter agree
    with lcbinint's convention by construction rather than by measurement.
    """
    if limb_darkening_c == 0.0:
        return np.ones_like(normalized_radius2)
    mu = np.sqrt(np.clip(1.0 - normalized_radius2, 0.0, None))
    return 1.0 - limb_darkening_c * (1.0 - mu)


def _composite_nodes(lower, upper, panels, order=PANEL_ORDER):
    """Composite Gauss-Legendre nodes and weights over ``[lower, upper]``."""
    base_nodes, base_weights = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(lower, upper, panels + 1)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    nodes = (mid[:, None] + half[:, None] * base_nodes[None, :]).ravel()
    weights = (half[:, None] * base_weights[None, :]).ravel()
    return nodes, weights


class PointSourceRows:
    """Batched point-source magnification along horizontal source-plane rows.

    ``point_source_threshold`` is pushed above anything reachable so that the
    finite-source machinery never runs: this class wants the point-source
    magnification at a coordinate, and any finite-source route taken here would
    smuggle the very averaging the arbiter is supposed to perform itself.
    """

    def __init__(self):
        self._curve = lcbinint.LightCurve(
            lens="binary",
            options=lcbinint.Options(
                coordinates="vbm", point_source_threshold=1.0e30))
        self.samples = 0

    def __call__(self, s, q, xs, y):
        info = self._curve.info(
            np.asarray(xs, dtype=float), t0=0.0, tE=1.0, u0=float(y),
            alpha=0.0, s=s, q=q, rho=1.0e-6, limb_darkening_c=0.0)
        self.samples += len(xs)
        return np.asarray(info.point_source_magnifications, dtype=float)


def integrate(rows, s, q, rho, x0, y0, limb_darkening_c, panels):
    """One brute-force disk integral at a fixed panel count.

    The disk is swept as horizontal chords.  Each chord's own extent shrinks to
    zero at the poles, so the x panels are laid inside the chord rather than
    across a fixed box: a fixed box would spend most of its nodes outside the
    disk and would resolve the equator far better than the poles.
    """
    y_nodes, y_weights = _composite_nodes(y0 - rho, y0 + rho, panels)
    weighted = 0.0
    norm = 0.0
    for y, wy in zip(y_nodes, y_weights):
        offset_y = y - y0
        half_chord2 = rho * rho - offset_y * offset_y
        if half_chord2 <= 0.0:
            continue
        half_chord = math.sqrt(half_chord2)
        x_nodes, x_weights = _composite_nodes(
            x0 - half_chord, x0 + half_chord, panels)
        offset_x = x_nodes - x0
        normalized_radius2 = np.clip(
            (offset_x * offset_x + offset_y * offset_y) / (rho * rho),
            0.0, 1.0)
        brightness = _limb_weight(normalized_radius2, limb_darkening_c)
        magnification = rows(s, q, x_nodes, y)
        if not np.all(np.isfinite(magnification)):
            return float("nan")
        weighted += wy * float(np.sum(x_weights * brightness * magnification))
        norm += wy * float(np.sum(x_weights * brightness))
    if norm <= 0.0 or not math.isfinite(norm):
        return float("nan")
    return weighted / norm


def arbitrate(case, *, levels=PANEL_LEVELS, verbose=True):
    """Integrate one geometry at every panel level and report the ladder."""
    rows = PointSourceRows()
    s, q, rho = case["s"], case["q"], case["rho"]
    x0, y0, c = case["x"], case["y"], case["limb_darkening_c"]

    ladder = []
    for panels in levels:
        started = time.perf_counter()
        before = rows.samples
        value = integrate(rows, s, q, rho, x0, y0, c, panels)
        entry = {
            "panels": panels,
            "nodes_per_dimension": panels * PANEL_ORDER,
            "value": value,
            "samples": rows.samples - before,
            "seconds": time.perf_counter() - started,
        }
        ladder.append(entry)
        if verbose:
            print(f"    panels={panels:>4} value={value:.12g} "
                  f"samples={entry['samples']:>10,d} "
                  f"{entry['seconds']:7.1f}s", flush=True)

    finite = [row for row in ladder if math.isfinite(row["value"])]
    if len(finite) < 2:
        return {"ladder": ladder, "value": float("nan"),
                "self_gap": float("inf"), "status": "did not converge"}
    value = finite[-1]["value"]
    self_gap = abs(value - finite[-2]["value"]) / max(abs(value), 1.0)
    return {
        "ladder": ladder,
        "value": value,
        "self_gap": self_gap,
        "status": "ok",
    }


# --------------------------------------------------------------------------
# The parties to the disagreement
# --------------------------------------------------------------------------

def witnesses(case, *, budget=600.0):
    """Re-run the three witnesses the campaign's reference is built from."""
    from . import reference

    s, q, rho = case["s"], case["q"], case["rho"]
    x, y, c = case["x"], case["y"], case["limb_darkening_c"]

    cartesian = reference.evaluate_ladder(
        s, q, rho, x, y, c, grid="cartesian",
        buckets=reference.LADDER_TOP, time_budget=budget)
    polar = reference.evaluate_ladder(
        s, q, rho, x, y, c, grid="polar",
        buckets=reference.LADDER_TOP, time_budget=budget)
    contour = reference.contour_reference(
        s, q, rho, x, y, c, time_budget=budget)

    def finest(ladder):
        for bucket in reversed(reference.LADDER_TOP):
            row = ladder.get(bucket) or {}
            if math.isfinite(row.get("magnification", float("nan"))):
                return {"bucket": bucket,
                        "value": row["magnification"],
                        "certified": bool(row.get("support_proven")),
                        "method": row.get("method")}
        return {"bucket": None, "value": float("nan"), "certified": False}

    return {
        "lcbinint_cartesian": finest(cartesian),
        "lcbinint_polar": finest(polar),
        "vbm_contour": {"value": contour["value"],
                        "self_gap": contour["self_gap"],
                        "reltol": contour.get("reltol"),
                        "values": contour.get("values")},
    }


def routed(case, reltol):
    """What the shipping automatic path returns, and by which route."""
    from .engines import lcbinint_auto

    engine = lcbinint_auto(reltol, case["limb_darkening_c"])
    result = engine(case["s"], case["q"], case["rho"], case["x"], case["y"])
    return {
        "value": result.magnification,
        "self_reported_error": result.error_estimate,
        "converged": result.converged,
        "method": result.method,
        "caustic_distance_over_rho":
            result.extra["caustic_distance"] / case["rho"],
        "point_magnification": result.extra["point_magnification"],
    }


def verdict(arbiter, parties):
    """Who agrees with the fourth witness, and is the gap wide enough to tell?

    The arbiter only speaks when its own convergence gap is a decade below the
    spread it is being asked to resolve.  Reporting a winner on a spread the
    arbiter cannot itself resolve would be inventing a result.
    """
    values = {name: entry["value"] for name, entry in parties.items()
              if math.isfinite(entry.get("value", float("nan")))}
    if not values or not math.isfinite(arbiter["value"]):
        return {"status": "no comparable values"}
    scale = max(abs(arbiter["value"]), 1.0)
    spread = (max(values.values()) - min(values.values())) / scale
    gaps = {name: abs(value - arbiter["value"]) / scale
            for name, value in values.items()}
    decidable = arbiter["self_gap"] <= VERDICT_MARGIN * spread
    return {
        "status": "decided" if decidable else "undecided",
        "party_spread": spread,
        "arbiter_self_gap": arbiter["self_gap"],
        "gap_to_arbiter": gaps,
        "closest": min(gaps, key=gaps.get) if decidable else None,
        "furthest": max(gaps, key=gaps.get) if decidable else None,
    }


# --------------------------------------------------------------------------
# The corpus: every epoch whose witnesses disagreed by more than a threshold
# --------------------------------------------------------------------------

def disputed_epochs(root, threshold=1.0e-3):
    """Rows from the speed sweep whose reference uncertainty exceeded a bound.

    The stored uncertainty is what the reference builder derived from how far
    the three witnesses fell apart, so it is exactly the disagreement this
    module exists to arbitrate.  The block stores one position per row and the
    reference epochs are indices into the block's own span in x; both are
    reconstructed here so the arbiter lands on the same coordinates.
    """
    import glob

    from .sweep_speed import BLOCK_SPAN_IN_RADII, DEFAULT_BLOCK

    found = []
    for path in sorted(glob.glob(f"{root}/block-*.json")):
        blob = json.load(open(path))
        for row in blob["rows"]:
            rho = row["rho"]
            span = BLOCK_SPAN_IN_RADII * rho
            times = np.linspace(
                row["x"] - 0.5 * span, row["x"] + 0.5 * span, DEFAULT_BLOCK)
            for key, entry in (row.get("references") or {}).items():
                value, uncertainty = entry.get("value"), entry.get("uncertainty")
                if value is None or uncertainty is None:
                    continue
                if not math.isfinite(value) or not math.isfinite(uncertainty):
                    continue
                relative = uncertainty / max(abs(value), 1.0)
                if relative <= threshold:
                    continue
                found.append({
                    "s": row["s"], "q": row["q"], "rho": rho,
                    "x": float(times[int(key)]), "y": row["y"],
                    "limb_darkening_c": row["limb_darkening_c"],
                    "profile": row["profile"],
                    "intended_distance_factor": row["intended_distance_factor"],
                    "reference_value": value,
                    "reference_uncertainty_relative": relative,
                    "source_block": path.rsplit("/", 1)[-1],
                    "epoch": int(key),
                })
    found.sort(key=lambda item: -item["reference_uncertainty_relative"])
    return found


def scan_dissents(root, party, threshold=1.0e-4, limit=0):
    """Positions from ``tangency_scan`` where one named witness stood alone.

    The scan names an outlier by majority, which is only as independent as the
    majority is: lcbinint's two grids share their seeding, their certificate
    and their flood fill, so "both grids against the contour" is two votes from
    one codebase.  Handing those rows to the arbiter is what turns a majority
    into a measurement, and it matters most exactly where the majority is
    lcbinint's -- otherwise the study would be marking its own paper.
    """
    import glob

    found = []
    for path in sorted(glob.glob(f"{root}/case-*.json")):
        blob = json.load(open(path))
        case = blob["case"]
        for row in blob["rows"]:
            decision = row.get("dissent") or {}
            if decision.get("status") != "ok":
                continue
            if decision["outlier"] != party:
                continue
            if decision["outlier_gap"] <= threshold:
                continue
            found.append({
                "s": case["s"], "q": case["q"], "rho": case["rho"],
                "x": row["x"], "y": row["y"],
                "limb_darkening_c": row["limb_darkening_c"],
                "profile": row["profile"],
                "intended_distance_factor": row["intended_distance_factor"],
                "achieved_distance_factor": row["achieved_distance_factor"],
                "scan_outlier": decision["outlier"],
                "scan_outlier_gap": decision["outlier_gap"],
                "scan_others_agree": decision["agreement_of_others"],
                "reference_value": decision["median"],
                "reference_uncertainty_relative": decision["spread"],
                "source_block": path.rsplit("/", 1)[-1],
                "epoch": -1,
            })
    found.sort(key=lambda item: -item["scan_outlier_gap"])
    return found[:limit] if limit else found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks",
        default="tests/diagnostics/results/recal2026/speed_discovery")
    parser.add_argument(
        "--from-scan", default="",
        help="arbitrate tangency_scan rows where --party stood alone")
    parser.add_argument(
        "--party", default="contour",
        choices=("contour", "cartesian", "polar"))
    parser.add_argument("--threshold", type=float, default=1.0e-3)
    parser.add_argument("--levels", type=int, nargs="+", default=PANEL_LEVELS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out",
        default="tests/diagnostics/results/recal2026/tangency_arbitration.json")
    arguments = parser.parse_args()

    if arguments.from_scan:
        cases = scan_dissents(arguments.from_scan, arguments.party,
                              arguments.threshold, arguments.limit)
        print(f"{len(cases)} scan positions where {arguments.party} stood "
              f"alone above {arguments.threshold:.0e} relative", flush=True)
    else:
        cases = disputed_epochs(arguments.blocks, arguments.threshold)
        if arguments.limit:
            cases = cases[:arguments.limit]
        print(f"{len(cases)} disputed epochs above {arguments.threshold:.0e} "
              f"relative", flush=True)

    results = []
    for index, case in enumerate(cases, 1):
        print(f"\n[{index}/{len(cases)}] s={case['s']:.4f} q={case['q']:.3e} "
              f"rho={case['rho']:.4e} d/rho~{case['intended_distance_factor']} "
              f"{case['profile']} "
              f"witness spread {case['reference_uncertainty_relative']:.3e}",
              flush=True)

        parties = witnesses(case)
        for reltol in (1.0e-3, 1.0e-4):
            parties[f"routed@{reltol:.0e}"] = routed(case, reltol)
        for name, entry in parties.items():
            print(f"    {name:<22} {entry['value']:.12g}", flush=True)

        arbiter = arbitrate(case, levels=tuple(arguments.levels))
        decision = verdict(arbiter, parties)
        print(f"    arbiter                {arbiter['value']:.12g} "
              f"(self gap {arbiter['self_gap']:.2e}) -> {decision['status']}",
              flush=True)
        if decision["status"] == "decided":
            print(f"    closest={decision['closest']} "
                  f"furthest={decision['furthest']}", flush=True)
            for name, gap in sorted(decision["gap_to_arbiter"].items(),
                                    key=lambda item: item[1]):
                print(f"      {name:<22} {gap:.3e}", flush=True)

        results.append({"case": case, "parties": parties,
                        "arbiter": arbiter, "verdict": decision})
        with open(arguments.out, "w") as handle:
            json.dump(results, handle, indent=1)

    print(f"\nwrote {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
