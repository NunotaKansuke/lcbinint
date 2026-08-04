#!/usr/bin/env python3
"""Do the router's own boundaries hold, measured by what each route delivered?

``lcbinint_auto`` is the only engine in the speed sweep that picks its own
route, and every timing entry records the methods it used.  That turns the
routing thresholds into something measurable rather than argued: group the
delivered error by the route taken, and a boundary is too loose exactly where a
cheap route was chosen and the requested tolerance was then missed.

The distinction this makes, and which a single overall miss rate hides, is
between a boundary that is wrong and a boundary that is merely approached.  The
point-source and pure inverse-ray routes miss nothing anywhere; the misses are
confined to the two routes that approximate over an extended source, and within
those they sit in one corner of the parameter space each.  A rate alone would
have averaged the two together and understated both.

For a route that does miss, ``separation`` asks the only question that can move
a boundary: do the misses separate from the hits on a quantity the router can
test *before* it commits?  It reports the cost of the tightest such predicate --
how many legitimate hits it would also reject, and what they would then cost --
because a predicate that catches every miss by refusing most of the route is not
a fix, it is the route's deletion written as a condition.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from .speed_analysis import ACCURACY_TARGETS, _error, _usable, load

PROFILES = ("uniform", "linear")

# The routes that approximate over the source rather than integrating a
# resolved image, and so are the only ones whose acceptance test can be wrong
# in a way the certificate does not catch.
APPROXIMATING = ("hexadecapole", "source_plane_quadrature")


def _entries(rows, profile, target, route=None):
    """Every ``lcbinint_auto`` result at one accuracy, with its row."""
    out = []
    for row in rows:
        if profile is not None and row.get("profile") != profile:
            continue
        if not _usable(row, target):
            continue
        for entry in row.get("engines", []):
            if entry.get("engine") != "lcbinint_auto":
                continue
            if entry.get("knob") != target:
                continue
            error = _error(entry)
            if error is None:
                continue
            methods = tuple(sorted(entry.get("methods") or ["<none>"]))
            if route is not None and route not in methods:
                continue
            out.append((methods, error, entry.get("seconds_per_epoch"), row))
    return out


def by_route(rows, profile, target):
    """Miss rate, worst error and cost, per route the auto engine chose."""
    grouped = defaultdict(list)
    for methods, error, seconds, row in _entries(rows, profile, target):
        grouped[methods].append((error, seconds, row))
    out = []
    for methods, items in grouped.items():
        errors = np.array([item[0] for item in items])
        seconds = np.array([item[1] * 1.0e3 for item in items
                            if item[1] is not None])
        missed = errors > target
        worst = max(items, key=lambda item: item[0])
        entry = {
            "route": "+".join(methods),
            "blocks": len(items),
            "miss_rate": float(missed.mean()),
            "misses": int(missed.sum()),
            "worst_error": float(errors.max()),
            "median_error": float(np.median(errors)),
            "median_milliseconds_per_epoch":
                float(np.median(seconds)) if seconds.size else None,
        }
        if missed.any():
            row = worst[2]
            entry["worst_case"] = {
                "s": row["s"], "q": row["q"], "rho": row["rho"],
                "distance_factor": row.get("intended_distance_factor"),
                "magnification": row.get("magnification"),
                "error": float(worst[0]),
                "over_target": float(worst[0] / target),
            }
        out.append(entry)
    out.sort(key=lambda item: -item["blocks"])
    return out


def separation(rows, route, target, profile=None):
    """Can a pre-commitment predicate separate this route's misses from its hits?

    ``rho`` and the magnification are both available before the route is chosen
    -- the multipole stage has already produced the second -- so a predicate on
    them is implementable.  What is reported is the tightest predicate that
    catches every miss, together with what it costs: the hits it also rejects,
    and the median cost of the route those hits would fall back to.
    """
    hits, misses = [], []
    for _, error, seconds, row in _entries(rows, profile, target, route):
        item = {"rho": row["rho"], "magnification": row.get("magnification"),
                "s": row["s"], "q": row["q"],
                "distance_factor": row.get("intended_distance_factor"),
                "error": error, "seconds": seconds,
                "profile": row.get("profile")}
        (misses if error > target else hits).append(item)
    out = {"route": route, "target": target, "profile": profile,
           "chosen": len(hits) + len(misses), "missed": len(misses)}
    if not misses:
        return out

    for name in ("rho", "magnification", "distance_factor"):
        for label, group in (("hits", hits), ("misses", misses)):
            values = np.array([item[name] for item in group], dtype=float)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            out.setdefault(name, {})[label] = {
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
                "n": int(values.size),
            }

    # The tightest axis-aligned box in (rho, A) that contains every miss.  A
    # box rather than a one-sided cut because the two routes fail at opposite
    # ends of the magnification axis -- the hexadecapole on faint extended
    # sources, the quadrature on bright grazing ones -- and a predicate shape
    # chosen to fit one of them would misreport the other as unseparable.
    rho_floor = min(item["rho"] for item in misses)
    rho_ceiling = max(item["rho"] for item in misses)
    magnifications = [item["magnification"] for item in misses
                      if item["magnification"] is not None]
    magnification_floor = min(magnifications)
    magnification_ceiling = max(magnifications)
    rejected = [item for item in hits
                if rho_floor <= item["rho"] <= rho_ceiling
                and item["magnification"] is not None
                and magnification_floor <= item["magnification"]
                <= magnification_ceiling]
    kept_seconds = [item["seconds"] for item in rejected
                    if item["seconds"] is not None]
    all_seconds = [item["seconds"] for item in hits + misses
                   if item["seconds"] is not None]
    out["predicate"] = {
        "rho_between": [float(rho_floor), float(rho_ceiling)],
        "magnification_between": [float(magnification_floor),
                                  float(magnification_ceiling)],
        "misses_caught": len(misses),
        "hits_also_rejected": len(rejected),
        "hits_rejected_share":
            float(len(rejected) / len(hits)) if hits else None,
        "rejected_median_milliseconds_per_epoch":
            float(np.median(kept_seconds) * 1.0e3) if kept_seconds else None,
        "route_median_milliseconds_per_epoch":
            float(np.median(all_seconds) * 1.0e3) if all_seconds else None,
    }
    out["every_miss"] = sorted(
        ({k: v for k, v in item.items() if k != "seconds"} for item in misses),
        key=lambda item: -item["error"])
    return out


def summarise(directory):
    rows = load(directory)
    out = {"directory": str(directory), "blocks": len(rows), "by_route": {},
           "separation": {}}
    for profile in PROFILES:
        for target in ACCURACY_TARGETS:
            out["by_route"][f"{profile}@{target:g}"] = by_route(
                rows, profile, target)
    for route in APPROXIMATING:
        for target in ACCURACY_TARGETS:
            result = separation(rows, route, target)
            if result["chosen"]:
                out["separation"][f"{route}@{target:g}"] = result
    return out


def _print(result):
    for key, routes in result["by_route"].items():
        print(f"\n===== {key} =====")
        for entry in routes:
            line = (f"  {entry['route']:52s} n={entry['blocks']:4d} "
                    f"miss {entry['miss_rate']:6.1%}  "
                    f"worst {entry['worst_error']:.2e}")
            if entry["median_milliseconds_per_epoch"] is not None:
                line += f"  {entry['median_milliseconds_per_epoch']:8.3f} ms/ep"
            print(line)
            worst = entry.get("worst_case")
            if worst:
                print(f"       worst miss: s={worst['s']:.4f} "
                      f"q={worst['q']:.3e} rho={worst['rho']:.3e} "
                      f"d/rho={worst['distance_factor']} "
                      f"A={worst['magnification']:.4g} "
                      f"({worst['over_target']:.1f}x target)")

    for key, entry in result["separation"].items():
        if not entry["missed"]:
            continue
        print(f"\n########## {key}: {entry['missed']} misses of "
              f"{entry['chosen']} ##########")
        predicate = entry["predicate"]
        rho_lo, rho_hi = predicate["rho_between"]
        mag_lo, mag_hi = predicate["magnification_between"]
        print(f"  predicate rho in [{rho_lo:.4g}, {rho_hi:.4g}] and "
              f"A in [{mag_lo:.4g}, {mag_hi:.4g}] catches "
              f"{predicate['misses_caught']}/{entry['missed']} misses")
        share = predicate["hits_rejected_share"]
        print(f"  and rejects {predicate['hits_also_rejected']} legitimate "
              f"hits" + (f" ({share:.1%})" if share is not None else ""))
        if predicate["route_median_milliseconds_per_epoch"] is not None:
            print(f"  route costs "
                  f"{predicate['route_median_milliseconds_per_epoch']:.3f} "
                  f"ms/ep today")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output", default="")
    arguments = parser.parse_args()
    result = summarise(arguments.directory)
    _print(result)
    if arguments.output:
        with open(arguments.output, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
