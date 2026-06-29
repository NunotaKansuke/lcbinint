from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import lcbinint


SEPARATION = 0.8
MASS_RATIO_PUBLIC = 0.01
MASS_RATIO_EFFECTIVE = 1.0 / MASS_RATIO_PUBLIC
U0 = -0.01
ALPHA = 0.3
RHO = 5.0e-3
TIME = 0.006015037593984918
SOURCE_BINS = 50


def vbm_geometry(source: complex):
    s = abs(SEPARATION)
    q_input = abs(MASS_RATIO_EFFECTIVE)
    q = q_input if q_input < 1.0 else 1.0 / q_input
    a = complex(-s if q_input < 1.0 else s, 0.0)
    m1 = 1.0 / (1.0 + q)
    m2 = q * m1
    y = source + a * m1
    return a, m1, m2, y


def binary_coefficients(source: complex) -> list[complex]:
    a, m1, m2, y = vbm_geometry(source)
    yc = y.conjugate()
    a2 = a * a
    a3 = a2 * a
    m2_2 = m2 * m2
    return [
        a2 * m2_2 * y,
        a * m2 * (a * (m1 + y * (2.0 * yc - a)) - 2.0 * y),
        y * (1.0 - a3 * yc)
        - a * (m1 + 2.0 * y * yc * (1.0 + m2))
        + a2 * (yc * (m1 - m2) + y * (1.0 + m2 + yc * yc)),
        2.0 * y * yc
        + a3 * yc
        + a2 * (yc * (2.0 * y - yc) - m1)
        - a * (y + 2.0 * yc * (yc * y - m2)),
        yc * (yc * (2.0 * a + y) - 1.0)
        - a * (yc * (2.0 * a + y) - m1),
        yc * (a - yc),
    ]


def lens_residual(source: complex, image: complex) -> float:
    a, m1, m2, y = vbm_geometry(source)
    zc = image.conjugate()
    return abs((y - image) + m1 / (zc - a) + m2 / zc)


def physical_images(source: complex) -> list[complex]:
    roots = lcbinint.polynomial_roots(binary_coefficients(source))
    candidates = sorted(
        ((lens_residual(source, root), root) for root in roots),
        key=lambda item: item[0],
    )
    if len(candidates) >= 5 and candidates[3][0] * 1.0e-4 > candidates[2][0] + 1.0e-12:
        return [root for _, root in candidates[:3]]
    return [root for _, root in candidates[:5]]


def map_binary_lens(image: complex) -> complex:
    a, m1, m2, _ = vbm_geometry(0.0j)
    zc = image.conjugate()
    shifted = image - m1 / (zc - a) - m2 / zc
    return shifted - a * m1


def inside_source(image: complex, source: complex) -> bool:
    return abs(map_binary_lens(image) - source) <= RHO


def cell_key(ix: int, iy: int) -> tuple[int, int]:
    return ix, iy


def flood_grid(source: complex, seeds: list[complex], source_bins: int) -> set[tuple[int, int]]:
    dx = RHO / source_bins
    filled: set[tuple[int, int]] = set()
    queued: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()

    def enqueue(ix: int, iy: int) -> None:
        key = cell_key(ix, iy)
        if key in queued:
            return
        queued.add(key)
        image = complex(ix * dx, iy * dx)
        if inside_source(image, source):
            filled.add(key)
            queue.append(key)

    for seed in seeds:
        ix0 = round(seed.real / dx)
        iy0 = round(seed.imag / dx)
        for radius in range(4):
            found = False
            for iy in range(iy0 - radius, iy0 + radius + 1):
                for ix in range(ix0 - radius, ix0 + radius + 1):
                    if max(abs(ix - ix0), abs(iy - iy0)) != radius:
                        continue
                    if inside_source(complex(ix * dx, iy * dx), source):
                        enqueue(ix, iy)
                        found = True
                        break
                if found:
                    break
            if found:
                break

    while queue:
        ix, iy = queue.popleft()
        enqueue(ix + 1, iy)
        enqueue(ix - 1, iy)
        enqueue(ix, iy + 1)
        enqueue(ix, iy - 1)
        if len(queued) > 2_000_000:
            raise RuntimeError("flood fill exceeded safety limit")
    return filled


def boundary_images(source: complex, samples: int = 1200) -> np.ndarray:
    points: list[complex] = []
    for index in range(samples):
        theta = 2.0 * math.pi * index / samples
        boundary_source = source + RHO * complex(math.cos(theta), math.sin(theta))
        points.extend(physical_images(boundary_source))
    return np.asarray([[point.real, point.imag] for point in points])


def near_filled(point: np.ndarray, filled: set[tuple[int, int]], source_bins: int, radius: int = 3) -> bool:
    dx = RHO / source_bins
    ix0 = round(float(point[0]) / dx)
    iy0 = round(float(point[1]) / dx)
    for iy in range(iy0 - radius, iy0 + radius + 1):
        for ix in range(ix0 - radius, ix0 + radius + 1):
            if (ix, iy) in filled:
                return True
    return False


def main() -> None:
    source = complex(
        TIME * math.cos(ALPHA) - U0 * math.sin(ALPHA),
        TIME * math.sin(ALPHA) + U0 * math.cos(ALPHA),
    )
    center_seeds = physical_images(source)
    filled = flood_grid(source, center_seeds, SOURCE_BINS)
    contours = boundary_images(source)
    missing_mask = np.asarray(
        [not near_filled(point, filled, SOURCE_BINS, radius=3) for point in contours],
        dtype=bool,
    )

    dx = RHO / SOURCE_BINS
    filled_xy = np.asarray([[ix * dx, iy * dx] for ix, iy in filled])
    missing = contours[missing_mask]
    if len(missing) == 0:
        raise RuntimeError("no missing contour points found")

    x0, x1 = np.quantile(missing[:, 0], [0.02, 0.98])
    y0, y1 = np.quantile(missing[:, 1], [0.02, 0.98])
    pad = max(x1 - x0, y1 - y0) * 0.25

    fig, (ax, inset) = plt.subplots(
        1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    for axis in (ax, inset):
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="0.88", linewidth=0.5)
        axis.set_xlabel("image x")
        axis.set_ylabel("image y")

    ax.scatter(filled_xy[:, 0], filled_xy[:, 1], s=1.0, c="#93a4b7", alpha=0.42, label="lcbinint filled grid")
    ax.scatter(contours[:, 0], contours[:, 1], s=1.2, c="black", alpha=0.65, label="boundary-mapped image contour")
    ax.scatter(missing[:, 0], missing[:, 1], s=4.0, c="#d62728", alpha=0.95, label="contour not covered by grid")
    ax.scatter([seed.real for seed in center_seeds], [seed.imag for seed in center_seeds], s=32, marker="x", c="#1f77b4", label="center point seeds")
    ax.set_title("Grazing-caustic finite-source failure")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    inset.scatter(filled_xy[:, 0], filled_xy[:, 1], s=4.0, c="#93a4b7", alpha=0.40)
    inset.scatter(contours[:, 0], contours[:, 1], s=3.5, c="black", alpha=0.60)
    inset.scatter(missing[:, 0], missing[:, 1], s=9.0, c="#d62728", alpha=0.95)
    inset.set_xlim(x0 - pad, x1 + pad)
    inset.set_ylim(y0 - pad, y1 + pad)
    inset.set_title("Inset: missed thin image boundary")

    fig.suptitle(
        "s=0.8, q=0.01, rho=5e-3, t=0.006015; source_bins=50",
        y=0.98,
        fontsize=11,
    )
    fig.tight_layout()

    out = Path(__file__).with_name("grazing-caustic-imagearea4-failure.png")
    fig.savefig(out, dpi=220)
    print(out)
    print(f"source=({source.real:.17g}, {source.imag:.17g})")
    print(f"center_seeds={len(center_seeds)} filled_cells={len(filled)} contour_points={len(contours)} missing_points={missing_mask.sum()}")


if __name__ == "__main__":
    main()
