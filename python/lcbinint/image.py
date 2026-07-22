"""Image-plane helpers for binary-lens geometry plots."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _lcbinint as lc


@dataclass(frozen=True)
class ImageRegion:
    """One inverse-ray image region for a finite circular source."""

    index: int
    seed: np.ndarray
    points: np.ndarray
    parity: int
    magnification: float


@dataclass(frozen=True)
class ImagePlane:
    """Binary-lens point-image and geometry helper.

    Parameters use the same VBM-compatible convention as ``LightCurve`` by
    default: ``q`` is the usual secondary/primary mass ratio and ``s`` is the
    binary separation. ``x`` and ``y`` are the source position in the lens frame.
    """

    q: float
    s: float
    x: float
    y: float
    rho: float = 0.0
    n_points: int = 512
    coordinates: str = "vbm"

    def images(self, *, full: bool = False):
        """Return image positions.

        By default this returns an ``(n_images, 2)`` numpy array of ``x, y``.
        With ``full=True`` it returns the bound ``lc.ImagePoint`` objects, which
        also expose ``magnification``, ``jacobian_determinant``, and ``parity``.
        """
        points = lc._binary_images(self.s, self._solver_q(), self.x, self.y)
        if full:
            return points
        return np.asarray([[p.x, p.y] for p in points], dtype=float)

    def image_table(self):
        """Return point-image seeds and diagnostics as a structured numpy array."""
        dtype = [
            ("x", "f8"),
            ("y", "f8"),
            ("magnification", "f8"),
            ("jacobian_determinant", "f8"),
            ("parity", "i4"),
        ]
        rows = [
            (p.x, p.y, p.magnification, p.jacobian_determinant, p.parity)
            for p in self.images(full=True)
        ]
        return np.asarray(rows, dtype=dtype)

    def seeds(self):
        """Return point-source image seeds as an ``(n_images, 2)`` array."""
        return self.images()

    def ray_shooting_images(
        self,
        *,
        resolution: int = 360,
        padding: float = 2.5,
        min_radius: float | None = None,
    ):
        """Return finite-source image regions from image-plane inverse shooting.

        The returned list contains ``ImageRegion`` objects. Each region has the
        point-source seed, parity, point-source magnification, and an ``(n, 2)``
        array of image-plane sample points whose lens-equation image falls
        inside the source disk of radius ``rho``.
        """
        if self.rho <= 0.0:
            raise ValueError("rho must be positive for ray_shooting_images()")
        if resolution <= 1:
            raise ValueError("resolution must be greater than 1")

        table = self.image_table()
        if len(table) == 0:
            return []

        seeds = np.column_stack([table["x"], table["y"]])
        mus = np.maximum(table["magnification"], 1.0)
        local_radii = padding * self.rho * np.sqrt(mus)
        floor = 6.0 * self.rho if min_radius is None else float(min_radius)
        local_radii = np.maximum(local_radii, floor)

        x_min = float(np.min(seeds[:, 0] - local_radii))
        x_max = float(np.max(seeds[:, 0] + local_radii))
        y_min = float(np.min(seeds[:, 1] - local_radii))
        y_max = float(np.max(seeds[:, 1] + local_radii))
        span = max(x_max - x_min, y_max - y_min)
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        x_min, x_max = cx - 0.5 * span, cx + 0.5 * span
        y_min, y_max = cy - 0.5 * span, cy + 0.5 * span

        xs = np.linspace(x_min, x_max, int(resolution))
        ys = np.linspace(y_min, y_max, int(resolution))
        xx, yy = np.meshgrid(xs, ys)
        sx, sy = self._lens_equation(xx, yy)
        inside = (sx - self.x) ** 2 + (sy - self.y) ** 2 <= self.rho ** 2
        if not np.any(inside):
            return [
                ImageRegion(
                    index=i,
                    seed=seeds[i].copy(),
                    points=np.empty((0, 2), dtype=float),
                    parity=int(table["parity"][i]),
                    magnification=float(table["magnification"][i]),
                )
                for i in range(len(table))
            ]

        points = np.column_stack([xx[inside], yy[inside]])
        distances2 = (
            (points[:, None, 0] - seeds[None, :, 0]) ** 2
            + (points[:, None, 1] - seeds[None, :, 1]) ** 2
        )
        labels = np.argmin(distances2, axis=1)

        regions = []
        for i in range(len(table)):
            regions.append(ImageRegion(
                index=i,
                seed=seeds[i].copy(),
                points=points[labels == i],
                parity=int(table["parity"][i]),
                magnification=float(table["magnification"][i]),
            ))
        return regions

    def caustics(self):
        """Return one closed polyline per physical caustic in the source plane."""
        return self._light_curve().caustics(s=self.s, q=self.q, n_points=self.n_points)

    def critical_curves(self):
        """Return one closed polyline per physical critical curve in the image plane."""
        return self._light_curve().critical_curves(
            s=self.s, q=self.q, n_points=self.n_points
        )

    def plot(
        self,
        *,
        ax=None,
        show_source_radius: bool | None = None,
        caustics: bool = True,
        critical_curves: bool = True,
        images: bool = True,
        image_resolution: int = 360,
        seeds: bool = True,
        source: bool = True,
        legend: bool = False,
    ):
        """Plot caustics, critical curves, source, finite images, and seeds."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(figsize=(3.8, 3.8))

        if critical_curves:
            self._plot_branches(
                ax, self.critical_curves(), color="tab:blue", lw=1.1,
                label="critical curve"
            )
        if caustics:
            self._plot_branches(
                ax, self.caustics(), color="tab:red", lw=1.1,
                label="caustic"
            )
        if source:
            ax.scatter(
                [self.x], [self.y], s=42, marker="*", color="#1f77b4",
                zorder=5, label="source"
            )
            draw_radius = self.rho > 0.0 if show_source_radius is None else show_source_radius
            if draw_radius and self.rho > 0.0:
                from matplotlib.patches import Circle

                ax.add_patch(Circle(
                    (self.x, self.y), self.rho, fill=False,
                    edgecolor="#1f77b4", linewidth=1.0, alpha=0.75
                ))
        if images and self.rho > 0.0:
            colors = {1: "#2ca25f", -1: "#5e3c99"}
            for region in self.ray_shooting_images(resolution=image_resolution):
                if len(region.points) == 0:
                    continue
                ax.scatter(
                    region.points[:, 0], region.points[:, 1],
                    s=3.0, marker="s", linewidth=0.0, alpha=0.65,
                    color=colors.get(region.parity, "0.25"),
                    zorder=3,
                    label="finite image" if region.index == 0 else None,
                )
        elif images:
            seed_points = self.images()
            if len(seed_points):
                ax.scatter(
                    seed_points[:, 0], seed_points[:, 1], s=30, marker="o",
                    facecolor="black", edgecolor="white", linewidth=0.6,
                    zorder=6, label="image"
                )
        if seeds:
            seed_points = self.seeds()
            if len(seed_points):
                ax.scatter(
                    seed_points[:, 0], seed_points[:, 1], s=42, marker="x",
                    color="#ffb000", linewidth=1.4, zorder=7,
                    label="seed"
                )

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), frameon=False)
        ax.figure.tight_layout()
        return ax

    def _light_curve(self):
        return lc.LightCurve(
            options=lc.Options(coordinates=self.coordinates, caustic_bins=self.n_points)
        )

    def _solver_q(self):
        if self.q <= 0.0:
            raise ValueError("q must be positive")
        if self.coordinates in ("vbm", "vbbl", "standard"):
            return 1.0 / self.q
        return self.q

    def _lens_equation(self, image_x, image_y):
        q_input = abs(self._solver_q())
        q = q_input if q_input < 1.0 else 1.0 / q_input
        a = -abs(self.s) if q_input < 1.0 else abs(self.s)
        m1 = 1.0 / (1.0 + q)
        m2 = q * m1

        z = image_x + 1j * image_y
        zc = np.conjugate(z)
        mapped = z - m1 / (zc - a) - m2 / zc - a * m1
        return np.real(mapped), np.imag(mapped)

    @staticmethod
    def _plot_branches(ax, branches, **kwargs):
        label = kwargs.pop("label", None)
        for i, (xs, ys) in enumerate(zip(branches.x, branches.y)):
            ax.plot(xs, ys, label=label if i == 0 else None, **kwargs)

def binary(q, s, x, y, *, rho=0.0, n_points=512, coordinates="vbm"):
    """Create an ``ImagePlane`` for a binary lens."""
    return ImagePlane(
        q=float(q),
        s=float(s),
        x=float(x),
        y=float(y),
        rho=float(rho),
        n_points=int(n_points),
        coordinates=coordinates,
    )


def binary_images(q, s, x, y, *, coordinates="vbm", full=False):
    """Return binary-lens point images for a source position."""
    return binary(q, s, x, y, coordinates=coordinates).images(full=full)


__all__ = ["ImagePlane", "ImageRegion", "binary", "binary_images"]
