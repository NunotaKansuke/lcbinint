"""Image-plane helpers for binary-lens geometry plots."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import lc


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
        """Return image diagnostics as a structured numpy array."""
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

    def caustics(self):
        """Return caustic branches in the source plane."""
        return self._light_curve().caustics(s=self.s, q=self.q, n_points=self.n_points)

    def critical_curves(self):
        """Return critical-curve branches in the image plane."""
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
        source: bool = True,
        legend: bool = True,
    ):
        """Plot caustics, critical curves, source position, and image positions."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(figsize=(6.0, 6.0))

        if critical_curves:
            self._plot_branches(
                ax, self.critical_curves(), color="0.45", lw=1.1,
                label="critical curve"
            )
        if caustics:
            self._plot_branches(
                ax, self.caustics(), color="#c43c35", lw=1.3,
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
        if images:
            points = self.images()
            if len(points):
                ax.scatter(
                    points[:, 0], points[:, 1], s=30, marker="o",
                    facecolor="black", edgecolor="white", linewidth=0.6,
                    zorder=6, label="image"
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


__all__ = ["ImagePlane", "binary", "binary_images"]
