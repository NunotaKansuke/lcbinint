"""M5: the full-epoch reference solver (plan sec. 12-13).

:func:`solve_epoch` is the single entry point a caller uses to get the
finite-source magnification (uniform or linear limb darkening) for one
source position, with an honest status:

* it classifies the radial cells (:mod:`topology`),
* routes the ``|zeta| ~ 0`` on-axis-origin locus to the connection-free
  singular patch (:mod:`singular`), everything else to the holonomic flux
  (:mod:`flux`, which re-anchors the period at every quadrature node),
* and -- in the default ``validated`` mode -- cross-checks the value against
  a fully independent source-plane quadrature of the Witt & Mao
  point-source magnification (:mod:`direct_quadrature`), promoting the
  status to ``OK_VALIDATED`` only when that check converges *and* agrees.

The status taxonomy is plan sec. 12.2.  ``value`` and gradient budgets are
tracked separately (gradients arrive at M6); this module reports the value
budget.  Nothing here trusts a single transported seed -- see
``checkpoint_M4.md`` sec. 4.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .direct_quadrature import image_plane_flux_grid, source_plane_flux
from .flux import epoch_flux
from .polynomial_family import LensParams
from .singular import (
    SINGULAR_ZETA_TOL,
    near_origin_source,
    on_axis_origin_flux,
    representation_events,
)

try:                                    # optional -- only for the value cross-check
    import lcbinint as _lcbinint
except Exception:                                            # pragma: no cover
    _lcbinint = None

# agreement bar against the independent reference: promotion to
# OK_VALIDATED requires rel <= this; a *trustworthy* reference that
# disagrees by more than this is a value failure (never silently returned
# as OK).
_VALIDATE_RTOL = 3.0e-3
# the reference is "trustworthy" when two methods of totally different
# construction (the production inverse-ray solver and a source-plane
# point-source quadrature) corroborate each other to this.
_REF_CORROBORATE_RTOL = 1.5e-2


@dataclass
class EpochResult:
    params: LensParams
    u: float
    mu: float                       # linear-LD magnification at ``u``
    mu_uniform: float               # ``u = 0`` magnification
    F0: float
    F_half: float
    r_max: float
    status: str
    mode: str
    cells: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    seconds: float = 0.0
    jacobian: "dict | None" = None    # set when solve_epoch(..., with_jacobian=True)

    def mu_linear_ld(self, u: float) -> float:
        rho2 = self.params.rho ** 2
        return (((1.0 - u) * self.F0 + u * self.F_half)
                / (math.pi * rho2 * (1.0 - u / 3.0)))


def _mu_from_flux(F0, F_half, rho, u):
    rho2 = rho * rho
    mu_u = F0 / (math.pi * rho2)
    mu = (((1.0 - u) * F0 + u * F_half) / (math.pi * rho2 * (1.0 - u / 3.0)))
    return mu, mu_u


def _ray_shooting_mu(params: LensParams, u: float):
    """Production inverse-ray ``mu`` (``lcbinint.binary_ray_shooting``) or ``None``.

    Source position is passed in the *primary* frame shifted to the centre
    of mass (``x = xs - m1 a``), the convention ``binary_ray_shooting``
    uses; ``s = a``.  This is the M7 A/B target and the plan's correctness
    bar ("現行 production solver 同等以上").
    """
    if _lcbinint is None:
        return None
    xs, ys = params.primary_frame_source()
    if min(math.hypot(xs, ys), math.hypot(xs - params.a, ys)) < 1e-9:
        # source centre sits *exactly* on a point mass -- inverse-ray
        # shooting adaptively refines forever near the enclosed singularity
        # and never returns.  (An off-centre disk that merely *covers* a lens
        # is fine; only the exact locus hangs.)
        return None
    x_com = xs - params.m1 * params.a
    ld = _lcbinint.LimbDarkening.linear(u) if u else None
    try:
        return float(_lcbinint.binary_ray_shooting(
            x_com, ys, s=params.a, q=params.q, rho=params.rho,
            limb_darkening=ld))
    except Exception:                                        # pragma: no cover
        return None


def _covers_lens(params: LensParams) -> bool:
    """True when the source disk encloses a point-lens centre.

    The source-plane point-source quadrature integrates ``A_pt``, which
    diverges at each lens; when a lens sits inside the disk that quadrature
    is unreliable at every resolution (``checkpoint_M5.md`` sec. 4).  The
    image-plane grid (bounded ``sqrt(phi) >= 0`` mask) stays convergent, so
    it is used as the corroborator there instead."""
    xs, ys = params.primary_frame_source()
    return (math.hypot(xs, ys) < params.rho
            or math.hypot(xs - params.a, ys) < params.rho)


def _source_plane_mu(params: LensParams, u: float, *, n_r: int = 96,
                     n_theta: int = 512):
    """Source-plane Witt & Mao point-source disk quadrature ``mu`` (independent)."""
    f0, fh = source_plane_flux(params, n_r=n_r, n_theta=n_theta)
    mu, _ = _mu_from_flux(f0, fh, params.rho, u)
    return mu


def _grid_mu(params: LensParams, u: float, *, n_r: int = 1000,
             n_theta: int = 16000):
    """Image-plane dense-grid ``mu`` -- bounded ``sqrt(phi) >= 0`` mask sum.

    Independent of the holonomic path *and* of the source-plane quadrature,
    and (unlike either) convergent both at an unresolved caustic and when the
    disk covers a lens.  Slightly noisy for a barely-resolved planetary
    caustic (``rho ~ sqrt(q)``), so it is used only where ``source_plane`` /
    ``binary_ray_shooting`` are themselves unavailable or suspect."""
    f0, fh = image_plane_flux_grid(params, n_r=n_r, n_theta=n_theta)
    mu, _ = _mu_from_flux(f0, fh, params.rho, u)
    return mu


def _independent_mu(params: LensParams, u: float):
    """``(ref_mu, trustworthy, corroboration_rel, method)``.

    The reference is the production inverse-ray ``mu`` when available,
    corroborated by a second quadrature of totally different construction --
    ``source_plane_flux`` away from a covered lens, the image-plane grid when
    the disk covers one (``source_plane`` is then unreliable).
    ``trustworthy`` is set only when the two agree; near an unresolved
    caustic they will not, and the caller then leaves the status unpromoted
    rather than guessing.
    """
    ray = _ray_shooting_mu(params, u)
    covers = _covers_lens(params)
    if covers:
        corr = _grid_mu(params, u)
        corr_method = "image_plane_flux_grid"
    else:
        corr = _source_plane_mu(params, u)
        corr_method = "source_plane_flux"
    if ray is None:
        # no production solver (or the source is exactly on a lens) -- fall
        # back to the corroborator alone, trustworthy only if a second,
        # higher-resolution evaluation of it agrees.
        if covers:
            hi = _grid_mu(params, u, n_r=1600, n_theta=24000)
        else:
            hi = _source_plane_mu(params, u, n_r=128, n_theta=768)
        rel = abs(corr - hi) / max(abs(hi), 1e-300)
        return hi, rel <= 3.0e-3, rel, f"{corr_method} x2 res"
    rel = abs(ray - corr) / max(abs(ray), 1e-300)
    return ray, rel <= _REF_CORROBORATE_RTOL, rel, (
        f"binary_ray_shooting corroborated by {corr_method}")


def solve_epoch(params: LensParams, u: float = 0.0, *, mode: str = "validated",
                validate: bool | None = None,
                with_jacobian: bool = False) -> EpochResult:
    """Finite-source magnification for one source position (plan sec. 1, 12).

    ``u`` is the linear limb-darkening coefficient (``u = 0`` -> uniform).
    ``mode`` is ``"validated"`` (default -- run the independent cross-check)
    or ``"fast"`` (skip it; status stays whatever the flux path reported).
    ``validate`` overrides the mode's cross-check choice if given.

    ``with_jacobian`` also returns the 5-component magnification Jacobian
    (:mod:`jacobian`) on ``result.jacobian``.  It is computed from a
    separate per-cell Gauss-Chebyshev reconstruction; to keep ``value``
    and the Jacobian mutually consistent (plan sec. 11) the result's
    ``F0`` / ``F_half`` / ``mu`` are then taken from that reconstruction
    (they agree with the default :func:`flux.epoch_flux` path to ~1e-4).
    A near-tangency / full-circle / singular-patch node marks
    ``result.jacobian['status']`` (and the epoch status) ``GRADIENT_UNRELIABLE``.
    """
    if mode not in ("validated", "fast"):
        raise ValueError(f"mode must be 'validated' or 'fast', got {mode!r}")
    do_validate = (mode == "validated") if validate is None else bool(validate)
    t0 = time.perf_counter()

    rho = params.rho
    notes: list = []
    cells: list = []

    # ---- routing -------------------------------------------------------
    routed_singular = near_origin_source(params)
    if routed_singular:
        from .topology import classify_cells
        topo = classify_cells(params)
        patch = on_axis_origin_flux(params, topo=topo)
        F0, F_half, status = patch.F0, patch.F_half, patch.status
        notes.append(patch.note)
        r_max = topo.r_max
    else:
        fr = epoch_flux(params)
        F0, F_half, status = fr.F0, fr.F_half, fr.status
        cells, r_max = fr.cells, fr.r_max
        notes.extend(fr.notes)

    for R, kind in representation_events(params):
        notes.append(f"representation event {kind} at R={R:.6f} inside the "
                     f"scan (crossed by direct QAWSE re-anchor; no bridging)")

    mu, mu_u = _mu_from_flux(F0, F_half, rho, u)

    # ---- optional 5-component Jacobian (plan sec. 11) ----------------
    jacobian = None
    if with_jacobian:
        from .jacobian import epoch_jacobian as _epoch_jacobian
        ej = _epoch_jacobian(params, u)
        F0, F_half = ej.F0, ej.F_half
        mu, mu_u = _mu_from_flux(F0, F_half, rho, u)
        jacobian = {
            "grad_mu": ej.grad_mu,
            "dmu_du": ej.dmu_du,
            "param_order": ("xs", "ys", "rho", "q", "a"),
            "status": ej.status,
            "notes": list(ej.notes),
        }
        if ej.status != "OK":
            status = _worst_status(status, ej.status)
            notes.append(f"jacobian: {ej.status}")

    # ---- independent value cross-check --------------------------------
    validation: dict = {}
    if do_validate and routed_singular:
        # the singular patch IS a connection-free direct quadrature and the
        # source sits on the primary lens, so neither the production
        # inverse-ray solver (hangs on the exact locus) nor the source-plane
        # quadrature (unreliable with a lens inside the disk) can be the
        # oracle.  The image-plane grid stays convergent here; record it as a
        # corroboration for the log but never promote past
        # LOCAL_REFERENCE_USED.
        try:
            g_mu = _grid_mu(params, u)
            rel = abs(mu - g_mu) / max(abs(g_mu), 1e-300)
            validation = {"ref_mu": g_mu, "rel": rel, "ref_trustworthy": False,
                          "method": "image_plane_flux_grid (patch "
                          "corroboration only; inverse-ray hangs on the lens, "
                          "source-plane diverges inside the disk)"}
            if rel > 1.0e-2:
                notes.append(f"image-plane-grid corroboration of the singular "
                             f"patch is loose (rel {rel:.1e})")
                status = _worst_status(status, "TRANSPORT_TOLERANCE_FAILED")
        except Exception as exc:                             # pragma: no cover
            validation = {"error": repr(exc)}
    elif do_validate:
        ref_mu, trustworthy, corr_rel, method = _independent_mu(params, u)
        rel = abs(mu - ref_mu) / max(abs(ref_mu), 1e-300)
        validation = {"ref_mu": ref_mu, "rel": rel,
                      "ref_trustworthy": trustworthy,
                      "ref_corroboration_rel": corr_rel, "method": method}
        if not trustworthy:
            notes.append(f"independent references disagree "
                         f"(rel {corr_rel:.1e}); status not promoted")
        elif rel <= _VALIDATE_RTOL:
            status = _promote(status, "OK_VALIDATED")
        else:
            notes.append(f"independent reference disagrees with the "
                         f"holonomic value (rel {rel:.1e} > {_VALIDATE_RTOL:g})")
            status = _worst_status(status, "TRANSPORT_TOLERANCE_FAILED")

    return EpochResult(
        params=params, u=u, mu=mu, mu_uniform=mu_u, F0=F0, F_half=F_half,
        r_max=r_max, status=status, mode=mode, cells=cells, notes=notes,
        validation=validation, seconds=time.perf_counter() - t0,
        jacobian=jacobian)


def magnification(params: LensParams, u: float = 0.0, **kw) -> float:
    """Just the number.  ``solve_epoch(...).mu``."""
    return solve_epoch(params, u, **kw).mu


# ------------------------------------------------------------------ status glue
_PROMOTABLE = {"OK"}


def _promote(current: str, better: str) -> str:
    """Raise ``current`` to ``better`` only if ``current`` is a clean ``OK``."""
    return better if current in _PROMOTABLE else current


def _worst_status(a: str, b: str) -> str:
    """The more severe of two statuses (``_worst`` extended with M5 names)."""
    sev = {"OK": 0, "OK_VALIDATED": 0, "OK_ESTIMATED": 1,
           "LOCAL_REFERENCE_USED": 2, "TOPOLOGY_UNCERTAIN": 3,
           "CONNECTION_ILL_CONDITIONED": 4, "TRANSPORT_TOLERANCE_FAILED": 5,
           "BASIS_DEGENERATE": 6, "GRADIENT_UNRELIABLE": 6,
           "ARC_TOPOLOGY_INVALID": 7, "EVENT_UNRESOLVED": 7,
           "RESOURCE_LIMIT": 8}
    return a if sev.get(a, 9) >= sev.get(b, 9) else b
