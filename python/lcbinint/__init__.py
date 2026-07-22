from . import _lcbinint as _native
from ._lcbinint import *          # noqa: F401, F403
from ._lcbinint import obs
from . import image
from .image import ImagePlane
from .vbm import binary_source_binary_lens as vbm_binary_source_binary_lens
import numpy as _np

Options = _native.Options
Parameters = _native.Parameters
# Compatibility name for the parameter container. Unlike the removed
# function-style evaluators, this does not create an alternate evaluation API.
LensParams = _native.Parameters
LimbDarkening = _native.LimbDarkening
_NativeLightCurve = _native.LightCurve
LightCurveInfo = _native.LightCurveInfo
SourceTrajectory = _native.SourceTrajectory
GeometryBranches = _native.GeometryBranches
Model = _native.Model
OrbitalMotionMode = _native.OrbitalMotionMode
XallarapParamType = _native.XallarapParamType
_binary_images = _native._binary_images
_binary_safety_diagnostic = _native._binary_safety_diagnostic

_PARAMETER_FIELDS = {
    "t0": "t0", "tE": "tE", "u0": "u0", "alpha": "alpha",
    "s": "s", "q": "q", "rho": "rho", "q2": "q2", "sep2": "sep2",
    "ang": "ang", "piEN": "piEN", "piEE": "piEE", "ra": "ra",
    "dec": "dec", "tfix": "tfix", "obs_lat": "obs_lat", "obs_lon": "obs_lon",
    "orbital_motion_mode": "orbital_motion_mode", "g1": "g1", "g2": "g2",
    "g3": "g3", "lom_szs": "lom_szs", "lom_ar": "lom_ar", "v_sep": "v_sep",
    "xi_1": "xi_1", "xi_2": "xi_2", "omega_xa": "omega_xa",
    "inc_xa": "inc_xa", "phi_xa": "phi_xa",
    "limb_darkening_c": "limb_darkening_c",
    "limb_darkening_d": "limb_darkening_d",
}

if not hasattr(LimbDarkening, "quadratic"):
    LimbDarkening.quadratic = staticmethod(lambda c, d: LimbDarkening(c, d))


class LightCurve:
    def __init__(self, *args, orbital_motion_mode=None, **kwargs):
        self.limb_darkening = kwargs.get("limb_darkening", LimbDarkening.none())
        if orbital_motion_mode is not None:
            kwargs["orbital_motion"] = _orbital_motion_name(orbital_motion_mode)
        self._native = _NativeLightCurve(*args, **kwargs)
        self.lens = self._native.lens

    @property
    def parallax(self):
        return bool(self._native.model.parallax)

    def _merge_params(self, params=None, **kwargs):
        if params is None:
            return dict(kwargs)
        elif isinstance(params, dict):
            if not kwargs:
                return params
            merged = dict(params)
        elif isinstance(params, Parameters):
            merged = {
                key: getattr(params, attribute)
                for key, attribute in _PARAMETER_FIELDS.items()
            }
        else:
            raise TypeError("params must be a dict, Parameters, or None")
        merged.update(kwargs)
        return merged

    def __call__(self, times, params=None, **kwargs):
        return self._native(times, self._merge_params(params, **kwargs))

    def magnification(self, times, params=None, **kwargs):
        return self.__call__(times, params, **kwargs)

    def magnification_batch(self, times, parameter_rows):
        """Evaluate independent parameter rows in one native call.

        Existing scalar APIs are unchanged. Inference libraries use this
        method to avoid one Python/pybind dispatch per walker.
        """
        merged = [self._merge_params(params) for params in parameter_rows]
        return self._native.magnification_batch(times, merged)

    @property
    def supports_fused_light_curve_likelihood(self):
        return True

    def light_curve_log_likelihood_batch(
        self,
        times,
        flux,
        flux_err,
        parameter_rows,
        distribution="gaussian",
        flux_mode="fit",
        nu=4.0,
        source_flux=None,
        blend_flux=None,
    ):
        """Internal inference fast path; scalar user APIs stay unchanged."""
        merged = [self._merge_params(params) for params in parameter_rows]
        return self._native.light_curve_log_likelihood_batch(
            times,
            flux,
            flux_err,
            merged,
            distribution,
            flux_mode,
            nu,
            source_flux,
            blend_flux,
        )

    def info(self, times, params=None, **kwargs):
        return self._native.info(times, self._merge_params(params, **kwargs))

    def source_trajectory(self, times, params=None, **kwargs):
        return self._native.source_trajectory(times, **self._merge_params(params, **kwargs))

    def finite_source_geometry(self, times, params=None, **kwargs):
        """Return trajectory-resolved geometry without solving lens roots."""
        return self._native.finite_source_geometry(
            times, **self._merge_params(params, **kwargs)
        )

    def separation(self, time=None, params=None, **kwargs):
        if params is None and isinstance(time, (dict, Parameters)):
            params = time
            time = None
        merged = self._merge_params(params, **kwargs)
        if time is None:
            return self._native.separation(merged)
        return self._native.separation(time, merged)

    def caustics(self, time=None, params=None, **kwargs):
        if params is None and isinstance(time, (dict, Parameters)):
            params = time
            time = None
        merged = self._merge_params(params, **kwargs)
        if time is None:
            return self._native.caustics(merged)
        return self._native.caustics(time, merged)

    def critical_curves(self, time=None, params=None, **kwargs):
        if params is None and isinstance(time, (dict, Parameters)):
            params = time
            time = None
        merged = self._merge_params(params, **kwargs)
        if time is None:
            return self._native.critical_curves(merged)
        return self._native.critical_curves(time, merged)

    def __getattr__(self, name):
        return getattr(self._native, name)

    def __repr__(self):
        return repr(self._native)


def _orbital_motion_name(mode):
    if mode == OrbitalMotionMode.CIRCULAR or int(mode) == int(OrbitalMotionMode.CIRCULAR):
        return "circular"
    if mode == OrbitalMotionMode.KEPLER or int(mode) == int(OrbitalMotionMode.KEPLER):
        return "kepler"
    return "static"


__all__ = [
    "obs",
    "Options", "Parameters", "LensParams", "LimbDarkening", "LightCurve",
    "LightCurveInfo", "SourceTrajectory", "GeometryBranches", "Model",
    "OrbitalMotionMode", "XallarapParamType",
    "binary_ray_shooting",
    "vbm_binary_source_binary_lens",
    "image", "ImagePlane",
]
