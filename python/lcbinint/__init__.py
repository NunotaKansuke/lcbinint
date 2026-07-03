from ._lcbinint import *          # noqa: F401, F403
from ._lcbinint import lc, obs, bayes, optimize, sample
from .sampler import SamplerOptions, run_sampler, load_chain, Reparameterization
import numpy as _np

Options = lc.Options
Parameters = lc.Parameters
LensParams = lc.Parameters
LimbDarkening = lc.LimbDarkening
_NativeLightCurve = lc.LightCurve
LightCurveInfo = lc.LightCurveInfo
SourceTrajectory = lc.SourceTrajectory
GeometryBranches = lc.GeometryBranches
Effects = lc.Effects
OrbitalMotionMode = lc.OrbitalMotionMode
XallarapParamType = lc.XallarapParamType

if not hasattr(LimbDarkening, "quadratic"):
    LimbDarkening.quadratic = staticmethod(lambda c, d: LimbDarkening(c, d))


class EventCoordinates:
    def __init__(self, ra=0.0, dec=0.0, tfix=0.0):
        self.ra = ra
        self.dec = dec
        self.tfix = tfix


class _LightCurveInfoCompat:
    def __init__(self, times, info):
        self.times = list(times)
        for name in (
            "magnifications",
            "point_source_magnifications",
            "finite_source_magnifications",
            "source_x",
            "source_y",
            "image_counts",
            "finite_source_methods",
            "finite_source_method_names",
            "finite_source_error_estimates",
            "finite_source_refinement_levels",
            "finite_source_converged",
            "root_candidate_counts",
            "root_duplicate_counts",
            "root_polish_failure_counts",
            "root_used_warm_start",
            "root_used_cold_retry",
            "root_used_high_precision",
            "root_needs_high_precision",
            "root_max_residuals",
            "all_converged",
            "unconverged_indices",
        ):
            setattr(self, name, getattr(info, name))


class LightCurve:
    def __init__(self, *args, event=None, orbital_motion_mode=None, **kwargs):
        self._event = event
        self.event = event
        self.lens = kwargs.get("lens", "binary_lens")
        self.limb_darkening = kwargs.get("limb_darkening", LimbDarkening.none())
        self.parallax = bool(kwargs.get("parallax", False))
        if orbital_motion_mode is not None:
            kwargs["orbital_motion"] = _orbital_motion_name(orbital_motion_mode)
        if event is not None:
            kwargs.setdefault("t_ref", event.tfix)
        self._native = _NativeLightCurve(*args, **kwargs)

    def _with_event(self, params):
        if self._event is None:
            return params
        if isinstance(params, dict):
            out = dict(params)
            out.setdefault("ra", self._event.ra)
            out.setdefault("dec", self._event.dec)
            out.setdefault("tfix", self._event.tfix)
            return out
        return params

    def _merge_params(self, params=None, **kwargs):
        if params is None:
            merged = dict(kwargs)
        elif isinstance(params, dict):
            merged = dict(params)
            merged.update(kwargs)
        else:
            return params
        return self._with_event(merged)

    def __call__(self, times, params=None, **kwargs):
        return self._native(times, self._merge_params(params, **kwargs))

    def magnification(self, times, params=None, **kwargs):
        return self.__call__(times, params, **kwargs)

    def light_curve(self, times, params=None, **kwargs):
        return self.__call__(times, params, **kwargs)

    def list(self, times, params=None, **kwargs):
        return self.__call__(times, params, **kwargs).tolist()

    def info(self, times, params=None, **kwargs):
        return self._native.info(times, self._merge_params(params, **kwargs))

    def source_trajectory(self, times, params=None, **kwargs):
        return self._native.source_trajectory(times, **self._merge_params(params, **kwargs))

    def separation(self, time=None, params=None, **kwargs):
        if params is None and isinstance(time, dict):
            params = time
            time = None
        merged = self._merge_params(params, **kwargs)
        if time is None:
            return self._native.separation(merged)
        return self._native.separation(time, merged)

    def caustics(self, time=None, params=None, **kwargs):
        if params is None and isinstance(time, dict):
            params = time
            time = None
        merged = self._merge_params(params, **kwargs)
        if time is None:
            return self._native.caustics(merged)
        return self._native.caustics(time, merged)

    def critical_curves(self, time=None, params=None, **kwargs):
        if params is None and isinstance(time, dict):
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


def _split_light_curve_kwargs(kwargs):
    kwargs = dict(kwargs)
    options = kwargs.pop("options", None) or Options()
    limb_darkening = kwargs.pop("limb_darkening", None) or LimbDarkening.none()
    event = kwargs.pop("event", None)
    orbital_motion_mode = kwargs.get("orbital_motion_mode", OrbitalMotionMode.STATIC)
    if event is not None:
        kwargs.setdefault("ra", event.ra)
        kwargs.setdefault("dec", event.dec)
        kwargs.setdefault("tfix", event.tfix)
    effects = {
        "orbital_motion": _orbital_motion_name(orbital_motion_mode),
        "parallax": abs(kwargs.get("piEN", 0.0)) > 0.0 or abs(kwargs.get("piEE", 0.0)) > 0.0,
    }
    if event is not None:
        effects["event"] = event
    return options, limb_darkening, effects, kwargs


def light_curve_info(times, **kwargs):
    options, limb_darkening, effects, params = _split_light_curve_kwargs(kwargs)
    curve = LightCurve(
        lens="binary_lens",
        options=options,
        limb_darkening=limb_darkening,
        **effects,
    )
    return _LightCurveInfoCompat(times, curve.info(times, params))


def binary_light_curve(times, **kwargs):
    return light_curve_info(times, **kwargs).magnifications


def light_curve(times, **kwargs):
    return _np.asarray(binary_light_curve(times, **kwargs))


def magnification(time, **kwargs):
    return binary_light_curve([time], **kwargs)[0]


def binary_magnification(time, **kwargs):
    return magnification(time, **kwargs)


def binary_mag0(separation, mass_ratio, y1, y2):
    return magnification(
        y1,
        t0=0.0,
        tE=1.0,
        u0=y2,
        alpha=0.0,
        s=separation,
        q=mass_ratio,
        rho=0.0,
        options=Options(coordinates="center_of_mass"),
    )


ParallaxLightCurve = LightCurve
ParallaxLightCurveFunc = LightCurve


# Build the Python-extended Model subclass and replace bayes.Model
from .model import _build_model_class as _bmc
bayes.Model = _bmc(bayes._Model)
del _bmc

__all__ = [
    "lc", "obs", "bayes", "optimize", "sample",
    "Options", "Parameters", "LensParams", "LimbDarkening", "LightCurve",
    "LightCurveInfo", "SourceTrajectory", "GeometryBranches", "Effects",
    "OrbitalMotionMode", "XallarapParamType",
    "EventCoordinates", "light_curve_info", "binary_light_curve", "light_curve",
    "magnification", "binary_mag0",
    "SamplerOptions", "run_sampler", "load_chain", "Reparameterization",
]
