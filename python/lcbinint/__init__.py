from . import _lcbinint as _native
from ._lcbinint import *          # noqa: F401, F403
from ._lcbinint import obs
from . import image
from .image import ImagePlane

_NativeOptions = _native.Options


class Options:
    """Public numerical options plus a Python-only JAX backend selector.

    The selector deliberately lives outside ``lcbi_options`` so adding JAX
    support does not change the layout of the public C ABI structure.
    """

    __slots__ = ("_native", "_jax")

    def __init__(self, *args, jax=False, **kwargs):
        self._native = _NativeOptions(*args, **kwargs)
        self._jax = bool(jax)

    @classmethod
    def _from_native(cls, native, *, jax=False):
        result = cls.__new__(cls)
        object.__setattr__(result, "_native", native)
        object.__setattr__(result, "_jax", bool(jax))
        return result

    @property
    def jax(self):
        return self._jax

    def __getattr__(self, name):
        native = object.__getattribute__(self, "_native")
        return getattr(native, name)

    def __setattr__(self, name, value):
        if name == "jax":
            object.__setattr__(self, "_jax", bool(value))
        elif name in self.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._native, name, value)

    def __repr__(self):
        try:
            native = repr(object.__getattribute__(self, "_native"))
        except AttributeError:
            return "<lc.Options (uninitialized)>"
        return native.replace(
            "<lc.Options ",
            f"<lc.Options backend='{'jax' if self.jax else 'native'}' ",
            1,
        )


Parameters = _native.Parameters
# Compatibility name for the parameter container. Unlike the removed
# function-style evaluators, this does not create an alternate evaluation API.
LensParams = _native.Parameters
LimbDarkening = _native.LimbDarkening
_NativeLightCurve = _native.LightCurve
LightCurveInfo = _native.LightCurveInfo
SourceTrajectory = _native.SourceTrajectory
BinarySourceComponent = _native.BinarySourceComponent
BinarySourceComponents = _native.BinarySourceComponents
GeometryBranches = _native.GeometryBranches
Model = _native.Model
OrbitalMotionMode = _native.OrbitalMotionMode
XallarapParamType = _native.XallarapParamType
_binary_images = _native._binary_images
_binary_safety_diagnostic = _native._binary_safety_diagnostic
_native_binary_ray_shooting = _native.binary_ray_shooting

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
        if orbital_motion_mode is not None:
            kwargs["orbital_motion"] = _orbital_motion_name(orbital_motion_mode)
        jax_argument = kwargs.pop("jax", None)
        jax = bool(jax_argument) if jax_argument is not None else False
        positional = list(args)
        if positional and isinstance(positional[0], Options):
            options = positional[0]
            positional[0] = options._native
            if jax_argument is None:
                jax = options.jax
        elif isinstance(kwargs.get("options"), Options):
            options = kwargs["options"]
            kwargs["options"] = options._native
            if jax_argument is None:
                jax = options.jax
        self._native = _NativeLightCurve(*positional, **kwargs)
        self.limb_darkening = LimbDarkening(self._native.ld_c, self._native.ld_d)
        self._options = Options._from_native(
            self._native.options, jax=jax
        )
        self.lens = self._native.lens

    @property
    def options(self):
        return self._options

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
        merged = self._merge_params(params, **kwargs)
        if self._options.jax:
            from .jax_backend import magnification

            return magnification(self._native, self._native.options, times, merged)
        return self._native(times, merged)

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

    def binary_source_components(self, times, params=None, **kwargs):
        """Return individual binary-source curves and trajectories.

        The returned ``BinarySourceComponents`` contains ``source1`` and
        ``source2`` components, each with ``magnification`` and ``trajectory``,
        plus the flux-weighted ``total`` curve. This method requires
        ``LightCurve(source="binary")``.
        """
        return self._native.binary_source_components(
            times, self._merge_params(params, **kwargs)
        )

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


def binary_ray_shooting(*args, **kwargs):
    """Native inverse-ray helper accepting the public :class:`Options`."""

    options = kwargs.get("options")
    if isinstance(options, Options):
        kwargs = dict(kwargs)
        kwargs["options"] = options._native
    return _native_binary_ray_shooting(*args, **kwargs)


__all__ = [
    "obs",
    "Options", "Parameters", "LensParams", "LimbDarkening", "LightCurve",
    "LightCurveInfo", "SourceTrajectory", "BinarySourceComponent",
    "BinarySourceComponents", "GeometryBranches", "Model",
    "OrbitalMotionMode", "XallarapParamType",
    "binary_ray_shooting",
    "image", "ImagePlane",
]
