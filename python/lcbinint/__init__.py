from . import _lcbinint as _native
from ._lcbinint import *          # noqa: F401, F403
from ._lcbinint import obs
from . import image
from .image import ImagePlane
from .warmup import WarmupReport

_NativeOptions = _native.Options


def _normalize_t_lim(value):
    if value is None:
        return None
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError("t_lim must be None or a two-element sequence") from error
    if len(values) != 2:
        raise ValueError("t_lim must contain exactly two time limits")
    try:
        lower, upper = (float(item) for item in values)
    except (TypeError, ValueError) as error:
        raise TypeError("t_lim values must be real numbers") from error
    import math

    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("t_lim values must be finite")
    if not lower < upper:
        raise ValueError("t_lim must satisfy lower < upper")
    return (lower, upper)


def _ephemeris_t_lim(t_lim):
    if t_lim is None:
        return None
    return tuple(
        value + 2450000.0 if value < 2450000.0 else value
        for value in t_lim
    )


class Options:
    """Public numerical options plus Python-only execution controls.

    ``jax`` and ``t_lim`` deliberately live outside ``lcbi_options`` so these
    controls do not change the layout of the public C ABI structure.
    """

    __slots__ = ("_native", "_jax", "_t_lim", "_t_lim_locked")

    def __init__(self, *args, jax=False, t_lim=None, **kwargs):
        self._native = _NativeOptions(*args, **kwargs)
        self._jax = bool(jax)
        self._t_lim = _normalize_t_lim(t_lim)
        self._t_lim_locked = False

    @classmethod
    def _from_native(cls, native, *, jax=False, t_lim=None):
        result = cls.__new__(cls)
        object.__setattr__(result, "_native", native)
        object.__setattr__(result, "_jax", bool(jax))
        object.__setattr__(result, "_t_lim", _normalize_t_lim(t_lim))
        object.__setattr__(result, "_t_lim_locked", True)
        return result

    @property
    def jax(self):
        return self._jax

    @property
    def t_lim(self):
        return self._t_lim

    def __getattr__(self, name):
        native = object.__getattribute__(self, "_native")
        return getattr(native, name)

    def __setattr__(self, name, value):
        if name == "jax":
            object.__setattr__(self, "_jax", bool(value))
        elif name == "t_lim":
            if getattr(self, "_t_lim_locked", False):
                raise AttributeError(
                    "LightCurve.options.t_lim is fixed at construction"
                )
            object.__setattr__(self, "_t_lim", _normalize_t_lim(value))
        elif name in self.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._native, name, value)

    def __repr__(self):
        try:
            native = repr(object.__getattribute__(self, "_native"))
        except AttributeError:
            return "<lc.Options (uninitialized)>"
        rendered = native.replace(
            "<lc.Options ",
            f"<lc.Options backend='{'jax' if self.jax else 'native'}' ",
            1,
        )
        if self.t_lim is not None:
            rendered = rendered[:-1] + f" t_lim={self.t_lim}>"
        return rendered


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


def _warmup_value_fingerprint(value):
    def scalar(item):
        if isinstance(item, float):
            import math

            if math.isnan(item):
                return ("float", "nan")
            if math.isinf(item):
                return ("float", "+inf" if item > 0.0 else "-inf")
        return item

    try:
        import numpy as np

        array = np.asarray(value)
        if array.ndim == 0:
            return scalar(array.item())
        return (
            str(array.dtype),
            tuple(array.shape),
            tuple(scalar(item) for item in array.ravel().tolist()),
        )
    except Exception:
        return repr(value)


def _warmup_parameter_fingerprint(values):
    return tuple(
        sorted((str(key), _warmup_value_fingerprint(value))
               for key, value in values.items())
    )

if not hasattr(LimbDarkening, "quadratic"):
    LimbDarkening.quadratic = staticmethod(lambda c, d: LimbDarkening(c, d))


class LightCurve:
    def __init__(self, *args, orbital_motion_mode=None, **kwargs):
        if orbital_motion_mode is not None:
            kwargs["orbital_motion"] = _orbital_motion_name(orbital_motion_mode)
        jax_argument = kwargs.pop("jax", None)
        jax = bool(jax_argument) if jax_argument is not None else False
        t_lim = None
        positional = list(args)
        if positional and isinstance(positional[0], Options):
            options = positional[0]
            positional[0] = options._native
            t_lim = options.t_lim
            if jax_argument is None:
                jax = options.jax
        elif isinstance(kwargs.get("options"), Options):
            options = kwargs["options"]
            kwargs["options"] = options._native
            t_lim = options.t_lim
            if jax_argument is None:
                jax = options.jax
        if t_lim is not None:
            site = None
            if len(positional) >= 4:
                site = positional[3]
            elif "site" in kwargs:
                site = kwargs["site"]
            if (
                site is not None
                and getattr(site, "kind", None) == "space"
                and site.has_position
            ):
                lower, upper = _ephemeris_t_lim(t_lim)
                limited_site = site._limited_to(lower, upper)
                if len(positional) >= 4:
                    positional[3] = limited_site
                else:
                    kwargs["site"] = limited_site
        self._native = _NativeLightCurve(*positional, **kwargs)
        self.limb_darkening = LimbDarkening(self._native.ld_c, self._native.ld_d)
        self._options = Options._from_native(
            self._native.options, jax=jax, t_lim=t_lim
        )
        self._warmup_profile = None
        self._warmup_methods = None
        self._warmup_values = None
        self.lens = self._native.lens
        if jax and self._native.model.parallax:
            # Materialize the ephemeris outside any later JAX trace.
            from .jax_backend import _earth_ephemeris

            _earth_ephemeris(t_lim, self._native.t_ref)

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

    def _validate_time_limit(self, times):
        if not self.parallax or self._options.t_lim is None or times is None:
            return
        try:
            import numpy as np

            concrete = np.asarray(times)
        except Exception:
            # A traced JAX time is checked with a device-side mask.
            return
        lower, upper = self._options.t_lim
        if not np.all(np.isfinite(concrete)):
            raise ValueError("times must be finite")
        if np.any(concrete < lower) or np.any(concrete > upper):
            raise ValueError(
                f"times must lie within Options.t_lim=[{lower}, {upper}]"
            )

    def _warmup_configuration_fingerprint(self):
        options = self._options
        model = self._native.model
        sky = model.sky
        sky_fingerprint = None if sky is None else (
            float(sky.ra_deg),
            float(sky.dec_deg),
        )
        return (
            model.lens,
            model.source,
            bool(model.finite_source),
            float(self._native.ld_c),
            float(self._native.ld_d),
            options.param_type,
            options.inverse_ray_grid,
            int(options.caustic_bins),
            float(options.grid_ratio),
            options.polar_source_bins,
            options.polar_grid_ratio,
            int(options.max_source_bins),
            float(options.finite_source_tol),
            float(options.finite_source_reltol),
            float(options.point_source_threshold),
            float(options.hexadecapole_threshold),
            float(options.adaptive_hex_threshold),
            bool(options.jax),
            model.orbital_motion,
            model.xallarap,
            model.source_orbit_coordinates,
            bool(model.parallax),
            bool(model.terrestrial),
            sky_fingerprint,
            model.t_ref,
        )

    @property
    def warmup_profile(self):
        return self._warmup_profile

    @property
    def warmup_plan(self):
        return self._warmup_profile

    def clear_warmup(self):
        self._warmup_profile = None
        self._warmup_methods = None
        self._warmup_values = None

    def warmup(self, times, params=None, **kwargs):
        """Prepare the selected backend for repeated evaluation.

        Native execution retains a baseline-anchored per-epoch plan and exact
        result. JAX execution compiles and synchronously executes the actual
        XLA path for this shape, dtype, and static configuration; it deliberately
        does not memoize values because doing so would sever JAX gradients.
        ``grid_timing_repeats`` applies only to native plan calibration.
        """
        grid_timing_repeats = kwargs.pop("grid_timing_repeats", 1)
        self._validate_time_limit(times)
        merged = self._merge_params(params, **kwargs)
        parameter_fingerprint = _warmup_parameter_fingerprint(merged)
        configuration_fingerprint = self._warmup_configuration_fingerprint()
        if self._options.jax:
            from .warmup import build_jax_warmup_report

            report = build_jax_warmup_report(
                self,
                times,
                merged,
                parameter_fingerprint=parameter_fingerprint,
                configuration_fingerprint=configuration_fingerprint,
            )
            import numpy as np

            self._warmup_profile = report
            self._warmup_methods = np.zeros(report.times.size, dtype=np.int64)
            self._warmup_values = None
            return report
        from .warmup import build_warmup_report

        report, methods = build_warmup_report(
            self,
            times,
            merged,
            parameter_fingerprint=parameter_fingerprint,
            configuration_fingerprint=configuration_fingerprint,
            grid_timing_repeats=grid_timing_repeats,
        )
        # A retained plan is valid only for the exact time, parameter, and
        # configuration fingerprints checked by _matching_warmup(). Compute
        # that deterministic result once now and retain it as immutable data.
        # This is stronger than retaining image seeds alone: repeated exact
        # calls need neither trajectory reconstruction, caustic scans, root
        # solves, support certification, nor another inverse-ray traversal.
        values = self._native._magnification_preplanned(
            times,
            merged,
            methods.tolist(),
            report.resolutions.tolist(),
        )
        import numpy as np

        values = np.asarray(values, dtype=float).copy()
        values.setflags(write=False)
        self._warmup_profile = report
        self._warmup_methods = methods
        self._warmup_values = values
        return report

    def _matching_warmup(self, times, merged):
        profile = self._warmup_profile
        if profile is None or self._warmup_methods is None:
            return False
        try:
            import numpy as np

            concrete_times = np.asarray(times, dtype=float)
            if concrete_times.ndim == 0:
                concrete_times = concrete_times.reshape(1)
            return (
                np.array_equal(concrete_times, profile.times)
                and _warmup_parameter_fingerprint(merged)
                == profile.parameter_fingerprint
                and self._warmup_configuration_fingerprint()
                == profile.configuration_fingerprint
            )
        except Exception:
            return False

    def __call__(self, times, params=None, **kwargs):
        self._validate_time_limit(times)
        merged = self._merge_params(params, **kwargs)
        if self._options.jax:
            from .jax_backend import magnification

            return magnification(self._native, self._options, times, merged)
        if self._matching_warmup(times, merged):
            if self._warmup_values is not None:
                # Return independent storage so a caller cannot mutate the
                # retained exact-result cache through the public array.
                return self._warmup_values.copy()
            try:
                return self._native._magnification_preplanned(
                    times,
                    merged,
                    self._warmup_methods.tolist(),
                    self._warmup_profile.resolutions.tolist(),
                )
            except RuntimeError:
                # A support or numerical failure invalidates the optimization
                # for this call; normal auto remains the fail-closed authority.
                pass
        return self._native(times, merged)

    def magnification(self, times, params=None, **kwargs):
        return self.__call__(times, params, **kwargs)

    def magnification_batch(self, times, parameter_rows):
        """Evaluate independent parameter rows with the selected backend.

        Native execution uses one GIL-free C++ call. JAX execution preserves
        the same row-major result shape and composes the public differentiable
        callable for each statically supplied parameter mapping.
        """
        self._validate_time_limit(times)
        merged = [self._merge_params(params) for params in parameter_rows]
        if self._options.jax:
            from .jax_backend import magnification_batch

            return magnification_batch(
                self._native, self._options, times, merged
            )
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
        """Evaluate the selected backend's fused/batched likelihood path."""
        self._validate_time_limit(times)
        merged = [self._merge_params(params) for params in parameter_rows]
        if self._options.jax:
            from .jax_backend import light_curve_log_likelihood_batch

            return light_curve_log_likelihood_batch(
                self._native,
                self._options,
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
        self._validate_time_limit(times)
        if self._options.jax:
            from .jax_backend import info

            return info(
                self._native,
                self._options,
                times,
                self._merge_params(params, **kwargs),
            )
        return self._native.info(times, self._merge_params(params, **kwargs))

    def source_trajectory(self, times, params=None, **kwargs):
        self._validate_time_limit(times)
        if self._options.jax:
            from .jax_backend import source_trajectory

            return source_trajectory(
                self._native,
                self._options,
                times,
                self._merge_params(params, **kwargs),
            )
        return self._native.source_trajectory(times, **self._merge_params(params, **kwargs))

    def binary_source_components(self, times, params=None, **kwargs):
        """Return individual binary-source curves and trajectories.

        The returned ``BinarySourceComponents`` contains ``source1`` and
        ``source2`` components, each with ``magnification`` and ``trajectory``,
        plus the flux-weighted ``total`` curve. This method requires
        ``LightCurve(source="binary")``.
        """
        self._validate_time_limit(times)
        merged = self._merge_params(params, **kwargs)
        if self._options.jax:
            from .jax_backend import binary_source_components

            return binary_source_components(
                self._native, self._options, times, merged
            )
        return self._native.binary_source_components(times, merged)

    def finite_source_geometry(self, times, params=None, **kwargs):
        """Return trajectory-resolved geometry without solving lens roots."""
        self._validate_time_limit(times)
        if self._options.jax:
            from .jax_backend import finite_source_geometry

            return finite_source_geometry(
                self._native,
                self._options,
                times,
                self._merge_params(params, **kwargs),
            )
        return self._native.finite_source_geometry(
            times, **self._merge_params(params, **kwargs)
        )

    def separation(self, time=None, params=None, **kwargs):
        if params is None and isinstance(time, (dict, Parameters)):
            params = time
            time = None
        merged = self._merge_params(params, **kwargs)
        self._validate_time_limit(time)
        if self._options.jax:
            from .jax_backend import separation

            return separation(
                self._native, self._options, time, merged
            )
        if time is None:
            return self._native.separation(merged)
        return self._native.separation(time, merged)

    def caustics(self, time=None, params=None, **kwargs):
        if params is None and isinstance(time, (dict, Parameters)):
            params = time
            time = None
        merged = self._merge_params(params, **kwargs)
        self._validate_time_limit(time)
        if time is None:
            return self._native.caustics(merged)
        return self._native.caustics(time, merged)

    def critical_curves(self, time=None, params=None, **kwargs):
        if params is None and isinstance(time, (dict, Parameters)):
            params = time
            time = None
        merged = self._merge_params(params, **kwargs)
        self._validate_time_limit(time)
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


def binary_ray_shooting(
    x,
    y,
    *,
    s,
    q,
    rho,
    limb_darkening=None,
    options=None,
    jax=None,
):
    """Evaluate a direct finite binary source with the selected backend.

    ``jax=True`` is the only API difference between native and differentiable
    execution. A public ``Options(jax=True)`` selects the same path; an
    explicit ``jax=...`` argument takes precedence, matching ``LightCurve``.
    """

    if limb_darkening is None:
        limb_darkening = LimbDarkening()
    if options is None:
        options = Options()
    if isinstance(options, Options):
        use_jax = options.jax if jax is None else bool(jax)
        native_options = options._native
    elif isinstance(options, _NativeOptions):
        use_jax = bool(jax) if jax is not None else False
        native_options = options
    else:
        raise TypeError("options must be lcbinint.Options or native Options")
    if use_jax:
        from .jax_backend import binary_ray_shooting as jax_binary_ray_shooting

        return jax_binary_ray_shooting(
            x,
            y,
            s=s,
            q=q,
            rho=rho,
            limb_darkening=limb_darkening,
            options=native_options,
        )
    return _native_binary_ray_shooting(
        x,
        y,
        s=s,
        q=q,
        rho=rho,
        limb_darkening=limb_darkening,
        options=native_options,
    )


__all__ = [
    "obs",
    "Options", "Parameters", "LensParams", "LimbDarkening", "LightCurve",
    "LightCurveInfo", "SourceTrajectory", "BinarySourceComponent",
    "BinarySourceComponents", "GeometryBranches", "Model",
    "OrbitalMotionMode", "XallarapParamType",
    "binary_ray_shooting",
    "image", "ImagePlane",
]
