from dataclasses import dataclass

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


def _normalize_finite_source_backend(value):
    if value is None:
        return None
    if not isinstance(value, str) or value not in ("default", "algebraic_boundary"):
        raise ValueError(
            "finite_source_backend must be None, 'default', or 'algebraic_boundary'"
        )
    return value


def _default_finite_source_backend(native_curve):
    """Select the branch default for the model supported by this backend."""
    model = native_curve.model
    if (
        model.lens == "binary"
        and model.source == "single"
        and not model.parallax
        and model.xallarap == "none"
        and native_curve.orbital_motion == "static"
    ):
        return "algebraic_boundary"
    return "default"


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
    ``finite_source_backend=None`` selects the branch default after the model
    is known: static single-source binary curves use algebraic boundary, while
    models outside that backend's support keep the standard router. Pass
    ``'default'`` explicitly to request the standard router for a binary curve.
    """

    __slots__ = (
        "_native", "_jax", "_t_lim", "_t_lim_locked",
        "_finite_source_backend",
    )

    def __init__(
        self, *args, jax=False, t_lim=None, finite_source_backend=None,
        **kwargs,
    ):
        self._native = _NativeOptions(*args, **kwargs)
        self._jax = bool(jax)
        self._t_lim = _normalize_t_lim(t_lim)
        self._t_lim_locked = False
        self._finite_source_backend = _normalize_finite_source_backend(
            finite_source_backend
        )

    @classmethod
    def _from_native(
        cls, native, *, jax=False, t_lim=None, finite_source_backend=None
    ):
        result = cls.__new__(cls)
        object.__setattr__(result, "_native", native)
        object.__setattr__(result, "_jax", bool(jax))
        object.__setattr__(result, "_t_lim", _normalize_t_lim(t_lim))
        object.__setattr__(result, "_t_lim_locked", True)
        object.__setattr__(
            result,
            "_finite_source_backend",
            _normalize_finite_source_backend(finite_source_backend),
        )
        return result

    @property
    def jax(self):
        return self._jax

    @property
    def t_lim(self):
        return self._t_lim

    @property
    def finite_source_backend(self):
        return self._finite_source_backend

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
        elif name == "finite_source_backend":
            object.__setattr__(
                self, "_finite_source_backend",
                _normalize_finite_source_backend(value),
            )
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
        if self.finite_source_backend not in (None, "default"):
            rendered = rendered[:-1] + (
                f" finite_source_backend={self.finite_source_backend!r}>"
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
_NativeLightCurveInfo = _native.LightCurveInfo
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


@dataclass(frozen=True)
class LightCurveInfo:
    """One light-curve result, with optional gradients and per-epoch status.

    ``magnifications`` is the best finite estimate available for each epoch.
    If finite-source support cannot be established before an integral estimate
    exists, the point-source value is returned and diagnostics mark that
    fallback. ``status`` uses 0 for accepted values, 1 for unconverged estimates,
    and 2 when image support could not be established. A requested but unusable
    gradient is represented by zeros plus ``gradient_valid=False``. Supported
    eager static binary calls use the algebraic-boundary value path and its
    forward-mode Jacobian. Gradients are marked invalid when the value status
    is not accepted.

    The object converts to its magnification array for NumPy/JAX expressions,
    while retaining status and gradient metadata for callers that need it.
    """

    OK = 0
    UNCONVERGED = 1
    SUPPORT_UNPROVEN = 2

    times: object
    magnifications: object
    status: object
    gradients: object = None
    gradient_valid: object = None
    gradient_parameters: tuple = ()
    value_converged: object = None
    support_certified: object = None
    diagnostics: object = None

    __array_priority__ = 1000

    @property
    def statuses(self):
        """Human-readable per-epoch status names."""
        return self.status_names

    @property
    def status_codes(self):
        return self.status

    @property
    def finite_source_converged(self):
        return self.value_converged

    @property
    def finite_source_magnifications(self):
        """Alias for the estimate in ``magnifications``.

        If finite-source image support is unproven, this may be the point-source
        fallback; inspect ``status`` and ``diagnostics`` before using it as a
        finite-source result.
        """
        return self.magnifications

    @property
    def all_converged(self):
        import numpy as np

        return bool(np.all(np.asarray(self.status) == self.OK))

    @property
    def unconverged_indices(self):
        import numpy as np

        return np.flatnonzero(np.asarray(self.status) != self.OK).tolist()

    @property
    def status_names(self):
        import numpy as np

        names = {
            self.OK: "ok",
            self.UNCONVERGED: "unconverged",
            self.SUPPORT_UNPROVEN: "support_unproven",
        }
        return np.asarray(
            [names.get(int(code), "unconverged") for code in np.asarray(self.status).reshape(-1)],
            dtype=object,
        ).reshape(np.shape(self.status))

    @property
    def shape(self):
        return self.magnifications.shape

    @property
    def ndim(self):
        return self.magnifications.ndim

    @property
    def dtype(self):
        return self.magnifications.dtype

    def __array__(self, dtype=None, copy=None):
        import numpy as np

        values = np.asarray(self.magnifications, dtype=dtype)
        return values.copy() if copy else values

    def __jax_array__(self):
        import jax.numpy as jnp

        return jnp.asarray(self.magnifications)

    def __getitem__(self, key):
        return self.magnifications[key]

    def __len__(self):
        return len(self.magnifications)

    def __iter__(self):
        return iter(self.magnifications)

    def __getattr__(self, name):
        diagnostics = object.__getattribute__(self, "diagnostics")
        if diagnostics is not None:
            if isinstance(diagnostics, dict) and name in diagnostics:
                return diagnostics[name]
            if isinstance(diagnostics, dict):
                aliases = {
                    "finite_source_error_estimates": "error_estimate",
                    "finite_source_converged": "value_converged",
                    "finite_source_method_names": "finite_source_method_name",
                }
                key = aliases.get(name)
                if key in diagnostics:
                    return diagnostics[key]
            if hasattr(diagnostics, name):
                return getattr(diagnostics, name)
        values = object.__getattribute__(self, "magnifications")
        if hasattr(values, name):
            return getattr(values, name)
        raise AttributeError(name)

    def __sub__(self, other):
        return self.magnifications - other

    def __rsub__(self, other):
        return other - self.magnifications

    def __add__(self, other):
        return self.magnifications + other

    def __radd__(self, other):
        return other + self.magnifications

    def __mul__(self, other):
        return self.magnifications * other

    def __rmul__(self, other):
        return other * self.magnifications

    def __truediv__(self, other):
        return self.magnifications / other

    def __neg__(self):
        return -self.magnifications

    def tree_flatten(self):
        children = (
            self.times,
            self.magnifications,
            self.status,
            self.gradients,
            self.gradient_valid,
            self.value_converged,
            self.support_certified,
        )
        return children, self.gradient_parameters

    @classmethod
    def tree_unflatten(cls, gradient_parameters, children):
        times, magnifications, status, gradients, gradient_valid, converged, support = children
        return cls(
            times=times,
            magnifications=magnifications,
            status=status,
            gradients=gradients,
            gradient_valid=gradient_valid,
            gradient_parameters=gradient_parameters,
            value_converged=converged,
            support_certified=support,
        )


_lightcurve_info_jax_registered = False


def _register_lightcurve_info_pytree():
    global _lightcurve_info_jax_registered
    if _lightcurve_info_jax_registered:
        return
    import jax

    try:
        jax.tree_util.register_pytree_node_class(LightCurveInfo)
    except ValueError:
        pass
    _lightcurve_info_jax_registered = True

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
        finite_source_backend = None
        t_lim = None
        positional = list(args)
        if positional and isinstance(positional[0], Options):
            options = positional[0]
            positional[0] = options._native
            t_lim = options.t_lim
            finite_source_backend = options.finite_source_backend
            if jax_argument is None:
                jax = options.jax
        elif isinstance(kwargs.get("options"), Options):
            options = kwargs["options"]
            kwargs["options"] = options._native
            t_lim = options.t_lim
            finite_source_backend = options.finite_source_backend
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
        if finite_source_backend is None:
            finite_source_backend = _default_finite_source_backend(self._native)
        self.limb_darkening = LimbDarkening(self._native.ld_c, self._native.ld_d)
        self._options = Options._from_native(
            self._native.options,
            jax=jax,
            t_lim=t_lim,
            finite_source_backend=finite_source_backend,
        )
        self._validate_backend_combination()
        self._warmup_profile = None
        self._warmup_methods = None
        self._warmup_jax_function = None
        self._warmup_drift_warned = False
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
            options.finite_source_backend,
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
        self._warmup_jax_function = None
        self._warmup_drift_warned = False

    def warmup(self, times, params=None, **kwargs):
        """Prepare the selected backend for repeated evaluation.

        Both backends retain a baseline-anchored per-epoch plan for reuse
        across nearby parameter proposals. Native execution keeps its C++ plan;
        JAX compiles a grouped fixed-route/``nbin`` callable and executes it
        once during warm-up. No magnification values are memoized, so JAX
        gradients remain connected to the current parameters. For automatic
        ``nbin``, JAX takes the initial route/bin proposal from its own
        dispatcher; the native calibration remains the numerical reference,
        and a generic certification/refinement pass is run if the two
        discretisations do not yet agree.
        """
        if self._options.finite_source_backend == "algebraic_boundary":
            raise NotImplementedError(
                "warmup is not implemented for finite_source_backend='algebraic_boundary'"
            )
        grid_timing_repeats = kwargs.pop("grid_timing_repeats", 1)
        self._validate_time_limit(times)
        merged = self._merge_params(params, **kwargs)
        parameter_fingerprint = _warmup_parameter_fingerprint(merged)
        configuration_fingerprint = self._warmup_configuration_fingerprint()
        # A failed replacement warmup must not leave an older proposal
        # neighbourhood active under the caller's feet.
        self.clear_warmup()
        if self._options.jax:
            from .warmup import build_jax_warmup_report

            report = build_jax_warmup_report(
                self,
                times,
                merged,
                parameter_fingerprint=parameter_fingerprint,
                configuration_fingerprint=configuration_fingerprint,
                grid_timing_repeats=grid_timing_repeats,
            )
            import numpy as np
            from .warmup import METHOD_NAMES

            self._warmup_profile = report
            self._warmup_methods = np.asarray(
                [METHOD_NAMES.index(method) for method in report.methods],
                dtype=np.int64,
            )
            self._warmup_jax_function = report.execution_plan
            self._warmup_drift_warned = False
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
        if not report.all_calibrated:
            raise RuntimeError(
                "warmup could not calibrate every epoch; no partial plan was retained"
            )
        self._warmup_profile = report
        self._warmup_methods = methods
        self._warmup_drift_warned = False
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
                and self._warmup_configuration_fingerprint()
                == profile.configuration_fingerprint
            )
        except Exception:
            return False

    def warmup_drift(self, times, params=None, **kwargs):
        """Compare concrete proposal geometry with the retained warmup plan."""
        from .warmup import compare_warmup_geometry

        merged = self._merge_params(params, **kwargs)
        profile = self._warmup_profile
        geometry = None if profile is None else profile.geometry
        return compare_warmup_geometry(self, times, merged, geometry)

    def _warn_if_warmup_drift(self, times, merged):
        report = self.warmup_drift(times, merged)
        if not report.warn:
            self._warmup_drift_warned = False
            return report
        if not self._warmup_drift_warned:
            import warnings

            warnings.warn(
                "warmup geometry has moved outside its calibrated neighbourhood: "
                + "; ".join(report.reasons)
                + ". The retained epoch methods/resolutions are still being used.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warmup_drift_warned = True
        return report

    def _validate_backend_combination(self):
        if self._options.finite_source_backend != "algebraic_boundary":
            return
        model = self._native.model
        if model.lens != "binary" or model.source != "single":
            raise ValueError(
                "finite_source_backend='algebraic_boundary' requires a single-source binary lens"
            )
        if model.parallax or model.xallarap != "none" or self._native.orbital_motion != "static":
            raise ValueError(
                "finite_source_backend='algebraic_boundary' requires a static binary trajectory"
            )

    def _evaluate_values(self, times, merged, *, with_gradient=False):
        """Build the common value/status record for every public call mode."""
        import numpy as np

        self._validate_backend_combination()
        self._validate_time_limit(times)

        def contains_tracer(value):
            try:
                import jax
            except ImportError:
                return False
            if isinstance(value, jax.core.Tracer):
                return True
            if isinstance(value, dict):
                return any(contains_tracer(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return any(contains_tracer(item) for item in value)
            return False

        traced = contains_tracer(times) or contains_tracer(merged)
        if traced:
            _register_lightcurve_info_pytree()
            import jax.numpy as jnp
            from .jax_backend import magnification

            time_array = jnp.asarray(times)
            if time_array.ndim == 0:
                time_array = jnp.reshape(time_array, (1,))
            elif time_array.ndim != 1:
                raise ValueError("times must be a scalar or one-dimensional array")
            if (
                self._options.finite_source_backend != "algebraic_boundary"
                and self._native.lens == "binary"
                and self._native.model.source == "single"
            ):
                from .jax_backend import info as jax_info

                raw = jax_info(
                    self._native, self._options, time_array, merged
                )
                primary = jnp.ravel(raw.magnifications)
                finite_source = jnp.ravel(raw.finite_source_magnifications)
                point_source = jnp.ravel(raw.point_source_magnifications)
                estimates = jnp.full_like(primary, 1.0)
                for candidate in (point_source, finite_source, primary):
                    estimates = jnp.where(
                        jnp.isfinite(candidate), candidate, estimates
                    )
                converged = jnp.ravel(raw.finite_source_converged)
                converged = converged & jnp.isfinite(primary)
                support = jnp.ravel(raw.support_certified)
                status = jnp.where(
                    converged,
                    LightCurveInfo.OK,
                    jnp.where(
                        support,
                        LightCurveInfo.UNCONVERGED,
                        LightCurveInfo.SUPPORT_UNPROVEN,
                    ),
                ).astype(jnp.int8)
                return LightCurveInfo(
                    times=time_array,
                    magnifications=estimates,
                    status=status,
                    value_converged=converged,
                    support_certified=support,
                    diagnostics=raw,
                )
            values = magnification(
                self._native,
                self._options,
                time_array,
                merged,
                finite_source_backend=self._options.finite_source_backend,
            )
            finite = jnp.isfinite(values)
            estimates = jnp.where(finite, values, 1.0)
            status = jnp.where(
                finite,
                LightCurveInfo.OK,
                LightCurveInfo.UNCONVERGED,
            ).astype(jnp.int8)
            return LightCurveInfo(
                times=time_array,
                magnifications=estimates,
                status=status,
                value_converged=finite,
            )

        time_array = np.asarray(times, dtype=float)
        if time_array.ndim == 0:
            time_array = time_array.reshape(1)
        elif time_array.ndim != 1:
            raise ValueError("times must be a scalar or one-dimensional array")

        if (
            self._options.jax
            and self._options.finite_source_backend != "algebraic_boundary"
            and self._matching_warmup(time_array, merged)
            and self._warmup_jax_function is not None
        ):
            self._warn_if_warmup_drift(time_array, merged)
            import jax.numpy as jnp

            values = jnp.ravel(self._warmup_jax_function(time_array, merged))
            finite = jnp.isfinite(values)
            return LightCurveInfo(
                times=time_array,
                magnifications=jnp.where(finite, values, 1.0),
                status=jnp.where(
                    finite, LightCurveInfo.OK,
                    LightCurveInfo.UNCONVERGED,
                ).astype(jnp.int8),
                value_converged=finite,
            )

        if (
            self._options.jax
            and self._options.finite_source_backend != "algebraic_boundary"
        ):
            _register_lightcurve_info_pytree()
            import jax.numpy as jnp
            from .jax_backend import info as jax_info, magnification

            if self._native.lens == "binary" and self._native.model.source == "single":
                raw = jax_info(
                    self._native, self._options, time_array, merged
                )
                primary = jnp.ravel(raw.magnifications)
                finite_source = jnp.ravel(raw.finite_source_magnifications)
                point_source = jnp.ravel(raw.point_source_magnifications)
                values = jnp.full_like(primary, 1.0)
                for candidate in (point_source, finite_source, primary):
                    values = jnp.where(
                        jnp.isfinite(candidate), candidate, values
                    )
                converged = jnp.ravel(raw.finite_source_converged)
                converged = converged & jnp.isfinite(primary)
                support = jnp.ravel(raw.support_certified)
                status = jnp.where(
                    converged,
                    LightCurveInfo.OK,
                    jnp.where(
                        support,
                        LightCurveInfo.UNCONVERGED,
                        LightCurveInfo.SUPPORT_UNPROVEN,
                    ),
                ).astype(jnp.int8)
                return LightCurveInfo(
                    times=jnp.asarray(time_array),
                    magnifications=values,
                    status=status,
                    value_converged=converged,
                    support_certified=support,
                    diagnostics=raw,
                )

            values = jnp.ravel(magnification(
                self._native,
                self._options,
                time_array,
                merged,
                finite_source_backend="default",
            ))
            finite = jnp.isfinite(values)
            return LightCurveInfo(
                times=jnp.asarray(time_array),
                magnifications=jnp.where(finite, values, 1.0),
                status=jnp.where(
                    finite, LightCurveInfo.OK,
                    LightCurveInfo.UNCONVERGED,
                ).astype(jnp.int8),
                value_converged=finite,
            )

        if self._options.finite_source_backend == "algebraic_boundary":
            from .algebraic_boundary import lightcurve_value

            raw = lightcurve_value(
                self, time_array, merged, with_gradient=with_gradient
            )
            values = np.asarray(raw["mu"], dtype=float).reshape(-1)
            converged = np.asarray(raw["fully_converged"], dtype=bool).reshape(-1)
            converged &= np.isfinite(values)
            support = np.asarray(raw["support_certified"], dtype=bool).reshape(-1)
            integrated = np.asarray(raw["integrator_used"], dtype=bool).reshape(-1)
            support_failed = integrated & ~support
            point_source = np.asarray(
                raw["point_source_magnification"], dtype=float
            ).reshape(-1)
            fallback_used = (
                ~np.isfinite(values) | (values <= 0.0)
            ) & np.isfinite(point_source) & (point_source > 0.0)
            values = np.where(fallback_used, point_source, values)
            raw["value_fallback_used"] = fallback_used.copy()
            raw["value_fallback_kind"] = np.where(
                fallback_used, "point_source", "none"
            )
            status = np.where(
                converged,
                LightCurveInfo.OK,
                np.where(
                    support_failed,
                    LightCurveInfo.SUPPORT_UNPROVEN,
                    LightCurveInfo.UNCONVERGED,
                ),
            ).astype(np.int8)
            values = self._finite_estimates(values)
            return LightCurveInfo(
                times=time_array,
                magnifications=values,
                status=status,
                value_converged=converged,
                support_certified=support,
                diagnostics=raw,
            )

        if self._native.model.source == "single":
            if self._matching_warmup(time_array, merged):
                self._warn_if_warmup_drift(time_array, merged)
                try:
                    values = np.asarray(
                        self._native._magnification_preplanned(
                            time_array,
                            merged,
                            self._warmup_methods.tolist(),
                            self._warmup_profile.resolutions.tolist(),
                        ),
                        dtype=float,
                    ).reshape(-1)
                except RuntimeError:
                    # A stale or unusable plan must still return a per-epoch
                    # status and the best available value from the ordinary
                    # diagnostic route.
                    pass
                else:
                    finite = np.isfinite(values)
                    return LightCurveInfo(
                        times=time_array,
                        magnifications=self._finite_estimates(values),
                        status=np.where(
                            finite, LightCurveInfo.OK,
                            LightCurveInfo.UNCONVERGED,
                        ).astype(np.int8),
                        value_converged=finite,
                    )
            raw = self._native.info(time_array, merged)
            primary = np.asarray(raw.magnifications, dtype=float).reshape(-1)
            finite_source = np.asarray(
                raw.finite_source_magnifications, dtype=float
            ).reshape(-1)
            point_source = np.asarray(
                raw.point_source_magnifications, dtype=float
            ).reshape(-1)
            values = self._finite_estimates(primary, finite_source, point_source)
            raw_statuses = np.asarray(raw.statuses, dtype=object).reshape(-1)
            status = np.asarray([
                LightCurveInfo.OK if item == "ok" else
                LightCurveInfo.SUPPORT_UNPROVEN
                if item in ("unsupported", "unsupported_tolerance") else
                LightCurveInfo.UNCONVERGED
                for item in raw_statuses
            ], dtype=np.int8)
            converged = np.asarray(
                raw.finite_source_converged, dtype=bool
            ).reshape(-1)
            status[(status == LightCurveInfo.OK) & ~converged] = (
                LightCurveInfo.UNCONVERGED
            )
            return LightCurveInfo(
                times=time_array,
                magnifications=values,
                status=status,
                value_converged=converged,
                diagnostics=raw,
            )

        try:
            values = np.asarray(
                self._native(time_array, merged), dtype=float
            ).reshape(-1)
        except RuntimeError as error:
            message = str(error)
            if message.startswith("unsupported"):
                code = LightCurveInfo.SUPPORT_UNPROVEN
            elif message.startswith("numerical error"):
                code = LightCurveInfo.UNCONVERGED
            else:
                raise
            values = np.ones(time_array.shape, dtype=float)
            status = np.full(time_array.shape, code, dtype=np.int8)
            return LightCurveInfo(
                times=time_array,
                magnifications=values,
                status=status,
                value_converged=np.zeros(time_array.shape, dtype=bool),
                support_certified=(
                    np.zeros(time_array.shape, dtype=bool)
                    if code == LightCurveInfo.SUPPORT_UNPROVEN else None
                ),
            )
        finite = np.isfinite(values)
        status = np.where(
            finite, LightCurveInfo.OK, LightCurveInfo.UNCONVERGED
        ).astype(np.int8)
        return LightCurveInfo(
            times=time_array,
            magnifications=self._finite_estimates(values),
            status=status,
            value_converged=finite,
        )

    @staticmethod
    def _finite_estimates(*candidates):
        import numpy as np

        if not candidates:
            return np.empty((0,), dtype=float)
        values = np.full(np.shape(candidates[0]), 1.0, dtype=float)
        for candidate in reversed(candidates):
            candidate = np.asarray(candidate, dtype=float)
            values = np.where(np.isfinite(candidate), candidate, values)
        return values

    def _evaluate_algebraic_boundary_gradients(
        self, times, merged, eligible, diagnostics=None
    ):
        """Return the native seven-parameter Jacobian for accepted epochs."""
        import numpy as np

        if (
            self._options.jax
            or self._options.finite_source_backend != "algebraic_boundary"
        ):
            return None
        model = self._native.model
        if (
            model.lens != "binary"
            or model.source != "single"
            or model.parallax
            or model.xallarap != "none"
            or self._native.orbital_motion != "static"
        ):
            return None

        core_order = ("t0", "u0", "tE", "alpha", "rho", "q", "s")
        if tuple(merged) == core_order and isinstance(diagnostics, dict):
            try:
                mapped = np.asarray(diagnostics["gradients"], dtype=np.float64)
                row_valid = np.asarray(
                    diagnostics["gradient_valid"], dtype=bool
                ).reshape(-1)
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if mapped.size == row_valid.size * len(core_order):
                    # The native combined call already clears every derivative
                    # row whose value did not converge or whose Jacobian failed.
                    return core_order, mapped.reshape(-1, len(core_order)), row_valid

        aliases = {
            "t_0": "t0", "t_E": "tE", "umin": "u0",
            "theta": "alpha", "sep": "s", "rho1": "rho",
        }
        canonical = {}
        for name, value in merged.items():
            canonical[aliases.get(name, name)] = value
        names = []
        for name, value in canonical.items():
            if name == "orbital_motion_mode":
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            names.append(name)

        core = set(core_order)
        unsupported = {"limb_darkening_c", "limb_darkening_d"} & set(names)
        active = set(names) - unsupported
        if not core.issubset(active) or active - core:
            return None

        try:
            time_array = np.asarray(times, dtype=np.float64)
            status_array = np.asarray(eligible, dtype=np.int8).reshape(-1)
        except (TypeError, ValueError):
            return None
        if time_array.ndim == 0:
            time_array = time_array.reshape(1)
        elif time_array.ndim != 1:
            return None
        if status_array.size != time_array.size:
            return None
        gradients = np.zeros((time_array.size, len(names)), dtype=np.float64)
        gradient_valid = np.zeros(time_array.size, dtype=bool)
        accepted = status_array == LightCurveInfo.OK
        if not np.any(accepted):
            return names, gradients, gradient_valid

        if not isinstance(diagnostics, dict):
            return None
        try:
            mapped = np.asarray(diagnostics["gradients"], dtype=np.float64)
            row_valid = np.asarray(diagnostics["gradient_valid"], dtype=bool)
        except (KeyError, TypeError, ValueError):
            return None
        if (
            mapped.size != time_array.size * 7
            or row_valid.size != time_array.size
        ):
            return None
        mapped = mapped.reshape(-1, 7)
        row_valid = row_valid.reshape(-1) & accepted
        name_to_core_column = dict(zip(core_order, range(len(core_order))))
        for column, name in enumerate(names):
            if name in name_to_core_column:
                gradients[:, column] = mapped[:, name_to_core_column[name]]
        if unsupported:
            row_valid[:] = False
        gradient_valid[:] = row_valid
        return names, gradients, gradient_valid

    def _evaluate_gradients(
        self, times, merged, eligible=None, diagnostics=None
    ):
        if eligible is not None:
            native_result = self._evaluate_algebraic_boundary_gradients(
                times, merged, eligible, diagnostics
            )
            if native_result is not None:
                return native_result
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as error:
            raise RuntimeError(
                "with_gradient=True requires JAX; install lcbinint with the 'jax' extra"
            ) from error
        import numpy as np
        from .jax_backend import _PARAMETER_ALIASES, magnification

        _register_lightcurve_info_pytree()
        base = dict(merged)
        canonical = {}
        for name, value in merged.items():
            canonical[_PARAMETER_ALIASES.get(name, name)] = value
        base.update(canonical)
        names = []
        values_by_name = {}
        for name, value in canonical.items():
            if name == "orbital_motion_mode":
                continue
            try:
                scalar = jnp.asarray(value, dtype=jnp.float64)
            except (TypeError, ValueError):
                continue
            if scalar.ndim != 0:
                continue
            names.append(name)
            values_by_name[name] = scalar

        unsupported_names = set()
        if self._options.finite_source_backend == "algebraic_boundary":
            unsupported_names.update(
                name for name in names
                if name in ("limb_darkening_c", "limb_darkening_d")
            )
        active_names = [name for name in names if name not in unsupported_names]

        time_array = jnp.asarray(times)
        if time_array.ndim == 0:
            time_array = jnp.reshape(time_array, (1,))
        elif time_array.ndim != 1:
            raise ValueError("times must be a scalar or one-dimensional array")
        epoch_count = int(time_array.shape[0])
        eligible_indices = None
        if eligible is not None:
            leaves = jax.tree_util.tree_leaves(eligible)
            if not any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
                eligible_mask = np.asarray(eligible, dtype=np.int8).reshape(-1)
                if eligible_mask.size != epoch_count:
                    raise ValueError("value status and times must have equal length")
                eligible_indices = np.flatnonzero(
                    eligible_mask == LightCurveInfo.OK
                )
        active_times = (
            time_array if eligible_indices is None
            else time_array[eligible_indices]
        )
        if not active_names:
            gradients = jnp.zeros((epoch_count, len(names)), dtype=jnp.float64)
            valid = (
                jnp.zeros((epoch_count,), dtype=jnp.bool_)
                if names else jnp.ones((time_array.shape[0],), dtype=jnp.bool_)
            )
            if not self._options.jax:
                return names, np.asarray(gradients), np.asarray(valid)
            return tuple(names), gradients, valid

        if active_times.shape[0] == 0:
            gradients = jnp.zeros((epoch_count, len(names)), dtype=jnp.float64)
            valid = jnp.zeros((epoch_count,), dtype=jnp.bool_)
            if not self._options.jax:
                return names, np.asarray(gradients), np.asarray(valid)
            return tuple(names), gradients, valid

        initial = jnp.stack([values_by_name[name] for name in active_names])
        is_traced = any(
            isinstance(leaf, jax.core.Tracer)
            for leaf in jax.tree_util.tree_leaves(initial)
        )

        def evaluate(parameter_vector):
            parameters = dict(base)
            for index, name in enumerate(active_names):
                parameters[name] = parameter_vector[index]
            return jnp.ravel(magnification(
                self._native,
                self._options,
                active_times,
                parameters,
                finite_source_backend=self._options.finite_source_backend,
            ))

        try:
            active_jacobian = jax.jacfwd(evaluate)(initial)
        except (NotImplementedError, RuntimeError):
            gradients = jnp.zeros(
                (epoch_count, len(names)), dtype=jnp.float64
            )
            valid = jnp.zeros((time_array.shape[0],), dtype=jnp.bool_)
            if not self._options.jax and not is_traced:
                return names, np.asarray(gradients), np.asarray(valid)
            return tuple(names), gradients, valid

        finite = jnp.isfinite(active_jacobian)
        active_valid = jnp.all(finite, axis=1)
        active_jacobian = jnp.where(finite, active_jacobian, 0.0)
        gradients = jnp.zeros(
            (epoch_count, len(names)), dtype=active_jacobian.dtype
        )
        if eligible_indices is None:
            active_indices = jnp.arange(epoch_count)
        else:
            active_indices = jnp.asarray(eligible_indices, dtype=jnp.int32)
        valid = jnp.zeros((epoch_count,), dtype=jnp.bool_)
        valid = valid.at[active_indices].set(active_valid)
        for index, name in enumerate(names):
            if name in active_names:
                gradients = gradients.at[active_indices, index].set(
                    active_jacobian[:, active_names.index(name)]
                )
        if unsupported_names:
            valid = valid.at[active_indices].set(False)
        if not self._options.jax and not is_traced:
            return names, np.asarray(gradients), np.asarray(valid)
        return tuple(names), gradients, valid

    def __call__(self, times, params=None, *, with_gradient=False, **kwargs):
        """Return magnifications and status, optionally with parameter gradients.

        The returned :class:`LightCurveInfo` is the same record in either
        mode. Set ``with_gradient=True`` to include ``gradients`` and
        ``gradient_valid``. Both modes return the same value and status. The
        derivative implementation follows the selected backend, and only runs
        for epochs whose value status is accepted.
        """
        merged = self._merge_params(params, **kwargs)
        native_algebraic_gradient = (
            with_gradient
            and not self._options.jax
            and self._options.finite_source_backend == "algebraic_boundary"
        )
        result = self._evaluate_values(
            times, merged, with_gradient=native_algebraic_gradient
        )
        if not with_gradient:
            return result
        core_gradient_parameters = ("t0", "u0", "tE", "alpha", "rho", "q", "s")
        if (
            native_algebraic_gradient
            and tuple(merged) == core_gradient_parameters
            and isinstance(result.diagnostics, dict)
            and "gradients" in result.diagnostics
            and "gradient_valid" in result.diagnostics
        ):
            # The combined native call already returns this common parameter
            # order and clears rows whose value or Jacobian was not accepted.
            gradients = result.diagnostics["gradients"].reshape(
                -1, len(core_gradient_parameters)
            )
            valid = result.diagnostics["gradient_valid"].reshape(-1)
            if gradients.shape[0] == result.status.shape[0] == valid.size:
                return LightCurveInfo(
                    times=result.times,
                    magnifications=result.magnifications,
                    status=result.status,
                    gradients=gradients,
                    gradient_valid=valid,
                    gradient_parameters=core_gradient_parameters,
                    value_converged=result.value_converged,
                    support_certified=result.support_certified,
                    diagnostics=result.diagnostics,
                )
        names, gradients, valid = self._evaluate_gradients(
            times, merged, eligible=result.status,
            diagnostics=result.diagnostics,
        )
        return LightCurveInfo(
            times=result.times,
            magnifications=result.magnifications,
            status=result.status,
            gradients=gradients,
            gradient_valid=valid,
            gradient_parameters=tuple(names),
            value_converged=result.value_converged,
            support_certified=result.support_certified,
            diagnostics=result.diagnostics,
        )

    def magnification(self, times, params=None, **kwargs):
        return self.__call__(times, params, **kwargs)

    def magnification_batch(self, times, parameter_rows):
        """Evaluate independent parameter rows with the selected backend.

        Native execution uses one GIL-free C++ call. JAX execution preserves
        the same row-major result shape and composes the public differentiable
        callable for each statically supplied parameter mapping.
        """
        if self._options.finite_source_backend == "algebraic_boundary":
            raise NotImplementedError(
                "magnification_batch is not implemented for finite_source_backend='algebraic_boundary'"
            )
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
        return self._options.finite_source_backend == "default"

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
        if self._options.finite_source_backend == "algebraic_boundary":
            raise NotImplementedError(
                "batched likelihood is not implemented for finite_source_backend='algebraic_boundary'"
            )
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

    def info(self, times, params=None, *, with_gradient=False, **kwargs):
        """Compatibility alias for :meth:`LightCurve.__call__`."""
        return self.__call__(
            times, params, with_gradient=with_gradient, **kwargs
        )

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
        if name in {
            "holonomic_gradient",
            "algebraic_boundary_magnification",
            "algebraic_boundary_info",
        }:
            raise AttributeError(name)
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
        if options.finite_source_backend not in (None, "default"):
            raise NotImplementedError(
                "finite_source_backend is only supported by LightCurve"
            )
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
