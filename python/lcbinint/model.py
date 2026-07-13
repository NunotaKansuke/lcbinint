"""Python-extended bayes.Model with decorators and sampling reparameterization."""
from __future__ import annotations

import inspect
import math

import numpy as np


class LikelihoodContext:
    """Auxiliary values from the base likelihood evaluation."""

    def __init__(self, *, fluxes):
        self.fluxes = fluxes


def _theta_star_result(fn, fluxes):
    result = fn(fluxes)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError(
            "theta_star() callable must return (log_center, log_sigma)"
        )
    center, sigma = (float(result[0]), float(result[1]))
    if not math.isfinite(center):
        raise ValueError("theta_star() log_center must be finite")
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("theta_star() log_sigma must be finite and >= 0")
    return center, sigma


def _logmeanexp(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("-inf")
    peak = float(np.max(values[finite]))
    return peak + math.log(float(np.mean(np.exp(values - peak))))


def _sobol_points(samples, dimensions, seed):
    import warnings

    from scipy.stats import qmc

    engine = qmc.Sobol(d=dimensions, scramble=True, seed=seed)
    if samples > 0 and samples & (samples - 1) == 0:
        points = engine.random_base2(int(math.log2(samples)))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            points = engine.random(samples)
    eps = np.finfo(float).eps
    return np.clip(points, eps, 1.0 - eps)


def _draw_fluxes(fluxes, conditionals, uniforms):
    from scipy.special import stdtrit

    draw = {name: dict(values) for name, values in fluxes.items()}
    for uniform, (name, params) in zip(uniforms, conditionals.items()):
        draw[name]["Fs"] = (
            float(params["mean"])
            + float(params["scale"]) * stdtrit(float(params["df"]), uniform)
        )
    return draw


def _theta_star_log_density(theta_star, center, sigma):
    if sigma == 0.0:
        raise RuntimeError(
            "theta_star() returned log_sigma=0, so thetaS is deterministic; "
            "do not register model.param('thetaS', ...)"
        )
    theta_star = float(theta_star)
    if not math.isfinite(theta_star) or theta_star <= 0.0:
        return float("-inf")
    z = (math.log(theta_star) - center) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def _apply_sampled_theta_star(fn, vals, fluxes, conditionals, options):
    if not conditionals:
        center, sigma = _theta_star_result(fn, fluxes)
        return _theta_star_log_density(vals["thetaS"], center, sigma)

    points = _sobol_points(
        options["samples"], len(conditionals), options["seed"]
    )
    log_weights = []
    for point in points:
        draw = _draw_fluxes(fluxes, conditionals, point)
        try:
            center, sigma = _theta_star_result(fn, draw)
            log_weights.append(
                _theta_star_log_density(vals["thetaS"], center, sigma)
            )
        except (ValueError, FloatingPointError, OverflowError):
            log_weights.append(float("-inf"))
    return _logmeanexp(log_weights)


def _evaluate_extra_priors_many(priors, values):
    current = [dict(vals) for vals in values]
    totals = np.zeros(len(current), dtype=float)
    for prior_index, fn in enumerate(priors):
        batch = fn.batch_log_prob(current) if hasattr(fn, "batch_log_prob") else None
        if batch is None:
            batch = np.asarray(
                [
                    _call_extra_prior(fn, vals) if math.isfinite(totals[i])
                    else float("-inf")
                    for i, vals in enumerate(current)
                ],
                dtype=float,
            )
        totals += batch
        if prior_index + 1 < len(priors) and hasattr(fn, "physical_values"):
            for i, vals in enumerate(current):
                if math.isfinite(totals[i]):
                    vals.update(fn.physical_values(vals))
    return totals


def _marginalize_theta_star(fn, vals, fluxes, conditionals, priors, options):
    from scipy.special import ndtri

    samples = []
    invalid_count = 0

    if conditionals:
        points = _sobol_points(
            options["samples"], len(conditionals) + 1, options["seed"]
        )
        flux_draws = (
            (_draw_fluxes(fluxes, conditionals, point[:-1]), point[-1])
            for point in points
        )
    else:
        flux_draws = ((fluxes, None),)

    for draw, theta_uniform in flux_draws:
        try:
            center, sigma = _theta_star_result(fn, draw)
        except (ValueError, FloatingPointError, OverflowError):
            invalid_count += 1 if conditionals else options["samples"]
            continue

        if sigma == 0.0:
            theta_stars = [math.exp(center)]
        elif conditionals:
            theta_stars = [math.exp(center + sigma * ndtri(theta_uniform))]
        else:
            theta_uniforms = _sobol_points(
                options["samples"], 1, options["seed"]
            )[:, 0]
            theta_stars = np.exp(
                center + sigma * ndtri(theta_uniforms)
            )

        for theta_star in theta_stars:
            current = dict(vals)
            current["thetaS"] = float(theta_star)
            samples.append(current)

    if not samples:
        return float("-inf")
    log_weights = _evaluate_extra_priors_many(priors, samples)
    if invalid_count:
        log_weights = np.concatenate(
            [log_weights, np.full(invalid_count, float("-inf"))]
        )
    return _logmeanexp(log_weights)


class _GalacticModelTerm:
    """Adapter for external Galactic model objects.

    The external object is expected to expose ``log_prob(theta, context=...)``.
    ``theta`` is assembled from lcbinint physical parameter values in ``names``
    order. Keeping this as a normal extra prior avoids making lcbinint depend on
    gapmoe directly.
    """

    def __init__(self, galaxy, *, names=None, context=None):
        if not hasattr(galaxy, "log_prob") or not callable(galaxy.log_prob):
            raise TypeError("galactic_prior() expects an object with log_prob()")
        if names is None:
            param_type = getattr(galaxy, "param_type", None)
            names = getattr(param_type, "names", None)
        if names is None:
            raise ValueError(
                "galactic_prior() requires names=... when "
                "galaxy.param_type.names is unavailable"
            )
        if isinstance(names, str):
            raise TypeError("galactic_prior() names must be a sequence, not a string")
        self.galaxy = galaxy
        self.names = tuple(names)
        if not self.names:
            raise ValueError("galactic_prior() requires at least one parameter name")
        self.context = context
        self._batch_fn = None
        self._batch_disabled = False

    def __call__(self, **vals):
        missing = [name for name in self.names if name not in vals]
        if missing:
            raise RuntimeError(
                "galactic_prior() missing model parameter(s): "
                + ", ".join(missing)
            )
        theta = [vals[name] for name in self.names]
        context = self._context(vals)
        if context is None:
            return float(self.galaxy.log_prob(theta))
        return float(self.galaxy.log_prob(theta, context=context))

    def physical_values(self, vals):
        theta = [vals[name] for name in self.names]
        context = self._context(vals)
        if hasattr(self.galaxy, "to_deterministic_physical"):
            if context is None:
                return dict(self.galaxy.to_deterministic_physical(theta))
            return dict(self.galaxy.to_deterministic_physical(theta, context=context))
        if not hasattr(self.galaxy, "to_physical"):
            return {}
        if context is None:
            physical = self.galaxy.to_physical(theta)
        else:
            physical = self.galaxy.to_physical(theta, context=context)
        keys = ["ML", "DL", "DS", "mu_N", "mu_E"]
        param_type = getattr(self.galaxy, "param_type", None)
        keys.extend(getattr(param_type, "derived_names", ()))
        return {
            key: value
            for key, value in zip(keys, physical)
        }

    def batch_log_prob(self, values):
        """Evaluate a JAX-compatible Galactic model for many latent states."""
        if self._batch_disabled:
            return None
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            self._batch_disabled = True
            return None

        if self._batch_fn is None:
            names = self.names
            context_fn = self.context
            galaxy = self.galaxy

            def one(vals):
                theta = jnp.stack([jnp.asarray(vals[name]) for name in names])
                context = context_fn(vals) if callable(context_fn) else context_fn
                if context is None:
                    return galaxy.log_prob(theta)
                return galaxy.log_prob(theta, context=context)

            self._batch_fn = jax.jit(jax.vmap(one))

        keys = tuple(values[0])
        try:
            batched = {
                key: jnp.asarray([vals[key] for vals in values])
                for key in keys
            }
            result = self._batch_fn(batched)
            return np.asarray(result, dtype=float)
        except (
            TypeError,
            ValueError,
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerArrayConversionError,
        ):
            self._batch_disabled = True
            self._batch_fn = None
            return None

    def _context(self, vals):
        if callable(self.context):
            return self.context(dict(vals))
        return self.context


def _call_extra_prior(fn, vals):
    missing = _missing_required_kwargs(fn, vals)
    if missing:
        raise RuntimeError(
            "prior() requested unavailable parameter(s): "
            + ", ".join(missing)
            + ". Only parameters that are deterministic for the current "
            "sampler step are passed to @model.prior. If these are "
            "marginalized Galactic quantities such as DL or DS, use "
            "model.get_galactic_physical(chain, galaxy, ...) after sampling, "
            "or sample those quantities explicitly."
        )
    return fn(**vals)


def _call_guard(fn, vals):
    missing = _missing_required_kwargs(fn, vals)
    if missing:
        raise RuntimeError(
            "guard() requested unavailable parameter(s): "
            + ", ".join(missing)
            + ". Guards run after reparameterization and may only use "
            "parameters that are deterministic for the current sampler step."
        )
    result = fn(**vals)
    if not isinstance(result, (bool, np.bool_)):
        raise TypeError(
            "guard() must return bool, got " + type(result).__name__
        )
    return bool(result)


def _missing_required_kwargs(fn, vals):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    missing = []
    for name, param in sig.parameters.items():
        if param.default is not inspect._empty:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name not in vals:
            missing.append(name)
    return missing


class _ReparamBlock:
    """One local replacement of physical parameters by sampling parameters."""

    def __init__(self, model, targets):
        self._model = model
        self.targets = list(targets)
        if not self.targets:
            raise ValueError("model.reparam() requires at least one target parameter")
        self._sample_params = []
        self._transform_fn = None
        self._log_jacobian_fn = None

    def param(self, name: str, prior):
        """Register a replacement sampling parameter and its prior."""
        self._sample_params.append((name, prior))
        return self

    def transform(self, fn):
        """Set sampling-parameter -> physical-parameter transform."""
        self._transform_fn = fn
        return fn

    def log_jacobian(self, fn):
        """Add an explicit log-Jacobian term for this reparameterization."""
        self._log_jacobian_fn = fn
        return fn

    @property
    def param_names(self):
        return [name for name, _ in self._sample_params]

    def _validate(self):
        if not self._sample_params:
            raise RuntimeError("model.reparam(): register at least one param()")
        if self._transform_fn is None:
            raise RuntimeError("model.reparam(): call transform() before sampling")


class _SamplingAdapter:
    """Sampler-space view of a Model.

    This object is intentionally not a C++ bayes.Model, so pybind dispatch uses
    the Python callback sampler path. It combines unchanged model parameters
    with local reparameterization blocks and evaluates the original posterior in
    physical parameter space.
    """

    def __init__(self, model):
        self._model = model
        self._phys_model = None
        self._phys_names = None
        self._base_names = [
            name for name in model.param_names
            if name not in model._reparam_targets()
        ]
        self._param_names = list(self._base_names)
        for block in model._reparam_blocks:
            block._validate()
            self._param_names.extend(block.param_names)
        self._bounds = self._build_bounds()
        self._transforms = self._build_transforms()
        self._aux_cache = {}

    @property
    def param_names(self):
        return list(self._param_names)

    @property
    def optimizer_bounds(self):
        return list(self._bounds)

    @property
    def _sample_transforms(self):
        return list(self._transforms)

    def n_params(self):
        return len(self._param_names)

    @property
    def n_data(self):
        return self._ensure_physical_model().n_data

    def log_prob(self, theta):
        phys = self._theta_to_physical(theta)
        lp = self._log_prior(theta, phys)
        if not math.isfinite(lp):
            return lp

        vals = dict(phys)
        for fn in self._model._guards:
            if not _call_guard(fn, vals):
                return float("-inf")

        phys_model = self._ensure_physical_model()
        phys_theta = [phys[name] for name in self._phys_names]
        base_prob, fluxes, conditionals = phys_model._log_prob_and_fluxes(phys_theta)
        if not math.isfinite(base_prob):
            return base_prob
        self._aux_cache[tuple(theta)] = (fluxes, conditionals)
        lp += base_prob - phys_model.log_prior(phys_theta)

        if self._model._theta_star_fn is not None:
            if "thetaS" in vals:
                lp += _apply_sampled_theta_star(
                    self._model._theta_star_fn,
                    vals,
                    fluxes,
                    conditionals,
                    self._model._theta_star_options,
                )
            else:
                lp += _marginalize_theta_star(
                    self._model._theta_star_fn,
                    vals,
                    fluxes,
                    conditionals,
                    self._model._extra_priors,
                    self._model._theta_star_options,
                )
            if not math.isfinite(lp):
                return lp

        if self._model._theta_star_fn is None or "thetaS" in vals:
            for fn in self._model._extra_priors:
                lp += _call_extra_prior(fn, vals)
                if not math.isfinite(lp):
                    return lp
                if hasattr(fn, "physical_values"):
                    vals.update(fn.physical_values(vals))

        context = LikelihoodContext(fluxes=fluxes)
        for fn, wants_context in self._model._extra_liks:
            if wants_context:
                lp += fn(**vals, context=context)
            else:
                lp += fn(**vals)
            if not math.isfinite(lp):
                return lp
        return lp

    def consume_current_aux(self, positions):
        """Return cached likelihood auxiliaries for accepted walker positions."""
        current = {}
        flux_rows = []
        scale_rows = []
        dfs = None
        dataset_names = None
        for position in np.asarray(positions):
            key = tuple(position.tolist())
            try:
                fluxes, conditionals = self._aux_cache[key]
            except KeyError as exc:
                raise RuntimeError(
                    "accepted sampler state has no cached likelihood auxiliaries"
                ) from exc
            current[key] = (fluxes, conditionals)
            if dataset_names is None:
                dataset_names = list(fluxes)
                dfs = np.asarray([
                    float(conditionals[name]["df"])
                    if name in conditionals else np.nan
                    for name in dataset_names
                ])
            flux_rows.append([
                value
                for name in dataset_names
                for value in (fluxes[name]["Fs"], fluxes[name]["Fb"])
            ])
            scale_rows.append([
                float(conditionals[name]["scale"])
                if name in conditionals else np.nan
                for name in dataset_names
            ])
        self._aux_cache = current
        return (
            np.asarray(flux_rows, dtype=float),
            np.asarray(scale_rows, dtype=float),
            dfs,
            dataset_names or [],
        )

    def fluxes(self, theta):
        phys = self._theta_to_physical(theta)
        phys_model = self._ensure_physical_model()
        phys_theta = [phys[name] for name in self._phys_names]
        return phys_model.fluxes(phys_theta)

    def _build_bounds(self):
        bounds = []
        for name in self._base_names:
            bounds.append(self._prior_bounds(self._model._param_priors[name]))
        for block in self._model._reparam_blocks:
            for _, prior in block._sample_params:
                bounds.append(self._prior_bounds(prior))
        return bounds

    def _build_transforms(self):
        transforms = []
        for name in self._base_names:
            transforms.append(
                "log" if self._model._is_log_prior(self._model._param_priors[name])
                else "identity")
        for block in self._model._reparam_blocks:
            for _, prior in block._sample_params:
                transforms.append("log" if self._model._is_log_prior(prior) else "identity")
        return transforms

    def _prior_bounds(self, prior):
        lo, hi = prior.bounds()
        if self._model._is_log_prior(prior):
            return (math.log(lo), math.log(hi))
        return (lo, hi)

    def _prior_log_prob(self, prior, theta_value):
        if self._model._is_log_prior(prior):
            return prior.log_prob(math.exp(theta_value)) + theta_value
        return prior.log_prob(theta_value)

    def _theta_to_values(self, theta):
        vals = {}
        idx = 0
        for name in self._base_names:
            prior = self._model._param_priors[name]
            vals[name] = math.exp(theta[idx]) if self._model._is_log_prior(prior) else theta[idx]
            idx += 1
        block_vals = []
        for block in self._model._reparam_blocks:
            current = {}
            for name, prior in block._sample_params:
                current[name] = (
                    math.exp(theta[idx]) if self._model._is_log_prior(prior) else theta[idx]
                )
                idx += 1
            block_vals.append(current)
        return vals, block_vals

    def _theta_to_physical(self, theta):
        phys, block_vals = self._theta_to_values(theta)
        for block, vals in zip(self._model._reparam_blocks, block_vals):
            out = block._transform_fn(**vals)
            if not isinstance(out, dict):
                raise TypeError("model.reparam().transform must return a dict")
            missing = [name for name in block.targets if name not in out]
            if missing:
                raise RuntimeError(
                    "model.reparam().transform did not return target(s): "
                    + ", ".join(missing)
                )
            phys.update(out)
        return phys

    def _log_prior(self, theta, phys):
        lp = 0.0
        idx = 0
        for name in self._base_names:
            lp += self._prior_log_prob(self._model._param_priors[name], theta[idx])
            idx += 1
            if not math.isfinite(lp):
                return lp
        for block in self._model._reparam_blocks:
            vals = {}
            for name, prior in block._sample_params:
                lp += self._prior_log_prob(prior, theta[idx])
                vals[name] = (
                    math.exp(theta[idx]) if self._model._is_log_prior(prior) else theta[idx]
                )
                idx += 1
                if not math.isfinite(lp):
                    return lp
            if block._log_jacobian_fn is not None:
                lp += block._log_jacobian_fn(**vals)
                if not math.isfinite(lp):
                    return lp
        return lp

    def _ensure_physical_model(self):
        if self._phys_model is None:
            self._build_physical_model()
        return self._phys_model

    def _build_physical_model(self):
        from ._lcbinint import bayes as _b

        names = []
        for name in self._base_names:
            if name not in names:
                names.append(name)
        for block in self._model._reparam_blocks:
            for name in block.targets:
                if name not in names:
                    names.append(name)
        for block in self._model._reparam_blocks:
            test_vals = {}
            for name, prior in block._sample_params:
                lo, hi = prior.bounds()
                test_vals[name] = (
                    math.exp(0.5 * (math.log(lo) + math.log(hi)))
                    if self._model._is_log_prior(prior) else 0.5 * (lo + hi)
                )
            out = block._transform_fn(**test_vals)
            if not isinstance(out, dict):
                raise TypeError("model.reparam().transform must return a dict")
            for name in out:
                if name not in names:
                    names.append(name)

        phys_model = _b._Model(self._model._lc, self._model._event_or_data)
        for name in names:
            phys_model.param(name, _b.Uniform(-1e15, 1e15))
        if self._model._lik_mode is not None:
            phys_model.likelihood(self._model._lik_mode, **self._model._lik_kwargs)
        self._phys_names = names
        self._phys_model = phys_model


def _build_model_class(cpp_base):
    """Build the Python Model subclass from the C++ _Model base.

    Called once from __init__.py after _lcbinint is loaded.
    """

    class Model(cpp_base):
        """Extended bayes.Model supporting custom Python prior/likelihood terms.

        Inherits all C++ bayes._Model functionality. Adds:

        - ``model.likelihood("gaussian")``  — set base likelihood mode (C++ path)
        - ``@model.likelihood``             — add a Python log-likelihood term
        - ``@model.guard``                  — reject unsafe parameter combinations
        - ``@model.theta_star``             — constrain or derive thetaS from flux
        - ``@model.prior``                  — add a Python log-prior term
        - ``model.galactic_prior(...)``     — add an external Galactic model
        - ``model.param(name)``             — flat improper prior (no bounds required)
        - ``model.reparam([...])``          — replace sampled coordinates locally

        User functions receive physical parameter values as **kwargs; unused
        params are absorbed by ``**_``.

        Example::

            model = bayes.Model(lc, event)
            model.param("tE",   bayes.LogUniform(1, 1000))
            model.param("piEN", bayes.Uniform(-1, 1))
            model.param("piEE", bayes.Uniform(-1, 1))
            model.likelihood("gaussian")

            @model.likelihood
            def gaia(piEN, piEE, **_):
                piE = (piEN**2 + piEE**2)**0.5
                return -0.5 * ((piE - 0.3) / 0.05)**2

            chain = run_sampler(model, nsteps=1000, start=result)
        """

        def __init__(
            self,
            lc=None,
            event_or_data=None,
            *,
            light_curve=None,
            event=None,
            data=None,
        ):
            if light_curve is not None:
                lc = light_curve
            provided = [x is not None for x in (event_or_data, event, data)]
            if sum(provided) != 1:
                raise TypeError(
                    "bayes.Model expects exactly one of event_or_data, event, or data"
                )
            if event is not None:
                event_or_data = event
            if data is not None:
                event_or_data = data
            if lc is None:
                raise TypeError("bayes.Model requires a light_curve")
            super().__init__(lc, event_or_data)
            self._lc = lc
            self._event_or_data = event_or_data
            self._extra_liks   = []
            self._extra_priors = []
            self._guards       = []
            self._theta_star_fn = None
            self._theta_star_options = None
            self._param_is_log = {}
            self._param_priors = {}
            self._lik_mode     = None
            self._lik_kwargs   = {}
            self._reparam_blocks = []

        # --- param: supports no-prior (flat) ---

        def param(self, name: str, prior=None):
            """Register a sampling parameter.

            If *prior* is omitted, a flat (improper) prior is used:
            ``log_prob = 0`` everywhere, no hard bounds.
            """
            from ._lcbinint import bayes as _b
            if name == "flux_all" and prior is None:
                raise ValueError("model.param('flux_all') requires an explicit prior")
            if prior is None:
                prior = _b.Uniform()
            if name == "flux_all":
                for dataset_name in self.dataset_names:
                    self.param(f"Fs_{dataset_name}", prior)
                    self.param(f"Fb_{dataset_name}", prior)
                return self
            super().param(name, prior)
            self._param_is_log[name] = isinstance(prior, _b.LogUniform)
            self._param_priors[name] = prior
            return self

        # --- local sampling reparameterization ---

        def reparam(self, targets):
            """Replace physical parameter(s) with custom sampling parameters.

            The target parameters do not need to be registered with
            ``model.param()``. Priors are attached to the replacement sampling
            parameters, not to the generated physical parameters.
            """
            if isinstance(targets, str):
                targets = [targets]
            overlap = sorted(set(targets) & self._reparam_targets())
            if overlap:
                raise ValueError(
                    "parameter(s) already reparameterized: " + ", ".join(overlap)
                )
            block = _ReparamBlock(self, targets)
            self._reparam_blocks.append(block)
            return block

        # --- likelihood ---

        def likelihood(self, arg="gaussian", **kwargs):
            """Set or add a likelihood term.

            ``model.likelihood("gaussian")`` — use C++ Gaussian chi2.
            ``model.likelihood("gaussian", flux="sample")`` — sample Fs/Fb params.
            ``model.likelihood("student_t", nu=4)`` — approximate robust likelihood.
            ``@model.likelihood``            — add a Python log-likelihood function.
            ``@model.likelihood(context=True)`` — pass likelihood context.
            """
            if "context" in kwargs:
                wants_context = bool(kwargs.pop("context"))
                if kwargs:
                    raise TypeError(
                        "callable likelihood does not accept keyword options "
                        "other than context"
                    )
                if callable(arg):
                    self._extra_liks.append((arg, wants_context))
                    return arg
                if arg == "gaussian":
                    def decorator(fn):
                        if not callable(fn):
                            raise TypeError(
                                "likelihood(context=True) expects a callable"
                            )
                        self._extra_liks.append((fn, wants_context))
                        return fn
                    return decorator
                raise TypeError("context=True is only valid for callable likelihoods")
            if isinstance(arg, str):
                allowed = {"nu", "flux"}
                unknown = sorted(set(kwargs) - allowed)
                if unknown:
                    raise TypeError(
                        "unknown likelihood option(s): " + ", ".join(unknown)
                    )
                super().likelihood(arg, **kwargs)
                self._lik_mode = arg
                self._lik_kwargs = dict(kwargs)
            elif callable(arg):
                if kwargs:
                    raise TypeError("callable likelihood does not accept keyword options")
                self._extra_liks.append((arg, False))
                return arg  # enables @model.likelihood
            else:
                raise TypeError(
                    f"likelihood() expects str or callable, got {type(arg).__name__}")

        # --- prior ---

        def theta_star(self, fn=None, *, samples=256, seed=0):
            """Set the log-space thetaS relation evaluated from fitted fluxes.

            The callable receives the complete flux dictionary and returns
            ``(log_center, log_sigma)``. A positive sigma adds a Gaussian term
            for explicitly sampled ``thetaS``. Without a registered ``thetaS``
            parameter, thetaS and marginalized fluxes are integrated using
            ``samples`` scrambled Sobol draws (default 256).
            """
            if fn is None:
                return lambda decorated: self.theta_star(
                    decorated,
                    samples=samples,
                    seed=seed,
                )
            if not callable(fn):
                raise TypeError(
                    f"theta_star() expects a callable, got {type(fn).__name__}"
                )
            if self._theta_star_fn is not None:
                raise RuntimeError("theta_star() is already configured")
            if isinstance(samples, bool) or int(samples) != samples or samples <= 0:
                raise ValueError("theta_star() samples must be a positive integer")
            if isinstance(seed, bool) or int(seed) != seed or seed < 0:
                raise ValueError("theta_star() seed must be a non-negative integer")
            self._theta_star_fn = fn
            self._theta_star_options = {
                "samples": int(samples),
                "seed": int(seed),
            }
            return fn

        def guard(self, fn):
            """Reject unsafe physical parameters before model evaluation.

            The callable receives physical parameter values as keyword
            arguments and must return bool. ``False`` makes the log posterior
            ``-inf`` before Galactic priors or magnification are evaluated.
            """
            if not callable(fn):
                raise TypeError(
                    f"guard() expects a callable, got {type(fn).__name__}"
                )
            self._guards.append(fn)
            return fn

        def prior(self, fn):
            """Add a Python log-prior term (decorator or direct call).

            *fn* receives physical parameter values as **kwargs.
            Multiple calls are additive.
            """
            if not callable(fn):
                raise TypeError(
                    f"prior() expects a callable, got {type(fn).__name__}")
            self._extra_priors.append(fn)
            return fn  # enables @model.prior

        def galactic_prior(self, galaxy, *, context=None, names=None):
            """Add an external Galactic model as a prior term.

            ``galaxy`` is typically a ``gapmoe.GalacticModel``. lcbinint does
            not import gapmoe; it only requires a ``log_prob(theta, context=...)``
            method. The parameter vector is assembled from model physical
            values in ``names`` order. If ``names`` is omitted,
            ``galaxy.param_type.names`` is used.

            ``context`` may be a dict-like object or a callable receiving the
            current physical parameter dict and returning a context object.
            """
            term = _GalacticModelTerm(galaxy, names=names, context=context)
            self._extra_priors.append(term)
            return term

        def get_galactic_physical(
            self,
            chain,
            galaxy,
            *,
            context=None,
            names=None,
            flat=True,
            discard=0,
            thin=1,
            rng=None,
        ):
            """Return gapmoe physical samples for a chain.

            For marginalized gapmoe dimensions this calls
            ``galaxy.sample_physical()``, so hidden physical variables are drawn
            from their conditional posterior for each chain sample. If thetaS
            was marginalized, its stored flux conditional is used to restore a
            posterior draw without recomputing magnification. The returned dict
            then also contains ``thetaS`` and ``Fs_<dataset>``.
            """
            if names is None:
                param_type = getattr(galaxy, "param_type", None)
                names = getattr(param_type, "names", None)
            if names is None:
                raise ValueError(
                    "get_galactic_physical() requires names=... when "
                    "galaxy.param_type.names is unavailable"
                )
            names = tuple(names)
            raw = np.asarray(
                chain.get_samples(
                    flat=flat,
                    discard=discard,
                    thin=thin,
                    physical=False,
                )
            )
            out_shape = raw.shape[:-1]
            rows = raw.reshape(-1, raw.shape[-1])
            if self.has_reparams():
                adapter = self._sampling_adapter()
                vals_iter = [adapter._theta_to_physical(row.tolist()) for row in rows]
            else:
                vals_iter = [self._theta_to_vals(row.tolist()) for row in rows]

            rng = np.random.default_rng() if rng is None else rng
            flux_dict = chain.get_fluxes(
                flat=flat, discard=discard, thin=thin
            )
            flux_rows = []
            for index in range(len(rows)):
                flux_rows.append({
                    name: {
                        "Fs": float(np.asarray(values["Fs"]).reshape(-1)[index]),
                        "Fb": float(np.asarray(values["Fb"]).reshape(-1)[index]),
                    }
                    for name, values in flux_dict.items()
                })

            scales = getattr(chain, "_flux_conditional_scales", None)
            dfs = getattr(chain, "_flux_conditional_dfs", None)
            if (
                self._theta_star_fn is not None
                and vals_iter
                and "thetaS" not in vals_iter[0]
                and self._lik_kwargs.get("flux") == "marginalize"
                and scales is None
            ):
                raise RuntimeError(
                    "this chain does not contain conditional flux statistics; "
                    "rerun sampling before restoring marginalized thetaS"
                )
            if scales is not None:
                scales = np.asarray(scales)[discard::thin]
                if flat:
                    scales = scales.reshape(-1, scales.shape[-1])
                else:
                    scales = scales.reshape(-1, scales.shape[-1])

            physical_draws = []
            latent_draws = []
            for index, vals in enumerate(vals_iter):
                vals = dict(vals)
                fluxes = flux_rows[index] if flux_rows else {}
                if self._theta_star_fn is not None and "thetaS" not in vals:
                    conditionals = {}
                    if scales is not None:
                        for d, name in enumerate(chain.dataset_names):
                            if np.isfinite(scales[index, d]):
                                conditionals[name] = {
                                    "mean": fluxes[name]["Fs"],
                                    "scale": scales[index, d],
                                    "df": np.asarray(dfs)[d],
                                }
                    candidates = self._latent_theta_star_candidates(
                        vals, fluxes, conditionals
                    )
                    weights = _evaluate_extra_priors_many(
                        self._extra_priors, [item[0] for item in candidates]
                    )
                    finite = np.isfinite(weights)
                    if not np.any(finite):
                        raise RuntimeError(
                            "no finite conditional thetaS/flux draw for chain sample"
                        )
                    peak = np.max(weights[finite])
                    probs = np.where(finite, np.exp(weights - peak), 0.0)
                    probs /= probs.sum()
                    vals, fluxes = candidates[int(rng.choice(len(candidates), p=probs))]

                missing = [name for name in names if name not in vals]
                if missing:
                    raise RuntimeError(
                        "get_galactic_physical() missing model parameter(s): "
                        + ", ".join(missing)
                    )
                theta = [vals[name] for name in names]
                ctx = context(dict(vals)) if callable(context) else context
                if hasattr(galaxy, "sample_physical"):
                    if ctx is None:
                        draw = galaxy.sample_physical(theta, rng=rng)
                    else:
                        draw = galaxy.sample_physical(theta, context=ctx, rng=rng)
                elif hasattr(galaxy, "to_deterministic_physical"):
                    if ctx is None:
                        draw = galaxy.to_deterministic_physical(theta)
                    else:
                        draw = galaxy.to_deterministic_physical(theta, context=ctx)
                else:
                    raise TypeError(
                        "galaxy must provide sample_physical() or "
                        "to_deterministic_physical()"
                    )
                physical_draws.append(dict(draw))
                latent_draws.append((vals, fluxes))

            if not physical_draws:
                return {}
            result = {
                key: np.asarray(
                    [draw[key] for draw in physical_draws], dtype=float
                ).reshape(out_shape)
                for key in physical_draws[0]
            }
            if self._theta_star_fn is not None:
                result["thetaS"] = np.asarray(
                    [vals["thetaS"] for vals, _ in latent_draws]
                ).reshape(out_shape)
                for name in flux_dict:
                    result[f"Fs_{name}"] = np.asarray(
                        [fluxes[name]["Fs"] for _, fluxes in latent_draws]
                    ).reshape(out_shape)
            return result

        def _latent_theta_star_candidates(self, vals, fluxes, conditionals):
            from scipy.special import ndtri

            options = self._theta_star_options
            dimensions = len(conditionals) + 1
            points = _sobol_points(options["samples"], dimensions, options["seed"])
            candidates = []
            for point in points:
                draw = _draw_fluxes(fluxes, conditionals, point[:-1])
                try:
                    center, sigma = _theta_star_result(self._theta_star_fn, draw)
                    theta_star = math.exp(
                        center if sigma == 0.0 else center + sigma * ndtri(point[-1])
                    )
                except (ValueError, FloatingPointError, OverflowError):
                    continue
                current = dict(vals)
                current["thetaS"] = theta_star
                candidates.append((current, draw))
            if not candidates:
                raise RuntimeError("theta_star() produced no valid conditional draws")
            return candidates

        # --- internal helpers ---

        def has_py_extras(self) -> bool:
            return bool(
                self._extra_liks or self._extra_priors or self._guards
                or self._theta_star_fn is not None
            )

        def has_reparams(self) -> bool:
            return bool(self._reparam_blocks)

        def validate(self):
            if self._theta_star_fn is not None and self._lik_mode is None:
                raise RuntimeError(
                    "model.theta_star requires a base likelihood configured with "
                    "model.likelihood(...)"
                )
            if self._lik_mode is None and not self._extra_liks:
                raise RuntimeError(
                    "bayes.Model: likelihood is not configured.\n"
                    "  Use model.likelihood('gaussian')  "
                    "— Gaussian chi2 (C++ fast path)\n"
                    "  or @model.likelihood               "
                    "— custom Python function"
                )

        def _theta_to_vals(self, theta) -> dict:
            return {
                name: math.exp(theta[i]) if self._param_is_log.get(name) else theta[i]
                for i, name in enumerate(self.param_names)
            }

        @property
        def _sample_transforms(self):
            return [
                "log" if self._param_is_log.get(name) else "identity"
                for name in self.param_names
            ]

        def _is_log_prior(self, prior) -> bool:
            try:
                from ._lcbinint import bayes as _b
                return isinstance(prior, _b.LogUniform)
            except Exception:
                return False

        def _reparam_targets(self):
            out = set()
            for block in self._reparam_blocks:
                out.update(block.targets)
            return out

        def _sampling_adapter(self):
            return _SamplingAdapter(self)

        def log_prob_python(self, theta) -> float:
            """log_prob including all Python extras."""
            lp = super().log_prior(theta)
            if not math.isfinite(lp):
                return lp
            vals = self._theta_to_vals(theta)
            for fn in self._guards:
                if not _call_guard(fn, vals):
                    return float("-inf")
            base_prob, fluxes, conditionals = super()._log_prob_and_fluxes(theta)
            if not math.isfinite(base_prob):
                return base_prob
            lp += base_prob - super().log_prior(theta)

            if self._theta_star_fn is not None:
                if "thetaS" in vals:
                    lp += _apply_sampled_theta_star(
                        self._theta_star_fn,
                        vals,
                        fluxes,
                        conditionals,
                        self._theta_star_options,
                    )
                else:
                    lp += _marginalize_theta_star(
                        self._theta_star_fn,
                        vals,
                        fluxes,
                        conditionals,
                        self._extra_priors,
                        self._theta_star_options,
                    )
                if not math.isfinite(lp):
                    return lp

            if self._theta_star_fn is None or "thetaS" in vals:
                for fn in self._extra_priors:
                    lp += _call_extra_prior(fn, vals)
                    if not math.isfinite(lp):
                        return lp
                    if hasattr(fn, "physical_values"):
                        vals.update(fn.physical_values(vals))

            context = LikelihoodContext(fluxes=fluxes)
            for fn, wants_context in self._extra_liks:
                if wants_context:
                    lp += fn(**vals, context=context)
                else:
                    lp += fn(**vals)
                if not math.isfinite(lp):
                    return lp
            return lp

        def log_prob(self, theta) -> float:
            return self.log_prob_python(theta)

    return Model
