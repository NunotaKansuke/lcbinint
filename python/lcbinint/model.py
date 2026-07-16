"""Python-extended bayes.Model with decorators and sampling reparameterization."""
from __future__ import annotations

import inspect
import math
from collections.abc import Mapping

import numpy as np


class LikelihoodContext:
    """Auxiliary values from the base likelihood evaluation.

    ``fluxes`` maps each dataset name to its current ``Fs`` and ``Fb``.
    ``conditionals`` contains conditional flux-posterior parameters when flux
    is marginalized.
    """

    def __init__(self, *, fluxes, conditionals=None, flux_mode=None):
        self.fluxes = fluxes
        self.conditionals = conditionals or {}
        self.flux_mode = flux_mode


def _context_callable_uses_likelihood(fn):
    """Return whether a context factory accepts ``(params, likelihood)``."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    try:
        signature.bind({}, object())
    except TypeError:
        try:
            signature.bind({})
        except TypeError as exc:
            raise TypeError(
                "galactic_prior() context callable must accept either "
                "(params) or (params, likelihood)"
            ) from exc
        return False
    return True


def _make_galactic_context(factory, vals, likelihood_context=None):
    if not callable(factory):
        context = factory
    elif not _context_callable_uses_likelihood(factory):
        context = factory(dict(vals))
    else:
        if likelihood_context is None:
            raise RuntimeError(
                "galactic_prior() context requires likelihood auxiliaries, but "
                "none are available"
            )
        if likelihood_context.flux_mode == "marginalize":
            raise RuntimeError(
                "galactic_prior() context cannot use likelihood fluxes with "
                "flux='marginalize' yet; the Galactic term must be integrated "
                "jointly over the conditional flux distribution. Use "
                "flux='sample' or the best-fit flux mode for now."
            )
        context = factory(dict(vals), likelihood_context)

    if "thetaS" not in vals:
        return context
    if context is None:
        context = {}
    if not isinstance(context, Mapping):
        raise TypeError(
            "galactic_prior() context must be a dict when model.theta_star() "
            "is configured"
        )
    context = dict(context)
    if "thS" in context:
        raise ValueError(
            "galactic_prior() context must not define 'thS'; configure "
            "model.theta_star() and lcbinint will inject it automatically"
        )
    context["thS"] = vals["thetaS"]
    return context


def _make_galactic_magnitudes(factory, vals, likelihood_context=None):
    if factory is None or not callable(factory):
        return factory
    if not _context_callable_uses_likelihood(factory):
        return factory(dict(vals))
    if likelihood_context is None:
        raise RuntimeError(
            "galactic_prior() magnitudes requires likelihood auxiliaries, "
            "but none are available"
        )
    if likelihood_context.flux_mode == "marginalize":
        raise RuntimeError(
            "galactic_prior() magnitudes cannot use likelihood fluxes with "
            "flux='marginalize' directly; source photometry must be integrated "
            "jointly over the conditional flux distribution. Configure "
            "model.theta_star() to perform that joint integration, or use "
            "flux='sample' or the best-fit flux mode."
        )
    result = factory(dict(vals), likelihood_context)
    if result is not None and not isinstance(result, dict):
        raise TypeError("galactic_prior() magnitudes callable must return a dict")
    return result


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


def _isochrone_magnitudes_result(fn, fluxes):
    result = fn(fluxes)
    if not isinstance(result, dict) or not result:
        raise TypeError(
            "theta_star(isochrone=...) callable must return a non-empty "
            "magnitudes dict"
        )
    values = {name: float(value) for name, value in result.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("theta_star() magnitudes must be finite")
    return values


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


def _evaluate_extra_priors_many(
    priors, values, likelihood_contexts=None, *, return_values=False
):
    current = [dict(vals) for vals in values]
    if likelihood_contexts is None:
        likelihood_contexts = [None] * len(current)
    elif len(likelihood_contexts) != len(current):
        raise ValueError("likelihood_contexts must match values")
    ordered = _ordered_extra_priors(priors, current[0] if current else {})
    totals = np.zeros(len(current), dtype=float)
    for prior_index, fn in enumerate(ordered):
        active = np.flatnonzero(np.isfinite(totals))
        if not len(active):
            break
        active_values = [current[index] for index in active]
        active_contexts = [likelihood_contexts[index] for index in active]
        if isinstance(fn, _GalacticModelTerm):
            active_batch = fn.batch_log_prob(active_values, active_contexts)
        else:
            active_batch = (
                fn.batch_log_prob(active_values)
                if hasattr(fn, "batch_log_prob")
                else None
            )
        if active_batch is None:
            active_batch = np.asarray(
                [
                    _call_extra_prior(fn, vals, active_contexts[i])
                    for i, vals in enumerate(active_values)
                ],
                dtype=float,
            )
        batch = np.full(len(current), float("-inf"), dtype=float)
        batch[active] = active_batch
        totals += batch
        if prior_index + 1 < len(ordered) and hasattr(fn, "physical_values"):
            for i in active:
                vals = current[i]
                if math.isfinite(totals[i]):
                    if isinstance(fn, _GalacticModelTerm):
                        derived = fn.physical_values(vals, likelihood_contexts[i])
                    else:
                        derived = fn.physical_values(vals)
                    vals.update(derived)
    return (totals, current) if return_values else totals


def _ordered_extra_priors(priors, vals):
    """Evaluate cheap available priors before Galactic-derived priors."""
    early = []
    galactic = []
    derived = []
    for fn in priors:
        if isinstance(fn, _GalacticModelTerm):
            galactic.append(fn)
        elif _missing_required_kwargs(fn, vals):
            derived.append(fn)
        else:
            early.append(fn)
    return early + galactic + derived


def _theta_star_candidates(
    fn,
    vals,
    fluxes,
    conditionals,
    options,
    flux_mode,
):
    from scipy.special import ndtri

    samples = []
    likelihood_contexts = []
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
            likelihood_contexts.append(
                LikelihoodContext(
                    fluxes=draw,
                    flux_mode=(
                        "conditional_draw" if conditionals else flux_mode
                    ),
                )
            )

    return samples, likelihood_contexts, invalid_count


def _marginalize_theta_star(
    fn,
    vals,
    fluxes,
    conditionals,
    priors,
    options,
    flux_mode,
):
    samples, likelihood_contexts, invalid_count = _theta_star_candidates(
        fn,
        vals,
        fluxes,
        conditionals,
        options,
        flux_mode,
    )
    if not samples:
        return float("-inf")
    log_weights = _evaluate_extra_priors_many(
        priors, samples, likelihood_contexts
    )
    if invalid_count:
        log_weights = np.concatenate(
            [log_weights, np.full(invalid_count, float("-inf"))]
        )
    return _logmeanexp(log_weights)


def _matching_isochrone_term(priors, isochrone):
    matches = []
    for term in priors:
        if not isinstance(term, _GalacticModelTerm):
            continue
        if (
            getattr(term.galaxy, "isochrone", None) is isochrone
            and hasattr(term.galaxy, "_isochrone_conditional_terms")
        ):
            matches.append(term)
    if len(matches) != 1:
        raise RuntimeError(
            "theta_star(isochrone=...) requires exactly one Galactic prior "
            "parameterized from that isochrone"
        )
    return matches[0]


def _isochrone_conditional_candidates(
    fn,
    isochrone,
    vals,
    fluxes,
    conditionals,
    priors,
    options,
    flux_mode,
):
    term = _matching_isochrone_term(priors, isochrone)
    if term.magnitudes is not None:
        raise RuntimeError(
            "theta_star(isochrone=...) supplies source magnitudes; remove "
            "magnitudes= from model.galactic_prior()"
        )
    if term.context_uses_likelihood:
        raise RuntimeError(
            "theta_star(isochrone=...) requires Galactic context independent "
            "of likelihood fluxes"
        )
    samples = int(getattr(term.galaxy, "integration_samples", 512))
    if conditionals:
        points = _sobol_points(samples, len(conditionals), options["seed"])
        flux_draws = [
            _draw_fluxes(fluxes, conditionals, point) for point in points
        ]
    else:
        flux_draws = [fluxes] * samples

    magnitudes = []
    valid = np.ones(samples, dtype=bool)
    first_valid = None
    for index, draw in enumerate(flux_draws):
        try:
            current = _isochrone_magnitudes_result(fn, draw)
            first_valid = current if first_valid is None else first_valid
            magnitudes.append(current)
        except (TypeError, ValueError, FloatingPointError, OverflowError):
            valid[index] = False
            magnitudes.append(None)
    if first_valid is None:
        return [], np.full(samples, float("-inf")), term
    magnitudes = [first_valid if item is None else item for item in magnitudes]
    keys = tuple(first_valid)
    if any(tuple(item) != keys for item in magnitudes):
        raise ValueError(
            "theta_star() magnitudes must use the same bands for every draw"
        )
    stacked_magnitudes = (
        first_valid
        if not conditionals
        else {
            key: np.asarray([item[key] for item in magnitudes], dtype=float)
            for key in keys
        }
    )
    missing = [name for name in term.names if name not in vals]
    if missing:
        raise RuntimeError(
            "galactic_prior() missing model parameter(s): " + ", ".join(missing)
        )
    theta = [vals[name] for name in term.names]
    context = term._context(vals, None)
    result = term.galaxy._isochrone_conditional_terms(
        theta,
        magnitudes=stacked_magnitudes,
        context=context,
    )
    provider_terms = np.asarray(result["log_terms"], dtype=float)
    physical = {
        key: np.asarray(value, dtype=float)
        for key, value in result["physical"].items()
    }
    candidate_values = []
    likelihood_contexts = []
    for index, draw in enumerate(flux_draws):
        current = dict(vals)
        current.update({key: value[index] for key, value in physical.items()})
        candidate_values.append(current)
        likelihood_contexts.append(
            LikelihoodContext(
                fluxes=draw,
                flux_mode="conditional_draw" if conditionals else flux_mode,
            )
        )
    remaining_priors = [prior for prior in priors if prior is not term]
    extra_terms = _evaluate_extra_priors_many(
        remaining_priors,
        candidate_values,
        likelihood_contexts,
    )
    log_terms = provider_terms + extra_terms
    log_terms = np.where(valid, log_terms, float("-inf"))
    candidates = [
        (candidate_values[index], flux_draws[index], {
            key: value[index] for key, value in physical.items() if key != "thetaS"
        })
        for index in range(samples)
    ]
    return candidates, log_terms, term


def _marginalize_isochrone_theta_star(*args, **kwargs):
    _, log_terms, _ = _isochrone_conditional_candidates(*args, **kwargs)
    return _logmeanexp(log_terms)


class _GalacticModelTerm:
    """Adapter for external Galactic model objects.

    The external object is expected to expose ``log_density(theta, ...)``.
    ``theta`` is assembled from lcbinint physical parameter values in ``names``
    order. Keeping this as a normal extra prior avoids making lcbinint depend on
    gapmoe directly.
    """

    def __init__(
        self,
        galaxy,
        *,
        names=None,
        context=None,
        magnitudes=None,
    ):
        if not hasattr(galaxy, "log_density") or not callable(galaxy.log_density):
            raise TypeError("galactic_prior() expects an object with log_density()")
        if names is None:
            names = getattr(galaxy, "names", None)
        if names is None:
            param_type = getattr(galaxy, "param_type", None)
            names = getattr(param_type, "names", None)
        if names is None:
            raise ValueError(
                "galactic_prior() requires names=... when "
                "galaxy.names is unavailable"
            )
        if isinstance(names, str):
            raise TypeError("galactic_prior() names must be a sequence, not a string")
        self.galaxy = galaxy
        self.names = tuple(names)
        if not self.names:
            raise ValueError("galactic_prior() requires at least one parameter name")
        self.context = context
        if isinstance(context, Mapping) and "thS" in context:
            raise ValueError(
                "galactic_prior() context must not define 'thS'; configure "
                "model.theta_star() and lcbinint will inject it automatically"
            )
        self.magnitudes = magnitudes
        self.context_uses_likelihood = bool(
            callable(context) and _context_callable_uses_likelihood(context)
        )
        self.magnitudes_use_likelihood = bool(
            callable(magnitudes)
            and _context_callable_uses_likelihood(magnitudes)
        )
        self.uses_likelihood_context = bool(
            self.context_uses_likelihood or self.magnitudes_use_likelihood
        )
        self._batch_fn = None
        self._batch_key = None
        self._batch_disabled = False

    def __call__(self, **vals):
        return self.evaluate(vals)

    def evaluate(self, vals, likelihood_context=None):
        missing = [name for name in self.names if name not in vals]
        if missing:
            raise RuntimeError(
                "galactic_prior() missing model parameter(s): "
                + ", ".join(missing)
            )
        theta = [vals[name] for name in self.names]
        context = self._context(vals, likelihood_context)
        magnitudes = self._magnitudes(vals, likelihood_context)
        kwargs = {}
        if context is not None:
            kwargs["context"] = context
        if magnitudes is not None:
            kwargs["magnitudes"] = magnitudes
        return float(self.galaxy.log_density(theta, **kwargs))

    def physical_values(self, vals, likelihood_context=None):
        theta = [vals[name] for name in self.names]
        context = self._context(vals, likelihood_context)
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

    def precheck(self, vals):
        validator = getattr(self.galaxy, "is_valid", None)
        if validator is None or self.context_uses_likelihood:
            return True
        if any(name not in vals for name in self.names):
            return True
        try:
            context = self._context(vals)
        except (KeyError, RuntimeError, ValueError):
            return True
        theta = [vals[name] for name in self.names]
        try:
            if context is None:
                return bool(validator(theta))
            return bool(validator(theta, context=context))
        except (KeyError, RuntimeError, ValueError):
            # A latent thetaS is only available after the likelihood has
            # produced fluxes. The complete Galactic evaluation will validate
            # the physical mapping once thetaS has been injected.
            return True

    def batch_log_prob(self, values, likelihood_contexts=None):
        """Evaluate a JAX-compatible Galactic model for many latent states."""
        if self._batch_disabled:
            return None
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            self._batch_disabled = True
            return None

        try:
            if likelihood_contexts is None:
                likelihood_contexts = [None] * len(values)
            contexts = [
                self._context(vals, likelihood_contexts[index])
                for index, vals in enumerate(values)
            ]
            magnitudes = [
                self._magnitudes(vals, likelihood_contexts[index])
                for index, vals in enumerate(values)
            ]
            has_context = contexts[0] is not None
            has_magnitudes = magnitudes[0] is not None
            if any((item is not None) != has_context for item in contexts):
                raise ValueError("Galactic contexts must be present for every batch item")
            if any((item is not None) != has_magnitudes for item in magnitudes):
                raise ValueError("Galactic magnitudes must be present for every batch item")

            batch_key = (has_context, has_magnitudes)
            if self._batch_fn is None or self._batch_key != batch_key:
                galaxy = self.galaxy
                if has_context and has_magnitudes:
                    one = lambda theta, ctx, mags: galaxy.log_density(
                        theta, context=ctx, magnitudes=mags
                    )
                    self._batch_fn = jax.jit(jax.vmap(one))
                elif has_context:
                    one = lambda theta, ctx: galaxy.log_density(
                        theta, context=ctx
                    )
                    self._batch_fn = jax.jit(jax.vmap(one))
                elif has_magnitudes:
                    one = lambda theta, mags: galaxy.log_density(
                        theta, magnitudes=mags
                    )
                    self._batch_fn = jax.jit(jax.vmap(one))
                else:
                    self._batch_fn = jax.jit(jax.vmap(galaxy.log_density))
                self._batch_key = batch_key

            theta = jnp.asarray([
                [vals[name] for name in self.names]
                for vals in values
            ])
            args = [theta]
            if has_context:
                args.append(_stack_pytrees(contexts, jax, jnp))
            if has_magnitudes:
                args.append(_stack_pytrees(magnitudes, jax, jnp))
            result = self._batch_fn(*args)
            return np.asarray(result, dtype=float)
        except (
            TypeError,
            ValueError,
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerArrayConversionError,
        ):
            self._batch_disabled = True
            self._batch_fn = None
            self._batch_key = None
            return None

    def _context(self, vals, likelihood_context=None):
        return _make_galactic_context(self.context, vals, likelihood_context)

    def _magnitudes(self, vals, likelihood_context=None):
        return _make_galactic_magnitudes(
            self.magnitudes,
            vals,
            likelihood_context,
        )


def _stack_pytrees(items, jax, jnp):
    return jax.tree_util.tree_map(
        lambda *values: jnp.asarray(values),
        *items,
    )


def _call_extra_prior(fn, vals, likelihood_context=None):
    if isinstance(fn, _GalacticModelTerm):
        return fn.evaluate(vals, likelihood_context)
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
        self._model.validate()
        phys = self._theta_to_physical(theta)
        lp = self._log_prior(theta, phys)
        if not math.isfinite(lp):
            return lp

        vals = dict(phys)
        for fn in self._model._guards:
            if not _call_guard(fn, vals):
                return float("-inf")
        for fn in self._model._extra_priors:
            if isinstance(fn, _GalacticModelTerm) and not fn.precheck(vals):
                return float("-inf")

        phys_model = self._ensure_physical_model()
        phys_theta = [phys[name] for name in self._phys_names]
        base_prob, fluxes, conditionals = phys_model._log_prob_and_fluxes(phys_theta)
        if not math.isfinite(base_prob):
            return base_prob
        self._aux_cache[tuple(theta)] = (fluxes, conditionals)
        lp += base_prob - phys_model.log_prior(phys_theta)
        likelihood_context = LikelihoodContext(
            fluxes=fluxes,
            conditionals=conditionals,
            flux_mode=self._model._lik_kwargs.get("flux"),
        )

        if self._model._theta_star_fn is not None:
            marginalizer = (
                _marginalize_isochrone_theta_star
                if self._model._theta_star_isochrone is not None
                else _marginalize_theta_star
            )
            args = [
                self._model._theta_star_fn,
            ]
            if self._model._theta_star_isochrone is not None:
                args.append(self._model._theta_star_isochrone)
            lp += marginalizer(
                *args,
                vals,
                fluxes,
                conditionals,
                self._model._extra_priors,
                self._model._theta_star_options,
                self._model._lik_kwargs["flux"],
            )
            if not math.isfinite(lp):
                return lp

        if self._model._theta_star_fn is None:
            for fn in _ordered_extra_priors(self._model._extra_priors, vals):
                lp += _call_extra_prior(fn, vals, likelihood_context)
                if not math.isfinite(lp):
                    return lp
                if hasattr(fn, "physical_values"):
                    if isinstance(fn, _GalacticModelTerm):
                        derived = fn.physical_values(vals, likelihood_context)
                    else:
                        derived = fn.physical_values(vals)
                    vals.update(derived)

        for fn, wants_context in self._model._extra_liks:
            if wants_context:
                lp += fn(**vals, context=likelihood_context)
            else:
                lp += fn(**vals)
            if not math.isfinite(lp):
                return lp
        return lp

    def log_prob_batch(self, theta_batch):
        """Evaluate an ensemble proposal batch with batched extra priors."""
        rows = [list(theta) for theta in theta_batch]
        if not rows:
            return []

        # Isochrone theta-star marginalization already batches its internal
        # quadrature. Combining that axis with walkers needs a provider-level
        # API, so retain the scalar behavior for this less common path.
        if self._model._theta_star_isochrone is not None:
            return [self.log_prob(theta) for theta in rows]

        self._model.validate()
        phys_model = self._ensure_physical_model()
        results = np.full(len(rows), float("-inf"), dtype=float)
        values = [None] * len(rows)
        contexts = [None] * len(rows)
        active = []

        for index, theta in enumerate(rows):
            phys = self._theta_to_physical(theta)
            lp = self._log_prior(theta, phys)
            if not math.isfinite(lp):
                results[index] = lp
                continue

            vals = dict(phys)
            if any(not _call_guard(fn, vals) for fn in self._model._guards):
                continue
            if any(
                isinstance(fn, _GalacticModelTerm) and not fn.precheck(vals)
                for fn in self._model._extra_priors
            ):
                continue

            phys_theta = [phys[name] for name in self._phys_names]
            base_prob, fluxes, conditionals = phys_model._log_prob_and_fluxes(
                phys_theta
            )
            if not math.isfinite(base_prob):
                results[index] = base_prob
                continue

            self._aux_cache[tuple(theta)] = (fluxes, conditionals)
            results[index] = (
                lp + base_prob - phys_model.log_prior(phys_theta)
            )
            values[index] = vals
            contexts[index] = LikelihoodContext(
                fluxes=fluxes,
                conditionals=conditionals,
                flux_mode=self._model._lik_kwargs.get("flux"),
            )
            active.append(index)

        if not active:
            return results.tolist()

        if self._model._theta_star_fn is None:
            prior_totals, prior_values = _evaluate_extra_priors_many(
                self._model._extra_priors,
                [values[index] for index in active],
                [contexts[index] for index in active],
                return_values=True,
            )
            for local, index in enumerate(active):
                results[index] += prior_totals[local]
                values[index] = prior_values[local]
        else:
            candidate_values = []
            candidate_contexts = []
            groups = []
            for index in active:
                samples, sample_contexts, invalid_count = _theta_star_candidates(
                    self._model._theta_star_fn,
                    values[index],
                    contexts[index].fluxes,
                    contexts[index].conditionals,
                    self._model._theta_star_options,
                    self._model._lik_kwargs["flux"],
                )
                begin = len(candidate_values)
                candidate_values.extend(samples)
                candidate_contexts.extend(sample_contexts)
                groups.append((index, begin, len(candidate_values), invalid_count))

            group_sizes = [end - begin for _, begin, end, _ in groups]
            if all(size == 1 for size in group_sizes):
                candidate_weights = _evaluate_extra_priors_many(
                    self._model._extra_priors,
                    candidate_values,
                    candidate_contexts,
                )
            else:
                # Keep nested flux/thetaS quadrature separate by walker. A
                # flattened walker x quadrature x Galactic-QMC array can be
                # much larger than either useful batching axis.
                candidate_weights = np.empty(len(candidate_values), dtype=float)
                for _, begin, end, _ in groups:
                    candidate_weights[begin:end] = _evaluate_extra_priors_many(
                        self._model._extra_priors,
                        candidate_values[begin:end],
                        candidate_contexts[begin:end],
                    )
            for index, begin, end, invalid_count in groups:
                weights = candidate_weights[begin:end]
                if invalid_count:
                    weights = np.concatenate([
                        weights,
                        np.full(invalid_count, float("-inf")),
                    ])
                results[index] += _logmeanexp(weights)

        for index in active:
            if not math.isfinite(results[index]):
                continue
            vals = values[index]
            likelihood_context = contexts[index]
            for fn, wants_context in self._model._extra_liks:
                if wants_context:
                    results[index] += fn(**vals, context=likelihood_context)
                else:
                    results[index] += fn(**vals)
                if not math.isfinite(results[index]):
                    break

        return results.tolist()

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
            self._theta_star_isochrone = None
            self._param_is_log = {}
            self._param_priors = {}
            self._lik_mode     = None
            self._lik_kwargs   = {}
            self._reparam_blocks = []

        # --- sampled parameters ---

        def param(self, name: str, prior=None):
            """Register a sampling parameter.

            Every sampled parameter requires an explicit prior.
            """
            if name == "thetaS":
                raise ValueError(
                    "thetaS is derived exclusively by model.theta_star(); "
                    "return (log(value), 0.0) there to fix it"
                )
            if prior is None:
                raise ValueError(
                    f"model.param({name!r}) requires an explicit prior"
                )
            from ._lcbinint import bayes as _b
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
            if "thetaS" in targets:
                raise ValueError(
                    "thetaS cannot be a reparameterization target; define it "
                    "with model.theta_star()"
                )
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
                kwargs.setdefault("flux", "fit")
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

        def theta_star(self, fn=None, *, samples=256, seed=0, isochrone=None):
            """Set the log-space thetaS relation evaluated from fitted fluxes.

            The callable receives the complete flux dictionary and returns
            ``(log_center, log_sigma)``. ``thetaS`` is always derived here and
            is never a sampling parameter. A positive sigma is marginalized;
            zero fixes ``thetaS = exp(log_center)``. Marginalized fluxes and
            thetaS use ``samples`` scrambled Sobol draws (default 256).
            With ``isochrone=...``, the callable instead returns apparent
            magnitudes. lcbinint and the matching Galactic prior then perform
            one paired QMC integral over flux, thetaS, and hidden physics.
            """
            if fn is None:
                return lambda decorated: self.theta_star(
                    decorated,
                    samples=samples,
                    seed=seed,
                    isochrone=isochrone,
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
            self._theta_star_isochrone = isochrone
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

        def galactic_prior(
            self,
            galaxy,
            *,
            context=None,
            magnitudes=None,
            names=None,
        ):
            """Add an external Galactic model as a prior term.

            lcbinint does not import gapmoe. It accepts any independently
            constructed object exposing ``names`` and ``log_density()``. The
            parameter vector is assembled from model physical values in
            ``names`` order.

            ``context`` may be a dict-like object or a callable receiving the
            current physical parameter dict and returning a context object.
            A two-argument callable ``context(params, likelihood)`` also
            receives a :class:`LikelihoodContext`. ``magnitudes`` may be a
            fixed dict or a callable with the same one- or two-argument forms.
            Source magnitudes condition the Galactic density; their marginal
            CMD density is not added as a prior on source flux.
            """
            if any(
                isinstance(item, _GalacticModelTerm)
                for item in self._extra_priors
            ):
                raise RuntimeError(
                    "galactic_prior() is already configured; add scientific "
                    "constraints with @model.prior or provider.prior"
                )
            term = _GalacticModelTerm(
                galaxy,
                names=names,
                context=context,
                magnitudes=magnitudes,
            )
            self._extra_priors.append(term)
            return term

        def get_galactic_physical(
            self,
            chain,
            galaxy,
            *,
            context=None,
            magnitudes=None,
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
                names = getattr(galaxy, "names", None)
            if names is None:
                param_type = getattr(galaxy, "param_type", None)
                names = getattr(param_type, "names", None)
            if names is None:
                raise ValueError(
                    "get_galactic_physical() requires names=... when "
                    "galaxy.names is unavailable"
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
                precomputed_draw = None
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
                    if self._theta_star_isochrone is not None:
                        candidates, weights, _ = _isochrone_conditional_candidates(
                            self._theta_star_fn,
                            self._theta_star_isochrone,
                            vals,
                            fluxes,
                            conditionals,
                            self._extra_priors,
                            self._theta_star_options,
                            self._lik_kwargs["flux"],
                        )
                    else:
                        candidates = self._latent_theta_star_candidates(
                            vals, fluxes, conditionals
                        )
                        candidate_contexts = [
                            LikelihoodContext(
                                fluxes=item[1],
                                flux_mode="conditional_draw" if conditionals else self._lik_kwargs.get("flux"),
                            )
                            for item in candidates
                        ]
                        weights = _evaluate_extra_priors_many(
                            self._extra_priors,
                            [item[0] for item in candidates],
                            candidate_contexts,
                        )
                    finite = np.isfinite(weights)
                    if not np.any(finite):
                        raise RuntimeError(
                            "no finite conditional thetaS/flux draw for chain sample"
                        )
                    peak = np.max(weights[finite])
                    probs = np.where(finite, np.exp(weights - peak), 0.0)
                    probs /= probs.sum()
                    selected = candidates[int(rng.choice(len(candidates), p=probs))]
                    vals, fluxes = selected[:2]
                    if self._theta_star_isochrone is not None:
                        precomputed_draw = selected[2]

                missing = [name for name in names if name not in vals]
                if missing:
                    raise RuntimeError(
                        "get_galactic_physical() missing model parameter(s): "
                        + ", ".join(missing)
                    )
                theta = [vals[name] for name in names]
                likelihood_context = LikelihoodContext(
                    fluxes=fluxes,
                    flux_mode=(
                        "conditional_draw"
                        if self._theta_star_fn is not None and "thetaS" not in vals_iter[index]
                        else self._lik_kwargs.get("flux")
                    ),
                )
                ctx = _make_galactic_context(
                    context, vals, likelihood_context
                )
                source_magnitudes = _make_galactic_magnitudes(
                    magnitudes,
                    vals,
                    likelihood_context,
                )
                if precomputed_draw is not None:
                    draw = precomputed_draw
                elif hasattr(galaxy, "sample_physical"):
                    kwargs = {"rng": rng}
                    if ctx is not None:
                        kwargs["context"] = ctx
                    if source_magnitudes is not None:
                        kwargs["magnitudes"] = source_magnitudes
                    draw = galaxy.sample_physical(theta, **kwargs)
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
                candidates.append((current, draw, None))
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
            if self._theta_star_isochrone is not None:
                term = _matching_isochrone_term(
                    self._extra_priors, self._theta_star_isochrone
                )
                if term.magnitudes is not None:
                    raise RuntimeError(
                        "theta_star(isochrone=...) supplies magnitudes; remove "
                        "magnitudes= from model.galactic_prior()"
                    )
                if term.context_uses_likelihood:
                    raise RuntimeError(
                        "theta_star(isochrone=...) requires Galactic context "
                        "independent of likelihood fluxes"
                    )
            if self._lik_mode is None and not self._extra_liks:
                raise RuntimeError(
                    "bayes.Model: likelihood is not configured.\n"
                    "  Use model.likelihood('gaussian')  "
                    "— Gaussian chi2 (C++ fast path)\n"
                    "  or @model.likelihood               "
                    "— custom Python function"
                )
            if self._lik_mode is None:
                return

            flux_mode = self._lik_kwargs["flux"]
            parameter_names = set(self.param_names)
            flux_names = {
                name
                for dataset in self.dataset_names
                for name in (f"Fs_{dataset}", f"Fb_{dataset}")
            }
            registered_flux = parameter_names & flux_names
            if flux_mode == "sample":
                missing = sorted(flux_names - registered_flux)
                if missing:
                    raise RuntimeError(
                        "flux='sample' requires sampled parameter(s): "
                        + ", ".join(missing)
                    )
            elif registered_flux:
                raise RuntimeError(
                    f"flux={flux_mode!r} cannot be combined with sampled flux "
                    "parameter(s): " + ", ".join(sorted(registered_flux))
                )

            if flux_mode == "marginalize":
                flux_aware_galaxy = any(
                    isinstance(fn, _GalacticModelTerm)
                    and fn.uses_likelihood_context
                    for fn in self._extra_priors
                )
                if flux_aware_galaxy and self._theta_star_fn is None:
                    raise RuntimeError(
                        "flux='marginalize' with flux-dependent Galactic "
                        "context or magnitudes requires marginalized thetaS: "
                        "configure model.theta_star(...)"
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
            self.validate()
            lp = super().log_prior(theta)
            if not math.isfinite(lp):
                return lp
            vals = self._theta_to_vals(theta)
            for fn in self._guards:
                if not _call_guard(fn, vals):
                    return float("-inf")
            for fn in self._extra_priors:
                if isinstance(fn, _GalacticModelTerm) and not fn.precheck(vals):
                    return float("-inf")
            base_prob, fluxes, conditionals = super()._log_prob_and_fluxes(theta)
            if not math.isfinite(base_prob):
                return base_prob
            lp += base_prob - super().log_prior(theta)
            likelihood_context = LikelihoodContext(
                fluxes=fluxes,
                conditionals=conditionals,
                flux_mode=self._lik_kwargs.get("flux"),
            )

            if self._theta_star_fn is not None:
                marginalizer = (
                    _marginalize_isochrone_theta_star
                    if self._theta_star_isochrone is not None
                    else _marginalize_theta_star
                )
                args = [self._theta_star_fn]
                if self._theta_star_isochrone is not None:
                    args.append(self._theta_star_isochrone)
                lp += marginalizer(
                    *args,
                    vals,
                    fluxes,
                    conditionals,
                    self._extra_priors,
                    self._theta_star_options,
                    self._lik_kwargs["flux"],
                )
                if not math.isfinite(lp):
                    return lp

            if self._theta_star_fn is None:
                for fn in _ordered_extra_priors(self._extra_priors, vals):
                    lp += _call_extra_prior(fn, vals, likelihood_context)
                    if not math.isfinite(lp):
                        return lp
                    if hasattr(fn, "physical_values"):
                        if isinstance(fn, _GalacticModelTerm):
                            derived = fn.physical_values(vals, likelihood_context)
                        else:
                            derived = fn.physical_values(vals)
                        vals.update(derived)

            for fn, wants_context in self._extra_liks:
                if wants_context:
                    lp += fn(**vals, context=likelihood_context)
                else:
                    lp += fn(**vals)
                if not math.isfinite(lp):
                    return lp
            return lp

        def log_prob(self, theta) -> float:
            return self.log_prob_python(theta)

    return Model
