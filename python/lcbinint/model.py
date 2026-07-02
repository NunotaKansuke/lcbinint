"""Python-extended bayes.Model with @model.prior / @model.likelihood decorators."""
from __future__ import annotations

import math


def _build_model_class(cpp_base):
    """Build the Python Model subclass from the C++ _Model base.

    Called once from __init__.py after _lcbinint is loaded.
    """

    class Model(cpp_base):
        """Extended bayes.Model supporting custom Python prior/likelihood terms.

        Inherits all C++ bayes._Model functionality. Adds:

        - ``model.likelihood("gaussian")``  — set base likelihood mode (C++ path)
        - ``@model.likelihood``             — add a Python log-likelihood term
        - ``@model.prior``                  — add a Python log-prior term
        - ``model.param(name)``             — flat improper prior (no bounds required)

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

        def __init__(self, lc, event_or_data):
            super().__init__(lc, event_or_data)
            self._extra_liks   = []
            self._extra_priors = []
            self._param_is_log = {}
            self._lik_mode     = None

        # --- param: supports no-prior (flat) ---

        def param(self, name: str, prior=None):
            """Register a sampling parameter.

            If *prior* is omitted, a flat (improper) prior is used:
            ``log_prob = 0`` everywhere, no hard bounds.
            """
            from ._lcbinint import bayes as _b
            if prior is None:
                prior = _b.Flat()
            super().param(name, prior)
            self._param_is_log[name] = isinstance(prior, _b.LogUniform)
            return self

        # --- likelihood ---

        def likelihood(self, arg="gaussian"):
            """Set or add a likelihood term.

            ``model.likelihood("gaussian")`` — use C++ Gaussian chi2.
            ``@model.likelihood``            — add a Python log-likelihood function.
            """
            if isinstance(arg, str):
                super().likelihood(arg)
                self._lik_mode = arg
            elif callable(arg):
                self._extra_liks.append(arg)
                return arg  # enables @model.likelihood
            else:
                raise TypeError(
                    f"likelihood() expects str or callable, got {type(arg).__name__}")

        # --- prior ---

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

        # --- internal helpers ---

        def has_py_extras(self) -> bool:
            return bool(self._extra_liks or self._extra_priors)

        def validate(self):
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

        def log_prob_python(self, theta) -> float:
            """log_prob including all Python extras."""
            lp = super().log_prob(theta)
            if not math.isfinite(lp):
                return lp
            vals = self._theta_to_vals(theta)
            for fn in self._extra_priors:
                lp += fn(**vals)
                if not math.isfinite(lp):
                    return lp
            for fn in self._extra_liks:
                lp += fn(**vals)
                if not math.isfinite(lp):
                    return lp
            return lp

        def log_prob(self, theta) -> float:
            return self.log_prob_python(theta)

    return Model
