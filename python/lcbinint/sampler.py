"""High-level sampler: run_sampler, SamplerOptions, load_chain."""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from . import sample as _sample

try:
    import h5py as _h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


# ---------------------------------------------------------------------------
# Reparameterization
# ---------------------------------------------------------------------------

class Reparameterization:
    """Custom sampling parameterization for run_sampler().

    Allows arbitrary transforms from sampling parameters to the physical
    parameters expected by lcbinint. The C++ ensemble sampler runs the
    stretch-move loop in C++; log_prob is called per-walker via GIL callback.

    Example — sampling piE + phi_piE instead of piEN + piEE::

        r = Reparameterization(lc, data)
        r.param("t0",      bayes.Uniform(7000, 9000))
        r.param("piE",     bayes.LogUniform(0.01, 1.0))
        r.param("phi_piE", bayes.Uniform(0, 2 * math.pi))

        @r.transform
        def my_transform(t0, piE, phi_piE):
            return {"t0":   t0,
                    "piEN": piE * math.cos(phi_piE),
                    "piEE": piE * math.sin(phi_piE)}

        chain = run_sampler(r, nsteps=1000)

    Notes
    -----
    The Jacobian of the transform is *not* applied automatically.
    Include it in the prior if the reparameterization requires it.
    """

    def __init__(self, lc, data):
        self._lc   = lc
        self._data = data
        self._sample_params    = []   # [(name, prior)]
        self._transform_fn     = None
        self._log_jacobian_fn  = None
        self._phys_model       = None  # C++ bayes.Model, built lazily
        self._phys_names       = []    # physical param names in registered order

    # --- setup ---

    def param(self, name: str, prior) -> "Reparameterization":
        """Register a sampling parameter with its prior."""
        self._sample_params.append((name, prior))
        return self

    def transform(self, fn):
        """Set the transform function (sampling kwargs -> physical dict).

        Usable as a decorator or called directly::

            @r.transform
            def my_transform(piE, phi): ...

            r.transform(lambda piE, phi: {...})
        """
        self._transform_fn = fn
        return fn

    def log_jacobian(self, fn):
        """Set the log |det J| correction for the reparameterization.

        ``fn`` receives the same keyword arguments as ``transform`` (physical
        sampling values, with LogUniform params already exp()-converted) and
        must return a scalar::

            @r.log_jacobian
            def jac(tE, **_):
                return -3 * math.log(tE)   # for tE*u0, tE*rho, tE*q → u0, rho, q

        The value is added to log_prior on every evaluation.
        """
        self._log_jacobian_fn = fn
        return fn

    # --- properties consumed by run_sampler ---

    @property
    def param_names(self) -> list:
        return [n for n, _ in self._sample_params]

    @property
    def n_params(self) -> int:
        return len(self._sample_params)

    @property
    def n_data(self) -> int:
        return self._ensure_built()._phys_model.n_data

    @property
    def optimizer_bounds(self) -> list:
        """Bounds in sampling space (log-space for LogUniform)."""
        out = []
        for _, prior in self._sample_params:
            lo, hi = prior.bounds()
            if self._is_log(prior):
                out.append((math.log(lo), math.log(hi)))
            else:
                out.append((lo, hi))
        return out

    @property
    def _sample_transforms(self) -> list:
        """'log'/'identity' per sampling param — used by Chain.samples."""
        return [
            "log" if self._is_log(p) else "identity"
            for _, p in self._sample_params
        ]

    # --- log_prob ---

    def log_prior(self, theta: list) -> float:
        lp = 0.0
        for i, (_, prior) in enumerate(self._sample_params):
            if self._is_log(prior):
                t = theta[i]
                lp += prior.log_prob(math.exp(t)) + t   # Jacobian of log-transform
            else:
                lp += prior.log_prob(theta[i])
            if not math.isfinite(lp):
                return lp
        if self._log_jacobian_fn is not None:
            lp += self._log_jacobian_fn(**self._theta_to_vals(theta))
        return lp

    def log_likelihood(self, theta: list) -> float:
        self._ensure_built()
        phys_dict  = self._transform_fn(**self._theta_to_vals(theta))
        phys_theta = [phys_dict[n] for n in self._phys_names]
        return self._phys_model.log_likelihood(phys_theta)

    def log_prob(self, theta: list) -> float:
        lp = self.log_prior(theta)
        if not math.isfinite(lp):
            return lp
        return lp + self.log_likelihood(theta)

    # --- internals ---

    def _is_log(self, prior) -> bool:
        try:
            from ._lcbinint import bayes as _b
            return isinstance(prior, _b.LogUniform)
        except Exception:
            return False

    def _theta_to_vals(self, theta: list) -> dict:
        """Internal theta -> sampling values (exp for LogUniform)."""
        return {
            name: math.exp(theta[i]) if self._is_log(prior) else theta[i]
            for i, (name, prior) in enumerate(self._sample_params)
        }

    def _ensure_built(self) -> "Reparameterization":
        if self._phys_model is None:
            self._build()
        return self

    def _build(self):
        if self._transform_fn is None:
            raise RuntimeError(
                "Reparameterization: call transform() before use")
        if not self._sample_params:
            raise RuntimeError(
                "Reparameterization: register at least one param()")

        # Discover physical param names by test-calling the transform.
        test_vals = {}
        for name, prior in self._sample_params:
            lo, hi = prior.bounds()
            test_vals[name] = (
                math.exp(0.5 * (math.log(lo) + math.log(hi)))
                if self._is_log(prior) else 0.5 * (lo + hi)
            )
        result = self._transform_fn(**test_vals)
        if not isinstance(result, dict):
            raise TypeError(
                "Reparameterization: transform must return a dict "
                "{param_name: value}")
        self._phys_names = list(result.keys())

        # Build C++ model with physical params.
        # Priors are dummy Uniform — only log_likelihood is called on this model.
        from ._lcbinint import bayes as _b
        model = _b.Model(self._lc, self._data)
        for name in self._phys_names:
            model.param(name, _b.Uniform(-1e15, 1e15))
        self._phys_model = model


# ---------------------------------------------------------------------------
# Helpers for Python-extras path in run_sampler (bayes.Model with @model.*)
# ---------------------------------------------------------------------------

def _py_init_state_extended(model, nwalkers: int, seed: int, start, log_prob_fn):
    """Create a SamplerState for a bayes.Model that has Python extras."""
    ndim   = model.n_params()
    bounds = list(model.optimizer_bounds)
    rng    = np.random.default_rng(seed)

    pos = np.empty((nwalkers, ndim))
    if start is None:
        for j, (lo, hi) in enumerate(bounds):
            pos[:, j] = rng.uniform(lo, hi, nwalkers)
    else:
        center = np.asarray(
            getattr(start, "position", start), dtype=float)
        for j, (lo, hi) in enumerate(bounds):
            span  = hi - lo
            sigma = min(1e-2 * span,
                        0.5 * min(center[j] - lo, hi - center[j]) + 1e-12)
            v = center[j] + rng.standard_normal(nwalkers) * sigma
            pos[:, j] = np.clip(v, lo, hi)

    lp = np.array([log_prob_fn(pos[w].tolist()) for w in range(nwalkers)])
    return _sample._make_state(nwalkers, ndim, int(seed), pos, lp)


def _py_collect_extended(model, state):
    """Build a Chain from a SamplerState for an extended bayes.Model."""
    chain_arr = state.get_chain
    lp_arr    = state.get_log_prob
    return _sample._chain_from_arrays(
        chain_arr.reshape(-1, state.ndim),
        lp_arr,
        state.nwalkers,
        param_names=model.param_names,
        acceptance=state.acceptance_fraction,
    )


# ---------------------------------------------------------------------------
# Helpers for Reparameterization path in run_sampler
# ---------------------------------------------------------------------------

def _py_init_state_reparam(model: Reparameterization,
                            nwalkers: int, seed: int, start=None):
    """Create a C++ SamplerState for a Reparameterization model."""
    ndim   = model.n_params
    bounds = model.optimizer_bounds
    rng    = np.random.default_rng(seed)

    pos = np.empty((nwalkers, ndim))
    if start is None:
        for j, (lo, hi) in enumerate(bounds):
            pos[:, j] = rng.uniform(lo, hi, nwalkers)
    else:
        center = np.asarray(
            getattr(start, "position", start), dtype=float)
        for j, (lo, hi) in enumerate(bounds):
            span  = hi - lo
            sigma = min(1e-2 * span,
                        0.5 * min(center[j] - lo, hi - center[j]) + 1e-12)
            v = center[j] + rng.standard_normal(nwalkers) * sigma
            pos[:, j] = np.clip(v, lo, hi)

    lp = np.array([model.log_prob(pos[w].tolist()) for w in range(nwalkers)])
    return _sample._make_state(nwalkers, ndim, int(seed), pos, lp)


def _py_collect_reparam(model: Reparameterization, state):
    """Build a Chain from a SamplerState for a Reparameterization model."""
    chain_arr = state.get_chain                    # (n_step, nw, ndim) C++ view
    lp_arr    = state.get_log_prob                 # (n_step * nw,)  C++ view
    return _sample._chain_from_arrays(
        chain_arr.reshape(-1, state.ndim),
        lp_arr,
        state.nwalkers,
        param_names=model.param_names,
        transforms=model._sample_transforms,
        acceptance=state.acceptance_fraction,
    )


# ---------------------------------------------------------------------------
# SamplerOptions
# ---------------------------------------------------------------------------

class SamplerOptions:
    """Configuration for run_sampler().

    Parameters
    ----------
    h5_path : str, optional
        If set, chain is saved to this HDF5 file in real time (every
        ``log_every`` steps). The file is readable via load_chain().
    log_path : str, optional
        Write log messages here instead of stdout. Pass an empty string
        to suppress all output.
    log_every : int
        Log + h5 flush interval (steps). Default 100.
    auto_stop : bool
        Stop early when the chain is converged (nsteps > tau_factor * tau).
    tau_factor : float
        Convergence criterion: stop when nsteps > tau_factor * max(tau).
        Default 50 (emcee standard).
    check_every : int
        How often to check convergence (steps). Defaults to log_every.
    nwalkers : int
        Number of ensemble walkers. Default 64.
    seed : int
        RNG seed. Default 0.
    """

    def __init__(
        self,
        h5_path: str = None,
        log_path: str = None,
        log_every: int = 100,
        auto_stop: bool = True,
        tau_factor: float = 50.0,
        check_every: int = None,
        nwalkers: int = 64,
        seed: int = 0,
    ):
        self.h5_path    = h5_path
        self.log_path   = log_path
        self.log_every  = log_every
        self.auto_stop  = auto_stop
        self.tau_factor = tau_factor
        self.check_every = check_every if check_every is not None else log_every
        self.nwalkers   = nwalkers
        self.seed       = seed


# ---------------------------------------------------------------------------
# run_sampler
# ---------------------------------------------------------------------------

def run_sampler(
    model,
    nsteps: int = 1000,
    burnin: int = 0,
    start=None,
    hessian_init: bool = False,
    options: SamplerOptions = None,
):
    """Run ensemble sampler and return a Chain.

    Parameters
    ----------
    model : bayes.Model or Reparameterization
    nsteps : int
        Maximum number of production steps (post-burnin).
    burnin : int
        Number of burn-in steps (discarded from the chain).
    start : optimize.Result or list-of-lists, optional
        Starting position(s). Random within prior bounds if not given.
    hessian_init : bool
        Use Hessian-based Laplace init (requires ``start``).
        Ignored for Reparameterization models.
    options : SamplerOptions, optional
        All optional settings (h5, logging, convergence, etc.).

    Returns
    -------
    sample.Chain
    """
    if options is None:
        options = SamplerOptions()

    if options.h5_path and not _HAS_H5PY:
        raise ImportError(
            "h5py is required for h5_path saving. "
            "Install it with:  pip install h5py"
        )

    is_reparam  = isinstance(model, Reparameterization)
    has_py_extra = (not is_reparam
                    and hasattr(model, 'has_py_extras')
                    and model.has_py_extras())

    if not is_reparam and hasattr(model, 'validate'):
        model.validate()

    sampler = _sample.EnsembleSampler(
        nwalkers=options.nwalkers, seed=options.seed
    )

    # ---- initialise state ----
    if is_reparam:
        state = _py_init_state_reparam(model, options.nwalkers, options.seed, start)
        def _collect():
            return _py_collect_reparam(model, state)
    elif has_py_extra:
        # Python log_prob callback path: model.log_prob_python called per walker
        _log_prob_fn = model.log_prob_python
        state = _py_init_state_extended(model, options.nwalkers, options.seed,
                                        start, _log_prob_fn)
        def _collect():
            return _py_collect_extended(model, state)
    elif start is not None:
        state = sampler.init_state(model, start, hessian_init)
        def _collect():
            return sampler.collect(model, state)
    else:
        state = sampler.init_state(model)
        def _collect():
            return sampler.collect(model, state)

    ndim        = state.ndim
    nw          = state.nwalkers
    param_names = model.param_names

    # ---- logging ----
    _log_buf = []
    _suppress = options.log_path == ""

    def _log(msg: str):
        if _suppress:
            return
        if options.log_path:
            _log_buf.append(msg)
            if len(_log_buf) >= 20:
                _flush_log()
        else:
            print(msg, flush=True)

    def _flush_log():
        if _log_buf and options.log_path:
            with open(options.log_path, "a") as f:
                f.write("\n".join(_log_buf) + "\n")
            _log_buf.clear()

    # ---- h5 setup ----
    h5file = None
    h5_saved = 0

    if options.h5_path:
        h5file = _h5py.File(options.h5_path, "w")
        chunk_s = min(options.log_every, 200)
        h5file.create_dataset(
            "chain",
            shape=(0, nw, ndim),
            maxshape=(None, nw, ndim),
            dtype="f8",
            chunks=(chunk_s, nw, ndim),
        )
        h5file.create_dataset(
            "log_prob",
            shape=(0, nw),
            maxshape=(None, nw),
            dtype="f8",
            chunks=(chunk_s, nw),
        )
        h5file.attrs["param_names"] = param_names
        h5file.attrs["nwalkers"]    = nw
        h5file.attrs["ndim"]        = ndim
        h5file.attrs["burnin"]      = burnin

    def _flush_h5():
        nonlocal h5_saved
        chain_arr = state.get_chain
        lp_arr    = state.get_log_prob.reshape(-1, nw)
        n_new = len(chain_arr) - h5_saved
        if n_new <= 0:
            return
        h5file["chain"].resize(h5_saved + n_new, axis=0)
        h5file["log_prob"].resize(h5_saved + n_new, axis=0)
        h5file["chain"][h5_saved:]    = chain_arr[h5_saved:]
        h5file["log_prob"][h5_saved:] = lp_arr[h5_saved:]
        h5file.flush()
        h5_saved += n_new

    # ---- burnin ----
    t0 = time.time()
    _log(
        f"lcbinint sampler | {nw} walkers | {ndim} params | "
        f"burnin={burnin} | nsteps={nsteps}"
    )

    for i in range(1, burnin + 1):
        sampler.step(model, state)
        if options.log_every and i % options.log_every == 0:
            _log(
                f"  [burnin {i:>{len(str(burnin))}}/{burnin}]"
                f"  accept={state.acceptance_fraction:.3f}"
            )

    state.reset_history()
    _log(f"  Burnin done. Starting production run.")

    # ---- production loop ----
    max_tau = None
    converged_at = None
    i = 0

    while i < nsteps:
        sampler.step(model, state)
        i += 1

        is_check = options.log_every and i % options.log_every == 0

        if is_check:
            elapsed = time.time() - t0
            lp_med  = float(np.median(state.log_prob))
            _log(
                f"  [step {i:>{len(str(nsteps))}}/{nsteps}]"
                f"  accept={state.acceptance_fraction:.3f}"
                f"  log_prob={lp_med:.2f}"
                f"  elapsed={elapsed:.1f}s"
            )

        if h5file and is_check:
            _flush_h5()

        if (
            options.auto_stop
            and options.check_every
            and i % options.check_every == 0
            and i >= 50
        ):
            try:
                tmp_chain = _collect()
                taus = tmp_chain.tau()
                finite = [t for t in taus if not np.isnan(t) and t > 0]
                if finite:
                    max_tau = max(finite)
                    if i > options.tau_factor * max_tau:
                        converged_at = i
                        _log(
                            f"  [step {i}] Converged: "
                            f"nsteps={i} > {options.tau_factor} × tau_max={max_tau:.1f}. Stopping."
                        )
                        break
            except Exception:
                pass

    # ---- final h5 flush ----
    if h5file:
        _flush_h5()
        h5file.attrs["acceptance"]   = state.acceptance_fraction
        h5file.attrs["converged_at"] = converged_at if converged_at else -1
        h5file.close()

    elapsed = time.time() - t0
    _log(
        f"  Done. {i} production steps | "
        f"accept={state.acceptance_fraction:.3f} | "
        f"elapsed={elapsed:.1f}s"
    )
    _flush_log()

    return _collect()


# ---------------------------------------------------------------------------
# load_chain
# ---------------------------------------------------------------------------

def load_chain(h5_path: str):
    """Load a Chain from an HDF5 file saved by run_sampler().

    Returns a sample.Chain with full tau() / ess() / summary() support.
    """
    if not _HAS_H5PY:
        raise ImportError("h5py is required: pip install h5py")

    with _h5py.File(h5_path, "r") as f:
        flat = f["chain"][:].reshape(-1, int(f.attrs["ndim"]))
        lp   = f["log_prob"][:].reshape(-1)
        nw          = int(f.attrs["nwalkers"])
        param_names = list(f.attrs.get("param_names", []))
        acceptance  = float(f.attrs.get("acceptance", 0.0))

    return _sample._chain_from_arrays(
        flat, lp, nw,
        param_names=param_names,
        acceptance=acceptance,
    )
