"""High-level sampler: run_sampler, SamplerOptions, load_chain."""
from __future__ import annotations

import time

import numpy as np

from . import sample as _sample

try:
    import h5py as _h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


# ---------------------------------------------------------------------------
# Helpers for Python-callback path in run_sampler
# ---------------------------------------------------------------------------

def _py_init_state_extended(model, nwalkers: int, seed: int, start, log_prob_fn):
    """Create a SamplerState for a Python callback sampling model."""
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
        if center.shape[0] != ndim:
            raise ValueError(
                f"start has {center.shape[0]} parameters, but sampling space has {ndim}"
            )
        for j, (lo, hi) in enumerate(bounds):
            span  = hi - lo
            sigma = min(1e-2 * span,
                        0.5 * min(center[j] - lo, hi - center[j]) + 1e-12)
            v = center[j] + rng.standard_normal(nwalkers) * sigma
            pos[:, j] = np.clip(v, lo, hi)

    lp = np.array([log_prob_fn(pos[w].tolist()) for w in range(nwalkers)])
    return _sample._make_state(nwalkers, ndim, int(seed), pos, lp)


def _py_collect_extended(model, state):
    """Build a Chain from a SamplerState for a Python callback model."""
    chain_arr = state.get_chain
    lp_arr    = state.get_log_prob
    flux_arr, dataset_names = _flux_array_from_samples(
        model, chain_arr.reshape(-1, state.ndim)
    )
    return _sample._chain_from_arrays(
        chain_arr.reshape(-1, state.ndim),
        lp_arr,
        state.nwalkers,
        param_names=model.param_names,
        transforms=getattr(model, "_sample_transforms", []),
        fluxes=flux_arr,
        dataset_names=dataset_names,
        acceptance=state.acceptance_fraction,
    )


def _flux_array_from_samples(model, samples):
    """Return flat flux array (ntot, ndatasets*2) and dataset names."""
    if not hasattr(model, "fluxes") or len(samples) == 0:
        return None, []
    rows = []
    dataset_names = None
    for theta in samples:
        flux_dict = model.fluxes(theta.tolist())
        if dataset_names is None:
            dataset_names = list(flux_dict.keys())
        row = []
        for name in dataset_names:
            row.extend([flux_dict[name]["Fs"], flux_dict[name]["Fb"]])
        rows.append(row)
    if dataset_names is None:
        return None, []
    return np.asarray(rows, dtype=float), dataset_names


def _state_flux_array(model, state, chain_arr):
    """Return flux history as (nsteps, nwalkers, n_fluxes), computing if needed."""
    state_fluxes = getattr(state, "get_fluxes", None)
    if state_fluxes is not None and len(state_fluxes) == len(chain_arr):
        return np.asarray(state_fluxes), None
    flat, dataset_names = _flux_array_from_samples(
        model, chain_arr.reshape(-1, state.ndim)
    )
    if flat is None:
        return None, dataset_names
    return flat.reshape(len(chain_arr), state.nwalkers, -1), dataset_names


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
    model : bayes.Model
    nsteps : int
        Maximum number of production steps (post-burnin).
    burnin : int
        Number of burn-in steps (discarded from the chain).
    start : optimize.Result or list-of-lists, optional
        Starting position(s). Random within prior bounds if not given.
    hessian_init : bool
        Use Hessian-based Laplace init (requires ``start``).
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

    has_reparam = (hasattr(model, 'has_reparams')
                   and model.has_reparams())
    has_py_extra = (hasattr(model, 'has_py_extras')
                    and model.has_py_extras())

    if hasattr(model, 'validate'):
        model.validate()

    sampler = _sample.EnsembleSampler(
        nwalkers=options.nwalkers, seed=options.seed
    )

    # ---- initialise state ----
    step_model = model
    if has_reparam or has_py_extra:
        # Python callback path. The adapter is not a C++ bayes.Model, which
        # keeps pybind overload resolution from taking the C++ fast path.
        step_model = (
            model._sampling_adapter()
            if hasattr(model, "_sampling_adapter") else model
        )
        state = _py_init_state_extended(
            step_model, options.nwalkers, options.seed, start, step_model.log_prob
        )
        def _collect():
            return _py_collect_extended(step_model, state)
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
    param_names = step_model.param_names
    transforms = getattr(step_model, "_sample_transforms", [])
    flux_dataset_names = []
    n_fluxes = int(getattr(state, "n_fluxes", 0))
    if n_fluxes > 0:
        flux_dataset_names = list(_collect().dataset_names)
    else:
        current_fluxes, flux_dataset_names = _flux_array_from_samples(
            step_model, np.asarray(state.pos)
        )
        n_fluxes = 0 if current_fluxes is None else current_fluxes.shape[1]

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
        if n_fluxes > 0:
            h5file.create_dataset(
                "fluxes",
                shape=(0, nw, n_fluxes),
                maxshape=(None, nw, n_fluxes),
                dtype="f8",
                chunks=(chunk_s, nw, n_fluxes),
            )
        h5file.attrs["param_names"] = param_names
        h5file.attrs["transforms"]   = transforms
        h5file.attrs["dataset_names"] = flux_dataset_names
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
        if "fluxes" in h5file:
            fl_arr, names = _state_flux_array(step_model, state, chain_arr)
            if fl_arr is not None:
                h5file["fluxes"].resize(h5_saved + n_new, axis=0)
                h5file["fluxes"][h5_saved:] = fl_arr[h5_saved:]
                if names:
                    h5file.attrs["dataset_names"] = names
        h5file.flush()
        h5_saved += n_new

    # ---- burnin ----
    t0 = time.time()
    _log(
        f"lcbinint sampler | {nw} walkers | {ndim} params | "
        f"burnin={burnin} | nsteps={nsteps}"
    )

    for i in range(1, burnin + 1):
        sampler.step(step_model, state)
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
        sampler.step(step_model, state)
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
        transforms  = list(f.attrs.get("transforms", []))
        dataset_names = list(f.attrs.get("dataset_names", []))
        fluxes = f["fluxes"][:].reshape(len(lp), -1) if "fluxes" in f else None
        acceptance  = float(f.attrs.get("acceptance", 0.0))

    return _sample._chain_from_arrays(
        flat, lp, nw,
        param_names=param_names,
        transforms=transforms,
        fluxes=fluxes,
        dataset_names=dataset_names,
        acceptance=acceptance,
    )
