"""High-level sampler: run_sampler, SamplerOptions, load_chain."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import sample as _sample

try:
    import h5py as _h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


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

    sampler = _sample.EnsembleSampler(
        nwalkers=options.nwalkers, seed=options.seed
    )

    # ---- initialise state ----
    if start is not None:
        state = sampler.init_state(model, start, hessian_init)
    else:
        state = sampler.init_state(model)

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
    h5_saved = 0  # production steps already written to h5

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
        chain_arr = state.get_chain          # (n_hist, nw, ndim) zero-copy
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

    # Reset history so production chain starts clean
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

        # h5 incremental save
        if h5file and is_check:
            _flush_h5()

        # convergence check
        if (
            options.auto_stop
            and options.check_every
            and i % options.check_every == 0
            and i >= 50
        ):
            try:
                tmp_chain = sampler.collect(model, state)
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

    return sampler.collect(model, state)


def load_chain(h5_path: str):
    """Load a Chain from an HDF5 file saved by run_sampler().

    Returns a sample.Chain with full tau() / ess() / summary() support.
    """
    if not _HAS_H5PY:
        raise ImportError("h5py is required: pip install h5py")

    with _h5py.File(h5_path, "r") as f:
        flat = f["chain"][:].reshape(-1, int(f.attrs["ndim"]))   # (nsteps*nw, ndim)
        lp   = f["log_prob"][:].reshape(-1)                       # (nsteps*nw,)
        nw          = int(f.attrs["nwalkers"])
        param_names = list(f.attrs.get("param_names", []))
        acceptance  = float(f.attrs.get("acceptance", 0.0))

    return _sample._chain_from_arrays(
        flat, lp, nw,
        param_names=param_names,
        acceptance=acceptance,
    )
