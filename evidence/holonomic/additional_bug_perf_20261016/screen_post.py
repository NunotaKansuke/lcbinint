"""Current-HEAD regression and performance screening for algebraic_boundary."""
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.update(
    OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
)
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "post_patch"
OLD = ROOT / "evidence/holonomic/stress_20261015"
MATRIX = ROOT / "evidence/holonomic/lightcurve_matrix_20261015"

def digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def load_api():
    sys.path.insert(0, str(MATRIX))
    from benchmark import load
    np, lc, VB = load()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    return np, lc, VB

def make_config(identifier, params, limb, span, origin, refine=False, reuse=False,
                special_times=None, n=25):
    return {
        "id": identifier, "params": params, "limb": float(limb),
        "span_tau": float(span), "origin": origin, "refine": bool(refine),
        "reuse_probe": bool(reuse), "special_times_days": special_times or [],
        "n_uniform": int(n), "tol": 1e-3,
    }

def configurations(np):
    old = json.loads((OLD / "cases.json").read_text())
    configs = []
    for idx in (4, 25, 33, 49, 64, 89, 117):
        c = old[idx]
        extra = [-0.15, 0.15] if idx == 33 else []
        configs.append(make_config(
            f"legacy_{idx:03d}", c["params"], c["limb"], c["span"],
            f"unresolved_or_stress_case_{idx:03d}_from_20261015",
            refine=True, reuse=idx in (4, 33, 49), special_times=extra,
        ))
    near = dict(s=.6, q=1e-4, rho=.05, t0=0., tE=10., u0=-1e-5,
                alpha=.5)
    configs.append(make_config(
        "near_double_root", near, .5, .08, "previously_reported_close_pair",
        refine=True, reuse=True, special_times=[-.704, -.700, -.696],
    ))
    near_limb = json.loads((ROOT / "evidence/holonomic/case117_near_limb_20261016/case117_after.json").read_text())
    p = near_limb["parameter_setup"]
    t_contact = float(near_limb["time"])
    configs.append(make_config(
        "near_limb_contact_117", p, near_limb["limb_darkening"],
        min(1.5, max(.03, 2*p["rho"])), "latest_case117_exact_contact",
        refine=True, reuse=True,
        special_times=[t_contact-1e-5, t_contact, t_contact+1e-5],
    ))

    rng = np.random.default_rng(20261016)
    s_values = [.08, .15, .3, .6, .95, .999999, 1., 1.000001, 1.5, 3., 10.]
    q_values = [1e-8, 1e-7, 1e-6, 1e-4, 1e-2, .1, 1.]
    rho_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .05, .2, .5, 1.5]
    impact_modes = ("axis", "tiny", "tiny_negative", "limb_in", "limb_out", "random")
    limb_values = [0., .5, .9, .999999]
    alpha_values = [0., .5, 1.5707963267948966, 3.141592653589793]
    for i in range(24):
        s = float(s_values[(i*7 + int(rng.integers(0, len(s_values)))) % len(s_values)])
        q = float(q_values[(i*3 + int(rng.integers(0, len(q_values)))) % len(q_values)])
        rho = float(rho_values[(i*5 + int(rng.integers(0, len(rho_values)))) % len(rho_values)])
        mode = impact_modes[i % len(impact_modes)]
        if mode == "axis": u0 = 0.
        elif mode == "tiny": u0 = 1e-9
        elif mode == "tiny_negative": u0 = -1e-5
        elif mode == "limb_in": u0 = float(rho*rng.uniform(.95, .99999))
        elif mode == "limb_out": u0 = float(rho*rng.uniform(1.00001, 1.05))
        else: u0 = float(rng.uniform(-.5, .5))
        alpha = float(alpha_values[i % len(alpha_values)] if i % 3 == 0
                      else rng.uniform(-np.pi, np.pi))
        p = dict(s=s, q=q, rho=rho, t0=0., tE=10., u0=u0, alpha=alpha)
        configs.append(make_config(
            f"stratified_{i:02d}", p, limb_values[i % len(limb_values)],
            max(.03, min(1.5, 2*rho)), "deterministic_stratified_seed_20261016",
            refine=i < 8, reuse=i in (0, 3, 6),
        ))
    return configs

def make_curve(lc, p, limb, tol):
    return lc.LightCurve(
        options=lc.Options(coordinates="vbm", nbin="auto",
                           finite_source_tol=tol,
                           finite_source_reltol=tol,
                           finite_source_backend="algebraic_boundary"),
        limb_darkening=lc.LimbDarkening.linear(limb),
    )

def worker_screen(config):
    import numpy as np
    np, lc, _ = load_api()
    p = config["params"]
    span = config["span_tau"]
    tau = list(np.linspace(-span, span, config["n_uniform"]))
    if abs(p["u0"]) <= p["rho"]:
        contact = np.sqrt(max(0., p["rho"]**2-p["u0"]**2))
        tau.extend([-float(contact), float(contact)])
    days = [float(p["t0"] + p["tE"]*x) for x in tau]
    days.extend(float(x) for x in config["special_times_days"])
    times = np.unique(np.asarray(days, dtype=float))
    curve = make_curve(lc, p, config["limb"], config["tol"])

    start = time.perf_counter()
    info = curve.info(times, p)
    info_ms = 1e3*(time.perf_counter()-start)
    mu = np.asarray(info["mu"], dtype=float)
    start = time.perf_counter()
    grad = curve.holonomic_gradient(times, p, mu_atol=config["tol"],
                                     mu_rtol=config["tol"])
    gradient_ms = 1e3*(time.perf_counter()-start)
    grads = np.asarray(grad["grad_mu"], dtype=float)
    gvalid = np.asarray(grad["gradient_valid"], dtype=bool)
    gvalue = np.asarray(grad["value_converged"], dtype=bool)

    start = time.perf_counter()
    plain = np.asarray(curve(times, p), dtype=float)
    plain_ms = 1e3*(time.perf_counter()-start)
    start = time.perf_counter()
    reversed_values = np.asarray(curve(times[::-1], p), dtype=float)[::-1]
    reverse_ms = 1e3*(time.perf_counter()-start)
    finite_pair = np.isfinite(mu) & np.isfinite(reversed_values)
    order_abs = np.abs(mu[finite_pair]-reversed_values[finite_pair])
    order_rel = order_abs/np.maximum(np.abs(mu[finite_pair]), 1e-12)
    finite_plain = np.isfinite(mu) & np.isfinite(plain)
    plain_abs = np.abs(mu[finite_plain]-plain[finite_plain])

    reuse = None
    if config["reuse_probe"]:
        changed = dict(p)
        changed["alpha"] += 1e-4
        curve(times, changed)
        restored = np.asarray(curve(times, p), dtype=float)
        fresh = np.asarray(make_curve(lc, p, config["limb"], config["tol"])(times, p), dtype=float)
        mask = np.isfinite(restored) & np.isfinite(fresh)
        delta = np.abs(restored[mask]-fresh[mask])
        reuse = {
            "max_abs_same_instance_vs_fresh": float(delta.max()) if delta.size else None,
            "exact_array_match": bool(np.array_equal(restored, fresh, equal_nan=True)),
        }

    tight = None
    arrays = dict(times=times, mu=mu, plain=plain, reverse=reversed_values,
                  value_converged=np.asarray(info["value_converged"], dtype=bool),
                  fully_converged=np.asarray(info["fully_converged"], dtype=bool),
                  support_certified=np.asarray(info["support_certified"], dtype=bool),
                  integral=np.asarray(info["integrator_used"], dtype=bool),
                  nodes=np.asarray(info["node_evaluations"], dtype=np.int64),
                  epoch_seconds=np.asarray(info["seconds"], dtype=float),
                  gradient=grads, gradient_valid=gvalid,
                  gradient_value_converged=gvalue)
    if config["refine"]:
        tight_tol = 1e-5
        tight_curve = make_curve(lc, p, config["limb"], tight_tol)
        i2 = tight_curve.info(times, p)
        g2 = tight_curve.holonomic_gradient(times, p, mu_atol=tight_tol,
                                             mu_rtol=tight_tol)
        mu2 = np.asarray(i2["mu"], dtype=float)
        grad2 = np.asarray(g2["grad_mu"], dtype=float)
        grad_mask = np.isfinite(grads) & np.isfinite(grad2)
        grad_delta = np.abs(grads[grad_mask]-grad2[grad_mask])
        value_mask = np.isfinite(mu) & np.isfinite(mu2)
        value_delta = np.abs(mu[value_mask]-mu2[value_mask])
        value_scaled = value_delta/np.maximum(config["tol"], config["tol"]*np.abs(mu2[value_mask]))
        tight = {
            "tol": tight_tol,
            "finite_values": int(np.isfinite(mu2).sum()),
            "value_converged": int(np.asarray(i2["value_converged"], dtype=bool).sum()),
            "fully_converged": int(np.asarray(i2["fully_converged"], dtype=bool).sum()),
            "finite_gradient_components": int(np.isfinite(grad2).sum()),
            "valid_gradient_components": int(np.asarray(g2["gradient_valid"], dtype=bool).sum()),
            "max_abs_value_change": float(value_delta.max()) if value_delta.size else None,
            "max_scaled_value_change_in_tol_1e3": float(value_scaled.max()) if value_scaled.size else None,
            "max_abs_gradient_change": float(grad_delta.max()) if grad_delta.size else None,
            "gradient_change_components_over_0p32_percent": int(np.sum(
                grad_delta > .0032*np.maximum(np.abs(grad2[grad_mask]), 1e-8))) if grad_delta.size else 0,
        }
        arrays.update(mu_tol1e5=mu2, gradient_tol1e5=grad2,
                      gradient_valid_tol1e5=np.asarray(g2["gradient_valid"], dtype=bool),
                      value_converged_tol1e5=np.asarray(i2["value_converged"], dtype=bool))

    methods = Counter(str(x) for x in info["finite_source_method_name"])
    stops = Counter(str(x) for x in info["stop"])
    integral = np.asarray(info["integrator_used"], dtype=bool)
    ep_seconds = np.asarray(info["seconds"], dtype=float)
    np.savez_compressed(OUT/(config["id"]+".npz"), **arrays)
    result = {
        **config, "status": "ok", "n_epochs": int(len(times)),
        "info_ms": info_ms, "plain_call_ms": plain_ms,
        "reverse_call_ms": reverse_ms, "gradient_ms": gradient_ms,
        "finite_values": int(np.isfinite(mu).sum()),
        "value_converged": int(np.asarray(info["value_converged"], dtype=bool).sum()),
        "fully_converged": int(np.asarray(info["fully_converged"], dtype=bool).sum()),
        "integral_epochs": int(integral.sum()),
        "integral_support_certified": int((integral & np.asarray(info["support_certified"], dtype=bool)).sum()),
        "finite_gradient_components": int(np.isfinite(grads).sum()),
        "gradient_components": int(grads.size),
        "valid_gradient_components": int(gvalid.sum()),
        "invalid_gradient_epochs": [float(times[i]) for i in range(len(times)) if not bool(np.all(gvalid[i]))],
        "nan_gradient_components": int((~np.isfinite(grads)).sum()),
        "plain_info_exact": bool(np.array_equal(mu, plain, equal_nan=True)),
        "plain_info_max_abs": float(plain_abs.max()) if plain_abs.size else None,
        "max_order_abs_difference": float(order_abs.max()) if order_abs.size else None,
        "max_order_relative_difference": float(order_rel.max()) if order_rel.size else None,
        "nodes_total": int(np.sum(info["node_evaluations"])),
        "nodes_max_epoch": int(np.max(info["node_evaluations"])) if len(times) else 0,
        "epoch_ms_median": float(np.median(ep_seconds)*1e3) if ep_seconds.size else None,
        "epoch_ms_p95": float(np.percentile(ep_seconds, 95)*1e3) if ep_seconds.size else None,
        "integral_epoch_ms_p95": float(np.percentile(ep_seconds[integral], 95)*1e3) if integral.any() else 0.,
        "routes": dict(methods), "stops": dict(stops),
        "parameter_reuse": reuse, "tolerance_sweep": tight,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "param_order": list(grad["param_order"]),
    }
    (OUT/(config["id"]+".json")).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("id", "n_epochs", "finite_values", "value_converged", "invalid_gradient_epochs", "max_order_relative_difference", "info_ms", "gradient_ms", "nodes_total")}, separators=(",", ":")), flush=True)

def performance_configs():
    return [
        dict(id="ordinary_point", params=dict(s=1.2,q=.01,rho=.003,t0=0.,tE=10.,u0=.3,alpha=0.), limb=.5, lo=-1., hi=1.),
        dict(id="wide_caustic", params=dict(s=1.2,q=.01,rho=.003,t0=0.,tE=10.,u0=.03,alpha=-.08), limb=0., lo=-.2, hi=.8),
        dict(id="resonant_uniform", params=dict(s=.95,q=.01,rho=.005,t0=0.,tE=10.,u0=.03,alpha=.5), limb=0., lo=-.15, hi=.15),
        dict(id="broad_source_uniform", params=dict(s=.6,q=1e-4,rho=.05,t0=0.,tE=10.,u0=-1e-5,alpha=.5), limb=0., lo=-.08, hi=.08),
        dict(id="large_source_uniform", params=dict(s=.6,q=.001,rho=.5,t0=0.,tE=10.,u0=-1e-5,alpha=.5), limb=0., lo=-.08, hi=.08),
        dict(id="large_source_ld09", params=dict(s=.6,q=.001,rho=.5,t0=0.,tE=10.,u0=-1e-5,alpha=.5), limb=.9, lo=-.08, hi=.08),
        dict(id="notebook_alpha180", params=dict(s=1.5,q=.01,rho=.001,t0=0.,tE=10.,u0=-.01,alpha=180.), limb=.5, lo=-.08, hi=.08),
    ]

def worker_performance(config):
    import numpy as np
    np, lc, VB = load_api()
    p = config["params"]
    times = p["t0"] + p["tE"]*np.linspace(config["lo"], config["hi"], 401)
    tol = 1e-3
    curve = make_curve(lc, p, config["limb"], tol)
    vbm = VB.VBMicrolensing()
    vbm.Tol = vbm.RelTol = tol
    vbm.a1 = config["limb"]
    vbm.a2 = 0.
    vbm_params = [np.log(p["s"]), np.log(p["q"]), p["u0"], p["alpha"],
                  np.log(p["rho"]), np.log(p["tE"]), p["t0"]]
    time_list = times.tolist()
    def info_call(): return curve.info(times, p)
    def plain_call(): return np.asarray(curve(times, p), dtype=float)
    def vbm_call(): return np.asarray(vbm.BinaryLightCurve(vbm_params, time_list)[0], dtype=float)

    started = time.perf_counter(); info = info_call(); info_cold_ms = 1e3*(time.perf_counter()-started)
    started = time.perf_counter(); plain = plain_call(); plain_cold_ms = 1e3*(time.perf_counter()-started)
    lc_warm=[]; vb_warm=[]
    for repeat in range(5):
        calls = [("lc", plain_call), ("vbm", vbm_call)] if repeat%2==0 else [("vbm", vbm_call), ("lc", plain_call)]
        for name, fn in calls:
            start = time.perf_counter(); value = fn(); elapsed = 1e3*(time.perf_counter()-start)
            if name == "lc": plain = value; lc_warm.append(elapsed)
            else: vbm_values = value; vb_warm.append(elapsed)
    info_mu = np.asarray(info["mu"], dtype=float)
    reverse_plain = np.asarray(curve(times[::-1], p), dtype=float)[::-1]
    finite = np.isfinite(plain) & np.isfinite(vbm_values) & (vbm_values > 0.)
    rel = np.full(len(times), np.nan); rel[finite] = np.abs(plain[finite]-vbm_values[finite])/np.abs(vbm_values[finite])
    scaled = np.full(len(times), np.nan); scaled[finite] = np.abs(plain[finite]-vbm_values[finite])/np.maximum(tol,tol*np.abs(vbm_values[finite]))
    diff = np.abs(info_mu-plain)
    integral = np.asarray(info["integrator_used"], dtype=bool)
    methods = Counter(str(x) for x in info["finite_source_method_name"])
    ep_seconds = np.asarray(info["seconds"], dtype=float)
    np.savez_compressed(OUT/(config["id"]+"_performance.npz"), times=times,
                        lc=plain, info=info_mu, vbm=vbm_values,
                        reverse=reverse_plain, nodes=np.asarray(info["node_evaluations"], dtype=np.int64),
                        epoch_seconds=ep_seconds,
                        value_converged=np.asarray(info["value_converged"], dtype=bool),
                        support_certified=np.asarray(info["support_certified"], dtype=bool))
    result = {
        **config, "tol": tol, "n_epochs": 401, "status": "ok",
        "metric": "constructor and parameter conversion excluded; one pinned CPU; five alternating warm calls; medians",
        "info_cold_ms": info_cold_ms, "plain_cold_ms": plain_cold_ms,
        "lc_warm_ms": lc_warm, "vbm_warm_ms": vb_warm,
        "lc_median_ms": float(np.median(lc_warm)), "vbm_median_ms": float(np.median(vb_warm)),
        "vbm_over_lc": float(np.median(vb_warm)/np.median(lc_warm)),
        "info_plain_exact": bool(np.array_equal(info_mu, plain, equal_nan=True)),
        "info_plain_max_abs": float(np.nanmax(diff)) if np.isfinite(diff).any() else None,
        "max_order_abs_difference": float(np.nanmax(np.abs(plain-reverse_plain))),
        "integral_epochs": int(integral.sum()),
        "value_converged": int(np.asarray(info["value_converged"], dtype=bool).sum()),
        "fully_converged": int(np.asarray(info["fully_converged"], dtype=bool).sum()),
        "support_certified_integral": int((integral & np.asarray(info["support_certified"], dtype=bool)).sum()),
        "nodes_total": int(np.sum(info["node_evaluations"])),
        "nodes_max_epoch": int(np.max(info["node_evaluations"])),
        "epoch_ms_median": float(np.median(ep_seconds)*1e3),
        "integral_epoch_ms_p95": float(np.percentile(ep_seconds[integral], 95)*1e3) if integral.any() else 0.,
        "routes": dict(methods), "max_relative_vbm_difference": float(np.nanmax(rel)) if np.isfinite(rel).any() else None,
        "max_scaled_vbm_difference": float(np.nanmax(scaled)) if np.isfinite(scaled).any() else None,
        "scaled_vbm_difference_over_one": int(np.sum(scaled > 1.)),
        "vbm_finite_values": int(np.isfinite(vbm_values).sum()),
        "lc_finite_values": int(np.isfinite(plain).sum()),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (OUT/(config["id"]+"_performance.json")).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("id", "lc_median_ms", "vbm_median_ms", "vbm_over_lc", "integral_epochs", "nodes_total", "max_relative_vbm_difference")}, separators=(",", ":")), flush=True)

def init_provenance(np, lc, VB):
    import _lcbinint_holonomic_fused as fused
    native = Path(lc._native.__file__).resolve()
    gradient = Path(fused.__file__).resolve()
    vbm_module = Path(VB.__file__).resolve()
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    prov = {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tracked_modified": subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines(),
        "python": sys.version, "platform": platform.platform(), "cpu_model": cpu,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": 1,
        "numpy": np.__version__, "vbmicrolensing_version": getattr(VB, "__version__", "unknown"),
        "native_module": str(native), "native_module_sha256": digest(native),
        "gradient_module": str(gradient), "gradient_module_sha256": digest(gradient),
        "vbm_module": str(vbm_module), "vbm_module_sha256": digest(vbm_module),
        "screen_script_sha256": digest(Path(__file__)),
        "limits": {"screen_worker_timeout_s": 20, "performance_worker_timeout_s": 45,
                   "sequential_workers": True, "threads_per_worker": 1},
        "notebook_dirty_preserved": True,
    }
    (OUT/"provenance.json").write_text(json.dumps(prov, indent=2)+"\n")
    return prov

def run_workers(mode):
    np, lc, VB = load_api()
    init_provenance(np, lc, VB)
    env = os.environ.copy()
    started = time.perf_counter()
    if mode in ("screen", "all"):
        cases = configurations(np)
        (OUT/"cases.json").write_text(json.dumps(cases, indent=2)+"\n")
        for config in cases:
            result = OUT/(config["id"]+".json")
            if result.exists(): continue
            try:
                proc = subprocess.run([sys.executable, str(Path(__file__)), "screen-worker", json.dumps(config)],
                                      cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
                if proc.returncode != 0:
                    raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
                print(proc.stdout.strip(), flush=True)
            except subprocess.TimeoutExpired:
                result.write_text(json.dumps({**config,"status":"timeout","timeout_seconds":20},indent=2)+"\n")
                print(config["id"], "TIMEOUT 20s", flush=True)
            except Exception as exc:
                result.write_text(json.dumps({**config,"status":"failed","error":str(exc)[-3000:]},indent=2)+"\n")
                print(config["id"], "FAILED", str(exc)[-300:], flush=True)
        (OUT/"screen_run.json").write_text(json.dumps({"elapsed_s":time.perf_counter()-started,"cases":len(cases),"timeout_seconds":20,"workers":"sequential, one pinned CPU"},indent=2)+"\n")
    if mode in ("performance", "all"):
        cases = performance_configs()
        (OUT/"performance_cases.json").write_text(json.dumps(cases,indent=2)+"\n")
        for config in cases:
            result = OUT/(config["id"]+"_performance.json")
            if result.exists(): continue
            try:
                proc = subprocess.run([sys.executable,str(Path(__file__)),"performance-worker",json.dumps(config)],
                                      cwd=ROOT,env=env,capture_output=True,text=True,timeout=45)
                if proc.returncode != 0:
                    raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
                print(proc.stdout.strip(),flush=True)
            except subprocess.TimeoutExpired:
                result.write_text(json.dumps({**config,"status":"timeout","timeout_seconds":45},indent=2)+"\n")
                print(config["id"],"TIMEOUT 45s",flush=True)
            except Exception as exc:
                result.write_text(json.dumps({**config,"status":"failed","error":str(exc)[-3000:]},indent=2)+"\n")
                print(config["id"],"FAILED",str(exc)[-300:],flush=True)
        (OUT/"performance_run.json").write_text(json.dumps({"elapsed_s":time.perf_counter()-started,"cases":len(cases),"timeout_seconds":45,"workers":"sequential, one pinned CPU"},indent=2)+"\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "screen-worker":
        worker_screen(json.loads(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "performance-worker":
        worker_performance(json.loads(sys.argv[2]))
    elif len(sys.argv) > 1:
        run_workers(sys.argv[1])
    else:
        run_workers("all")
