from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[3];DATA=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
md('''# LightCurve / VBMicrolensing 性能比較

10軌道 × 一様・線形LD係数0.5・0.9 × atol=reltol=1e-3・1e-4、各401点。
主比較は通常の `curve.info(times, params)` と `vbm.BinaryLightCurve(...)`。1 CPU、初回呼び出し後に順序を交互にした3反復の中央値です。

**VBMとの差は誤差の保証ではありません。** 大きい光源のLDでは、独立した一様円盤の半径積分がlcbinint側を支持する反例がありました。
値の収束と像領域サポートの証明は別です。微分の有限性も微分精度の保証ではありません。
''')
code('''from pathlib import Path
import os, json, csv, sys
root = next((p for p in [Path.cwd(), *Path.cwd().parents,
    Path("/rogue1_8/nunota/lcbinint/experiments/algebraic_boundary")]
    if (p / "evidence/holonomic/lightcurve_matrix_20261015/cases.json").is_file()), None)
if root is None:
    raise FileNotFoundError("Open this notebook from the algebraic_boundary checkout")
data = root / "evidence/holonomic/lightcurve_matrix_20261015"
cache = root / "build-algebraic-boundary/plot-cache"
cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
results = pd.read_csv(data / "results.csv")
configs = {c["id"]: c for c in json.loads((data / "cases.json").read_text())}
''')
code('''columns = ["case", "limb", "tol", "lc_ms", "vbm_ms", "speed_vbm_over_lc",
    "relative_difference_max", "value_converged", "integral_epochs",
    "integral_support_certified"]
display(results[columns])
# speed_vbm_over_lc > 1: lcbinint is faster. Gradients are excluded from these timings.
''')
md('''## 曲線を選ぶ

`case_id` を表の `case + _ld..._tol...` に変更してください。
例：`large_source_ld0_tol0.001`、`broad_source_ld0.9_tol0.001`。
`notebook_alpha180_rad` は元ノートブックの数値180を**ラジアン**としてそのまま使ったケースです。
''')
code('''case_id = "wide_caustic_ld0.5_tol0.0001"
config = configs[case_id]
record = json.loads((data / f"{case_id}.json").read_text())
curve_data = np.load(data / f"{case_id}.npz")
print(config)
print("lcbinint / VBM [ms]:", record["lc_ms"], record["vbm_ms"])
print("routes:", record["routes"])
print("value converged:", record["value_converged"], "/ 401")
print("integral support certified:", record["integral_support_certified"], "/", record["integral_epochs"])
fig, axes = plt.subplots(2, 1, sharex=True, constrained_layout=True)
axes[0].plot(curve_data["times"], curve_data["lc"], label="lcbinint")
axes[0].plot(curve_data["times"], curve_data["vbm"], "--", label="VBMicrolensing")
axes[0].set_ylabel("magnification")
axes[0].legend()
axes[1].semilogy(curve_data["times"], np.maximum(curve_data["relative_difference"], 1e-16))
axes[1].set_ylabel("|LC - VBM| / |VBM|")
axes[1].set_xlabel("time from t0 (days)")
if globals().get("EXPORT_PREVIEW", False):
    fig.savefig(data / "preview.png")
plt.show()
''')
md('''## 選んだ設定を普通の呼び出しで再計算

`RUN_AGAIN=True` にすると、このチェックアウトのビルドを明示して同じ公開APIを呼びます。
全60設定を再計測するコマンドや独立参照の式は同じevidenceディレクトリのREADMEにあります。
''')
code('''RUN_AGAIN = False
if RUN_AGAIN:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(data))
    from benchmark import load
    _, lcbinint, VBMicrolensing = load()
    import time
    params = config["params"]
    atol = rtol = config["tol"]
    times = params["t0"] + params["tE"] * np.linspace(config["tau_min"], config["tau_max"], 401)
    curve = lcbinint.LightCurve(
        options=lcbinint.Options(coordinates="vbm", nbin="auto",
            finite_source_tol=atol, finite_source_reltol=rtol,
            finite_source_backend="algebraic_boundary"),
        limb_darkening=lcbinint.LimbDarkening.linear(config["limb"]),
    )
    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol, vbm.RelTol = atol, rtol
    vbm.a1, vbm.a2 = config["limb"], 0.0
    vbm_params = [np.log(params["s"]), np.log(params["q"]), params["u0"],
        params["alpha"], np.log(params["rho"]), np.log(params["tE"]), params["t0"]]
    start = time.perf_counter()
    info = curve.info(times, params)
    lc_ms = 1e3 * (time.perf_counter() - start)
    start = time.perf_counter()
    vbm_values = np.asarray(vbm.BinaryLightCurve(vbm_params, times.tolist())[0])
    vbm_ms = 1e3 * (time.perf_counter() - start)
    print("single calls [ms]:", lc_ms, vbm_ms)
    print("max relative difference:", np.max(np.abs(info["mu"] - vbm_values) / np.abs(vbm_values)))
''')
md('''## 独立したLD参照

一様円盤の値を使い、部分積分で得られる

`A_LD = ((1-c)*A_uniform(rho) + c*integral_0^1 (1-z^2)*A_uniform(rho*sqrt(1-z^2)) dz)/(1-c/3)`

を32→64→128→256点で計算しました。VBM内蔵LD計算を参照として固定していません。
この参照の次数間差も、元の一様円盤ソルバーも、厳密な全体誤差証明ではありません。
''')
code('''display(pd.read_csv(data / "reference_summary.csv"))
''')
for i,c in enumerate(cells):c['id']=f'cell-{i:02d}'
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.10')),nbformat=4,nbformat_minor=5)
path=ROOT/'example/compare-vbm/lcbinint_vbm_lightcurve_performance.ipynb';path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
# Execute the read/plot cells directly to verify the artifact without creating a Jupyter runtime elsewhere.
os.environ['MPLBACKEND']='Agg';ns={'EXPORT_PREVIEW':True}
for c in cells:
 if c['cell_type']=='code':exec(compile(''.join(c['source']),str(path),'exec'),ns)
print(path)
