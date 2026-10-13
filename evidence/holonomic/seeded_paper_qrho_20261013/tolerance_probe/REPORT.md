# 低増光率でVBMicrolensingが速い理由と許容誤差スイープ

## 結果

VBMicrolensingの低増光率側の速さは計測ミスでも、今回の一様光源で多極近似のショートカットが効いているためでもありません。一様光源3,608入力をVBMicrolensing 5.5で再評価し、`BinaryMag2`の多極近似条件は0件、点光源値への完全な短絡も0件でした。

主因は固定費です。今回の経路ではソース中心像の求根と有限光源seed構築だけで中央値約40 µs、続く境界計画が約22–24 µsかかります。同一ケースごとの`seed+planning`中央値は一様で62.40 µs、線形で64.68 µsとなり、VBMicrolensing呼び出し全体の中央値46.39 µs、48.66 µsをすでに上回ります。許容誤差を緩めて積分を速くしても、この差は埋まりません。

## VBMの実経路

インストール済みVBMicrolensingは5.5です。計測した共有入力で保存済みVBM時間を再現し、低増光率の中央値は一様光源で保存値46.394 µsに対して新規計測46.043 µs、線形減光で48.663 µsに対して48.853 µsでした。元の時間表を生成したスクリプトは残っていませんが、現行の直接呼び出しは保存値を約1%以内で再現します。

一様光源は`BinaryMag2`、線形減光は`BinaryMagDark`を通ります。`BinaryMag2`には点光源値から戻る多極近似ゲートがありますが、今回の一様光源3,608点では一度も通っていません。したがって、低増光率で速い理由はこのゲートではありません。

代わりに、一様光源では半径方向のannulus適応が実質1区間で終わります。VBMの初期条件は`minannuli=1`で、初期annulus数も1です。`a1=0`なら`LDprofile`は中心からlimbまで一定の1となり、初期annulus誤差式にある`f(center)-f(limb)`が厳密にゼロです。そのため半径方向の追加細分化は発生しません。中心の`BinaryMag0`評価と、外縁の`BinaryMagSafe`評価は行われます。線形減光はannulusを適応細分化しますが、低増光率では滑らかなため、許容誤差を`1e-3`から`1e-2`に緩めると積分node中央値は53から29に減ります。

## 許容誤差スイープ

本ベンチと同じ入力変換、seed provider、1400-bin caustic branchesを使い、308入力を5回ずつ実行しました。各光源プロファイルで増光率帯ごとに層化抽出し、現行最大誤差の上位例と以前の5つの端点失敗配置も加えました（重複除外後、一様155件・線形153件）。`mu_atol=1e-8`、勾配計算なしです。caustic branchesの構築は計時から除外し、中心像求根と有限光源seed構築は計時に含めています。誤差は保存済みVBM `RelTol=1e-6`値との差です。

低増光サンプル（`A_ref<1.1`、一様62件・線形61件）の中央値です。各時間は入力ごとの5反復中央値を集計し、`current/VBM`は同一入力ごとの比の中央値です。

| 光源 | `mu_rtol` | VBM µs | seed µs | 計画 µs | 積分 µs | 全体 µs | current/VBM | 実誤差 p95 | 実誤差 最大 | 1e-3超過数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 一様 | 1e-3 | 46.39 | 40.19 | 23.63 | 32.53 | 104.30 | 2.35x | 1.84e-5 | 5.61e-4 | 0/62 |
| 一様 | 1e-2 | 46.39 | 40.19 | 23.14 | 25.13 | 95.12 | 2.10x | 3.67e-4 | 2.50e-3 | 2/62 |
| 一様 | 1e-1 | 46.39 | 40.19 | 22.23 | 24.92 | 92.37 | 2.10x | 5.59e-4 | 1.05e-2 | 3/62 |
| 線形 | 1e-3 | 48.66 | 40.16 | 23.12 | 50.92 | 128.30 | 2.41x | 4.71e-5 | 5.30e-4 | 0/61 |
| 線形 | 1e-2 | 48.66 | 40.16 | 22.29 | 33.20 | 103.00 | 2.00x | 6.21e-4 | 9.37e-4 | 0/61 |
| 線形 | 1e-1 | 48.66 | 40.16 | 22.15 | 28.53 | 102.22 | 1.96x | 9.27e-4 | 3.48e-3 | 2/61 |

同一入力での`mu_rtol=1e-3`比では、`1e-2`に緩めたとき全体時間の中央値は一様光源で約5%、線形減光で約16%短縮しました。`1e-1`でも約6%・17%短縮で、線形減光の積分は速くなりますが、seedと計画の費用が残るためVBMより約2倍遅いままです。

一方、精度には余裕があります。`mu_rtol=1e-3`の低増光サンプルでは、実誤差中央値は一様光源で`5.0e-7`、線形減光で`2.1e-6`です。許容誤差を`1e-2`にすると線形のサンプルは最大でも`9.37e-4`でしたが、一様では2/62点が`1e-3`を超えました。これは小さな経験サンプルの結果で、より緩い許容誤差を標準設定にできるという保証ではありません。

## 解釈と残る課題

「積分が細かすぎる」は線形減光には確かに当てはまり、`1e-2`への緩和で積分時間中央値が約35%減りました。しかし低増光率の一様光源では積分時間自体が約33 µsで、全体の律速はseedと計画です。ここを削らないとVBMには勝てません。したがって本命は許容誤差を緩めるだけでなく、低増光・滑らかなケースで全像seedの作成と半径境界計画を安くすることです。

このスイープで全入力の内部値収束フラグは立ちましたが、frontierの`coverage_certified`は従来どおりfalseです。ここで示す誤差は保存済み参照との経験比較であり、完全な支持領域保証ではありません。既定許容誤差や積分器の標準動作は変更していません。

## 再現

```bash
python3 evidence/holonomic/seeded_paper_qrho_20261013/tolerance_probe/prepare_tolerance_probe.py
g++ -std=c++17 -O3 -DNDEBUG -march=native -fext-numeric-literals -I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include evidence/holonomic/seeded_paper_qrho_20261013/tolerance_probe/bench_tolerance_probe.cpp build-seeded-events/seed_after_v26.o -Wl,--start-group build-standard-current/liblcbinint_magnification.a build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group -L/home/nunota/.miniconda3/envs/myenv/lib -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lquadmath -fopenmp -o build-seeded-events/bench_tolerance_probe
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$PWD/build-seeded-events" build-seeded-events/bench_tolerance_probe evidence/holonomic/seeded_paper_qrho_20261013/tolerance_probe/probe_input.tsv evidence/holonomic/seeded_paper_qrho_20261013/tolerance_probe/raw.tsv 5
python3 evidence/holonomic/seeded_paper_qrho_20261013/tolerance_probe/summarize_tolerance_probe.py
PYTHONDONTWRITEBYTECODE=1 python3 evidence/holonomic/seeded_paper_qrho_20261013/tolerance_probe/audit_vbm_path.py
```

VBM source fingerprint: SHA256 `378451e561e9ad13f71418cd927a3e27cbfc6abb429d55723c8338a3c5f2f73b` (`/home/nunota/.miniconda3/envs/myenv/lib/python3.10/site-packages/VBMicrolensing/lib/VBMicrolensingLibrary.cpp`). The VBM 5.5 `BinaryMag2` gate is at lines 1170–1189; the annulus initialization and stopping test are at lines 1230–1284 of that source. The benchmark wrapper's actual method dispatch is in `/rogue1_8/nunota/lcbinint/tests/diagnostics/recal2026/engines.py` lines 313–324.

### Artifacts

- `raw.tsv`: every selected input, tolerance, and repetition.
- `summary.tsv`: per-profile, magnification-bin, and tolerance summary.
- `vbm_uniform_shortcut_audit.tsv`: direct audit of the multipole gate over all 3,608 uniform inputs.
- `vbm_low_magnification_timing.tsv`: fresh direct VBM timings for the low-magnification sample, paired with saved timings.
- `selection.tsv`: stratified sample and forced-tail selection reasons.
