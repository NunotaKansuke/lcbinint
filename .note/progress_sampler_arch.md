# Sampler Architecture — 実装進捗

設計元: `.note/lcbinint_sampler_design.md`
ブランチ: `feature/sampler-arch`

---

## Phase 1: クラス大枠 (2026-06-30) ✅

### 追加したファイル

#### C++ ヘッダ / スタブ実装

| ファイル | 内容 |
|---|---|
| `src/lcbinint/lc/evaluator.hpp` | `lc::IEvaluator` 抽象インターフェース |
| `src/lcbinint/obs/light_curve_data.{hpp,cpp}` | `obs::LightCurveData` (C++ 配列所有) |
| `src/lcbinint/obs/event.{hpp,cpp}` | `obs::Event` (複数データセットのコンテナ) |
| `src/lcbinint/bayes/prior.{hpp,cpp}` | `Prior` 基底 + `Uniform`, `Normal`, `LogUniform` |
| `src/lcbinint/bayes/model.{hpp,cpp}` | `bayes::Model` (パラメータ定義・事前分布・尤度設定) |
| `src/lcbinint/optimize/result.hpp` | `optimize::Result` (header-only) |
| `src/lcbinint/optimize/differential_evolution.{hpp,cpp}` | `DifferentialEvolution` スタブ |
| `src/lcbinint/sample/chain.{hpp,cpp}` | `sample::Chain` |
| `src/lcbinint/sample/ensemble_sampler.{hpp,cpp}` | `EnsembleSampler` スタブ |

#### Python バインディング

| ファイル | サブモジュール |
|---|---|
| `python/bind_obs.{hpp,cpp}` | `lcbinint.obs` |
| `python/bind_bayes.{hpp,cpp}` | `lcbinint.bayes` |
| `python/bind_optimize.{hpp,cpp}` | `lcbinint.optimize` |
| `python/bind_sample.{hpp,cpp}` | `lcbinint.sample` |

### 設計上のポイント

- **後方互換:** 既存の flat `lcbinint.LightCurve` 等は変更なし。新サブモジュールを追加するだけ。
- **IEvaluator 抽象クラス:** `bayes::Model` が `lc::IEvaluator` を持つことで、magnification 計算の実装詳細から分離。既存の `PyLightCurveEvaluator` を後でこのインターフェースに適合させる。
- **C++17:** `std::span` は C++20 なので `const std::vector<double>&` に変更。
- **スタブのみ:** `DifferentialEvolution::minimize`, `EnsembleSampler::run`, `Model::log_likelihood`, `Model::chi2` は `throw std::runtime_error("not yet implemented")`。

### ビルド確認

```
PYTHONPATH=build_new python3 -c "import lcbinint; lcbinint.obs.LightCurveData(...); ..."
```

全サブモジュール (`obs`, `bayes`, `optimize`, `sample`) が `import` 可能で Python から呼び出せることを確認。

---

## Phase 2b: IEvaluator 廃止・Parameters ラッパー廃止 (2026-06-30) ✅

### 設計変更

#### Parameters ラッパークラスを廃止
- `lc::Parameters` という中間クラスを削除
- `lcbi_params` 自体を `bind_lc.cpp` で `lc.Parameters` として直接バインド
- setter/getter のオーバーヘッドなし。`def_readwrite` で構造体フィールドに直書き

#### IEvaluator 抽象クラスを廃止
- `lc/evaluator.hpp` と `IEvaluator` を削除
- `lc::LightCurve` はスタンドアロンの Python 向けラッパーとして残す（継承なし）
- `bayes::Model` は `lcbi_options` だけを受け取り、ホットパスで `lcbi_magnification_array` を直接呼ぶ

#### bayes::Model の新コンストラクタ
```cpp
// Before
Model(shared_ptr<IEvaluator>, shared_ptr<Event>)

// After
Model(lcbi_options, shared_ptr<Event>)
Model(lcbi_options, shared_ptr<LightCurveData>)  // 単一データセット便利版
```

#### Python API
```python
model = lcbinint.bayes.Model(opts, event)   # opts は lc.Options (= lcbi_options)
model = lcbinint.bayes.Model(opts, data)    # data は obs.LightCurveData
```

---

## Phase 2: lc サブモジュール完成・旧 API 削除 (2026-06-30) ✅

### 追加したファイル

| ファイル | 内容 |
|---|---|
| `src/lcbinint/lc/parameters.{hpp,cpp}` | `lc::Parameters` — `lcbi_params` の名前付きラッパー |
| `src/lcbinint/lc/light_curve.{hpp,cpp}` | `lc::LightCurve : IEvaluator` — `lcbi_magnification_array` を呼ぶだけのクリーンな実装 |
| `python/bind_lc.{hpp,cpp}` | `lcbinint.lc` サブモジュール |

### 削除したもの

- `python/lcbinint_pybind.cpp` の全 3926 行 → 20 行の thin shell に置き換え
  - 旧 `PyLightCurveEvaluator`、`PyBinaryParams`、`PyOptions`、`PyLimbDarkening`、`PyEventCoordinates`、文字列ディスパッチ、dict パース — 全廃

### 新 API の概要

```python
import lcbinint.lc as lc
import numpy as np

opts   = lc.Options(source_bins=12, vbm_compatible=True)
ld     = lc.LimbDarkening.linear(0.5)
lc_obj = lc.LightCurve(options=opts, limb_darkening=ld)

params = lc.Parameters(t0=9000.0, tE=20.0, u0=0.3, alpha=0.5, s=1.5, q=1e-3, rho=1e-2)
times  = np.linspace(8990.0, 9010.0, 1000)
A      = lc_obj(times, params)   # np.ndarray, shape=(1000,)
```

Triple lens: `params.q2`, `params.sep2`, `params.ang` を設定するだけ。
Parallax: `params.piEN`, `params.piEE`, `params.ra`, `params.dec` を設定 + `Options(parallax_mode=1)`.
Xallarap: `params.xi_1`, `params.xi_2`, ... + `Options(xallarap_param_type=lc.XallarapParamType.ANGULAR_VELOCITY)`.

### 設計上のポイント

- `lc::LightCurve` は `IEvaluator` を実装 → `bayes::Model` がそのまま使える
- `lc::LightCurve` は `shared_ptr` で Python に露出 → `bayes::Model` に渡せる
- 旧 API の複雑な文字列ディスパッチは不要 — `lcbi_magnification_array` がパラメータを見て全て処理する

---

## 次のステップ

### Phase 3: パラメータ名→`lcbi_params` マッピング ← 次

- `Model::theta_to_params()` の中身: `"t0"`, `"tE"`, `"u0"`, `"sep"`, `"q"`, ... → `lcbi_params` フィールドへのマッピングテーブル
- サンプリング空間 (theta) と物理空間 (lcbi_params) の変換 (log変換など)

### Phase 4: 線形フラックスソルバー + 尤度

- `FluxSolver`: 各 `LightCurveData` に対して `Fs, Fb` を線形最小二乗で解く (C++)
- `Model::log_likelihood` の実装 (Gaussian chi2)
- ワークスペースの再利用 (評価ごとに allocate しない)

### Phase 5: `DifferentialEvolution` 実装

- GSL `gsl_multimin` か手書き DE
- `bayes::Model::chi2(theta)` を objective に

### Phase 6: `EnsembleSampler` 実装

- affine-invariant ensemble sampler (emcee スタイル)
- `bayes::Model::log_prob(theta)` のみを呼ぶ
