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

## 次のステップ

### Phase 2: `lc::IEvaluator` の実装接続

- 既存 `PyLightCurveEvaluator` に `IEvaluator` を継承させる (または adapter を作る)
- `bayes::Model` のコンストラクタが Python 側から `LightCurve` オブジェクトを受け取れるようにする

### Phase 3: パラメータ名→`lcbi_params` マッピング

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
