# MOASARCからのD14明示ルート統合メモ

## 結論

現行integration系へD14を接続する最小のABIを、通常routerから分離した
明示ルートとして実装した。`lcbinint`の既存`auto`/algebraic/inverse-ray経路は
変更していない。MOASARC側では
`EvaluationPlan(refinement_policy_id="holonomic-d14-v1")`を指定した場合だけ
このABIを選ぶ。

この変更はD14をproduction defaultへ昇格したものではない。今回のABIは1回の
batch呼び出しで各rowをstatelessに評価するため、`PreparedEpochGeometry`の
epoch間warm reuseはまだ接続していない。速度・coverage・通常経路との同一case
比較を終えるまで、明示的な研究/比較ルートとして扱う。

## 位置づけ

確認した基準は次の通り。

| 対象 | checkout | 基準 |
| --- | --- | --- |
| 現行integration LCBININT | `eco-system/integration` | `4b9888f` |
| D14研究実装 | `dev/holonomic` | `71ef8bd`（local snapshot） |
| 現行MOASARC integration | `eco-system/integration` | `3885463` |

`dev/holonomic`と現行integrationは履歴が大きく分岐しているため、branch全体を
mergeせず、D14の`src/lcbinint/magnification/holonomic/`をintegration側へ
移し、ABIとadapterだけを追加した。過去の`48daa49`をそのまま復活させる方式
ではない。

## ABIと座標

LCBININTに次を追加した。

```text
lcbinint_holonomic_batch_value_jacobian_f64_v1
```

入力はbatchの
`(source_x, source_y, separation)`とscalarの`(rho, q, limb_c, limb_d, n_r)`。
出力は、値、5成分のgeometry Jacobian、行ごとのsupport certificate。
Jacobianの順序は

```text
(source_x, source_y, rho, q, separation)
```

である。MOASARCのpublic geometry座標
`(x, y, rho, q, separation)`をそのまま渡し、古いalgebraic adapterの
`(-x, -y, 1/q)`変換はD14ルートには適用しない。

D14研究実装が現在扱うのはlinear limb darkeningなので、`limb_d != 0`は
Python planでもC++ FFIでもfail-closedにした。無視してlinear扱いすることはない。
入力非有限、`rho <= 0`、`q <= 0`、`separation <= 0`もsupport=falseとする。

MOASARCの`NativeM1Batch`は、D14の値・analytic 5Jac・supportを受け、
supportが落ちたrowを`VALUE_UNCERTIFIED`として返す。自動で普通のrouterへ
差し替えるfallbackは今回のexplicit planには入れていない。

## 検証

次の環境で確認した。

```bash
GSL_ROOT=/rogue1_8/nunota/local/gsl \
cmake -S . -B /tmp/lcbinint-d14-python-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLCBININT_BUILD_PYTHON=ON \
  -DLCBININT_BUILD_TESTS=OFF \
  -DLCBININT_ENABLE_OPENMP=OFF \
  -DLCBININT_ENABLE_JAX_FFI=ON
cmake --build /tmp/lcbinint-d14-python-build --target lcbinint_python -j2
```

editable installのredirectを除外して、LCBININTのD14 batch/planned testを実行した。

```bash
PYTHONPATH=/tmp/lcbinint-d14-python-build \
LD_LIBRARY_PATH=/rogue1_8/nunota/local/gsl/lib \
JAX_ENABLE_X64=1 python - <<'PY'
import sys
sys.meta_path[:] = [x for x in sys.meta_path
                    if type(x).__module__ != '_lcbinint_editable']
import pytest
raise SystemExit(pytest.main([
    '-q', 'tests/jax_ir/test_holonomic_batch.py',
    'tests/jax_ir/test_planned.py']))
PY
```

結果: **6 passed**。

MOASARC側は同じnative buildを使い、D14 plan、JAX JVP、quadratic LDのfail-closed、
既存LCBININT route selectionを確認した。

```bash
PYTHONPATH=/tmp/moasarc-d14-integration/python:/tmp/lcbinint-d14-python-build \
LD_LIBRARY_PATH=/rogue1_8/nunota/local/gsl/lib \
JAX_ENABLE_X64=1 python - <<'PY'
import sys
sys.meta_path[:] = [x for x in sys.meta_path
                    if type(x).__module__ != '_lcbinint_editable']
import pytest
raise SystemExit(pytest.main([
    '-q', 'tests/test_d14_route.py',
    'tests/test_lcbinint_route_selection.py']))
PY
```

結果: **6 passed**。

## 未接続のもの

- `PreparedEpochGeometry`を保持するMOASARC trajectory/warm ABI
- D14のcoverage/statusを通常algebraic routeと同一条件で比較するwhole-epoch benchmark
- explicit D14を`auto`やdefault planから選ぶproduction router
- D14のquadratic limb darkening
- MOASARC既存integration checkoutへの直接merge

したがって今回確認できたのは「明示planからD14の値/Jacobianを呼べること」と
「通常経路を壊していないこと」であり、production速度の勝利ではない。
