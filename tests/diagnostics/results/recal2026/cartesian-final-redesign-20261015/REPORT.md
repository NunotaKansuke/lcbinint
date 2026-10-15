# Cartesian誤差処理の修正と最適化 — 2026-10-15

## 採用結果

採用は **v7、初期16×16**。flood fill、保持行、方向別細分化を使用する。全体画像の楕円近似・差し引き積分・ケースID別の救済は追加していない。`1.09e-4`程度の軽微な超過を潰すための一律安全係数・要求精度変更も追加していない。

| 輝度 | 要求相対精度 | ケース別速度比中央値（変更前/採用版） | 精度を満たした共通ペア | 要求精度外の受理（全件、変更前→採用版） |
|---|---:|---:|---:|---:|
| Uniform | 1e-03 | 1.0788 | 116/128 | 81 → 12 |
| Uniform | 1e-04 | 1.0770 | 114/128 | 155 → 26 |
| Linear LD | 1e-03 | 1.0544 | 112/128 | 54 → 4 |
| Linear LD | 1e-04 | 1.0592 | 108/128 | 118 → 19 |

速度は256入力（Uniform/LD各128）の同じ入力を交互に実行し、各ケース3回の中央値を比較したもの。初期格子16×16、CPU0・1スレッド。**seedキャッシュmiss時の構築を含む**。中心像・causticの準備、LDテーブルの初期化は計時外。速いseed実装への既存接続は維持した。倍率はケース別比の中央値で、時間中央値同士の比ではない。失敗を速いケースとして混ぜていない。

変更前に正しく受理できたケースからの新しい失敗は、この計時標本ではUniform1e-3に1、LD1e-4に1（いずれも未達）。ほかの2群は0。各ケースの時間・精度・例外・除外理由は [計時raw/集計](selected-v7-paired256/summary.json) に残した。全VBMの再計時や6パネル再描画は今回行っていない。既存の6パネルは今回のCartesian修正前の図である。

## 実装で直した箇所

1. **境界帯だけの再計算が、未再計算の内部誤差を捨てていた。** 内部の符号付き差分と剰余を保持し、内部だけで予算超過なら境界帯だけの修復を試さない。
2. **旧Bennett外側重みと適応欠損補正を混在させていた。** 適応の欠損補正経路では旧外側重みを外す。横方向の実際の境界根補正は維持する。固定格子の旧経路は9比較すべて変更前と一致。
3. **行のrun数が同じでも、幅の急変で4階差分が相殺していた。** 2階差分の尺度に対し粗い4階差分では絶対剰余を残す。滑らかな部分の符号付き相殺は維持する。
4. **同じ4行から推定した端点位置が偏ることがあった。** レンズ写像の実際の境界 `F=0, Fx=0` を局所Newton法で確認する。解析微分、曲率・向き・内点の確認を使い、最大8反復。解けない場合は既存経路へ戻す。物理端点を固定して幅の二乗モデルを合わせ、独立の保持行で確認する。
5. **粗い格子で欠落画像を補修した際、同じ画像を二重加算することがあった。** 同一占有runは重複追加しない。合流時は共有する実際の占有格子点でownerを統合し、受理された合流だけを確定する。合流できなければ画像全体の不確実性を残す。
6. **処理の再利用。** Newtonの受理済み評価を次反復で再利用。物理端点が分かる場合の不要な位相fitを省く。通常の初期16で重複していた所有領域チェックを省く。横方向の境界根も既存残差から片側の括弧を作り、符号確認と最終補正は残す。

ソース3ファイルの変更は [開始時のdirty sourceに対する差分](cartesian-before-this-turn.patch)。既存のstaged変更を含むHEADとの差分とは区別した。

## 精度・内部確認

- 保存済みVBM RelTol1e-6参照と3件の独立裁定を使い、7,216入力×2精度の **14,432条件** を比較。要求精度外の受理は **408→61**。
- 変更前の誤受理408のうち **318は正確な受理に改善、33は未達として返す**。以前は正確だった入力から新たな厳密閾値超過が4条件ある。内訳は [全件比較](selected-v7-full16-comparison.json)。
- 全件の未達はUniform198/245、LD192/230（1e-3/1e-4）。評価予算の例外は変更前と同じ **312形状、624条件**。例外・未達を成功に数えていない。
- 独立参照108入力×2精度では、誤受理 **4→0**。216条件のうち114受理、72例外、30未達であり、216全成功ではない。[独立比較](selected-v7-heldout.json)
- 最終nativeで **132内部チェック** 通過。反復再現、traceの非干渉、低予算、失敗後の回復、誤差予算による受理、成分flux合計を確認。[契約](final-contracts.json)
- 全件保存結果のうち有限13,808条件で成分flux合計と返却倍率が一致、失敗0。[flux ledger](selected-v7-full-flux-ledger.json)
- 端点単体: 独立根との48条件比較、既知の物理端点20,000条件（適格19,802中19,797で受理、最大位相差1.47e-5）。幅モデルの合成10,000条件、占有領域の異なる縦横格子10,000条件も確認。原記録は `physical-tip-unit.json`, `constructed-tip-unit.txt`, `fixed-phase-unit.txt`, `retained-overlap-unit.txt`。
- **Polarは数値を維持。** 全14,432条件で倍率・誤差指標・新規評価数・受理状態が以前の採用版と完全一致。全件参照比較の誤受理0、189内部チェック通過。[不変性](polar-invariance-v7.json)

これは誤差保証の完成宣言ではない。残る61条件には要求1e-4に対する1.03e-3程度の超過もある。`certified=false`, `complete_error_model=false` を保持する。ケース1604の約1.09e-4も未修正。入力の切り替えルーターや難例用の救済分岐は増やしていない。

## 不採用にした高速案

横16・縦8の開始を試した。面積規格化の不具合を修正し、粗い行数にも既存チェックを適用した後のv12は、全件の誤受理74、例外164形状。seed込み256入力の倍率中央値はUniform1.150/1.049、LD1.061/1.019。**LD1e-4では採用した正方格子版1.059を下回る**。既存4行から次数を上げる境界モデルも、採用を正当化する改善は得られなかった。両案を外し、初期16×16に戻した。

少数パイロットのLD1.57/1.28倍を採用版の性能として報告しない。失敗した面積規格化版、追加チェックにより失われた速度、契約テスト自身の不備、その他の不採用案を [EXPERIMENTS.md](EXPERIMENTS.md) と各出力に残した。

## 識別と再現

- branch `feature/boundary-error-and-discriminant-support`、HEAD `ee4d1c4e`。未コミット・未push。
- 最終native SHA256: `cbb4f10d7725786de0656e57bcccc64c1e041ddd0e474d2a1b2a662c088fd278`。
- 最終再ビルドのnativeは全件検証済みv7と **byte単位で一致**。静的archiveはmtime欄だけが異なり、すべてのobjectが一致。最終archiveから作り直したrunnerも計時したv7とbyte単位で一致。この同一性に基づき完了済みの全件検証・計時を再利用した。[build同一性](final-build-parity.json)、[runner同一性](final-runner-parity.json)
- [final-identity.json](final-identity.json): source/native/input/reference/実験実行境界。実際のsourceは `final-source/`、基準native/sourceは `baseline/`。最終固定格子比較は `final-fixed-grid-comparison.json`。

プロジェクトrootから（新しい出力先を指定する）:

```bash
TMPDIR="$PWD/build-adaptive-production/task-tmp" cmake --build build-adaptive-production -j2
python tests/diagnostics/recal2026/cartesian_full_accuracy_audit.py --output NEW_CARTESIAN_AUDIT
python tests/diagnostics/recal2026/cartesian_internal_contract_audit.py --output NEW_CONTRACTS.json
TMPDIR="$PWD/build-adaptive-production/task-tmp" python tests/diagnostics/results/recal2026/cartesian-final-redesign-20261015/paired.py --name NEW_PAIRED --count 256 --cartesian-row-ratio 1
```

rawのsource xには`time`、qには入力qの逆数を使用する。保存された別の`x`列をそのまま代入しない。VBMのsource xは逆符号。参照裁定を適用せずに精度違反件数を集計しない。
