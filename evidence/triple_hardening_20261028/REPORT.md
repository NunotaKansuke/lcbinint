# Triple LightCurve hardening: faint-image recovery

## 採用した変更と根拠

`experiment/algebraic_boundary` / `37e7f6118a656455aad95b5a88e08bbc0b8f8f49` の既存dirty checkoutを継続。今回新たに採用した実装変更は `src/lcbinint/magnification/point_source_magnifier.cpp` のみ。以前の修正を含むその他のsource hashesは前回finalと一致する。コミット・push・新規worktree作成はしていない。

次数10の多項式求根で非常に暗い負パリティ像が失われると、物理像が3個になり、独立caustic countとの一致を要求するpoint/hex guardが通らず全面積分へ進む。cold retryでも同じ取りこぼしを再現。alphaの2π変更などの微小な丸め差で4像へ戻る場合があり、計算経路と近似微分の不変性を損なっていた。

既存の高精度polish後も4像未満の場合に限り、各lens poleの局所負像から追加候補を作る。D = source − pole + Σ(other masses / conjugate pole separation), r = −2m/(|D| + hypot(|D|,2√m)) を使う。候補を既存long double Newtonで元のレンズ方程式へpolishし、既存の残差1e-7以下、有限の負Jacobian、既存の重複判定を通ったものだけ加える。多項式の非物理候補は診断tailに残す。像数・caustic clearance・point/hex誤差の既存guardを維持する。要求許容誤差は変更しない。

本修正で全画像の完全性を一般に証明したとはしない。double位置へ戻すだけで残差を満たせない極端な低質量・近接配置は回復させない。

## 全件の再検証

前回固定の1528条件 / 6520 epochを再利用。wide_random 400、near_collinear_or_clustered 120、rho=0 48、sampled_caustic_contact 960。rtol=1e-3 / 1e-4、atol=0、linear LD=0/.5/1。native値・値＋微分を3056独立workerで再計算。VBM SinglePoly/MultiPoly/NoPolyの同一入力baselineは前回保存物を再利用し、変化・回復・速度測定の条件に厳しいMultiPoly/NoPoly参照を追加した。

| 全曲線の結果 | 修正前 | 修正後 |
|---|---:|---:|
| 値 成功 | 1420/1528 | 1460/1528 |
| 値＋微分 成功 | 1400/1528 | 1441/1528 |
| 以前成功→今回失敗 | — | 0 |

45条件・81 engine-curvesが回復。残りは値68 exception、微分77 exception＋10 invalid_gradient。成功のみの統計に失敗を混ぜない。3056 raw結果について入力hashとnative DSO hashを全件確認した。

厳しい独立参照はRelTol=1e-7。MultiPoly/NoPolyの相互差が要求誤差の1/10以下の点だけ精度集計へ入れる。以前特定したLD annulus上限の誤値3点には、VBM一様diskから別途積分した独立LD値を用いる。

646条件・2286source points・4537比較で要求誤差超過0。最大相対誤差0.00081673459、要求誤差に対する最大比0.816735。

曖昧な参照151比較、参照engine失敗59行は除外理由付きで保存し、成功扱いにしない。参照が互いに一致しても真値の証明ではない。

## 不変性・微分

130軌道に対して、alpha/angを2π変更、時刻原点変更、逆順＋重複epochの390検査を実施。全検査でstatus=0、gradient_valid=true。値の許容差超過0。旧版と共通して成功していた359検査の最大微分差 abs(g′−g)/(1+abs(g)) は 0.0119597292 → 4.61596e-9。これは表現変更への安定性であり、物理的な全微分精度の証明とは分ける。

case21/255の各10列についてVBM MultiPoly/NoPolyの中央差分を3段階のhで保存。NoPolyのrho以外18列は診断許容差 abs1e-3 + rel3e-3内。MultiPolyには微小hでのノイズが見える。point近似が選ばれたrho微分は0で、真の有限光源のrho微分精度を保証しない。case255のNoPoly差分は約0.0121046で安定しており、この既存近似の限界を隠さない。微分誤差を使うadaptive controllerは導入していない。

## 同条件での速度

CPU0、1 thread、逐次独立process、2 warmup＋15回のthread CPU中央値。import/constructorを除く7 epoch全曲線の時間。build・参照・profile・他campaignの終了後に測定。外部負荷とCPU周波数は固定していない。値＋微分は値を含む。

| case | 値 前→後 ms | 値＋微分 前→後 ms | SinglePoly ms | MultiPoly ms | NoPoly ms |
|---|---:|---:|---:|---:|---:|
| 6 | 失敗→0.459 | 失敗→0.487 | 1.523 | 1.638 | 1.719 |
| 21 | 16.671→0.397 | 21.821→0.417 | 1.919 | 2.408 | 3.694 |
| 255 | 10.476→0.401 | 13.337→0.419 | 1.864 | 2.461 | 3.799 |
| 384 | 32.692→0.368 | 30.350→0.388 | 0.997 | 1.297 | 1.849 |
| 192 | 1.721→1.719 | 2.262→2.276 | 1.210 | 1.572 | 1.751 |
| 54 | 12.666→6.575 | 20.753→15.792 | 608.229 (誤値) | 2.257 | 6.651 |
| 116 | 8.446→1.741 | 14.654→4.264 | 56.993 | 1.907 | 3.659 |
| 200 | 1.875→1.820 | 5.567→4.624 | 7.932 (誤値) | 4.006 | 5.798 |

全12条件と全試行はtiming-comparison.csv / timing-with-accuracy.json。SinglePoly case54は相対誤差約1.35e8、case200は約0.118で不適合。精度を満たす速度として比較しない。VBMのTol=0なので、VBM既定のpoint shortcutを使う速度とは異なる。

内訳の確認: case21の7 epoch中6、255の5、384の6が全面積分から、独立caustic guard付きpoint近似へ移った。case6は7点すべて。根の回収そのものを高速化したというより、取りこぼしによる不要な積分・証明・endpoint作業をなくした効果である。通常の積分を一律40倍高速化したとは説明しない。詳細stage profileは前回triple_stress_20261028/REPORT.mdを参照し、新版のstage実測とは区別する。

## 接触点の調査と不採用案

値のみ成功・微分失敗の接触22条件を独立C++probeで再現。geometry中点のphiが丸め誤差見積もり以下なのにdouble符号だけで棄却される例を確認。635ではphi≈−1.11e-14、roundoff≈4.12e-12でquad値は正だった。tangent endpointのNewton早期returnがquad fallbackを迂回する経路も確認した。

局所quad membership＋endpoint試験変更では一部が回復するが、途中1665 engine-curvesのうち既存成功58件の回帰が出たため採用せず、プロセスを停止しsourceを前回と完全一致に戻した。試験DSO、partial rows、source、ログはfixed-package / geometry-trial-* / trial-*.hppに保存した。初回launchのbuild参照先設定エラーもfailed-launch-*へ分離し、科学結果には使わない。

次の停止は、radial panelが隣接double値まで縮み、空間分解能witnessが満たせずerror=infのまま二分不能となるもの。例613の区間[0.44928038235871848,0.44928038235871853]。判定を緩めたり微小panelを誤差評価なしに捨てたりする変更はしていない。635/782では独立topologyもoverlapping_root_bracketsを返す。これらは未解決。

最終版の失敗理由は重複を許す集計で、値側overlapping_root_brackets 40条件、degenerate_radial_event 29条件、unresolved_root_cluster 7条件が中心。微分にはpatterson_geometry_repair 10、stream_local_repair 7、panel_resolution_limit 5も残る。全域収束や1e-6精度を認定しない。

## 検証と再現

- Python対象51 pass。追加7回帰テストは旧版7 fail→新版7 pass。
- 必要なC++全targetを再buildし、ctest 30/30 pass。
- git diff --check pass。無関係のdirty/untracked作業を保持。
- 採用DSO: fbb36b30818d0f1f141e8b088f773ca8ab8d38592987744d1b6b85177602dd75。
- 計算予算: 最大8 CPU / RAM16 GiB。全件nativeは5 worker、各AS上限2 GiB、CPU affinity1..5、BLAS/OMP各1。同時buildは-j2。追加参照は4 worker。速度測定は全て終了後CPU0で逐次。
- final-provenance.json、comparison.json、validated-reference-*.json、invariants*.json、route-changes.json、fd-summary.jsonに根拠。全件入力はcases.json / inputs/、最終rawはrows/。
- 再開: 同じディレクトリでPython run.py。既存結果は入力/DSO hashを照合して再利用。buildを変更した場合は新規出力先と新規freezeを使う。現行run.py/worker.pyの既定buildはseeds-package。
