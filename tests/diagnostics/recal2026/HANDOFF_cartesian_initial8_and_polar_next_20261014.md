# Cartesian初期8の固定・polar適応への引き継ぎ（2026-10-14）

## 現在地と次の作業

- ブランチ: `feature/boundary-error-and-discriminant-support`。この資料と実装を同じコミットで保存する。直前HEADは`a7b15c1601abecdf9c019f3f1fe9fbd9636209bd`。masterへのマージは未許可。
- checkout: `/rogue1_8/nunota/lcbinint`。別worktreeを作らず、他の未追跡研究ファイルを保持する。
- Cartesian初期8の補修削減は一旦区切る。次は高増光でpolarの角度方向・半径方向を独立に局所適応化する。次の実装はまだない。
- 標準initial_bins=16は変更しない。Cartesian8は`binary_adaptive_flux(..., initial_bins=8)`で明示選択する。許容値は1e-3/1e-4、Uniformとlinear LDで評価する。
- 数学的な一般保証は付いていない。`certified=false`、`complete_error_model=false`を維持。推定受理と実精度達成を混同しない。

## 保存した実装

1. 粗格子で分断された同じ像が細分化で結合する場合、旧成分のrun seedを追加し、同じ細格子で完成済み行・rayを再利用する。採用後だけdonorを退役して二重計上を避ける。より細かいdonorを破棄しない。
2. 初期8の外縁継続チェックと、既存行のshifted capモデルによる感度推定。新規完全求根は追加せず、同一格子の情報と少数rayを使う。
3. 粗い外縁の補修は可能な場合に上下capだけx/yを増やす。内部積分と誤差を保持。失敗時は計算済みrayと解像度が一致する完成行を使って必要な補修を続ける。
4. 既存行の滑らかなcore、および新しく挟む行のLD momentを再利用する。端点とBennett補正は実評価。薄い像、gap、junction、推定誤差過大では通常の走査。
5. 初期8の遅延rayキャッシュ。小さい評価予算では即時重複排除。
6. 先行ターンのpolar最適化も含む: occupied angular columnだけの積分保存・連結処理、linear LDの解析sqrt評価。角度適応はまだ実装していない。

主なファイルは`src/lcbinint/magnification/adaptive_cartesian_impl.inc`、`adaptive_cartesian_run_fill.hpp`、`adaptive_outer_endpoint_model.hpp`、`probe_diagnostics.hpp`、`finite_source_magnifier.cpp`と`python/bind_lc.cpp`。

## 最新の速度結果

`tests/diagnostics/results/recal2026/initial8-repair-cost-20261014/REPORT.md`に全条件と失敗を保存。前版8・新版8・初期16を同じ入力ごとに順序を回転して計測した。相対許容値を実際に満たし、双方が推定受理した組の幾何平均。

| 光源 | 目標 | 前版8 / 新版8 | 初期16 / 新版8 | 比較数 |
|---|---:|---:|---:|---:|
| Uniform | 1e-3 | 1.016 | 1.098 | 21 |
| Uniform | 1e-4 | 1.033 | 1.140 | 19 |
| LD | 1e-3 | 1.167 | 1.164 | 23 |
| LD | 1e-4 | 1.059 | 0.994 | 22 |

180調整・診断入力＋未使用レンズ配置46入力で、前版が実精度を満たした入力に新しい未達0。初期16は360比較で数値・受理・停止・ray数が前版と一致。12高増光LD呼び出しで再利用行105232個を内部全走査と照合しgap0。予算・例外後復帰36チェック成功。

未使用46入力ではUniformは初期16より約1.26/1.40倍、LDは1.21/0.97倍（1e-3/1e-4）。高増光の未使用配置は少なく、各profileのA>=300は1入力のみ。全体2倍達成とは主張しない。

### 残る精度問題

評価予算の削減で以前は止まっていたID1723/1727が受理まで進み、実相対誤差1.688e-4/1.199e-4を1e-4で誤受理した。Uniformは初期16と増光率一致、LDも予算を1000万へ上げた初期16で1.181e-4。既存の推定器の問題として残る。速度改善を精度保証と呼ばない。既知の誤受理と停止結果を表から消さない。

### 不採用

y方向だけの補修、端点予測の小さなwalk、補間の割当誤差予算の緩和は節約が小さいか遅くなったため戻した。LD1e-4を一律初期16にするような自動選択も入れていない。光源/目標だけで勝敗は分かれない。

## 次: 高増光polar適応

先行結果は`tests/diagnostics/results/recal2026/polar-cartesian-start-grid-20261014/REPORT.md`。高増光の精度達成部分では固定polar32がCartesian16より約2倍速いが、1e-4でpolar64が必要になると優位がほぼ消える。新しいCartesian8と比較すると差も縮む。polarの`converged`は指定許容値の受理ではない。

推奨する順番:

1. 現行polarの半径方向・角度方向の誤差と費用を分離する。面積要素r dr dphi、2π wrap、run連結、複数像成分を確認する。
2. 不足した角度区間だけ分割し、評価済み半径積分を保持、新しい角度列だけ評価する。全円を倍増したり過去の積分を捨てない。
3. 半径方向は境界の局所補修、LD内部momentは既存情報を利用する。Uniform/LDの構造を共有する。
4. 最新Cartesian16/8、polar適応、VBMを同一入力・同一目標・同一計時範囲で比較。両profile、1e-3/1e-4、fold/cusp/contact/高増光を扱う。
5. polar単体の効果を確認した後に、既存像位置やJacobianを使う安い座標選択を考える。増光率の硬い閾値だけで切り替えたり二つの増光率をblendしたりしない。

光源が動く/レンズ配置が変わる実運用を前提にする。前epochで同じgeometryが続くことを速度向上の主因にしない。新規求根・caustic構築・保証の追加費用で節約を失わない。

## buildと再現

- 通常production Release: `cmake --build build-adaptive-production -j2`。実験support経路OFF。
- 保存したrunnerのcompile引数は`initial8-local-repair-20261014/compile.json`と`paired-compile.json`。絶対パスはこのhost/checkout用なので別hostでは変更する。
- `initial8-repair-cost-20261014/paired.py`、`paired-fresh.py`、`verify.py`、`audit.py`は保存された入力と現行buildを用いる。前版native binaryはローカル`initial8-local-repair-20261014/runner`に残しているがGitには入れていない。別hostで前版比較するには当該実装のsnapshot/diffまたは保存TSVを使う。
- CPU0固定、1thread、warmup後3回の中央値。caustic/index構築と中心像求根は除外。seed取得、fill、Bennett/LD、推定、追加補修は含む。毎回の適応状態は新規。
- 入力の列は`geom_id,s,q,rho,time,y,c`。保存xではなくtimeを使う。native lens呼び出しのqは1/q、VBM physical sourceは(-time,y)。この規約を勝手に変えない。
- referenceは保存VBM RelTol1e-6。結果の倍率だけでなく実相対誤差と目標、拒否・誤受理を集計する。
- 結果ディレクトリはローカルexclude対象なので、今回必要な小さい入力・TSV・JSON・報告書・runnerソースだけ明示的にforce-stageした。実行ファイル、大きなtrace、buildや他研究の未追跡ファイルはコミット対象にしない。

対応する詳しい報告は`initial8-local-repair-20261014/REPORT.md`（初期8の補修）と`initial8-repair-cost-20261014/REPORT.md`（補修費用削減）。履歴の古い報告は当時の測定を表す。以後の正本はGitに保存した今回の入力・TSV・provenance。
