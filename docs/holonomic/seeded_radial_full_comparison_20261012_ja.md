# D14対seed経路：準備込み比較

## 結論

caustic構築を除き、seed生成を含めた場合、通常群の中央値は約20%改善。fold・浅い接触群の中央値はほぼ同等で、合計時間はそれぞれ29%、33%増加。全般的な高速化は未達。前のseed-ready比較から全体速度の改善を主張することはできない。

| 群 | D14 ms | 新：seed生成込み・caustic除外 ms | 新：causticも込み ms | caustic除外の合計時間比 新/D14 |
|---|---:|---:|---:|---:|
|通常|0.719|0.576|1.860|0.894|
|fold|0.583|0.593|1.941|1.287|
|浅い接触|0.575|0.577|1.899|1.330|

時間欄は各評価3反復の中央値を取り、群の中央値を算出。合計時間比は比較可能な全評価の時間を合計して比較。中央値同士の比と異なる。

## 内訳（群中央値、ms）

|群|中心像求根|caustic構築・距離|seed追加生成・座標変換|新境界準備|新積分|D14 topology|D14積分|
|---|---:|---:|---:|---:|---:|---:|---:|
|通常|0.0016|1.304|0.234|0.075|0.260|0.492|0.213|
|fold|0.0017|1.325|0.161|0.096|0.323|0.357|0.210|
|浅い接触|0.0017|1.323|0.096|0.106|0.338|0.341|0.195|

各内訳の中央値は同じ評価から選ばれるとは限らないため、和は総時間中央値と厳密には一致しない。

D14準備を軽くした利点を、seed追加費用と新経路の積分増加が相殺している。caustic除外でのp90も通常3.092対D14 2.426、fold1.274対1.236、浅い接触1.217対1.291ms。難しいケースのtailは未解決。

## 計測境界と実装

D14はepoch_adaptive（D14 event/topologyと積分）。新経路は毎回新しいFiniteSourceMagnifierを構築し、中心像、caustic構築・距離、最小seed生成、座標変換、seed radial frontier、既存角度kernelによる適応積分を実行。geometry/epochをまたぐ再利用なし。同一epoch内のcaustic再利用はseed生成に必要なもの。

ユーザー指定に合わせたcaustic除外値は、各反復の総時間から直接計測したcaustic構築・距離stageを除いてから中央値を取る。中心像求根、seed生成、オブジェクト構築、frontier、積分は残す。これはcausticがselector等の別用途で同epoch内に準備済みの場合の費用配分。causticを使わないアルゴリズムにしたものではない。D14側には不要なseed/caustic費用を足さない。全ABI/routing benchmarkではなく指定有限光源epoch経路の比較。

最小seedコードは通常ライブラリでなく、保存済みdiscriminant-algebraic-minimal-seeds-20261012-v1/sourceを使用。critical-seedsを選択。現在のroot checkoutや他worktreeは変更なし。

## 精度・条件

224geometry、u=0/0.5、tol=1e-3/1e-4で896評価。D14収束754評価すべて新経路も収束し、D14との差は指定許容差内。D14未収束142評価は別記録し、採用速度集合から除外。独立真値・全成分の数学的完全性・gradientは未保証。CALIBRATED=false、標準経路への昇格なし。

rogue1、GCC11.5.0、-O3 -march=native、各回fresh workspace、3反復、A/B交互、1thread。base f3547723b79828d9c501182746028e99f7bad116。source/input hashとpatchはevidence/holonomic/seeded_radial_full_v6/manifest.json等に保存。生データと再現条件は同directoryのREADME.md参照。
