# Seed frontier最適化とCartesian/polar比較

## 結果

seed生成を含み、caustic構築・距離stageを除いた比較。通常群の相対許容差1e-4、87 source/u条件で、新経路の合計時間はCartesianより約5.5%短く、polarより約52%短い。しかし中央値ではCartesianが速く、新経路が全ケースを上回る段階にはない。

|経路|群中央値 ms|87条件の合計 ms|新経路が速い条件数|
|---|---:|---:|---:|
|新経路|0.535|63.584|—|
|Cartesian|0.463|67.288|39/87|
|polar|1.038|133.621|56/87|

通常inverse ray側に有利な比較として、各ケースでnbin16/32/64/128/256/512のうち、D14との差が1e-4以下で最速の結果を事後選択した。nbin選択の実行費用は含まない。この比較用選択を実装側のルールには入れていない。両inverse rayとも87条件すべてに許容差内の解像度があった。caustic込みの合計比も新/Cartesian=0.996、新/polar=0.734。中心求根・seed・探索・積分を含め、各回fresh object/workspace。geometry再利用なし。

独立真値でなくD14との差を精度条件としている点に注意。参照自体もtol1e-4であり、厳密な誤差保証ではない。Cartesianの合計差5.5%は小さく、繰り返しcampaignで安定した優位を確認したものではない。

## 効いた変更

1. 積分点で根数変化を見つけたら、最後まで積分する前に中断し、局所foldを追加して再開する。
2. 途中で得た信頼できるsampleと、変更しないpanelのtreeを保持する。
3. 根数変化を検出しても局所foldを確定できない場合、部分評価した同じpanelで追加の積分点を得る。D14へ丸ごと戻さない。
4. 根数チェックと角度境界探索で同じ四次式を再生成しない。
5. band中央の無条件分割を外す。
6. warm seed選択は点が存在するslotだけ走査する。既存のdense走査と同じseedを選ぶ監査を実行。

配置別の閾値・例外表は追加していない。Newton/精度/予算の既存数値条件は維持。通常群178評価の積分点中央値81→69。途中試作での合計254ms→早期修復版128msは、過剰な細分化の削減を示す。最後の採用計測は128.616ms。long double係数生成を積分点にも揃える追加実験は、点数・速度の改善が確認できず取り下げた。

## D14との継続比較

|群|評価数|新経路中央値 ms|D14中央値 ms|新/D14 合計時間比|
|---|---:|---:|---:|---:|
|通常|178|0.522|0.703|0.528|
|fold|288|0.613|0.568|0.978|
|浅い接触|288|0.679|0.567|1.164|

これもseed込み・caustic除外。全754評価で新経路収束かつ指定tol内の一致。D14未収束142評価を分離し、全896評価を保存。浅い接触では依然D14より遅い。現在の通常群固定側費用中央値はseed0.227ms、境界準備0.079ms、積分0.212ms。seed費用が依然大きい。

## foldのbounded pilot

5 geometry×u=0/0.5の10条件でCartesian/polarも追加比較。inverse rayはnbin16/32/64/128から精度を満たす最速を事後選択。新経路はCartesianに4/10、polarに8/10で勝つ。群中央値は新0.499ms、Cartesian0.464ms、polar1.644ms。Cartesianへの全面的な優位はない。小さいplanetary sourceのpolarが非常に遅く、合計速度比を一般化しない。6番目の候補geometryはD14未収束で参照比較対象外。これはfold全体を代表すると認定したcampaignではない。

## 検証・再現

共有adaptive回帰1608 checks / 0 failures。新frontier回帰20 checks / 0 failures。早期中断が誤って収束を宣言しないこと、部分panelのsample保持・再開を追加検証。HOLO_ADAPTIVE_ANCHOR_AUDITによるsparse/denseのseed一致監査込み。

rogue1、GCC11.5.0、-O3 -march=native、1thread、3反復。新/D14のA/B順序は交互。標準inverse rayは現行main checkout b51c00c系のdirty sourceを独立buildし、最小seedは保存済みsnapshotを使用。元checkoutや他worktreeは変更なし。manifestにhashとdiffを記録。

生データ、集計script、採用source、再現条件はevidence/holonomic/seeded_inverse_comparison_v7/。*_final.tsvとcurrent_inverse_native*.tsvが採用計測。失敗試作・不採用の精度実験・中断した高nbin探索は消さず明示して保存。

全成分の数学的完全性、独立高精度真値、gradient、ABI全体、trajectory campaignは未検証。CALIBRATED=false。標準dispatchへの昇格なし。
