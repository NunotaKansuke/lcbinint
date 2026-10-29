# Triple LightCurve: event precision / exterior clearance

## 結果と限界

branch `experiment/algebraic_boundary`、基点 `43e1f083ac6b682b46cd01a146c62397102d14da`。1800条件・7880 epoch、値と全10パラメータ微分付きの両経路で1791条件成功、9条件停止。元の同一集合に対して値+9、微分+15、既存成功の分類回帰0。新規固定seed32条件・160 epochは両経路32/32（旧25/21）。最大8 CPU・RAM16 GiB内、通常6 worker、各ライブラリ1 thread、worker AS上限2 GiB、観測最大RSS47,208 KiB。

参照を再監査した4502比較（2251点、471条件）で要求値誤差超過0、最大error/requested=0.816735。これは全7880点や全微分の精度認証ではない。160曲線・480不変性検査（角度2π、epoch/t0同時移動、逆順・重複）で失敗0。最大値相対差5.833e-7、最大scaled微分差2.378e-5（検査閾値1e-4）。要求rtol=1e-3/1e-4、atol=0を維持。

残る停止は主集合ID 400,403,433,439,460,467,484,499,515。両経路ともroot cluster / degenerate radial eventの精度・証明限界で安全に拒否する。個々の極に適した局所chartと退化イベントの処理は未解決。重いLDの角度分割・endpoint処理も残る。完全解決とは扱わない。

## 採用修正

* 小さいfoldの中心箱を段階的に縮め、毎回物理Krawczyk一意性を再証明する。丸め床・元の箱との共通部分を維持し、成功を強制しない。
* support半径の下限を各像自身のスケールで外向き丸めする。遠い像の半径で小さい像の正の下限が0になる問題を解消。全像・全IFT寄与を保持。
* 通常の全chartが失敗した後、fresh supportとイベント精度救済を実施。必要時は残るsecondary-anchor置換でBernstein/quad角度分離を使い、全微分を元の物理順序へ戻す。case483の角度2πでの停止を解消。
* 物理質量とレンズ間隔から全causticを含む円盤の和を区間演算で包む。円盤の外側というだけでは穴を排除できないため、無限遠へ抜ける軸方向rayも証明する。winding=0と4像を保証できた遠方配置だけ、既存のpoint/hex判定へ進める。失敗時は従来のcaustic treeへ戻る。幾何だけを厳密な9成分キーでcacheし、値・微分はcacheしない。rho=0経路は変更しない。

包絡の導出：M=総質量、δ=最小レンズ間隔、critical pointから最近接レンズまでrとするとr≤√M。r≥δ/2なら対応causticはそのレンズから√M+2M/δ以内。r<δ/2ならm_j/r²≤1+4(M−m_j)/δ²より、min(√M,δ/2)+√(m_j(1+4(M−m_j)/δ²))+2(M−m_j)/δ以内。両者の最大を全レンズで外向き評価する。正質量・異なる極・通常丸め環境を要求し、証明できなければ拒否する。

productionは既存point/hexまたは値で制御する結合積分を使う。微分誤差による適応制御、primal分割のtangent再生、暗い像の省略、source-disk積分への置換は導入していない。

## 独立精度とVBMの注意点

case504のLDをuniform diskの独立積分で検査（NoPoly、GL64/128/256、central差分とRichardson）。t0微分はnative −0.01537605634236、独立 −0.01537605643846、rhoは0.13254048871515対0.13254048868002。case483もGL128/256/512のMP/NP独立値と差分で検査し、t0/rhoは約1e-9以内で一致。LD専用VBMは分割上限100で未収束のまま一致する場合があり、相互一致だけを基準にしない。

case514は80桁・原レンズ方程式・4像すべてを用いる独立source-disk積分を検証専用に実施。3段階メッシュが一致し、残差最大1.427e-69。倍率は各epochで約1.00000024。nativeと約1 ULPで一致する。一方VBM5.5の公開Python APIでもMultiPoly/NoPolyがrtol=1e-7で全点1.0を返すことを再現した（`reference-py-tight-0514.json`, `py-tight-0514-*.json`）。通常の比較rtol=1e-3ではこの偏差は許容内だが、厳しい参照には使えない。VBM由来の1から8 ULP以内の値は独立な裏付けなしに厳密参照へ採用しない。514は上の80桁参照で置換した。

新規32条件の通常VBM比較はSinglePoly29成功・3件CPU20秒上限、MultiPoly/NoPoly32成功。SinglePolyのcase3 epoch4は適格MP/NP参照に対して相対誤差1.942。厳しいMP参照は6曲線失敗し、さらに誤差推定超過や未検証のunlensed値を除外した。失敗も生データに保存。MP/NP一致自体は真値の証明ではない。

無条件の極近傍frame優先試作は値の比較を通ったが、case321のalpha微分が独立差分約0.3957046に対し0.39594235となったため棄却。採用値は0.3957049573。この反例を回帰試験へ追加した。NPによるこの差分診断のSIGSEGVも保存。早期prepared-support優先試作もcase483の仕事量増加により撤回した。

## 速度と内訳

同一入力・許容値、CPU0固定、各1 thread、2 warmup後9回のthread CPU中央値。import/constructorを除く公開APIの曲線全体。外部負荷・周波数は固定していない。修正前の速度基準は**正確性修正済み中間DSO 3fbe908e…**であり、514が失敗する元HEADではない。VBM列は値のみ、native微分列は全10微分。

|case|native値 前→後 ms|native微分 前→後 ms|VBM Single / Multi / NoPoly ms|
|---|---:|---:|---:|
|21|0.392→0.390|0.338→0.401|1.457 / 1.823 / 2.783|
|514|656.451→0.300|478.199→0.318|0.00308 / 0.00392 / 0.00285|
|483|288.540→295.118|298.928→301.496|wall75秒上限 / 94.126 / 178.386|
|940|20.897→18.225|212.643→220.617|wall75秒上限 / 50.164 / 224.222|
|972|105.019→93.132|1047.374→1039.379|8.461 / 11.065 / 38.873|
|1508|43.742→44.935|647.696→649.510*|24.687 / 34.136 / 77.057|

*1508微分の初回測定は662.535→852.901msだったが、独立processのA/B/B/A再測定では約0.65秒で同等。表は再測定各18sample中央値。初回結果も保存する。普遍的高速化やfit全体の高速化は主張しない。精度参照coverageは21全7点、514全5点、483全5点、972は3点中2点、940/1508は現参照集合では0点（速度のみの行）。

514のcoldも値645.741→0.885ms、微分463.397→1.127ms。旧center-epoch単独profileではframe0準備34.06ms+失敗後support31.74ms、frame1準備1.79ms+global26.37msで拒否、frame2準備0.390ms+結合積分3.604msだった。新しい幾何証明はこの試行列を避け、既存の4像IFT評価へ進める。これは該当遠方曲線での改善であり、重いLD全般の解決ではない。

## 検査・再現性

C++33/33、対象Python120/120通過。critical曲線216包絡、独立tree winding12件、6置換、円盤の和に穴がある反例、丸め環境拒否も検査した。全repo pytestの成功は主張しない。既存の別JAX/legacy失敗9件は以前baseline再現済みで今回の変更対象外。

cold/warm試験は初め誤って1e-12の同値を要求していた。cold結合積分とwarm hexの差が要求値精度内かつ既存scaled微分閾値1e-4内であることを独立参照でも確認し、試験の要求を契約へ合わせた。production許容値は変更していない。C++診断の初回batchはOpenBLAS thread環境未設定で起動失敗したため保存し、各1 threadで全件再実行した。

`SUMMARY.json`, `PROVENANCE.json`, `timing-summary.json`, `remaining-failures-final.json`と生データarchiveを参照。採用DSO SHA256は`0cb54c74b5e67ac30732722fd49d8162c2801a8f706d70c5112f5a4cffdd6fb5`。VBM5.5、g++11.5、Python3.10.19、ビルドflag・全source hashはprovenanceに保存。既存の未コミットbinding変更を含むDSOを測定しており、その変更は今回コミットせずsnapshot/hashで固定した。clean checkoutとbit同一なビルドと誤解しない。

archiveには入力、全variantの行、棄却試作、log、source/test snapshot、実行scriptを保存し、DSO・実行binary・compiler scratchを除外する。`archive.json`に件数・SHAを記録し、全tar memberをmanifestと照合する。元の証拠は削除しない。scriptの再実行前にはarchiveを元の相対パスへ展開し、PROVENANCEのsnapshotとnative hashを照合すること。これは固定環境での再現手順であり、任意環境の一発再実行保証ではない。
