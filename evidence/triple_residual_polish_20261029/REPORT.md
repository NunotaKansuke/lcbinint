# Triple residual hardening — 2026-10-29

## 結果と適用範囲

`experiment/algebraic_boundary`、base `26c684b9` からの追加修正。通常の `LightCurve(lens="triple", param_type="vbm")`、static / single source、一様・linear LD、rtol=1e-3 / 1e-4、atol=0を検証した。最大8 CPU・16 GiBを維持し、新規worktreeやbranchは作成していない。pushは行っていない。

全1768条件・7720 epochで、値は **1743→1757成功、停止25→11**、微分付きは **1741→1755成功、停止27→13**。以前成功していた条件の回帰は0。接触960条件と既存holdout240条件も両経路で全件成功した。失敗も除外せず `remaining-failures.json` に保存している。今回の成功は全物理配置・全微分の精度認証ではない。

採用したDSOは `a9b2ca2c3d1cff8795a0329c009719c7865259f3fcc08143460694308f7030a6`、packageは `deferred-package`。`deferred-provenance.json` に関連source 68ファイルのhash、`source-snapshots/accepted` に内容を保存した。ビルド時の他作業のdirty bindingもhash/snapshotで識別し、その作業ファイルは今回の変更・stageに含めない。nativeはg++ / O3、C++独立検査はno-fast-math・ffp-contract=off。bit-identicalなclean rebuildを保証する識別子ではない。

## 修正した原因

1. **暗い像の座標の消失。** 極近傍の変位をレンズ位置へ加えると、binary64で変位が丸め落ちていた。33条件・195 witnessで、保持した局所変位なら残差約1e-15、投影した大域座標では最大約24まで悪化することを再現した。`TripleImageCandidate` に極とdouble変位を保持し、long double Newtonで磨き、保持する表現そのものを元のレンズ方程式の既存残差基準1e-7で検証する。場・曲率・IFT微分も同じ局所変位を使う。残差基準は緩めていない。`position` と低レベルの `TripleImage.position` は丸めた診断用投影であり、その値だけから極近傍の場を再構成してはいけない。全点源計算をquad化した変更ではない。
2. **偶数像数だけでは欠落を検出できない。** 通常配置147/482では、多項式由来の種が別の像へ収束し、正負の像が一対欠ける近傍もあった。原方程式の残差が小さいことと全像の存在は別である。独立したcaustic winding countをpoint / hex packetへ渡し、不足時のみ極の局所種と有限の多方向Newton種で回復する。追加の大域種は残差1e-12、非臨界Jacobianを要求する。像数が合わなければ近似を受理しない。像数の証明が取得できないrho=0は既存の数値的な点源契約を保ち、全点源をcoverage認証したとは主張しない。微分は回収した同じ像のIFTから計算し、多項式を微分しない。
3. **狭いcomponent maskの数値的拒否。** 完全になった像集合がoptional component partitionを有効にし、既存成功419/448が一時的に失敗した。通常の全primary frameを試した後に、同じsupport/IFT証明を保持した非分割積分、それでも必要なら独立した全support再構築へ進む。全像・全微分を保持し、許容誤差も変えない。救済を元frameで先行すると972の勾配付き呼出しが約1.99 sへ増えたため、試行順序を変更し約1.06 sへ戻した。これは以前の正式baselineに対する2倍高速化ではなく、今回の中間版の余分な救済費用を除いた比較である。

C++検査では丸め投影が元方程式の残差基準を破る実例、局所場と物理座標の有限差分、6像領域のcold/warm解・近傍有限差分を確認した。Pythonでは同じ2配置のrho=0を、独立した多方向探索＋winding countで確認した6像の参照値と比較し、scalar / gradient双方を検査する。

## 精度とVBMの扱い

既存の監査済み4289比較を全て再検証。追加23条件でVBM SinglePoly / MultiPoly / NoPolyとtight MP/NPの115実行を保存した。tightはRelTol=1e-7、LD max annuli=100で、両方の誤差見積もりが1e-7×abs(mu)以下、方法間の差が要求native rtolの0.1倍以内の点だけを参照にする。追加の比較は81点・157件、要求誤差超過0、最大の誤差/要求誤差は0.0941以下。

旧参照と重複を除いた総計は **460条件・2202点・4394比較、要求誤差超過0**、最大の相対誤差/要求誤差は0.81673459。`deferred-joint-validation.json` と全比較行を保存した。この2202点以外の全epochの精度、また全パラメータ微分の物理精度を認証した集計ではない。

MP/NP不一致2点、LD誤差見積もりが要求を満たさない9点、参照curve failure 7条件は追加参照から除いた。SinglePolyにはCPU上限・SIGSEGV等と大きな正の誤値があり、189では参照が一致した点でも誤差が最大約1.6e13倍に達した。VBMが値を返しただけでは正解に扱っていない。通常設定のtiming結果にも同じ品質上の制限がある。

## 速度

全campaign・build終了後、rogue1のCPU0、各library 1 threadで逐次測定。constructor/importを除く全epochのmagnification呼出しを2 warmup後に9回測り、thread-CPU時間の中央値を用いた。比較前は前commitの85ef83…、比較後は上記a9b2ca…で、同一入力・同一許容誤差。一方が停止した場合の高速化倍率は定義しない。外部負荷・CPU周波数は固定していない。

単位ms。勾配付きは10パラメータを返すnative全呼出しで、VBM列は値のみ。

| case | rtol | native値 前 | native値 後 | native微分 後 | VBM single | VBM multi | VBM no poly |
|---|---:|---:|---:|---:|---:|---:|---:|
| 21 | 0.0001 | 0.371 | 0.409 | 0.436 | 1.911 | 1.806 | 2.524 |
| 5 | 0.0001 | 停止 | 0.403 | 0.408 | 1691.201 | 1.625 | 2.007 |
| 52 | 0.001 | 停止 | 0.254 | 0.325 | CPU上限 | 1.228 | 1.797 |
| 135 | 0.0001 | 73.471 | 0.365 | 0.323 | CPU上限 | 2.273 | 3.976 |
| 189 | 0.0001 | 44.336 | 0.256 | 0.292 | 1610.859 | 26.386 | 4.904 |
| 940 | 0.001 | 17.674 | 17.055 | 177.504 | CPU上限 | 44.703 | 238.499 |
| 972 | 0.001 | 82.705 | 89.820 | 1058.822 | 9.207 | 12.804 | 35.705 |
| 1508 | 0.0001 | 37.408 | 44.754 | 743.774 | 31.688 | 35.382 | 80.262 |
| 489 | 0.0001 | 停止 | 0.239 | 0.252 | SIGSEGV | CPU上限 | 2.264 |

135の値は約201倍、189は約173倍。失敗していた52/489の値も回復した。一方、この測定では通常ケース21と重いLD1508の値は約10%、20%増えた。全配置の高速化ではない。189のsingle/multiには誤値があり、速度だけを横並びに評価しない。`deferred-controlled-timing` に生の9 samples、first-call値、返値、異常終了を保存した。

## 段階別profileと残るボトルネック

`stage_profile.cpp` は同じ物理kernelの独立したO2 header buildで、nested scopeをexclusive thread-CPU時間に分けた。cold support準備も別計測した。通常APIのO3 / warm計測そのものではなく、scopeの計測費用も含む。

- 972の中央epoch、元frameは約874 msを使ってrefinement budgetで拒否された。別frame2は約5.69 msで収束した。公開node数は採用試行のみなので、失敗試行の費用を表さない。救済順序の修正で二重の拒否試行を除いたが、最初の拒否試行自体の費用は残る。
- 1508の中央epoch、元frame約189 msのうちangular約126 ms、endpoint約26 ms、chart約19 ms。採用される全3 epochの微分付き評価点は約24.5万。別frame1は約1.57 msで収束した。
- 940の中央epochも元frame約166 ms、別frame1約2.61 ms。座標系で仕事量が大きく変わる。call履歴や微分誤差でframeを切り替える短絡は今回採用していない。値・証明だけで安定して選ぶ方法は別途検証が必要。

## 検証と未解決条件

C++ **32/32**、対象Python **96/96**。既存の別legacy-backendテスト9件が前baselineでも失敗する件は前報告の通りで、全projectの全pytestがgreenだとは主張しない。

154曲線で回転2π、epoch/t0同時平行移動、逆順＋重複epochの462チェックを実施し例外・status/valid違反なし。値の最大相対差5.84e-7、微分の最大scaled差2.38e-5。像欠落があった189では、中間版のrho微分差約2.4e-3が、修正後は約1.8e-14以下になった。これも任意の物理微分精度を保証するものではない。

7720点の元方程式監査では、返したphysical候補の残差基準違反0。通常配置の像数不整合も追加探索で解消し、残る72点はsampled caustic contactの点源候補である。臨界点付近で像数だけを証明に置き換えず、有限光源経路は別のsupport/IFT証明で判定する。

残る停止は値11条件、微分13条件で、全てnear_collinear_or_clustered。主に未分離角度root cluster・退化event、1件のgradient_event_precision。28失敗epoch×3 frameの84 support planについて、分割できない半径軸の代わりに角度軸を継続する実験も行ったが、追加成功0（完成10→10、48件は両方向で精度限界、26件はbudget）だったためproductionには入れていない。極端なfaint componentを局所座標で証明・積分する設計と、安定したframe選択が残件である。NaNや停止を成功に変えること、微分寄与の省略、許容誤差の緩和は行っていない。

## 再現用証拠

入力・全variantのraw結果・失敗・profile・native flags・source snapshotは `raw-evidence.tar.gz`、各ファイルのSHA256は `raw-evidence-manifest.json.gz`、archive全体の識別子は `archive.json`。DSO・static library・compiler一時ファイルはarchiveに含めず、検証したDSO hashと対応sourceを残す。profileのscope用headerだけは計測用overlayで、productionではない。

workerは6並列、各2 GiB address-space上限・CPU60/62秒・wall120秒。参照workerはCPU20/22秒、timingはCPU30/32秒。初期timing launcherが非atomicなstatus更新を読んで停止した履歴も残し、statusのatomic replaceとread retryを追加した。nativeのraw結果は影響を受けていない。最終timingは未開始だった分を、他のjobが終了した後に独立に実行した。
