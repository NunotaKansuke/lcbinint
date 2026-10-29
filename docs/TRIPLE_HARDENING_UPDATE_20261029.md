# Triple hardening update — 2026-10-29

**2026-10-29 更新:** この資料の旧「一旦区切り」以後、ユーザー依頼で大規模検証を再開。
最新結果・採用修正・失敗manifest・再現用archiveは
[triple_final_polish_20261029/REPORT.md](../evidence/triple_final_polish_20261029/REPORT.md) を参照。
1528条件で値1503、微分1501成功、接触960条件は両経路とも全件成功、既存成功の回帰0。
値25・微分27の極端配置で停止が残る。VBMのLD参照には分割上限未収束が多く、
参照gateを監査し直した1931比較で要求誤差超過0。全配置・全微分の認証ではない。
branchは引き続き `experiment/algebraic_boundary`。新規worktreeは作成していない。

追加の未使用240条件 / 1200 epochでも値・微分付きとも全件成功。参照gateを通した2358比較で要求誤差超過0。計1768条件・7720 epochを検証した。既知の停止25/27条件は個別manifestに残す。

元の `TRIPLE_IMPLEMENTATION_HANDOFF.md` は既存の未追跡資料のため、このコミットには含めない。この更新とevidenceだけを追跡する。

## 残差・像数の追加磨き込み

[triple_residual_polish_20261029/REPORT.md](../evidence/triple_residual_polish_20261029/REPORT.md) が最新。計1768条件・7720 epochで値1757、微分1755成功、以前の成功条件の回帰0。停止は値11・微分13条件。極の局所変位を保持し、独立した像数の証明でpoint/hexの欠落を検出・回収する。C++32件・対象Python96件、監査済み4394精度比較で要求誤差超過0。全配置・全微分の認証ではなく、極端配置の停止とLDのframe依存の仕事量が残る。

## LD座標選択の追試

[triple_ld_frame_selection_20261029/REPORT.md](../evidence/triple_ld_frame_selection_20261029/REPORT.md) に不採用実験と独立微分検査を記録。値だけの予備積分によるframe選択は、全体のCPU費用増加と2条件の独立微分不一致により棄却。productionは656b7e03の採用DSOへ復元し、hashも一致。独立LD差分4件を追加して対象Python100件、C++32件が通過した。11/13条件の停止と全微分の精度認証は依然別の課題である。

## イベント精度・遠方caustic証明の追加検証

最新は [triple_event_precision_20261029/REPORT.md](../evidence/triple_event_precision_20261029/REPORT.md)。1800条件・7880 epochで値／微分とも1791条件成功、9条件停止、成功分類の回帰0。再監査した4502精度比較で要求誤差超過0、480不変性検査・C++33件・対象Python120件通過。小さいfoldと像ごとのsupport丸めを修正し、遠方で全causticの外側かつ無限遠へ接続することを証明する幾何guardを追加。遠方case514は正確性修正済み中間版との同条件測定で値656→0.300ms、微分478→0.318ms。VBMの1.0返却を独立80桁参照と照合し、未検証のflat値は厳密参照から除外した。無条件frame優先試作は独立alpha微分反例により棄却。残る9条件と重いLDは未解決で、全微分精度の認証ではない。
