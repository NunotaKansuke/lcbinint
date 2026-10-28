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
