# Phase46: bracket点積専用化によるtiny-v追跡の再評価（不採用）

基準8cacc87。Phase45はquartic戻りを減らせるが認証費用で負けたため、区間積の既知point operandを専用化して再評価した。

## 試作

HOLO_BRACKET_POINT_MULで、片方の区間端点が等しい有限regular値なら重複する4積を2積へ減らす。外向きdown/up、subnormal/overflow/underflow拒否、ok propagationを維持。一般interval積はそのまま。Phase45のtiny_bracket_accept.patchと併用した。

既存test_quartic_local_bracket.cppはPASS。40000 interval comparisons、1000中999 polynomial enclosure check。非有限・underflow拒否、誤root/重複/root at chart cut拒否、FTZ/DAZ入力guard、FE_UPWARD拒否を既存testで確認した。全入力の浮動小数ビット同値を新たに証明したわけではない。

## A/B

同じ7216行、CPU0、warm-timing、candidate→baseline逐次単回。各steady2706行。

|中央値ms|uniform baseline→candidate|LD baseline→candidate|
|---|---:|---:|
|whole profile|.247007→.2486035|.3060455→.3077045|
|arc|.0689175→.0705975|.0793855→.0802465|

quartic戻り件数はPhase45受理案と同じuniform37869、LD39730。Phase45受理案とのmu差0、rowごとの戻り件数差0。今回baselineとはok/node差0、mu差/(1+|mu|)最大9.69e-12。

区間演算を専用化しても通常追跡に比べてまだ遅い。利益がないので双方をsourceから撤去。point_mul.patchと既存Phase45 patchで再現可能。production default/router/VFloor gate変更なし。Phase45で実施済みのreferenceは使い回し、新規reference/5Jac/whole14,432行は実施していない。今回のコーパスで前回受理案とmu差0であることと、別の未評価入力での同等性は区別する。

## 再現

Phase45 tiny_bracket_accept.patchと今回point_mul.patchを適用。

```
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -DHOLO_BRACKET_POINT_MUL tests/holonomic_cpp/test_quartic_local_bracket.cpp -o /tmp/p46_test -lquadmath
/tmp/p46_test
```

profile runnerはPhase45と同flags＋-DHOLO_MV_TINY_BRACKET_ACCEPT -DHOLO_BRACKET_POINT_MUL。bench_adaptive_stage_profile.cpp、input_snapshot.tsv warm-timing、CPU0。baselineは両defineなし（Phase44 baseline binary、solver behavior同一）。raw/summary/patchはevidence/holonomic/adaptive_extreme_phase46/。

## 判断

fallback件数を減らすだけでは速くならないことを再確認した。このbranch認証をさらに小さく磨くより、既にある同等認証の共有や通常quartic warm補正の仕事削減など、別のまとまった費用削減が必要。現時点で受理を変更する理由はない。VBM1e-3勝利は未達。
