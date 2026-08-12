# レビュー：純グリッド LCB-in と VBM の速度比較

作成日：2026-08-09
対象：[`HANDOFF_pure_grid_vbm_speed_comparison_20260809.md`](HANDOFF_pure_grid_vbm_speed_comparison_20260809.md)
対象データ：`near_caustic_pure_grid_large_equal_timeout_20260809/merged/results.json`
対象ブランチ：`master`

引き継ぎ資料の記述を harness と merged raw result に突き合わせた結果をまとめる。
記述そのものに誤りは見つかっていない。以下は、結論の解釈と次の一手を変える
可能性がある指摘である。

## 0. 先に確認できたこと

引き継ぎ資料の主張のうち、独立に裏を取れたものを先に挙げる。

- 対象が純粋な有限光源 grid ルートに限定されている点は
  [`bench_grid_vs_vbm_dark.py`](bench_grid_vs_vbm_dark.py) の `_pure_grid_route` と整合する。
  `point_source`、`hexadecapole`、`source_plane_quadrature` は確かに排除されている。
- `BinaryMagDark` の第6引数に関する §4 の注意はコード側の実装と一致している。
  `a1` はオブジェクトに設定され、第6引数には `VBM_ABSOLUTE_FLOOR = 1e-12` が渡っている。
- reference は VBM 由来ではない。[`reference.py`](reference.py) の `build` は
  LCB-in Cartesian の最細 certified バケット値を採用し、polar と VBM contour の
  不一致から uncertainty を決めている。
- §3 の「timeout 勝敗は発生していない」は raw result と一致する。
  `ratio_status` の内訳は `measured` 5671、`vbm_unresolved` 1 のみである。

## 1. VBM の tolerance 階段は交絡していない

VBM は `RelTol = target` から 10 倍刻みで締めながら、reference との誤差が
target 以下になった最初の tolerance を採用している。刻みが粗いため、
「target で通らず 1 段締めた結果、必要より 10 倍厳しい精度で計時されて
VBM が不当に遅く見える」という交絡があり得る。もしこれが linear LD、
`epsilon=1e-4` で効いていれば、§5 の逆転はアーティファクトになる。

merged raw result の `vbm.selected_reltol` を集計した結果、これは起きていない。

| profile | target | `reltol = target` で確定 | 1 段締めた | 2 段締めた |
|---|---|---:|---:|---:|
| uniform | `1e-2` | 1132 | 0 | 0 |
| uniform | `1e-3` | 1132 | 0 | 0 |
| uniform | `1e-4` | 512 | 0 | 0 |
| linear | `1e-2` | 1106 | 17 | 1 |
| linear | `1e-3` | 1122 | 2 | 0 |
| linear | `1e-4` | 646 | 1 | 0 |

linear LD、`epsilon=1e-4` では 647 点中 646 点が一発で確定している。
§5 の勝率表と median R は tolerance 階段の副作用ではない。ここは
そのまま主張してよい。

## 2. 支配項は速度則ではなく固定費である

これが最も重要な指摘である。§6 と §7 は `A_finite`、実測 `d/rho`、`rho` で
速度則を探しているが、大半の点では変動項が固定費に埋もれている。

merged raw result の `chosen_seconds` を集計すると次の通り。

- 全 5672 点で LCB-in 側の最小値は **1.22 ms**。
- p10 は 6 条件すべてで **1.66 ms から 2.33 ms** の狭い帯に張り付いている。
  target を 1e-2 から 1e-4 に締めても、profile を uniform から linear に変えても、
  この床はほとんど動かない。
- 全 5672 点のうち **4074 点、71.8% が `nbin=16`**、つまり探索ラダーの最小格子で
  精度条件を満たしている。

格子を上げたときの伸び方も、固定費支配を示している。

| method | nbin | points | median seconds |
|---|---:|---:|---:|
| cartesian | 16 | 1574 | 2.13 ms |
| cartesian | 32 | 129 | 3.14 ms |
| cartesian | 64 | 55 | 4.40 ms |

セル数が 16 倍になっても中央値は 2 倍にしかならない。積分本体が
`O(nbin^2)` で効いているなら 16 倍になるはずである。

対する VBM の中央値は uniform `1e-2` で **53.8 µs**、uniform `1e-4` で **200.3 µs**。

したがって uniform の `median R = 0.024` は「LCB-in の積分が 40 倍遅い」ではなく、
**「1 エポックあたり約 2 ms の固定費と、VBM の 54 µs の比」**である。
`A_finite` 単独の単調な速度則が見つからないのは当然で、
測っている量の大半が積分仕事量ではないためである。

この固定費は計時区間の内側にある。[`light_curve.cpp`](../../../src/lcbinint/lc/light_curve.cpp) の
`evaluate_preplanned_diagnostic` は `LensModel` の構築を計時区間の外に置き、
`lens_model.magnification()` だけを `steady_clock` で挟んでいる。
つまり 1.2 ms は `magnification()` の中で消えている。

**提案**：§11 の holdout 層別化より先に、この 1.2 ms の内訳を実測して切り分ける。
`nbin=16` の 1 エポックで何が 1.2 ms を使っているかが分かるまで、
`A_finite` と `d/rho` と `rho` の 3 次元層別化はノイズを層別化することになる。

## 3. 計時が非対称である

LCB-in 側と VBM 側で、測っている区間が違う。

- LCB-in：C++ 内部の `steady_clock` で `lens_model.magnification()` のみ。
  pybind11 の呼び出しオーバーヘッドと `LensModel` の構築は含まれない。
- VBM：Python の `perf_counter` で `vbm.BinaryMagDark(...)` 全体。
  pybind11 のオーバーヘッドも VBM 側のレンズ設定も含まれる。

向きは LCB-in 有利である。LCB-in 側は ms オーダーなので影響は無視できるが、
VBM の 34 µs から 54 µs に対しては数パーセント効く。結論を覆す量ではないため
再測定は不要だが、レポートには一行明記すべきである。

## 4. reference が LCB-in Cartesian 自身の値である

[`reference.py`](reference.py) の `build` は、reference の**値**を LCB-in Cartesian の
最細 certified バケットから取り、polar と VBM contour は uncertainty の算出にのみ
使っている。したがって VBM は「LCB-in Cartesian の答えを再現する」ことを
要求されている。Cartesian 格子に収束しない系統誤差があれば、それは
LCB-in 側には見えず、VBM の誤差として計上される。

ただし `_usable` が `reference_floor <= 0.1 * target` を要求しているため、
contour witness が 0.1×target より大きくずれる行は最初から除外されている。
偏りは target の 10% に抑えられており、致命的ではない。

とはいえ、勝率が 50% 付近で拮抗する linear LD、`epsilon=1e-4` では
この 10% は無視できる大きさではない。§4 と同じ扱いで、
「reference は中立ではなく LCB-in Cartesian 基準であり、偏りは target の
10% に bounded である」と明記するのが誠実である。

## 5. harness の潜在バグ（今回の結果には影響なし）

[`bench_grid_vs_vbm_dark.py`](bench_grid_vs_vbm_dark.py) の `_time_vbm_candidate` に、
timeout 後に後続点を取りこぼす条件がある。

```python
        if message is None:
            statuses[point_index] = "timeout"
            process.terminate()
            process.join(5.0)
            remaining = remaining[position + 1:]
            break
        ...
    if position >= len(remaining):
        remaining = []
        finished = True
```

`remaining` を切り詰めた**後**の長さと、切り詰め**前**の `position` を
比較している。reference epoch が 4 点で `position=2` の点が timeout すると、
`remaining` は長さ 1 になり、`2 >= 1` が成立して `finished = True` となり、
外側の再スポーンループを抜けてしまう。残る 1 点は `unrequested` のままになる。

同じ形が `error` 分岐にもある。

影響は測定漏れではなく、精度要求の水増しである。`unrequested` の点は
`_time_vbm` の `pending` に残るため、次の 10 倍厳しい `reltol` で測り直される。
つまり **timeout した点の巻き添えで、隣のエポックが必要より 1 段厳しい
tolerance で計時され、VBM が不当に遅く見える**。

今回の大規模結果では timeout が 1 件も発生していないため、この経路は
発火していない。§1 の集計で 1 段締めた 20 点は、このバグではなく
VBM の `RelTol` が実際に届かなかった点である。

`finished` フラグはすでに存在するので、長さ比較ではなく
内側ループが正常終了したかどうかで判定すれば直る。
引き継ぎ資料 §11-5 の harness 改善と同時にやるのがよい。

## 6. 結論への影響

| 指摘 | 結論への影響 |
|---|---|
| §1 tolerance 階段 | 影響なし。§5 の勝率表はそのまま有効 |
| §2 固定費支配 | **大きい**。§6 と §7 の解釈と §11 の優先順位が変わる |
| §3 計時の非対称 | 小さい。VBM 側に数パーセント。注記のみ |
| §4 reference の基準 | 小さいが bounded。linear `1e-4` では注記すべき |
| §5 harness バグ | 今回は影響なし。次回 harness で修正 |

引き継ぎ資料 §5 の勝率表と median R は、そのまま信頼してよい。
変えるべきなのはその解釈である。追記するとすれば次の一文になる。

> uniform での敗北の主因は積分効率ではなく、1 エポックあたり約 1.2 ms から
> 2 ms の固定費である。全 5672 点の 71.8% は最小格子 `nbin=16` で
> すでに精度条件を満たしており、格子を 16 倍にしても実測時間は 2 倍に
> しかならない。

この一文があると、§6 と §7 で `A_finite` 単独の速度則が見つからなかった
理由が説明でき、§11 の次の一手が holdout 層別化から固定費の
プロファイリングに変わる。

## 7. 再現方法

本レビューの集計は既存の merged raw result のみを使っており、
再測定は行っていない。

```bash
python - <<'PY'
import json, collections, math, numpy as np
p = ("tests/diagnostics/results/recal2026/"
     "near_caustic_pure_grid_large_equal_timeout_20260809/merged/results.json")
d = json.load(open(p))

# 1. VBM tolerance の階段数
steps = collections.defaultdict(collections.Counter)
# 2. 絶対時間と nbin
by_nbin = collections.defaultdict(list)
grid_all = collections.defaultdict(list)
vbm_all = collections.defaultdict(list)
for r in d["results"]:
    key = (r["profile"], r["target"])
    for i in range(4):
        rel = r["vbm"]["selected_reltol"][i]
        if r["vbm"]["selected_status"][i] == "completed" and rel is not None:
            steps[key][round(math.log10(r["target"] / rel))] += 1
        g, v = r["chosen_seconds"][i], r["vbm"]["selected_seconds"][i]
        if g is not None:
            grid_all[key].append(g)
            by_nbin[(r["chosen_grid"][i], r["chosen_nbin"][i])].append(g)
        if v is not None:
            vbm_all[key].append(v)

for k in sorted(steps):
    print(k, dict(sorted(steps[k].items())))
for k in sorted(grid_all):
    g, v = np.array(grid_all[k]), np.array(vbm_all[k])
    print(k, "grid p10/med %.0f/%.0f us" % (np.percentile(g, 10) * 1e6,
                                            np.median(g) * 1e6),
          "vbm med %.1f us" % (np.median(v) * 1e6))
total = sum(len(v) for v in by_nbin.values())
n16 = sum(len(v) for k, v in by_nbin.items() if k[1] == 16)
print("nbin=16 share: %d/%d = %.1f%%" % (n16, total, 100 * n16 / total))
for k in sorted(by_nbin, key=lambda k: (k[0] or "", k[1] or 0)):
    v = np.array(by_nbin[k])
    print(k, len(v), "med %.0f us" % (np.median(v) * 1e6))
PY
```

## 8. 提案する次の一手

引き継ぎ資料 §11 を置き換えるのではなく、順序を差し替える。

1. `nbin=16` の 1 エポックで消える 1.2 ms の内訳をプロファイルする。
   これが下がれば uniform の勝敗そのものが変わる可能性がある。
2. 固定費を切り分けたうえで、変動項だけで `A_finite`、実測 `d/rho`、`rho`、
   `q` の層別化をやり直す。固定費込みの層別化は、大半の点で
   定数を層別化していることになる。
3. `_time_vbm_candidate` の終了判定を `finished` フラグに直す。
4. `d/rho` の intended 値と actual 値を benchmark raw result に
   最初から保存する（元の §11-5）。
5. production dispatcher 全体への拡張（元の §11-3）は、
   固定費の切り分けが終わってからでよい。
