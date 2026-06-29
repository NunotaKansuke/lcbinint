# Terrestrial Parallax 使い方メモ

## 基本

`parallax=True` に加えて `terrestrial_parallax=True` も必要。obs_lat/obs_lon だけでは有効にならない。

```python
import numpy as np
import lcbinint

event = lcbinint.EventCoordinates(
    ra=270.0,         # イベントの赤経 (degrees)
    dec=-30.0,        # イベントの赤緯 (degrees)
    tfix=2459000.0,   # 視差の基準時刻 (JD or HJD-2450000)
    obs_lat=43.0,     # 観測所の緯度 (degrees, 北正)
    obs_lon=172.5,    # 観測所の東経 (degrees)
)

lc = lcbinint.LightCurve(event=event, parallax=True, terrestrial_parallax=True)

params = {
    "t0": 2459000.0,
    "tE": 100.0,
    "u0": 0.1,
    "alpha": 0.5,
    "s": 1.2,
    "q": 1e-3,
    "rho": 0.0,
    "piEN": 0.3,   # annual と terrestrial で共通
    "piEE": 0.1,
}

times = np.array([2459000.0, 2459001.0, 2459005.0])
magnifications = lc(times, params)
info = lc.info(times, params)
```

## フラグの組み合わせ

| parallax | terrestrial_parallax | 効果 |
|---|---|---|
| False | False | 視差なし（デフォルト） |
| True | False | annual parallax のみ |
| True | True | annual + terrestrial |
| False | True | 無効（piEN/piEE が0になるので terrestrial も効かない） |

## 複数望遠鏡

望遠鏡ごとに LightCurve を作る。piEN/piEE は共通。

```python
lc_moa = lcbinint.LightCurve(
    event=lcbinint.EventCoordinates(ra=270.0, dec=-30.0, tfix=2459000.0,
                                    obs_lat=-43.0, obs_lon=170.5),  # Mt John
    parallax=True,
    terrestrial_parallax=True,
)
lc_ogle = lcbinint.LightCurve(
    event=lcbinint.EventCoordinates(ra=270.0, dec=-30.0, tfix=2459000.0,
                                    obs_lat=-29.0, obs_lon=-70.7),  # Las Campanas
    parallax=True,
    terrestrial_parallax=True,
)
```

## obs_lat=obs_lon=0 のとき

`terrestrial_parallax=True` でも地心観測と同じ結果になる（(0,0) は geocenter 扱い）。

## 注意

- piEN=piEE=0 なら terrestrial parallax も効果なし
- 時刻の単位は annual parallax と同じ (JD or HJD-2450000 どちらでも可)
- tfix は annual parallax の基準時刻。terrestrial parallax は絶対時刻から GAST を計算するので tfix に依存しない
- 球面地球近似 (R_⊕/AU ≈ 4.26e-5)、GAST ≈ GMST（章動無視、誤差 < 1 arcsec）
