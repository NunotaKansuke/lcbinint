# Terrestrial Parallax Implementation Plan

Date: 2026-06-29

## 概要

地上視差 (terrestrial parallax) を実装する。望遠鏡の緯度経度を指定すると、
annual parallax に加えて terrestrial parallax 補正が加わる。

## 物理

年周視差 (annual parallax) は地球の公転運動による地心位置のずれを πE でスケールした補正:
  δu_annual = πE × (r_geocenter(t) - r_geocenter(t0par) - v(t0par)(t-t0par)) / AU

地上視差は望遠鏡の地心からのずれを同じ πE でスケールした補正:
  δu_terrestrial = πE × r_tel(t) / AU  (projected onto sky plane)

両者は同じ piEN, piEE パラメータを共有し、必ず annual parallax と同時に有効になる。

望遠鏡の地心位置 (赤道座標 J2000):
  r_tel_x = R_⊕ cos(lat) cos(GAST + lon)
  r_tel_y = R_⊕ cos(lat) sin(GAST + lon)
  r_tel_z = R_⊕ sin(lat)
  R_⊕ / AU ≈ 4.2635×10⁻⁵ (球面近似)

GAST ≈ GMST = 280.46061837° + 360.98564736629° × (JD - 2451545.0)
(歳差章動は無視、誤差 < 1 arcsec)

スカイ面への投影は annual parallax と同じ project() ロジックを使う:
  proj_N = −(r_tel · ê_N)   (ê_N = sky_north)
  proj_E = −(r_tel · ê_E)   (ê_E = sky_east)

τ   += piEN * proj_N + piEE * proj_E
β   += −piEE * proj_N + piEN * proj_E

## 有効条件

- has_annual_parallax(params) が true (piEN ≠ 0 or piEE ≠ 0)
- obs_lat ≠ 0 or obs_lon ≠ 0 (地心 = (0,0) なら無補正と扱う)

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| include/lcbinint/lcbinint.h | lcbi_params に obs_lat, obs_lon を追加 |
| src/lcbinint/model/lens_parameters.hpp | LensParameters に obs_lat, obs_lon を追加 |
| src/lcbinint/model/lens_parameters.cpp | from_c_params でコピー |
| src/lcbinint/model/trajectory.cpp | apply_terrestrial_parallax() 関数を追加、source_position で呼び出し |
| python/lcbinint_pybind.cpp | params dict で obs_lat, obs_lon を受け付ける |
| tests/regression/test_parallax.py (新規) | 地上視差のテスト |

## テスト方針

1. piEN=0.1, piEE=0 の状況で obs_lat/obs_lon を変えた2点の差が手計算値と一致するか
2. obs_lat=obs_lon=0 のとき annual parallax only と同じ結果になるか
3. 同一観測時刻を2つの望遠鏡 (異なる lat/lon) で計算し、差が理論値と合うか

## ログ

- 2026-06-29: 計画作成、コード調査完了
  - trajectory.cpp: apply_annual_parallax → EarthOrbitalParallaxProjector 経由
  - project() が符号反転 (-dot) することを確認
  - sky_north/sky_east 関数が既に実装済み

- 2026-06-29: C++ 実装完了、ビルド成功
  - include/lcbinint/lcbinint.h: lcbi_params に obs_lat, obs_lon 追加
  - src/lcbinint/model/lens_parameters.hpp/.cpp: LensParameters に obs_lat, obs_lon 追加
  - src/lcbinint/model/trajectory.cpp: apply_terrestrial_parallax() 追加
    - GMST = 280.46061837 + 360.98564736629 * (JD - 2451545.0) 度 で GAST 近似
    - r_tel = R_⊕/AU * (cos(lat)*cos(GAST+lon), cos(lat)*sin(GAST+lon), sin(lat))
    - proj_N/proj_E = -dot(r_tel, sky_north/east) (annual parallax と同符号規則)
    - source_position() から apply_annual_parallax の直後に呼び出し
  - python/lcbinint_pybind.cpp:
    - PyEventCoordinates に obs_lat, obs_lon 追加
    - EventCoordinates Python クラスのコンストラクタと def_readwrite に追加
    - lcbi_params Python binding に obs_lat, obs_lon 追加
    - make_binary_params で params.obs_lat/obs_lon = event.obs_lat/obs_lon を設定
  - 次: テスト追加

- 2026-06-29: テスト追加、全テスト通過
  - tests/regression/test_terrestrial_parallax.py 新規追加 (3テスト)
    1. test_terrestrial_zero_obs_same_as_geocentric: (0,0) は geocenter と同一
    2. test_terrestrial_inter_observatory_displacement: 2望遠鏡間の source 位置差が理論値と一致
    3. test_terrestrial_diurnal_variation: 日周変動 (geocenter との差) が理論値と一致
  - full suite: 110 passed, 2 errors (既知の診断スイート errors)
  - 実装完了
