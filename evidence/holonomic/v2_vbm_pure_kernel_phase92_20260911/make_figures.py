#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import platform
import subprocess
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

SPEED_PATH = Path('/tmp/lcbinint_pure_kernel_confirm_full_20260909/merged_final/results.json')
REF_ROOT = Path('/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/plot_parts_vbm_reltol_1e-6_reference')
ROOT = Path(__file__).resolve().parent
V2_PATH = ROOT / 'v2_results.tsv'
OUT = ROOT
FIG_DIR = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_INDICES = [0, 7, 15, 23]
MIN_CELL_POPULATION = 8


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_speed():
    return json.loads(SPEED_PATH.read_text())['results']


def load_reference():
    rows = []
    for path in sorted(REF_ROOT.glob('*/results.json')):
        rows.extend(json.loads(path.read_text())['results'])
    out = {}
    for row in rows:
        key = (int(row['case_id']), str(row['profile']), float(row['s']),
               float(row['q']), float(row['rho']), float(row['x']), float(row['y']))
        # The reference run contains the same 1e-6 values for both target labels.
        if key in out and out[key]['reference'] != row['reference']:
            raise RuntimeError(f'reference mismatch for {key}')
        out[key] = row
    return out


def load_v2():
    rows = []
    with V2_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            v = line.split()
            if len(v) != 22:
                raise RuntimeError(f'V2 row has {len(v)} columns')
            rows.append({
                'case_id': int(v[0]),
                'configuration_id': int(v[1]),
                'profile': v[2],
                'd_bin_index': int(v[3]),
                'epoch_index': int(v[4]),
                's': float(v[5]), 'q': float(v[6]), 'rho': float(v[7]),
                'x': float(v[8]), 'y': float(v[9]), 'time': float(v[10]),
                'u': float(v[11]), 'X': float(v[12]),
                'input_reference': float(v[13]),
                'v2_mu': float(v[14]), 'F0': float(v[15]),
                'F_half': float(v[16]), 'status_code': int(v[17]),
                'status': v[18], 'ms': float(v[19]),
                'input_ratio': float(v[20]),
                'input_relative_error': float(v[21]),
            })
    return rows


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def join_rows():
    speed = load_speed()
    ref = load_reference()
    v2 = load_v2()
    v2_by_key = {}
    for row in v2:
        key = (row['case_id'], row['profile'], row['d_bin_index'], row['epoch_index'])
        if key in v2_by_key:
            raise RuntimeError(f'duplicate V2 key {key}')
        v2_by_key[key] = row

    out = []
    missing = []
    ref_mismatch = []
    for sr in speed:
        base = (int(sr['case_id']), str(sr['profile']), float(sr['s']),
                float(sr['q']), float(sr['rho']), float(sr['x']), float(sr['y']))
        rr = ref.get(base)
        if rr is None:
            raise RuntimeError(f'missing 1e-6 reference {base}')
        d_idx = int(sr['d_bin_index'])
        selected = sr['vbm']['selected_seconds']
        if len(selected) != len(REFERENCE_INDICES):
            raise RuntimeError('unexpected VBM timing length')
        for epoch_pos, epoch_index in enumerate(REFERENCE_INDICES):
            vr = v2_by_key.get((int(sr['case_id']), str(sr['profile']), d_idx, epoch_index))
            if vr is None:
                missing.append((sr['case_id'], sr['profile'], d_idx, epoch_index))
                continue
            reference = float(rr['reference'][epoch_pos])
            if abs(vr['input_reference'] - reference) > 5e-13 * max(1.0, abs(reference)):
                ref_mismatch.append((base, epoch_pos, vr['input_reference'], reference))
            v2_ms = vr['ms']
            vbm_ms = float(selected[epoch_pos]) * 1000.0
            v2_mu = vr['v2_mu']
            relerr = (abs(v2_mu - reference) / abs(reference)
                      if finite(v2_mu) and finite(reference) and reference != 0.0
                      else float('nan'))
            speed_ratio_all = (vbm_ms / v2_ms
                               if finite(vbm_ms) and finite(v2_ms) and v2_ms > 0.0
                               else float('nan'))
            speed_ratio_ok = (speed_ratio_all
                              if vr['status'] == 'OK' else float('nan'))
            out.append({
                'case_id': int(sr['case_id']),
                'configuration_id': int(sr.get('configuration_id', sr['case_id'])),
                'profile': str(sr['profile']),
                'target': float(sr['target']),
                'd_bin_index': d_idx,
                'requested_factor': float(sr.get('requested_factor', sr.get('d_over_rho', float('nan')))),
                'actual_d_over_rho': float(sr.get('actual_d_over_rho', float('nan'))),
                'epoch_index': int(epoch_index),
                's': float(sr['s']), 'q': float(sr['q']), 'rho': float(sr['rho']),
                'x': float(sr['x']), 'y': float(sr['y']),
                'reference_1e-6': reference,
                'v2_mu': v2_mu,
                'v2_F0': vr['F0'], 'v2_F_half': vr['F_half'],
                'v2_status_code': vr['status_code'], 'v2_status': vr['status'],
                'v2_ms': v2_ms, 'vbm_ms': vbm_ms,
                'speed_ratio_vbm_over_v2_all_finite': speed_ratio_all,
                'speed_ratio_vbm_over_v2_status_ok': speed_ratio_ok,
                'relative_error_v2_over_vbm_1e-6': relerr,
            })
    if missing:
        raise RuntimeError(f'missing V2 rows: {missing[:3]} ({len(missing)} total)')
    if ref_mismatch:
        raise RuntimeError(f'V2 input reference mismatch: {ref_mismatch[:2]}')
    return out, {'speed_rows': len(speed), 'reference_rows': len(ref),
                 'v2_rows': len(v2), 'joined_rows': len(out)}


def percentile(values, p):
    values = np.asarray([x for x in values if finite(x)], dtype=float)
    if values.size == 0:
        return None
    return float(np.percentile(values, p))


def stats(rows):
    out = {}
    for profile in sorted({r['profile'] for r in rows}):
        for target in sorted({r['target'] for r in rows}):
            g = [r for r in rows if r['profile'] == profile and r['target'] == target]
            ok = [r for r in g if r['v2_status'] == 'OK']
            ratios_all = [r['speed_ratio_vbm_over_v2_all_finite'] for r in g]
            ratios_ok = [r['speed_ratio_vbm_over_v2_status_ok'] for r in g]
            ratios_positive = [r['speed_ratio_vbm_over_v2_all_finite'] for r in g
                               if finite(r['speed_ratio_vbm_over_v2_all_finite'])
                               and r['speed_ratio_vbm_over_v2_all_finite'] > 0.0
                               and finite(r['v2_mu']) and r['v2_mu'] > 0.0]
            errors = [r['relative_error_v2_over_vbm_1e-6'] for r in g]
            errors_ok = [r['relative_error_v2_over_vbm_1e-6'] for r in ok]
            ms = [r['v2_ms'] for r in g]
            out[f'{profile}:target={target:g}'] = {
                'rows': len(g), 'status_counts': dict(Counter(r['v2_status'] for r in g)),
                'status_ok': len(ok), 'status_ok_fraction': len(ok) / len(g),
                'v2_ms': {'p50': percentile(ms, 50), 'p90': percentile(ms, 90),
                          'p95': percentile(ms, 95), 'p99': percentile(ms, 99),
                          'max': percentile(ms, 100)},
                'speed_ratio_vbm_over_v2_all_finite': {
                    'p50': percentile(ratios_all, 50), 'p90': percentile(ratios_all, 90),
                    'win_fraction_ratio_gt_1': (sum(x > 1 for x in ratios_all if finite(x)) /
                                                sum(finite(x) for x in ratios_all)),
                },
                'speed_ratio_vbm_over_v2_positive': {
                    'count': len(ratios_positive),
                    'p50': percentile(ratios_positive, 50),
                    'p90': percentile(ratios_positive, 90),
                    'p95': percentile(ratios_positive, 95),
                    'p99': percentile(ratios_positive, 99),
                    'max': percentile(ratios_positive, 100),
                    'win_fraction_ratio_gt_1': (sum(x > 1 for x in ratios_positive) /
                                                len(ratios_positive)
                                                if ratios_positive else None),
                },
                'speed_ratio_vbm_over_v2_status_ok': {
                    'p50': percentile(ratios_ok, 50), 'p90': percentile(ratios_ok, 90),
                    'p95': percentile(ratios_ok, 95), 'p99': percentile(ratios_ok, 99),
                    'max': percentile(ratios_ok, 100),
                    'win_fraction_ratio_gt_1': (sum(x > 1 for x in ratios_ok if finite(x)) /
                                                sum(finite(x) for x in ratios_ok)),
                },
                'relative_error_v2_over_vbm_1e-6_all_statuses': {
                    'p50': percentile(errors, 50), 'p95': percentile(errors, 95),
                    'p99': percentile(errors, 99), 'max': percentile(errors, 100)},
                'relative_error_v2_over_vbm_1e-6_status_ok': {
                    'p50': percentile(errors_ok, 50), 'p95': percentile(errors_ok, 95),
                    'p99': percentile(errors_ok, 99), 'max': percentile(errors_ok, 100)},
            }
    return out


SPEED_BOUNDS = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 65]
SPEED_COLORS = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7',
                '#ef8a62', '#d73027', '#b2182b', '#7f0000', '#4d0010']
SPEED_LABELS = ['< 0.25 ×', '0.25 ×', '0.5 ×', '1 ×', '2 ×', '4 ×',
                '8 ×', '16 ×', '32 ×', '≥ 64 ×']
ERROR_BOUNDS = [0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1.1e-2]
ERROR_COLORS = ['#006d2c', '#31a354', '#74c476', '#d9f0a3', '#fee08b',
                '#fdae61', '#f46d43', '#c5002f']
ERROR_LABELS = ['≤ 10⁻⁴', '2 × 10⁻⁴', '5 × 10⁻⁴', '10⁻³',
                '2 × 10⁻³', '5 × 10⁻³', '10⁻²', '≥ 10⁻²']


def edges_for(kind):
    if kind == 'q':
        return np.logspace(-4, 0, 13), r'$q$'
    if kind == 'rho':
        return np.logspace(-4, 0, 11), r'$\rho$'
    if kind == 'A':
        return np.logspace(0, 5, 13), r'$A_{\rm fs}$'
    raise ValueError(kind)


def axis_data(r, kind):
    return float(r['q'] if kind == 'q' else r['rho'] if kind == 'rho' else r['reference_1e-6'])


def cell_stat(group, xkind, ykind, value_key, reducer):
    xe, _ = edges_for(xkind)
    ye, _ = edges_for(ykind)
    vals = [[[] for _ in range(len(xe) - 1)] for _ in range(len(ye) - 1)]
    for r in group:
        x = axis_data(r, xkind); y = axis_data(r, ykind)
        if not (finite(x) and finite(y) and x > 0.0 and y > 0.0):
            continue
        ix = int(np.searchsorted(xe, x, side='right') - 1)
        iy = int(np.searchsorted(ye, y, side='right') - 1)
        if 0 <= ix < len(xe) - 1 and 0 <= iy < len(ye) - 1:
            v = r[value_key]
            if finite(v):
                vals[iy][ix].append(float(v))
    z = np.full((len(ye) - 1, len(xe) - 1), np.nan)
    counts = np.zeros_like(z, dtype=int)
    for iy, row in enumerate(vals):
        for ix, v in enumerate(row):
            counts[iy, ix] = len(v)
            if len(v) >= MIN_CELL_POPULATION:
                z[iy, ix] = reducer(v)
    return xe, ye, z, counts


def all_counts(group, xkind, ykind):
    xe, _ = edges_for(xkind); ye, _ = edges_for(ykind)
    total = np.zeros((len(ye)-1, len(xe)-1), dtype=int)
    ok = np.zeros_like(total)
    for r in group:
        x = axis_data(r, xkind); y = axis_data(r, ykind)
        if not (finite(x) and finite(y) and x > 0 and y > 0):
            continue
        ix = int(np.searchsorted(xe, x, side='right') - 1)
        iy = int(np.searchsorted(ye, y, side='right') - 1)
        if 0 <= ix < total.shape[1] and 0 <= iy < total.shape[0]:
            total[iy, ix] += 1
            if r['v2_status'] == 'OK':
                ok[iy, ix] += 1
    return total, ok


def format_target(target):
    return '10^{-3}' if abs(target - 1e-3) < 1e-12 else '10^{-4}'


def draw_axis(ax, xkind, ykind, z, title, cmap, norm, cbar_kind, counts_total, counts_ok):
    xe, xlabel = edges_for(xkind)
    ye, ylabel = edges_for(ykind)
    cm = cmap.copy()
    cm.set_bad('#d9d9d9')
    im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(z), cmap=cm, norm=norm,
                       shading='flat', edgecolors='#9a9a9a', linewidth=0.35)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, pad=10)
    ax.tick_params(labelsize=10)
    # A small corner mark identifies cells whose valid status coverage is incomplete.
    for iy in range(z.shape[0]):
        for ix in range(z.shape[1]):
            if counts_total[iy, ix] >= MIN_CELL_POPULATION and counts_ok[iy, ix] < counts_total[iy, ix]:
                x0, x1 = xe[ix], xe[ix+1]
                y0, y1 = ye[iy], ye[iy+1]
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                       hatch='//', edgecolor='#555555', linewidth=0.0, alpha=0.23))
    return im


def make_figure(rows, profile, target):
    group = [r for r in rows if r['profile'] == profile and abs(r['target'] - target) < 1e-15]
    ok = [r for r in group if r['v2_status'] == 'OK']
    # Match the established paper-style map: retain finite positive V2
    # timings/value outputs in the map, and expose status delivery separately.
    speed_group = [r for r in group
                   if finite(r['speed_ratio_vbm_over_v2_all_finite'])
                   and r['speed_ratio_vbm_over_v2_all_finite'] > 0.0
                   and finite(r['v2_mu']) and r['v2_mu'] > 0.0]
    error_group = [r for r in group
                   if finite(r['relative_error_v2_over_vbm_1e-6'])]
    if not group:
        raise RuntimeError(f'no rows for {profile} {target}')
    speed_cmap = ListedColormap(SPEED_COLORS, name='speed')
    err_cmap = ListedColormap(ERROR_COLORS, name='error')
    speed_norm = BoundaryNorm(SPEED_BOUNDS, speed_cmap.N)
    err_norm = BoundaryNorm(ERROR_BOUNDS, err_cmap.N)

    fig = plt.figure(figsize=(18, 10.5), dpi=180)
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.84, bottom=0.20,
                          hspace=0.98, wspace=0.28)
    specs = [('q', 'rho', r'$q \times \rho$'),
             ('A', 'rho', r'$A_{\rm fs} \times \rho$'),
             ('A', 'q', r'$A_{\rm fs} \times q$')]
    top_ims = []
    for j, (xkind, ykind, label) in enumerate(specs):
        xe, ye, z, counts = cell_stat(speed_group, xkind, ykind,
                                       'speed_ratio_vbm_over_v2_all_finite', np.median)
        total, ok_counts = all_counts(group, xkind, ykind)
        ax = fig.add_subplot(gs[0, j])
        top_ims.append(draw_axis(ax, xkind, ykind, z, label + ' — runtime',
                                  speed_cmap, speed_norm, 'speed', total, ok_counts))
    # Top colorbar is shared and uses the same bins as the earlier paper-style figure.
    cax1 = fig.add_axes([0.055, 0.51, 0.93, 0.037])
    cb1 = fig.colorbar(top_ims[0], cax=cax1, orientation='horizontal',
                       boundaries=SPEED_BOUNDS,
                       ticks=[0.125, 0.375, 0.75, 1.5, 3, 6, 12, 24, 48, 64.5])
    cb1.set_ticklabels(SPEED_LABELS)
    cb1.ax.tick_params(labelsize=11, length=3)
    cb1.set_label(r'median $R=t_{\rm VBM}/t_{\rm V2}$  ( $R>1$: V2 faster )', fontsize=13, labelpad=7)

    bottom_ims = []
    for j, (xkind, ykind, label) in enumerate(specs):
        xe, ye, z, counts = cell_stat(error_group, xkind, ykind,
                                       'relative_error_v2_over_vbm_1e-6', lambda v: np.percentile(v, 95))
        total, ok_counts = all_counts(group, xkind, ykind)
        ax = fig.add_subplot(gs[1, j])
        bottom_ims.append(draw_axis(ax, xkind, ykind, z, label + ' — p95 difference',
                                    err_cmap, err_norm, 'error', total, ok_counts))
    cax2 = fig.add_axes([0.055, 0.065, 0.93, 0.037])
    cb2 = fig.colorbar(bottom_ims[0], cax=cax2, orientation='horizontal',
                       boundaries=ERROR_BOUNDS,
                       ticks=[5e-5, 1.5e-4, 3.5e-4, 7.5e-4, 1.5e-3, 3.5e-3, 7.5e-3, 1.05e-2])
    cb2.set_ticklabels(ERROR_LABELS)
    cb2.ax.tick_params(labelsize=11, length=3)
    cb2.set_label(r'cell p95 $|A_{\rm V2}-A_{\rm VBM,1e-6}|/|A_{\rm VBM,1e-6}|$', fontsize=13, labelpad=0)

    ld = 'LD off (uniform)' if profile == 'uniform' else 'LD on (linear, c=0.5)'
    valid_fraction = len(ok) / len(group)
    fig.suptitle(f'{ld}, VBM speed target $\\epsilon_{{rel}}={format_target(target)}$', fontsize=20, y=0.965)
    fig.text(0.5, 0.905,
             f'current V2: n_r=64, flux_value_integrate body only; pf/topology prebuilt; '
             f'V2 status OK {len(ok)}/{len(group)} ({100*valid_fraction:.1f}%), '
             f'map keeps finite positive V2 outputs; hatched cells contain at least one non-OK V2 point',
             ha='center', va='center', fontsize=11)
    fig.text(0.5, 0.135,
             'Grey cells: fewer than 8 mapped points. Error uses all finite V2 values against the existing VBM RelTol=1e-6 reference.',
             ha='center', va='center', fontsize=10)
    stem = f'q_rho_{profile}_{"1e-3" if target == 1e-3 else "1e-4"}_v2_current_vbm1e-6'
    png = FIG_DIR / f'{stem}.png'
    pdf = FIG_DIR / f'{stem}.pdf'
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf, {'rows': len(group), 'status_ok': len(ok), 'status_ok_fraction': valid_fraction}


def write_joined(rows):
    path = OUT / 'joined_v2_vbm.tsv'
    fields = list(rows[0].keys())
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t',
                           lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def main():
    rows, counts = join_rows()
    joined_path = write_joined(rows)
    figures = []
    for profile in ['uniform', 'linear']:
        for target in [1e-3, 1e-4]:
            png, pdf, meta = make_figure(rows, profile, target)
            figures.append({'profile': profile, 'target': target,
                            'png': str(png), 'pdf': str(pdf), **meta})
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='/rogue1_8/nunota/lcbinint', text=True).strip()
    except Exception:
        head = None
    summary = {
        'purpose': 'Fair four-way pure-kernel comparison using current V2 flux_value_integrate and existing VBM direct-kernel timing; error against existing VBM RelTol=1e-6 reference.',
        'v2_commit': head,
        'v2_runner': '/tmp/v2_pure_kernel_phase92_runner',
        'v2_runner_sha256': sha256(Path('/tmp/v2_pure_kernel_phase92_runner')),
        'inputs': {
            'v2_results': str(V2_PATH),
            'v2_results_sha256': sha256(V2_PATH),
            'vbm_speed_results': str(SPEED_PATH),
            'vbm_speed_results_sha256': sha256(SPEED_PATH),
            'vbm_reference_root': str(REF_ROOT),
        },
        'matching': {
            **counts,
            'reference_indices': REFERENCE_INDICES,
            'v2_calls_added': counts['v2_rows'],
            'speed_comparison': 'VBM RelTol=1e-3 and 1e-4, uniform and linear profiles',
            'error_reference': 'VBM RelTol=1e-6 reference values',
            'v2_timing_scope': 'flux_value_integrate(64,u,pf,topo) body; LensParams, PrimaryFrame, classify_cells (D14+topology), epoch_value_blend, and TSV parsing/output excluded; no epoch_value' ,
            'profiles': {'uniform': 'LD off, c=0.0', 'linear': 'LD on, c=0.5'},
        },
        'plot': {
            'min_cell_population': MIN_CELL_POPULATION,
            'speed_ratio': 'VBM selected_seconds * 1000 / V2 ms; finite positive V2 outputs enter the paper-style runtime cells, regardless of status',
            'error': 'abs(V2 mu - VBM 1e-6 reference) / abs(VBM 1e-6 reference); all finite V2 values enter the paper-style p95 cells',
            'nonok_diagnostic': 'non-OK points are retained in joined TSV and status_counts; hatched cells identify incomplete status coverage',
        },
        'system': {'platform': platform.platform(), 'python': platform.python_version()},
        'figures': figures,
        'summary_by_condition': stats(rows),
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = []
    report.append(f'# Current V2 pure-kernel vs existing VBM four-way comparison (HEAD `{head}`)\n')
    report.append('The four speed conditions are VBM `RelTol=1e-3` and `RelTol=1e-4`, each for uniform (LD off) and linear (LD on, `c=0.5`). Only current V2 was evaluated in this step. Relative error is evaluated against the existing VBM `RelTol=1e-6` reference.\n')
    report.append('V2 timing covers the value-only `flux_value_integrate(64,u,pf,topo)` path; no gradient/Jacobian is requested or timed. `LensParams`, `PrimaryFrame`, `classify_cells` (D14+topology), `epoch_value_blend`, input parsing, and output formatting are outside the timer; no `epoch_value` or full epoch is timed. The existing VBM timing is the warmed direct `BinaryMag`/`BinaryMagDark` kernel.\n')
    report.append('Non-OK V2 points remain in the machine-readable join and status counts. The paper-style runtime map keeps finite positive V2 outputs, while the p95 error map keeps all finite V2 outputs; hatched cells identify incomplete status coverage.\n')
    report.append('## Conditions\n')
    report.append('| profile | VBM timing target | rows | V2 OK | V2 p50 ms | V2 p95 ms | median VBM/V2 (finite positive) | median VBM/V2 (OK) | all p95 error | OK p95 error |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for key, s in summary['summary_by_condition'].items():
        prof, targettxt = key.split(':target=')
        report.append(f"| {prof} | {targettxt} | {s['rows']} | {s['status_ok']} ({100*s['status_ok_fraction']:.1f}%) | {s['v2_ms']['p50']:.6g} | {s['v2_ms']['p95']:.6g} | {s['speed_ratio_vbm_over_v2_positive']['p50']:.6g} | {s['speed_ratio_vbm_over_v2_status_ok']['p50']:.6g} | {s['relative_error_v2_over_vbm_1e-6_all_statuses']['p95']:.6g} | {s['relative_error_v2_over_vbm_1e-6_status_ok']['p95']:.6g} |")
    report.append('\n## Reproduction\n')
    report.append('The V2 runner was compiled at the current branch HEAD and run as:\n\n```bash\n/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 -I/rogue1_8/nunota/lcbinint/src -march=native -funroll-loops -ffp-contract=fast -fno-math-errno evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/v2_pure_kernel_runner.cpp -o /tmp/v2_pure_kernel_phase92_runner -lquadmath\n/tmp/v2_pure_kernel_phase92_runner evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/input_snapshot.tsv /tmp/v2_pure_kernel_phase92_results.tsv\ncp /tmp/v2_pure_kernel_phase92_results.tsv evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/v2_results.tsv\npython3 evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/make_figures.py\n```\n')
    report.append('## Figures\n')
    for f in figures:
        report.append(f"- `{f['png']}`\n- `{f['pdf']}`")
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({'out': str(OUT), 'joined': str(joined_path), 'figures': figures,
                      'summary_by_condition': summary['summary_by_condition']}, indent=2))


if __name__ == '__main__':
    main()
