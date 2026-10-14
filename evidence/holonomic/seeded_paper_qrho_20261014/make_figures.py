#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

V2_PATH = ROOT / 'raw_results.tsv'
INPUT_SNAPSHOT = ROOT / 'input_snapshot.tsv'
VBM_JOIN_PATH = ROOT / 'vbm_timing_reference.tsv'
OUT = ROOT
FIG_DIR = OUT / 'figures'
REFERENCE_INDICES = [0, 7, 15, 23]
MIN_CELL_POPULATION = 8
Q_RHO_EDGE_SPEC = {
    'q': 'geomspace(1e-4, 1, 13)',
    'rho': 'geomspace(3e-5, 1, 13)',
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_v2():
    rows = []
    with V2_PATH.open(newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            for name in ('case_id', 'configuration_id', 'd_bin_index', 'epoch_index',
                         'seed_count', 'center_seed_count', 'reps',
                         'value_converged_reps', 'radius_probes', 'bands',
                         'skipped_seeds', 'nodes', 'evaluations', 'reused_nodes',
                         'panels', 'splits', 'local_repairs', 'retained_nodes',
                         'evaluated_nodes', 'event_topology_reuses',
                         'event_radius_reuses', 'event_qf_refinements'):
                if name in row and row[name] != '':
                    row[name] = int(float(row[name]))
            for name in ('target', 's', 'q', 'rho', 'x', 'y', 'time', 'u', 'X',
                         'reference_1e-6', 'caustic_prebuild_ms', 'seed_ms',
                         'planning_ms', 'integration_ms', 'ms', 'v2_mu',
                         'value_error', 'setup_ms', 'physics_ms', 'estimator_ms',
                         'scheduler_ms', 'setup_event_ms'):
                if name in row and row[name] != '':
                    row[name] = float(row[name])
            for name in ('support_proven', 'coverage_certified', 'value_converged'):
                row[name] = int(row[name])
            rows.append(row)
    return rows


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def join_rows():
    base_rows = []
    with VBM_JOIN_PATH.open(newline='') as f:
        base_rows = list(csv.DictReader(f, delimiter='\t'))
    baseline = {}
    for row in base_rows:
        key = (int(row['case_id']), row['profile'], int(row['d_bin_index']),
               int(row['epoch_index']), float(row['target']))
        if key in baseline:
            raise RuntimeError(f'duplicate saved VBM key {key}')
        baseline[key] = row
    v2 = load_v2()
    v2_by_key = {}
    for row in v2:
        key = (row['case_id'], row['profile'], row['d_bin_index'],
               row['epoch_index'], row['target'])
        if key in v2_by_key:
            raise RuntimeError(f'duplicate seeded key {key}')
        v2_by_key[key] = row

    out = []
    ref_mismatch = []
    base_keys = set(baseline)
    current_keys = set(v2_by_key)
    if current_keys != base_keys:
        missing = sorted(base_keys - current_keys)
        extra = sorted(current_keys - base_keys)
        raise RuntimeError(f'key mismatch: missing={missing[:3]} ({len(missing)}), '
                           f'extra={extra[:3]} ({len(extra)})')
    for key, vr in v2_by_key.items():
        sr = baseline[key]
        for field in ('s', 'q', 'rho', 'x', 'y'):
            if abs(float(vr[field]) - float(sr[field])) > 5e-13 * max(1.0, abs(float(sr[field]))):
                ref_mismatch.append((key, field, vr[field], sr[field]))
        reference = float(sr['reference_1e-6'])
        if abs(float(vr['reference_1e-6']) - reference) > 5e-13 * max(1.0, abs(reference)):
            ref_mismatch.append((key, 'reference', vr['reference_1e-6'], reference))
        v2_ms = float(vr['ms'])
        vbm_ms = float(sr['vbm_ms'])
        v2_mu = float(vr['v2_mu'])
        relerr = (abs(v2_mu - reference) / abs(reference)
                  if finite(v2_mu) and finite(reference) and reference != 0.0
                  else float('nan'))
        speed_ratio_all = (vbm_ms / v2_ms
                           if finite(vbm_ms) and finite(v2_ms) and v2_ms > 0.0
                           else float('nan'))
        speed_ratio_ok = speed_ratio_all if vr['status'] == 'OK' else float('nan')
        out.append({
            'case_id': int(vr['case_id']),
            'configuration_id': int(vr['configuration_id']),
            'profile': str(vr['profile']),
            'target': float(vr['target']),
            'd_bin_index': int(vr['d_bin_index']),
            'requested_factor': float(sr['requested_factor']),
            'actual_d_over_rho': float(sr['actual_d_over_rho']),
            'epoch_index': int(vr['epoch_index']),
            's': float(vr['s']), 'q': float(vr['q']), 'rho': float(vr['rho']),
            'x': float(vr['x']), 'y': float(vr['y']),
            'reference_1e-6': reference,
            'v2_mu': v2_mu,
            'v2_value_error': float(vr['value_error']),
            'v2_value_converged': int(vr['value_converged']),
            'v2_stop': vr['stop'], 'v2_value_stop': vr['value_stop'],
            'v2_numerical_status': vr['numerical_status'],
            'v2_frontier_reason': vr['frontier_reason'],
            'v2_status_code': -1,
            'v2_support_proven': int(vr['support_proven']),
            'v2_coverage_certified': int(vr['coverage_certified']),
            'v2_seed_count': int(vr['seed_count']),
            'v2_center_seed_count': int(vr['center_seed_count']),
            'v2_caustic_prebuild_ms': float(vr['caustic_prebuild_ms']),
            'v2_seed_ms': float(vr['seed_ms']),
            'v2_planning_ms': float(vr['planning_ms']),
            'v2_integration_ms': float(vr['integration_ms']),
            'v2_status': str(vr['status']),
            'v2_ms': v2_ms, 'vbm_ms': vbm_ms,
            'v2_nodes': int(vr['nodes']), 'v2_evaluations': int(vr['evaluations']),
            'v2_reused_nodes': int(vr['reused_nodes']),
            'v2_panels': int(vr['panels']), 'v2_splits': int(vr['splits']),
            'v2_setup_ms': float(vr['setup_ms']),
            'v2_physics_ms': float(vr['physics_ms']),
            'v2_estimator_ms': float(vr['estimator_ms']),
            'v2_scheduler_ms': float(vr['scheduler_ms']),
            'v2_setup_event_ms': float(vr['setup_event_ms']),
            'v2_event_topology_reuses': int(vr['event_topology_reuses']),
            'v2_event_radius_reuses': int(vr['event_radius_reuses']),
            'v2_event_qf_refinements': int(vr['event_qf_refinements']),
            'speed_ratio_vbm_over_v2_all_finite': speed_ratio_all,
            'speed_ratio_vbm_over_v2_status_ok': speed_ratio_ok,
            'relative_error_v2_over_vbm_1e-6': relerr,
        })
    if ref_mismatch:
        raise RuntimeError(f'input/reference metadata mismatch: {ref_mismatch[:3]} '
                           f'({len(ref_mismatch)} total)')
    return out, {'saved_vbm_rows': len(base_rows), 'seeded_rows': len(v2),
                 'joined_rows': len(out)}


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
                'numerical_status_counts': dict(Counter(r['v2_numerical_status'] for r in g)),
                'value_stop_counts': dict(Counter(r['v2_value_stop'] for r in g)),
                'seed_support_proven': sum(r['v2_support_proven'] == 1 for r in g),
                'coverage_certified': sum(r['v2_coverage_certified'] == 1 for r in g),
                'status_ok': len(ok), 'status_ok_fraction': len(ok) / len(g),
                'adaptive_nodes': {
                    'p50': percentile([r['v2_nodes'] for r in g], 50),
                    'p90': percentile([r['v2_nodes'] for r in g], 90),
                    'p95': percentile([r['v2_nodes'] for r in g], 95),
                    'p99': percentile([r['v2_nodes'] for r in g], 99),
                    'max': percentile([r['v2_nodes'] for r in g], 100),
                },
                'adaptive_evaluations': {
                    'p50': percentile([r['v2_evaluations'] for r in g], 50),
                    'p95': percentile([r['v2_evaluations'] for r in g], 95),
                    'max': percentile([r['v2_evaluations'] for r in g], 100),
                },
                'adaptive_timing_ms': {
                    'setup_p50': percentile([r['v2_setup_ms'] for r in g], 50),
                    'physics_p50': percentile([r['v2_physics_ms'] for r in g], 50),
                    'estimator_p50': percentile([r['v2_estimator_ms'] for r in g], 50),
                    'scheduler_p50': percentile([r['v2_scheduler_ms'] for r in g], 50),
                    'setup_event_p50': percentile([r['v2_setup_event_ms'] for r in g], 50),
                },
                'stage_timing_ms': {
                    'caustic_prebuild_excluded_p50': percentile([r['v2_caustic_prebuild_ms'] for r in g], 50),
                    'seed_p50': percentile([r['v2_seed_ms'] for r in g], 50),
                    'planning_p50': percentile([r['v2_planning_ms'] for r in g], 50),
                    'integration_p50': percentile([r['v2_integration_ms'] for r in g], 50),
                },
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
ERROR_BOUNDS = [0, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
ERROR_COLORS = ['#006d2c', '#31a354', '#74c476', '#d9f0a3', '#fee08b',
                '#fdae61', '#f46d43', '#c5002f']
ERROR_LABELS = ['≤ 10⁻⁸', '10⁻⁷', '10⁻⁶', '3 × 10⁻⁶',
                '10⁻⁵', '3 × 10⁻⁵', '10⁻⁴', '≥ 3 × 10⁻⁴']


def edges_for(kind):
    if kind == 'q':
        return np.geomspace(1e-4, 1, 13), r'$q$'
    if kind == 'rho':
        return np.geomspace(3e-5, 1, 13), r'$\rho$'
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


def q_rho_population(rows):
    """Return the canonical q-rho population counts used by the paper panels."""
    out = {}
    for profile in ['uniform', 'linear']:
        for target in [1e-3, 1e-4]:
            group = [r for r in rows
                     if r['profile'] == profile and abs(r['target'] - target) < 1e-15]
            total, _ = all_counts(group, 'q', 'rho')
            populated = total[total >= MIN_CELL_POPULATION]
            out[f'{profile}:target={target:g}'] = {
                'cells': int(total.size),
                'populated_cells': int(populated.size),
                'empty_or_underpopulated_cells': int((total < MIN_CELL_POPULATION).sum()),
                'min_nonzero_population': int(total[total > 0].min()) if np.any(total > 0) else 0,
                'max_population': int(total.max()),
            }
    return out


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
    # A finite partial value or a row without seed-provider support remains in
    # the joined raw data, but is not presented as an accepted accuracy result.
    error_group = [r for r in group
                   if r['v2_status'] == 'OK' and r['v2_value_converged'] == 1
                   and finite(r['relative_error_v2_over_vbm_1e-6'])]
    max_accepted_error = max(
        (r['relative_error_v2_over_vbm_1e-6'] for r in error_group),
        default=float('nan'))
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
    cb1.set_label(r'median $R=t_{\rm VBM}/t_{\rm seeded}$  ( $R>1$: seeded faster )', fontsize=13, labelpad=7)

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
                       ticks=[5e-9, 5.5e-8, 5.5e-7, 2e-6,
                              6.5e-6, 2e-5, 6.5e-5, 2e-4])
    cb2.set_ticklabels(ERROR_LABELS)
    cb2.ax.tick_params(labelsize=11, length=3)
    cb2.set_label(r'cell p95 $|A_{\rm seeded}-A_{\rm VBM,1e-6}|/|A_{\rm VBM,1e-6}|$', fontsize=13, labelpad=0)

    ld = 'LD off (uniform)' if profile == 'uniform' else 'LD on (linear, c=0.5)'
    valid_fraction = len(ok) / len(group)
    fig.suptitle(f'{ld}, VBM speed target $\\epsilon_{{rel}}={format_target(target)}$', fontsize=20, y=0.965)
    fig.text(0.5, 0.905,
             f'seeded radial frontier: seed construction + radial discovery/integration; caustic build excluded\n'
             f'seed-supported and value-converged {len(ok)}/{len(group)} ({100*valid_fraction:.1f}%); '
             f'hatching marks cells with non-OK rows',
             ha='center', va='center', fontsize=10)
    fig.text(0.5, 0.135,
             'Grey: fewer than 8 samples. Error maps show cell p95 vs saved VBM RelTol=1e-6.\n'
             f'Worst pointwise error among internally OK rows: {max_accepted_error:.3g}; '
             'radial coverage is not certified.',
             ha='center', va='center', fontsize=9)
    stem = f'q_rho_{profile}_{"1e-3" if target == 1e-3 else "1e-4"}_seeded_radial_vbm1e-6'
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


def write_error_outliers(rows):
    path = OUT / 'accepted_error_outliers.tsv'
    selected = [r for r in rows
                if r['v2_status'] == 'OK'
                and finite(r['relative_error_v2_over_vbm_1e-6'])
                and r['relative_error_v2_over_vbm_1e-6'] > 1e-3]
    selected.sort(key=lambda r: r['relative_error_v2_over_vbm_1e-6'], reverse=True)
    fields = ['case_id', 'configuration_id', 'profile', 'target', 'd_bin_index',
              'epoch_index', 's', 'q', 'rho', 'x', 'y', 'v2_mu',
              'reference_1e-6', 'relative_error_v2_over_vbm_1e-6',
              'v2_seed_count', 'v2_center_seed_count', 'v2_seed_ms', 'v2_ms',
              'v2_frontier_reason', 'v2_stop', 'v2_value_stop']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t',
                                lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(selected)
    return path, selected


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows, counts = join_rows()
    joined_path = write_joined(rows)
    outlier_path, outliers = write_error_outliers(rows)
    figures = []
    for profile in ['uniform', 'linear']:
        for target in [1e-3, 1e-4]:
            png, pdf, meta = make_figure(rows, profile, target)
            figures.append({'profile': profile, 'target': target,
                            'png': str(png), 'pdf': str(pdf), **meta})

    project = ROOT.parents[2]
    runner = project / 'build-seeded-events/bench_seeded_paper_qrho'
    runner_source = project / 'benchmarks/holonomic/bench_seeded_paper_qrho.cpp'
    build_script = project / 'benchmarks/holonomic/build_seeded_paper_qrho.sh'
    try:
        head = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip()
        branch = subprocess.check_output(
            ['git', 'branch', '--show-current'], cwd=project, text=True).strip()
    except Exception:
        head = None
        branch = None

    non_ok = [r for r in rows if r['v2_status'] != 'OK']
    accepted = [r for r in rows if r['v2_status'] == 'OK'
                and finite(r['relative_error_v2_over_vbm_1e-6'])]
    error_thresholds = [1e-3, 1e-2, 0.1]
    external_error_screen = {
        'rows_internal_status_ok': len(accepted),
        'max_relative_error_among_internal_ok': max(
            (r['relative_error_v2_over_vbm_1e-6'] for r in accepted),
            default=None),
        'p99_relative_error_among_internal_ok': percentile(
            [r['relative_error_v2_over_vbm_1e-6'] for r in accepted], 99),
        'counts_above_threshold': {
            f'{threshold:g}': sum(r['relative_error_v2_over_vbm_1e-6'] > threshold
                                  for r in accepted)
            for threshold in error_thresholds
        },
        'distinct_case_epoch_geometries_above_1e-3': len({
            (r['case_id'], r['epoch_index']) for r in accepted
            if r['relative_error_v2_over_vbm_1e-6'] > 1e-3}),
        'outlier_rows_written': len(outliers),
        'outlier_threshold': 1e-3,
        'meaning': 'status OK checks seed-provider support and value convergence only; it is not a radial-support coverage certificate',
    }
    summary = {
        'purpose': 'Seeded radial-frontier benchmark on the filled q-rho paper corpus; speed vs saved direct VBM RelTol 1e-3/1e-4 timings and accuracy vs saved VBM RelTol 1e-6 references.',
        'checkout': str(project),
        'branch': branch,
        'head': head,
        'runner': str(runner),
        'runner_sha256': sha256(runner),
        'runner_source': str(runner_source),
        'runner_source_sha256': sha256(runner_source),
        'build_script': str(build_script),
        'build_script_sha256': sha256(build_script),
        'figure_script': str(Path(__file__).resolve()),
        'figure_script_sha256': sha256(Path(__file__).resolve()),
        'inputs': {
            'input_snapshot': str(INPUT_SNAPSHOT),
            'input_snapshot_sha256': sha256(INPUT_SNAPSHOT),
            'saved_vbm_timing_and_reference_join': str(VBM_JOIN_PATH),
            'saved_vbm_timing_and_reference_join_sha256': sha256(VBM_JOIN_PATH),
            'raw_seeded_results': str(V2_PATH),
            'raw_seeded_results_sha256': sha256(V2_PATH),
        },
        'matching': {
            **counts,
            'targets': [1e-3, 1e-4],
            'profiles': {'uniform': 'u=0; LD off', 'linear': 'u=0.5; linear LD c=0.5'},
            'timing_reference': 'saved warmed direct VBM BinaryMag2 (uniform) / BinaryMagDark (linear) measurements at matched tolerance; direct VBM 5.5 calls reproduce the saved low-magnification medians within 1%',
            'accuracy_reference': 'saved VBM RelTol=1e-6 magnifications',
            'timing_scope': 'center point-source roots + finite-source seed generation + epoch_seeded_radial_frontier; caustic branch construction excluded; no D14/classify_cells call; input parsing and TSV output excluded',
            'seed_stage': 'center PointSourceMagnifier::binary_images plus FiniteSourceMagnifier::cached_binary_image_seeds, including the caustic-polyline scan and additional seed roots',
            'caustic_handling': '1400-bin binary caustic branches prepared outside each timed repetition and consumed by seed generation; caustic construction time reported separately but excluded from ms',
            'full_time_definition': 'median across 3 paired repetitions of seed_ms + epoch_seeded_radial_frontier wall time',
            'frontier_coverage': 'coverage_certified is recorded; the current seeded radial frontier reports false and does not claim a universal radial-support proof',
            'adaptive_config': {
                'mu_rtol': 'matched target 1e-3 or 1e-4',
                'mu_atol': 1e-8,
                'gradient_policy': 'None',
                'with_jacobian': False,
                'reuse_samples': True,
                'repetitions': 3,
                'reducer': 'median',
            },
        },
        'status_summary': {
            'rows': len(rows),
            'ok': len(rows) - len(non_ok),
            'non_ok': len(non_ok),
            'counts': dict(Counter(r['v2_status'] for r in rows)),
            'value_stop_counts': dict(Counter(r['v2_value_stop'] for r in rows)),
            'support_proven': sum(r['v2_support_proven'] == 1 for r in rows),
            'coverage_certified': sum(r['v2_coverage_certified'] == 1 for r in rows),
        },
        'external_error_screen': external_error_screen,
        'plot': {
            'panels_per_figure': 6,
            'figures': 'one 2x3 runtime/error map for each profile x tolerance condition',
            'min_cell_population': MIN_CELL_POPULATION,
            'q_rho_edges': Q_RHO_EDGE_SPEC,
            'q_rho_bins': '12 x 12; cells require at least 8 points',
            'q_rho_population': q_rho_population(rows),
            'error_color_bounds': ERROR_BOUNDS,
            'speed_ratio': 'saved VBM milliseconds / seeded full milliseconds; R>1 means seeded route is faster',
            'error': 'cell p95 of abs(seeded mu - VBM 1e-6 reference) / abs(VBM 1e-6 reference); only internal status OK rows enter error cells; worst pointwise error is separately shown and outliers are listed',
            'nonok_diagnostic': 'non-OK points are retained in raw and joined tables; speed cells include finite positive values and hatch cells with incomplete OK status coverage',
        },
        'system': {'platform': platform.platform(), 'python': platform.python_version()},
        'figures': figures,
        'summary_by_condition': stats(rows),
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    report = [
        f'# Seeded radial frontier vs saved VBMicrolensing benchmark (HEAD `{head}`)\n',
        'The full 7,216-row q-rho corpus was evaluated for VBM speed targets `RelTol=1e-3` and `1e-4`, with uniform and linear limb darkening. The reference values are the saved VBM `RelTol=1e-6` values. Each condition has 3 repetitions.\n',
        'The tested path is the current v29 seed provider plus `epoch_seeded_radial_frontier`. Every repetition creates fresh point-source and finite-source objects, builds the 1400-bin caustic branches before timing, then times the center-image roots and finite-source seed construction. The timed seed step scans the prepared caustic branches and solves any additional seed roots. It then times the seeded radial discovery and integration. D14 and `classify_cells` are not called. The reported full time is the median of paired `seed_ms + radial_ms`; caustic construction, object construction, input/output, and figure generation are excluded.\n',
        'The existing VBM timing is the saved warmed direct `BinaryMag2` (uniform) / `BinaryMagDark` (linear) kernel timing. Fresh VBM 5.5 calls on the low-magnification tolerance sample reproduce the saved medians within 1%; the original timing-table generator was not retained. Thus this comparison adds seed construction to the current route and compares it with the direct VBM call.\n',
        'The seed provider support flag and the frontier `coverage_certified` flag are retained. The frontier reports `coverage_certified=false`. `status=OK` means seed-provider support plus value convergence; it is not a radial-support coverage certificate. The original run had 20 internally OK rows above 1e-3. Investigation found that endpoint polishing could accept a different fold outside the count-changing radius bracket and truncate an image band. The bracket-local fallback and post-polish band checks are now in place; the corrected full rerun has zero internally OK rows above 1e-3. See `OUTLIER_DIAGNOSIS.md` for the trace and limits of this empirical check.\n',
        'The four output figures each have six panels: three median speed-ratio maps and three p95 relative-error maps over q-rho, finite-source magnification-rho, and finite-source magnification-q. Bins use the established q-rho coverage edges and require at least eight points. Non-OK rows stay in raw and joined tables; finite positive outputs remain visible in runtime panels, while error panels use rows with seed support and value convergence.\n',
        '## Conditions\n',
        '| profile | target | rows | OK | full p50 ms | seed p50 ms | radial p50 ms | median row-wise R | accepted p95 error | accepted max error |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for key, item in summary['summary_by_condition'].items():
        profile, target_text = key.split(':target=')
        stage = item['stage_timing_ms']
        report.append(
            f"| {profile} | {target_text} | {item['rows']} | "
            f"{item['status_ok']} ({100*item['status_ok_fraction']:.1f}%) | "
            f"{item['v2_ms']['p50']:.6g} | {stage['seed_p50']:.6g} | "
            f"{stage['planning_p50'] + stage['integration_p50']:.6g} | "
            f"{item['speed_ratio_vbm_over_v2_positive']['p50']:.6g} | "
            f"{item['relative_error_v2_over_vbm_1e-6_status_ok']['p95']:.6g} | "
            f"{item['relative_error_v2_over_vbm_1e-6_status_ok']['max']:.6g} |"
        )
    worst_by_geometry = {}
    for row in accepted:
        geometry = (row['case_id'], row['epoch_index'])
        previous = worst_by_geometry.get(geometry)
        if previous is None or row['relative_error_v2_over_vbm_1e-6'] > previous['relative_error_v2_over_vbm_1e-6']:
            worst_by_geometry[geometry] = row
    worst_rows = sorted(worst_by_geometry.values(),
                        key=lambda r: r['relative_error_v2_over_vbm_1e-6'],
                        reverse=True)
    if outliers:
        outlier_note = (
            'The internal OK flag did not catch these discrepancies. '
            'Full rows are in `accepted_error_outliers.tsv`.\n'
        )
    else:
        outlier_note = (
            'No internally OK row exceeds 1e-3 relative error against the saved '
            'VBM 1e-6 reference. The radial-support coverage flag remains false.\n'
        )
    report.extend([
        '\n## Worst internally accepted reference discrepancies\n',
        f"Across internally OK rows, the maximum pointwise relative error is {external_error_screen['max_relative_error_among_internal_ok']:.6g}; "
        f"{external_error_screen['counts_above_threshold']['0.001']} rows exceed 1e-3 across "
        f"{external_error_screen['distinct_case_epoch_geometries_above_1e-3']} distinct case/epoch geometries. "
        + outlier_note,
        '| case | epoch | profile | target | q | rho | seeded mu | VBM 1e-6 mu | relative error |\n|---:|---:|---|---:|---:|---:|---:|---:|---:|',
    ])
    for row in worst_rows[:10]:
        report.append(
            f"| {row['case_id']} | {row['epoch_index']} | {row['profile']} | {row['target']:g} | "
            f"{row['q']:.6g} | {row['rho']:.6g} | {row['v2_mu']:.8g} | "
            f"{row['reference_1e-6']:.8g} | {row['relative_error_v2_over_vbm_1e-6']:.6g} |"
        )
    report.extend([
        '\n## Artifacts\n',
        f'- Raw results: `{V2_PATH}`',
        f'- Joined results: `{joined_path}`',
        f'- Internally OK rows with relative error >1e-3: `{outlier_path}`',
        f'- Diagnosis and pre-fix rows: `{ROOT / "OUTLIER_DIAGNOSIS.md"}` and `{ROOT / "accepted_error_outliers_pre_endpoint_fix.tsv"}`',
        f'- Six-panel figures: `{FIG_DIR}`',
        '\n## Reproduction\n',
        'From this checkout, run `bash benchmarks/holonomic/build_seeded_paper_qrho.sh`, then `bash evidence/holonomic/seeded_paper_qrho_20261014/run_full.sh`, then `python3 evidence/holonomic/seeded_paper_qrho_20261014/make_figures.py`. The full-run script fixes the single-thread environment; the plot script keeps its Matplotlib cache inside this evidence directory.\n',
    ])
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({'out': str(OUT), 'joined': str(joined_path),
                      'figures': figures,
                      'external_error_screen': external_error_screen,
                      'summary_by_condition': summary['summary_by_condition']}, indent=2))


if __name__ == '__main__':
    main()
