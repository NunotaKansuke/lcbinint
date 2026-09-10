#!/usr/bin/env python3
"""Independent mathematical checks for the V2 adaptive-radial design.

Not an lcbinint implementation or a lens/whole-epoch speed benchmark.
Python >= 3.10, numpy. scipy is NOT required. Outputs JSON next to this file.
Tests nested Fejer-II weights, cache reuse, interpolation-detail norm,
fold mappings, and a finite-sample error-estimator counterexample.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Rule:
    level: int
    x: np.ndarray
    w: np.ndarray
    theta: np.ndarray
    norm_w: np.ndarray
    bary_w: np.ndarray


def key(level: int, k: int) -> tuple[int, int]:
    """Canonical dyadic-angle ID: theta = pi*k/2**level."""
    if level < 1 or not 0 < k < (1 << level):
        raise ValueError("An interior node is required")
    while k % 2 == 0:
        k //= 2
        level -= 1
    return level, k


def node_x(level: int, k: int) -> float:
    level, k = key(level, k)
    # One canonical evaluation; the midpoint is exact.
    if level == 1:
        return 0.0
    return math.cos(math.pi * k / (1 << level))


@lru_cache(None)
def rule(level: int) -> Rule:
    if not 1 <= level <= 11:
        raise ValueError("Test rule level must be in [1, 11]")
    m = 1 << level
    k = np.arange(1, m)
    theta = math.pi * k / m
    x = np.array([node_x(level, int(j)) for j in k])
    # Integrate U_j exactly: integral U_j = 2/(j+1) for even j, else 0.
    odd = np.arange(1, m, 2)
    w = (4.0 / m) * np.sin(theta) * (
        np.sin(theta[:, None] * odd[None, :]) / odd[None, :]
    ).sum(axis=1)
    norm_w = (math.pi / m) * np.sin(theta) ** 2
    bary_w = (-1.0) ** k * np.sin(theta) ** 2
    return Rule(level, x, w, theta, norm_w, bary_w)


def interpolate(x: np.ndarray, y: np.ndarray, bw: np.ndarray,
                targets: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = []
    for t in targets:
        hit = np.flatnonzero(x == t)
        if len(hit):
            out.append(y[hit[0]])
        else:
            a = bw / (t - x)
            out.append(np.tensordot(a, y, axes=(0, 0)) / a.sum())
    return np.asarray(out)


def detail_norm(level: int, values: np.ndarray) -> np.ndarray:
    """Exact (in exact arithmetic) weighted L2 norm of p_fine-p_coarse.

    Coarse nodes are the even k nodes, hence values[1::2]. The norm's GC2
    rule integrates the squared polynomial difference exactly.
    """
    if level < 2:
        raise ValueError("A coarse level is required")
    fine, coarse = rule(level), rule(level - 1)
    pred = interpolate(coarse.x, values[1::2], coarse.bary_w, fine.x)
    diff = values - pred
    # Force inherited nodes to exact zero, avoiding interpolation noise.
    diff[1::2] = 0.0
    return np.sqrt(np.tensordot(fine.norm_w, diff * diff, axes=(0, 0)))


class SampleCache:
    def __init__(self, function):
        self.function = function
        self.samples: dict[tuple[int, int], np.ndarray] = {}
        self.evals = 0

    def at_level(self, level: int) -> np.ndarray:
        out = []
        for k in range(1, 1 << level):
            ident = key(level, k)
            if ident not in self.samples:
                self.samples[ident] = np.asarray(self.function(node_x(level, k)))
                self.evals += 1
            out.append(self.samples[ident])
        return np.asarray(out)


def u_poly(n: int, x):
    if n == 0:
        return np.ones_like(x)
    a, b = np.ones_like(x), 2*x
    for _ in range(1, n):
        a, b = b, 2*x*b - a
    return b


def sample_rows():
    exact = np.array([2/3 + 0.3*2/5 + 0.2*2/7, 2 + 0.3*2/3])
    def unmapped(x):
        r = (x+1)*0.5
        return np.array([math.sqrt(r)*(1+0.3*r+0.2*r*r)*0.5,
                         (1+0.3*r)/math.sqrt(r)*0.5])
    def mapped(x):
        t = (x+1)*0.5
        r = t*t
        # dR/dx=t. Evaluate removable factors algebraically, not 0*infinity.
        return np.array([t*t*(1+0.3*r+0.2*r*r), 1+0.3*r])
    rows = []
    for name, f in [('affine', unmapped), ('left_fold_squared', mapped)]:
        cache = SampleCache(f)
        for level in range(3, 9):
            vals = cache.at_level(level)
            q = np.tensordot(rule(level).w, vals, axes=(0, 0))
            d = detail_norm(level, vals)
            rows.append(dict(mapping=name, nodes=(1<<level)-1,
                             total_unique_evals=cache.evals,
                             value_error=abs(float(q[0]-exact[0])),
                             derivative_model_error=abs(float(q[1]-exact[1])),
                             detail_indicator=(math.sqrt(math.pi)*d).tolist()))
    return rows


def main() -> None:
    res = {}
    max_exactness = 0.0
    nested = []
    for level in range(2, 9):
        q = rule(level)
        assert np.all(q.w > 0)
        assert abs(float(q.w.sum()) - 2) < 2e-14
        # Test in bounded Chebyshev-T basis, avoiding monomial conditioning.
        t0, t1 = np.ones_like(q.x), q.x.copy()
        for degree in range(len(q.x)):
            if degree == 0: vals = t0
            elif degree == 1: vals = t1
            else:
                t0, t1 = t1, 2*q.x*t1-t0
                vals = t1
            exact = 0.0 if degree % 2 else 2.0/(1-degree*degree)
            max_exactness = max(max_exactness, abs(float(q.w@vals)-exact))
        if level > 2:
            assert np.array_equal(q.x[1::2], rule(level-1).x)
            nested.append(True)
    assert max_exactness < 1e-12
    res['max_polynomial_exactness_error'] = max_exactness
    res['bitwise_nested_nodes'] = all(nested)

    cache = SampleCache(lambda x: np.array([math.exp(x), 1+x*x]))
    counts=[]
    increments=[]
    previous=0
    for level in (3,4,5,6):
        cache.at_level(level)
        counts.append(cache.evals)
        increments.append(cache.evals-previous)
        previous=cache.evals
    assert counts == [7,15,31,63]
    assert increments == [7,8,16,32]
    res['cache_unique_counts'] = counts
    res['cache_added_counts'] = increments

    # Independent exactness test for the norm: orthogonal U coefficients.
    rng=np.random.default_rng(84261)
    max_norm_error=0.0
    max_cs_violation=0.0
    for level in (3,4,5,6):
        q=rule(level)
        vals=rng.normal(size=len(q.x))
        c=rule(level-1)
        pred=interpolate(c.x, vals[1::2], c.bary_w, q.x)
        diff=vals-pred
        b=(2.0/(1<<level))*(np.sin(q.theta[:,None]*np.arange(1,1<<level))
                 .T @ (diff*np.sin(q.theta)))
        via_coeff=math.sqrt(math.pi*0.5*float(b@b))
        via_nodes=float(detail_norm(level, vals))
        max_norm_error=max(max_norm_error, abs(via_coeff-via_nodes))
        integral_diff=abs(float(q.w@vals-c.w@vals[1::2]))
        max_cs_violation=max(max_cs_violation,
                             integral_diff-math.sqrt(math.pi)*via_nodes)
    assert max_norm_error < 1e-12
    assert max_cs_violation <= 1e-12
    res['weighted_L2_norm_identity_error'] = max_norm_error
    res['cauchy_schwarz_bound_violation'] = max(0.0,max_cs_violation)

    # A finite-sample failure example: positive analytic polynomial which
    # vanishes at ALL nodes of levels 1..4. No estimator based only on these
    # samples is an unconditional integration-error bound.
    q=rule(4)
    vals=u_poly(15, q.x)**2
    estimate=float(q.w@vals)
    detail=float(math.sqrt(math.pi)*detail_norm(4, vals))
    exact=2.0*sum(1.0/(2*k+1) for k in range(16))
    assert estimate < 1e-20 and exact > 1.0
    res['aliasing_counterexample'] = dict(function='U_15(x)^2',
                                        Q15=estimate, detail_indicator=detail,
                                        exact_integral=exact,
                                        conclusion='finite-sample indicator is not a rigorous bound')
    res['fold_model_rows']=sample_rows()
    p=Path(__file__).with_name('validation_results.json')
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
