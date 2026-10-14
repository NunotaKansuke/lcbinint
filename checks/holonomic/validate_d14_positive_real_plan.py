#!/usr/bin/env python3
"""Exact-arithmetic checks for the proposed D14 positive-real-root plan.

This is a design check, NOT the production solver, a floating-point enclosure
implementation, an lcbinint accuracy benchmark, or a performance comparison.
Requires SymPy. Run: python validate_plan_math.py > validation.json
"""
from __future__ import annotations
import json
import random
import time
from functools import reduce
from math import gcd
import sympy as s

x = s.symbols('x')
checks = 0

def check(condition, message):
    global checks
    if not condition:
        raise AssertionError(message)
    checks += 1


def primitive_positive(p):
    """Remove rational content using a strictly positive multiplier."""
    p = s.Poly(p, x, domain=s.QQ)
    if p.is_zero:
        return p
    den = s.ilcm(*[c.q for c in p.all_coeffs()]) if len(p.all_coeffs()) > 1 else p.LC().q
    ints = [int(c*den) for c in p.all_coeffs()]
    g = reduce(gcd, (abs(c) for c in ints))
    return s.Poly(p.as_expr() * s.Rational(den, g), x, domain=s.QQ)


def prem_positive(a, b):
    """R <- |lc(b)| R - sign(lc(b))*lc(R)*x^shift*b.

    The final result is a positive scalar multiple of rem(a,b).
    Production normalization is by positive powers of two, not rational gcd.
    """
    r = a
    while not r.is_zero and r.degree() >= b.degree():
        lc = b.LC()
        sigma = s.sign(lc)
        shift = r.degree() - b.degree()
        r = s.Poly(abs(lc)*r.as_expr() - sigma*r.LC()*x**shift*b.as_expr(), x, domain=s.QQ)
        r = primitive_positive(r)
    return r


def chain(p):
    p = primitive_positive(p)
    if p.degree() == 0:
        return [p]
    seq = [p, primitive_positive(p.diff())]
    while seq[-1].degree() > 0:
        r = prem_positive(seq[-2], seq[-1])
        if r.is_zero:
            break
        seq.append(primitive_positive(-r))
    return seq


def side_sign(p, point, side):
    # Exact one-sided convention; roots at endpoints are excluded by count_open.
    k = 0
    while not p.is_zero:
        val = p.eval(point)
        if val:
            return int(s.sign(val)) * (side**k)
        p = p.diff()
        k += 1
    return 0


def variation(seq, point, side):
    signs = [side_sign(p, point, side) for p in seq]
    signs = [v for v in signs if v]
    return sum(a != b for a, b in zip(signs, signs[1:]))


def count_open(seq, lo, hi):
    return variation(seq, lo, +1) - variation(seq, hi, -1)


def d14(a, m, X, Y, rho):
    # Exact version of the C3/G4/Z3 formula; all five inputs are rationals.
    e = x-m
    d = x-1
    beta = X*X+Y*Y-rho*rho
    L = x*d+a*a*e
    U = a*e*d+X*L+a*x*beta
    C = x*d*d+a*a*e*e+x*(x+a*a)*beta+2*a*X*x*(3*m-1-2*x)
    G = U*U+Y*Y*L*L-4*a*e*X*C-4*a*a*e*e*x*(4*X*X+Y*Y)
    Z = a*a*(1-m)*Y*Y*e*((a*m+(X-a)*x)**2+x*x*(Y*Y-rho*rho))
    D = (C*C-4*x*G)*G*G+8*C*(2*C*C-9*x*G)*Z-432*x*x*Z*Z
    return s.Poly(D, x), s.Poly(C, x), s.Poly(G, x), s.Poly(Z, x)


def main():
    start = time.perf_counter()
    rng = random.Random(20260911)
    polynomials = []
    for degree in [2, 3, 4, 7, 10, 14]:
        for _ in range(4):
            coeff = [rng.choice([-9,-5,-1,1,3,7])] + [rng.randint(-9,9) for _ in range(degree)]
            polynomials.append(s.Poly.from_list(coeff,x,domain=s.QQ))
    close = s.Rational(1,2) + s.Rational(1,2**28)
    special = [
        (x-s.Rational(1,2))*(x-close)*(x+1)*(x*x+1),
        ((x-s.Rational(1,2))**2+s.Rational(1,2**56))*(x-s.Rational(3,4)),
        (x-s.Rational(1,3))**2*(x-s.Rational(2,3))**3*(x+1),
        x*x*(x-1)**2*(x-s.Rational(1,2)),
        -(x-1)**4*(x-2)*(x*x+4),
        x*x+1,
    ]
    polynomials.extend(s.Poly(p,x,domain=s.QQ) for p in special)
    R=s.Rational
    d14_counts=[]
    for vals in [(R(1),R(1,2),R(1,5),R(1,7),R(1,10)),
                 (R(6,5),R(99,100),R(1,3),R(1,1000),R(1,100)),
                 (R(3,2),R(3,4),R(-2,5),R(0),R(1,6))]:
        D,C,G,Z = d14(*vals)
        polynomials.append(D)
        if Z.is_zero:
            check(D == G*G*(C*C-s.Poly(4*x,x)*G), 'Z=0 factorization')
        seq=chain(D)
        d14_counts.append({'degree': int(D.degree()), 'positive_roots_in_0_16': count_open(seq,0,16), 'axis':vals[3]==0})
    for p in polynomials:
        own=chain(p)
        oracle=[s.Poly(q,x,domain=s.QQ) for q in s.sturm(p.as_expr(),x)]
        for lo,hi in [(-3,0),(0,1),(0,16),(R(1,4),R(3,4)),(1,3)]:
            check(count_open(own,lo,hi)==count_open(oracle,lo,hi), 'Sturm count mismatch')
        for a,b in zip(own,own[1:]):
            if b.degree()>0:
                nr = prem_positive(a,b)
                rr = a.rem(b)
                if not rr.is_zero:
                    lam = nr.LC()/rr.LC()
                    check(lam>0 and s.expand(nr.as_expr()-lam*rr.as_expr())==0, 'wrong PRS sign')
        # Split accounting includes an exact root at the split only once.
        mid=R(1,2)
        check(count_open(own,0,1)==count_open(own,0,mid)+count_open(own,mid,1)+int(p.eval(mid)==0), 'split accounting')
    # Warm completeness: disjoint opposite-sign brackets + global N suffice.
    p=s.Poly((x-R(1,4))*(x-R(3,4))*(x*x+1),x)
    brackets=[(R(1,8),R(3,8)),(R(5,8),R(7,8))]
    seq=chain(p)
    check(count_open(seq,0,1)==len(brackets), 'total count')
    check(all(p.eval(a)*p.eval(b)<0 for a,b in brackets), 'bracket signs')
    check(all(count_open(seq,a,b)==1 for a,b in brackets), 'per-bracket count consequence')
    # Same total root count does NOT validate old neighborhoods by itself.
    p_now=s.Poly((x-R(9,20))*(x-R(11,20))*(x*x+1),x)
    check(count_open(chain(p_now),0,1)==2, 'current N')
    check(all(p_now.eval(a)*p_now.eval(b)>0 for a,b in brackets), 'old brackets not accepted')
    # Pair birth must be discovered even if old positive roots are unchanged.
    p_new=s.Poly(p.as_expr()*((x-R(1,2))**2-R(1,2**20)),x)
    check(count_open(chain(p_new),0,1)==4, 'birth detection')
    # D14(v) need not be even: +/- symmetry has already been used in v=R^2.
    D, *_=d14(R(1),R(1,2),R(1,5),R(1,7),R(1,10))
    check(D.as_expr().subs(x,-x)!=D.as_expr(), 'D14 is not generally even in v')
    print(json.dumps({'status':'PASS','checks':checks,'polynomial_cases':len(polynomials),
                      'd14_examples':d14_counts,
                      'validated':['positive-scaling PRS equals positive multiples of Euclidean remainder',
                                   'Sturm open-interval counts against independent SymPy chain',
                                   'multiple roots, close roots, complex near-real pair, endpoint atoms',
                                   'global-count + disjoint sign-bracket completeness conditions',
                                   'exact Z=0 factorization and absence of general +/-v symmetry'],
                      'not_validated':['floating-point enclosure implementation','C++ runtime','lcbinint corpus accuracy','soft-cut removal'],
                      'elapsed_seconds':time.perf_counter()-start},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
