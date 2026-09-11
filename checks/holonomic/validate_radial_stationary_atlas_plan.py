#!/usr/bin/env python3
"""Exact algebra checks for the radial stationary-atlas DESIGN, not a solver benchmark.
Requires Python 3 and sympy. No network access, repository build, or production edits.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import sympy as sp


def main() -> None:
    checks: list[str] = []
    def zero(name: str, expression: sp.Expr) -> None:
        if sp.expand(expression) != 0 and sp.simplify(expression) != 0:
            raise AssertionError(name)
        checks.append(name)
    def truth(name: str, value: bool) -> None:
        if not value:
            raise AssertionError(name)
        checks.append(name)

    R, s, u, a, m, X, Y, rho = sp.symbols('R s u a m X Y rho', real=True)
    n0 = -(X + sp.I*Y)*R**2
    n1 = R*(R**2 - 1 + a*(X + sp.I*Y))
    n2 = a*(m-R**2)
    T = sp.expand(n0+n1+n2 + 2*sp.I*(n2-n0)*s + (n1-n0-n2)*s**2)
    P = sp.expand(rho**2*R**2*(1+s**2)*((R-a)**2+(R+a)**2*s**2) - T*sp.conjugate(T))
    truth('boundary bidegree at most (6,4)', sp.degree(P,R)==6 and sp.degree(P,s)==4)
    Q = sp.Poly(P,s)
    coeff = [Q.nth(k) for k in range(5)]
    Qrec = sum((-1)**k*coeff[4-k]*u**k for k in range(5))
    zero('reciprocal polynomial uses coefficient reversal, not 1/u evaluation', sp.expand(u**4*P.subs(s,-1/u))-Qrec)
    zero('nonzero boundary numerator at first lens R=0', P.subs(R,0)+a*a*m*m*(1+s*s)**2)
    zero('nonzero boundary numerator at second lens', P.subs({R:a,s:0})+a*a*(1-m)**2)
    zero('axis polynomial is even', P.subs(Y,0)-P.subs({Y:0,s:-s}))
    tr, ti = sp.expand(sp.re(T)), sp.expand(sp.im(T))
    zero('direct real polynomial construction', P-(rho**2*R**2*(1+s**2)*((R-a)**2+(R+a)**2*s**2)-tr**2-ti**2))

    p0,p1,p2,p3,p4,z,w = sp.symbols('p0 p1 p2 p3 p4 z w', real=True)
    M, h = sp.symbols('M h', real=True)
    pp=p0+p1*z+p2*z*z+p3*z**3+p4*z**4
    E=p0+p1*M+p2*(M*M+w)+p3*(M**3+3*M*w)+p4*(M**4+6*M*M*w+w*w)
    O=p1+2*p2*M+p3*(3*M*M+w)+4*p4*M*(M*M+w)
    zero('root-pair E is symmetric even polynomial', ((pp.subs(z,M+h)+pp.subs(z,M-h))/2).expand()-E.subs(w,h*h))
    zero('root-pair O has no sqrt(w) division', ((pp.subs(z,M+h)-pp.subs(z,M-h))/(2*h)).expand()-O.subs(w,h*h))
    # On a fold pp(M)=pp'(M)=0. E_M vanishes; E_w=pp''/2, O_M=pp'', O_w=pp'''/6.
    zero('E_M at w=0 is P_s', sp.diff(E,M).subs(w,0)-sp.diff(pp,z).subs(z,M))
    zero('E_w at w=0 is P_ss/2', sp.diff(E,w).subs(w,0)-sp.diff(pp,z,2).subs(z,M)/2)
    zero('O_M at w=0 is P_ss', sp.diff(O,M).subs(w,0)-sp.diff(pp,z,2).subs(z,M))
    zero('O_w at w=0 is P_sss/6', sp.diff(O,w).subs(w,0)-sp.diff(pp,z,3).subs(z,M)/6)

    v, lam, C, G, Z, A, b = sp.symbols('v lam C G Z A b', real=True)
    cubic=v*lam**3+C*lam**2+G*lam-4*Z
    delta=(C*C-4*v*G)*G*G+8*C*(2*C*C-9*v*G)*Z-432*v*v*Z*Z
    zero('cubic lift discriminant identity', sp.discriminant(cubic,lam)-delta)
    aa=C*C-3*v*G
    bb=2*C**3-9*v*C*G-108*v*v*Z
    zero('stationary-value scaled discriminant identity', 27*v*v*delta-(4*aa**3-bb**2))
    for sigma in (-1,1):
        l=(-C+sigma*b)/(3*v)
        # b^2=A, and choose G=(C^2-b^2)/(3v); then exactly on f_lambda=0.
        subst={G:(C*C-b*b)/(3*v)}
        zero(f'cubic stationary branch {sigma} identity', 27*v*v*cubic.subs(lam,l).subs(subst)-(bb.subs(subst)-2*sigma*b**3))

    # Degree-(2,2) toy: the same endpoint crossing count can hide two interior events.
    f=s*s-(R-sp.Rational(2,5))*(R-sp.Rational(3,5))
    truth('equal endpoint topology does not certify an event-free slab', len(sp.real_roots(f.subs(R,0),s))==len(sp.real_roots(f.subs(R,1),s))==2 and len(sp.real_roots(f.subs(R,sp.Rational(1,2)),s))==0)
    zero('stationary branch envelope derivative has no branch derivative term', sp.diff(f.subs(s,0),R)-sp.diff(f,R).subs(s,0))
    truth('two crossings need a derivative/cover test, not one endpoint sign test', sp.solve(f.subs(s,0),R)==[sp.Rational(2,5),sp.Rational(3,5)])
    fdeg=s*s+(R-sp.Rational(1,2))**2
    truth('even contact is not an ordinary fold', fdeg.subs({s:0,R:sp.Rational(1,2)})==0 and sp.diff(fdeg,R).subs({s:0,R:sp.Rational(1,2)})==0)

    # Exact Bernstein conversion and subdivision checks, arbitrary rational coefficients.
    x,y=sp.symbols('x y')
    def bern(n,i,t): return sp.binomial(n,i)*t**i*(1-t)**(n-i)
    bbtab=[[sp.Rational((i+2)*(j+3)%13-6,17) for j in range(5)] for i in range(7)]
    poly=sp.expand(sum(bbtab[i][j]*bern(6,i,x)*bern(4,j,y) for i in range(7) for j in range(5)))
    ds=sp.expand(sum(4*(bbtab[i][j+1]-bbtab[i][j])*bern(6,i,x)*bern(3,j,y) for i in range(7) for j in range(4)))
    zero('Bernstein derivative tensor identity', sp.diff(poly,y)-ds)
    def split_half(seq):
        rows=[seq]
        while len(rows[-1])>1:
            row=rows[-1]; rows.append([(row[i]+row[i+1])/2 for i in range(len(row)-1)])
        return [row[0] for row in rows], [row[-1] for row in rows][::-1]
    left=[[None]*5 for _ in range(7)]; right=[[None]*5 for _ in range(7)]
    for j in range(5):
        ll,rr=split_half([bbtab[i][j] for i in range(7)])
        for i in range(7): left[i][j],right[i][j]=ll[i],rr[i]
    for name,tab,sub in [('left',left,x/2),('right',right,(1+x)/2)]:
        subpoly=sp.expand(sum(tab[i][j]*bern(6,i,x)*bern(4,j,y) for i in range(7) for j in range(5)))
        zero(f'exact de Casteljau subdivision {name}', poly.subs(x,sub)-subpoly)
    # Discrete evaluation sanity, not a proof of enclosure; convex hull proof is in plan.
    for xx,yy in [(sp.Rational(1,3),sp.Rational(2,7)),(0,1),(1,0)]:
        val=poly.subs({x:xx,y:yy}); flat=[b for row in bbtab for b in row]
        truth(f'Bernstein hull sample {xx},{yy}', min(flat)<=val<=max(flat))

    # Exact Krawczyk contraction for G=(s^2-R,2s) on R,s in [-1/4,1/4].
    # Y=[[ -1,0],[0,1/2]], so YG(0)=0 and I-YJ=[[0,2s],[0,0]].
    r=sp.Rational(1,4)
    truth('ordinary-fold Krawczyk toy contraction and strict inclusion', 2*r<1 and 2*r*r<r)
    result={'status':'PASS','checks':len(checks),'validated':checks,
            'scope':'Exact algebra and small theorem witnesses for a design; no production solver run.',
            'not_validated':['floating point outward enclosures','global coverage implementation','new C++ performance','14432-row corpus','new full-epoch speed','soft-hint removal correctness']}
    dest=Path(sys.argv[1]) if len(sys.argv)>1 else Path('validation.json')
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':len(checks),'output':str(dest)},ensure_ascii=False))

if __name__=='__main__': main()
