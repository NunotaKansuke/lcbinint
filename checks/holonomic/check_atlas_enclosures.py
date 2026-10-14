#!/usr/bin/env python3
"""Exact-rational source polynomial -> coefficients, independent of C++ tensor algebra."""
import json,sys
from fractions import Fraction as F
from math import comb
from pathlib import Path
import sympy as S

def hexq(x):
    mant,ex=x.lower().split('p');sg=-1 if mant[0]=='-' else 1
    a,b=mant.lstrip('+-')[2:].split('.')
    return sg*F(int(a+b,16),16**len(b))*F(2)**int(ex)
x,y,R,s=S.symbols('x y R s',real=True)
cache={};n=0
for line in Path(sys.argv[1]).read_text().splitlines():
    f=line.split();key=tuple(map(int,(f[0],f[2])));i,j=int(f[3]),int(f[4]);vals=list(map(hexq,f[5:]));a,m,X,Y,rho,r0,r1,s0,s1,lo,hi=vals
    if key not in cache:
        a,m,X,Y,rho,r0,r1,s0,s1=map(lambda z:S.Rational(z.numerator,z.denominator),vals[:9])
        T=-(X+S.I*Y)*R**2*(1-S.I*s)**2+R*(R**2-1+a*(X+S.I*Y))*(1+s*s)+a*(m-R**2)*(1+S.I*s)**2
        p=S.Poly(S.expand(rho**2*R**2*(1+s*s)*((R-a)**2+(R+a)**2*s*s)-T*S.conjugate(T)),s)
        expr=p.as_expr() if key[1]==0 else sum((-1)**k*p.nth(4-k)*s**k for k in range(5))
        p=S.Poly(S.expand(expr.subs({R:r0+(r1-r0)*x,s:s0+(s1-s0)*y})),x,y)
        cache[key]={(ii,jj):sum(p.coeff_monomial(x**k*y**l)*S.Rational(comb(ii,k),comb(6,k))*S.Rational(comb(jj,l),comb(4,l)) for k in range(ii+1) for l in range(jj+1)) for ii in range(7) for jj in range(5)}
    exact=cache[key][i,j]
    if not lo<=exact<=hi:raise AssertionError((f[:5],str(lo),str(exact),str(hi)))
    n+=1
out={'status':'PASS','coefficient_enclosures':n,'boxes':len(cache),'tiers':['double interval','D14Real ball','qf interval'],'oracle':'SymPy exact rational; hexadecimal bounds exact parsed'}
Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n');print(out)
