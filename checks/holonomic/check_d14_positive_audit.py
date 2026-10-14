#!/usr/bin/env python3
"""Compare emitted C++ enclosures and counts to exact dyadic SymPy H*."""
import json, sys
import sympy as s
from validate_d14_positive_real_plan import d14
def exacthex(t):
 mant,ex=t.split('p');n,f=mant.split('.') if '.' in mant else (mant,'')
 sign=-1 if n.startswith('-') else 1
 return sign*s.Rational(int(n.lstrip('-')+f,16),16**len(f))*s.Rational(2)**int(ex)
cases=[]
for line in open(sys.argv[1]):
 t=line.split()
 if t[0]=='P':
  a=[s.Rational(float.fromhex(v)) for v in t[2:7]]
  p,*_=d14(*a)
  c={'id':int(t[1]),'assurance':int(t[8]),'reason':t[10],'tier':int(t[11]),'roots':0,'coefficients':0}
  cases.append(c)
  if c['assurance']==2:
   expected=p.count_roots(0,s.Rational(2)**int(t[7]))
   assert expected==int(t[9]),(c,expected,t[9])
   c['exact_count']=int(expected)
 elif t[0]=='C':
  j=int(t[2]);lo,hi=map(exacthex,t[3:])
  assert lo<=p.nth(j)<=hi,(c,j)
  c['coefficients']+=1
 elif t[0]=='R':
  lo,hi=map(exacthex,t[1:]);assert p.count_roots(lo,hi)==1,(c,lo,hi)
  c['roots']+=1
print(json.dumps({'status':'PASS','cases':cases,'note':'Exact dyadic H*, independent count and each output enclosure; hexadecimal endpoints preserve every bit'},indent=2))
