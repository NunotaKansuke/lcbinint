import sympy as S
from pathlib import Path
r,a,m,x,h=S.symbols('R a m X rho');c0=-x*r*r+r*(r*r-1+a*x)+a*(m-r*r);c2=r*(r*r-1+a*x)+x*r*r-a*(m-r*r);c1=2*(a*(m-r*r)+x*r*r)
p0=h*h*r*r*(r-a)**2-c0*c0;p2=h*h*r*r*((r-a)**2+(r+a)**2)-c1*c1-2*c0*c2;p4=h*h*r*r*(r+a)**2-c2*c2
f=S.factor(p2*p2-4*p0*p4);print('p2^2-4*p0*p4 =',f)
Path('evidence/holonomic/stress_cause_20261015/symmetry.txt').write_text(str(f)+'\n')
v=S.symbols('v');C=a*a*m*m+(1-2*a*a*m+a*a*(x*x-h*h)+2*a*x*(3*m-1))*v+(-2+a*a+x*x-h*h-4*a*x)*v*v+v**3
L=-a*a*m+(a*a-1)*v+v*v;U=a*m-a*a*m*x+(-a*(m+1)+x*(a*a-1)+a*(x*x-h*h))*v+(a+x)*v*v;E=v-m
G=S.expand(U*U-4*a*x*E*C-16*a*a*x*x*v*E*E)
print('G symmetric',S.factor(G.subs({m:S.Rational(1,2),x:a/2})))
print('D symmetric',S.factor(((C*C-4*v*G)*G*G).subs({m:S.Rational(1,2),x:a/2})))
