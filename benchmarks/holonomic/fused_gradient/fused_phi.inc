// A boundary square root can be much smaller than the two quantities whose
// difference defines phi. Re-evaluate that rare cancellation in binary128,
// preserving the radial low part; a genuinely negative result still fails.
inline PhiValDP fused_precise_phi(double R,double theta,const PrimaryFrame&pf){
 using Q=__float128;struct Z{Q re,im;};
 auto add=[](Z a,Z b){return Z{a.re+b.re,a.im+b.im};};
 auto sub=[](Z a,Z b){return Z{a.re-b.re,a.im-b.im};};
 auto mul=[](Z a,Z b){return Z{a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re};};
 auto scale=[](Z a,Q s){return Z{a.re*s,a.im*s};};
 auto inverse=[](Z a){Q n=a.re*a.re+a.im*a.im;return Z{a.re/n,-a.im/n};};
 auto dot=[](Z a,Z b){return a.re*b.re+a.im*b.im;};
 const Q r=Q(R)+fused_gradient::radius_offset,m=pf.m0,n=1-m,rho=pf.rho;
 const Z z{r*cosq(Q(theta)),r*sinq(Q(theta))},zb{z.re,-z.im};
 const Z i=inverse(zb),j=inverse(sub(zb,{pf.a,0}));
 const Z g=sub(sub(sub(z,scale(i,m)),scale(j,n)),{pf.X,pf.Y});
 const Q norm=dot(g,g),r2=rho*rho;
 PhiValDP out;out.phi=double((r2-norm)/r2);
 out.dP={double(2*g.re/r2),double(2*g.im/r2),double(2*norm/(rho*r2)),
          double(-2*dot(g,sub(j,i))/r2),double(2*n*dot(g,mul(j,j))/r2)};
 if(const auto*motion=fused_gradient::active_motion){
  const Z radial=scale(add(add(z,scale(i,m)),scale(mul(mul(j,j),zb),n)),1/r);
  const Q slope=-2*dot(g,radial)/r2;
  for(int k=0;k<5;++k)out.dP[k]+=double(slope*motion->radius[k]);
 }
 return out;
}
