#pragma once
#include "boundary_bernstein.hpp"
#include "boundary_polynomial.hpp"
namespace lcbinint::holonomic::atlas_detail {
// Local physical-coordinate Krawczyk, avoiding differences of almost equal
// Bernstein coefficients after strong contraction. Y is only a proposal.
inline LocalProof contact_krawczyk(const PrimaryFrame& pf,const Box& b){
 LocalProof out;Q x[2]={(b.r0+b.r1)/2,(b.s0+b.s1)/2};Q h[2]={fmaxq(up(x[0]-b.r0),up(b.r1-x[0])),fmaxq(up(x[1]-b.s0),up(b.s1-x[1]))};if(h[0]<=0||h[1]<=0)return out;
 auto mid=local_fold_quantities<IQ>(IQ(x[0],x[0]),IQ(x[1],x[1]),pf,b.chart);
 auto all=local_fold_quantities<IQ>(IQ(b.r0,b.r1),IQ(b.s0,b.s1),pf,b.chart);
 IQ f[2]={mid.P,mid.Pt},j[2][2]={{all.PR,all.Pt},{all.Ptr,all.Ptt}};
 Q a=midpoint(mid.PR),v=midpoint(mid.Pt),c=midpoint(mid.Ptr),d=midpoint(mid.Ptt);Q det=a*d-v*c;if(!finiteq(det)||det==0)return out;
 Q Y[2][2]={{d/det,-v/det},{-c/det,a/det}};IQ K[2];Q norm=0;
 for(int i=0;i<2;++i){K[i]=IQ(x[i],x[i]);Q row=0;
  for(int k=0;k<2;++k)K[i]=K[i]-IQ(Y[i][k],Y[i][k])*f[k];
  for(int l=0;l<2;++l){IQ e(double(i==l));for(int k=0;k<2;++k)e=e-IQ(Y[i][k],Y[i][k])*j[k][l];K[i]=K[i]+e*IQ(-h[l],h[l]);row=up(row+up(up(mag(e)*h[l])/h[i]));}norm=fmaxq(norm,row);
 }
 if(K[0].hi<b.r0||K[0].lo>b.r1||K[1].hi<b.s0||K[1].lo>b.s1){out.kind=2;return out;}
 out.x=(K[0]-IQ(b.r0,b.r0))*inverse(IQ(b.r1,b.r1)-IQ(b.r0,b.r0));out.y=(K[1]-IQ(b.s0,b.s0))*inverse(IQ(b.s1,b.s1)-IQ(b.s0,b.s0));
 if(norm<1&&K[0].lo>b.r0&&K[0].hi<b.r1&&K[1].lo>b.s0&&K[1].hi<b.s1)out.kind=3;
 return out;
}
inline bool contact_candidate(const PrimaryFrame& pf,const Box& domain,Box& candidate){
 D14Real r=d14_from_qf((domain.r0+domain.r1)/2),s=d14_from_qf((domain.s0+domain.s1)/2);
 Q dr=0,ds=0;
 for(int i=0;i<16;++i){auto g=local_fold_quantities(r,s,pf,bool(domain.chart));auto det=g.PR*g.Ptt-g.Pt*g.Ptr;
  if(!det.finite()||det.hi==0)return false;
  auto xr=(g.P*g.Ptt-g.Pt*g.Pt)/det,ys=(g.PR*g.Pt-g.Ptr*g.P)/det;
  r=r-xr;s=s-ys;dr=fabsq(d14_to_qf(xr));ds=fabsq(d14_to_qf(ys));
  if(!r.finite()||!s.finite())return false;
  if(d14_to_qf(r)<domain.r0||d14_to_qf(r)>domain.r1||d14_to_qf(s)<domain.s0||d14_to_qf(s)>domain.s1)return false;
  if(dr<1e-25Q&&ds<1e-25Q)break;
 }
 // Candidate radius is not an error estimate: a fresh enclosure certificate
 // is mandatory below. Current exact-dyadic source is reevaluated there.
 Q hr=fmaxq(dr*8,1e-18Q*fmaxq(1,fabsq(d14_to_qf(r)))),hs=fmaxq(ds*8,1e-16Q);
 candidate=domain;candidate.r0=down(d14_to_qf(r)-hr);candidate.r1=up(d14_to_qf(r)+hr);candidate.s0=down(d14_to_qf(s)-hs);candidate.s1=up(d14_to_qf(s)+hs);
 return candidate.r0>=domain.r0&&candidate.r1<=domain.r1&&candidate.s0>=domain.s0&&candidate.s1<=domain.s1;
}
} // namespace
