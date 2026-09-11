#pragma once
#include "radial_stationary_atlas.hpp"
#include "adaptive_radial.hpp"
namespace lcbinint::holonomic {
struct AtlasSampleContext {const std::vector<ContactAnchor>* contacts=nullptr;long attempts=0,accepted=0;};
inline thread_local AtlasSampleContext* active_atlas_samples=nullptr;
struct AtlasSampleScope {AtlasSampleContext* old;explicit AtlasSampleScope(AtlasSampleContext* s):old(active_atlas_samples){active_atlas_samples=s;}~AtlasSampleScope(){active_atlas_samples=old;}};
namespace atlas_detail {
template<class T> T fraction(int n,int d){if constexpr(std::is_same_v<T,IQ>)return ratio<IQ>(n,d);else return T(double(n))/T(double(d));}
template<class T> struct PairJet {T e,o,em,ew,om,ow;};
template<class T> PairJet<T> pair_jet(T R,T m,T w,const PrimaryFrame& pf,int chart){auto g=local_fold_quantities(R,m,pf,bool(chart));T p4=g.Pssss*fraction<T>(1,24);return {g.P+w*g.Ptt*T(.5)+w*w*p4,g.Pt+w*g.Psss*fraction<T>(1,6),g.Pt+w*g.Psss*T(.5),g.Ptt*T(.5)+T(2)*w*p4,g.Ptt+T(4)*w*p4,g.Psss*fraction<T>(1,6)};}
struct CertifiedPair {IQ m,w;int chart=0;bool valid=false;};
inline CertifiedPair contact_pair(Q R,const ContactAnchor& c,const PrimaryFrame& pf){
 CertifiedPair out;out.chart=c.uniqueness.chart;
 if(R<c.uniqueness.r0||R>c.uniqueness.r1)return out;
 auto g=local_fold_quantities(c.R_center,c.s_center,pf,bool(out.chart));D14Real m=c.s_center,w=D14Real(-2)*g.PR/g.Ptt*(d14_from_qf(R)-c.R_center);
 if(!(w.hi>0))return out;
 Q dm=0,dw=0;
 for(int i=0;i<12;++i){auto f=pair_jet(d14_from_qf(R),m,w,pf,out.chart);auto det=f.em*f.ow-f.ew*f.om;if(det.hi==0||!det.finite())return out;auto a=(f.e*f.ow-f.o*f.ew)/det,b=(f.em*f.o-f.om*f.e)/det;m=m-a;w=w-b;dm=fabsq(d14_to_qf(a));dw=fabsq(d14_to_qf(b));if(dm<1e-27Q&&dw<1e-27Q)break;}
 Q x[2]={d14_to_qf(m),d14_to_qf(w)},h[2]={fmaxq(8*dm,1e-24Q),fmaxq(8*dw,1e-26Q)};
 IQ box[2]={{down(x[0]-h[0]),up(x[0]+h[0])},{down(x[1]-h[1]),up(x[1]+h[1])}};if(box[1].lo<=0)return out;
 h[0]=fmaxq(up(x[0]-box[0].lo),up(box[0].hi-x[0]));h[1]=fmaxq(up(x[1]-box[1].lo),up(box[1].hi-x[1]));
 auto mid=pair_jet(IQ(R,R),IQ(x[0],x[0]),IQ(x[1],x[1]),pf,out.chart),all=pair_jet(IQ(R,R),box[0],box[1],pf,out.chart);
 Q a=midpoint(mid.em),b=midpoint(mid.ew),cc=midpoint(mid.om),d=midpoint(mid.ow),det=a*d-b*cc;if(det==0||!finiteq(det))return out;
 Q Y[2][2]={{d/det,-b/det},{-cc/det,a/det}};IQ f[2]={mid.e,mid.o},j[2][2]={{all.em,all.ew},{all.om,all.ow}},K[2];Q norm=0;
 for(int i=0;i<2;++i){K[i]=IQ(x[i],x[i]);Q row=0;for(int k=0;k<2;++k)K[i]=K[i]-IQ(Y[i][k],Y[i][k])*f[k];for(int l=0;l<2;++l){IQ e(double(i==l));for(int k=0;k<2;++k)e=e-IQ(Y[i][k],Y[i][k])*j[k][l];K[i]=K[i]+e*IQ(-h[l],h[l]);row=up(row+up(up(mag(e)*h[l])/h[i]));}norm=fmaxq(norm,row);}
 if(!(norm<1&&K[0].lo>box[0].lo&&K[0].hi<box[0].hi&&K[1].lo>box[1].lo&&K[1].hi<box[1].hi))return out;
 out.m=K[0];out.w=K[1];out.valid=true;return out;
}
inline IQ sqrt_bound(IQ x){if(x.lo<0)return {-HUGE_VALQ,HUGE_VALQ};return {down(sqrtq(x.lo)),up(sqrtq(x.hi))};}
inline bool try_pair_sample(double Rhi,double Rlo,double jac,const LensParams& p,double u,const PrimaryFrame& pf,const CellPlan& cell,AdaptiveSample& out){
 if(!active_atlas_samples||!active_atlas_samples->contacts||cell.n_crossings!=2)return false;
 Q R=(Q)Rhi+Rlo;const ContactAnchor* closest=nullptr;Q dist=HUGE_VALQ;
 for(const auto& c:*active_atlas_samples->contacts){Q d=fabsq(R-d14_to_qf(c.R_center));if(d<dist){dist=d;closest=&c;}}
 if(!closest||R<closest->uniqueness.r0||R>closest->uniqueness.r1)return false;
 ++active_atlas_samples->attempts;auto pair=contact_pair(R,*closest,pf);if(!pair.valid)return false;
 auto center=local_fold_quantities(IQ(R,R),pair.m,pf,bool(pair.chart));if(center.P.lo<=0)return false; // complement arc stays on incumbent evaluator
 IQ w=pair.w,m=pair.m,root=sqrt_bound(w),den=IQ(1)+m*m-w;
 if(den.lo<=0)return false;
 IQ tangent=IQ(2)*root*inverse(den);IQ angle(down(2*atanq(tangent.lo)),up(2*atanq(tangent.hi)));
 Q f0=R*midpoint(angle),f0err=up(fabsq(R)*(angle.hi-angle.lo));IQ fh(0);double inner=0;
 if(u!=0){auto base=local_fold_quantities(IQ(R,R),IQ(0),pf,bool(pair.chart));IQ p4=base.Pssss*ratio<IQ>(1,24),p3=base.Psss*ratio<IQ>(1,6),p2=base.Ptt*IQ(.5);
  IQ d1=p3+IQ(2)*m*p4,d0=p2+IQ(2)*m*d1-(m*m-w)*p4;
  auto evaluate_rule=[&](const GC2Rule& rule,int n){IQ sum(0);IQ rm=IQ(R,R)-IQ(pf.a),rp=IQ(R,R)+IQ(pf.a);if(pair.chart)std::swap(rm,rp);
   for(int i=0;i<n;++i){IQ t=m+root*IQ(rule.x[i]),S=-(d0+t*(d1+t*p4)),A=IQ(1)+t*t,B=rm*rm+rp*rp*t*t;if(S.lo<=0||B.lo<=0)return IQ(-HUGE_VALQ,HUGE_VALQ);sum=sum+IQ(rule.w[i])*sqrt_bound(S)*inverse(A*sqrt_bound(A)*sqrt_bound(B));}
   return IQ(2)*inverse(IQ(p.rho))*w*sum;
  };
  auto hi=evaluate_rule(gc2_rule16(),16),lo=evaluate_rule(gc2_rule8(),8);if(!hi.valid()||!lo.valid())return false;
  inner=double(mag(hi-lo));if(inner>kHoloKRelTol*double(mag(hi)))return false;fh=hi;
 }
 double D=kPi*p.rho*p.rho*(1-u/3),scale=jac/D;
 out.value[0]=scale*double((1-u)*f0+u*midpoint(fh));out.inner[0]=std::fabs(scale*u)*inner;
 out.geometry[0]=std::fabs(scale)*double(f0err+fabsq(u)*(fh.hi-fh.lo));out.roundoff[0]=32*std::numeric_limits<double>::epsilon()*std::fabs(out.value[0]);
 out.reliable=std::isfinite(out.value[0])&&std::isfinite(out.geometry[0]);if(out.reliable)++active_atlas_samples->accepted;return out.reliable;
}
} // namespace
} // namespace
