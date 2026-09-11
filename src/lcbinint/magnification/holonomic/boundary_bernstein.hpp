#pragma once
#include "d14_positive_roots.hpp"
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math","fp-contract=off")
#endif
namespace lcbinint::holonomic::atlas_detail {
using positive_detail::Interval;using positive_detail::Ball;
using positive_detail::up;using positive_detail::down;using positive_detail::bounds;
using Q=__float128;using IQ=Interval<Q>;
struct Box { Q r0=0,r1=1,s0=-1,s1=1; int chart=0,depth=0; };
template<class I> struct Tensor { std::array<std::array<I,5>,7> c{};int nr=0,ns=0; };
template<class I> Tensor<I> tensor_constant(I v){Tensor<I> p;p.c[0][0]=v;return p;}
template<class I> Tensor<I> add(Tensor<I> a,const Tensor<I>& b){a.nr=std::max(a.nr,b.nr);a.ns=std::max(a.ns,b.ns);for(int i=0;i<=b.nr;++i)for(int j=0;j<=b.ns;++j)a.c[i][j]=a.c[i][j]+b.c[i][j];return a;}
template<class I> Tensor<I> neg(Tensor<I> a){for(auto& row:a.c)for(auto& v:row)v=-v;return a;}
template<class I> Tensor<I> mul(const Tensor<I>& a,const Tensor<I>& b){Tensor<I> c;c.nr=a.nr+b.nr;c.ns=a.ns+b.ns;if(c.nr>6||c.ns>4){c.nr=6;c.ns=4;c.c[0][0]=I(INFINITY);return c;}for(int i=0;i<=a.nr;++i)for(int j=0;j<=a.ns;++j)if(!a.c[i][j].zero())for(int k=0;k<=b.nr;++k)for(int l=0;l<=b.ns;++l)c.c[i+k][j+l]=c.c[i+k][j+l]+a.c[i][j]*b.c[k][l];return c;}
template<class I> Tensor<I> times(Tensor<I> a,I c){for(auto& row:a.c)for(auto& v:row)v=v*c;return a;}
inline int choose(int n,int k){int v=1;for(int j=1;j<=k;++j)v=v*(n-j+1)/j;return v;}
template<class I> I ratio(int n,int d){return positive_detail::point<I>((Q)n/d);}
// Denominator binomials are not generally dyadic: enclose the division too.
template<> inline Interval<double> ratio(int n,int d){double x=double(n)/d;return {down(x),up(x)};}
template<> inline IQ ratio(int n,int d){Q x=(Q)n/d;return {down(x),up(x)};}
template<> inline Ball ratio(int n,int d){auto b=positive_detail::point<Ball>((Q)n/d);b.r=up(b.r+double(fabsq(nextafterq((Q)n/d,HUGE_VALQ)-(Q)n/d)));return b;}
template<class I> Tensor<I> bernstein(const Tensor<I>& p){
 struct Weights {I w[7][7][7]{};Weights(){for(int n=0;n<=6;++n)for(int i=0;i<=n;++i)for(int k=0;k<=i;++k)w[n][i][k]=i==n||k==0?I(1):ratio<I>(choose(i,k),choose(n,k));}};
 static const Weights weights;
 Tensor<I> temp,b;temp.nr=b.nr=p.nr;temp.ns=b.ns=p.ns;
 for(int i=0;i<=p.nr;++i)for(int j=0;j<=p.ns;++j)for(int k=0;k<=i;++k)temp.c[i][j]=temp.c[i][j]+p.c[k][j]*weights.w[p.nr][i][k];
 for(int i=0;i<=p.nr;++i)for(int j=0;j<=p.ns;++j)for(int k=0;k<=j;++k)b.c[i][j]=b.c[i][j]+temp.c[i][k]*weights.w[p.ns][j][k];
 return b;
}
// Direct real T construction in LOCAL coordinates; no R-polynomial family.
// Reciprocal chart reverses homogeneous angular coefficients before evaluation.
template<class I> Tensor<I> boundary_power(const PrimaryFrame& pf,const Box& box){
 auto one=tensor_constant(I(1));Tensor<I> R,S;R.nr=1;R.c[0][0]=positive_detail::point<I>(box.r0);R.c[1][0]=(positive_detail::point<I>(box.r1)-positive_detail::point<I>(box.r0));S.ns=1;S.c[0][0]=positive_detail::point<I>(box.s0);S.c[0][1]=(positive_detail::point<I>(box.s1)-positive_detail::point<I>(box.s0));
 I a(pf.a),m(pf.m0),X(pf.X),Y(pf.Y),rho(pf.rho);
 auto R2=mul(R,R),n0r=times(R2,-X),n0i=times(R2,-Y);
 auto n1r=mul(R,add(R2,tensor_constant(I(-1)+a*X))),n1i=times(R,a*Y),n2=times(add(tensor_constant(m),neg(R2)),a);
 std::array<Tensor<I>,3> re={add(add(n0r,n1r),n2),times(n0i,I(2)),add(add(n1r,neg(n0r)),neg(n2))};
 std::array<Tensor<I>,3> im={add(n0i,n1i),times(add(n2,neg(n0r)),I(2)),add(n1i,neg(n0i))};
 if(box.chart){std::swap(re[0],re[2]);std::swap(im[0],im[2]);re[1]=neg(re[1]);im[1]=neg(im[1]);}
 auto S2=mul(S,S);auto tr=add(add(re[0],mul(re[1],S)),mul(re[2],S2));auto ti=add(add(im[0],mul(im[1],S)),mul(im[2],S2));
 auto minus=add(R,tensor_constant(-a)),plus=add(R,tensor_constant(a));auto bm=mul(minus,minus),bp=mul(plus,plus);if(box.chart)std::swap(bm,bp);
 auto pos=times(mul(mul(R2,add(one,S2)),add(bm,mul(bp,S2))),rho*rho);
 return add(pos,neg(add(mul(tr,tr),mul(ti,ti))));
}
template<class I> Tensor<I> boundary(const PrimaryFrame& pf,const Box& box){return bernstein(boundary_power<I>(pf,box));}
template<class I> Tensor<I> power_derivative(const Tensor<I>& p,int axis){Tensor<I> d;d.nr=axis? p.nr:std::max(0,p.nr-1);d.ns=axis?std::max(0,p.ns-1):p.ns;int n=axis?p.ns:p.nr;if(!n)return d;for(int i=0;i<=d.nr;++i)for(int j=0;j<=d.ns;++j)d.c[i][j]=p.c[i+(axis==0)][j+(axis==1)]*I(double(axis?j+1:i+1));return d;}
template<class I> Tensor<I> derivative(const Tensor<I>& b,int axis){Tensor<I> d;d.nr=axis==0?std::max(0,b.nr-1):b.nr;d.ns=axis==1?std::max(0,b.ns-1):b.ns;int n=axis?b.ns:b.nr;if(!n)return d;for(int i=0;i<=d.nr;++i)for(int j=0;j<=d.ns;++j)d.c[i][j]=(b.c[i+(axis==0)][j+(axis==1)]-b.c[i][j])*I(double(n));return d;}
template<class I> IQ hull(const Tensor<I>& b){IQ v(HUGE_VALQ,-HUGE_VALQ);for(int i=0;i<=b.nr;++i)for(int j=0;j<=b.ns;++j){auto a=bounds(b.c[i][j]);if(!a.valid())return {-HUGE_VALQ,HUGE_VALQ};v.lo=fminq(v.lo,a.lo);v.hi=fmaxq(v.hi,a.hi);}return v;}
template<class I> I evaluate(const Tensor<I>& b,I x,I y){std::array<I,7> c{};for(int i=0;i<=b.nr;++i){auto row=b.c[i];for(int n=b.ns;n>0;--n)for(int j=0;j<n;++j)row[j]=(I(1)-y)*row[j]+y*row[j+1];c[i]=row[0];}for(int n=b.nr;n>0;--n)for(int i=0;i<n;++i)c[i]=(I(1)-x)*c[i]+x*c[i+1];return c[0];}
template<class I> std::pair<Tensor<I>,Tensor<I>> split(const Tensor<I>& p,int axis,I fraction=I(.5)){bool half=false;if constexpr(std::is_same_v<I,Ball>)half=fraction.c.hi==.5&&fraction.c.lo==0&&fraction.r==0;else half=fraction.lo==.5&&fraction.hi==.5;Tensor<I> a=p,b=p;int n=axis?p.ns:p.nr,other=axis?p.nr:p.ns;for(int k=0;k<=other;++k){std::array<I,7> v{};for(int j=0;j<=n;++j)v[j]=axis?p.c[k][j]:p.c[j][k];for(int level=0;level<=n;++level){if(axis){a.c[k][level]=v[0];b.c[k][n-level]=v[n-level];}else{a.c[level][k]=v[0];b.c[n-level][k]=v[n-level];}for(int j=0;j<n-level;++j)v[j]=half?(v[j]+v[j+1])*I(.5):(I(1)-fraction)*v[j]+fraction*v[j+1];}}return {a,b};}
inline IQ inverse(IQ a){if(a.sign()!=1&&a.sign()!=-1)return {-HUGE_VALQ,HUGE_VALQ};return {down(1/a.hi),up(1/a.lo)};}
inline Q midpoint(IQ a){return a.lo+(a.hi-a.lo)/2;}
inline Q mag(IQ a){return fmaxq(fabsq(a.lo),fabsq(a.hi));}
// After a non-dyadic contraction the rounded physical midpoint need not
// represent exactly half the parent interval. Enclose the true split ratio.
// The EFT fast test is evaluated with contraction disabled by this header.
template<class I> I interval_cast(IQ v){
 if constexpr(std::is_same_v<I,IQ>)return v;
 else if constexpr(std::is_same_v<I,Interval<double>>)return {down(double(v.lo)),up(double(v.hi))};
 else{Q m=midpoint(v);I b=positive_detail::point<I>(m);Q r=fmaxq(up(m-v.lo),up(v.hi-m));b.r=up(b.r+up(double(r)));return b;}
}
template<class I> std::pair<Tensor<I>,Tensor<I>> split_physical(const Tensor<I>& p,const Box& box,int axis){
 Q a=axis?box.s0:box.r0,b=axis?box.s1:box.r1;
 Q sum=a+b,mid=sum/2,bv=sum-a,error=(a-(sum-bv))+(b-bv);
 if(error==0&&mid*2==sum)return split(p,axis);
 IQ f=(IQ(mid,mid)-IQ(a,a))*inverse(IQ(b,b)-IQ(a,a));
 return split(p,axis,interval_cast<I>(f));
}
struct LocalProof {int kind=0; // 0 unknown,1 no-boundary,2 no-tangency,3 unique
 IQ x{0,1},y{0,1};bool tube=false;};
template<class I> LocalProof prove_jets(const Tensor<I>& p,const Tensor<I>& py,const Tensor<I>& px,const Tensor<I>& pyy,const Tensor<I>& pyx,bool tubes=true){
 LocalProof out;if(hull(p).sign()==1||hull(p).sign()==-1){out.kind=1;return out;}
 if(hull(py).sign()==1||hull(py).sign()==-1){out.kind=2;return out;}

 IQ j[2][2]={{hull(px),hull(py)},{hull(pyx),hull(pyy)}};
 if(tubes&&j[1][1].sign()!=2&&j[1][1].sign()!=0){
  Tensor<I> l=py,h=py;l.ns=h.ns=0;for(int i=0;i<=py.nr;++i){l.c[i][0]=py.c[i][0];h.c[i][0]=py.c[i][py.ns];}
  int sl=hull(l).sign(),sh=hull(h).sign();
  if(sl*sh==-1&&sl!=2&&sh!=2&&(j[0][0].sign()==1||j[0][0].sign()==-1)){
   int fs[2]={2,2};for(int end=0;end<2;++end){Q lo=0,hi=1;
    for(int k=0;k<24;++k){Q m=(lo+hi)/2;int sm=bounds(evaluate(py,I(double(end)),positive_detail::point<I>(m))).sign();if(sm==2||sm==0)break;if(sm==sl)lo=m;else hi=m;}
    // Interval de Casteljau encloses all y in the stationary bracket.
    I yi;if constexpr(std::is_same_v<I,Ball>){yi=positive_detail::point<Ball>((lo+hi)/2);yi.r=up(yi.r+double((hi-lo)/2));}else yi={decltype(I{}.lo)(lo),decltype(I{}.hi)(hi)};
    fs[end]=bounds(evaluate(p,I(double(end)),yi)).sign();
   }
   if(fs[0]!=2&&fs[1]!=2&&fs[0]*fs[1]==1){out.kind=2;out.tube=true;return out;}
   if(fs[0]!=2&&fs[1]!=2&&fs[0]*fs[1]==-1){out.kind=3;out.tube=true;}
  }
 }
 // Krawczyk on [0,1]^2, using a point inverse only as a preconditioner.
 Q a=midpoint(j[0][0]),b=midpoint(j[0][1]),c=midpoint(j[1][0]),d=midpoint(j[1][1]);Q det=a*d-b*c;if(det==0||!finiteq(det))return out;
 Q Y[2][2]={{d/det,-b/det},{-c/det,a/det}};IQ f[2]={bounds(evaluate(p,I(.5),I(.5))),bounds(evaluate(py,I(.5),I(.5)))};
 IQ K[2];Q norm=0;for(int i=0;i<2;++i){K[i]=IQ(.5);Q row=0;for(int k=0;k<2;++k)K[i]=K[i]-IQ(Y[i][k],Y[i][k])*f[k];for(int l=0;l<2;++l){IQ v(double(i==l));for(int k=0;k<2;++k)v=v-IQ(Y[i][k],Y[i][k])*j[k][l];K[i]=K[i]+v*IQ(-.5,.5);row=up(row+mag(v));}norm=fmaxq(norm,row);}
 if(K[0].hi<0||K[0].lo>1||K[1].hi<0||K[1].lo>1){out.kind=2;return out;}
 out.x={fmaxq(0,K[0].lo),fminq(1,K[0].hi)};out.y={fmaxq(0,K[1].lo),fminq(1,K[1].hi)};
 if(norm<1&&K[0].lo>0&&K[0].hi<1&&K[1].lo>0&&K[1].hi<1)out.kind=3;
 return out;
}
template<class I> int coefficient_sign(const Tensor<I>& p){int sg=p.c[0][0].sign();if(sg!=1&&sg!=-1)return 2;for(int i=0;i<=p.nr;++i)for(int j=0;j<=p.ns;++j)if(p.c[i][j].sign()!=sg)return 2;return sg;}
template<class I> LocalProof prove(const Tensor<I>& p,bool tubes=true){
 LocalProof out;int sg=coefficient_sign(p);if(sg==1||sg==-1){out.kind=1;return out;}
 auto py=derivative(p,1);sg=coefficient_sign(py);if(sg==1||sg==-1){out.kind=2;return out;}
 return prove_jets(p,py,derivative(p,0),derivative(py,1),derivative(py,0),tubes);
}
template<class I> LocalProof prove_local(const PrimaryFrame& pf,const Box& box,bool tubes=true){auto p=boundary_power<I>(pf,box),py=power_derivative(p,1);return prove_jets(bernstein(p),bernstein(py),bernstein(power_derivative(p,0)),bernstein(power_derivative(py,1)),bernstein(power_derivative(py,0)),tubes);}
} // namespace

#if defined(__GNUC__)
#pragma GCC pop_options
#endif
