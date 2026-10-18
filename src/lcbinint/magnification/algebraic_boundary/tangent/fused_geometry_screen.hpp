#pragma once
#include "lcbinint/magnification/algebraic_boundary/tangent/cells.hpp"
#include <optional>
#include <cstdint>
#include <cstring>
#include "lcbinint/magnification/algebraic_boundary/tangent/d14_real_sturm.hpp"
namespace lcbinint::algebraic_boundary::tangent::geometry_experiment {
// Same nextafter enclosure as Interval<double>, with an inlined IEEE step.
// The screen is disabled when the floating-point environment is unsuitable.
inline double screen_up(double x){
 static_assert(sizeof(double)==sizeof(std::uint64_t)&&std::numeric_limits<double>::is_iec559);
 std::uint64_t bits;std::memcpy(&bits,&x,sizeof(bits));
 const auto absolute=bits&UINT64_C(0x7fffffffffffffff);
 if(absolute==0){bits=1;}
 else if(absolute>UINT64_C(0x7ff0000000000000)||bits==UINT64_C(0x7ff0000000000000))return x;
 else if(bits>>63)--bits;else ++bits;
 std::memcpy(&x,&bits,sizeof(bits));return x;
}
inline double screen_down(double x){return -screen_up(-x);}
struct ScreenInterval {
 double lo=0,hi=0;
 ScreenInterval()=default;ScreenInterval(double x):lo(x),hi(x){}ScreenInterval(double l,double h):lo(l),hi(h){}
 bool zero()const{return lo==0&&hi==0;}
 int sign()const{return !std::isfinite(lo)||!std::isfinite(hi)||lo>hi?2:lo>0?1:hi<0?-1:zero()?0:2;}
};
inline ScreenInterval operator-(ScreenInterval a){return {-a.hi,-a.lo};}
inline ScreenInterval operator+(ScreenInterval a,ScreenInterval b){if(a.zero())return b;if(b.zero())return a;return {screen_down(a.lo+b.lo),screen_up(a.hi+b.hi)};}
inline ScreenInterval operator-(ScreenInterval a,ScreenInterval b){return a+(-b);}
inline ScreenInterval operator*(ScreenInterval a,ScreenInterval b){
 if(a.zero()||b.zero())return {};
 double lo,hi;
 // A sign-known interval needs only the two extremal products.
 if(a.lo>=0){
  if(b.lo>=0){lo=a.lo*b.lo;hi=a.hi*b.hi;}
  else if(b.hi<=0){lo=a.hi*b.lo;hi=a.lo*b.hi;}
  else {lo=a.hi*b.lo;hi=a.hi*b.hi;}
 }else if(a.hi<=0){
  if(b.lo>=0){lo=a.lo*b.hi;hi=a.hi*b.lo;}
  else if(b.hi<=0){lo=a.hi*b.hi;hi=a.lo*b.lo;}
  else {lo=a.lo*b.hi;hi=a.lo*b.lo;}
 }else{
  if(b.lo>=0){lo=a.lo*b.hi;hi=a.hi*b.hi;}
  else if(b.hi<=0){lo=a.hi*b.lo;hi=a.lo*b.lo;}
  else {lo=std::min(a.lo*b.hi,a.hi*b.lo);hi=std::max(a.lo*b.lo,a.hi*b.hi);}
 }
 return {screen_down(lo),screen_up(hi)};
}
inline ScreenInterval scale(ScreenInterval a,int exponent){if(a.zero())return a;return {screen_down(std::ldexp(a.lo,exponent)),screen_up(std::ldexp(a.hi,exponent))};}
inline double screen_upper(ScreenInterval a){return std::max(std::fabs(a.lo),std::fabs(a.hi));}
inline double screen_lower(ScreenInterval a){return a.lo>0?a.lo:a.hi<0?-a.hi:0;}
// Cheap enclosure of an arithmetic expression. `mag` evaluates the same
// expression with absolute inputs and positive additions. `depth` bounds the
// number of rounding factors in every monomial. With exact binary64 inputs,
// |computed-exact| <= gamma_n/(1-gamma_n)*computed_majorant. We use the
// larger 2*n*u/(1-2*n*u) bound. Subnormal majorants/overflow are inconclusive.
struct ScreenEstimate {
 double c=0,mag=0;unsigned depth=0;
 ScreenEstimate()=default;ScreenEstimate(double x):c(x),mag(std::fabs(x)){if(mag>0&&mag<std::numeric_limits<double>::min())depth=1000001;}
 bool zero()const{return mag==0&&depth==0;}
 double error()const{
  if(depth>1000000||!std::isfinite(c)||!std::isfinite(mag))return INFINITY;
  if(depth==0)return 0;
  const double nu=2*double(depth)*0x1p-53;
  return screen_up(screen_up(nu/screen_down(1-nu))*mag);
 }
 int sign()const{const double e=error();return !std::isfinite(e)?2:c>e?1:c<-e?-1:zero()?0:2;}
};
inline ScreenEstimate operator-(ScreenEstimate a){a.c=-a.c;return a;}
inline ScreenEstimate operator+(ScreenEstimate a,ScreenEstimate b){if(a.zero())return b;if(b.zero())return a;ScreenEstimate out;out.c=a.c+b.c;out.mag=a.mag+b.mag;out.depth=std::min(1000001u,std::max(a.depth,b.depth)+1);if(out.c!=0&&std::fabs(out.c)<std::numeric_limits<double>::min())out.depth=1000001;return out;}
inline ScreenEstimate operator-(ScreenEstimate a,ScreenEstimate b){return a+(-b);}
inline ScreenEstimate operator*(ScreenEstimate a,ScreenEstimate b){if(a.zero()||b.zero())return {};ScreenEstimate out;out.c=a.c*b.c;out.mag=a.mag*b.mag;out.depth=a.depth>1000000||b.depth>1000000?1000001:std::min(1000001u,a.depth+b.depth+1);if(out.mag<std::numeric_limits<double>::min()||(out.c==0&&a.c!=0&&b.c!=0)||(out.c!=0&&std::fabs(out.c)<std::numeric_limits<double>::min()))out.depth=1000001;return out;}
inline ScreenEstimate scale(ScreenEstimate a,int exponent){if(a.zero())return a;const double before=a.c;a.c=std::ldexp(a.c,exponent);a.mag=std::ldexp(a.mag,exponent);if(a.mag<std::numeric_limits<double>::min()||(a.c==0&&before!=0)||(a.c!=0&&std::fabs(a.c)<std::numeric_limits<double>::min()))a.depth=1000001;return a;}
inline double screen_upper(ScreenEstimate a){return screen_up(std::fabs(a.c)+a.error());}
inline double screen_lower(ScreenEstimate a){return std::max(0.,screen_down(std::fabs(a.c)-a.error()));}
// Symmetric convolution evaluates each off-diagonal product once. This
// is the same polynomial identity, with bounds supplied by scalar arithmetic.
template<class I> inline positive_detail::Poly<I> screen_square(const positive_detail::Poly<I>&p){
 positive_detail::Poly<I> out;out.degree=2*p.degree;
 if(out.degree>14){out.degree=14;out.c[0]=I(INFINITY);return out;}
 for(int i=0;i<=p.degree;++i){out.c[2*i]=out.c[2*i]+p.c[i]*p.c[i];
  for(int j=i+1;j<=p.degree;++j)out.c[i+j]=out.c[i+j]+I(2)*(p.c[i]*p.c[j]);}
 return out;
}
// Build the disk polynomial before expanding the low-degree blocks. Shifting
// an already-expanded D14 loses the small coefficients around narrow images.
template<class I> struct ScreenBlocks {
 positive_detail::Poly<I> C,L,U,E,B;I factor;
 explicit ScreenBlocks(const PrimaryFrame&pf){
  I a(pf.a),m(pf.m0),x(pf.X),y(pf.Y),rho(pf.rho),one(1),a2=a*a,beta=x*x+y*y-rho*rho;
  C.degree=3;C.c[0]=a2*m*m;C.c[1]=one-I(2)*a2*m+a2*beta+I(2)*a*x*(I(3)*m-one);C.c[2]=I(-2)+a2+beta-I(4)*a*x;C.c[3]=one;
  L.degree=2;L.c[0]=-a2*m;L.c[1]=a2-one;L.c[2]=one;
  U.degree=2;U.c[0]=a*m-a2*m*x;U.c[1]=-a*(m+one)+x*(a2-one)+a*beta;U.c[2]=a+x;
  E.degree=1;E.c[0]=-m;E.c[1]=one;
  B.degree=2;B.c[0]=a2*m*m;B.c[1]=I(2)*a*m*(x-a);B.c[2]=(x-a)*(x-a)+y*y-rho*rho;
  factor=a2*(one-m)*y*y;
 }
 I value(const PrimaryFrame&pf,double v0)const{
  using namespace positive_detail;I v(v0),a(pf.a),x(pf.X),y(pf.Y),a2=a*a;
  I c=eval(C,v),l=eval(L,v),u=eval(U,v),e=eval(E,v),b=eval(B,v);
  I g=u*u+y*y*l*l-I(4)*a*x*e*c-I(4)*a2*(I(4)*x*x+y*y)*v*e*e;
  I z=factor*e*b,cc=c*c,vg=v*g;
  return (cc-I(4)*vg)*g*g+I(8)*c*(I(2)*cc-I(9)*vg)*z-I(432)*v*v*z*z;
 }
 positive_detail::Poly<I> disk(const PrimaryFrame&pf,double center,double radius)const{
  using namespace positive_detail;
  Poly<I> V;V.degree=1;V.c[0]=I(center);V.c[1]=I(radius);
  auto translate=[&](const Poly<I>&p){auto out=constant(p.c[p.degree]);for(int k=p.degree-1;k>=0;--k)out=add(mul(out,V),constant(p.c[k]));return out;};
  const auto c=translate(C),l=translate(L),u=translate(U),e=translate(E),b=translate(B);
  I a(pf.a),x(pf.X),y(pf.Y),a2=a*a;
  auto G=add(add(screen_square(u),mul(constant(y*y),screen_square(l))),add(mul(constant(I(-4)*a*x),mul(e,c)),mul(constant(I(-4)*a2*(I(4)*x*x+y*y)),mul(V,screen_square(e)))));
  auto Z=mul(constant(factor),mul(e,b)),CC=screen_square(c),VG=mul(V,G);
  return add(add(mul(add(CC,times(VG,-4)),screen_square(G)),times(mul(mul(c,add(times(CC,2),times(VG,-9))),Z),8)),times(mul(screen_square(V),screen_square(Z)),-432));
 }
};
inline double screen_upper(positive_detail::Interval<double> a){return std::max(std::fabs(a.lo),std::fabs(a.hi));}
inline double screen_lower(positive_detail::Interval<double> a){return a.lo>0?a.lo:a.hi<0?-a.hi:0;}
inline double screen_upper(positive_detail::Ball a){return positive_detail::up(positive_detail::up(std::fabs(a.c.hi)+std::fabs(a.c.lo))+a.r);}
inline double screen_lower(positive_detail::Ball a){return std::max(0.,positive_detail::down(positive_detail::down(std::fabs(a.c.hi)-std::fabs(a.c.lo))-a.r));}
struct ScreenStats {unsigned long epochs=0,bands=0,clear=0,disks=0,dominance_failed=0,witness_failed=0,double_counts=0,precise_counts=0,epochs_clear=0,fine_epochs=0,special_epochs=0;};
inline thread_local ScreenStats screen_stats;
// A successful screen proves that each covering disk contains only distinct
// real roots already bracketed around physical events. It never interprets an
// inconclusive arithmetic sign or a failed Pellet test as absence of a root.
inline std::vector<bool> screen_bands(const PrimaryFrame&pf,const TopologyResult&topo,int resolution){
 // Start with the direct interval certificate. The expression-majorant
 // prepass usually failed on these narrow bands and repeated the same disk
 // expansion before interval arithmetic. Inconclusive intervals still escalate.
 using I=ScreenInterval;using Q=__float128;
 std::vector<bool> needs(topo.cells.size(),true);++screen_stats.epochs;
 if(resolution<=0||!positive_detail::environment_ok())return needs;
 const ScreenBlocks<I> blocks(pf);
 std::optional<ScreenBlocks<positive_detail::Ball>> precise_blocks;
 struct Witness {double radius=0,lo=0,hi=0;int valid=-1;};
 std::vector<Witness>witnesses;
 for(const auto&e:topo.events)if(e.physically_real&&e.radius>0)witnesses.push_back({e.radius});
 std::sort(witnesses.begin(),witnesses.end(),[](auto a,auto b){return a.radius<b.radius;});
 witnesses.erase(std::unique(witnesses.begin(),witnesses.end(),[](auto a,auto b){return a.radius==b.radius;}),witnesses.end());
 for(size_t k=0;k<witnesses.size();++k){auto&w=witnesses[k];
  double gap=w.radius;
  if(k)gap=std::min(gap,w.radius-witnesses[k-1].radius);
  if(k+1<witnesses.size())gap=std::min(gap,witnesses[k+1].radius-w.radius);
  // These disjoint real brackets contain the event proposals, but proposals
  // alone do not establish root existence: opposite interval signs below do.
  const double l=w.radius-gap*.25,h=w.radius+gap*.25;
  w.lo=positive_detail::down(l*l);w.hi=positive_detail::up(h*h);
 }
 auto disk_clear=[&](Q left,Q right,Q imag,double padding=2.){
  ++screen_stats.disks;
  const double center=double((left+right)/2);
  Q extent=fmaxq(fabsq(left-Q(center)),fabsq(right-Q(center)));
  // A padded disk keeps known physical roots away from its boundary.
  // The padding changes proof efficiency only; acceptance still requires a
  // strict root count and disjoint real-root witnesses.
  // Outward binary64 arithmetic encloses the quad extents. IEEE sqrt
  // is correctly rounded in the environment accepted above; the successor
  // therefore bounds it from above. Avoid a software quad square root on
  // every ordinary band. Overflow remains inconclusive below.
  const double ex=screen_up(double(extent)),im=screen_up(double(imag));
  const double squared=screen_up(screen_up(ex*ex)+screen_up(im*im));
  const double radius=padding*screen_up(std::sqrt(squared));
  if(!(radius>0&&std::isfinite(radius)))return false;
  auto dominance=[&](const auto&source){
   using namespace positive_detail;
   auto t=source.disk(pf,center,radius);
   using Scalar=std::decay_t<decltype(t.c[0])>;
   for(int step=0;step<4;++step){
    std::array<double,15> upper{},lower{},prefix{},suffix{};double maximum=0;
    for(int k=0;k<=t.degree;++k){upper[k]=screen_upper(t.c[k]);maximum=std::max(maximum,upper[k]);}
    for(int k=0;k<t.degree;++k)prefix[k+1]=screen_up(prefix[k]+upper[k]);
    for(int k=t.degree;k>0;--k)suffix[k-1]=screen_up(suffix[k]+upper[k]);
    for(int k=0;k<=t.degree;++k){lower[k]=screen_lower(t.c[k]);if(lower[k]>screen_up(prefix[k]+suffix[k]))return k;}
    if(step==3){
     // If interval bounds already rule out every possible dominant term,
     // more arithmetic precision cannot make this SAME Pellet test pass.
     // Fall back to the detailed detector without rebuilding the proof.
     std::array<double,15> prefix_lower{},suffix_lower{};
     for(int k=0;k<t.degree;++k)prefix_lower[k+1]=screen_down(prefix_lower[k]+lower[k]);
     for(int k=t.degree;k>0;--k)suffix_lower[k-1]=screen_down(suffix_lower[k]+lower[k]);
     bool resolved=true;
     for(int k=0;k<=t.degree;++k){const double rest_lower=screen_down(prefix_lower[k]+suffix_lower[k]);if(!std::isfinite(upper[k])||upper[k]>rest_lower){resolved=false;break;}}
     return resolved?-2:-1;
    }
    if(!(maximum>0&&std::isfinite(maximum)))return -1;
    int exponent=0;std::frexp(maximum,&exponent);for(int k=0;k<=t.degree;++k)t.c[k]=scale(t.c[k],-exponent);
    // Graeffe maps every root z to z^2, preserving the number inside the
    // unit circle. Pellet dominance after squaring is still a root count
    // certificate, but separates moduli that the first test cannot resolve.
    Poly<Scalar> even,odd;even.degree=t.degree/2;odd.degree=(t.degree-1)/2;
    for(int k=0;k<=t.degree;++k)(k&1?odd.c[k/2]:even.c[k/2])=t.c[k];
    t=add(screen_square(even),neg(shift(screen_square(odd),1)));
   }
   return -1;
  };
  int count=dominance(blocks);
  if(count>=0)++screen_stats.double_counts;
  if(count==-1){++screen_stats.precise_counts;if(!precise_blocks)precise_blocks.emplace(pf);count=dominance(*precise_blocks);}
  if(count<0){++screen_stats.dominance_failed;return false;}
  if(count==0)return true;
  int established=0;double last_witness_hi=-INFINITY;
  for(auto&w:witnesses){
   const Q event=Q(w.radius)*w.radius,margin=Q(radius)-fabsq(event-Q(center));
   if(!(margin>0))continue;
   const double l=std::max(w.lo,double(event-margin/2)),h=std::min(w.hi,double(event+margin/2));
   if(l>w.lo||h<w.hi){w.lo=l;w.hi=h;w.valid=-1;}
   const Q distance=fmaxq(fabsq(Q(w.lo)-Q(center)),fabsq(Q(w.hi)-Q(center)));
   if(!(distance<Q(radius))||!(w.hi>w.lo))continue;
   if(w.valid<0){int l=blocks.value(pf,w.lo).sign(),h=blocks.value(pf,w.hi).sign();
    if((l!=1&&l!=-1)||(h!=1&&h!=-1)){if(!precise_blocks)precise_blocks.emplace(pf);l=precise_blocks->value(pf,w.lo).sign();h=precise_blocks->value(pf,w.hi).sign();}
    w.valid=(l==1||l==-1)&&(h==1||h==-1)&&l==-h;}
   if(w.valid&&w.lo>last_witness_hi){++established;last_witness_hi=w.hi;}
  }
  if(established!=count){++screen_stats.witness_failed;return false;}
  return true;
 };
 for(size_t first=0;first<topo.cells.size();){
  if(topo.cells[first].kind==ArcKind::kEmpty){needs[first++]=false;continue;}
  size_t last=first;while(last+1<topo.cells.size()&&topo.cells[last+1].kind!=ArcKind::kEmpty)++last;
  ++screen_stats.bands;
  const Q a=topo.cells[first].r_lo,b=topo.cells[last].r_hi;
  const Q delta=(b-a)/Q(resolution),imag=2*(b+delta)*delta;
  // An overlarge circumscribed disk can include unrelated complex roots.
  // If that proof is inconclusive, cover the SAME target rectangle by
  // overlapping smaller disks. Every disk must prove that all its roots are
  // already bracketed distinct real folds; otherwise retain detailed detection.
  // The structured polynomial blocks and certified witnesses above are reused.
  auto cover_clear=[&](Q left,Q right){
   if(disk_clear(left,right,imag))return true;
   if(!(imag>0))return false;
   const int pieces=std::max(2,int(ceilq((right-left)/(2*imag))));
   Q l=left;
   for(int j=1;j<=pieces;++j){
    const Q r=j==pieces?right:left+(right-left)*j/pieces;
    if(!disk_clear(l,r,imag,1.))return false;
    l=r;
   }
   return true;
  };
  bool clear=cover_clear(a*a-delta*delta,b*b);
  if(!clear&&first!=last){clear=true;for(size_t k=first;k<=last;++k){Q l=topo.cells[k].r_lo,h=topo.cells[k].r_hi;if(!cover_clear(l*l-delta*delta,h*h)){clear=false;break;}}}
  if(clear){++screen_stats.clear;for(size_t k=first;k<=last;++k)needs[k]=false;}
  first=last+1;
 }
 if(std::none_of(needs.begin(),needs.end(),[](bool b){return b;}))++screen_stats.epochs_clear;
 return needs;
}
}
