// Close-root geometry detector used by the compiled gradient API.
#pragma once
#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include "lcbinint/magnification/holonomic/cells.hpp"
#include "fused_geometry_screen.hpp"
namespace lcbinint::holonomic::geometry_experiment {
using Q=__float128;
struct Pair {double center=0,scale=0,left=0,right=0,band_lo=0,band_hi=0;bool real=false,existing=false;};
// Existing support edges are normal image endpoints. Only mutually closest
// INTERNAL physical folds are candidates for additional near-pair treatment.
inline std::vector<Pair> known_internal_pairs(const PrimaryFrame&pf,const TopologyResult&topo,int resolution){
 std::vector<Pair> out;
 std::optional<ScreenBlocks<ScreenInterval>> blocks;
 std::optional<ScreenBlocks<positive_detail::Ball>> precise;
 auto isolated=[&](const auto&source,double center,double radius){auto p=source.disk(pf,center,radius);double rest=0;for(int k=0;k<=p.degree;++k)if(k!=2)rest=screen_up(rest+screen_upper(p.c[k]));return screen_lower(p.c[2])>rest;};
 for(size_t first=0;first<topo.cells.size();){
  if(topo.cells[first].kind==ArcKind::kEmpty){++first;continue;}
  size_t last=first;while(last+1<topo.cells.size()&&topo.cells[last+1].kind!=ArcKind::kEmpty)++last;
  if(last-first<2){first=last+1;continue;}
  double lo=topo.cells[first].r_lo,hi=topo.cells[last].r_hi;
  std::vector<double> events;
  for(const auto&e:topo.events)if(e.physically_real&&e.radius>=lo&&e.radius<=hi)events.push_back(e.radius);
  std::sort(events.begin(),events.end());events.erase(std::unique(events.begin(),events.end()),events.end());
  for(size_t k=1;k+2<events.size();++k){double a=events[k],b=events[k+1],gap=b-a;
   if(!(a>lo&&b<hi&&gap<events[k]-events[k-1]&&gap<events[k+2]-events[k+1]&&.5*gap<(hi-lo)/resolution))continue;
   const Q av=Q(a)*a,bv=Q(b)*b;const double center=double((av+bv)/2),radius=std::nextafter(double(bv-av),INFINITY);
   if(!blocks)blocks.emplace(pf);bool ok=isolated(*blocks,center,radius);
   if(!ok){if(!precise)precise.emplace(pf);ok=isolated(*precise,center,radius);}
   if(ok){Pair pair;pair.center=a+gap*.5;pair.scale=gap*.5;pair.left=a;pair.right=b;pair.band_lo=lo;pair.band_hi=hi;pair.real=pair.existing=true;out.push_back(pair);}
  }
  first=last+1;
 }
 return out;
}
// Stationary isolation is local to seeded support bands; it does not certify
// global image support completeness. Solve D'(v)=0
// before splitting its quadratic normal form; the stationary point is broad
// even when D's two roots are too close for radial quadrature to see.
inline std::vector<Pair> detect(const PrimaryFrame&pf,const TopologyResult&topo,
                               const std::vector<Cplx<double>>&seeds,bool*complete_out=nullptr,int resolution=7) {
 bool completed=true;
 if(resolution<=0){if(complete_out)*complete_out=false;return {};}
 auto known=known_internal_pairs(pf,topo,resolution);
 std::vector<bool> needs;
 const char* screen_setting=std::getenv("LCBININT_GEOMETRY_SCREEN");
 if(!screen_setting||screen_setting[0]!='0'){
  needs=screen_bands(pf,topo,resolution);
  if(std::none_of(needs.begin(),needs.end(),[](bool b){return b;})){if(complete_out)*complete_out=true;if(!known.empty())++screen_stats.special_epochs;return known;}
 }
 ++screen_stats.fine_epochs;
 auto c=re_detail::d14_expanded_from_struct(re_detail::d14_struct_build(pf.a,pf.m0,pf.X,pf.Y,pf.rho));
 std::vector<Pair> out=std::move(known);if(c.size()<3){if(complete_out)*complete_out=false;return out;}
 auto jet=[&](Q v){std::array<Q,4> j{};for(int k=int(c.size())-1;k>=0;--k){j[3]=j[3]*v+3*j[2];j[2]=j[2]*v+2*j[1];j[1]=j[1]*v+j[0];j[0]=j[0]*v+c[k];}return j;};
 auto approximate_jet=[&](long double v){std::array<long double,3> j{};long double norm=0;for(int k=int(c.size())-1;k>=0;--k){j[2]=j[2]*v+2*j[1];j[1]=j[1]*v+j[0];j[0]=j[0]*v+(long double)c[k];norm=norm*std::fabs(v)+std::fabs((long double)c[k]);}return std::make_pair(j,norm);};
 std::vector<Q> descending(c.rbegin(),c.rend());const int degree=int(c.size())-1;
  using I=positive_detail::Interval<Q>;
  const auto polynomial=positive_detail::polynomial<I>(pf);
  auto dp=positive_detail::derivative(polynomial);
  // The epoch polynomial is identical for every occupied band. Construct
  // each precision once, rather than rebuilding D14 at every isolation call.
  const auto dp_double=positive_detail::derivative(positive_detail::polynomial<positive_detail::Interval<double>>(pf));

 for(size_t start=0;start<topo.cells.size();){
  if(topo.cells[start].kind==ArcKind::kEmpty || (!needs.empty()&&!needs[start])){++start;continue;}
  size_t end=start;while(end+1<topo.cells.size()&&topo.cells[end+1].kind!=ArcKind::kEmpty)++end;
  const double lo=topo.cells[start].r_lo,hi=topo.cells[end].r_hi;
  const Q vl=Q(lo)*lo,vh=Q(hi)*hi;
  // Bernstein/Descartes exclusion: same-sign coefficients prove there
  // is no stationary root on the interval. One variation isolates one root;
  // multiple variations are bisected with de Casteljau, without dense sampling.
  auto isolate=[&](auto prototype){
  using J=decltype(prototype);
  const auto&d=[&]() -> const positive_detail::Poly<J>& {
   if constexpr(std::is_same_v<J,I>)return dp;else return dp_double;
  }();
  bool complete=true;
  std::array<J,14> bern{};bern[0]=d.c[d.degree];int n=0;
  for(int k=d.degree-1;k>=0;--k){std::array<J,14> next{};
   using T=decltype(J{}.lo);J inv(positive_detail::down(T(1)/T(n+1)),positive_detail::up(T(1)/T(n+1)));
   for(int j=0;j<=n+1;++j){
    if(j<=n)next[j]=next[j]+bern[j]*positive_detail::point<J>(vl)*J(double(n+1-j))*inv;
    if(j)next[j]=next[j]+bern[j-1]*positive_detail::point<J>(vh)*J(double(j))*inv;
    next[j]=next[j]+d.c[k];
   }bern=next;++n;
  }
  struct Bracket{Q l,h;std::array<J,14> c;int depth;};
  std::vector<Bracket> pending{{vl,vh,bern,0}};
  std::vector<Q> stationary_roots;int subdivisions=0;
  while(!pending.empty()&&subdivisions++<512){auto b=pending.back();pending.pop_back();
   int variations=0,last=0;bool uncertain=false;
   for(int k=0;k<=n;++k){int sign=b.c[k].sign();if(sign==2){uncertain=true;break;}if(!sign)continue;if(last&&last!=sign)++variations;last=sign;}
   if(uncertain){complete=false;break;}
   if(!variations)continue;
   if(!uncertain&&variations==1){
    long double l=(long double)b.l,h=(long double)b.h,v=(l+h)/2;auto jl=approximate_jet(l).first;
    for(int i=0;i<48;++i){auto evaluated=approximate_jet(v);auto j=evaluated.first;long double step=j[2]!=0?j[1]/j[2]:h-l;
     const long double floor=256*std::numeric_limits<long double>::epsilon()*evaluated.second/std::max(std::fabs(j[2]),1e-4900L);
     if(std::fabs(step)<=std::max(floor,1e-16L*std::max(1.L,std::fabs(v))))break;
     if((j[1]>0)==(jl[1]>0)){l=v;jl=j;}else h=v;
     long double next=v-step;v=next>l&&next<h?next:(l+h)/2;
    }
    Q refined=v;bool converged=false;
    for(int i=0;i<16;++i){auto j=jet(refined);if(j[2]==0)break;Q step=j[1]/j[2];if(fabsq(step)<1e-18Q*fmaxq(1,fabsq(refined))){converged=true;break;}Q next=refined-step;if(!(next>=b.l&&next<=b.h))break;refined=next;}
    if(converged)stationary_roots.push_back(refined);else complete=false;continue;
   }
   if(b.depth>=48){complete=false;break;}
   std::array<J,14> left{},right{},work=b.c;left[0]=work[0];right[n]=work[n];
   for(int k=1;k<=n;++k){for(int j=0;j<=n-k;++j)work[j]=(work[j]+work[j+1])*J(.5);left[k]=work[0];right[n-k]=work[n-k];}
   Q mid=(b.l+b.h)/2;pending.push_back({b.l,mid,left,b.depth+1});pending.push_back({mid,b.h,right,b.depth+1});
  }
  if(!pending.empty())complete=false;
  return std::make_pair(stationary_roots,complete);
  };
  auto isolated=isolate(positive_detail::Interval<double>{});
  if(!isolated.second)isolated=isolate(I{});
  if(!isolated.second){
   // Bernstein coefficients can straddle zero at a subdivision boundary.
   // Use signed Sturm variations as a bounded independent isolation fallback.
   positive_detail::Chain<I> chain;
   if(chain.build(dp)){
    struct SBracket{Q l,h;int vl,vh;};std::vector<SBracket> pending;
    int a=chain.variation(vl,1),b=chain.variation(vh,-1);
    bool ok=a>=b&&b>=0;if(ok)pending.push_back({vl,vh,a,b});
    std::vector<Q> roots;int steps=0;
    while(!pending.empty()&&steps++<512){auto bracket=pending.back();pending.pop_back();int n=bracket.vl-bracket.vh;if(!n)continue;
     if(n==1){Q l=bracket.l,h=bracket.h,v=(l+h)/2;auto jl=jet(l);bool found=false;
      for(int j=0;j<128;++j){auto z=jet(v);Q step=z[2]!=0?z[1]/z[2]:h-l;
       if(fabsq(step)<1e-18Q*fmaxq(1,fabsq(v)) || h-l<1e-20Q*fmaxq(1,fabsq(v))){found=true;break;}
       if((z[1]>0)==(jl[1]>0)){l=v;jl=z;}else h=v;Q next=v-step;v=next>l&&next<h?next:(l+h)/2;
      }if(found)roots.push_back(v);else ok=false;continue;
     }
     Q mid=(bracket.l+bracket.h)/2;int count=chain.variation(mid,1);
     if(count<0||count>bracket.vl||count<bracket.vh){ok=false;break;}
     pending.push_back({bracket.l,mid,bracket.vl,count});pending.push_back({mid,bracket.h,count,bracket.vh});
    }
    if(!pending.empty())ok=false;if(ok)isolated={roots,true};
   }
  }
  if(!isolated.second)completed=false;
  const auto&stationary_roots=isolated.first;
  for(Q v:stationary_roots){
   auto j=jet(v);if(j[2]==0)continue;
   const Q eta=2*j[0]/j[2];if(eta==0)continue;const Q b=sqrtq(fabsq(eta));
   // Require the local quadratic pair to lie inside this support band.
   if(!(v-b>vl&&v+b<vh))continue;
   // Rouche: the quadratic term dominates all other Taylor terms on
   // |z-v|=2*b. Interval coefficients bound the original exact dyadic input.
   // This proves exactly two discriminant roots in the disk, and rejects
   // clusters where the quadratic normal form is not locally isolated.
   std::array<I,15> shifted{};int nd=0;shifted[0]=polynomial.c[polynomial.degree];
   for(int k=polynomial.degree-1;k>=0;--k){shifted[nd+1]=shifted[nd];for(int j=nd;j>0;--j)shifted[j]=shifted[j]*I(v)+shifted[j-1];shifted[0]=shifted[0]*I(v)+polynomial.c[k];++nd;}
   I radius(2*b),power(1),remainder{};Q main_lower=0;
   for(int k=0;k<=nd;++k){I term=shifted[k]*power;
    if(k==2)main_lower=term.lo>0?term.lo:term.hi<0?-term.hi:0;
    else remainder=remainder+I(0,fmaxq(fabsq(term.lo),fabsq(term.hi)));
    power=power*radius;
   }
   if(!(main_lower>remainder.hi))continue;
   int known=0;for(const auto&e:topo.events)if(e.physically_real){Q rr=Q(e.radius)+e.radius_lo;if(fabsq(rr*rr-v)<2*b)++known;}
   if(known>=2)continue;
   Cplx<Q> z[2]={eta>0?Cplx<Q>(v,b):Cplx<Q>(v-b,0),eta>0?Cplx<Q>(v,-b):Cplx<Q>(v+b,0)};
   bool good=true;
   for(auto&root:z){bool converged=false;for(int i=0;i<24;++i){auto f=poly_eval_c(descending.data(),degree,root),df=polyder_eval_c(descending.data(),degree,root);if(cabs2(df)==0)break;auto step=f/df;root=root-step;
     if(cabs(step)<1e-23Q*fmaxq(1,cabs(root))){converged=true;break;}}
    good=good&&converged&&root.re>vl&&root.re<vh;
   }
   if(!good)continue;
   auto sqrtc=[](Cplx<Q> z){Q re=sqrtq((cabs(z)+z.re)/2);return Cplx<Q>(re,z.im/(2*re));};
   auto a=sqrtc(z[0]),broot=sqrtc(z[1]);Pair pair;pair.real=eta<0;
   pair.center=double((a.re+broot.re)/2);pair.scale=pair.real?double(fabsq(a.re-broot.re)/2):double(fabsq(a.im));
   pair.left=double(fminq(a.re,broot.re));pair.right=double(fmaxq(a.re,broot.re));pair.band_lo=lo;pair.band_hi=hi;
   if(pair.real&&!(pair.left<pair.right)){completed=false;continue;}
   if(!(pair.scale>0&&pair.scale<.5*(hi-lo)))continue;
   // The Newton roots must retain the proposed real/conjugate pair structure.
   if(pair.real && (fabsq(a.im)>1e-20Q || fabsq(broot.im)>1e-20Q))continue;
   if(!pair.real && (fabsq(a.re-broot.re)>1e-18Q || fabsq(a.im+broot.im)>1e-18Q))continue;
   bool duplicate=false;for(auto old:out)if(std::fabs(old.center-pair.center)<1e-12*(1+pair.center))duplicate=true;
   if(!duplicate)out.push_back(pair);
  }
  start=end+1;
 }
 if(!out.empty())++screen_stats.special_epochs;
 if(complete_out)*complete_out=completed;
 std::sort(out.begin(),out.end(),[](auto a,auto b){return a.center<b.center;});return out;
}
}
