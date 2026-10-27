#include "lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include <iostream>
#include <random>
using namespace lcbinint;
namespace tr=algebraic_boundary::triple;
namespace rs=tr::regular_support_detail;
using Q=__float128;using CQ=tr::coefficient_detail::C<Q>;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
CQ exact_image(CQ z,Q phase,const tr::Geometry&p){
 const CQ target{Q(p.X)+Q(p.rho)*cosq(phase),Q(p.Y)+Q(p.rho)*sinq(phase)};
 const auto lens=p.lenses();std::array<Q,3>m{Q(1)-Q(p.m1)-Q(p.m2),Q(p.m1),Q(p.m2)};
 for(int k=0;k<16;++k){CQ f=z-target,shear{};for(int j=0;j<3;++j){CQ d=tr::coefficient_detail::conj(z-CQ{Q(lens[j].real()),Q(lens[j].imag())});CQ v=tr::coefficient_detail::conj(d)*CQ{Q(1)/(d.re*d.re+d.im*d.im)};f=f-CQ{m[j]}*v;shear=shear+CQ{m[j]}*v*v;}Q det=1-shear.re*shear.re-shear.im*shear.im;CQ step=(f-shear*tr::coefficient_detail::conj(f))*CQ{Q(1)/det};z=z-step;if(fabsq(step.re)+fabsq(step.im)<Q(1e-30))break;}
 return z;
}
int main(){
 std::vector<tr::Geometry> geometries{{.5,.4,.03,.2,.08,1.,-.4,.6},
 {.1,.05,.01,.0009989012086704624,.00009989012086704624,1.2,.7,.3},
 {.5,.2886751345948129,.01,1./3,1./3,1.,.5,.8660254037844386}};
 std::mt19937_64 random(674103);magnification::PointSourceMagnifier point;
 int accepted=0,counts=0,multiple=0,motion_checks=0,values=0,bounded=0,max_count=0;
 for(auto base:geometries){
  rs::Tree tree(base);tr::PreparedRegularSupport prepared;int local_values=0;
  require(!prepared.plan(base,1e-3,1).completed,"prepared event budget must be respected");
  for(int k=0;k<40;++k){auto p=base;if(k){p.X=-.6+1.8*double(random()%10000)/10000;p.Y=-.5+1.6*double(random()%10000)/10000;p.rho=.001+.009*double(random()%1000)/1000;}
   const int expected=tree.count({p.X,p.Y},p.rho);
   model::TripleLensGeometry geometry;geometry.masses=p.masses();auto lenses=p.lenses();for(int j=0;j<3;++j)geometry.lens_positions[j]={lenses[j].real(),lenses[j].imag()};
   // Also pull the same model implementation into the standalone audit link.
   const auto witness=model::triple_lens_equation(geometry,{1.73,.43});require(std::isfinite(witness.x+witness.y),"finite model witness");
   const auto images=point.triple_images(geometry,{p.X,p.Y});
   if(expected>=0){++counts;multiple+=expected>4;max_count=std::max(max_count,expected);require(images.size()==std::size_t(expected),"caustic winding image count disagrees with point solver");}
   if(expected>=0) {
    const auto before=tree.witness_hits();
    require(tree.count({p.X,p.Y},p.rho)==expected && tree.witness_hits()==before+1,"caustic witness count not reused");
   }
   auto plan=prepared.plan(p,1e-3);if(!plan.completed)continue;++accepted;bounded+=!plan.excluded_images.empty();
   auto full=tr::radial_event_plan(p,1000000,1e-6);require(full.completed,"full reference event plan");
   int retained=0;
   for(auto event:full.events){double theta=(event.reciprocal?rs::pi:0)+2*std::atan(event.coordinate);auto z=std::polar(event.radius,theta);bool excluded=false;for(auto b:plan.excluded_images)excluded=excluded || std::abs(z-b.center)<b.radius;if(excluded)continue;++retained;bool found=false;
    for(auto fast:plan.events){double angle=(fast.reciprocal?rs::pi:0)+2*std::atan(fast.coordinate);double dh=2*(fast.coordinate_hi-fast.coordinate_lo)+1e-11;if(event.radius_hi>=fast.radius_lo && event.radius_lo<=fast.radius_hi && std::abs(std::remainder(theta-angle,2*rs::pi))<dh)found=true;}
    require(found,"source-boundary plan missed a globally isolated event");
   }
   require(retained==int(plan.events.size()),"unexpected duplicate or missing stationary event");
   if(k<12)for(auto image:images){double phi=.37,half=.005;auto center=Complex(p.X,p.Y)+std::polar(p.rho,phi);auto z=rs::solve_image({image.position.x,image.position.y},center,p);auto local=p;local.X=center.real();local.Y=center.imag();local.rho=2*p.rho*std::sin(half*.5)+rs::eps*(1+std::abs(center)+p.rho);auto ball=rs::image_ball(z,local);if(!ball)continue;auto enclosure=rs::arc_motion(*ball,phi,half,p);if(!enclosure)continue;
    for(double offset:{-half,0.,half}){Q angle=Q(phi)+Q(offset);auto root=exact_image({Q(ball->z.real()),Q(ball->z.imag())},angle,p);require(hypotq(root.re-Q(ball->z.real()),root.im-Q(ball->z.imag()))<Q(ball->r),"IFT disk lost a quad image");auto f=rs::stationary_motion(root,cosq(angle),sinq(angle),p);require(bool(f),"quad stationary numerator");require(f->first>=Q(enclosure->first.lo)&&f->first<=Q(enclosure->first.hi)&&f->second>=Q(enclosure->second.lo)&&f->second<=Q(enclosure->second.hi),"stationary numerator forward bound");++motion_checks;}
   }
   if(local_values<4){for(double limb:{0.,.5}){tr::IntegrationOptions strict;strict.absolute_tolerance=0;strict.relative_tolerance=1e-7;auto reference=tr::integrate(p,limb,strict);require(reference.converged,"strict value reference");for(double tolerance:{1e-3,1e-4}){auto fast=prepared.plan(p,tolerance);require(fast.completed,"selected prepared support");auto options=strict;options.relative_tolerance=tolerance;auto value=tr::integrate(p,limb,options,&fast);require(value.converged,"prepared value convergence");require(std::abs(value.magnification-reference.magnification)<=value.error_estimate+reference.error_estimate+1e-10,"prepared value outside reported error");require(std::abs(value.magnification-reference.magnification)<=tolerance*reference.magnification,"prepared relative accuracy");}}++values;++local_values;}
  }
 }
 // Overlapping radial ranges must still integrate as separate certified
 // image domains. This reuses the ordinary weighted panel rule, not a point
 // source or single-annulus LD approximation.
 {
  auto p=geometries.front();tr::PreparedRegularSupport split_prepared;
  auto plan=split_prepared.plan(p,1e-4);require(plan.completed,"component fixture plan");
  tr::IntegrationOptions opt;opt.absolute_tolerance=0;opt.relative_tolerance=1e-4;
  for(double limb:{0.,.5,1.}) {
   opt.separate_components=true;auto split=tr::integrate(p,limb,opt,&plan);
   opt.separate_components=false;auto whole=tr::integrate(p,limb,opt,&plan);
   require(split.converged && whole.converged && split.component_domains==4,"component integration domains");
   require(std::abs(split.magnification-whole.magnification)<=split.error_estimate+whole.error_estimate,
           "partition value within combined reported error");
   // Uniform integrates angular lengths analytically, so its value-sample
   // count is zero in both routes. Check work reduction on the LD fixture.
   if(limb==.5)require(split.value_evaluations<whole.value_evaluations,"partition reduces LD work");
   require(split.parameter_derivative_evaluations==0,"component values avoid parameter derivatives");
  }
  bool overlap_checked=false;
  for(const auto&ball:plan.component_images)if(ball.radius<.01){
   tr::AngularOptions masked;masked.inclusion=&ball;
   auto filtered=tr::angular_topology(std::abs(ball.center),p,masked);
   auto full=tr::angular_topology(std::abs(ball.center),p);
   require(filtered.completed && filtered.roots.size()==2 && full.completed && full.roots.size()>2,
           "inclusion mask isolates radially overlapping components");overlap_checked=true;
  }
  require(overlap_checked,"overlapping component fixture exercised");
 }
 // Both caustic-free and caustic-contained eligibility have reusable
 // geometry witnesses. A failed budget must still permit later progress.
 {
  auto p=geometries.front();rs::Tree tree(p);std::vector<Complex> roots;
  auto critical=rs::cover(p,rs::pi/32,0,roots);require(critical.size()==6,"critical witness fixture");
  auto source=critical.front().w;
  require(tree.count(source,.03)<0,"contained caustic must refuse regular path");
  auto hits=tree.refusal_hits();
  require(tree.count(source+Complex(.001,0),.03)<0 && tree.refusal_hits()>hits,
          "contained critical point should prove a nearby refusal without a new traversal");
  require(tree.count({.5,.4},.03)==4,"interior witness must not reject a disjoint regular disk");
  rs::Tree limited(p);(void)limited.count({.5,.4},.03,0);
  require(limited.count({.5,.4},.03)==4,"budget refusal must not suppress retained-tree progress");
 }
 // The source limb creates additional boundary events across contact even
 // when centre-image proposals are reused. Global complement coverage must
 // discover them rather than assuming that the seed list is complete.
 {
  tr::PreparedRegularSupport partial;
  tr::Geometry outside{-.087243824247718094,-.52604606110908503,.03,.2,.08,1.,-.4,.6};
  auto inside=outside;inside.X=-.087211699298757425;inside.Y=-.5259953857930596;
  std::size_t prior_events=0;
  for(auto p:{outside,inside}) {
   auto plan=partial.plan(p,1e-4);auto full=tr::radial_event_plan(p,1000000,1e-5);
   require(plan.completed && full.completed && !plan.component_images.empty(),"partial contact fixture");
   require(plan.events.size()==full.events.size(),"partial seeds must retain all newly created events");
   for(auto e:full.events){bool found=false;for(auto f:plan.events)
    if(e.reciprocal==f.reciprocal && e.radius_lo<=f.radius_hi && e.radius_hi>=f.radius_lo &&
       e.coordinate_lo<=f.coordinate_hi && e.coordinate_hi>=f.coordinate_lo)found=true;
    require(found,"partial support lost a physical event");
   }
   if(prior_events)require(prior_events!=plan.events.size(),"contact must exercise an event-count change");
   prior_events=plan.events.size();
   for(double limb:{0.,.5}) {
    tr::IntegrationOptions opt;opt.absolute_tolerance=0;opt.relative_tolerance=1e-4;
    auto value=tr::integrate(p,limb,opt,&plan);opt.relative_tolerance=1e-7;
    auto reference=tr::integrate(p,limb,opt);
    require(value.converged && reference.converged && value.component_domains>1,"partial contact integration");
    require(std::abs(value.magnification-reference.magnification)<=value.error_estimate+reference.error_estimate,
            "partial contact value exceeds reported error");
   }
  }
 }
 // The projected route must prove all folds, including new images at
 // source-limb contact. It retains local coverage when projection is partial.
 for(auto p:std::array<tr::Geometry,3>{{
      {.1,.15,.15,.2,.08,1.,-.4,.6},
      {-.087243824247718094,-.52604606110908503,.03,.2,.08,1.,-.4,.6},
      {-.087211699298757425,-.5259953857930596,.03,.2,.08,1.,-.4,.6}}}) {
  int component_calls=0;
  auto projected=tr::projected_radial_event_plan(p,1000000,1e-5,nullptr,nullptr,[&]() {
   ++component_calls;return tr::PartialComponents{};
  });
  auto full=tr::radial_event_plan(p,1000000,1e-5);
  require(component_calls==(p.rho==.15?0:1),"partial components must be built only for unresolved coverage");
  require(projected.completed && full.completed && projected.events.size()==full.events.size(),"projected coverage");
  require(projected.resultant_nodes>0 && projected.resultant_events>0,"projected route exercised");
  if(p.rho==.15)require(projected.boxes==0 && projected.resultant_events==12,"full 1D contact coverage");
  else require(projected.boxes>0,"local complement fallback exercised");
  for(auto e:full.events){bool found=false;
   const double angle=(e.reciprocal?rs::pi:0)+2*std::atan(e.coordinate);
   for(auto f:projected.events){
    const double other=(f.reciprocal?rs::pi:0)+2*std::atan(f.coordinate);
    if(e.radius_lo<=f.radius_hi && e.radius_hi>=f.radius_lo && std::abs(std::remainder(angle-other,2*rs::pi))<1e-10)found=true;
   }
   require(found,"projected route lost a physical fold");
  }
  require(!tr::projected_radial_event_plan(p,1,1e-5).completed,"projected budget fails closed");
 }
 // Carry event proposals across queries without reusing their proofs.
 // The resultant must still certify coverage, and a lens change invalidates
 // the proposals before they can influence the next query.
 {
  tr::Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};tr::PreparedRegularSupport carried;
  auto cold=carried.plan(p,1e-4),warm=carried.plan(p,1e-4);
  require(carried.native_seed_solves()==0,"complete projection must avoid centre-image construction");
  require(cold.completed && warm.completed && cold.resultant_solves>0 && warm.resultant_solves==0,
          "event positions should replace repeated polynomial proposals");
  require(warm.resultant_nodes>0 && warm.events.size()==cold.events.size(),"event proposals retain full resultant proof");
  p.X+=.001;auto moved=carried.plan(p,1e-4);
  auto global=tr::radial_event_plan(p,1000000,1e-6);
  require(moved.completed && global.completed && moved.events.size()==global.events.size(),"transported event count");
  for(auto e:global.events) {
   bool found=false;const double angle=(e.reciprocal?rs::pi:0)+2*std::atan(e.coordinate);
   for(auto f:moved.events)if(e.radius_lo<=f.radius_hi && f.radius_lo<=e.radius_hi &&
      std::abs(std::remainder(angle-(f.reciprocal?rs::pi:0)-2*std::atan(f.coordinate),2*rs::pi))<1e-9)found=true;
   require(found,"transported proposals missed a globally certified event");
  }
  p.a+=.01;require(!carried.plan(p,1e-4,1).completed,"failed budget must not accept cached event proof");
  auto changed=carried.plan(p,1e-4);
  require(changed.completed && changed.resultant_solves>0,"event proposal bank invalidated by lens geometry");
 }
 // Nearby-source seed continuation must avoid another global polynomial
 // solve while retaining complete support. A geometry change invalidates it.
 tr::PreparedRegularSupport seed_bank;auto nearby=geometries.front();
 require(seed_bank.plan(nearby,1e-4).completed,"initial seed-bank plan");
 auto solves=seed_bank.native_seed_solves();nearby.X+=1e-4;
 require(seed_bank.plan(nearby,1e-4).completed,"continued seed-bank plan");
 require(seed_bank.continued_seed_hits()>0 && seed_bank.native_seed_solves()==solves,
         "nearby seed proof must reuse positions without global solve");
 nearby.a+=.01;auto switched=seed_bank.plan(nearby,1e-4);
 require(switched.completed && seed_bank.native_seed_solves()>solves,"seed bank invalidated on geometry change");
 auto extreme=geometries.front();extreme.a=1e150;rs::Tree extreme_tree(extreme);
 require(extreme_tree.count({extreme.X,extreme.Y},extreme.rho)<0,"nonfinite critical coefficients must refuse the shortcut");
 require(accepted>=40 && counts>=60 && multiple>0 && max_count==10 && motion_checks>=60 && bounded>0,"nontrivial prepared-support coverage");
 // A strict small-source request must promote local arithmetic early,
 // rather than exhaust the radial budget against a double-precision floor.
 // Regression from evidence/triple_ld_final_cost/tight.json: rho=.01,
 // rtol=1e-9 converged before the LD cost changes but stalled when the
 // loose-value absolute-budget policy was applied to every tolerance.
 for(auto request:{std::array<double,2>{.01,1e-9},std::array<double,2>{.001,1e-7}}) {
  tr::Geometry p{1.0561478501382953,-0.96718632134775,request[0],.2,.08,1.,-.4,.6};
  tr::PreparedRegularSupport prepared;tr::IntegrationOptions opt;
  opt.absolute_tolerance=0;opt.relative_tolerance=request[1];opt.maximum_refinements=1000;
  const auto plan=prepared.plan(p,opt.relative_tolerance,opt.event_box_budget,true);
  const auto result=tr::integrate(p,.5,opt,plan.completed?&plan:nullptr);
  require(result.converged && result.support_complete,"strict LD arithmetic promotion convergence");
  require(result.error_estimate<=opt.relative_tolerance*std::abs(result.magnification),"strict LD unchanged relative error contract");
 }
 std::cout<<"PASS: "<<accepted<<" prepared plans, "<<counts<<" winding counts ("<<multiple<<" with >4 images, maximum "<<max_count<<"), "<<motion_checks<<" quad motion checks, "<<values*4<<" relative value comparisons, "<<bounded<<" bounded-component plans\n";
}
