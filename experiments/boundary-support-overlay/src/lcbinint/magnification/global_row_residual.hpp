#pragma once
#include "cartesian_row_residual.hpp"
#include <complex>
namespace lcbinint::magnification::detail {
// Euler--Maclaurin continuation, s=-1/2 or -3/2. Not a remainder bound.
inline double endpoint_hurwitz(double s,double eta) {
 double value=0;
 for(int n=0;n<16;++n)value+=std::pow(n+eta,-s);
 const double t=16+eta;
 value+=std::pow(t,1-s)/(s-1)+.5*std::pow(t,-s);
 constexpr std::array<double,5> bernoulli={1./12,-1./720,1./30240,-1./1209600,1./47900160};
 double rising=s;
 for(int k=1;k<=5;++k) {
  if(k>1)rising*=(s+2*k-3)*(s+2*k-2);
  value+=bernoulli[k-1]*rising*std::pow(t,-s-2*k+1);
 }
 return value;
}
struct GlobalRowResidual {
 double outer=0,gap=0,rotated_patch=0,missing_image=0;
 std::size_t local_images=0,unresolved_images=0;
 std::size_t rotated_patches=0,unresolved_patches=0;
 std::size_t fitted=0,local_turns=0,unresolved=0,map_evaluations=0;
 double total()const{return outer+gap+rotated_patch+missing_image;}
};
struct BoundaryTurnGeometry {
 double separation,m1,m2,shift,rho2,source_x,source_y;
};
inline std::array<double,5> boundary_turn_values(const BoundaryTurnGeometry& m,double x,double y) {
 using C=std::complex<double>;const C z{x,y};
 const C first=z-m.separation;
 if(std::norm(z)==0||std::norm(first)==0)return {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
 const C f=z-m.m1/std::conj(first)-m.m2/std::conj(z)-m.shift-C(m.source_x,m.source_y);
 const C k=m.m1/(first*first)+m.m2/(z*z),kp=-2.*m.m1/(first*first*first)-2.*m.m2/(z*z*z);
 const C fx=1.+std::conj(k),fy=C(0,1)*(1.-std::conj(k));
 const C fxx=std::conj(kp),fxy=C(0,-1)*std::conj(kp);
 return {std::norm(f)-m.rho2,2*std::real(std::conj(f)*fx),2*std::real(std::conj(f)*fy),
  2*(std::norm(fx)+std::real(std::conj(f)*fxx)),2*std::real(std::conj(fy)*fx+std::conj(f)*fxy)};
}
// Repair a completely unrepresented, small regular image using its already
// solved centre image. Fixed local quadrature; no additional quintic solve,
// no global grid retry. This is a diagnostic estimate, not a coverage proof.
inline void add_missing_centre_image(const BoundaryTurnGeometry& m,double sx,double sy,
 const std::vector<CartesianRunBoundaryTraceEvent>& events,GlobalRowResidual& out) {
 if(events.empty())return;const double h=events.front().cell_spacing;
 const auto ix=std::llround(sx/h),iy=std::llround(sy/h);
 for(const auto& e:events)if(e.run.iy==iy&&e.run.lo<=ix&&ix<=e.run.hi)return;
 using C=std::complex<double>;const C z{sx,sy},first=z-m.separation;
 const C k=m.m1/(first*first)+m.m2/(z*z);const double J=1-std::norm(k);
 const double rho=std::sqrt(m.rho2),ey=rho*std::abs(1.+k)/std::abs(J),ex=rho/std::abs(1.+k);
 if(!(std::isfinite(ey)&&std::isfinite(ex)&&2*ey<h&&2*ex<h&&std::abs(k)>2)) {++out.unresolved_images;return;}
 const auto evaluate=[&](double x,double y){++out.map_evaluations;return boundary_turn_values(m,x,y);};
 std::array<std::array<double,2>,2> turns;
 const double covariance=2*std::imag(k)/(J*J);
 for(int side=0;side<2;++side) {
  const double sign=side==0?-1:1;double x=sx+sign*m.rho2*covariance/ey,y=sy+sign*ey;
  bool okay=false;
  for(int iteration=0;iteration<30;++iteration) {
   const auto v=evaluate(x,y);const double det=v[1]*v[4]-v[2]*v[3];
   if(!std::isfinite(det)||det==0)break;
   const double dx=(-v[0]*v[4]+v[2]*v[1])/det,dy=(v[0]*v[3]-v[1]*v[1])/det;
   x+=dx;y+=dy;if(std::abs(x-sx)>3*(ex+ey)||std::abs(y-sy)>3*ey)break;
   if(std::abs(dx)+std::abs(dy)<h*1e-11) {okay=true;break;}
  }
  const auto v=evaluate(x,y);
  if(!okay||std::abs(v[0])>m.rho2*1e-9||v[3]<=0) {++out.unresolved_images;return;}
  turns[side]={x,y};
 }
 const double ya=turns[0][1],yb=turns[1][1],mid=.5*(ya+yb),half=.5*(yb-ya);
 if(!(ya<sy&&sy<yb&&yb-ya<2*h)) {++out.unresolved_images;return;}
 // If any occupied coarse node lies in this local enclosure, adding a whole
 // image would double count it. Keep that case unresolved instead.
 const double xlo=std::min(turns[0][0],turns[1][0])-2*ex,xhi=std::max(turns[0][0],turns[1][0])+2*ex;
 for(const auto& e:events)if(e.run.iy*h>=ya&&e.run.iy*h<=yb&&e.run.lo*h<=xhi&&e.run.hi*h>=xlo) {++out.unresolved_images;return;}
 constexpr std::array<double,16> nodes={-.9894009349916499,-.9445750230732326,-.8656312023878318,-.7554044083550030,-.6178762444026438,-.4580167776572274,-.2816035507792589,-.09501250983763744,.09501250983763744,.2816035507792589,.4580167776572274,.6178762444026438,.7554044083550030,.8656312023878318,.9445750230732326,.9894009349916499};
 constexpr std::array<double,16> weights={.027152459411754095,.06225352393864789,.09515851168249278,.12462897125553387,.14959598881657673,.16915651939500254,.1826034150449236,.1894506104550685,.1894506104550685,.1826034150449236,.16915651939500254,.14959598881657673,.12462897125553387,.09515851168249278,.06225352393864789,.027152459411754095};
 double area=0;
 for(int n=0;n<16;++n) {
  const double theta=nodes[n]*1.5707963267948966,py=mid+half*std::sin(theta);
  const double cx=turns[0][0]+(turns[1][0]-turns[0][0])*(py-ya)/(yb-ya);
  std::array<double,2> roots;
  for(int side=0;side<2;++side) {
   double x=cx+(side==0?-1:1)*ex*std::cos(theta);bool okay=false;
   for(int iteration=0;iteration<30;++iteration) {
    const auto v=evaluate(x,py);if(!std::isfinite(v[0])||!std::isfinite(v[1])||v[1]==0)break;
    const double step=v[0]/v[1];x-=step;if(x<xlo||x>xhi)break;
    if(std::abs(step)<h*1e-11) {okay=true;break;}
   }
   const auto v=evaluate(x,py);
   if(!okay||std::abs(v[0])>m.rho2*1e-9) {++out.unresolved_images;return;}
   roots[side]=x;
  }
  if(roots[1]<=roots[0]) {++out.unresolved_images;return;}
  area+=weights[n]*(roots[1]-roots[0])*half*std::cos(theta)*1.5707963267948966;
 }
 if(!(std::isfinite(area)&&area>0)) {++out.unresolved_images;return;}
 out.missing_image+=area;++out.local_images;
}
inline bool short_boundary_turn_residual(const BoundaryTurnGeometry& geometry,
 const CartesianRunBoundaryTraceEvent& event,int direction,int sign,double& residual,GlobalRowResidual& counters) {
 const double h=event.cell_spacing,xlo=(event.run.lo-1.)*h,xhi=(event.run.hi+1.)*h,y0=event.run.iy*h;
 double x=.5*(event.run.lo+event.run.hi)*h,y=y0-.5*direction*h;
 const auto evaluate=[&](double px,double py){++counters.map_evaluations;return boundary_turn_values(geometry,px,py);};
 bool converged=false;
 for(int iteration=0;iteration<40;++iteration) {
  const auto v=evaluate(x,y);const double merit=v[0]*v[0]+h*h*v[1]*v[1];
  if(std::abs(v[0])<geometry.rho2*1e-11&&std::abs(v[1])*h<geometry.rho2*1e-11) {converged=true;break;}
  const double det=v[1]*v[4]-v[2]*v[3];if(!std::isfinite(det)||det==0) return false;
  const double dx=(-v[0]*v[4]+v[2]*v[1])/det,dy=(v[0]*v[3]-v[1]*v[1])/det;
  bool accepted=false;
  for(double factor=1;factor>=1./256;factor*=.5) {
   const double nx=x+factor*dx,ny=y+factor*dy;
   if(nx<xlo||nx>xhi||std::abs(ny-y0)>1.6*h)continue;
   const auto w=evaluate(nx,ny);const double trial=w[0]*w[0]+h*h*w[1]*w[1];
   if(std::isfinite(trial)&&trial<merit) {x=nx;y=ny;accepted=true;break;}
  }
  if(!accepted)return false;
 }
 if(!converged)return false;
 const auto v=evaluate(x,y);const double eta=direction*(y0-y)/h,coefficient=-2*direction*v[2]*h/v[3];
 if(!(eta>0&&eta<1.6&&coefficient>0&&sign*v[3]>0))return false;
 const double a=2*std::sqrt(coefficient);std::array<double,2> bvalues;
 for(int sample=0;sample<2;++sample) {
  const double t=sample==0?.0625:.125,py=y+direction*t*h,offset=std::sqrt(coefficient*t);
  std::array<double,2> roots;
  for(int side=0;side<2;++side) {
   double px=x+(side==0?-offset:offset);bool okay=false;
   for(int iteration=0;iteration<30;++iteration) {
    const auto u=evaluate(px,py);if(!std::isfinite(u[0])||!std::isfinite(u[1])||u[1]==0)return false;
    const double step=u[0]/u[1];px-=step;
    if(std::abs(step)<h*1e-11) {okay=true;break;}
   }
   if(!okay)return false;roots[side]=px;
  }
  if(roots[1]<=roots[0])return false;
  bvalues[sample]=((roots[1]-roots[0])/std::sqrt(t)-a)/t;
 }
 const double b=2*bvalues[0]-bvalues[1];
 residual=-h*(a*endpoint_hurwitz(-.5,eta)+b*endpoint_hurwitz(-1.5,eta));
 return std::isfinite(residual);
}
inline bool rotated_boundary_patch(const BoundaryTurnGeometry& geometry,
 const CartesianRunBoundaryTraceEvent& low,const CartesianRunBoundaryTraceEvent& high,
 double& residual,GlobalRowResidual& counters) {
 const double h=low.cell_spacing,ya=low.run.iy*h,yb=high.run.iy*h;
 constexpr std::array<double,8> nodes={-.9602898564975363,-.7966664774136267,-.5255324099163290,-.1834346424956498,.1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363};
 constexpr std::array<double,8> weights={.1012285362903763,.2223810344533745,.3137066458778873,.3626837833783620,.3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763};
 const auto evaluate=[&](double x,double y){++counters.map_evaluations;return boundary_turn_values(geometry,x,y);};
 std::array<double,2> integrals,derivative_a,derivative_b;
 for(int side=0;side<2;++side) {
  const double xa=(side==0?low.run.lo-low.diagnostic.left_fraction:low.run.hi+low.diagnostic.right_fraction)*h;
  const double xb=(side==0?high.run.lo-high.diagnostic.left_fraction:high.run.hi+high.diagnostic.right_fraction)*h;
  if(!std::isfinite(xa)||!std::isfinite(xb))return false;
  const auto va=evaluate(xa,ya),vb=evaluate(xb,yb);
  derivative_a[side]=-va[2]/va[1];derivative_b[side]=-vb[2]/vb[1];
  const bool integrate_x=std::abs(xb-xa)>h;
  double sum=0;
  for(int n=0;n<8;++n) {
   const double fixed=integrate_x?.5*(xa+xb)+.5*(xb-xa)*nodes[n]:.5*(ya+yb)+.5*h*nodes[n];
   double variable=integrate_x?ya+h*(fixed-xa)/(xb-xa):xa+(xb-xa)*(fixed-ya)/h;
   bool converged=false;
   for(int iteration=0;iteration<30;++iteration) {
    const auto v=integrate_x?evaluate(fixed,variable):evaluate(variable,fixed);
    const double derivative=integrate_x?v[2]:v[1];
    if(!std::isfinite(v[0])||!std::isfinite(derivative)||std::abs(derivative)<1e-30)return false;
    const double step=v[0]/derivative;variable-=step;
    if(integrate_x&&std::abs(variable-.5*(ya+yb))>3*h)return false;
    if(std::abs(step)<h*1e-11) {converged=true;break;}
   }
   if(!converged)return false;sum+=weights[n]*variable;
  }
  integrals[side]=integrate_x?xb*yb-xa*ya-.5*(xb-xa)*sum:.5*h*sum;
 }
 const double area=integrals[1]-integrals[0];
 const double trap=.5*(low.midpoint_area+high.midpoint_area)*h*h;
 const double tail=h*h*((derivative_b[1]-derivative_b[0])-(derivative_a[1]-derivative_a[0]))/12;
 residual=area-trap+tail;
 return std::isfinite(residual);
}
inline GlobalRowResidual global_row_residual_candidate(const std::vector<CartesianRunBoundaryTraceEvent>& events,
 const BoundaryTurnGeometry& geometry) {
 GlobalRowResidual result;const RowResidualGraph graph(events);
 const auto process=[&](const auto& nodes,const RowResidualGraph& links,int sign,bool check_parent) {
  for(std::size_t i=0;i<nodes.size();++i)for(int direction:{1,-1}) {
   const auto& next=direction==1?links.after:links.before;
   const auto& previous=direction==1?links.before:links.after;
   if(!previous[i].empty())continue;
   if(check_parent) {
    const auto parent=graph.rows.find(nodes[i].run.iy-direction);bool enclosed=false;
    if(parent!=graph.rows.end())for(auto p:parent->second)
     if(events[p].run.lo<=nodes[i].run.lo&&events[p].run.hi>=nodes[i].run.hi)enclosed=true;
    if(!enclosed)continue;
   }
   std::array<std::size_t,4> panel{i,0,0,0};int count=1;
   while(count<4&&next[panel[count-1]].size()==1) {
    const auto j=next[panel[count-1]][0];if(previous[j].size()!=1)break;
    panel[count++]=j;
   }
   double residual=0;bool resolved=false;
   if(count==4) {
    const auto model=row_endpoint_model(nodes,panel);
    const double h=nodes[i].cell_spacing;
    std::array<double,4> widths,uncertainties;
    for(int k=0;k<4;++k) {
        const auto& e=nodes[panel[k]];
        widths[k]=e.midpoint_area*h;
        uncertainties[k]=h*((e.diagnostic.left_bracket_upper-e.diagnostic.left_bracket_lower)+
            (e.diagnostic.right_bracket_upper-e.diagnostic.right_bracket_lower));
    }
    const double mismatch=std::abs(widths[0]*widths[0]-3*widths[1]*widths[1]+3*widths[2]*widths[2]-widths[3]*widths[3]);
    double noise=0;constexpr std::array<double,4> factors={1,3,3,1};
    for(int k=0;k<4;++k)noise+=factors[k]*(2*widths[k]*uncertainties[k]+uncertainties[k]*uncertainties[k]);
    // Reject a quadratic model contradicted by the held-out fourth sample
    // beyond the already-recorded numerical boundary uncertainty.
    if(model.resolved&&mismatch<=noise) {
     const double w0=nodes[panel[0]].midpoint_area*h,w1=nodes[panel[1]].midpoint_area*h,w2=nodes[panel[2]].midpoint_area*h;
     const double c2=.5*(w2*w2-2*w1*w1+w0*w0),c1=w1*w1-w0*w0-c2;
     const double a=std::sqrt(c1-2*model.eta*c2),b=c2/(2*a);
     residual=-h*(a*endpoint_hurwitz(-.5,model.eta)+b*endpoint_hurwitz(-1.5,model.eta));
     resolved=std::isfinite(residual);if(resolved)++result.fitted;
    }
   }
   if(!resolved) {++result.local_turns;resolved=short_boundary_turn_residual(geometry,nodes[i],direction,sign,residual,result);}
   if(!resolved) {++result.unresolved;continue;}
   if(sign>0)result.outer+=residual;else result.gap-=residual;
  }
 };
 process(events,graph,1,false);
 std::vector<CartesianRunBoundaryTraceEvent> gaps;
 for(const auto& [iy,indices]:graph.rows) {
  auto ordered=indices;std::sort(ordered.begin(),ordered.end(),[&](auto i,auto j){return events[i].run.lo<events[j].run.lo;});
  for(std::size_t k=1;k<ordered.size();++k) {
   const auto& l=events[ordered[k-1]];const auto& u=events[ordered[k]];const double h=l.cell_spacing;
   const double width=(u.run.lo-u.diagnostic.left_fraction-l.run.hi-l.diagnostic.right_fraction)*h;
   if(!(std::isfinite(width)&&width>0))continue;
   CartesianRunBoundaryTraceEvent gap;gap.run={iy,l.run.hi,u.run.lo};gap.midpoint_area=width/h;gap.cell_spacing=h;
   gap.diagnostic.left_fraction=-l.diagnostic.right_fraction;
   gap.diagnostic.right_fraction=-u.diagnostic.left_fraction;
   gap.diagnostic.left_bracket_lower=l.diagnostic.right_bracket_lower;
   gap.diagnostic.left_bracket_upper=l.diagnostic.right_bracket_upper;
   gap.diagnostic.right_bracket_lower=u.diagnostic.left_bracket_lower;
   gap.diagnostic.right_bracket_upper=u.diagnostic.left_bracket_upper;
   gaps.push_back(gap);
  }
 }
 // Empty intervals are geometric gaps, not occupied 8-neighbour cells.
 // A diagonal lattice adjacency can join an enclosed hole to the exterior.
 const RowResidualGraph gap_graph(gaps,true);
 process(gaps,gap_graph,-1,true);
 // A new tip and closing gap can both lie between the same two rows,
 // leaving the sampled run count unchanged. Scout the largest boundary
 // displacement away from all already-visible endpoints and junctions.
 double largest_score=.5;std::size_t patch_a=events.size(),patch_b=events.size();
 for(std::size_t i=0;i<events.size();++i) {
  if(graph.after[i].size()!=1)continue;const auto j=graph.after[i][0];
  if(graph.before[j].size()!=1)continue;bool regular=true;
  for(auto start:{i,j})for(int direction:{-1,1}) {
   auto current=start;
   for(int step=0;step<2;++step) {
    if(graph.before[current].size()!=1||graph.after[current].size()!=1) {regular=false;break;}
    current=direction>0?graph.after[current][0]:graph.before[current][0];
   }
  }
  if(!regular)continue;
  const auto& a=events[i].run;const auto& b=events[j].run;
  const double displacement=std::max(std::abs((static_cast<double>(a.lo)-static_cast<double>(b.lo))),std::abs((static_cast<double>(a.hi)-static_cast<double>(b.hi))));
  const double width=std::min(static_cast<double>(a.hi-a.lo+1),static_cast<double>(b.hi-b.lo+1));
  const double score=displacement/width;
  if(score>largest_score) {largest_score=score;patch_a=i;patch_b=j;}
 }
 if(patch_a!=events.size()) {
  double residual=0;
  if(rotated_boundary_patch(geometry,events[patch_a],events[patch_b],residual,result)) {
   result.rotated_patch=residual;++result.rotated_patches;
  } else ++result.unresolved_patches;
 }
 // Remove the actual Bennett weight adjustment once per physical row.
 for(const auto& e:events)result.outer-=(e.outer_weight-1)*e.midpoint_area*e.cell_spacing*e.cell_spacing;
 return result;
}
}
