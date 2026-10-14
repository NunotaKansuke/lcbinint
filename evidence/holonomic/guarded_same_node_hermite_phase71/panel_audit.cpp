#include "../../../tests/holonomic_cpp/adaptive_reference.hpp"
#include <iomanip>
#include <iostream>
using namespace lcbinint::holonomic;
double fref(double R,const LensParams&p,const PrimaryFrame&pf){auto a=reference_arcs(R,pf);if(a.kind==ArcKind::kEmpty)return 0;if(a.kind==ArcKind::kFull)return 2*kPi*R/(kPi*p.rho*p.rho);if(a.kind!=ArcKind::kArcs)return NAN;double w=0;for(auto arc:a.arcs){auto l=polish_endpoint(R,arc[0],pf),r=polish_endpoint(R,arc[1],pf);if(!l.reliable||!r.reliable)return NAN;double z=r.theta-l.theta;if(z<=0)z+=kTwoPi;w+=z;}return R*w/(kPi*p.rho*p.rho);}
double detail(const std::array<double,7>& values,int level){const int m=1<<level,step=256/m,coarse=m/2-1;const auto& rule=fejer_rule(level);double sum=0;for(int k=1;k<m;k+=2){double pred=0;for(int h=1;h<=coarse;++h){const int ci=(2*h*step)/32-1;pred+=rule.interp[(k/2)*coarse+h-1]*values[ci];}const int fi=(k*step)/32-1;const double d=values[fi]-pred;sum+=rule.norm_w[k-1]*d*d;}return std::sqrt(kPi*sum);}
std::array<double,2> panel_map(const AdaptivePanel& p,double x){
 const long double aa=(long double)p.map.a+p.left_radius_lo;
 const long double bb=(long double)p.map.b+p.right_radius_lo;
 if(p.left_radius_lo==0&&p.right_radius_lo==0)return p.map(x);
 const long double t=(1.0L+x)*0.5L,w=bb-aa;
 if(p.map.left&&p.map.right){
  const long double z=std::sin(3.1415926535897932384626433832795029L*(t<=0.5L?t:1.0L-t)*0.5L);
  return {(double)(t<=0.5L?aa+w*z*z:bb-w*z*z),(double)(3.1415926535897932384626433832795029L*w*0.25L*std::sin(3.1415926535897932384626433832795029L*t))};
 }
 if(p.map.left)return {(double)(aa+w*t*t),(double)(w*t)};
 if(p.map.right)return {(double)(bb-w*(1.0L-t)*(1.0L-t)),(double)(w*(1.0L-t))};
 return {(double)(aa+w*t),(double)(w*0.5L)};
}
int main(int argc,char**argv){std::ifstream in(argv[1]);std::string line;std::cout<<std::setprecision(17)<<"name tol panel cell depth value_resolved q7 qh3 qh7 correction model inner geometry event roundoff model_budget_pass_ignoring_value_detail panel_local_budget_pass ref64 ref128 ref_unc qh_error q7_error model_under jet_count jet_endpoint_ift_max_scaled derivative_detail3 derivative_detail7 derivative_detail_decays\n";
 while(std::getline(in,line)){LensParams p;double u,dummy;int b;std::string name;std::istringstream ss(line);if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>b>>u>>dummy>>name)||u!=0)continue;p.barycentric=b;auto pf=PrimaryFrame::from(p);
 for(double tol:{1e-3,1e-4}){AdaptiveConfig cfg;cfg.same_node_hermite_shadow=true;cfg.gradient_policy=GradientPolicy::None;cfg.tol.mu_rtol=tol;cfg.max_level=3;AdaptiveWorkspace w;auto result=epoch_adaptive(p,u,cfg,w);(void)result;GL lo(64),hi(128);
 for(size_t k=0;k<w.panels.size();++k){const auto& pan=w.panels[k];const auto& hp=w.hermite_panels[k];if(pan.level!=3||!hp.seen)continue;const double h=.5*(pan.xr-pan.xl),mid=.5*(pan.xr+pan.xl);int jet_count=0;for(int slot=32;slot<256;slot+=32){int sid=pan.samples[slot];if(sid>=0&&static_cast<size_t>(sid)<w.hermite_jets.size()&&w.hermite_jets[sid].radial_jet)++jet_count;}
 if(!hp.valid){std::cout<<name<<' '<<tol<<' '<<k<<' '<<pan.cell<<' '<<pan.depth<<' '<<pan.value_resolved<<' '<<pan.q[0]<<" nan nan nan nan "<<pan.inner[0]<<' '<<pan.geometry[0]<<' '<<pan.event[0]<<' '<<pan.roundoff[0]<<" 0 0 nan nan nan nan nan "<<jet_count<<" nan nan nan 0\n";continue;}
 auto integ=[&](const GL&g){double s=0;for(size_t j=0;j<g.x.size();++j){double x=mid+h*g.x[j];auto m=panel_map(pan,x);double v=fref(m[0],p,pf);s+=g.w[j]*v*m[1]*h;}return s;};
 const double q64=integ(lo),q128=integ(hi),unc=std::fabs(q128-q64),err=std::fabs(hp.q7-q128),err7=std::fabs(pan.q[0]-q128);const double budget=cfg.tol.budget(0,hp.q7);const double model=hp.model_error+pan.inner[0]+pan.geometry[0]+pan.event[0]+pan.roundoff[0];const bool model_pass_ignoring_detail=model<=budget,early=pan.value_resolved&&model_pass_ignoring_detail;
 double jet_ift=std::numeric_limits<double>::quiet_NaN();
 double derivative_detail3=std::numeric_limits<double>::quiet_NaN(),derivative_detail7=std::numeric_limits<double>::quiet_NaN();bool derivative_detail_decays=false;
 if(jet_count==7){std::array<double,7> d{};for(int j=1;j<=7;++j)d[j-1]=w.hermite_jets[pan.samples[j*32]].g_xi;derivative_detail3=detail(d,2);derivative_detail7=detail(d,3);derivative_detail_decays=derivative_detail7<=.5*derivative_detail3;}
 if(jet_count==7){jet_ift=0;for(int j=1;j<=7;++j){const int sid=pan.samples[j*32];const double xi=std::cos(kPi*j/8.0),x=mid+h*xi;const auto mapped=panel_map(pan,x);const auto arcs=reference_arcs(mapped[0],pf);if(arcs.kind!=ArcKind::kArcs){jet_ift=std::numeric_limits<double>::infinity();break;}double width=0,width_R=0;for(auto arc:arcs.arcs){auto l=polish_endpoint(mapped[0],arc[0],pf),r=polish_endpoint(mapped[0],arc[1],pf);if(!l.reliable||!r.reliable){jet_ift=std::numeric_limits<double>::infinity();break;}double tl=l.theta,tr=r.theta;if(tr<=tl)tr+=kTwoPi;width+=tr-tl;const double t0=std::tan(.5*tl),t1=std::tan(.5*tr);auto dl=endpoint_dR(t0,mapped[0],pf),dr=endpoint_dR(t1,mapped[0],pf);if(!dl.ok||!dr.ok){jet_ift=std::numeric_limits<double>::infinity();break;}width_R+=2*dr.dt_dR/(1+t1*t1)-2*dl.dt_dR/(1+t0*t0);}if(!std::isfinite(jet_ift))break;const double norm=kPi*p.rho*p.rho,F=mapped[0]*width/norm,FR=(width+mapped[0]*width_R)/norm;double Rxx=0;const double t=.5*(1+x),map_width=pan.map.b-pan.map.a+pan.left_radius_lo+pan.right_radius_lo;if(pan.map.left&&pan.map.right)Rxx=kPi*kPi*map_width*.125*std::cos(kPi*t);else if(pan.map.left)Rxx=.5*map_width;else if(pan.map.right)Rxx=-.5*map_width;const double J=mapped[1]*h,expected=FR*J*J+F*Rxx*h*h;const auto& jet=w.hermite_jets[sid];if(!jet.fixed_r_derivative_finite){jet_ift=std::numeric_limits<double>::infinity();break;}const double efr=std::fabs(jet.FR_over_norm-FR)/(1+std::fabs(FR)),eg=std::fabs(jet.g_xi-expected)/(1+std::fabs(expected));jet_ift=std::max(jet_ift,std::max(efr,eg));}}
 std::cout<<name<<' '<<tol<<' '<<k<<' '<<pan.cell<<' '<<pan.depth<<' '<<pan.value_resolved<<' '<<pan.q[0]<<' '<<hp.q3<<' '<<hp.q7<<' '<<hp.correction<<' '<<hp.model_error<<' '<<pan.inner[0]<<' '<<pan.geometry[0]<<' '<<pan.event[0]<<' '<<pan.roundoff[0]<<' '<<model_pass_ignoring_detail<<' '<<early<<' '<<q64<<' '<<q128<<' '<<unc<<' '<<err<<' '<<err7<<' '<<(err>model+unc)<<' '<<jet_count<<' '<<jet_ift<<' '<<derivative_detail3<<' '<<derivative_detail7<<' '<<derivative_detail_decays<<'\n';
 }}
 }
}
