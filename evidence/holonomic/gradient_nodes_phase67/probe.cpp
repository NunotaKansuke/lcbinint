#include "../../../tests/holonomic_cpp/adaptive_reference.hpp"
#include <iostream>
#include <iomanip>
double f0(double R,LensParams p){auto pf=PrimaryFrame::from(p);auto a=reference_arcs(R,pf);if(a.kind!=ArcKind::kArcs)return NAN;double sum=0;for(auto b:a.arcs){auto e=polish_endpoint(R,b[0],pf),l=polish_endpoint(R,b[1],pf);double w=l.theta-e.theta;if(w<=0)w+=kTwoPi;sum+=R*w;}return sum/(kPi*p.rho*p.rho);}
int main(){
 std::cout<<std::setprecision(17);
 std::cout<<"rounds panel cell depth level slot R jac weight grad cold fd fdhalf value coldvalue reliable a b\n";
 LensParams p{.12,0,.02,.5,1,false};auto pf=PrimaryFrame::from(p);
 for(int rounds:{0,1,2,3,4,8}){
 AdaptiveWorkspace w;AdaptiveConfig c;c.gradient_policy=GradientPolicy::ValueFirst;c.tol.mu_rtol=1e-3;c.value_first_gradient_round_budget=rounds;
 PreparedEpochGeometry state;(void)epoch_adaptive_prepared(p,0,c,w,state);auto result=epoch_adaptive_prepared(p,0,c,w,state);
 std::cerr<<rounds<<" "<<result.mu<<" "<<result.grad_mu[4]<<" "<<result.stats.unique_nodes<<" "<<int(result.grad_quality[4])<<"\n";
 for(size_t i=0;i<w.panels.size();++i){auto&pan=w.panels[i];if(!pan.active||pan.level<1)continue;auto&rule=fejer_rule(pan.level);int m=1<<pan.level;
 for(int k=1;k<m;++k){auto&s=w.samples[pan.samples[k*256/m]];double x=(pan.xl+pan.xr)/2+(pan.xr-pan.xl)/2*rule.x[k-1];
 double jac=pan.map(x)[1]*(pan.xr-pan.xl)/2;
 auto cold=adaptive_detail::mapped_radius(s.R,jac,p,0,pf,w.cells[pan.cell],true,false,nullptr,&c,s.R_lo);
 double h=std::min(p.rho*1e-5,std::min(s.R-pan.map.a,pan.map.b-s.R)*1e-3);
 double fd=NAN,fh=NAN;if(h>1e-12){auto pp=p,pm=p;pp.a+=h;pm.a-=h;fd=jac*(f0(s.R,pp)-f0(s.R,pm))/(2*h);pp=p;pm=p;pp.a+=h/2;pm.a-=h/2;fh=jac*(f0(s.R,pp)-f0(s.R,pm))/h;}
 std::cout<<rounds<<" "<<i<<" "<<pan.cell<<" "<<pan.depth<<" "<<pan.level<<" "<<k<<" "<<s.R<<" "<<jac<<" "<<rule.w[k-1]<<" "<<s.value[5]<<" "<<cold.value[5]<<" "<<fd<<" "<<fh<<" "<<s.value[0]<<" "<<cold.value[0]<<" "<<cold.reliable<<" "<<pan.map.a<<" "<<pan.map.b<<"\n";
 }}
 }
}
