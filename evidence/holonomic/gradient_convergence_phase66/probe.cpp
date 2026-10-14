#include "../../../tests/holonomic_cpp/adaptive_reference.hpp"
#include <iomanip>
#include <iostream>
int main(int argc,char**argv){
 std::ifstream f(argv[1]);std::string line;std::cout<<std::setprecision(17);
 std::cout<<"kind name u parameter safety tol level nodes converged mu grad error quality n h stop grad_reason radial inner geometry\n";
 while(std::getline(f,line)){
 LensParams p;double u,dummy;int bary;std::string name;std::istringstream ss(line);
 if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>u>>dummy>>name))continue;
 int j=-1;
 if(name=="caustic-cross")j=u==0?4:0;
 if(name=="rand030"&&u==0)j=2;
 if(j<0)continue;p.barycentric=bary;
 for(double safety:{1.,2.,4.})for(double tol:{1e-3,1e-4,1e-5,1e-6,1e-7,1e-8}){
 AdaptiveConfig cfg;cfg.nested_difference_safety=safety;cfg.tol.mu_rtol=tol;cfg.gradient_policy=GradientPolicy::ValueFirst;
 cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;
 AdaptiveWorkspace w;PreparedEpochGeometry state;(void)epoch_adaptive_prepared(p,u,cfg,w,state);
 auto r=epoch_adaptive_prepared(p,u,cfg,w,state);
 std::cout<<"adaptive "<<name<<" "<<u<<" "<<j<<" "<<safety<<" "<<tol<<" 3 "<<r.stats.unique_nodes<<" "<<r.value_converged<<" "<<r.mu<<" "<<r.grad_mu[j]<<" "<<r.estimated_abs_error_grad[j]<<" "<<int(r.grad_quality[j])<<" 0 0 "<<adaptive_stop_name(r.value_stop_reason)<<" "<<int(r.grad_reason[j])<<" "<<r.radial_error[j+1]<<" "<<r.inner_error[j+1]<<" "<<r.geometry_error[j+1]<<"\n";
 }
 for(int level:{5,6,7}) {
 AdaptiveConfig cfg;cfg.initial_level=level;cfg.tol.mu_rtol=1e-8;cfg.gradient_policy=GradientPolicy::ValueFirst;
 cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;AdaptiveWorkspace w;
 auto r=epoch_adaptive(p,u,cfg,w);
 std::cout<<"dense "<<name<<" "<<u<<" "<<j<<" 2 1e-8 "<<level<<" "<<r.stats.unique_nodes<<" "<<r.value_converged<<" "<<r.mu<<" "<<r.grad_mu[j]<<" "<<r.estimated_abs_error_grad[j]<<" "<<int(r.grad_quality[j])<<" 0 0 "<<adaptive_stop_name(r.value_stop_reason)<<" "<<int(r.grad_reason[j])<<" "<<r.radial_error[j+1]<<" "<<r.inner_error[j+1]<<" "<<r.geometry_error[j+1]<<"\n";
 }
 for(int n:{128,256})for(double hs:{1e-3,3e-4,1e-4}){
 double h=p.rho*hs;LensParams pp=p,pm=p;
 double *xp=j==0?&pp.xs:j==2?&pp.rho:&pp.a;
 double *xm=j==0?&pm.xs:j==2?&pm.rho:&pm.a;
 *xp+=h;*xm-=h;
 auto tp=classify_cells(PrimaryFrame::from(pp),nullptr,nullptr,true);
 auto tm=classify_cells(PrimaryFrame::from(pm),nullptr,nullptr,true);
 double vp=reference(pp,u,tp,n),vm=reference(pm,u,tm,n);
 std::cout<<"fd "<<name<<" "<<u<<" "<<j<<" 0 0 0 0 "<<(std::isfinite(vp)&&std::isfinite(vm))<<" "<<(vp+vm)/2<<" "<<(vp-vm)/(2*h)<<" 0 0 "<<n<<" "<<h<<" NA 0 0 0 0\n";std::cout.flush();
 }
 }
}
