#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace lcbinint::holonomic;
int main(int argc,char**argv){
  if(argc!=2)return 2;
  std::ifstream in(argv[1]);if(!in)return 3;
  std::cout<<std::setprecision(17)
    <<"row\tname\tu\trtol\tpolicy\twall_ms\tvalue_ok\tvalue_stop\tstatus\tmu\tvalue_error\tnodes\tpanels\tsplits";
  for(int j=0;j<5;++j)std::cout<<"\tgrad"<<j<<"\tgrad_error"<<j<<"\tgrad_quality"<<j;
#ifdef SOFTCUTFOLD_CANDIDATE
  std::cout<<"\tsoftcut_fold_merges";
#endif
  std::cout<<'\n';
  std::string line;int row=0;
  while(std::getline(in,line)){
    ++row;LensParams p;double u,reference;int bary;std::string name;
    std::istringstream ss(line);
    if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>u>>reference>>name))continue;
    p.barycentric=bary!=0;
    for(double rtol:{1e-3,1e-4})for(auto policy:{GradientPolicy::None,GradientPolicy::ValueFirst}){
      AdaptiveConfig cfg;cfg.gradient_policy=policy;cfg.with_jacobian=policy!=GradientPolicy::None;
      cfg.tol.mu_rtol=rtol;
      AdaptiveWorkspace workspace;
      const auto begin=std::chrono::steady_clock::now();
      const auto result=epoch_adaptive(p,u,cfg,workspace);
      const double ms=std::chrono::duration<double,std::milli>(
          std::chrono::steady_clock::now()-begin).count();
      std::cout<<row<<'\t'<<name<<'\t'<<u<<'\t'<<rtol<<'\t'
        <<gradient_policy_name(policy)<<'\t'<<ms<<'\t'<<result.value_converged<<'\t'
        <<adaptive_stop_name(result.value_stop_reason)<<'\t'
        <<static_cast<int>(result.numerical_status)<<'\t'<<result.mu<<'\t'
        <<result.estimated_abs_error_mu<<'\t'<<result.stats.unique_nodes<<'\t'
        <<result.stats.panels<<'\t'<<result.stats.splits;
      for(int j=0;j<5;++j)std::cout<<'\t'<<result.grad_mu[j]<<'\t'
        <<result.estimated_abs_error_grad[j]<<'\t'
        <<gradient_quality_name(result.grad_quality[j]);
#ifdef SOFTCUTFOLD_CANDIDATE
      std::cout<<'\t'<<result.stats.softcut_fold_merges;
#endif
      std::cout<<'\n';
    }
  }
}
