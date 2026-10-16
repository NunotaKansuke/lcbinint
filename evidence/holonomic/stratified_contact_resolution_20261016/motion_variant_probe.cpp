#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "../../../src/lcbinint/magnification/holonomic/fused_gradient.hpp"
using namespace lcbinint::holonomic;
int main(int argc,char**argv){if(argc!=4)return 2;const std::string name=argv[1];const double t=std::stod(argv[2]),tol=std::stod(argv[3]);std::ifstream in(std::string("evidence/holonomic/stratified_contact_resolution_20261016/seeds_")+name+".dat");std::string label;std::size_t n=0;std::getline(in,label);n=std::stoull(label.substr(label.find('=')+1));std::vector<std::array<double,2>> seeds;seeds.reserve(n);for(std::size_t i=0;i<n;++i){double x,y;in>>x>>y;seeds.push_back({x,y});}if(!in||seeds.size()!=n)return 3;const double u0=9.846481110931334e-7,alpha=.5,tE=10,rho=1e-6,q=1e7,s=.95,tau=t/tE,co=cos(alpha),si=sin(alpha);LensParams p{tau*co-u0*si,tau*si+u0*co,rho,q,s,true};auto r=epoch_fused_gradient(p,.5,seeds,{tol,tol});double t0=(-co*r.grad_mu[0]-si*r.grad_mu[1])/tE;std::cout<<std::setprecision(17)<<name<<" t="<<t<<" tol="<<tol<<" mu="<<r.mu<<" value="<<r.value_converged<<" stop="<<r.stop<<" retry="<<r.reliability_retried<<" support="<<r.support_certified<<" seeds="<<n<<" t0="<<t0<<" gx="<<r.grad_mu[0]<<" gy="<<r.grad_mu[1]<<" rho="<<r.grad_mu[2]<<" valid=";for(bool v:r.gradient_valid)std::cout<<v;std::cout<<" nodes="<<r.node_evaluations<<"\n";}
