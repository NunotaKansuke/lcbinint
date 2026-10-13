#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "fused_k.hpp"
#include <iostream>
using namespace lcbinint::holonomic;
int main() {
    int checks=0;
    for(double m:{-.5,0.,.3})for(double v:{0.,1e-6,.1}) {
        const std::array<double,3> a{{m*m-v,-2*m,1}},b{{-4.8,1.4,1}};
        std::array<double,5> pc{};
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)pc[i+j]+=a[i]*b[j];
        std::array<std::array<double,5>,5> dp{};
        for(int j=0;j<5;++j)dp[j][j]=1;
        const RootPair pair{m,v};
        const auto batch=fused_gradient::root_pair_tangents(pair,pc,dp);
        const auto J=eo_jacobian(pair,pc);
        for(int j=0;j<5;++j) {
            const auto scalar=root_pair_dR(pair,pc,dp[j]);
            if(!batch[j].ok || !scalar.ok)return 1;
            const auto rhs=eo_residuals(pair,dp[j]);
            const double E=J.E_m*batch[j].dm_dR+J.E_v*batch[j].dv_dR+rhs.E;
            const double O=J.O_m*batch[j].dm_dR+J.O_v*batch[j].dv_dR+rhs.O;
            if(std::fabs(E)>1e-12 || std::fabs(O)>1e-12 ||
               std::fabs(batch[j].dm_dR-scalar.dm_dR)>1e-12 ||
               std::fabs(batch[j].dv_dR-scalar.dv_dR)>1e-12)return 2;
            ++checks;
        }
    }
    std::array<std::array<double,5>,5> dp{};
    const auto singular=fused_gradient::root_pair_tangents({0,0},{},dp);
    for(const auto& d:singular)if(d.ok)return 3;
    std::cout<<checks<<" shared root-pair solves satisfy implicit equations and match scalar solves; singular pair rejected\n";
}
