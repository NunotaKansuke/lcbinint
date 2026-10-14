#include "../../../tests/holonomic_cpp/adaptive_reference.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace lcbinint::holonomic;

namespace {
double& parameter(LensParams& p,int j) {
    switch(j) {
    case 0:return p.xs;
    case 1:return p.ys;
    case 2:return p.rho;
    case 3:return p.q;
    default:return p.a;
    }
}
}

int main(int argc,char** argv) {
    if(argc!=3) {
        std::cerr<<"usage: reference_fd INPUT_TSV OUTPUT_TSV\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    std::ofstream out(argv[2],std::ios::out|std::ios::trunc);
    out<<std::setprecision(17)
       <<"row\tcase\tu\tparameter\tn\th\ttopology_plus\ttopology_minus\tmu_plus\tmu_minus\tfd\n";
    std::string line; int row=0;
    while(std::getline(in,line)) {
        ++row; std::istringstream ss(line);
        LensParams p; int bary=0; double u=0,mu=0; std::string name;
        if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>u>>mu>>name))continue;
        p.barycentric=bary!=0;
        if(name!="caustic-cross"||u!=0.0)continue;
        constexpr int j=4;
        for(int n:{128,256,512,1024}) {
            for(double scale:{1e-2,3e-3,1e-3,3e-4,1e-4}) {
                const double h=p.rho*scale;
                LensParams pp=p,pm=p;
                parameter(pp,j)+=h; parameter(pm,j)-=h;
                const auto tp=classify_cells(PrimaryFrame::from(pp),nullptr,nullptr,true);
                const auto tm=classify_cells(PrimaryFrame::from(pm),nullptr,nullptr,true);
                const double vp=reference(pp,u,tp,n),vm=reference(pm,u,tm,n);
                out<<row<<'\t'<<name<<'\t'<<u<<"\ta\t"<<n<<'\t'<<h<<'\t'
                   <<int(tp.status)<<'\t'<<int(tm.status)<<'\t'<<vp<<'\t'<<vm<<'\t'
                   <<(vp-vm)/(2*h)<<'\n';
            }
        }
    }
}
