#include "lcbinint/magnification/adaptive_route_features.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
using namespace lcbinint::magnification;
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    std::ifstream in(argv[1]);std::ofstream out(argv[2]);std::string line;
    out<<"id\ttarget\tfeature_us\troot_us";
    for(int k=0;k<15;++k)out<<"\tf"<<k;
    out<<'\n'<<std::setprecision(17);
    while(std::getline(in,line)){
        int id;double s,q,r,x,y,c;std::istringstream row(line);
        if(!(row>>id>>s>>q>>r>>x>>y>>c))continue;
        try {
            PointSourceMagnifier point;auto start=std::chrono::steady_clock::now();
            auto images=point.binary_images(s,1/q,{x,y});
            double root=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
            for(double target:{1e-3,1e-4}){
                std::array<double,15> f{};start=std::chrono::steady_clock::now();
                // Repetition amortizes clock overhead; volatile sum prevents elimination.
                volatile double sink=0;
                for(int k=0;k<64;++k){f=adaptive_detail::adaptive_route_features(s,1/q,r,c,target,images);sink+=f[k%15];}
                double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/64;
                out<<id<<'\t'<<target<<'\t'<<us<<'\t'<<root;
                for(double v:f)out<<'\t'<<v;
                out<<'\n';
            }
        }catch(const std::exception& e){std::cerr<<id<<' '<<e.what()<<'\n';return 1;}
    }
}
