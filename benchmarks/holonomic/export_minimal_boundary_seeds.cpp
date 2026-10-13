#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
using namespace lcbinint;
using namespace lcbinint::magnification;
int main(int argc,char**argv){
    if(argc<3||argc>4)return 2;
    const std::string mode=argc==4?argv[3]:"legacy";
    std::ifstream in(argv[1]);std::ofstream out(argv[2]);
    out<<std::setprecision(17);
    int id;double x,y,rho,q,s;
    while(in>>id>>x>>y>>rho>>q>>s){
        PointSourceMagnifier pm;FiniteSourceSettings cfg;cfg.caustic_bins=1400;
        FiniteSourceMagnifier fm(cfg);bool proven=false;
        auto t=std::chrono::steady_clock::now();
        std::vector<SourcePosition> seeds;
        if(mode=="point-candidates"||mode=="centre"){
            // Ghost polynomial roots are proposals, never certified images.
            for(const auto& candidate:pm.binary_image_candidates(s,q,{x,y}))
                if((mode=="point-candidates"||candidate.physical)&&
                   std::isfinite(candidate.position.x+candidate.position.y))
                    seeds.push_back(candidate.position);
        }else if(mode=="legacy"){
            seeds=fm.cached_binary_image_seeds(pm,s,q,{x,y},rho,
                std::numeric_limits<double>::infinity(),nullptr,&proven);
        }else return 2;
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
        out<<id<<' '<<x<<' '<<y<<' '<<rho<<' '<<q<<' '<<s<<' '<<proven<<' '<<ms<<' '<<seeds.size();
        for(auto z:seeds)out<<' '<<z.x+(q<1?s:0)<<' '<<z.y;
        out<<'\n';
    }
}
