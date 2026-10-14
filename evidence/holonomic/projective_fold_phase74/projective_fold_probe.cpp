#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;

int main(int argc,char**argv) {
    if(argc!=3) {
        std::cerr<<"usage: projective_fold_probe CASES_TSV OUTPUT_TSV\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    std::ofstream out(argv[2],std::ios::trunc);
    if(!in||!out)return 2;
    out<<std::setprecision(17)
       <<"row\tcase\tu\tevent_index\tR_chart\tdecision\t"
         "p4_relative\tcontact_relative\tangular_curvature_relative\t"
         "radial_crossing_relative\tD14_event_index\tD14_delta\t"
         "D14_match_budget\tR_lo\tuncertainty\tseed_u\n";
    std::string line;
    int row=0;
    while(std::getline(in,line)) {
        ++row;
        std::istringstream ss(line);
        LensParams p;
        int barycentric=0;
        double u=0,mu=0;
        std::string name;
        if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>barycentric>>u>>mu>>name))continue;
        p.barycentric=barycentric!=0;
        const PrimaryFrame pf=PrimaryFrame::from(p);
        const TopologyResult topo=classify_cells(pf,nullptr,nullptr,true);
        for(std::size_t i=0;i<topo.events.size();++i) {
            const auto& event=topo.events[i];
            if(event.kind!="chart_p4")continue;
            const auto probe=adaptive_detail::probe_projective_p4_fold(
                event,topo.events,pf);
            out<<row<<'\t'<<name<<'\t'<<u<<'\t'<<i<<'\t'<<event.radius<<'\t'
               <<adaptive_detail::projective_fold_reject_name(probe.reject)<<'\t'
               <<probe.p4_relative<<'\t'<<probe.contact_relative<<'\t'
               <<probe.angular_curvature_relative<<'\t'
               <<probe.radial_crossing_relative<<'\t'<<probe.d14_event_index<<'\t'
               <<probe.d14_delta<<'\t'<<probe.d14_match_budget<<'\t'
               <<probe.radius_lo<<'\t'<<probe.uncertainty<<'\t'
               <<(probe.accepted()?0.0:std::numeric_limits<double>::quiet_NaN())
               <<'\n';
        }
    }
    return 0;
}
