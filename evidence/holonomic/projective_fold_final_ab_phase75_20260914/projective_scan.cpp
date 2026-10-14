#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;

int main(int argc,char** argv) {
    if(argc!=3){std::cerr<<"usage: projective_scan INPUT OUTPUT\n";return 2;}
    std::ifstream in(argv[1]);std::ofstream out(argv[2]);
    if(!in||!out){std::cerr<<"cannot open input/output\n";return 2;}
    out<<"# Per-epoch chart_p4 candidate audit; diagnostic-only, outside benchmark timers\n";
    out<<"case_id configuration_id profile d_bin_index epoch_index event_index radius "
          "decision contact_relative angular_curvature_relative radial_crossing_relative "
          "d14_delta d14_match_budget topology_status topology_cells topology_events\n";
    out<<std::setprecision(17);
    std::string line;std::size_t input_rows=0,chart_records=0,accepted=0;
    while(std::getline(in,line)){
        if(line.empty()||line[0]=='#')continue;
        std::istringstream ss(line);
        int case_id=0,configuration_id=0,bin=0,epoch=0,barycentric=0;
        std::string profile;double s=0,q=0,rho=0,x=0,y=0,time=0,u=0,X=0,reference=0;
        if(!(ss>>case_id>>configuration_id>>profile>>bin>>epoch>>s>>q>>rho>>x>>y>>time>>u>>X>>reference))continue;
        ++input_rows;
        const LensParams p{time,y,rho,1.0/q,s,true};
        const auto pf=PrimaryFrame::from(p);
        const auto topo=classify_cells(pf,nullptr,nullptr,true);
        for(std::size_t i=0;i<topo.events.size();++i){
            const auto& event=topo.events[i];if(event.kind!="chart_p4")continue;
            ++chart_records;
            const auto probe=adaptive_detail::probe_projective_p4_fold(event,topo.events,pf);
            accepted+=probe.accepted();
            out<<case_id<<' '<<configuration_id<<' '<<profile<<' '<<bin<<' '<<epoch<<' '
               <<i<<' '<<event.radius<<' '<<adaptive_detail::projective_fold_reject_name(probe.reject)<<' '
               <<probe.contact_relative<<' '<<probe.angular_curvature_relative<<' '
               <<probe.radial_crossing_relative<<' '<<probe.d14_delta<<' '
               <<probe.d14_match_budget<<' '<<to_string(topo.status)<<' '
               <<topo.cells.size()<<' '<<topo.events.size()<<'\n';
        }
    }
    std::cerr<<"input_rows="<<input_rows<<" chart_p4_records="<<chart_records
             <<" accepted_projective_folds="<<accepted<<'\n';
    return input_rows==7216?0:3;
}
