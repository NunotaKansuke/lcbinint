#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;
using Clock=std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point t) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}

int main(int argc,char**argv) {
    if(argc!=4){std::cerr<<"usage: projective_screen_scan INPUT OUTPUT snapshot|reference\n";return 2;}
    std::ifstream in(argv[1]);std::ofstream out(argv[2]);
    if(!in||!out)return 2;
    const std::string mode=argv[3];
    if(mode!="snapshot"&&mode!="reference")return 2;
    out<<std::setprecision(17)
       <<"row profile bin epoch event_index X Y rho q a radius topology "
         "old_decision old_p4_steps old_probe_ms fast_screen fast_screen_ms "
         "p4_interval_lo p4_interval_hi p3_interval_lo p3_interval_hi "
         "p3_qf_gate_hi d14_gap overlap_budget fast_reject_old_accept\n";
    std::string line;std::size_t inputs=0,charts=0,screen_rejects=0,old_accepts=0,
        false_rejects=0,screen_possible=0,screen_unresolved=0;
    double old_ms=0,screen_ms=0,screened_ms=0;
    while(std::getline(in,line)) {
        if(line.empty()||line[0]=='#')continue;
        std::istringstream ss(line);
        LensParams p;std::string profile="reference";int row_id=0,bin=0,epoch=0;
        if(mode=="reference") {
            int bary=0;double u=0,mu=0;std::string name;
            if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>u>>mu>>name))continue;
            p.barycentric=bary!=0;profile=name;row_id=(int)++inputs;
        } else {
            int config=0,bary=0;double s=0,q=0,rho=0,x=0,y=0,time=0,u=0,X=0,ref=0;
            if(!(ss>>row_id>>config>>profile>>bin>>epoch>>s>>q>>rho>>x>>y>>time>>u>>X>>ref))continue;
            p=LensParams{time,y,rho,1.0/q,s,true};
            ++inputs;
        }
        const auto pf=PrimaryFrame::from(p);
        const auto topo=classify_cells(pf,nullptr,nullptr,true);
        for(std::size_t i=0;i<topo.events.size();++i) {
            const auto& event=topo.events[i];if(event.kind!="chart_p4")continue;
            ++charts;
            const auto sb=Clock::now();
            const auto screen=adaptive_detail::projective_p4_fast_screen(event,topo.events,pf);
            const double sm=elapsed_ms(sb);screen_ms+=sm;
            const bool rejected=screen.result==adaptive_detail::ProjectiveFastScreen::NoD14Overlap||
                                screen.result==adaptive_detail::ProjectiveFastScreen::NoReciprocalContact;
            const auto ob=Clock::now();
            const auto old=adaptive_detail::probe_projective_p4_fold(event,topo.events,pf);
            const double om=elapsed_ms(ob);old_ms+=om;
            if(rejected){++screen_rejects;screened_ms+=sm;}
            if(screen.result==adaptive_detail::ProjectiveFastScreen::Possible)++screen_possible;
            if(screen.result==adaptive_detail::ProjectiveFastScreen::Unresolved)++screen_unresolved;
            old_accepts+=old.accepted();
            const bool false_reject=rejected&&old.accepted();
            false_rejects+=false_reject;
            out<<row_id<<' '<<profile<<' '<<bin<<' '<<epoch<<' '<<i<<' '
               <<pf.X<<' '<<pf.Y<<' '<<pf.rho<<' '<<p.q<<' '<<pf.a<<' '
               <<event.radius<<' '<<to_string(topo.status)<<' '
               <<adaptive_detail::projective_fold_reject_name(old.reject)<<' '
               <<old.p4_newton_steps<<' '<<om<<' '
               <<adaptive_detail::projective_fast_screen_name(screen.result)<<' '
               <<sm<<' '<<screen.radius_lo<<' '<<screen.radius_hi<<' '
               <<screen.p3_lo<<' '<<screen.p3_hi<<' '<<screen.p3_gate_hi<<' '
               <<screen.nearest_d14_gap<<' '<<screen.overlap_budget<<' '
               <<false_reject<<'\n';
        }
    }
    std::cerr<<"mode="<<mode<<" inputs="<<inputs<<" chart_p4="<<charts
             <<" old_accepted="<<old_accepts<<" screen_rejects="<<screen_rejects
             <<" false_rejects="<<false_rejects<<" screen_possible="<<screen_possible
             <<" screen_unresolved="<<screen_unresolved<<" old_probe_ms="<<old_ms
             <<" screen_only_ms="<<screened_ms<<" screen_eval_ms="<<screen_ms<<'\n';
    return false_rejects?4:0;
}
