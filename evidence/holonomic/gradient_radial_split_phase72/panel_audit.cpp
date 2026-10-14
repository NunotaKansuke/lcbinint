#include "../../../tests/holonomic_cpp/adaptive_reference.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace lcbinint::holonomic;

namespace {
struct DetailPair { double fine=0, coarse=0, q_fine=0, q_coarse=0; };

DetailPair panel_details(const AdaptivePanel& p, const AdaptiveWorkspace& w,
                         int component) {
    DetailPair out;
    const int m=1<<p.level, step=256/m;
    const auto& rule=fejer_rule(p.level);
    adaptive_detail::Sum q;
    for(int k=1;k<m;++k) {
        const int id=p.samples[k*step];
        if(id>=0)q.add(rule.w[k-1]*w.samples[id].value[component]);
    }
    out.q_fine=q.get();
    const int cm=m/2, cstep=256/cm;
    const auto& crule=fejer_rule(p.level-1);
    adaptive_detail::Sum cq;
    for(int k=1;k<cm;++k) {
        const int id=p.samples[k*cstep];
        if(id>=0)cq.add(crule.w[k-1]*w.samples[id].value[component]);
    }
    out.q_coarse=cq.get();
    for(int lev=p.level-1;lev<=p.level;++lev) {
        const int mm=1<<lev, st=256/mm, coarse=mm/2-1;
        const auto& r=fejer_rule(lev);
        double norm=0;
        for(int k=1;k<mm;k+=2) {
            const int fid=p.samples[k*st];
            if(fid<0)continue;
            adaptive_detail::Sum pred;
            bool valid=true;
            for(int h=1;h<=coarse;++h) {
                const int cid=p.samples[2*h*st];
                if(cid<0){valid=false;break;}
                pred.add(r.interp[(k/2)*coarse+h-1]*w.samples[cid].value[component]);
            }
            if(!valid)continue;
            const double d=w.samples[fid].value[component]-pred.get();
            norm+=r.norm_w[k-1]*d*d;
        }
        const double detail=std::sqrt(kPi*norm);
        if(lev==p.level)out.fine=detail;else out.coarse=detail;
    }
    return out;
}

int selected_parameter(const std::string& name,double u) {
    if(name=="caustic-cross" && u==0.0)return 4; // a
    if(name=="caustic-cross" && u==0.5)return 0; // X
    if(name=="rand030" && u==0.0)return 2;       // rho
    if(name=="rand035" && (u==0.0 || u==0.5))return 0; // X; Phase69 quality flip
    return -1;
}

double& parameter(LensParams& p,int j) {
    switch(j) {
    case 0:return p.xs;
    case 1:return p.ys;
    case 2:return p.rho;
    case 3:return p.q;
    default:return p.a;
    }
}

AdaptiveResult run(const LensParams& p,double u,bool local,int split_level,
                   AdaptiveWorkspace& w) {
    AdaptiveConfig cfg;
    cfg.gradient_policy=GradientPolicy::ValueFirst;
    cfg.tol.mu_rtol=1e-3;
    cfg.tol.grad_atol.fill(1e-4);
    cfg.tol.grad_rtol.fill(1e-2); // same explicit research tolerance as Phase68/69
    cfg.value_first_gradient_round_budget=64;
    cfg.value_first_gradient_node_budget=8192;
    cfg.max_node_evals=16384;
    cfg.gradient_local_refinement=local;
    cfg.gradient_split_min_level=split_level;
    cfg.gradient_difference_safety=2.0;
    cfg.collect_diagnostics=false;
    PreparedEpochGeometry state;
    (void)epoch_adaptive_prepared(p,u,cfg,w,state); // warm D14 seed only
    return epoch_adaptive_prepared(p,u,cfg,w,state);
}

void finite_difference(const LensParams& p,double u,int j,const std::string& name,
                       int input_row) {
    for(int n:{128,256})for(double hscale:{1e-3,3e-4,1e-4}) {
        const double h=p.rho*hscale;
        LensParams pp=p,pm=p;
        parameter(pp,j)+=h; parameter(pm,j)-=h;
        const auto tp=classify_cells(PrimaryFrame::from(pp),nullptr,nullptr,true);
        const auto tm=classify_cells(PrimaryFrame::from(pm),nullptr,nullptr,true);
        const double vp=reference(pp,u,tp,n),vm=reference(pm,u,tm,n);
        const double fd=(vp-vm)/(2*h);
        std::cout<<"FD "<<input_row<<' '<<name<<' '<<u<<' '<<j<<' '<<n<<' '
                 <<h<<' '<<int(tp.status)<<' '<<int(tm.status)<<' '
                 <<vp<<' '<<vm<<' '<<fd<<'\n';
    }
}

void emit_case(const LensParams& p,double u,const std::string& name,int row) {
    for(int mode=0;mode<3;++mode) {
        const bool local=mode!=0;
        const int split_level=mode==1?4:5;
        AdaptiveWorkspace w;
        const auto r=run(p,u,local,split_level,w);
        std::cout<<"RUN "<<row<<' '<<name<<' '<<u<<' '<<mode<<' '
                 <<(mode==0?0:split_level)<<' '<<r.value_converged<<' '
                 <<adaptive_stop_name(r.value_stop_reason)<<' '
                 <<gradient_quality_name(r.grad_quality[0])<<' '
                 <<r.mu<<' '<<r.stats.unique_nodes<<' '<<r.stats.splits<<' '
                 <<r.stats.node_evaluations;
        for(int j=0;j<5;++j)
            std::cout<<' '<<r.grad_mu[j]<<' '<<r.estimated_abs_error_grad[j]<<' '
                     <<gradient_quality_name(r.grad_quality[j])<<' '
                     <<gradient_reason_name(r.grad_reason[j]);
        for(int j=0;j<6;++j)
            std::cout<<' '<<r.radial_error[j]<<' '<<r.inner_error[j]<<' '
                     <<r.geometry_error[j]<<' '<<r.event_error[j]<<' '
                     <<r.roundoff_error[j];
        std::cout<<'\n';

        for(int j=0;j<5;++j) {
            double pos=0,neg=0,abs_sum=0,total=0;
            for(const auto& panel:w.panels)if(panel.active) {
                const double q=panel.q[j+1];
                total+=q;abs_sum+=std::fabs(q);
                if(q>0)pos+=q;else neg+=q;
            }
            const double cancellation=std::fabs(total)>0?abs_sum/std::fabs(total):INFINITY;
            std::cout<<"CANCEL "<<row<<' '<<name<<' '<<u<<' '<<mode<<' '
                     <<j<<' '<<total<<' '<<pos<<' '<<neg<<' '<<abs_sum<<' '
                     <<cancellation<<' '<<r.grad_mu[j]<<' '
                     <<(total-r.grad_mu[j])<<'\n';
        }

        for(size_t ip=0;ip<w.panels.size();++ip) {
            const auto& panel=w.panels[ip];if(!panel.active)continue;
            for(int j=0;j<5;++j) {
                const int component=j+1;
                const auto d=panel_details(panel,w,component);
                const double floor=panel.inner[component]+panel.geometry[component]+
                                   panel.roundoff[component];
                const bool decay=d.fine<=0.5*d.coarse || d.fine<=floor;
                const double radial_local=local
                    ?std::max(d.fine,2.0*std::fabs(d.q_fine-d.q_coarse)):d.fine;
                const double T=1e-4+1e-2*std::fabs(r.grad_mu[j]);
                const double global_nonradial=r.inner_error[component]+
                    r.geometry_error[component]+r.event_error[component]+
                    r.roundoff_error[component];
                const bool candidate=!decay && radial_local>T &&
                    panel.inner[component]+panel.geometry[component]+
                        panel.event[component]+panel.roundoff[component]<panel.radial[component] &&
                    global_nonradial<=T;
                std::cout<<"PANEL "<<row<<' '<<name<<' '<<u<<' '<<mode<<' '
                         <<ip<<' '<<panel.cell<<' '<<panel.parent<<' '
                         <<panel.depth<<' '<<panel.level<<' '
                         <<panel.map.a<<' '<<panel.map.b<<' '
                         <<panel.map.left<<' '<<panel.map.right<<' '
                         <<j<<' '<<d.q_fine<<' '<<d.fine<<' '<<d.coarse<<' '
                         <<decay<<' '<<radial_local<<' '<<panel.inner[component]<<' '
                         <<panel.geometry[component]<<' '<<panel.event[component]<<' '
                         <<panel.roundoff[component]<<' '<<panel.error[component]<<' '
                         <<T<<' '<<candidate<<' '<<panel.gradient_resolved[j]<<'\n';
            }
        }
    }
}
}

int main(int argc,char** argv) {
    if(argc!=2)return 2;
    std::ifstream in(argv[1]);
    std::cout<<std::setprecision(17);
    std::cout<<"TYPE row name u mode split_level value_converged value_stop quality0 mu nodes splits evals"
             <<" grad0 error0 quality0 reason0 ... grad4 error4 quality4 reason4"
             <<" [six x radial inner geometry event roundoff]\n";
    std::string line;int row=0;
    while(std::getline(in,line)) {
        ++row;
        LensParams p;double u,dummy;int bary;std::string name;
        std::istringstream ss(line);
        if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>u>>dummy>>name))continue;
        if(selected_parameter(name,u)<0)continue;
        p.barycentric=bary;
        emit_case(p,u,name,row);
        finite_difference(p,u,selected_parameter(name,u),name,row);
    }
}
