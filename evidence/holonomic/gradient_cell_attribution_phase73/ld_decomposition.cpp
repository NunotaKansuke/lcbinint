// Shadow decomposition of d(vK)=dv*K+v*dK on Phase72's actual Fejer nodes.
// No production files or numerical gates are changed.
#include "../../../tests/holonomic_cpp/adaptive_reference.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace lcbinint::holonomic;

namespace {
struct RowNode {
    double R=0.0,J=0.0,w=0.0;
    bool source_ok=false;
};
struct ArcTerms {
    double f0p=0.0,dvK=0.0,v_dK=0.0,check=0.0;
    double max_decomp_residual=0.0;
    int accepted=0,rejected=0;
};
struct CellTerms {
    long double uniform=0.0L,dvK=0.0L,v_dK=0.0L,total_ld=0.0L;
    double abs_dvK=0.0,abs_v_dK=0.0;
    int nodes=0,source_rejects=0,sample_rejects=0,k_nodes=0,k_rejects=0;
    double max_decomp_residual=0.0;
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out; std::size_t at=0;
    for (;;) { const auto e=s.find('\t',at); out.push_back(s.substr(at,e==std::string::npos?e:e-at));
        if(e==std::string::npos)break; at=e+1; }
    return out;
}
std::map<std::string,std::size_t> header(const std::string& s) {
    const auto v=split(s);std::map<std::string,std::size_t> h;
    for(std::size_t i=0;i<v.size();++i)h[v[i]]=i;return h;
}
std::string field(const std::vector<std::string>& v,const std::map<std::string,std::size_t>& h,const std::string& k) {
    auto i=h.at(k);return i<v.size()?v[i]:std::string{};
}
double user_parameter(LensParams& p,const std::string& axis) {
    if(axis=="X")return p.xs;
    if(axis=="Y")return p.ys;
    return 0.0;
}

ArcTerms decompose_at_R(double R,const LensParams& p,const PrimaryFrame& pf,
                        const CellPlan& cell,int j,const AdaptiveSample& sample) {
    ArcTerms result;
    QuarticWarm qw=sample.quartic;
    RootPairWarm rp=sample.roots;
    QuarticCoeffs pc{};
    const ArcSet arcs=adaptive_detail::adaptive_arc_intervals(
        R,pf,&qw,&rp,&pc,cell.n_crossings);
    if(arcs.kind==ArcKind::kEmpty)return result;
    if(arcs.kind!=ArcKind::kArcs){++result.rejected;return result;}
    const QuarticParamJac dpc=boundary_quartic_dp(R,pf);
    const std::array<double,5> zero{};
    for(const auto& arc:arcs.arcs) {
        auto pe=polish_endpoint(R,arc[0],pf,6);
        auto pl=polish_endpoint(R,arc[1],pf,6);
        double te=pe.theta,tl=pl.theta;if(tl<=te)tl+=kTwoPi;
        if(!pe.reliable||!pl.reliable){++result.rejected;continue;}
        const PhiGrad ge=phi_grad(R,te,pf),gl=phi_grad(R,tl,pf);
        if(ge.dphi_dtheta==0.0||gl.dphi_dtheta==0.0){++result.rejected;continue;}
        const double dte=-ge.dP[j]/ge.dphi_dtheta;
        const double dtl=-gl.dP[j]/gl.dphi_dtheta;
        result.f0p += R*(dtl-dte);
        bool got=false;
        for(bool reciprocal:{false,true}) {
            ArcPairJac ap=reciprocal
                ?arc_pair_jac_reciprocal(te,tl,zero,zero)
                :arc_pair_jac(te,tl,zero,zero);
            if(!ap.ok)continue;
            const QuarticCoeffs cp=reciprocal?boundary_quartic_reciprocal(pc):pc;
            const QuarticParamJac dp=reciprocal?boundary_quartic_reciprocal_dp(dpc):dpc;
            const RootPairDR dr=root_pair_dR(RootPair{ap.m,ap.v},cp.p,dp.dp[j]);
            if(!dr.ok)continue;
            ap.dm[j]=dr.dm_dR;ap.dv[j]=dr.dv_dR;
            VKJacobian actual=v_times_K_jac(ap.m,ap.v,ap.dm,ap.dv,R,pf,
                                             cp.p,dp.dp,reciprocal);
            if(!actual.ok)continue;

            const double m=ap.m,v=ap.v,s=std::sqrt(v);
            const DeflatedQuad dq=deflate_mv(cp.p,m,v);
            const double p4=dq.p4,d1=dq.d1,d0=dq.d0;
            const double dp4=dp.dp[j][4];
            const double dd1=dp.dp[j][3]+2.0*(ap.dm[j]*p4+m*dp4);
            const double dd0=dp.dp[j][2]+2.0*(ap.dm[j]*d1+m*dd1)-
                (2.0*m*ap.dm[j]-ap.dv[j])*p4-(m*m-v)*dp4;
            const double Rma=R-pf.a,Rpa=R+pf.a;
            const double b0=reciprocal?Rpa:Rma;
            const double b2=reciprocal?Rma*Rma:Rpa*Rpa;
            const auto& gc=gc2_rule16();
            double G=0.0,dG=0.0;
            for(int k=0;k<kHoloNK;++k) {
                const double xi=gc.x[k],wi=gc.w[k],t=m+s*xi;
                const double S2=-(d0+t*(d1+t*p4));
                const double A=1.0+t*t,B=b0*b0+b2*t*t;
                if(!(S2>0.0)||!(B>0.0)){G=NAN;break;}
                const double wv=std::sqrt(S2)/(A*std::sqrt(A)*std::sqrt(B));
                G+=wi*wv;
                const double dt=ap.dm[j]+xi*(0.5/s)*ap.dv[j];
                const double dS2dt=-(d1+2.0*t*p4);
                const double dS2=-(dd0+t*(dd1+t*dp4))+dS2dt*dt;
                const double dA=2.0*t*dt;
                double dB=2.0*b2*t*dt;
                if(j==4)dB+=reciprocal?2.0*Rpa-2.0*Rma*t*t
                                       :-2.0*Rma+2.0*Rpa*t*t;
                dG+=wi*wv*(0.5*dS2/S2-1.5*dA/A-0.5*dB/B);
            }
            if(!std::isfinite(G)||!std::isfinite(dG))continue;
            const double dv_term=(2.0/pf.rho)*ap.dv[j]*G;
            const double vdK_term=(2.0/pf.rho)*v*dG;
            const double residual=dv_term+vdK_term-actual.dvK[j];
            result.dvK+=dv_term;result.v_dK+=vdK_term;
            result.check+=actual.dvK[j];result.max_decomp_residual=
                std::max(result.max_decomp_residual,std::fabs(residual));
            ++result.accepted;got=true;break;
        }
        if(!got)++result.rejected;
    }
    return result;
}

} // namespace

int main(int argc,char** argv) {
    if(argc!=5){std::cerr<<"usage: ld_decomposition CASES_TSV NODES_TSV CELLS_TSV OUTPUT_PREFIX\n";return 2;}
    std::map<std::pair<int,std::string>,LensParams> params;
    {
        std::ifstream in(argv[1]);std::string line;int row=0;
        while(std::getline(in,line)){++row;std::istringstream ss(line);LensParams p;int bary;double u,mu;std::string name;
            if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>u>>mu>>name))continue;
            if(name!="rand035"||row!=100||u!=0.5)continue;
            p.barycentric=bary!=0;params[{row,"X"}]=p;params[{row,"Y"}]=p;}
    }
    std::map<std::string,std::size_t> ch;std::vector<std::vector<std::string>> cell_fields;
    {std::ifstream in(argv[3]);std::string line;std::getline(in,line);ch=header(line);while(std::getline(in,line))cell_fields.push_back(split(line));}
    std::ofstream node_out(std::string(argv[4])+"_nodes.tsv",std::ios::out|std::ios::trunc);
    std::ofstream sum_out(std::string(argv[4])+"_cells.tsv",std::ios::out|std::ios::trunc);
    node_out<<std::setprecision(17)<<"row\tcase\tu\tparameter\tcell\tk\tR\tJ\tK_arcs\tK_rejected\t"
        "uniform_Fp\tdvK\tv_dK\tdvK_plus_vdK\tproduction_vK_jac\tdecomp_residual\t"
        "sample_Fp_D\tidentity_residual\tquadrature_dvK\tquadrature_v_dK\n";
    sum_out<<std::setprecision(17)<<"row\tcase\tu\tparameter\tcell\tk_nodes\taccepted_K_nodes\t"
        "rejected_K_nodes\tsample_rejected_nodes\tsource_invalid_nodes\tuniform_contribution\tdvK_contribution\t"
        "v_dK_contribution\tLD_total\tdvK_abs_sum\tv_dK_abs_sum\tterm_abs_sum\t"
        "radial_cancel_dvK\tradial_cancel_v_dK\tterm_cancel_after_radial\tmax_pointwise_residual\n";

    for(const auto& pp:params) {
        const int row=pp.first.first;const std::string axis=pp.first.second;
        const LensParams p=pp.second;const PrimaryFrame pf=PrimaryFrame::from(p);
        const TopologyResult topo=classify_cells(pf,nullptr,nullptr,true);
        std::vector<CellPlan> plans;if(topo.status!=Status::OK||!adaptive_detail::restore_physical_cuts(topo,pf,plans))continue;
        const int j=axis=="X"?0:1;const double u=0.5;
        std::map<int,std::map<int,RowNode>> input;
        {
            std::ifstream in(argv[2]);std::string line;std::getline(in,line);const auto h=header(line);
            while(std::getline(in,line)){
                const auto f=split(line);
                if(std::stoi(field(f,h,"row"))!=row||field(f,h,"case")!="rand035"||
                   field(f,h,"u")!="0.5"||field(f,h,"parameter")!=axis||field(f,h,"level")!="8")continue;
                const int cell=std::stoi(field(f,h,"cell")),k=std::stoi(field(f,h,"k"));
                if(cell!=2&&cell!=3&&cell!=8&&cell!=11)continue;
                RowNode n;n.R=std::stod(field(f,h,"R"));n.J=std::stod(field(f,h,"J"));
                n.w=std::stod(field(f,h,"fejer_weight"));n.source_ok=field(f,h,"ok")=="1";
                input[cell][k]=n;
            }
        }
        const double D=kPi*p.rho*p.rho*(1.0-u/3.0);
        for(auto& by_cell:input) {
            const int ci=by_cell.first;if(ci<0||ci>=static_cast<int>(plans.size()))continue;
            CellTerms acc;AdaptiveSample prev;bool have_prev=false;
            for(int k=1;k<256;++k) {
                auto it=by_cell.second.find(k);if(it==by_cell.second.end())continue;
                const RowNode& n=it->second;++acc.nodes;
                if(!n.source_ok){++acc.source_rejects;continue;}
                AdaptiveSample s=adaptive_detail::mapped_radius(
                    n.R,1.0,p,u,pf,plans[ci],true,topo.from_warm_d14,
                    have_prev?&prev:nullptr,nullptr,0.0,false,nullptr);
                if(!s.reliable&&have_prev)
                    s=adaptive_detail::mapped_radius(n.R,1.0,p,u,pf,plans[ci],true,
                                                     false,nullptr,nullptr,0.0,false,nullptr);
                if(!s.reliable){++acc.sample_rejects;continue;}
                prev=s;have_prev=true;
                ArcTerms terms=decompose_at_R(n.R,p,pf,plans[ci],j,s);
                acc.k_nodes+=terms.accepted;
                acc.k_rejects+=terms.rejected;
                acc.max_decomp_residual=std::max(acc.max_decomp_residual,terms.max_decomp_residual);
                const double sample_fp_D=s.value[j+1]*D;
                const double identity=(1.0-u)*terms.f0p+u*(terms.dvK+terms.v_dK);
                const double id_res=sample_fp_D-identity;
                const double qdv=n.w*n.J*u*terms.dvK/D;
                const double qvd=n.w*n.J*u*terms.v_dK/D;
                acc.uniform+=static_cast<long double>(n.w*n.J*(1.0-u)*terms.f0p/D);
                acc.dvK+=static_cast<long double>(qdv);
                acc.v_dK+=static_cast<long double>(qvd);
                acc.total_ld+=static_cast<long double>(qdv+qvd);
                acc.abs_dvK+=std::fabs(qdv);acc.abs_v_dK+=std::fabs(qvd);
                node_out<<row<<"\trand035\t"<<u<<'\t'<<axis<<'\t'<<ci<<'\t'<<k<<'\t'
                    <<n.R<<'\t'<<n.J<<'\t'<<terms.accepted<<'\t'<<terms.rejected<<'\t'
                    <<terms.f0p<<'\t'<<terms.dvK<<'\t'<<terms.v_dK<<'\t'
                    <<terms.dvK+terms.v_dK<<'\t'<<terms.check<<'\t'
                    <<terms.dvK+terms.v_dK-terms.check<<'\t'<<sample_fp_D<<'\t'<<id_res<<'\t'
                    <<qdv<<'\t'<<qvd<<'\n';
            }
            const double abs_terms=acc.abs_dvK+acc.abs_v_dK;
            const double radial_dv=std::fabs(static_cast<double>(acc.dvK))>1e-300
                ?acc.abs_dvK/std::fabs(static_cast<double>(acc.dvK)):INFINITY;
            const double radial_vdk=std::fabs(static_cast<double>(acc.v_dK))>1e-300
                ?acc.abs_v_dK/std::fabs(static_cast<double>(acc.v_dK)):INFINITY;
            const double after_radial=(std::fabs(static_cast<double>(acc.total_ld))>1e-300)
                ?(std::fabs(static_cast<double>(acc.dvK))+std::fabs(static_cast<double>(acc.v_dK)))/std::fabs(static_cast<double>(acc.total_ld)):INFINITY;
            sum_out<<row<<"\trand035\t"<<u<<'\t'<<axis<<'\t'<<ci<<'\t'<<acc.nodes<<'\t'
                <<acc.k_nodes<<'\t'<<acc.k_rejects<<'\t'<<acc.sample_rejects<<'\t'<<acc.source_rejects<<'\t'
                <<static_cast<double>(acc.uniform)<<'\t'<<static_cast<double>(acc.dvK)<<'\t'
                <<static_cast<double>(acc.v_dK)<<'\t'<<static_cast<double>(acc.total_ld)<<'\t'
                <<acc.abs_dvK<<'\t'<<acc.abs_v_dK<<'\t'<<abs_terms<<'\t'
                <<radial_dv<<'\t'<<radial_vdk<<'\t'<<after_radial<<'\t'<<acc.max_decomp_residual<<'\n';
        }
    }
}
