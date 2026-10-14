// Compare production Phase72 fixed-R analytic gradients with an independent
// QF direct-phi angular derivative at exactly the same Fejer nodes.
// This separates local derivative evaluation from radial quadrature.
#define main phase72_reference_main
#include "../moving_map_gradient_phase72/rand035_reference/reference.cpp"
#undef main

#include <map>
#include <tuple>

struct DerivativePair {
    long double f0=0.0L,fh=0.0L;
    bool ok=true;
    int rejected=0;
    int arc_count=0;
    std::array<long double,4> arc_f0{},arc_fh{};
};

Q qphi_dp(Q R,Q theta,const PrimaryFrame&pf,const std::string&parameter,bool barycentric){
    const Q ct=cosq(theta),st=sinq(theta);const QC z{R*ct,R*st},zb{R*ct,-R*st};
    const Q m0=Q(pf.m0),m1=Q(1)-m0,a=Q(pf.a);const QC iz=qdiv({Q(1),Q(0)},zb),delta=qsub(zb,{a,Q(0)});
    const QC iza=qdiv({Q(1),Q(0)},delta);const QC f=qsub(qsub(z,{m0*iz.re,m0*iz.im}),{m1*iza.re,m1*iza.im});
    const QC g=qsub(f,{Q(pf.X),Q(pf.Y)});QC gp{Q(0),Q(0)};
    if(parameter=="X")gp={Q(-1),Q(0)};else if(parameter=="Y")gp={Q(0),Q(-1)};
    else if(parameter=="a"){gp=qdiv({-m1,Q(0)},qmul(delta,delta));if(barycentric)gp.re-=m1;}
    else return std::numeric_limits<Q>::quiet_NaN();
    const Q rho=Q(pf.rho);return -Q(2)*(g.re*gp.re+g.im*gp.im)/(rho*rho);
}

Q qphi_dp_over_endpoint(Q R,Q theta,const PrimaryFrame&pf,const std::string&parameter,bool barycentric){
    const Q pp=qphi_dp(R,theta,pf,parameter,barycentric),pt=qphi_dtheta(R,theta,pf);
    if(!finiteq(pp)||!finiteq(pt)||pt==Q(0))return std::numeric_limits<Q>::quiet_NaN();
    return -pp/pt;
}

DerivativePair independent_fixed_r_derivative(double R,const PrimaryFrame&pf,const std::string&parameter,bool barycentric,int angular_nodes){
    DerivativePair out;RefValue local;const RefArcs arcs=certified_arcs(R,pf,local);
    if(local.fail||arcs.kind==ArcKind::kDegenerate){out.ok=false;out.rejected=local.fail?local.fail:1;return out;}
    if(arcs.kind==ArcKind::kEmpty)return out;
    if(arcs.kind==ArcKind::kFull){
        const Q pi=acosq(Q(-1));Q sum=Q(0);
        for(int j=0;j<angular_nodes;++j){const Q th=Q(2)*pi*(Q(j)+Q(0.5))/Q(angular_nodes);const Q ph=qphi(Q(R),th,pf);const Q php=qphi_dp(Q(R),th,pf,parameter,barycentric);if(!(ph>Q(0))||!finiteq(php)){out.ok=false;out.rejected=2;return out;}sum+=php/(Q(2)*sqrtq(ph));}
        out.fh=static_cast<long double>(Q(R)*Q(2)*pi*sum/Q(angular_nodes));return out;
    }
    if(arcs.kind!=ArcKind::kArcs){out.ok=false;out.rejected=3;return out;}
    const Q pi=acosq(Q(-1));Q f0p=Q(0),fhp=Q(0);
    if(arcs.arcs.size()>out.arc_f0.size()){out.ok=false;out.rejected=7;return out;}
    out.arc_count=static_cast<int>(arcs.arcs.size());
    for(std::size_t ai=0;ai<arcs.arcs.size();++ai){
        const auto&arc=arcs.arcs[ai];const Q lo=arc.lo,hi=arc.hi;
        const Q dlo=qphi_dp_over_endpoint(Q(R),lo,pf,parameter,barycentric),dhi=qphi_dp_over_endpoint(Q(R),hi,pf,parameter,barycentric);
        if(!finiteq(dlo)||!finiteq(dhi)){out.ok=false;out.rejected=4;return out;}
        const Q arc0=Q(R)*(dhi-dlo),mid=(lo+hi)/Q(2),half=(hi-lo)/Q(2);Q arch=Q(0);
        for(int j=0;j<angular_nodes;++j){
            const Q x=cosq(pi*(Q(2*j+1))/Q(2*angular_nodes)),omx2=Q(1)-x*x,th=mid+half*x;
            const Q ph=qphi(Q(R),th,pf),php=qphi_dp(Q(R),th,pf,parameter,barycentric);
            if(!(omx2>Q(0))||!(ph>Q(0))||!finiteq(php)){out.ok=false;out.rejected=5;return out;}
            const Q H=ph/omx2;if(!(H>Q(0))||!finiteq(H)){out.ok=false;out.rejected=6;return out;}
            arch+=Q(R)*half*(pi/Q(angular_nodes))*php/(Q(2)*sqrtq(H));
        }
        out.arc_f0[ai]=static_cast<long double>(arc0);out.arc_fh[ai]=static_cast<long double>(arch);
        f0p+=arc0;fhp+=arch;
    }
    out.f0=static_cast<long double>(f0p);out.fh=static_cast<long double>(fhp);return out;
}

namespace {
std::vector<std::string> split_tsv(const std::string&s){std::vector<std::string>v;std::size_t a=0;for(;;){auto e=s.find('\t',a);v.push_back(s.substr(a,e==s.npos?e:e-a));if(e==s.npos)break;a=e+1;}return v;}
std::map<std::string,std::size_t> columns(const std::string&s){auto v=split_tsv(s);std::map<std::string,std::size_t>m;for(std::size_t i=0;i<v.size();++i)m[v[i]]=i;return m;}
std::string fld(const std::vector<std::string>&v,const std::map<std::string,std::size_t>&h,const char*k){auto i=h.at(k);return i<v.size()?v[i]:std::string{};}
struct Selected{int row=0;std::string name,param;double u=0;LensParams p;};
struct Sum{long double prod=0,d128=0,d256=0,abs_delta=0;double max_delta=0,max_rel=0;int n=0,prod_ok=0,direct_ok=0,direct_fail=0,prod_rejected=0;};
struct ArcSum{long double f0=0,fh=0,combined=0,abs_combined=0,abs_f0=0,abs_fh=0;int nodes=0;};
std::string key(int row,const std::string&parameter,int cell){return std::to_string(row)+"/"+parameter+"/"+std::to_string(cell);}
std::string arc_key(int row,const std::string&parameter,int cell,int arc){return key(row,parameter,cell)+"/"+std::to_string(arc);}
}

int main(int argc,char**argv){
    if(argc!=6){std::cerr<<"usage: same_node_audit CASES_TSV NODES_TSV NODE_OUT CELL_OUT ARC_OUT\n";return 2;}
    std::vector<Selected> selected;
    {std::ifstream in(argv[1]);std::string line;int row=0;while(std::getline(in,line)){++row;std::istringstream ss(line);Selected c;int bary;double mu;if(!(ss>>c.p.xs>>c.p.ys>>c.p.rho>>c.p.q>>c.p.a>>bary>>c.u>>mu>>c.name))continue;c.p.barycentric=bary!=0;c.row=row;
        if(row==17&&c.name=="caustic-cross"){c.param="a";selected.push_back(c);}
        if((row==99||row==100)&&c.name=="rand035"){for(const auto&par:{"X","Y"}){c.param=par;selected.push_back(c);}}}}
    std::map<std::string,Selected> by_id;for(const auto&c:selected)by_id[std::to_string(c.row)+"/"+c.param]=c;
    std::ofstream node_out(argv[3],std::ios::trunc),cell_out(argv[4],std::ios::trunc),arc_out(argv[5],std::ios::trunc);
    node_out<<std::setprecision(17)<<"row\tcase\tu\tparameter\tcell\tk\tR\tJ\tfejer_weight\t"
        "production_Fp\tqf_Fp_128\tqf_Fp_256\tproduction_old\tqf_old_128\tqf_old_256\t"
        "point_error_256\tpoint_rel_error_256\tprod_sample_ok\tprod_node_ok\tprod_reject_reason\t"
        "qf128_ok\tqf256_ok\n";
    cell_out<<std::setprecision(17)<<"row\tcase\tu\tparameter\tcell\tfejer_nodes\tprod_valid_nodes\t"
        "qf256_valid_nodes\tqf_failures\tproduction_invalid_nodes\tproduction_old_q8\tqf_old_q8_128\t"
        "qf_old_q8_256\tlocal_derivative_delta\tmax_point_abs_delta\tmax_point_rel_delta\n";
    arc_out<<std::setprecision(17)<<"row\tcase\tu\tparameter\tcell\tarc\tfejer_nodes\t"
        "uniform_arc_contribution\tLD_arc_contribution\tcombined_arc_contribution\t"
        "arc_radial_abs_sum\twithin_arc_radial_cancellation\tcell_interarc_cancellation\n";
    std::map<std::string,Sum> sums;
    std::map<std::string,ArcSum> arc_sums;
    std::ifstream in(argv[2]);std::string line;std::getline(in,line);const auto h=columns(line);
    while(std::getline(in,line)){
        const auto f=split_tsv(line);
        if(fld(f,h,"level")!="8")continue;
        const int row=std::stoi(fld(f,h,"row"));const std::string param=fld(f,h,"parameter");
        auto ci=by_id.find(std::to_string(row)+"/"+param);if(ci==by_id.end())continue;
        const Selected& c=ci->second;const int cell=std::stoi(fld(f,h,"cell"));
        const double R=std::stod(fld(f,h,"R")),J=std::stod(fld(f,h,"J")),w=std::stod(fld(f,h,"fejer_weight"));
        const bool sample_ok=fld(f,h,"sample_ok")=="1",node_ok=fld(f,h,"ok")=="1";
        const double D=kPi*c.p.rho*c.p.rho*(1.0-c.u/3.0);
        const double prodFp=std::stod(fld(f,h,"Fp"));
        const double prodOld=std::stod(fld(f,h,"old"));
        const std::string reason=fld(f,h,"reject_reason");
        DerivativePair d128,d256;
        if(fld(f,h,"kind")=="empty"){d128.ok=true;d256.ok=true;}
        else {
            const PrimaryFrame pf=PrimaryFrame::from(c.p);
            d128=independent_fixed_r_derivative(R,pf,param,c.p.barycentric,128);
            d256=independent_fixed_r_derivative(R,pf,param,c.p.barycentric,256);
        }
        const double qf128=(1.0-c.u)*static_cast<double>(d128.f0)+c.u*static_cast<double>(d128.fh);
        const double qf256=(1.0-c.u)*static_cast<double>(d256.f0)+c.u*static_cast<double>(d256.fh);
        const double old128=J*qf128/D,old256=J*qf256/D;
        const double point_delta=qf256-prodFp;
        const double rel=std::fabs(point_delta)/std::max(1e-300,std::fabs(qf256));
        node_out<<row<<'\t'<<c.name<<'\t'<<c.u<<'\t'<<param<<'\t'<<cell<<'\t'
            <<fld(f,h,"k")<<'\t'<<R<<'\t'<<J<<'\t'<<w<<'\t'<<prodFp<<'\t'<<qf128<<'\t'<<qf256<<'\t'
            <<prodOld<<'\t'<<old128<<'\t'<<old256<<'\t'<<point_delta<<'\t'<<rel<<'\t'
            <<sample_ok<<'\t'<<node_ok<<'\t'<<reason<<'\t'<<d128.ok<<'\t'<<d256.ok<<'\n';
        auto& s=sums[key(row,param,cell)];++s.n;if(node_ok){++s.prod_ok;s.prod+=static_cast<long double>(w)*prodOld;}else ++s.prod_rejected;
        if(d256.ok){++s.direct_ok;s.d128+=static_cast<long double>(w)*old128;s.d256+=static_cast<long double>(w)*old256;if(node_ok){s.abs_delta+=std::fabs(static_cast<long double>(w)*old256-static_cast<long double>(w)*prodOld);s.max_delta=std::max(s.max_delta,std::fabs(point_delta));s.max_rel=std::max(s.max_rel,rel);}}else ++s.direct_fail;
        if(d256.ok&&fld(f,h,"kind")=="arcs")for(int ai=0;ai<d256.arc_count;++ai){
            const long double f0=static_cast<long double>(w*J*(1.0-c.u)/D)*d256.arc_f0[ai];
            const long double fh=static_cast<long double>(w*J*c.u/D)*d256.arc_fh[ai];
            const long double combined=f0+fh;auto&a=arc_sums[arc_key(row,param,cell,ai)];
            a.f0+=f0;a.fh+=fh;a.combined+=combined;a.abs_combined+=std::fabs(combined);a.abs_f0+=std::fabs(f0);a.abs_fh+=std::fabs(fh);++a.nodes;
        }
    }
    for(const auto&entry:sums){const auto&k=entry.first;const auto&s=entry.second;const auto slash=k.find('/');const int row=std::stoi(k.substr(0,slash));const auto slash2=k.find('/',slash+1);const std::string param=k.substr(slash+1,slash2-slash-1);const int cell=std::stoi(k.substr(slash2+1));const auto&c=by_id.at(std::to_string(row)+"/"+param);
        cell_out<<row<<'\t'<<c.name<<'\t'<<c.u<<'\t'<<param<<'\t'<<cell<<'\t'<<s.n<<'\t'<<s.prod_ok<<'\t'<<s.direct_ok<<'\t'<<s.direct_fail<<'\t'<<s.prod_rejected<<'\t'
            <<static_cast<double>(s.prod)<<'\t'<<static_cast<double>(s.d128)<<'\t'<<static_cast<double>(s.d256)<<'\t'
            <<static_cast<double>(s.abs_delta)<<'\t'<<s.max_delta<<'\t'<<s.max_rel<<'\n';}
    for(const auto&entry:arc_sums){
        const auto&k=entry.first;const auto&s=entry.second;const auto a=k.find('/'),b=k.find('/',a+1),d=k.find('/',b+1);
        const int row=std::stoi(k.substr(0,a));const std::string param=k.substr(a+1,b-a-1);const int cell=std::stoi(k.substr(b+1,d-b-1)),arc=std::stoi(k.substr(d+1));
        const auto&c=by_id.at(std::to_string(row)+"/"+param);long double cell_net=0,cell_abs=0;
        for(const auto&other:arc_sums){const auto&ok=other.first;const auto oa=ok.find('/'),ob=ok.find('/',oa+1),od=ok.find('/',ob+1);if(std::stoi(ok.substr(0,oa))==row&&ok.substr(oa+1,ob-oa-1)==param&&std::stoi(ok.substr(ob+1,od-ob-1))==cell){cell_net+=other.second.combined;cell_abs+=std::fabs(other.second.combined);}}
        const double within=std::fabs(static_cast<double>(s.combined))>1e-300?static_cast<double>(s.abs_combined/std::fabs(s.combined)):INFINITY;
        const double across=std::fabs(static_cast<double>(cell_net))>1e-300?static_cast<double>(cell_abs/std::fabs(cell_net)):INFINITY;
        arc_out<<row<<'\t'<<c.name<<'\t'<<c.u<<'\t'<<param<<'\t'<<cell<<'\t'<<arc<<'\t'<<s.nodes<<'\t'
            <<static_cast<double>(s.f0)<<'\t'<<static_cast<double>(s.fh)<<'\t'<<static_cast<double>(s.combined)<<'\t'
            <<static_cast<double>(s.abs_combined)<<'\t'<<within<<'\t'<<across<<'\n';
    }
}
