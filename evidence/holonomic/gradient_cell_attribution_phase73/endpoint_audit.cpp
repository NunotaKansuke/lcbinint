// Independent QF audit of the two caustic-cross cells whose boundary is
// chart_p4 (the projective theta=pi endpoint).  Diagnostic only.
#define main phase72_reference_main
#include "../moving_map_gradient_phase72/rand035_reference/reference.cpp"
#undef main

#include <map>

namespace {
std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v; std::size_t at=0;
    for (;;) { auto e=s.find('\t',at); v.push_back(s.substr(at,e==s.npos?e:e-at));
        if(e==s.npos)break; at=e+1; }
    return v;
}
std::string qstr(Q x) {
    char b[128]; quadmath_snprintf(b,sizeof(b),"%.36Qg",x); return b;
}
}

int main(int argc,char**argv) {
    if(argc!=4){std::cerr<<"usage: endpoint_audit CASES_TSV CELLS_TSV OUTPUT_TSV\n";return 2;}
    LensParams p; bool found=false; int row=0;
    {std::ifstream in(argv[1]);std::string line;while(std::getline(in,line)){++row;
        std::istringstream ss(line);int bary;double u,mu;std::string name;
        if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>u>>mu>>name))continue;
        if(row==17&&name=="caustic-cross"){p.barycentric=bary!=0;found=true;break;}}}
    if(!found)return 3;
    const PrimaryFrame pf=PrimaryFrame::from(p);
    std::ofstream out(argv[3],std::ios::trunc);
    out<<"row\tcell\tR_chart_double\tR_p4_qf\tR_delta_qf\tp4_at_double\tp4_at_root\t"
          "phi_pi\tphi_theta_pi\tphi_R_pi\tphi_thetatheta_pi\tg_pi\tordinary_fold_check\n";
    std::ifstream in(argv[2]);std::string line;std::getline(in,line);
    const auto h=split(line);std::map<std::string,std::size_t> col;
    for(std::size_t i=0;i<h.size();++i)col[h[i]]=i;
    while(std::getline(in,line)){
        const auto f=split(line);
        if(std::stoi(f[col["row"]])!=17||f[col["case"]]!="caustic-cross"||
           f[col["parameter"]]!="a"||std::stoi(f[col["cell"]])!=5&&std::stoi(f[col["cell"]])!=7||
           std::stoi(f[col["level"]])!=8)continue;
        const int cell=std::stoi(f[col["cell"]]);
        const Q rd=Q(std::stod(f[col["a"]]));
        const Q aa=Q(pf.a),m0=Q(pf.m0),m1=Q(1)-m0,X=Q(pf.X),Y=Q(pf.Y),rho=Q(pf.rho);
        auto p4=[&](Q R){return qf_boundary_coeffs(R,pf)[4];};
        auto dp4=[&](Q R){
            const Q c2r=R*(R*R-Q(1)+aa*X)+X*R*R+aa*(R*R-m0);
            const Q c2i=Y*R*(aa+R);
            const Q c2rp=Q(3)*R*R-Q(1)+aa*X+Q(2)*X*R+Q(2)*aa*R;
            const Q c2ip=Y*(aa+Q(2)*R);
            return rho*rho*(Q(2)*R*(R+aa)*(R+aa)+Q(2)*R*R*(R+aa))-
                   Q(2)*(c2r*c2rp+c2i*c2ip);
        };
        Q r=rd;
        for(int k=0;k<5;++k)r-=p4(r)/dp4(r);
        const Q pi=acosq(Q(-1));
        const Q phi=qphi(r,pi,pf),phit=qphi_dtheta(r,pi,pf);
        const Q g=-r+m0/r+m1/(r+aa)-X;
        const Q gr=-Q(1)-m0/(r*r)-m1/((r+aa)*(r+aa));
        const Q gt=-r+m0/r+m1*r/((r+aa)*(r+aa));
        const Q gtt=r-m0/r-m1*r*(r-aa)/((r+aa)*(r+aa)*(r+aa));
        const Q phir=-Q(2)*g*gr/(rho*rho);
        const Q phitt=-Q(2)*(gt*gt+g*gtt)/(rho*rho);
        out<<17<<'\t'<<cell<<'\t'<<std::setprecision(17)<<static_cast<double>(rd)<<'\t'
           <<qstr(r)<<'\t'<<qstr(r-rd)<<'\t'<<qstr(p4(rd))<<'\t'<<qstr(p4(r))<<'\t'
           <<qstr(phi)<<'\t'<<qstr(phit)<<'\t'<<qstr(phir)<<'\t'<<qstr(phitt)<<'\t'
           <<qstr(g)<<'\t'<<((phir!=Q(0)&&phitt!=Q(0))?1:0)<<'\n';
    }
}
