#include "../../../tests/holonomic_cpp/adaptive_reference.hpp"
int main(int argc,char**argv){
std::ifstream f(argv[1]);std::string line;
puts("case_id profile d_bin epoch n reference");
while(std::getline(f,line)){
 std::istringstream s(line);int id,cid,db,ep;std::string profile;double a,q,rho,x,y,time,u,X,ref;
 if(!(s>>id>>cid>>profile>>db>>ep>>a>>q>>rho>>x>>y>>time>>u>>X>>ref))continue;
 if(id!=115||db!=2||(ep!=15&&ep!=7))continue;
 LensParams p{time,y,rho,1/q,a,true};auto t=classify_cells(PrimaryFrame::from(p),nullptr,nullptr,true);
 for(int n:{128,256,512,1024}){double z=reference(p,u,t,n);printf("%d %s %d %d %d %.17g\n",id,profile.c_str(),db,ep,n,z);fflush(stdout);}
}}
