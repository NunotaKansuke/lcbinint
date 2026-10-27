#include "lcbinint/magnification/algebraic_boundary/tangent/geometry.hpp"
namespace ab=lcbinint::algebraic_boundary::tangent;
extern "C" void* build(const double*p){return new ab::geometry_experiment::exact_detail::Chain({p[4],1/(1+p[3]),p[0],p[1],p[2]});}
extern "C" void destroy(void*p){delete static_cast<ab::geometry_experiment::exact_detail::Chain*>(p);}
extern "C" bool equal(void*ptr,int index,int degree,const char*value){auto&s=*static_cast<ab::geometry_experiment::exact_detail::Chain*>(ptr);return index<int(s.p.size())&&degree<int(s.p[index].size())&&s.p[index][degree].str()==value;}
extern "C" int size(void*ptr,int index){auto&s=*static_cast<ab::geometry_experiment::exact_detail::Chain*>(ptr);return index<0?s.p.size():s.p[index].size();}
extern "C" const char* value(void*ptr,int index,int degree){static thread_local std::string text;text=static_cast<ab::geometry_experiment::exact_detail::Chain*>(ptr)->p[index][degree].str();return text.c_str();}
