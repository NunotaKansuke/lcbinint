#pragma once
#include "../nested_fejer2.hpp"
namespace lcbinint::algebraic_boundary::triple {
// Binary tangent/fused_gaussian.hpp uses this same seven-node product rule:
// exact for A(x)+sqrt(1-x*x)*B(x), with quadratic A and B. The background
// and the component born between two real folds share the physical nodes.
inline const Fejer2Rule& mixed_fold_rule(int level){
 if(level==1)return fejer_rule(1);
 if(level==2){static const auto q=[](){auto r=fejer_rule(2);const double w=(2-std::acos(-1.)*.5)/(2*(1-std::sqrt(.5)));r.w={w,2-2*w,w};return r;}();return q;}
 if(level==3){static const auto q=[](){auto r=fejer_rule(3);r.w={.22294318587385589,.079567012897208009,.7050754885397208,-.015171374621569443,.7050754885397208,.079567012897208009,.22294318587385589};return r;}();return q;}
 throw std::invalid_argument("mixed fold level");
}
}
