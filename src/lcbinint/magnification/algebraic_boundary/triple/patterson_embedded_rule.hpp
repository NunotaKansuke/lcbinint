#pragma once
#include "radial_panel.hpp"
#include "patterson_rule.hpp"
namespace lcbinint::algebraic_boundary::triple::embedded_patterson_detail {
// Reuse the value controller's positive embedded-rule construction. Patterson
// nodes need different omissions from Fejer nodes to retain positive weights.
inline const panel_detail::EmbeddedRadialRule& embedded_radial_rule(int level){
    if(level==3){static const panel_detail::EmbeddedRadialRule q(patterson_rule(3),1,5);return q;}
    if(level==4){static const panel_detail::EmbeddedRadialRule q(patterson_rule(4),0,14);return q;}
    if(level==5){static const panel_detail::EmbeddedRadialRule q(patterson_rule(5),0,30);return q;}
    throw std::invalid_argument("embedded Patterson level");
}
template<class Samples,class Accessor> std::pair<double,double> embedded_estimate(int level,const Samples& samples,Accessor value){
    return panel_detail::embedded_estimate_from(patterson_rule(level),embedded_radial_rule(level),samples,value);
}
}
