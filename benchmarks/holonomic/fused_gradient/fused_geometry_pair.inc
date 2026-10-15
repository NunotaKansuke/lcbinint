  // Mixed product integration: exact for A(x)+sqrt(1-x^2)*B(x),
  // with quadratic A and B, including the smooth background. The seven
  // fixed cosine nodes stay away from rounded physical endpoints.
  const bool pair_interior=hot&&hot->real&&panel.xl==-1&&panel.xr==1&&
      std::fabs(panel.map.a-hot->left)<std::min(hot->scale*.25,1e-10*(1+hot->left))&&
      std::fabs(panel.map.b-hot->right)<std::min(hot->scale*.25,1e-10*(1+hot->right));
  if(pair_interior){
   const int weighted_level=cfg.initial_level;
   const std::array<double,7> mixed_x{{0.92387953251128674,0.70710678118654757,0.38268343236508978,0,-0.38268343236508978,-0.70710678118654757,-0.92387953251128674}};
   const std::array<double,7> mixed_w{{0.22294318587385589,0.079567012897208009,0.7050754885397208,-0.015171374621569443,0.7050754885397208,0.079567012897208009,0.22294318587385589}};
   AdaptiveSample previous;bool have_previous=false;
   const long double a=(long double)panel.map.a+panel.left_radius_lo;
   const long double b=(long double)panel.map.b+panel.right_radius_lo;
   for(size_t k=0;k<mixed_x.size();++k){
    const double x=mixed_x[k];
    const long double radius=a+(b-a)*(1.L+x)*.5L;
    const double R=double(radius),lo=double(radius-(long double)R),jac=double((b-a)*.5L);
    if(primal_nodes+result.stats.node_evaluations>=cfg.max_node_evals){valid=false;failure=AdaptiveStop::BudgetExceeded;break;}
    auto sample=eval(R,jac,panel.evaluator_cell,have_previous?&previous:nullptr,false,lo,weighted_level,nullptr);
    ++result.stats.node_evaluations;
    if(!sample.reliable&&have_previous&&primal_nodes+result.stats.node_evaluations<cfg.max_node_evals){sample=eval(R,jac,panel.evaluator_cell,nullptr,true,lo,weighted_level,nullptr);++result.stats.node_evaluations;}
    if(!sample.reliable){valid=false;break;}
    for(int j=0;j<5;++j){
     const double term=mixed_w[k]*sample.value[j+1];
     valid=valid&&std::isfinite(term);sum[j].add(term);
    }
    sample.R=R;previous=std::move(sample);have_previous=true;
   }
   if(!valid)break;
   continue;
  }
