#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
struct GaussRule {
    std::vector<double> x,w;
    explicit GaussRule(int n):x(n),w(n) {
        const long double pi=acosl(-1.L);
        for(int i=0;i<(n+1)/2;++i) {
            long double z=cosl(pi*(i+.75L)/(n+.5L)),derivative=0;
            for(int iteration=0;iteration<30;++iteration) {
                long double a=1,b=z;
                for(int j=2;j<=n;++j){const auto c=((2*j-1)*z*b-(j-1)*a)/j;a=b;b=c;}
                derivative=n*(z*b-a)/(z*z-1);
                const auto step=b/derivative;z-=step;
                if(fabsl(step)<4e-19L)break;
            }
            x[i]=-double(z);x[n-1-i]=double(z);
            w[i]=w[n-1-i]=double(2/((1-z*z)*derivative*derivative));
        }
    }
};
inline const GaussRule& gauss_rule(int n) {
    switch(n){
        case 10:{static const GaussRule r(10);return r;}
        case 8:{static const GaussRule r(8);return r;}
        case 12:{static const GaussRule r(12);return r;}
        case 16:{static const GaussRule r(16);return r;}
        case 24:{static const GaussRule r(24);return r;}
        case 32:{static const GaussRule r(32);return r;}
        case 64:{static const GaussRule r(64);return r;}
        case 128:{static const GaussRule r(128);return r;}
        default:throw std::invalid_argument("gradient Gauss order");
    }
}
}
