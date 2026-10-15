   if(hot&&hot->real){
    const double matching=std::min(hot->scale*.25,1e-10*(1+std::fabs(hot->center)));
    const bool left=std::fabs(panel.map.b-hot->left)<matching;
    const bool right=std::fabs(panel.map.a-hot->right)<matching;
    if(left||right){
     const long double a=(long double)panel.map.a+panel.left_radius_lo;
     const long double b=(long double)panel.map.b+panel.right_radius_lo;
     auto bound=[&](double x){if(x==-1)return a;if(x==1)return b;const long double t=(1.L+x)*.5L,w=b-a;
      if(map.left&&map.right){const long double s=sinl(3.1415926535897932384626433832795029L*t*.5L);return a+w*s*s;}
      if(map.left)return a+w*t*t;if(map.right)return b-w*(1-t)*(1-t);return a+w*t;
     };
     const long double lo=bound(panel.xl),hi=bound(panel.xr),critical=left?b:a,scale=hot->scale;
     auto y=[&](long double R){const long double distance=left?critical-R:R-critical;return 2*asinhl(sqrtl(distance/(2*scale)));};
     FoldRadialMap hyperbolic{double(y(lo)),double(y(hi)),
        map.left&&panel.xl==-1&&!right,map.right&&panel.xr==1&&!left};
     auto yy=hyperbolic(rule.x[k]);const long double sh=sinhl((long double)yy[0]*.5L);
     const long double radius=critical+(left?-1:1)*2*scale*sh*sh;
     mapped[0]=double(radius);radius_lo=double(radius-(long double)mapped[0]);
     jac=double((left?-1:1)*scale*sinhl((long double)yy[0])*yy[1]);
    }
   }
