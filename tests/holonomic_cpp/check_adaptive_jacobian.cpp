#include "adaptive_reference.hpp"
int main() {
    puts("name,u,parameter,step,n,stop,analytic,estimated_error,finite_difference,difference");
    const LensParams cases[]={ { .2,.142857142857,.125,.5,1.2,false },
        {.00027748104958798373,.000043678813767388989,8.1133264861482082e-5,1/.00017213162835819022,1.6869965260211353,true} };
    (void)fejer_rule(8);
    for(int k=0;k<2;++k)for(double u:{0.,.5}){
        auto p=cases[k];AdaptiveConfig c;c.with_jacobian=true;c.tol.mu_rtol=1e-7;c.tol.grad_rtol.fill(1e-4);
        AdaptiveWorkspace w;auto a=epoch_adaptive(p,u,c,w);
        for(int j=0;j<5;++j)for(double fraction:{1e-3,3e-4,1e-4})for(int n:{128,256}) {
            auto plus=p,minus=p;
            double* pp[]={&plus.xs,&plus.ys,&plus.rho,&plus.q,&plus.a};
            double* pm[]={&minus.xs,&minus.ys,&minus.rho,&minus.q,&minus.a};
            double scale=j<3?p.rho:std::fabs(*pp[j]);double h=fraction*scale;
            *pp[j]+=h;*pm[j]-=h;
            auto tp=classify_cells(PrimaryFrame::from(plus)),tm=classify_cells(PrimaryFrame::from(minus));
            double fd=(reference(plus,u,tp,n)-reference(minus,u,tm,n))/(2*h);
            printf("%s,%g,%d,%.17g,%d,%s,%.17g,%.9g,%.17g,%.9g\n",k?"paper_highA":"plan15",u,j,h,n,adaptive_stop_name(a.stop),a.grad_mu[j],a.estimated_abs_error_grad[j],fd,std::fabs(fd-a.grad_mu[j]));fflush(stdout);
        }
    }
}
