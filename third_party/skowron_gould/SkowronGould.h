#pragma once

#include <cmath>

#define MR 8
#define MT 10
#define MAXIT (MT*MR)
#define MAXM 30

class complex;

class complex {
public:
	double re;
	double im;
	inline complex(double a, double b) : re(a), im(b) {}
	inline complex(double a) : re(a), im(0.0) {}
	inline complex(void) : re(0.0), im(0.0) {}
};

inline double abs2(complex z) { return z.re * z.re + z.im * z.im; }
inline double abs(complex z) { return std::sqrt(abs2(z)); }
inline complex conj(complex z) { return complex(z.re, -z.im); }
inline complex sqrt(complex z)
{
	double md = std::sqrt(z.re * z.re + z.im * z.im);
	return (md > 0.0) ?
		complex(std::sqrt((md + z.re) / 2.0),
			std::sqrt((md - z.re) / 2.0) * ((z.im > 0.0) ? 1.0 : -1.0)) :
		0.0;
}
inline double real(complex z) { return z.re; }
inline double imag(complex z) { return z.im; }
inline complex cbrt(complex z)
{
	double r_cube = std::pow(abs(z), 1.0 / 3.0);
	double theta_cube = std::atan2(z.im, z.re) / 3.0;
	return complex(r_cube * std::cos(theta_cube), r_cube * std::sin(theta_cube));
}
inline complex expcmplx(complex z)
{
    double r = std::exp(z.re);
    return complex(r * std::cos(z.im), r * std::sin(z.im));
}
inline complex operator+(complex p1, complex p2) { return complex(p1.re + p2.re, p1.im + p2.im); }
inline complex operator-(complex p1, complex p2) { return complex(p1.re - p2.re, p1.im - p2.im); }
inline complex operator*(complex p1, complex p2)
{
	return complex(p1.re * p2.re - p1.im * p2.im, p1.re * p2.im + p1.im * p2.re);
}
inline complex operator/(complex p1, complex p2)
{
	double md = p2.re * p2.re + p2.im * p2.im;
	return complex((p1.re * p2.re + p1.im * p2.im) / md,
		(p1.im * p2.re - p1.re * p2.im) / md);
}
inline complex operator+(complex z, double a) { return complex(z.re + a, z.im); }
inline complex operator-(complex z, double a) { return complex(z.re - a, z.im); }
inline complex operator*(complex z, double a) { return complex(z.re * a, z.im * a); }
inline complex operator/(complex z, double a) { return complex(z.re / a, z.im / a); }
inline complex operator+(double a, complex z) { return complex(z.re + a, z.im); }
inline complex operator-(double a, complex z) { return complex(a - z.re, -z.im); }
inline complex operator*(double a, complex z) { return complex(a * z.re, a * z.im); }
inline complex operator/(double a, complex z)
{
	double md = z.re * z.re + z.im * z.im;
	return complex(a * z.re / md, -a * z.im / md);
}
inline complex operator+(complex z, int a) { return complex(z.re + a, z.im); }
inline complex operator-(complex z, int a) { return complex(z.re - a, z.im); }
inline complex operator*(complex z, int a) { return complex(z.re * a, z.im * a); }
inline complex operator/(complex z, int a) { return complex(z.re / a, z.im / a); }
inline complex operator+(int a, complex z) { return complex(z.re + a, z.im); }
inline complex operator-(int a, complex z) { return complex(a - z.re, -z.im); }
inline complex operator*(int a, complex z) { return complex(a * z.re, a * z.im); }
inline complex operator/(int a, complex z)
{
	double md = z.re * z.re + z.im * z.im;
	return complex(a * z.re / md, -a * z.im / md);
}
inline complex operator-(complex z) { return complex(-z.re, -z.im); }
inline bool operator==(complex p1, complex p2) { return p1.re == p2.re && p1.im == p2.im; }
inline bool operator!=(complex p1, complex p2) { return !(p1 == p2); }


void cmplx_roots_gen(complex *, complex *, int, bool, bool);
void cmplx_laguerre(complex *, int, complex *, int &, bool &);
void cmplx_newton_spec(complex *, int, complex *, int &, bool &);
void cmplx_laguerre2newton(complex *, int, complex *, int &, bool &, int);
void solve_quadratic_eq(complex &, complex &, complex *);
void solve_cubic_eq(complex &, complex &, complex &, complex *);
void divide_poly_1(complex *, complex, complex, complex *, int);
void cmplx_roots_5(complex *, bool, complex *, bool);
void sort_5_points_by_separation(complex *);
void sort_5_points_by_separation_i(int *, complex *);
void find_2_closest_from_5(int *, int *, double *, complex *);
