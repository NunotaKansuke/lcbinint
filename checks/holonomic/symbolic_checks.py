"""Symbolic / numerical re-verification of the ATPT design document.

Re-derives every boxed formula in implementation_plan_ja.md that is used
*before* any C++ is written, from the lens equation, independently of the
prototype scripts [V1]-[V4] (not in this repo).

Heavy resultant/discriminant identities are checked by specialising the
physical parameters (a, m0, xs, ys, rho2) to several rational tuples while
keeping the geometric variables (t, R, v=R^2) symbolic.  Three independent
tuples guard against accidental cancellation; a spot symbolic check is kept
where it is cheap.

Run:  python checks/holonomic/symbolic_checks.py
"""

import json
import math
import os
import random
import sys
import time
import sympy as sp

RESULTS, LOG = {}, []
T0 = time.time()


def log(m):
    s = f"[{time.time()-T0:6.1f}s] {m}"
    LOG.append(s)
    print(s, flush=True)


def record(name, ok, detail=""):
    RESULTS[name] = {"ok": bool(ok), "detail": str(detail)}
    log(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


t, R, a, m0, xs, ys, rho2, v = sp.symbols("t R a m0 xs ys rho2 v", real=True)
m1 = 1 - m0
I = sp.I

# boundary quartic, symbolic in everything ----------------------------------
A = 1 + t**2
B = (R - a) ** 2 + (R + a) ** 2 * t**2
zeta = xs + I * ys
n0 = -zeta * R**2
n1 = R * (R**2 - 1 + a * zeta)
n2 = a * (m0 - R**2)
Tt = n0 * (1 - I * t) ** 2 + n1 * (1 + t**2) + n2 * (1 + I * t) ** 2
Tb = (sp.conjugate(n0) * (1 + I * t) ** 2 + sp.conjugate(n1) * (1 + t**2)
      + sp.conjugate(n2) * (1 - I * t) ** 2)
P = sp.expand(sp.re(sp.expand(rho2 * R**2 * A * B - sp.expand(Tt * Tb))))
Ppoly = sp.Poly(P, t)
record("P_is_quartic_in_t", Ppoly.degree() == 4, f"deg_t P = {Ppoly.degree()}")

# rational parameter tuples (a, m0, xs, ys, rho2) ---------------------------
TUPLES = [
    (sp.Rational(6, 5), sp.Rational(2, 3), sp.Rational(1, 5), sp.Rational(1, 7), sp.Rational(1, 64)),
    (sp.Rational(9, 7), sp.Rational(2, 5), sp.Rational(-3, 8), sp.Rational(5, 9), sp.Rational(1, 100)),
    (sp.Rational(1, 2), sp.Rational(7, 10), sp.Rational(4, 11), sp.Rational(-2, 13), sp.Rational(9, 400)),
]


def subs_t(tp):
    return dict(zip((a, m0, xs, ys, rho2), tp))


def build_num(tp):
    """P, A, B, Q with (a,m0,xs,ys,rho2) fixed to the rational tuple tp."""
    av, m0v, xsv, ysv, r2v = tp
    An = 1 + t**2
    Bn = (R - av) ** 2 + (R + av) ** 2 * t**2
    zt = xsv + I * ysv
    a0, a1, a2 = -zt * R**2, R * (R**2 - 1 + av * zt), av * (m0v - R**2)
    Tn = a0 * (1 - I * t) ** 2 + a1 * (1 + t**2) + a2 * (1 + I * t) ** 2
    Tbn = (sp.conjugate(a0) * (1 + I * t) ** 2 + sp.conjugate(a1) * (1 + t**2)
           + sp.conjugate(a2) * (1 - I * t) ** 2)
    Pn = sp.expand(sp.re(sp.expand(r2v * R**2 * An * Bn - sp.expand(Tn * Tbn))))
    return Pn, An, Bn, sp.expand(Pn * An * Bn)

# --- 1. phi = P / (rho^2 R^2 A B)  (numeric, many random points) ----------
Pfun = sp.lambdify((t, R, a, m0, xs, ys, rho2), P, "math")
rng = random.Random(0)
num_ok, maxrel = True, 0.0
for _ in range(20000):
    A_, m0_ = rng.uniform(0.2, 3.0), rng.uniform(0.05, 0.95)
    xs_, ys_ = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
    rho_, R_ = rng.uniform(0.01, 0.3), rng.uniform(0.1, 2.5)
    th = rng.uniform(-3.14159, 3.14159)
    tt = math.tan(th / 2)
    zz = R_ * complex(math.cos(th), math.sin(th))
    zb = zz.conjugate()
    fzz = zz - m0_ / zb - (1 - m0_) / (zb - A_)
    phiv = 1 - abs(fzz - complex(xs_, ys_)) ** 2 / rho_**2
    den = rho_**2 * R_**2 * (1 + tt**2) * ((R_ - A_) ** 2 + (R_ + A_) ** 2 * tt**2)
    rel = abs(phiv - Pfun(tt, R_, A_, m0_, xs_, ys_, rho_**2) / den) / (abs(phiv) + 1e-30)
    maxrel = max(maxrel, rel)
    num_ok &= rel < 1e-9
record("phi_equals_P_over_rho2R2AB_numeric", num_ok, f"20000 pts, max rel = {maxrel:.2e}")

# --- 2. Disc_t P = R^4 D14(R^2), deg_v D14 <= 14 -------------------------
ok_all, det = True, []
for i, tp in enumerate(TUPLES):
    Ps = build_num(tp)[0]
    disc = sp.expand(sp.discriminant(Ps, t))
    Rpw = sorted({mm[0] for mm in sp.Poly(disc, R).monoms()})
    q_v = sp.expand((disc / R**4).subs(R, sp.sqrt(v)))
    dv = sp.Poly(q_v, v)
    intpow = all(mp[0] == int(mp[0]) for mp in dv.monoms())
    good = min(Rpw) >= 4 and intpow and dv.degree() <= 14
    ok_all &= good
    det.append(f"t{i}: Rpowers={Rpw} deg_vD14={dv.degree()}")
record("disc_P_is_R4_times_D14_of_R2", ok_all, "; ".join(det))

# Res_t(P,P_t) = p4 * Disc_t P  (symbolic, cheap-ish per tuple)
ok_all = True
for tp in TUPLES:
    Ps = build_num(tp)[0]
    p4 = sp.Poly(Ps, t).nth(4)
    r = sp.expand(sp.resultant(Ps, sp.diff(Ps, t), t) - p4 * sp.discriminant(Ps, t))
    ok_all &= r == 0
record("res_P_Pt_equals_p4_times_disc", ok_all, "")

# --- 3. Disc/Res of A, B, and Res(P,A), Res(P,B) -----------------------
record("disc_A_is_minus4", sp.simplify(sp.discriminant(A, t) + 4) == 0, "")
record("disc_B_is_minus4_R2ma2_sq",
       sp.expand(sp.discriminant(B, t) + 4 * (R**2 - a**2) ** 2) == 0, "")
record("res_A_B_is_16a2R2", sp.expand(sp.resultant(A, B, t) - 16 * a**2 * R**2) == 0, "")

okPA, okPB, dPA, dPB = True, True, [], []
for i, tp in enumerate(TUPLES):
    av, m0v, xsv, ysv, r2v = tp
    Ps, An, Bn, _ = build_num(tp)
    rPA = sp.expand(sp.resultant(Ps, An, t))
    tgtPA = sp.expand(256 * av**2 * R**4 * (m0v - R**2) ** 2 * (xsv**2 + ysv**2))
    okPA &= sp.expand(rPA - tgtPA) == 0
    dPA.append(f"t{i}:{sp.expand(rPA - tgtPA)==0}")

    rPB = sp.expand(sp.resultant(Ps, sp.expand(Bn), t).subs(R, sp.sqrt(v)))
    Lv = (av - xsv) * v * (v - av**2) - av * v + av**3 * m0v
    tgtPB = sp.expand(256 * av**2 * v**2 * (1 - m0v) ** 2
                      * (Lv**2 + ysv**2 * v**2 * (v - av**2) ** 2))
    okPB &= sp.expand(rPB - tgtPB) == 0
    dPB.append(f"t{i}:{sp.expand(rPB - tgtPB)==0}")
record("res_P_A_formula", okPA, " ".join(dPA))
record("res_P_B_formula", okPB, " ".join(dPB))

# --- 4. LD reduction: P/A = H + (S/A)' Q + (S/A) Q'/2  (symbolic) -------
Q = sp.expand(P * A * B)
S = -t / (4 * a * R)
Hplan = (2 * (R + a) ** 2 * P + t * (B * sp.diff(P, t) + sp.diff(B, t) * P)) / (8 * a * R)
red = sp.simplify(sp.expand(P / A) - sp.expand(
    Hplan + sp.diff(S / A, t) * Q + (S / A) * sp.diff(Q, t) / 2))
record("LD_reduction_identity", red == 0, f"residual={red}")
record("H_degree_le_6", sp.Poly(sp.expand(Hplan), t).degree() <= 6,
       f"deg_t H = {sp.Poly(sp.expand(Hplan), t).degree()}")

# --- 5. Gauss-Manin connection polynomial identity (numeric R) ----------
gm_ok, gm_det = True, []
for tp in TUPLES:
    Pn, An, Bn, Qsym = build_num(tp)
    for Rv in (sp.Rational(9, 10), sp.Rational(13, 10), sp.Rational(23, 20)):
        sd = {R: Rv}
        Qn = sp.Poly(sp.expand(Qsym.subs(sd)), t)
        QRn = sp.Poly(sp.expand(sp.diff(Qsym, R).subs(sd)), t)
        Qtn = sp.Poly(sp.expand(sp.diff(Qsym, t).subs(sd)), t)
        if Qn.degree() != 8 or sp.gcd(Qn, Qtn).degree() != 0:
            continue
        inv = Qtn.invert(Qn)  # (Q_t)^{-1} mod Q
        for k in range(7):
            Sk = (sp.Poly(t**k, t) * QRn * inv).rem(Qn)
            num2 = sp.Poly(sp.expand(Sk.as_expr() * Qtn.as_expr() - t**k * QRn.as_expr()), t)
            Tk, rem = sp.div(num2, sp.Poly(sp.expand(2 * Qn.as_expr()), t))
            Ck = sp.expand(Tk.as_expr() - sp.diff(Sk.as_expr(), t))
            idr = sp.expand(-sp.Rational(1, 2) * t**k * QRn.as_expr()
                            - (Ck * Qn.as_expr() + sp.diff(Sk.as_expr(), t) * Qn.as_expr()
                               - sp.Rational(1, 2) * Sk.as_expr() * Qtn.as_expr()))
            deg_ok = Ck == 0 or sp.Poly(Ck, t).degree() <= 6
            if idr != 0 or not deg_ok or sp.expand(rem.as_expr()) != 0:
                gm_ok = False
                gm_det.append(f"tp={tp} R={Rv} k={k}")
record("gauss_manin_polynomial_identity", gm_ok,
       "; ".join(gm_det) or "all tuples/R, k=0..6: exact, deg C_k<=6, T_k polynomial")

# --- 6. regularized root pair (m, v) -----------------------------------
mm, vv, s = sp.symbols("mm vv s", real=True)
c0, c1, c2, c3, c4 = sp.symbols("c0 c1 c2 c3 c4", real=True)
Pg = c4 * t**4 + c3 * t**3 + c2 * t**2 + c1 * t + c0
d2, d3, d4 = (sp.diff(Pg, t, n) for n in (2, 3, 4))
E = Pg.subs(t, mm) + vv / 2 * d2.subs(t, mm) + vv**2 / 24 * d4.subs(t, mm)
O = sp.diff(Pg, t).subs(t, mm) + vv / 6 * d3.subs(t, mm)
record("rootpair_E_exact",
       sp.expand(Pg.subs(t, mm + s) + Pg.subs(t, mm - s) - (2 * E).subs(vv, s**2)) == 0, "")
record("rootpair_O_exact",
       sp.expand(Pg.subs(t, mm + s) - Pg.subs(t, mm - s) - (2 * s * O).subs(vv, s**2)) == 0, "")
J = sp.Matrix([[sp.diff(E, mm), sp.diff(E, vv)], [sp.diff(O, mm), sp.diff(O, vv)]])
detJ0 = sp.expand(J.det().subs(vv, 0))
# at a genuine double root v=0 we also have P'(m)=0; impose it (solve c1)
Pprime_m = sp.diff(Pg, t).subs(t, mm)
c1_sol = sp.solve(Pprime_m, c1)[0]
detJ0_dr = sp.expand(detJ0.subs(c1, c1_sol))
tgt_dr = sp.expand((-sp.Rational(1, 2) * d2.subs(t, mm) ** 2).subs(c1, c1_sol))
record("rootpair_jacobian_det_at_v0",
       sp.expand(detJ0_dr - tgt_dr) == 0,
       "det = -1/2 P''(m)^2 modulo P'(m)=0 ; raw(det+1/2P''^2) = (c3+4c4 m) P'(m)")

# --- 7. R_max search bound (numeric) ---------------------------------
rmax_ok = True
for _ in range(3000):
    A_, m0_ = rng.uniform(0.1, 3.0), rng.uniform(0.05, 0.95)
    xs_, ys_ = rng.uniform(-2, 2), rng.uniform(-2, 2)
    rho_ = rng.uniform(1e-3, 0.3)
    W = math.hypot(xs_, ys_) + rho_
    Rmax = (A_ + W + math.hypot(A_ - W, 2.0)) / 2 + 1e-9
    for Rt in (Rmax * 1.0001, Rmax * 1.5, Rmax * 5):
        for k in range(36):
            th = k * math.pi / 18
            zz = Rt * complex(math.cos(th), math.sin(th))
            zb = zz.conjugate()
            fzz = zz - m0_ / zb - (1 - m0_) / (zb - A_)
            if 1 - abs(fzz - complex(xs_, ys_)) ** 2 / rho_**2 > 0:
                rmax_ok = False
record("Rmax_bound_no_image_outside", rmax_ok, "3000 configs x 3 radii x 36 angles")

summary = {"all_pass": all(r["ok"] for r in RESULTS.values()), "results": RESULTS}
os.makedirs("evidence/holonomic", exist_ok=True)
json.dump(summary, open("evidence/holonomic/symbolic_checks.json", "w"), indent=2)
open("evidence/holonomic/symbolic_checks.txt", "w").write("\n".join(LOG) + "\n")
print("\nALL PASS" if summary["all_pass"] else "\nSOME FAILURES", flush=True)
sys.exit(0 if summary["all_pass"] else 1)
