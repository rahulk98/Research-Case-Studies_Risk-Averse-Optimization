#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wobbly Rosenbrock (parameter noise in 'a'): nominal vs. CVaR-optimized

Usage examples:
  python wobbly_rosenbrock_cvar_param_noise.py               # default: compare
  python wobbly_rosenbrock_cvar_param_noise.py --mode compare
  python wobbly_rosenbrock_cvar_param_noise.py --alpha 0.99 --N 500 --sigma-a 0.1

Requires: casadi>=3.6, numpy
"""
import argparse
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
def rosenbrock(x, a, b):
    x1, x2 = x[0], x[1]
    return (a - x1)**2 + b*(x2 - x1**2)**2

def cvar_empirical(values, alpha: float):
    """
    Empirical CVaR_alpha for a 1D array of loss values:
      CVaR = mean of the upper (1-alpha) tail (>= VaR_alpha).
    """
    values = np.asarray(values, dtype=float)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if values.size == 0:
        return np.nan
    # Empirical VaR at level alpha
    var_alpha = np.quantile(values, alpha)
    tail = values[values >= var_alpha]
    return float(tail.mean())

def solve_nominal(a=1.0, b=100.0, lbx=(-2.0,-2.0), ubx=(2.0,2.0), x0=(-1.2,1.0)):
    """Minimize the deterministic Rosenbrock (no noise)."""
    x = ca.MX.sym('x', 2) # type: ignore
    f = rosenbrock(x, a, b)
    nlp = {'x': x, 'f': f}
    solver = ca.nlpsol('nom', 'ipopt', nlp, {'ipopt.print_level':0,'print_time':0})
    sol = solver(x0=x0, lbx=lbx, ubx=ubx)
    x_nom = np.array(sol['x']).squeeze()
    f_nom = float(sol['f'])
    return x_nom, f_nom

def solve_cvar(a=1.0, b=100.0, alpha=0.95, Xi=None, lbx=(-2.0,-2.0), ubx=(2.0,2.0), x0=(-1.2,1.0)):
    """
    Solve CVaR minimization using RU formulation with parameter noise in 'a'.
    Xi: array of shape (N,), zero-mean noise samples for 'a'.
    """
    assert Xi is not None and Xi.ndim == 1
    N = Xi.shape[0]

    x = ca.MX.sym('x', 2) # type: ignore
    t = ca.MX.sym('t') # type: ignore
    u = ca.MX.sym('u', N) # type: ignore

    g = []
    for i in range(N):
        a_eff = a + float(Xi[i])
        fi = rosenbrock(x, a_eff, b)
        g.append(u[i] - (fi - t))   # u_i >= f - t
        g.append(u[i])              # u_i >= 0

    obj = t + (1.0/((1.0 - alpha)*N))*ca.sum1(u)

    w  = ca.vertcat(x, t, u)
    lbg = [0.0]*len(g)
    ubg = [ca.inf]*len(g)
    lbw = list(lbx) + [-ca.inf] + [0.0]*N
    ubw = list(ubx) + [ ca.inf] + [ca.inf]*N
    w0  = np.r_[x0, 1.0, np.zeros(N)]

    nlp = {'x': w, 'f': obj, 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('cvar', 'ipopt', nlp, {'ipopt.print_level':0,'print_time':0})
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = np.array(sol['x']).squeeze()
    x_cvar = w_opt[:2]
    t_cvar = w_opt[2]
    obj_cvar = float(sol['f'])
    return x_cvar, t_cvar, obj_cvar

def evaluate_losses(xval, a, b, Xi):
    """Compute array of losses f(x, xi_a) for samples Xi."""
    vals = np.array([float((a + z - xval[0])**2 + b*(xval[1] - xval[0]**2)**2) for z in Xi], dtype=float)
    return vals

# def plot(losses_nom, losses_cvar):
#     """Plot losses in two separate subplots."""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     ax1.hist(losses_nom, bins=100, alpha=0.7, color='C0')
#     ax1.set_xscale('log')
#     ax1.set_title('Nominal')
#     ax1.set_xlabel('Loss f(x,a+ξ)')
#     ax1.set_ylabel('Frequency')

#     ax2.hist(losses_cvar, bins=100, alpha=0.7, color='C1')
#     ax2.set_xscale('log')
#     ax2.set_title('CVaR-optimized')
#     ax2.set_xlabel('Loss f(x,a+ξ)')
#     ax2.set_ylabel('Frequency')

#     fig.tight_layout()
#     # labels and legends are set per-axis above
#     plt.show()


def main():
    parser = argparse.ArgumentParser(description="Wobbly Rosenbrock: nominal vs CVaR-optimized comparison")
    parser.add_argument('--mode', choices=['compare','nominal','cvar'], default='compare',
                        help="Run nominal only, CVaR only, or both and compare (default).")
    parser.add_argument('--alpha', type=float, default=0.95, help="CVaR level in (0,1).")
    parser.add_argument('--N', type=int, default=300, help="Number of scenarios for training (optimization).")
    parser.add_argument('--M', type=int, default=5000, help="Number of scenarios for evaluation (oos CVaR).")
    parser.add_argument('--sigma-a', type=float, default=0.05, dest='sigma_a', help="Std-dev of noise in 'a'.")
    parser.add_argument('--a', type=float, default=1.0, help="Nominal parameter a.")
    parser.add_argument('--b', type=float, default=100.0, help="Rosenbrock parameter b>0.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    a, b = args.a, args.b
    alpha = args.alpha
    N, M = args.N, args.M
    sigma_a = args.sigma_a

    rng = np.random.default_rng(args.seed)
    Xi_train = rng.normal(0.0, sigma_a, size=N) 
    Xi_eval  = rng.normal(0.0, sigma_a, size=M) 

    # Testing with heavy-tailed noise (Student's t with 3 dof)
    # Xi_train = rng.standard_t(df=3, size=N) * sigma_a
    # Xi_eval  = rng.standard_t(df=3, size=M) * sigma_a


    lbx, ubx = (-2.0,-2.0), (2.0,2.0)
    x0 = (-1.2, -1.0)

    if args.mode in ('compare','nominal'):
        x_nom, f_nom = solve_nominal(a=a, b=b, lbx=lbx, ubx=ubx, x0=x0)
        losses_nom = evaluate_losses(x_nom, a, b, Xi_eval)
        cvar_nom = cvar_empirical(losses_nom, alpha)
        mean_nom = float(losses_nom.mean())
        print("Nominal solution (deterministic Rosenbrock):")
        print("  x_nom =", x_nom, "  f(x_nom; a,b) =", f_nom)
        print("  OOS (M=%d) mean=%.6f  CVaR@alpha=%.2f = %.6f" % (M, mean_nom, alpha, cvar_nom))
        print()

    if args.mode in ('compare','cvar'):
        x_cvar, t_cvar, obj_cvar = solve_cvar(a=a, b=b, alpha=alpha, Xi=Xi_train, lbx=lbx, ubx=ubx, x0=x0)
        losses_cvar = evaluate_losses(x_cvar, a, b, Xi_eval)
        cvar_cvar = cvar_empirical(losses_cvar, alpha)
        mean_cvar = float(losses_cvar.mean())
        print("CVaR-optimized solution (train N=%d, alpha=%.2f):" % (N, alpha))
        print("  x_cvar =", x_cvar, "  t* =", t_cvar, "  training objective (RU) =", obj_cvar)
        print("  OOS (M=%d) mean=%.6f  CVaR@alpha=%.2f = %.6f" % (M, mean_cvar, alpha, cvar_cvar))
        print()

    if args.mode == 'compare':
        # If both were computed, add a compact comparison line
        print("Comparison (OOS): CVaR@alpha=%.2f" % alpha)
        print("  CVaR(x_cvar)  vs  CVaR(x_nom)  -->  %.6f  vs  %.6f" % (cvar_cvar, cvar_nom))
        print("  Mean(x_cvar)  vs  Mean(x_nom)  -->  %.6f  vs  %.6f" % (mean_cvar, mean_nom))
        # plot(losses_nom, losses_cvar)
if __name__ == "__main__":
    main()
