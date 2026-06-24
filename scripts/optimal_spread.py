#!/usr/bin/env python3
"""The equation that balances reward harvesting against adverse selection.

The live strategies peg to the touch (s->0) because Polymarket's reward score ((v-s)/v)^2 is
maximised there. But the touch is also where fills are most frequent and most toxic. The right
objective trades the two off explicitly. Per side, per minute, in CENTS-per-share units:

    Net(s) = reward_subsidy(s)  +  lambda(s) * (s - eta(s)) * size

where
    s              quoted half-spread (distance of our quote from mid), in cents
    reward_subsidy reward $/min we earn for resting at depth s; proportional to the order score
                   ((v - s)/v)^2 capped by the band v (== reward_v_cents). reward_subsidy(0) = R0.
    lambda(s)      fill arrival rate (fills/min) at depth s; decays as we quote deeper.
    eta(s)         adverse selection per fill at depth s (the post-fill markout against us, in
                   cents/share) -- exactly the taker pickoff we measure from the tape.
    s - eta(s)     net trading edge per fill: we capture the half-spread s, lose eta to markout.

Two forces:
  * the SUBSIDY is largest at s=0 and pulls quotes IN (reward harvesting),
  * the TRADING term is negative near the touch (eta > s: toxic fills) and improves with depth,
    pushing quotes OUT.
The maximiser s* is therefore INTERIOR -- tighter than a naked market maker (the subsidy pays
you to lean in) but strictly wider than the touch (toxic fills aren't worth the extra reward).
That interior point is what touch-pegged quoting throws away.

Pure / dependency-free (stdlib only), unit-tested in tests/test_optimal_spread.py.
"""
from __future__ import annotations

import math
from typing import Callable

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_model import order_score  # noqa: E402


# --------------------------------------------------------------------------- components

def reward_subsidy(s: float, v: float, r0: float) -> float:
    """Reward $/min for resting one side at half-spread s (cents) given band v (cents).

    Proportional to the Polymarket order score; r0 is the subsidy at the touch (s=0), which is
    where the score equals 1. Zero beyond the band (s > v) or for degenerate inputs.
    """
    if v <= 0 or r0 <= 0:
        return 0.0
    return r0 * order_score(v, s)


def exp_lambda(a: float, k: float) -> Callable[[float], float]:
    """Parametric fill-rate model lambda(s) = A * exp(-k*s) (fills/min). A = rate at the touch."""
    def lam(s: float) -> float:
        return a * math.exp(-k * max(s, 0.0))
    return lam


def const_eta(eta0: float) -> Callable[[float], float]:
    """Constant adverse-selection-per-fill model eta(s) = eta0 (cents/share)."""
    return lambda s: eta0


def linear_eta(eta0: float, slope: float) -> Callable[[float], float]:
    """eta(s) = eta0 + slope*s. slope<0 if quoting deeper sheds the most toxic fills first."""
    return lambda s: eta0 + slope * s


# --------------------------------------------------------------------------- objective

def net_rate(s: float, *, v: float, r0: float, lam: Callable[[float], float],
             eta: Callable[[float], float], size: float = 1.0,
             inv: float = 0.0, inv_risk: float = 0.0) -> float:
    """Net $/min from resting one side at half-spread s.

    Net = reward_subsidy(s) + lambda(s)*(s - eta(s))*size - inv_risk*inv^2 (optional A-S-style
    inventory penalty, independent of s; included so callers can compare placements at a given
    inventory). s and eta are in cents/share, so the trading term is in cent-shares/min; divide by
    100 outside if dollars are wanted -- all tests work in consistent cent units.
    """
    trade = lam(s) * (s - eta(s)) * size
    return reward_subsidy(s, v, r0) + trade - inv_risk * inv * inv


def net_deriv(s: float, *, v: float, r0: float, lam: Callable[[float], float],
              eta: Callable[[float], float], size: float = 1.0, h: float = 1e-6) -> float:
    """Central finite-difference dNet/ds (the FOC residual is zero at an interior optimum)."""
    fp = net_rate(s + h, v=v, r0=r0, lam=lam, eta=eta, size=size)
    fm = net_rate(s - h, v=v, r0=r0, lam=lam, eta=eta, size=size)
    return (fp - fm) / (2 * h)


# --------------------------------------------------------------------------- solver

def solve_optimal_spread(*, v: float, r0: float, lam: Callable[[float], float],
                         eta: Callable[[float], float], size: float = 1.0,
                         lo: float = 0.0, hi: float | None = None,
                         tol: float = 1e-7) -> dict:
    """Maximise net_rate over s in [lo, hi] (hi defaults to the band v).

    Golden-section search refined against a coarse grid scan, so we are robust to the objective
    being multi-modal (the subsidy can create a second bump at the touch). Returns the optimal
    half-spread, its net rate, and the touch baseline for comparison.
    """
    if hi is None:
        hi = v
    f = lambda s: net_rate(s, v=v, r0=r0, lam=lam, eta=eta, size=size)

    # coarse scan to bracket the global max (guards against multi-modality)
    N = 200
    best_s, best_f = lo, f(lo)
    for i in range(N + 1):
        s = lo + (hi - lo) * i / N
        fs = f(s)
        if fs > best_f:
            best_f, best_s = fs, s
    a = max(lo, best_s - (hi - lo) / N)
    b = min(hi, best_s + (hi - lo) / N)

    # golden-section refine within the bracket
    gr = (math.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    while abs(b - a) > tol:
        if f(c) < f(d):
            a = c
        else:
            b = d
        c = b - gr * (b - a)
        d = a + gr * (b - a)
    s_star = (a + b) / 2
    return {
        "s_star": s_star,
        "net_star": f(s_star),
        "net_touch": f(0.0),
        "lambda_star": lam(s_star),
        "eta_star": eta(s_star),
        "reward_star": reward_subsidy(s_star, v, r0),
        "interior": tol < s_star < v - tol,
    }


def foc_closed_form_residual(s: float, *, v: float, r0: float, a: float, k: float,
                             eta0: float, size: float = 1.0) -> float:
    """Analytic FOC residual for the exponential-lambda / constant-eta case.

    Net(s) = r0*((v-s)/v)^2 + A*e^{-ks}*(s - eta0)*size
    Net'(s) = -2*r0*(v-s)/v^2 + A*e^{-ks}*size*(1 - k*(s - eta0))
    Returns Net'(s); a root in (0, v) is the interior optimum. Used to cross-check the numeric
    solver against calculus done by hand.
    """
    dreward = -2.0 * r0 * (v - s) / (v * v)
    dtrade = a * math.exp(-k * s) * size * (1.0 - k * (s - eta0))
    return dreward + dtrade


if __name__ == "__main__":
    # illustrative: toxic touch (eta0=0.6c), band v=3c, modest reward, fast fill decay
    lam = exp_lambda(a=40.0, k=1.2)
    eta = const_eta(0.6)
    for r0 in (0.0, 0.2, 1.0, 4.0):
        r = solve_optimal_spread(v=3.0, r0=r0, lam=lam, eta=eta, size=1.0)
        print(f"r0={r0:4.1f}  s*={r['s_star']:.3f}c  net*={r['net_star']:8.3f}  "
              f"net@touch={r['net_touch']:8.3f}  interior={r['interior']}")
