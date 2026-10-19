"""Black–Scholes(–Merton) and Black-76: prices, Greeks, implied volatility.

Black–Scholes–Merton (spot S, continuous dividend yield q, risk-free r, time T in years):
    d1 = [ln(S/K) + (r − q + σ²/2)T] / (σ√T),   d2 = d1 − σ√T
    C  = S e^{−qT} N(d1) − K e^{−rT} N(d2)
    P  = K e^{−rT} N(−d2) − S e^{−qT} N(−d1)

Black-76 (forward F for the option's expiry; what we use for NSE options, because the forward can be read off the
futures price or put–call parity, so the dividend yield never has to be guessed):
    F  = S e^{(r−q)T}
    d1 = [ln(F/K) + σ²T/2] / (σ√T),   d2 = d1 − σ√T
    C  = e^{−rT} [F N(d1) − K N(d2)]
    P  = e^{−rT} [K N(−d2) − F N(−d1)]
The two are the same model: plug F = S e^{(r−q)T} into Black-76 and you get BSM.

Put–call parity (model-free):  C − P = e^{−rT} (F − K)   ⇒   F = K + e^{rT} (C − P)

Greeks (Black-76, per unit of F):
    Δ_call = e^{−rT} N(d1),   Δ_put = −e^{−rT} N(−d1)
    Γ      = e^{−rT} φ(d1) / (F σ √T)
    vega   = e^{−rT} F φ(d1) √T            (per 1.00 of σ; divide by 100 for "per vol point")
    Θ_call = −e^{−rT} F φ(d1) σ / (2√T) + r C          (per year; /365 for per calendar day)

Implied volatility σ_imp solves model(σ) = market price. Price is strictly increasing in σ (vega > 0), so the root is
unique when the price lies inside the no-arbitrage bounds  e^{−rT}(F−K)⁺ < C < e^{−rT}F  (puts analogously).
Solved with Brent's method on σ ∈ [1e-4, 5]; NaN when the price is outside the bounds.
"""
from __future__ import annotations
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

N, phi = norm.cdf, norm.pdf


def _d1d2(F, K, T, sigma):
    s = sigma * np.sqrt(T); d1 = (np.log(F / K) + 0.5 * s * s) / s
    return d1, d1 - s


def black76(F, K, T, r, sigma, kind="call"):
    """Price of a European option on a forward (vectorised)."""
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma)); df = np.exp(-r * T); d1, d2 = _d1d2(F, K, T, sigma)
    call = df * (F * N(d1) - K * N(d2)); put = df * (K * N(-d2) - F * N(-d1))
    return np.where(np.asarray(kind) == "call", call, put) if np.ndim(kind) else (call if kind == "call" else put)


def bsm(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black–Scholes–Merton on spot with dividend yield q (= Black-76 with F = S e^{(r−q)T})."""
    return black76(np.asarray(S) * np.exp((r - q) * np.asarray(T)), K, T, r, sigma, kind)


def greeks(F, K, T, r, sigma, kind="call") -> dict:
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma)); df = np.exp(-r * T); d1, d2 = _d1d2(F, K, T, sigma)
    sq = np.sqrt(T); price = black76(F, K, T, r, sigma, kind)
    delta = df * N(d1) if kind == "call" else -df * N(-d1)
    gamma = df * phi(d1) / (F * sigma * sq)
    vega = df * F * phi(d1) * sq
    theta = -df * F * phi(d1) * sigma / (2 * sq) + r * price
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "d1": d1, "d2": d2}


def bounds(F, K, T, r, kind="call") -> tuple[float, float]:
    df = math.exp(-r * T)
    return (df * max(F - K, 0.0), df * F) if kind == "call" else (df * max(K - F, 0.0), df * K)


def implied_vol(price, F, K, T, r, kind="call", lo=1e-4, hi=5.0, tol=1e-8) -> float:
    """σ such that black76(F, K, T, r, σ) = price; NaN if the price breaks no-arbitrage bounds or T ≤ 0."""
    if not (price > 0 and F > 0 and K > 0 and T > 0) or any(map(math.isnan, (price, F, K, T))): return float("nan")
    lb, ub = bounds(F, K, T, r, kind)
    if not (lb + 1e-10 < price < ub - 1e-10): return float("nan")
    f = lambda s: float(black76(F, K, T, r, s, kind)) - price
    if f(lo) > 0 or f(hi) < 0: return float("nan")
    return float(brentq(f, lo, hi, xtol=tol, maxiter=200))


def implied_vols(prices, F, K, T, r, kinds) -> np.ndarray:
    """Row-wise implied_vol over arrays (r may be scalar or array)."""
    n = len(prices); r = np.broadcast_to(np.asarray(r, dtype=float), (n,))
    return np.array([implied_vol(float(p), float(f), float(k), float(t), float(rr), str(kd))
                     for p, f, k, t, rr, kd in zip(prices, F, K, T, r, kinds)])


def forward_from_parity(call, put, K, T, r) -> float:
    """F = K + e^{rT}(C − P)."""
    return float(K + math.exp(r * T) * (call - put))
