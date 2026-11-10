"""Volatility spillover network (Diebold & Yilmaz 2012): who transmits volatility shocks to whom.

Input: daily log realised variance y_t = (ln RV_{1,t}, …, ln RV_{N,t}) for N nodes (stocks and indices).

1. VAR(p):  y_t = c + Σ_{l=1..p} Φ_l y_{t−l} + ε_t,   ε_t ~ (0, Σ),   p by BIC (≤ 5)
2. MA form: y_t = Σ_{h≥0} A_h ε_{t−h}
3. Generalised forecast-error variance decomposition at horizon H (Pesaran–Shin; invariant to variable order):
       θ_ij(H) = σ_jj⁻¹ Σ_{h<H} (e_i′ A_h Σ e_j)²  /  Σ_{h<H} (e_i′ A_h Σ A_h′ e_i)
   row-normalised:  θ̃_ij = θ_ij / Σ_j θ_ij     → share of node i's H-step forecast-error variance due to shocks in j
4. Connectedness (all in %):
       FROM_i  = Σ_{j≠i} θ̃_ij            how much of i's volatility comes from the others  (received)
       TO_j    = Σ_{i≠j} θ̃_ij            how much j contributes to the others              (transmitted)
       NET_j   = TO_j − FROM_j            > 0: net transmitter,  < 0: net receiver
       NPDC_ij = θ̃_ji − θ̃_ij             net pairwise flow  j → i
       TOTAL   = (1/N) Σ_{i≠j} θ̃_ij      system-wide spillover index
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def realised_log_variance(prices: pd.DataFrame, symbols: dict[str, str]) -> pd.DataFrame:
    """prices: long table (Date, Symbol, Open, High, Low, Close, Adj Close, Volume) → wide ln RV, one column per node.
    RV_t = 10⁴ [ln(O_t/C_{t−1})² + ln(H_t/L_t)²/(4 ln 2)] on dividend-adjusted prices (same as the rest of the project)."""
    from .live.model import adjusted
    out = {}
    for name, sym in symbols.items():
        p = adjusted(prices, sym)
        rv = 1e4 * (np.log(p.Open / p.Close.shift(1)) ** 2 + np.log(p.High / p.Low) ** 2 / (4 * np.log(2)))
        out[name] = np.log(rv.replace(0, np.nan))
    d = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)
    return d.dropna()


class Connectedness:
    def __init__(self, horizon: int = 10, max_lags: int = 5, lags: int | None = None):
        self.H, self.max_lags, self.lags = horizon, max_lags, lags

    def gfevd(self, y: pd.DataFrame) -> pd.DataFrame:
        from statsmodels.tsa.api import VAR
        m = VAR(y.values)
        p = self.lags or max(1, int(m.select_order(self.max_lags).selected_orders["bic"]))
        r = m.fit(p)
        A = r.ma_rep(self.H - 1)                       # (H, N, N): A_0 = I, …
        S = r.sigma_u; N = y.shape[1]
        num = np.zeros((N, N)); den = np.zeros(N)
        for h in range(self.H):
            AS = A[h] @ S
            num += AS ** 2                             # (e_i′ A_h Σ e_j)²
            den += np.diag(A[h] @ S @ A[h].T)
        theta = num / np.diag(S)[None, :] / den[:, None]
        theta = theta / theta.sum(1, keepdims=True)
        self.p_ = p
        return pd.DataFrame(theta, index=y.columns, columns=y.columns)

    @staticmethod
    def table(theta: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        t = 100 * theta.values; N = len(t); off = t - np.diag(np.diag(t))
        df = pd.DataFrame({"FROM": off.sum(1), "TO": off.sum(0)}, index=theta.index)
        df["NET"] = df.TO - df.FROM; df["OWN"] = np.diag(t)
        return df, float(off.sum() / N)

    @staticmethod
    def pairwise_net(theta: pd.DataFrame) -> pd.DataFrame:
        """NPDC[i, j] = θ̃_ji − θ̃_ij (in %): positive = net flow from j to i … returned as a source→target edge list."""
        t = 100 * theta.values; names = list(theta.index); rows = []
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                f = t[b, a] - t[a, b]                 # a→b minus b→a
                src, dst = (names[a], names[b]) if f > 0 else (names[b], names[a])
                rows.append({"source": src, "target": dst, "net_pct": abs(f)})
        return pd.DataFrame(rows).sort_values("net_pct", ascending=False).reset_index(drop=True)

    def rolling(self, y: pd.DataFrame, window: int = 250, step: int = 5) -> tuple[pd.Series, pd.DataFrame]:
        """Total spillover index and NET per node on a moving window (value dated at the window's last day)."""
        tot, net, idx = [], [], []
        for e in range(window, len(y) + 1, step):
            th = self.gfevd(y.iloc[e - window:e]); tb, ti = self.table(th)
            tot.append(ti); net.append(tb.NET.values); idx.append(y.index[e - 1])
        return pd.Series(tot, index=idx, name="total"), pd.DataFrame(net, index=idx, columns=y.columns)
