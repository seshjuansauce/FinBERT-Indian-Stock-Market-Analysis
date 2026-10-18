"""VolatilityModel (walk-forward correction of GJR-GARCH) and Evaluator (QLIKE, Diebold–Mariano, call-window test)."""
from __future__ import annotations

import numpy as np
import pandas as pd

HAR = ["lrv_d", "lrv_w", "lrv_m", "lmkt_d", "lmkt_w"]
ATT = ["log_n", "log_primary", "results_day", "log_focus", "no_news"]


class VolatilityModel:
    """u_{t+1} = ln RV_{t+1} − ln h_{t+1|t} regressed on feature set x_t, walk-forward (model/run_ladder.walk_forward)."""

    def __init__(self, feature_sets: dict[str, list[str] | None], refit: int = 21, min_train: int = 120, start: str = "2022-07-01"):
        self.sets, self.refit, self.min_train, self.start = feature_sets, refit, min_train, start

    @staticmethod
    def with_market(table: pd.DataFrame) -> pd.DataFrame:
        from run_ladder import market_rv
        t = table.join(market_rv(), how="left") if "lmkt_d" not in table else table
        return t

    def forecasts(self, table: pd.DataFrame) -> pd.DataFrame:
        from run_ladder import walk_forward
        d = self.with_market(table).dropna(subset=["gjr_fc", "rv_target"] + HAR)
        out = {}
        for name, feats in self.sets.items():
            if feats is None: out[name] = d["gjr_fc"]; continue
            missing = [f for f in feats if f not in d]
            if missing: raise KeyError(f"{name}: missing features {missing}")
            out[name] = walk_forward(d.fillna({f: 0.0 for f in feats}), feats, self.refit, self.min_train, self.start)
        return pd.DataFrame(out).assign(rv_target=d["rv_target"])

    def latest(self, table: pd.DataFrame, feats: list[str]) -> dict:
        """Fit on every row with a known target, forecast RV for the day after the last row."""
        d = self.with_market(table).dropna(subset=["gjr_fc"] + HAR).fillna({f: 0.0 for f in feats})
        tr = d.dropna(subset=["rv_target"]); X = tr[feats].values; mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
        Z = np.column_stack([np.ones(len(tr)), (X - mu) / sd]); beta, *_ = np.linalg.lstsq(Z, tr["u_gjr"].values, rcond=None)
        s2 = np.var(tr["u_gjr"].values - Z @ beta, ddof=Z.shape[1]); last = d.iloc[-1]
        z = np.concatenate([[1.0], (last[feats].values.astype(float) - mu) / sd])
        rv = float(last["gjr_fc"] * np.exp(z @ beta + s2 / 2))
        return {"as_of": str(d.index[-1].date()), "forecast_for": "next trading day", "rv_pct2": rv, "sigma_pct": rv ** 0.5,
                "garch_sigma_pct": float(last["gjr_fc"]) ** 0.5, "features": feats}


class Evaluator:
    def __init__(self, validation=("2022-07-01", "2023-12-31"), test_start="2024-01-01"):
        self.val, self.test_start = validation, test_start

    @staticmethod
    def qlike(rv, f): x = np.asarray(rv) / np.asarray(f); return x - np.log(x) - 1

    def table(self, fc: pd.DataFrame, extra_masks: dict[str, pd.Series] | None = None, baseline: str | None = None) -> pd.DataFrame:
        from run_ladder import dm_test
        names = [c for c in fc.columns if c != "rv_target"]; fc = fc.dropna()
        masks = {"validation": (fc.index >= self.val[0]) & (fc.index <= self.val[1]), "test": fc.index >= self.test_start}
        for k, m in (extra_masks or {}).items(): masks[k] = masks["test"] & m.reindex(fc.index).fillna(False).astype(bool).values
        rows = []
        for p, m in masks.items():
            if m.sum() < 5: continue
            for n in names:
                l = self.qlike(fc.rv_target[m], fc[n][m]); row = {"period": p, "model": n, "n": int(m.sum()), "qlike": float(l.mean())}
                if baseline and n != baseline:
                    st, pv = dm_test(l, self.qlike(fc.rv_target[m], fc[baseline][m])); row.update({"dm_vs_base": st, "p": pv})
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def call_window_test(panel: pd.DataFrame, features: list[str], target: str = "u_window") -> pd.DataFrame:
        """Pooled call-level regression: target (mean GARCH miss over the window after publication) on standardised call
        features, with company fixed effects and heteroskedasticity-robust (HC3) errors. One row per call."""
        import statsmodels.formula.api as smf
        d = panel.dropna(subset=[target] + features).copy()
        for f in features: d[f] = (d[f] - d[f].mean()) / (d[f].std() or 1)
        rows = []
        for f in features:
            r = smf.ols(f"{target} ~ {f} + C(company)", d).fit(cov_type="HC3")
            rows.append({"feature": f, "beta_per_sd": r.params[f], "t": r.tvalues[f], "p": r.pvalues[f], "n_calls": int(r.nobs), "r2": r.rsquared})
        return pd.DataFrame(rows).sort_values("p")
