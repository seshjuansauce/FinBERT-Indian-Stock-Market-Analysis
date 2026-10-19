"""NSE equity-derivatives daily bhavcopy (every option and future, every day) → one normalised table.

Two file formats exist (run on your Mac — NSE blocks most cloud IPs):
  old    ≤ 2024-07-05  …/content/historical/DERIVATIVES/{YYYY}/{MON}/fo{DD}{MON}{YYYY}bhav.csv.zip
         INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, …, OPEN_INT, …, TIMESTAMP
  UDiFF  ≥ 2024-07-08  …/content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip
         TradDt, FinInstrmTp (STO/STF/IDO/IDF), TckrSymb, XpryDt, StrkPric, OptnTp, ClsPric, SttlmPric, UndrlygPric,
         OpnIntrst, TtlTradgVol, …

Normalised columns: date, symbol, kind (option / future), expiry, strike, cp (call / put / ""), close, settle,
                    underlying (UDiFF only), oi, volume
Cached per day as data/options/raw/{YYYYMMDD}.parquet (filtered to the symbols you ask for), so re-runs are free.
"""
from __future__ import annotations
import io
import random
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = "https://nsearchives.nseindia.com"
OLD_URL = BASE + "/content/historical/DERIVATIVES/{y}/{mon}/fo{d:%d}{mon}{y}bhav.csv.zip"
NEW_URL = BASE + "/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
UDIFF_FROM = date(2024, 7, 8)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
COLS = ["date", "symbol", "kind", "expiry", "strike", "cp", "close", "settle", "underlying", "oi", "volume"]


def url_for(d: date) -> str:
    mon = d.strftime("%b").upper()
    return NEW_URL.format(d=d) if d >= UDIFF_FROM else OLD_URL.format(y=d.year, mon=mon, d=d)


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Either NSE format → COLS."""
    c = {k.strip(): k for k in raw.columns}; raw = raw.rename(columns=lambda k: k.strip())
    if "TckrSymb" in c:                                    # UDiFF
        kind = raw["FinInstrmTp"].map({"STO": "option", "IDO": "option", "STF": "future", "IDF": "future"})
        out = pd.DataFrame({"date": pd.to_datetime(raw["TradDt"]), "symbol": raw["TckrSymb"].str.strip(), "kind": kind,
                            "expiry": pd.to_datetime(raw["XpryDt"]), "strike": pd.to_numeric(raw["StrkPric"], errors="coerce"),
                            "cp": raw["OptnTp"].fillna("").str.strip().map({"CE": "call", "PE": "put"}).fillna(""),
                            "close": pd.to_numeric(raw["ClsPric"], errors="coerce"), "settle": pd.to_numeric(raw["SttlmPric"], errors="coerce"),
                            "underlying": pd.to_numeric(raw.get("UndrlygPric"), errors="coerce"),
                            "oi": pd.to_numeric(raw["OpnIntrst"], errors="coerce"), "volume": pd.to_numeric(raw["TtlTradgVol"], errors="coerce")})
    elif "INSTRUMENT" in c:                                # old format
        kind = raw["INSTRUMENT"].str.strip().map({"OPTSTK": "option", "OPTIDX": "option", "FUTSTK": "future", "FUTIDX": "future"})
        out = pd.DataFrame({"date": pd.to_datetime(raw["TIMESTAMP"], format="%d-%b-%Y"), "symbol": raw["SYMBOL"].str.strip(), "kind": kind,
                            "expiry": pd.to_datetime(raw["EXPIRY_DT"], format="%d-%b-%Y"), "strike": pd.to_numeric(raw["STRIKE_PR"], errors="coerce"),
                            "cp": raw["OPTION_TYP"].str.strip().map({"CE": "call", "PE": "put"}).fillna(""),
                            "close": pd.to_numeric(raw["CLOSE"], errors="coerce"), "settle": pd.to_numeric(raw["SETTLE_PR"], errors="coerce"),
                            "underlying": float("nan"), "oi": pd.to_numeric(raw["OPEN_INT"], errors="coerce"),
                            "volume": pd.to_numeric(raw["CONTRACTS"], errors="coerce")})
    else:
        raise ValueError(f"unknown bhavcopy format: {list(raw.columns)[:8]}")
    out = out[out["kind"].notna()]
    out.loc[out["kind"] == "future", "strike"] = float("nan")
    return out[COLS].reset_index(drop=True)


class NSEBhavcopy:
    def __init__(self, cache: Path, symbols=("TCS",), delay: float = 1.0, session=None):
        import requests
        self.cache, self.symbols, self.delay = Path(cache), set(symbols), delay
        self.cache.mkdir(parents=True, exist_ok=True); self.s = session or requests.Session(); self._seeded = False

    def _seed(self):
        try: self.s.get("https://www.nseindia.com/", headers={"User-Agent": UA}, timeout=20)
        except Exception: pass
        self._seeded = True

    def day(self, d: date) -> pd.DataFrame | None:
        """One trading day (None on holidays / missing files)."""
        p = self.cache / f"{d:%Y%m%d}.parquet"
        if p.exists(): return pd.read_parquet(p)
        if not self._seeded: self._seed()
        for attempt in range(3):
            try:
                r = self.s.get(url_for(d), headers={"User-Agent": UA, "Referer": "https://www.nseindia.com/"}, timeout=60)
            except Exception as e:
                time.sleep(2 ** attempt + 1); continue
            if r.status_code == 404: return None
            if r.status_code == 200 and r.content[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    raw = pd.read_csv(z.open(z.namelist()[0]))
                df = normalise(raw); df = df[df["symbol"].isin(self.symbols)]
                df.to_parquet(p, index=False); return df
            time.sleep(2 ** attempt + random.uniform(0.5, 2)); self._seed()
        return None

    def range(self, start: date, end: date, log=print) -> pd.DataFrame:
        out, d, n = [], start, 0
        while d <= end:
            if d.weekday() < 5:
                cached = (self.cache / f"{d:%Y%m%d}.parquet").exists()
                x = self.day(d); n += 1
                if x is not None and len(x): out.append(x)
                if not cached: time.sleep(random.uniform(0.6, 1.6) * self.delay)
                if log and d.day == 1: log(f"  {d:%Y-%m}: {sum(len(o) for o in out):,} rows so far")
            d += timedelta(days=1)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLS)
