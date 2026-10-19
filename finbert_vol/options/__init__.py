"""Options-implied volatility for NSE stocks: Black–Scholes / Black-76 (bs), NSE bhavcopy loader (data),
IV surface → constant-maturity 30-day IV (surface), and comparison with realised volatility and GARCH (compare)."""
from .bs import black76, bsm, greeks, implied_vol, implied_vols, forward_from_parity  # noqa: F401
