"""NIFTY 50 universe for the earnings-call panel: symbol → BSE scrip code, short name (call-model context), sector.

BSE scrip codes are checked at fetch time against the company name BSE returns (BSETranscriptFetcher.verify);
a mismatch is reported, never silently used. Tata Motors is left out: after the 2025 demerger (TMPV) the price
history restarts, so there is no continuous series for its GARCH baseline.
"""
from __future__ import annotations

# symbol: (BSE scrip code, short name, sector, word that must appear in BSE's company name)
NIFTY50 = {
    "TCS": ("532540", "tcs", "IT", "tata consultancy"), "INFY": ("500209", "infosys", "IT", "infosys"),
    "WIPRO": ("507685", "wipro", "IT", "wipro"), "HCLTECH": ("532281", "hcltech", "IT", "hcl"),
    "TECHM": ("532755", "tech mahindra", "IT", "tech mahindra"),
    "HDFCBANK": ("500180", "hdfc bank", "Banks", "hdfc bank"), "ICICIBANK": ("532174", "icici bank", "Banks", "icici bank"),
    "SBIN": ("500112", "sbi", "Banks", "state bank"), "AXISBANK": ("532215", "axis bank", "Banks", "axis bank"),
    "KOTAKBANK": ("500247", "kotak bank", "Banks", "kotak"),
    "BAJFINANCE": ("500034", "bajaj finance", "Financials", "bajaj finance"), "BAJAJFINSV": ("532978", "bajaj finserv", "Financials", "bajaj finserv"),
    "SHRIRAMFIN": ("511218", "shriram finance", "Financials", "shriram"), "JIOFIN": ("543940", "jio financial", "Financials", "jio financial"),
    "SBILIFE": ("540719", "sbi life", "Financials", "sbi life"), "HDFCLIFE": ("540777", "hdfc life", "Financials", "hdfc life"),
    "RELIANCE": ("500325", "reliance", "Energy", "reliance industries"), "ONGC": ("500312", "ongc", "Energy", "oil and natural gas"),
    "COALINDIA": ("533278", "coal india", "Energy", "coal india"), "NTPC": ("532555", "ntpc", "Energy", "ntpc"),
    "POWERGRID": ("532898", "power grid", "Energy", "power grid"),
    "ADANIENT": ("512599", "adani enterprises", "Industrials", "adani enterprises"), "ADANIPORTS": ("532921", "adani ports", "Industrials", "adani ports"),
    "LT": ("500510", "larsen and toubro", "Industrials", "larsen"), "BEL": ("500049", "bharat electronics", "Industrials", "bharat electronics"),
    "INDIGO": ("539448", "indigo", "Industrials", "interglobe"),
    "MARUTI": ("532500", "maruti suzuki", "Autos", "maruti"), "M&M": ("500520", "mahindra and mahindra", "Autos", "mahindra & mahindra"),
    "BAJAJ-AUTO": ("532977", "bajaj auto", "Autos", "bajaj auto"), "EICHERMOT": ("505200", "eicher motors", "Autos", "eicher"),
    "SUNPHARMA": ("524715", "sun pharma", "Health", "sun pharmaceutical"), "DRREDDY": ("500124", "dr reddy's", "Health", "reddy"),
    "CIPLA": ("500087", "cipla", "Health", "cipla"), "APOLLOHOSP": ("508869", "apollo hospitals", "Health", "apollo hospitals"),
    "MAXHEALTH": ("543220", "max healthcare", "Health", "max healthcare"),
    "ITC": ("500875", "itc", "Consumer", "itc"), "HINDUNILVR": ("500696", "hindustan unilever", "Consumer", "hindustan unilever"),
    "NESTLEIND": ("500790", "nestle india", "Consumer", "nestle"), "TATACONSUM": ("500800", "tata consumer", "Consumer", "tata consumer"),
    "TITAN": ("500114", "titan", "Consumer", "titan"), "TRENT": ("500251", "trent", "Consumer", "trent"),
    "ASIANPAINT": ("500820", "asian paints", "Consumer", "asian paints"), "ETERNAL": ("543320", "eternal", "Consumer", "eternal"),
    "BHARTIARTL": ("532454", "bharti airtel", "Telecom", "bharti airtel"),
    "TATASTEEL": ("500470", "tata steel", "Materials", "tata steel"), "JSWSTEEL": ("500228", "jsw steel", "Materials", "jsw steel"),
    "HINDALCO": ("500440", "hindalco", "Materials", "hindalco"), "ULTRACEMCO": ("532538", "ultratech cement", "Materials", "ultratech"),
    "GRASIM": ("500300", "grasim", "Materials", "grasim"),
}
BSE_CODE = {s: v[0] for s, v in NIFTY50.items()}
NAME = {s: v[1] for s, v in NIFTY50.items()}
SECTOR = {s: v[2] for s, v in NIFTY50.items()}
NAME_CHECK = {s: v[3] for s, v in NIFTY50.items()}
