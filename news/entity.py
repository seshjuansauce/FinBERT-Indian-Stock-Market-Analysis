"""Entity-centric feature extraction: what does THIS article say about THIS company?

For each (article, target company):
  title_norm      source suffixes / section prefixes stripped
  mentions        tracked companies named in the title (alias dictionary below)
  n_companies     tracked + other company-like names in list structures
  target_span     the clause about the target ("HCLTech drags"), else the list segment, else the title
  role            primary | co_mention | list_member | peripheral (target only in description) | wrong_entity
  article_type    market_wrap, stocks_to_watch, buy_sell_reco, results_date, results_preview, results, deal_win,
                  analyst_view, dividend, management, legal, non_business, other
  relevance       primary 1.0 / co_mention 0.5 / list_member 0.2 / peripheral 0.1 / wrong_entity|non_business 0
"""
from __future__ import annotations
import re

ALIASES = {
    "TCS":     [r"tata consultancy services", r"\btcs\b"],
    "INFY":    [r"\binfosys\b", r"\binfy\b"],
    "WIPRO":   [r"\bwipro\b"],
    "HCLTECH": [r"\bhcl ?tech(?:nologies)?\b", r"\bhcl\b(?! infosystems)"],
    "TECHM":   [r"\btech ?mahindra\b", r"\btechm\b"],
}
NOT_ENTITY = {  # same string, different thing
    "TCS": r"tcs property management|tax collected at source|\btcs (?:rate|deduction|on (?:sale|foreign|lrs))",
    "HCLTECH": r"hcl infosystems",
}
NON_BUSINESS = r"chess league|cricket|marathon|award(?:s|ed)? (?:night|ceremony)|csr\b|sponsorship|divorce|accused|court case|wedding"
SUFFIX = re.compile(r"\s+[-|–]\s+(?:the economic times|economic times|mint|livemint|moneycontrol|ndtv profit|business standard|"
                    r"financial express|cnbc ?tv18|business today|the hindu businessline|times of india|news18)\s*$", re.I)
PREFIX = re.compile(r"^(?:business news|markets news|stock market news|breaking)\s*[|:–-]\s*", re.I)
TYPES = [  # first match wins
    ("non_business", NON_BUSINESS),
    ("results_date", r"results? (?:date|on (?:oct|nov|dec|jan|feb|mar|apr|may|jun|jul|aug|sep)\w*|announcement date)|to (?:declare|announce) q[1-4]|earnings call schedule|record date"),
    ("results_preview", r"preview|ahead of q[1-4]|q[1-4] (?:preview|expectations)|what to expect|investors (?:eye|watch)|earnings risk"),
    ("results", r"q[1-4] (?:results|profit|net profit|revenue)|reports? q[1-4]|posts? (?:q[1-4]|profit)|net profit (?:rises|falls|jumps|drops)"),
    ("live_updates", r"live updates|opening bell|share price live|price movement today|moves past \d+-day"),
    ("market_wrap", r"\bsensex\b|\bnifty\b(?! it)|gift nifty|stock market updates|closing bell|midday|top gainers|top losers|biggest moves"),
    ("stocks_to_watch", r"stocks? (?:to watch|in focus|in news|in spotlight)|top stocks|stocks to (?:buy|sell)|trade spotlight|shares in focus"),
    ("buy_sell_reco", r"\b(?:buy|sell|hold)\b|target price|stop[- ]loss|top picks|stocks? to add|investment ideas"),
    ("dividend", r"dividend|ex-dividend"),
    ("analyst_view", r"morgan stanley|jefferies|nomura|goldman|nuvama|kotak|motilal|clsa|bullish|bearish|upgrade|downgrade|rating"),
    ("deal_win", r"\bdeal\b|contract|partnership|partners with|selected by|wins?\b|bags?\b|acquisition|acquire"),
    ("management", r"appoint|promoted|resign|steps down|ceo|cfo|chief|head of"),
    ("legal", r"court|tribunal|lawsuit|penalt|probe|sebi order|tax demand"),
]
_TYPES = [(t, re.compile(p, re.I)) for t, p in TYPES]

# Price-move verbs FinBERT misses in market-wrap headlines (applied to the TARGET span only)
MOVE_NEG = re.compile(r"\bdrags?\b|\bdragged\b|worst hit|top los(?:er|ers)|\bslump|\bslides?\b|\btanks?\b|\bplunge|\bplummet|"
                      r"\bsinks?\b|\bsheds?\b|\bcracks?\b|\bcrash|\bfalls?\b|\bfell\b|\bdips?\b|\bdrops?\b|\bdeclines?\b|"
                      r"under pressure|\bweighs?\b|52-week low|lower circuit|\bsell-?off\b|\btumbles?\b|\bskids?\b", re.I)
MOVE_POS = re.compile(r"top gainers?|leads? gains|\brall(?:y|ies|ied)\b|\bjumps?\b|\bsurges?\b|\bsoars?\b|\bclimbs?\b|"
                      r"\bgains?\b|\brises?\b|\brose\b|52-week high|record high|upper circuit|\blifts?\b|\bzooms?\b|\bspurts?\b", re.I)
# Recommendations / analyst actions (separate feature, not "sentiment")
RECO_BUY = re.compile(r"stocks? to buy|\bbuy\b(?! or sell)|top picks?|stocks to add|\baccumulate\b|\boutperform\b|\boverweight\b", re.I)
RECO_SELL = re.compile(r"stocks? to sell|\bsell\b(?!,? or)|\breduce\b|\bunderperform\b|\bunderweight\b", re.I)
RECO_AMBIG = re.compile(r"buy,? sell or hold|buy or sell|how should you trade", re.I)
ANALYST_POS = re.compile(r"bullish|upgrade[sd]?|raises? target|target (?:price )?raised|eyes upside|sees upside", re.I)
ANALYST_NEG = re.compile(r"bearish|downgrade[sd]?|cuts? target|target (?:price )?cut|earnings risk|sees downside", re.I)
STUB = re.compile(r"^(?:[\w&\. ]{0,40})(?:opening bell updates?|share price live updates?|live updates?|in focus|stock in focus|shares? in focus)\s*$", re.I)

def move_signal(span: str) -> int:
    neg, pos = bool(MOVE_NEG.search(span)), bool(MOVE_POS.search(span))
    return 0 if neg == pos else (-1 if neg else 1)

def reco_signal(text: str) -> int:
    if RECO_AMBIG.search(text): return 0
    b, s_ = bool(RECO_BUY.search(text)), bool(RECO_SELL.search(text))
    return 0 if b == s_ else (1 if b else -1)

def analyst_signal(text: str) -> int:
    p, n = bool(ANALYST_POS.search(text)), bool(ANALYST_NEG.search(text))
    return 0 if p == n else (1 if p else -1)
_AL = {c: [re.compile(a, re.I) for a in al] for c, al in ALIASES.items()}
_NOT = {c: re.compile(p, re.I) for c, p in NOT_ENTITY.items()}

def normalise(title: str) -> str:
    t = PREFIX.sub("", (title or "").strip())
    return SUFFIX.sub("", t).strip()

def mentions(text: str) -> dict[str, tuple[int, int]]:
    out = {}
    for c, rxs in _AL.items():
        if c in _NOT and _NOT[c].search(text):
            continue
        hits = [m for rx in rxs for m in rx.finditer(text)]
        if hits:
            h = min(hits, key=lambda m: m.start()); out[c] = (h.start(), h.end())
    return out

def _is_name_list(seg: str) -> bool:
    """'HDFC Bank, TCS, YES Bank, Hindalco among …' — mostly short capitalised comma-separated items."""
    parts = [p.strip() for p in re.split(r",|\band\b|\bto\b", seg) if p.strip()]
    if len(parts) < 3: return False
    short = [p for p in parts if len(p.split()) <= 3 and p[:1].isupper()]
    return len(short) >= 0.6 * len(parts)

MARKET = re.compile(r"\bsensex\b|\bnifty\b|gift nifty|\bmarkets?\b|\bindices\b|\bcrude\b|\brupee\b|\bfii|\brbi\b", re.I)
CAP_ITEM = re.compile(r"^[A-Z][\w&\.\-]*(?: [A-Z][\w&\.\-]*){0,2}$")

def _segments(title: str) -> list[tuple[int, int]]:
    cuts = [0] + [m.end() for m in re.finditer(r";|\||:|–|—|\s-\s", title)] + [len(title) + 1]
    return [(cuts[i], cuts[i + 1] - 1) for i in range(len(cuts) - 1)]

def _comma_run(seg: str, rel: int) -> int:
    """Number of short capitalised comma/and-separated items in the run that contains the target."""
    items, pos = [], 0
    for p in re.split(r"(,\s|\sand\s|\s&\s|\sto\s)", seg):
        if not re.fullmatch(r",\s|\sand\s|\s&\s|\sto\s", p or ""): items.append((pos, p))
        pos += len(p or "")
    idx = max(i for i, (st, _) in enumerate(items) if st <= rel)
    def cap(i): return CAP_ITEM.match(items[i][1].strip().split(" among")[0].split(" shares")[0].strip()) is not None
    n, j = 1, idx - 1
    while j >= 0 and cap(j): n += 1; j -= 1
    j = idx + 1
    while j < len(items) and cap(j): n += 1; j += 1
    return n

def target_span(title: str, span: tuple[int, int]) -> tuple[str, int]:
    """Returns (span_text, n_names_in_target_run). Lists keep the WHOLE title (the predicate is elsewhere)."""
    a = span[0]
    segs = _segments(title)
    s0, s1 = next((x, y) for x, y in segs if x <= a <= y)
    seg = title[s0:s1]
    if STUB.match(seg.strip(" ;|:–—-")) and len(segs) > 1:      # "Infosys Opening Bell Updates: Global Cues Lift INFY…"
        rest = " ".join(title[x:y].strip(" ;|:–—-") for x, y in segs if (x, y) != (s0, s1)).strip()
        if len(rest.split()) >= 3: return rest, 1
    run = _comma_run(seg, a - s0)
    if run >= 2 or _is_name_list(seg):
        return title, max(run, 3 if _is_name_list(seg) else run)
    # split only when the rest of the title is about something else (market / other company): contrast exists
    others = title[:s0] + " " + title[s1:]
    contrast = bool(MARKET.search(others) or mentions(others))
    sub = [m for m in re.finditer(r"[^,]+", seg)]
    clause = next((m.group(0) for m in sub if m.start() <= a - s0 <= m.end()), seg).strip()
    if not contrast:
        # also check contrast inside the segment's other comma clauses
        rest = " ".join(m.group(0) for m in sub if not (m.start() <= a - s0 <= m.end()))
        contrast = bool(MARKET.search(rest) or mentions(rest) or re.search(r"\b[A-Z]{2,}\b", rest))
        if not contrast: return title, 1
        return clause if len(clause.split()) >= 2 else seg.strip(), 1
    return (clause if len(clause.split()) >= 2 else seg.strip(" ;|:–—-")), 1

def article_type(text: str) -> str:
    return next((t for t, rx in _TYPES if rx.search(text)), "other")

def analyse(title: str, description: str, target: str) -> dict:
    t = normalise(title); d = description or ""
    if target not in ALIASES:   # sector-level pull (e.g. SECTOR): the whole title is the target
        atype = article_type(t)
        return dict(title_norm=t, role="sector", target_span=t, in_list=False, n_tracked=len(mentions(t)),
                    article_type=atype, relevance=0.0 if atype == "non_business" else 1.0,
                    move_signal=move_signal(t), reco_signal=reco_signal(t), analyst_signal=analyst_signal(t))
    m_title = mentions(t); m_desc = mentions(d)
    if target in _NOT and (_NOT[target].search(t) or _NOT[target].search(d)) and target not in m_title:
        return dict(title_norm=t, role="wrong_entity", target_span="", in_list=False, n_tracked=len(m_title),
                    article_type=article_type(t), relevance=0.0, move_signal=0, reco_signal=0, analyst_signal=0)
    atype = article_type(t)
    if target in m_title:
        span, run = target_span(t, m_title[target])
        n = max(len(m_title), run)
        role = "list_member" if n >= 3 else "co_mention" if n == 2 else "primary"
        in_list = n >= 3
    elif target in m_desc:
        span, in_list, role = t, False, "peripheral"
    else:
        span, in_list, role = t, False, "wrong_entity"
    rel = {"primary": 1.0, "co_mention": 0.5, "list_member": 0.2, "peripheral": 0.1, "wrong_entity": 0.0}[role]
    if atype == "non_business": rel = 0.0
    # list headlines: a buy-list / gainers-list applies to every listed name; otherwise signals come from the target span
    if role == "list_member" and target in m_title:   # list: use only the segment holding the list, not the market clause
        a0 = m_title[target][0]
        sig_text = next(t[x:y] for x, y in _segments(t) if x <= a0 <= y)
    else:
        sig_text = span
    return dict(title_norm=t, role=role, target_span=span, in_list=in_list, n_tracked=len(m_title), article_type=atype, relevance=rel,
                move_signal=move_signal(sig_text) if atype in ("market_wrap", "live_updates", "stocks_to_watch", "other") or role == "primary" else 0,
                reco_signal=reco_signal(t), analyst_signal=analyst_signal(t))


# ───────────── model input: typed entity markers + contrast-word tags ─────────────
# Input to the sentiment model is a pair:  text_a = target name,  text_b = marked headline
#   <t> … </t>  every mention of the target          <o> … </o>  every other tracked / global IT peer
#   <c> … </c>  concessive connective (main clause decides, wherever it sits)
#   <a> … </a>  adversative connective (clause after it usually carries the weight)
# Markers use existing wordpieces (<, t, >, /) so no new embeddings are needed.
PEERS = {  # global peers: marked as <o>, never a target, not counted in roles
    "ACN":   [r"\baccenture\b"],
    "CTSH":  [r"\bcognizant\b"],
    "CAP":   [r"\bcapgemini\b"],
    "LTIM":  [r"\blti ?mindtree\b", r"\bltim\b", r"\bmindtree\b"],
    "MPHASIS": [r"\bmphasis\b"],
}
TARGET_NAME = {"TCS": "tcs", "INFY": "infosys", "WIPRO": "wipro", "HCLTECH": "hcltech", "TECHM": "tech mahindra"}
CONCESSIVE = re.compile(r"\b(?:despite|in spite of|even as|even though|although|though|notwithstanding)\b", re.I)
ADVERSATIVE = re.compile(r"\b(?:but(?! for\b)|however)\b|(?<=[,;] )yet\b", re.I)  # 'but for' = except; bare 'yet' is usually temporal
_ALL_AL = {**_AL, **{c: [re.compile(a, re.I) for a in al] for c, al in PEERS.items()}}

def all_mentions(text: str) -> list[tuple[int, int, str]]:
    """Every (start, end, company) occurrence of tracked companies and global peers, non-overlapping."""
    hits = []
    for c, rxs in _ALL_AL.items():
        if c in _NOT and _NOT[c].search(text):
            continue
        hits += [(m.start(), m.end(), c) for rx in rxs for m in rx.finditer(text)]
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    out, last = [], -1
    for s, e, c in hits:
        if s >= last: out.append((s, e, c)); last = e
    return out

def model_input(title: str, target: str, entity_markers: bool = True, other_markers: bool = True,
                contrast_tags: bool = True) -> tuple[str, str]:
    """(text_a, text_b) for the sentence-pair sentiment model. Flags exist for the input-format ablation."""
    t = normalise(title)
    spans = []
    if entity_markers or other_markers:
        for s, e, c in all_mentions(t):
            if c == target and entity_markers: spans.append((s, e, "t"))
            elif c != target and other_markers: spans.append((s, e, "o"))
    if contrast_tags:
        spans += [(m.start(), m.end(), "c") for m in CONCESSIVE.finditer(t)]
        spans += [(m.start(), m.end(), "a") for m in ADVERSATIVE.finditer(t)]
    for s, e, k in sorted(spans, reverse=True):          # right-to-left keeps offsets valid
        t = f"{t[:s]}<{k}> {t[s:e]} </{k}>{t[e:]}"
    return TARGET_NAME.get(target, "the company"), t
