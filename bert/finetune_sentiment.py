"""Fine-tune the sentiment head and run the A / B / C comparison + input-format ablation (Mac MPS / CUDA / CPU).

  python bert/build_label_sets.py               # first: assemble train_final.csv / test_final.csv
  python bert/finetune_sentiment.py --quick     # A, B, C, one seed (~10 min)
  caffeinate -i python bert/finetune_sentiment.py          # everything, 3 seeds (~1 h)

Runs
  A_full / A_crop   ProsusAI FinBERT as shipped, no training (full headline / entity crop)
  B                 ProsusAI encoder + head, fine-tuned on PhraseBank + our labels, input pair_tocm
  C                 DAPT encoder (models/finbert-in-dapt) + ProsusAI head as warm start, same data + recipe as B
  C_full, C_crop, C_pair_t, C_pair_to   C with other input formats (ablation)

Input formats
  full       headline as is                          crop      entity.py target span
  pair_t     ("tcs", headline with <t> target </t>)  pair_to   + <o> peers </o>
  pair_tocm  + <c>/<a> contrast words + <m> market indices </m>   (default)

Data   train: PhraseBank AllAgree (2,264, downloaded once) + data/labels/train_final.csv (85%, upsampled x3,
       target-name swap augmentation p=0.5); dev: other 15% of our rows (epoch selection only);
       test: data/labels/test_final.csv (2024, never trained on) + bert/eval/minimal_pairs.csv
Out    results/sentiment/summary.md, summary.json, preds_<run>_s<seed>.csv
       models/finbert-in-sentiment (C, seed 0), models/finbert-b-sentiment (B, seed 0) for corpus scoring
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, re, sys, time, zipfile
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, BertForMaskedLM, BertForSequenceClassification, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "news"))
from entity import analyse, model_input  # noqa: E402

LAB, OUT, MODELS = ROOT / "data" / "labels", ROOT / "results" / "sentiment", ROOT / "models"
BASE, DAPT = "ProsusAI/finbert", MODELS / "finbert-in-dapt"
FORMATS = {
    "full":      dict(pair=False, crop=False),
    "crop":      dict(pair=False, crop=True),
    "pair_t":    dict(pair=True, t=True, o=False, c=False, m=False),
    "pair_to":   dict(pair=True, t=True, o=True, c=False, m=False),
    "pair_tocm": dict(pair=True, t=True, o=True, c=True, m=True),
}
RUNS = {  # name: (encoder, train?, format)
    "A_full": ("prosus", False, "full"), "A_crop": ("prosus", False, "crop"),
    "B": ("prosus", True, "pair_tocm"), "C": ("dapt", True, "pair_tocm"),
    "C_full": ("dapt", True, "full"), "C_crop": ("dapt", True, "crop"),
    "C_pair_t": ("dapt", True, "pair_t"), "C_pair_to": ("dapt", True, "pair_to"),
}
SWAP_NAMES = ["tcs", "infosys", "wipro", "hcltech", "tech mahindra", "accenture", "cognizant", "ltimindtree"]

# ───────────────────────────── data ─────────────────────────────
def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def phrasebank() -> pd.DataFrame | None:
    """Financial PhraseBank, sentences with 100% annotator agreement. Cached to data/phrasebank_allagree.csv."""
    cache = ROOT / "data" / "phrasebank_allagree.csv"
    if cache.exists(): return pd.read_csv(cache)
    from huggingface_hub import hf_hub_download
    errs = []
    try:
        p = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
        with zipfile.ZipFile(p) as z:
            name = next(n for n in z.namelist() if n.endswith("Sentences_AllAgree.txt"))
            rows = [l.rsplit("@", 1) for l in z.read(name).decode("latin-1").splitlines() if "@" in l]
        df = pd.DataFrame(rows, columns=["title", "label"])
    except Exception as e:  # noqa: BLE001
        errs.append(e)
        try:
            p = hf_hub_download("takala/financial_phrasebank", "sentences_allagree/train/0000.parquet",
                                repo_type="dataset", revision="refs/convert/parquet")
            df = pd.read_parquet(p).rename(columns={"sentence": "title"})
            df["label"] = df["label"].map({0: "negative", 1: "neutral", 2: "positive"})
        except Exception as e2:  # noqa: BLE001
            print(f"PhraseBank unavailable ({errs[0]!r}; {e2!r}) — training on our labels only"); return None
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["positive", "negative", "neutral"])]
    cache.parent.mkdir(parents=True, exist_ok=True); df.to_csv(cache, index=False)
    return df

def encode(title: str, target: str | None, fmt: str) -> tuple[str, str | None]:
    """-> (text, text_pair). PhraseBank rows have target=None (generic company, no entity markers)."""
    f = FORMATS[fmt]
    if not f["pair"]:
        if f.get("crop") and target:
            span = analyse(title, "", target)["target_span"]
            return (span or title), None
        return title, None
    if target is None:
        a, b = model_input(title, "__none__", entity_markers=False, other_markers=False, contrast_tags=f["c"], market_tags=f["m"])
        return "the company", b
    a, b = model_input(title, target, entity_markers=f["t"], other_markers=f["o"], contrast_tags=f["c"], market_tags=f["m"])
    return a, b

def swap_target(a: str, b: str | None, rng: random.Random) -> tuple[str, str | None]:
    """Name-swap augmentation: the model must follow the <t> marker, not the word 'tcs'."""
    if b is None or "<t>" not in b or rng.random() >= 0.5: return a, b
    present = {n for n in SWAP_NAMES if n in b.lower()}
    name = rng.choice([n for n in SWAP_NAMES if n not in present] or SWAP_NAMES)
    return name, re.sub(r"<t> .*? </t>", f"<t> {name} </t>", b)

def split_ours(ours: pd.DataFrame, dev_frac=0.15, seed=0):
    rng = np.random.default_rng(seed); dev = []
    for _, g in ours.groupby("label"):
        dev += list(rng.choice(g.index, max(1, round(len(g) * dev_frac)), replace=False))
    return ours.drop(index=dev), ours.loc[dev]

# ───────────────────────────── model ─────────────────────────────
def load_tokenizer(): return AutoTokenizer.from_pretrained(BASE)

def load_model(encoder: str):
    m = BertForSequenceClassification.from_pretrained(BASE)            # ProsusAI encoder + pooler + 3-way head
    if encoder == "dapt":
        if not DAPT.exists(): raise SystemExit(f"{DAPT} missing — run bert/dapt_mlm.py first")
        enc = BertForMaskedLM.from_pretrained(DAPT).bert.state_dict()
        missing, unexpected = m.bert.load_state_dict(enc, strict=False)
        assert all("pooler" in k for k in missing) and not unexpected, (missing, unexpected)
    return m

def batches(rows, tok, bs, max_len, shuffle, rng=None):
    idx = list(range(len(rows)))
    if shuffle: rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [rows[j] for j in idx[i:i + bs]]
        a = [r[0] for r in chunk]; b = [r[1] for r in chunk]
        enc = tok(a, b if all(x is not None for x in b) else None, truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        yield enc, torch.tensor([r[2] for r in chunk])

@torch.no_grad()
def predict(model, rows, tok, dev, max_len, bs=64):
    model.eval(); out = []
    for enc, _ in batches(rows, tok, bs, max_len, shuffle=False):
        out.append(model(**{k: v.to(dev) for k, v in enc.items()}).logits.float().softmax(-1).cpu())
    model.train()
    return torch.cat(out).numpy() if out else np.zeros((0, 3))

# ───────────────────────────── metrics ─────────────────────────────
def metrics(y: np.ndarray, p: np.ndarray, labels: list[str]) -> dict:
    k = len(labels); cm = np.zeros((k, k), int)
    for a, b in zip(y, p): cm[a, b] += 1
    f1 = []
    for c in range(k):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    pos, neg = labels.index("positive"), labels.index("negative")
    return {"n": int(len(y)), "acc": float((y == p).mean()) if len(y) else float("nan"), "macro_f1": float(np.mean(f1)),
            **{f"f1_{labels[c]}": float(f1[c]) for c in range(k)},
            "sign_flips": int(cm[pos, neg] + cm[neg, pos]), "confusion": cm.tolist()}

def macro_f1(y, p, k=3):
    f = []
    for c in range(k):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

# ───────────────────────────── one run ─────────────────────────────
def run_one(name, seed, a, tok, labels, data, dev):
    encoder, train, fmt = RUNS[name]
    L2I = {l: i for i, l in enumerate(labels)}
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rng = random.Random(seed)
    mk = lambda df, tgt=True: [(*encode(r.title, r.target if tgt else None, fmt), L2I[r.label]) for r in df.itertuples()]
    dev_rows, test_rows, mp_rows = mk(data["dev"]), mk(data["test"]), mk(data["pairs"])
    model = load_model(encoder).to(dev); t0 = time.time(); best = {"epoch": 0, "dev_macro_f1": None}
    if train:
        ours = mk(data["train"]) * a.upsample
        pb = mk(data["phrasebank"], tgt=False) if data["phrasebank"] is not None else []
        rows = ours + pb
        counts = np.bincount([r[2] for r in rows], minlength=3)
        w = torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))) ** 0.5, dtype=torch.float, device=dev)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        steps = math.ceil(len(rows) / a.batch_size) * a.epochs
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        best_state, best_f1 = None, -1.0
        for ep in range(1, a.epochs + 1):
            ep_rows = [(*swap_target(x, y_, rng), lab) for x, y_, lab in rows] if FORMATS[fmt]["pair"] else rows
            tot, n = 0.0, 0
            for enc, y in batches(ep_rows, tok, a.batch_size, a.max_len, shuffle=True, rng=rng):
                loss = lossf(model(**{k: v.to(dev) for k, v in enc.items()}).logits, y.to(dev))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True); tot += loss.item() * len(y); n += len(y)
            yd = np.array([r[2] for r in dev_rows]); f1 = macro_f1(yd, predict(model, dev_rows, tok, dev, a.max_len).argmax(1))
            print(f"    {name} s{seed} epoch {ep}: train loss {tot / n:.3f}  dev macro-F1 {f1:.3f}", flush=True)
            if f1 > best_f1:
                best_f1, best = f1, {"epoch": ep, "dev_macro_f1": f1}
                best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
    res = {"run": name, "seed": seed, "encoder": encoder, "trained": train, "format": fmt, **best,
           "train_rows": (len(data["train"]) * a.upsample + (0 if data["phrasebank"] is None else len(data["phrasebank"]))) if train else 0,
           "minutes": round((time.time() - t0) / 60, 2)}
    for split, rows_, df in [("dev", dev_rows, data["dev"]), ("test", test_rows, data["test"]), ("pairs", mp_rows, data["pairs"])]:
        pr = predict(model, rows_, tok, dev, a.max_len); y = np.array([r[2] for r in rows_]); p = pr.argmax(1)
        res[split] = metrics(y, p, labels)
        if split == "test":
            for role, g in df.reset_index(drop=True).groupby("role"):
                res["test"][f"macro_f1_{role}"] = macro_f1(y[g.index], p[g.index]); res["test"][f"n_{role}"] = len(g)
            pd.DataFrame({"id": df["id"].values, "role": df["role"].values, "y": [labels[i] for i in y], "pred": [labels[i] for i in p],
                          **{f"p_{l}": pr[:, i] for i, l in enumerate(labels)}}).to_csv(OUT / f"preds_{name}_s{seed}.csv", index=False)
            res["_test_pred"] = p.tolist(); res["_test_y"] = y.tolist()
        if split == "pairs":
            ok = (y == p); pid = df["pair_id"].values
            res["pairs"]["pair_acc"] = float(np.mean([ok[pid == i].all() for i in np.unique(pid)]))
            res["pairs"]["by_test"] = {t: float(ok[df["test"].values == t].mean()) for t in df["test"].unique()}
    if train and seed == a.seeds[0] and name in ("B", "C"):
        dst = MODELS / ("finbert-in-sentiment" if name == "C" else "finbert-b-sentiment")
        model.save_pretrained(dst); tok.save_pretrained(dst)
        (dst / "input_format.json").write_text(json.dumps({"format": fmt, **FORMATS[fmt], "labels": labels}, indent=1))
    del model
    if dev.type == "mps": torch.mps.empty_cache()
    return res

# ───────────────────────────── summary ─────────────────────────────
def bootstrap_diff(results, r1, r2, n_boot=2000, seed=0):
    """Paired bootstrap over test headlines of mean-over-seeds macro-F1(r1) − macro-F1(r2)."""
    s1 = [r for r in results if r["run"] == r1]; s2 = [r for r in results if r["run"] == r2]
    if not s1 or not s2: return None
    y = np.array(s1[0]["_test_y"]); P1 = [np.array(r["_test_pred"]) for r in s1]; P2 = [np.array(r["_test_pred"]) for r in s2]
    rng = np.random.default_rng(seed); n = len(y); d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(np.mean([macro_f1(y[i], p[i]) for p in P1]) - np.mean([macro_f1(y[i], p[i]) for p in P2]))
    d = np.array(d); point = np.mean([macro_f1(y, p) for p in P1]) - np.mean([macro_f1(y, p) for p in P2])
    return {"diff": float(point), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_le_0": float((d <= 0).mean())}

def md_table(df: pd.DataFrame) -> str:   # no `tabulate` dependency
    cols = [df.index.name or ""] + list(df.columns)
    rows = [[str(i)] + [("" if pd.isna(v) else f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r] for i, r in zip(df.index, df.values)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])

def summarise(results, label_source):
    df = pd.DataFrame([{"run": r["run"], "seed": r["seed"], "format": r["format"], "test_macro_f1": r["test"]["macro_f1"],
                        "test_acc": r["test"]["acc"], "sign_flips": r["test"]["sign_flips"],
                        "f1_co_mention": r["test"].get("macro_f1_co_mention", np.nan), "f1_list": r["test"].get("macro_f1_list_member", np.nan),
                        "pairs_acc": r["pairs"]["acc"], "pair_acc": r["pairs"]["pair_acc"], "dev_f1": r["dev"]["macro_f1"]} for r in results])
    g = df.groupby("run", sort=False)
    agg = g.agg(format=("format", "first"), seeds=("seed", "count"),
                test_macro_f1=("test_macro_f1", "mean"), sd=("test_macro_f1", "std"), test_acc=("test_acc", "mean"),
                sign_flips=("sign_flips", "mean"), f1_co_mention=("f1_co_mention", "mean"), f1_list=("f1_list", "mean"),
                minimal_pairs=("pair_acc", "mean"), dev_f1=("dev_f1", "mean")).round(3)
    comps = {f"{a} - {b}": bootstrap_diff(results, a, b) for a, b in
             [("B", "A_full"), ("C", "B"), ("C", "A_full"), ("C", "C_full"), ("C", "C_crop"), ("C_pair_to", "C_pair_t"), ("C", "C_pair_to")]}
    comps = {k: v for k, v in comps.items() if v}
    lines = [f"# Sentiment model comparison\n", f"Test set: {label_source}\n", md_table(agg), "\n## Paired bootstrap, test macro-F1 difference (mean over seeds)\n",
             "| comparison | diff | 95% CI | P(diff ≤ 0) |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['p_le_0']:.3f} |" for k, v in comps.items()]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps({"label_source": label_source, "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
                                                  "comparisons": comps}, indent=1))
    print("\n" + "\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS)); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--quick", action="store_true", help="A_full, B, C with one seed")
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--upsample", type=int, default=3); ap.add_argument("--no-phrasebank", action="store_true")
    a = ap.parse_args()
    if a.quick: a.runs, a.seeds = "A_full,B,C", "0"
    a.runs = [r for r in a.runs.split(",") if r]; a.seeds = [int(s) for s in a.seeds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True); dev = device(); print("device:", dev)
    tok = load_tokenizer()
    labels = [BertForSequenceClassification.from_pretrained(BASE).config.id2label[i] for i in range(3)]
    labels = [l.lower() for l in labels]
    ours = pd.read_csv(LAB / "train_final.csv"); test = pd.read_csv(LAB / "test_final.csv")
    pairs = pd.read_csv(ROOT / "bert" / "eval" / "minimal_pairs.csv")
    assert not set(ours["id"]) & set(test["id"]), "train/test overlap"
    tr, dv = split_ours(ours)
    data = {"train": tr, "dev": dv, "test": test, "pairs": pairs, "phrasebank": None if a.no_phrasebank else phrasebank()}
    label_source = ("YOUR blind labels" if (test["label_source"] == "user").all() else
                    f"PROVISIONAL — Claude's hidden drafts for {(test['label_source'] != 'user').sum()} of {len(test)} rows")
    print(f"labels {labels} | train {len(tr)} (+PhraseBank {0 if data['phrasebank'] is None else len(data['phrasebank'])}) "
          f"dev {len(dv)} test {len(test)} [{label_source}] pairs {len(pairs)}")
    results = []
    for name in a.runs:
        for seed in (a.seeds if RUNS[name][1] else a.seeds[:1]):
            print(f"\n== {name} seed {seed} ({RUNS[name][0]} encoder, {RUNS[name][2]})", flush=True)
            r = run_one(name, seed, a, tok, labels, data, dev); results.append(r)
            print(f"   test macro-F1 {r['test']['macro_f1']:.3f}  acc {r['test']['acc']:.3f}  sign flips {r['test']['sign_flips']}  "
                  f"minimal pairs {r['pairs']['pair_acc']:.2f}  ({r['minutes']} min)", flush=True)
    summarise(results, label_source)

if __name__ == "__main__":
    main()
