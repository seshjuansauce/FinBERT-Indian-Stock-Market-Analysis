"""FinBERT-IN-call: fine-tune the news model (FinBERT-IN) on labelled earnings-call sentences (Mac MPS / CUDA / CPU).

  python bert/finetune_call.py              # ~5 min on Apple-silicon GPU
  python bert/finetune_call.py --seeds 0,1,2

Labels  data/labels/calls_validate.csv  (your_label if filled, else draft_label)  → 85% train / 15% dev
        data/labels/calls_test_blind.csv (your_label) — falls back to Claude's hidden drafts, marked PROVISIONAL
Runs    news_zero_shot  FinBERT-IN (news) as is, call inputs — what the news model already knows about calls
        call            fine-tuned from FinBERT-IN with call inputs: (company + speaker role + section, sentence with
                        <t>/<o>/<c>/<a>/<m> markers + <h> hedging tags)
        call_noctx      same, but text_a is only the company (no role/section) and no <h> tags — ablation of what is
                        call-specific
Out     models/finbert-in-call (+ input_format.json), results/calls/summary.md|json, temperature in
        results/sentiment/calibration.json under key "CALL"
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch
from scipy.optimize import minimize_scalar
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bert")); sys.path.insert(0, str(ROOT / "news"))
import finetune_sentiment as ft  # noqa: E402
from finbert_vol.calls import COMPANY, call_input  # noqa: E402

LAB, OUT, BASE = ROOT / "data" / "labels", ROOT / "results" / "calls", ROOT / "models" / "finbert-in-sentiment"
DST = ROOT / "models" / "finbert-in-call"
FMT = {"format": "call", "pair": True, "t": True, "o": True, "c": True, "m": True, "h": True}
RUNS = {"news_zero_shot": (False, "full_ctx"), "call": (True, "full_ctx"), "call_noctx": (True, "noctx")}


def norm(x):
    x = str(x).strip().lower(); return x if x in ("positive", "negative", "neutral") else None


def load_labels():
    tr = pd.read_csv(LAB / "calls_validate.csv")
    tr["label"] = tr["your_label"].map(norm).fillna(tr["draft_label"])
    te = pd.read_csv(LAB / "calls_test_blind.csv", encoding="utf-8-sig")
    hidden = pd.read_csv(LAB / ".hidden" / "calls_test_claude_hidden.csv")
    te = te.merge(hidden[["sid", "draft_label"]], on="sid", how="left")
    te["label"] = te["your_label"].map(norm)
    src = "YOUR blind labels" if te["label"].notna().all() else f"PROVISIONAL — Claude's drafts for {te['label'].isna().sum()} of {len(te)}"
    te["label"] = te["label"].fillna(te["draft_label"])
    return tr, te, src


def encode(df, variant):
    out = []
    for r in df.itertuples():
        a, b = call_input(r.sentence, r.company, r.context, {**FMT, "h": variant == "full_ctx"})
        if variant == "noctx": a = COMPANY.get(r.company, "the company")
        out.append((a, b))
    return out


def split(df, frac=0.15, seed=0):
    rng = np.random.default_rng(seed); dev = []
    for _, g in df.groupby("label"): dev += list(rng.choice(g.index, max(1, round(len(g) * frac)), replace=False))
    return df.drop(index=dev), df.loc[dev]


@torch.no_grad()
def logits(model, tok, pairs, dev, bs=64):
    model.eval(); out = []
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        enc = tok([p[0] for p in ch], [p[1] for p in ch], truncation=True, max_length=160, padding=True, return_tensors="pt").to(dev)
        out.append(model(**enc).logits.float().cpu().numpy())
    model.train(); return np.concatenate(out)


def run(name, seed, a, data, dev):
    train_flag, variant = RUNS[name]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForSequenceClassification.from_pretrained(BASE).to(dev)
    labels = [model.config.id2label[i].lower() for i in range(3)]; L2I = {l: i for i, l in enumerate(labels)}
    P = {k: encode(v, variant) for k, v in data.items()}; Y = {k: np.array([L2I[x] for x in v["label"]]) for k, v in data.items()}
    best = {"epoch": 0}
    if train_flag:
        rows = [(*p, y) for p, y in zip(P["train"], Y["train"])]
        counts = np.bincount([r[2] for r in rows], minlength=3)
        lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))) ** 0.5, dtype=torch.float, device=dev))
        steps = math.ceil(len(rows) / a.batch_size) * a.epochs
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        best_f1, best_state = -1, None
        for ep in range(1, a.epochs + 1):
            rng.shuffle(rows); tot = 0.0
            for i in range(0, len(rows), a.batch_size):
                ch = rows[i:i + a.batch_size]
                enc = tok([r[0] for r in ch], [r[1] for r in ch], truncation=True, max_length=160, padding=True, return_tensors="pt").to(dev)
                loss = lossf(model(**enc).logits, torch.tensor([r[2] for r in ch], device=dev))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
                tot += loss.item() * len(ch)
            f1 = ft.macro_f1(Y["dev"], logits(model, tok, P["dev"], dev).argmax(1))
            print(f"    {name} s{seed} epoch {ep}: loss {tot / len(rows):.3f}  dev macro-F1 {f1:.3f}", flush=True)
            if f1 > best_f1: best_f1, best = f1, {"epoch": ep, "dev_macro_f1": f1}; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
    zd, zt = logits(model, tok, P["dev"], dev), logits(model, tok, P["test"], dev)
    nll = lambda z, y, T: float(-np.mean(np.log(ft_softmax(z / T)[np.arange(len(y)), y] + 1e-12)))
    T = float(np.exp(minimize_scalar(lambda lt: nll(zd, Y["dev"], np.exp(lt)), bounds=(np.log(.05), np.log(50)), method="bounded").x))
    p = zt.argmax(1); res = {"run": name, "seed": seed, **best, "T": T, "test": ft.metrics(Y["test"], p, labels)}
    for ctx, g in data["test"].reset_index(drop=True).groupby("context"):
        res["test"][f"macro_f1_{ctx.replace(' ', '_')}"] = ft.macro_f1(Y["test"][g.index], p[g.index])
    res["_pred"], res["_y"] = p.tolist(), Y["test"].tolist()
    if name == "call" and seed == a.seeds[0]:
        model.save_pretrained(DST); tok.save_pretrained(DST); (DST / "input_format.json").write_text(json.dumps({**FMT, "labels": labels}, indent=1))
        cal_path = ROOT / "results" / "sentiment" / "calibration.json"; cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
        cal["CALL"] = {"T": T, "format": "call", "labels": labels, "fitted_on": f"{len(data['dev'])} dev call sentences"}
        cal_path.parent.mkdir(parents=True, exist_ok=True); cal_path.write_text(json.dumps(cal, indent=1))
    del model
    if dev.type == "mps": torch.mps.empty_cache()
    return res


def ft_softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0"); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args(); a.seeds = [int(s) for s in a.seeds.split(",")]
    if not BASE.exists(): raise SystemExit(f"{BASE} missing — run bert/finetune_sentiment.py first")
    OUT.mkdir(parents=True, exist_ok=True); dev = ft.device(); print("device:", dev)
    tr, te, src = load_labels(); trn, dv = split(tr)
    data = {"train": trn, "dev": dv, "test": te}
    print(f"train {len(trn)} dev {len(dv)} test {len(te)} [{src}]  labels {tr.label.value_counts().to_dict()}")
    results = []
    for name in RUNS:
        for seed in (a.seeds if RUNS[name][0] else a.seeds[:1]):
            print(f"\n== {name} seed {seed}", flush=True); t0 = time.time()
            r = run(name, seed, a, data, dev); results.append(r)
            print(f"   test macro-F1 {r['test']['macro_f1']:.3f} acc {r['test']['acc']:.3f} sign flips {r['test']['sign_flips']}  T {r['T']:.2f}  ({(time.time()-t0)/60:.1f} min)")
    df = pd.DataFrame([{"run": r["run"], "seed": r["seed"], "macro_f1": r["test"]["macro_f1"], "acc": r["test"]["acc"], "sign_flips": r["test"]["sign_flips"],
                        **{k: v for k, v in r["test"].items() if k.startswith("macro_f1_")}, "T": r["T"]} for r in results])
    agg = df.groupby("run", sort=False).mean(numeric_only=True).round(3)
    comps = {}
    for x, y in [("call", "news_zero_shot"), ("call", "call_noctx")]:
        A = [np.array(r["_pred"]) for r in results if r["run"] == x]; B = [np.array(r["_pred"]) for r in results if r["run"] == y]
        if A and B:
            yv = np.array(results[0]["_y"]); rng = np.random.default_rng(0); d = []
            for _ in range(2000):
                i = rng.integers(0, len(yv), len(yv)); d.append(np.mean([ft.macro_f1(yv[i], p[i]) for p in A]) - np.mean([ft.macro_f1(yv[i], p[i]) for p in B]))
            comps[f"{x} - {y}"] = {"diff": float(np.mean([ft.macro_f1(yv, p) for p in A]) - np.mean([ft.macro_f1(yv, p) for p in B])),
                                   "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    md = [f"# Earnings-call sentiment model\n\nTest: {len(te)} sentences from 2024–26 calls ({src}).\n", ft.md_table(agg), "\n## Paired bootstrap, macro-F1\n",
          "| comparison | diff | 95% CI |", "|---|---|---|"] + [f"| {k} | {v['diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] |" for k, v in comps.items()]
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    (OUT / "summary.json").write_text(json.dumps({"label_source": src, "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results], "comparisons": comps}, indent=1))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
