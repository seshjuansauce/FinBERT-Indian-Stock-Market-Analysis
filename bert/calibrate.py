"""Temperature scaling for the sentiment models (A, B, C): fix over-confident probabilities without changing any label.

  python bert/calibrate.py        # after bert/finetune_sentiment.py; ~2 min on the Mac GPU

    p = softmax(z / T),   T* = argmin_T  NLL(dev split)

The dev split is the same 15% of our labelled rows that finetune_sentiment.py held out for epoch selection, so the
2024 test set is never used to fit T. argmax(z / T) = argmax(z): accuracy, F1, precision and recall are unchanged;
only the probabilities (and therefore the sentiment-strength feature |P(pos) − P(neg)|) move.

Out   results/sentiment/calibration.json   T per model + dev/test NLL, ECE, Brier, confidence, saturation before/after
      results/sentiment/calibration.md     the same as a table
      results/sentiment/preds_logits_<M>_test.csv   raw test logits (re-calibrate later without the GPU)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize_scalar
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "bert"))
import finetune_sentiment as ft  # noqa: E402  (reuses encode / split / batching so inputs match training exactly)

OUT = ROOT / "results" / "sentiment"
MODELS = {"A": ("ProsusAI/finbert", "full"), "B": (str(ft.MODELS / "finbert-b-sentiment"), None),
          "C": (str(ft.MODELS / "finbert-in-sentiment"), None)}

@torch.no_grad()
def logits(model, tok, rows, dev, max_len=128):
    model.eval(); out = []
    for enc, _ in ft.batches(rows, tok, 64, max_len, shuffle=False):
        out.append(model(**{k: v.to(dev) for k, v in enc.items()}).logits.float().cpu().numpy())
    return np.concatenate(out)

def softmax(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def nll(z, y, T): return float(-np.mean(np.log(softmax(z, T)[np.arange(len(y)), y] + 1e-12)))

def ece(p, y, bins=5):
    conf, ok = p.max(1), (p.argmax(1) == y).astype(float); edges = np.linspace(1 / 3, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf <= hi) if hi == 1 else (conf < hi))
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)

def report(z, y, T, ip, ineg):
    p = softmax(z, T); s = p[:, ip] - p[:, ineg]; onehot = np.eye(3)[y]
    return {"nll": nll(z, y, T), "ece": ece(p, y), "brier": float(np.mean(np.sum((p - onehot) ** 2, 1))),
            "accuracy": float((p.argmax(1) == y).mean()), "mean_conf": float(p.max(1).mean()),
            "share_conf_gt_0.95": float((p.max(1) > 0.95).mean()), "share_abs_s_gt_0.9": float((np.abs(s) > 0.9).mean()),
            "share_abs_s_0.1_to_0.9": float(((np.abs(s) >= 0.1) & (np.abs(s) <= 0.9)).mean())}

def main():
    dev = ft.device(); print("device:", dev)
    ours = pd.read_csv(ft.LAB / "train_final.csv"); test = pd.read_csv(ft.LAB / "test_final.csv")
    _, dv = ft.split_ours(ours)                                    # identical dev split to training (seed 0)
    label_source = "user" if (test["label_source"] == "user").all() else "claude_provisional"
    out, lines = {"test_label_source": label_source, "n_dev": len(dv), "n_test": len(test)}, []
    for name, (path, fmt) in MODELS.items():
        if not path.startswith("ProsusAI") and not Path(path).exists():
            print(f"skip {name}: {path} missing"); continue
        tok = AutoTokenizer.from_pretrained(path); mdl = AutoModelForSequenceClassification.from_pretrained(path).to(dev)
        if fmt is None: fmt = json.loads((Path(path) / "input_format.json").read_text())["format"]
        labels = [mdl.config.id2label[i].lower() for i in range(3)]; L2I = {l: i for i, l in enumerate(labels)}
        ip, ineg = L2I["positive"], L2I["negative"]
        mk = lambda df: [(*ft.encode(r.title, r.target, fmt), L2I[r.label]) for r in df.itertuples()]
        rd, rt = mk(dv), mk(test)
        zd, zt = logits(mdl, tok, rd, dev), logits(mdl, tok, rt, dev)
        yd, yt = np.array([r[2] for r in rd]), np.array([r[2] for r in rt])
        T = float(np.exp(minimize_scalar(lambda lt: nll(zd, yd, np.exp(lt)), bounds=(np.log(0.05), np.log(50)), method="bounded").x))
        res = {"T": T, "format": fmt, "labels": labels,
               "dev": {"raw": report(zd, yd, 1.0, ip, ineg), "calibrated": report(zd, yd, T, ip, ineg)},
               "test": {"raw": report(zt, yt, 1.0, ip, ineg), "calibrated": report(zt, yt, T, ip, ineg)}}
        out[name] = res
        pd.DataFrame(zt, columns=[f"z_{l}" for l in labels]).assign(id=test["id"].values, y=[labels[i] for i in yt]) \
          .to_csv(OUT / f"preds_logits_{name}_test.csv", index=False)
        r, c = res["test"]["raw"], res["test"]["calibrated"]
        lines.append(f"| {name} | {T:.2f} | {r['nll']:.3f} → {c['nll']:.3f} | {r['ece']:.3f} → {c['ece']:.3f} | {r['brier']:.3f} → {c['brier']:.3f} "
                     f"| {r['mean_conf']:.3f} → {c['mean_conf']:.3f} | {r['accuracy']:.3f} | {r['share_abs_s_gt_0.9']:.2f} → {c['share_abs_s_gt_0.9']:.2f} |")
        print(f"{name}: T = {T:.2f}  test ECE {r['ece']:.3f} → {c['ece']:.3f}  NLL {r['nll']:.3f} → {c['nll']:.3f}", flush=True)
        del mdl
        if dev.type == "mps": torch.mps.empty_cache()
    (OUT / "calibration.json").write_text(json.dumps(out, indent=1))
    md = [f"# Temperature scaling (T fitted on {out['n_dev']} dev headlines; test = {out['n_test']} headlines, labels: {label_source})\n",
          "| model | T | test NLL | test ECE | test Brier | mean confidence | accuracy (unchanged) | share \\|s\\| > 0.9 |",
          "|---|---|---|---|---|---|---|---|"] + lines
    (OUT / "calibration.md").write_text("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
