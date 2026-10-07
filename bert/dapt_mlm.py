"""Domain-adaptive pretraining (MLM) of FinBERT on Indian financial news — runs on a Mac (Apple MPS).

  python bert/dapt_mlm.py --smoke            # ~5 min: 200 steps, proves everything works, prints ETA for full run
  caffeinate -i python bert/dapt_mlm.py      # full run (1 epoch over data/mlm/train.txt)
  caffeinate -i python bert/dapt_mlm.py --resume   # continue after a stop / crash

Model : bert-base-uncased MLM head  +  ProsusAI/finbert encoder (FinBERT ships without an MLM head)
Mask  : 15% whole-word masking (80% [MASK] / 10% random / 10% keep), max_len 128
Opt   : AdamW lr 5e-5, wd 0.01, 6% warmup, linear decay
Eval  : perplexity on a fixed 2,000-line 2024 subset (same masks before/after) + fill-mask probes
Out   : models/finbert-in-dapt/  (final)   models/finbert-in-dapt/ckpt/ (resume state)   .../log.jsonl
"""
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForMaskedLM, BertModel, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data" / "mlm", ROOT / "models" / "finbert-in-dapt"
PROBES = [
    "TCS shares [MASK] 6% after Q3 results beat street estimates.",
    "Infosys [MASK] its FY25 revenue guidance to 4.5-5%.",
    "Wipro reported a net [MASK] of Rs 3,000 crore for the quarter.",
    "The Nifty IT index fell as brokerages [MASK] the stock to sell.",
    "HCLTech's deal wins rose, but attrition [MASK] to 14%.",
]

def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def read_lines(p: Path, n: int | None = None, seed: int = 42):
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    if n and n < len(lines): lines = random.Random(seed).sample(lines, n)
    return lines

class WWMCollator:
    """Tokenise a batch of raw lines and apply whole-word masking (all sub-tokens of a chosen word together)."""
    def __init__(self, tok, max_len=128, p=0.15, seed=None):
        self.tok, self.max_len, self.p = tok, max_len, p
        self.rng = random.Random(seed) if seed is not None else random
        self.special = set(tok.all_special_ids)
    def __call__(self, lines):
        enc = self.tok(lines, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        ids, labels = enc["input_ids"].clone(), torch.full_like(enc["input_ids"], -100)
        V = len(self.tok)
        for i in range(ids.size(0)):
            wids = enc.word_ids(i); words = sorted({w for w in wids if w is not None})
            if not words: continue
            k = max(1, round(len(words) * self.p)); chosen = set(self.rng.sample(words, min(k, len(words))))
            for j, w in enumerate(wids):
                if w is None or w not in chosen or ids[i, j].item() in self.special: continue
                labels[i, j] = ids[i, j]; r = self.rng.random()
                if r < 0.8: ids[i, j] = self.tok.mask_token_id
                elif r < 0.9: ids[i, j] = self.rng.randrange(V)
        return {"input_ids": ids, "attention_mask": enc["attention_mask"], "labels": labels}

def build_model(init: str | None):
    if init:                                   # resume / already-built model directory
        return BertForMaskedLM.from_pretrained(init)
    mlm = BertForMaskedLM.from_pretrained("bert-base-uncased")
    enc = BertModel.from_pretrained("ProsusAI/finbert", add_pooling_layer=False)
    missing, unexpected = mlm.bert.load_state_dict(enc.state_dict(), strict=False)
    assert not [m for m in missing if "pooler" not in m], f"encoder weights missing: {missing}"
    mlm.tie_weights()                          # decoder weights = FinBERT word embeddings
    print(f"FinBERT encoder loaded into MLM model (missing={len(missing)}, unexpected={len(unexpected)})")
    return mlm

@torch.no_grad()
def evaluate(model, batches, dev):
    model.eval(); tot, n = 0.0, 0
    for b in batches:
        b = {k: v.to(dev) for k, v in b.items()}
        m = (b["labels"] != -100).sum().item()
        tot += model(**b).loss.item() * m; n += m
    model.train(); loss = tot / max(n, 1)
    return loss, math.exp(loss)

@torch.no_grad()
def probe(model, tok, dev):
    model.eval(); out = {}
    for s in PROBES:
        enc = tok(s, return_tensors="pt").to(dev)
        pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0].item()
        top = model(**enc).logits[0, pos].softmax(-1).topk(5)
        out[s] = [(tok.decode([i]).strip(), round(p, 3)) for p, i in zip(top.values.tolist(), top.indices.tolist())]
    model.train(); return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None, help="optimizer steps cap (overrides epochs)")
    ap.add_argument("--sample", type=int, default=None, help="use only N training lines")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch-size x grad-accum")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-lines", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="200 steps on 20k lines, writes to models/finbert-in-dapt-smoke")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = OUT
    if a.smoke:
        a.max_steps, a.sample, a.eval_every, a.save_every, a.val_lines = 200, 20000, 100, 10**9, 500
        out = OUT.parent / "finbert-in-dapt-smoke"
    out.mkdir(parents=True, exist_ok=True); ck = out / "ckpt"
    random.seed(a.seed); torch.manual_seed(a.seed); dev = device(); print("device:", dev)

    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    assert tok.is_fast, "need a fast tokenizer for word_ids()"
    resuming = a.resume and (ck / "state.pt").exists()
    model = build_model(str(ck) if resuming else None).to(dev); model.train()

    train = read_lines(DATA / "train.txt", a.sample, a.seed)
    val = read_lines(DATA / "val.txt", a.val_lines, a.seed)
    print(f"train lines {len(train):,}  val lines {len(val):,}")
    val_batches = list(DataLoader(val, batch_size=32, collate_fn=WWMCollator(tok, a.max_len, seed=123)))  # fixed masks
    coll = WWMCollator(tok, a.max_len)

    steps_per_epoch = math.ceil(len(train) / (a.batch_size * a.grad_accum))
    total = a.max_steps or int(steps_per_epoch * a.epochs)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": 0.01},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=a.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    step, log = 0, open(out / "log.jsonl", "a")
    if resuming:
        st = torch.load(ck / "state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
        random.setstate(st["py_rng"]); print(f"resumed at step {step}/{total}")
        if st.get("total", total) != total: print(f"WARNING: checkpoint was planned for {st['total']} steps; resume with the same flags")
    else:
        l0, p0 = evaluate(model, val_batches, dev)
        print(f"BEFORE  val loss {l0:.3f}  perplexity {p0:.2f}")
        pr0 = probe(model, tok, dev)
        for s, t in pr0.items(): print("  ", s, "->", t)
        log.write(json.dumps({"step": 0, "val_loss": l0, "ppl": p0, "probes": pr0}) + "\n"); log.flush()

    def save_ckpt():
        model.save_pretrained(ck); tok.save_pretrained(ck)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "total": total, "py_rng": random.getstate()}, ck / "state.pt")

    # deterministic epoch order; on resume skip the micro-batches already consumed
    micro_per_epoch = math.ceil(len(train) / a.batch_size)
    done_micro = step * a.grad_accum
    t0, run_loss, k, win = time.time(), 0.0, 0, 0
    epoch = done_micro // micro_per_epoch
    while step < total:
        order = list(range(len(train))); random.Random(a.seed + epoch).shuffle(order)
        start = done_micro % micro_per_epoch if epoch == done_micro // micro_per_epoch else 0
        for mb in range(start, micro_per_epoch):
            batch = coll([train[i] for i in order[mb * a.batch_size:(mb + 1) * a.batch_size]])
            batch = {kk: v.to(dev) for kk, v in batch.items()}
            loss = model(**batch).loss / a.grad_accum; loss.backward(); run_loss += loss.item(); k += 1
            if k % a.grad_accum: continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1; win += 1
            if step % 25 == 0:
                el = time.time() - t0; rate = (k / a.grad_accum) / el
                print(f"step {step}/{total}  loss {run_loss / win:.3f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{rate:.2f} steps/s  ETA {(total - step) / rate / 3600:.1f} h", flush=True)
                run_loss, win = 0.0, 0
            if step % a.eval_every == 0 or step == total:
                l, p = evaluate(model, val_batches, dev); print(f"  [eval] step {step}  val loss {l:.3f}  perplexity {p:.2f}")
                log.write(json.dumps({"step": step, "val_loss": l, "ppl": p}) + "\n"); log.flush()
            if step % a.save_every == 0: save_ckpt()
            if step >= total: break
        epoch += 1

    l1, p1 = evaluate(model, val_batches, dev); pr1 = probe(model, tok, dev)
    print(f"\nAFTER   val loss {l1:.3f}  perplexity {p1:.2f}")
    for s, t in pr1.items(): print("  ", s, "->", t)
    log.write(json.dumps({"step": step, "final": True, "val_loss": l1, "ppl": p1, "probes": pr1}) + "\n"); log.close()
    model.save_pretrained(out); tok.save_pretrained(out)
    print(f"\nsaved -> {out}\nNext: re-fine-tune a sentiment head on top of this encoder (the MLM step does not touch sentiment).")

if __name__ == "__main__":
    main()
