"""SentimentModel: one fine-tuned FinBERT checkpoint + its input format + its calibration temperature."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LABELS = ("positive", "negative", "neutral")


class SentimentModel:
    """Wraps a sequence-classification checkpoint.

    The checkpoint folder may hold `input_format.json` (written by bert/finetune_sentiment.py or bert/finetune_call.py):
      {"format": "pair_tocm", "pair": true, "t": true, "o": true, "c": true, "m": true}       news models
      {"format": "call", "pair": true, ...}                                                     call model
    Probabilities are softmax(logits / T); T comes from results/sentiment/calibration.json (1.0 if absent).
    Torch/transformers are imported lazily so the rest of the package works without them.
    """

    def __init__(self, path: str | Path, temperature: float = 1.0, device: str | None = None,
                 batch_size: int = 64, max_len: int = 128, name: str | None = None):
        self.path, self.temperature, self.batch_size, self.max_len = str(path), float(temperature), batch_size, max_len
        self.name = name or Path(self.path).name
        fmt_file = Path(self.path) / "input_format.json"
        self.fmt = json.loads(fmt_file.read_text()) if fmt_file.exists() else {"format": "full", "pair": False}
        self._device_pref, self._model, self._tok, self.labels = device, None, None, list(LABELS)

    # ---------- construction ----------
    @classmethod
    def from_settings(cls, path: str | Path, calibration: str | Path | None, key: str | None, device: str | None = None):
        T = 1.0
        if calibration and Path(calibration).exists() and key:
            cal = json.loads(Path(calibration).read_text())
            T = float(cal.get(key, {}).get("T", 1.0))
        return cls(path, temperature=T, device=device, name=key)

    def _load(self):
        if self._model is not None: return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        dev = self._device_pref or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(dev)
        self._tok = AutoTokenizer.from_pretrained(self.path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.path).to(self._device).eval()
        self.labels = [self._model.config.id2label[i].lower() for i in range(self._model.config.num_labels)]

    # ---------- input formatting ----------
    def format_inputs(self, texts: Sequence[str], targets: Sequence[str | None] | str | None = None,
                      contexts: Sequence[str] | None = None) -> list[tuple[str, str | None]]:
        """(text_a, text_b) per row. News: target-aware markers. Call: (speaker role + section, marked sentence)."""
        from entity import model_input
        if isinstance(targets, str) or targets is None: targets = [targets] * len(texts)
        f = self.fmt
        if not f.get("pair"):
            return [(t, None) for t in texts]
        if f.get("format") == "call":
            from .calls import call_input
            ctx = contexts or ["management prepared"] * len(texts)
            return [call_input(t, tg, c, f) for t, tg, c in zip(texts, targets, ctx)]
        out = []
        for t, tg in zip(texts, targets):
            if tg is None:
                _, b = model_input(t, "__none__", entity_markers=False, other_markers=False,
                                   contrast_tags=f.get("c", True), market_tags=f.get("m", True))
                out.append(("the company", b))
            else:
                out.append(model_input(t, tg, entity_markers=f.get("t", True), other_markers=f.get("o", True),
                                       contrast_tags=f.get("c", True), market_tags=f.get("m", True)))
        return out

    # ---------- inference ----------
    def logits(self, texts: Sequence[str], targets=None, contexts=None) -> np.ndarray:
        if len(texts) == 0: return np.zeros((0, 3))
        self._load()
        import torch
        pairs = self.format_inputs(texts, targets, contexts); out = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                chunk = pairs[i:i + self.batch_size]
                a = [p[0] for p in chunk]; b = [p[1] for p in chunk]
                enc = self._tok(a, b if all(x is not None for x in b) else None, truncation=True,
                                max_length=self.max_len, padding=True, return_tensors="pt").to(self._device)
                out.append(self._model(**enc).logits.float().cpu().numpy())
        z = np.concatenate(out)
        # reorder columns to LABELS so every caller sees positive, negative, neutral
        return z[:, [self.labels.index(l) for l in LABELS]]

    @staticmethod
    def probs(z: np.ndarray, T: float = 1.0) -> np.ndarray:
        z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    def predict(self, texts: Sequence[str], targets=None, contexts=None) -> pd.DataFrame:
        """One row per text: logits, calibrated probabilities, s = P(pos) − P(neg), label."""
        z = self.logits(texts, targets, contexts); p = self.probs(z, self.temperature)
        df = pd.DataFrame(z, columns=[f"z_{l}" for l in LABELS])
        for j, l in enumerate(LABELS): df[f"p_{l}"] = p[:, j]
        df["s"] = df["p_positive"] - df["p_negative"]
        df["label"] = np.array(LABELS)[p.argmax(1)] if len(df) else []
        return df

    def __repr__(self): return f"SentimentModel({self.name!r}, T={self.temperature:.2f}, format={self.fmt.get('format')})"
