"""Assemble the sentiment label sets.

  python bert/build_label_sets.py
    data/labels/train_final.csv     train rows: TCS sample (397) + contrast (53)
                                    label = your_label if given > train_recheck suggestion > Claude draft
    data/labels/test_to_label.csv   2024 blind test, NO drafts — label column `your_label`
    data/labels/test_final.csv      test rows with gold labels: your blind labels if test_to_label.csv is filled,
                                    else Claude's hidden drafts (marked label_source=claude_provisional)
"""
from pathlib import Path
import pandas as pd
L = Path(__file__).resolve().parents[1] / "data" / "labels"
VALID = {"positive", "negative", "neutral"}

def norm(x): x = str(x).strip().lower(); return x if x in VALID else None

def user_labels(path, sheet=None):
    if not path.exists(): return {}
    df = pd.read_excel(path, sheet_name=sheet) if path.suffix == ".xlsx" else pd.read_csv(path, encoding="utf-8-sig")
    return {r.id: norm(r.your_label) for r in df.itertuples() if norm(r.your_label)}

def main():
    # ---- train ----
    tr = pd.read_csv(L / "sample_train.csv").assign(target="TCS")
    drafts = pd.read_csv(L / "train_validate.csv")[["id", "draft_label"]]
    tr = tr.merge(drafts, on="id")                                            # drops the 3 tax-TCS rows
    co = pd.read_csv(L / "sample_contrast.csv").merge(pd.read_csv(L / "contrast_drafts.csv")[["id", "draft_label"]], on="id")
    tr = pd.concat([tr, co], ignore_index=True)
    tr["label"], tr["label_source"] = tr["draft_label"], "claude_draft_accepted"
    rc = pd.read_csv(L / "train_recheck.csv")
    for r in rc.itertuples():
        m = tr["id"] == r.id
        if r.suggested_label != r.draft_label: tr.loc[m, ["label", "label_source"]] = [r.suggested_label, "recheck_rule"]
    yours = {}
    for f, sh in [(L / "test_blind.xlsx", "test_blind"), (L / "train_validate.xlsx", None), (L / "contrast_validate.csv", None)]:
        yours.update(user_labels(f, sh))
    for i, lab in yours.items():
        m = tr["id"] == i
        if m.any(): tr.loc[m, ["label", "label_source"]] = [lab, "user"]
    cols = ["id", "date", "source", "target", "title", "role", "article_type", "label", "label_source"]
    tr[cols].to_csv(L / "train_final.csv", index=False)
    print(f"train {len(tr)}: {tr.label.value_counts().to_dict()}  sources {tr.label_source.value_counts().to_dict()}")

    # ---- test ----
    hidden = pd.read_csv(L / ".hidden" / "test_claude_hidden.csv")
    te = pd.read_csv(L / "sample_test.csv").assign(target="TCS")
    te = te[te["id"].isin(hidden["id"])]                                      # same 148 (tax rows removed)
    tl = L / "test_to_label.csv"
    if not tl.exists():
        te[["id", "date", "title", "target_span"]].assign(your_label="", note="").to_csv(tl, index=False, encoding="utf-8-sig")
        print(f"wrote {tl.name} ({len(te)} rows) — label this one blind")
    mine = user_labels(tl)
    te = te.merge(hidden[["id", "draft_label"]], on="id")
    te["label"] = te["id"].map(mine).fillna(te["draft_label"])
    te["label_source"] = te["id"].map(lambda i: "user" if i in mine else "claude_provisional")
    te[cols].to_csv(L / "test_final.csv", index=False)
    print(f"test {len(te)}: {te.label.value_counts().to_dict()}  sources {te.label_source.value_counts().to_dict()}")
    if (te.label_source == "user").any() and (te.label_source == "user").all():
        from collections import Counter
        a, b = te["label"], te["draft_label"]; po = (a == b).mean()
        pe = sum(Counter(a)[k] * Counter(b)[k] for k in VALID) / len(a) ** 2
        print(f"agreement you vs Claude drafts: {po:.1%}, Cohen's kappa {(po - pe) / (1 - pe):.3f}")

if __name__ == "__main__":
    main()
