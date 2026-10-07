"""Download the HF storage bucket into data/hf/ using your Hugging Face token (hf://buckets paths).

.env needs:  HUGGINGFACE_API_KEY=hf_...      (Read permission is enough)
Optional S3 route instead: AWS_ACCESS_KEY_ID=HFAK...  AWS_SECRET_ACCESS_KEY=...  (Settings → Access Tokens → Generate S3 credentials)

  pip install -U huggingface_hub
  python news/download_hf_bucket.py --list      # list files + sizes
  python news/download_hf_bucket.py --small     # skip the 2.4 GB + 857 MB files for a first look
  python news/download_hf_bucket.py             # everything (4.2 GB)
Skips files already downloaded with the same size.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "data" / "hf"
BUCKET_PATH = "buckets/seshjuansauce/Six_Year_Indian_Stock_Market_Dataset-News_and_Ticker-bucket"
PREFIX = "dataset"
BIG = {"tier_segregated_news.csv", "processed_news_dataset.csv"}

def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--small", action="store_true"); ap.add_argument("--list", action="store_true")
    a = ap.parse_args(); load_env()
    token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    if not token: raise SystemExit("HUGGINGFACE_API_KEY missing in .env")
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=token)
    root = f"{BUCKET_PATH}/{PREFIX}"
    files = [f for f in fs.find(root, detail=True).values() if f.get("type") != "directory"]
    print(f"{len(files)} files, {sum(f['size'] for f in files)/1e9:.2f} GB")
    for f in files: print(f"  {f['size']/1e6:9.1f} MB  {f['name'][len(root)+1:]}")
    if a.list: return
    for f in files:
        rel = f["name"][len(root) + 1:]
        if a.small and Path(rel).name in BIG: print(f"skip (big) {rel}"); continue
        dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == f["size"]: print(f"skip (have) {rel}"); continue
        print(f"downloading {rel} ({f['size']/1e6:.0f} MB) ...", flush=True)
        fs.get(f["name"], str(dst))
    print(f"\ndone → {OUT}")

if __name__ == "__main__":
    main()
