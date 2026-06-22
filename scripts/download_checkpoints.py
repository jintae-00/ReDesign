#!/usr/bin/env python3
"""
Download every model checkpoint the ReDesign agent and evaluation need.

Run from the repository root:

    python scripts/download_checkpoints.py            # tool + eval checkpoints
    python scripts/download_checkpoints.py --with-qwen  # also prefetch Qwen-Image-Layered (large)

What it fetches
---------------
Tool checkpoints downloaded automatically (saved to ./weights/):
  * groundingdino_swinb_cogcoor.pth   GroundingDINO SwinB  (open detection)   [IDEA-Research]
  * sam2.1_hiera_large.pt             SAM 2.1 Hiera-Large  (segmentation)      [Meta]
  * sam_vit_h_4b8939.pth              SAM ViT-H backbone (used by Hi-SAM)      [Meta]
  * big-lama.pt                       LaMa inpainting (TorchScript)            [IOPaint/Sanster mirror]
  * jixin0101/ObjectClear             ObjectClear inpainting pipeline          [HF, cached under weights/]

Manual download (authors' OneDrive only — the script prints instructions):
  * sam_tss_h_textseg.pth             Hi-SAM text-segmentation head            [ymy-k/Hi-SAM OneDrive]

Generation model (only with --with-qwen, ~tens of GB; otherwise it is fetched
automatically on first agent run):
  * Qwen/Qwen-Image-Layered           QwenImageLayeredPipeline                 [HF]

Evaluation models:
  * facebook/dino-vits16              DINO ViT-S/16 features (DINO metric)     [HF]
  * LPIPS (AlexNet) and PaddleOCR PP-OCRv5 download themselves on first use.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = REPO_ROOT / "weights"

# Checkpoints with stable public URLs (verified to match the originals) ---------
URL_WEIGHTS = {
    # GroundingDINO SwinB (open-vocabulary detection) — IDEA-Research
    "groundingdino_swinb_cogcoor.pth":
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
    # SAM 2.1 Hiera-Large (segmentation) — Meta
    "sam2.1_hiera_large.pt":
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    # SAM ViT-H backbone — required by Hi-SAM (it stores only the text-seg head) — Meta
    "sam_vit_h_4b8939.pth":
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    # LaMa inpainting (TorchScript) — IOPaint/Sanster mirror
    "big-lama.pt":
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
}

# Checkpoints distributed only via the authors' OneDrive (not script-automatable);
# the user downloads these manually. (name -> (source_url, note))
MANUAL_WEIGHTS = {
    "sam_tss_h_textseg.pth": (
        "https://1drv.ms/u/s!AimBgYV7JjTlgco1z9sdUi1vXCsKgA?e=U3WPJy",
        "Hi-SAM 'SAM-TS-H (TextSeg)' checkpoint — official release by ymy-k/Hi-SAM "
        "(https://github.com/ymy-k/Hi-SAM). Download and place at weights/sam_tss_h_textseg.pth",
    ),
}

# HF pipelines / models --------------------------------------------------------
OBJECTCLEAR_REPO = "jixin0101/ObjectClear"
QWEN_REPO = "Qwen/Qwen-Image-Layered"
DINO_REPO = "facebook/dino-vits16"

MIN_BYTES = 1_000_000  # treat a file smaller than this as a failed download


def _download_url(name: str, url: str) -> None:
    dst = WEIGHTS / name
    if dst.exists() and dst.stat().st_size > MIN_BYTES:
        print(f"[skip] {name} already present ({dst.stat().st_size/1e6:.0f} MB)")
        return
    print(f"[get ] {name}  <-  {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r       {pct:3d}%  ({done/1e6:.0f}/{total/1e6:.0f} MB)", end="", flush=True)
    print()
    tmp.rename(dst)



def _hf_snapshot(repo: str, repo_type: str = "model") -> None:
    from huggingface_hub import snapshot_download
    print(f"[get ] snapshot {repo}  ->  weights/ cache")
    snapshot_download(repo_id=repo, repo_type=repo_type, cache_dir=str(WEIGHTS))


def main() -> int:
    ap = argparse.ArgumentParser(description="Download ReDesign checkpoints")
    ap.add_argument("--with-qwen", action="store_true",
                    help="Also prefetch Qwen/Qwen-Image-Layered (large; otherwise fetched on first run)")
    ap.add_argument("--skip-objectclear", action="store_true", help="Skip ObjectClear download")
    ap.add_argument("--skip-eval", action="store_true", help="Skip evaluation models (DINO)")
    args = ap.parse_args()

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    print(f"Weights directory: {WEIGHTS}\n")

    # 1) public-URL tool checkpoints
    for name, url in URL_WEIGHTS.items():
        _download_url(name, url)

    # 2) manual-only checkpoints (OneDrive)
    manual_pending = []
    for fn, (src, note) in MANUAL_WEIGHTS.items():
        dst = WEIGHTS / fn
        if dst.exists() and dst.stat().st_size > MIN_BYTES:
            print(f"[skip] {fn} already present")
        else:
            manual_pending.append((fn, src, note))

    # 3) ObjectClear pipeline (cached under weights/)
    if not args.skip_objectclear:
        try:
            _hf_snapshot(OBJECTCLEAR_REPO)
        except Exception as e:
            print(f"[WARN] ObjectClear download failed: {e}")

    # 4) Qwen-Image-Layered (optional, large)
    if args.with_qwen:
        try:
            _hf_snapshot(QWEN_REPO)
        except Exception as e:
            print(f"[WARN] Qwen-Image-Layered download failed: {e}")
    else:
        print("[note] Skipping Qwen/Qwen-Image-Layered (use --with-qwen to prefetch; "
              "otherwise it downloads automatically on the first agent run).")

    # 5) evaluation models
    if not args.skip_eval:
        try:
            from huggingface_hub import snapshot_download
            print(f"[get ] {DINO_REPO} (DINO metric)")
            snapshot_download(repo_id=DINO_REPO)
        except Exception as e:
            print(f"[WARN] DINO model prefetch failed: {e}")
        print("[note] LPIPS (AlexNet) and PaddleOCR PP-OCRv5 download on first use.")

    print("\nDownloaded checkpoints are in ./weights/ and the HuggingFace cache.")
    if manual_pending:
        print("\n" + "=" * 70)
        print("MANUAL DOWNLOAD REQUIRED (distributed only via the authors' OneDrive):")
        for fn, src, note in manual_pending:
            print(f"\n  • {fn}")
            print(f"      url:  {src}")
            print(f"      {note}")
        print("=" * 70)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
