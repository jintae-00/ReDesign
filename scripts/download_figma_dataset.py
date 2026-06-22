#!/usr/bin/env python3
"""
Download the ReDesign Figma-909 benchmark from the HuggingFace Hub into ./figma_data.

    python scripts/download_figma_dataset.py

After download, ./figma_data/ contains:
    valid_frames/<episode_id>.json          (909 GT metadata files, CC BY 4.0)
    unit_images/<figma_dir>/...             (per-episode GT layers + reconstruction)
    reconstructed_images/<episode_id>.png   (GT reconstruction, episode-id keyed)
    ATTRIBUTIONS.csv                        (per-episode author / source / license)

This is exactly the layout expected by:
    python -m ReDesign.run_agent_figma --data_dir figma_data --output_dir outputs/figma_agent
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "Jintae-Park/ReDesign-Figma909"


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the ReDesign Figma-909 dataset")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"HF dataset repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--out", default=str(REPO_ROOT / "figma_data"), help="Output directory")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo}  ->  {out}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(out),
    )

    n = len(list((out / "valid_frames").glob("*.json"))) if (out / "valid_frames").exists() else 0
    print(f"\nDone. valid_frames episodes: {n} (expected 909)")
    if n != 909:
        print("[WARN] expected 909 episodes — verify the download.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
