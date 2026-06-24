#!/usr/bin/env python3
"""Precompute per-element selection masks for the project-page playground.

The playground (docs/index.html) hit-tests clicks against each element's opaque
pixels. At runtime it reads those pixels with canvas.getImageData(), which the
browser blocks on a tainted canvas (e.g. when the page is opened as a local
file:// document) -- so it falls back to the whole bounding box and the entire
image becomes clickable.

This script bakes a compact, slightly *dilated* opaque mask into each element of
the per-episode JSON files, so the playground can hit-test without getImageData
(works everywhere) and elements are a touch easier to select.

Run it as the LAST step of the playground asset pipeline (after regenerating
docs/assets/playground/), then commit the updated *.json files:

    python scripts/build_playground_masks.py

Fields added per element:  mk (base64 packed bitmask, LSB-first), mw, mh.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO_ROOT / "docs" / "assets" / "playground"


def build_mask(img_path: Path, max_dim: int, dilate_frac: float, alpha_thr: int):
    """Return (mw, mh, base64 packed bits) for the dilated opaque mask."""
    im = Image.open(img_path).convert("RGBA")
    iw, ih = im.size
    s = min(1.0, max_dim / max(iw, ih))
    mw, mh = max(1, round(iw * s)), max(1, round(ih * s))
    # Area-averaging downscale (BOX) preserves coverage; BILINEAR/BICUBIC alias
    # badly at large downscale factors and would drop thin/sparse opaque regions.
    a = np.asarray(im.resize((mw, mh), Image.BOX))[:, :, 3]
    occ = a > alpha_thr

    # Morphological dilation by a small radius (square structuring element).
    r = max(1, round(dilate_frac * max(mw, mh)))
    if occ.any() and r > 0:
        pad = np.pad(occ, r)
        out = np.zeros_like(occ)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                out |= pad[r + dy: r + dy + mh, r + dx: r + dx + mw]
        occ = out

    bits = occ.flatten().astype(np.uint8)
    packed = np.packbits(bits, bitorder="little")
    return mw, mh, base64.b64encode(packed.tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help="playground assets dir (default: docs/assets/playground)")
    ap.add_argument("--max-dim", type=int, default=96,
                    help="mask resolution along the longer side (default: 96)")
    ap.add_argument("--dilate-frac", type=float, default=0.03,
                    help="dilation radius as a fraction of the longer side (default: 0.03)")
    ap.add_argument("--alpha-thr", type=int, default=12,
                    help="alpha threshold for 'opaque' (default: 12)")
    args = ap.parse_args()

    base = Path(args.dir)
    jsons = sorted(p for p in base.glob("*.json") if p.name != "index.json")
    print(f"Processing {len(jsons)} episode JSON files in {base}")

    total_el = 0
    for jp in jsons:
        m = json.loads(jp.read_text())
        key = m.get("key", jp.stem)
        ep_dir = base / key
        changed = 0
        for el in m.get("elements", []):
            img = el.get("img")
            if not img:
                continue
            ip = ep_dir / img
            if not ip.exists():
                print(f"  [warn] missing image: {ip}")
                continue
            mw, mh, mk = build_mask(ip, args.max_dim, args.dilate_frac, args.alpha_thr)
            el["mw"], el["mh"], el["mk"] = mw, mh, mk
            changed += 1
        jp.write_text(json.dumps(m, separators=(",", ":")))
        total_el += changed
        print(f"  {jp.name}: {changed} element masks")
    print(f"Done. Baked masks for {total_el} elements across {len(jsons)} episodes.")


if __name__ == "__main__":
    main()
