# prepare_crello_records.py
# -*- coding: utf-8 -*-
"""
Prepare Crello records for the ReDesign agent and evaluation.

Reads the Crello test Parquet shards (download them first; see
crello_data/README.md) and renders each design into one record directory
containing the composited canvas (`composite.png`, the agent's input image),
the per-element GT images, and `gt_metadata.json`.

Usage:
    # Render all test records into ./crello_data/records
    python scripts/prepare_crello_records.py \
        --parquet-glob "crello_data/test-*.parquet" \
        --output-dir crello_data/records

    # Quick smoke test: only the first 3 records
    python scripts/prepare_crello_records.py \
        --parquet-glob "crello_data/test-*.parquet" \
        --output-dir crello_data/records --limit 3

Arguments:
    --parquet-glob  Glob for the downloaded Crello test Parquet shards.
    --output-dir    Where the crello_test_*/ record directories are written.
    --limit         Process only the first N records (0 = all).
    --filter        'none' (default; keep every record) or 'paper'
                    (reproduce the paper subset: text_chars>60, 5<=non_text<=25).
    --cache-dir     HuggingFace datasets cache dir (default: <output-dir>/.hf_cache).

Output structure (per record):
    <output-dir>/
    ├── crello_test_0000/
    │   ├── gt_metadata.json          # GT metadata (evaluation-compatible)
    │   ├── composite.png             # full composited canvas (agent input)
    │   ├── elements/
    │   │   ├── element_000.png       # each element RGBA placed on the full canvas
    │   │   └── ...
    ├── crello_test_0001/
    │   └── ...
    └── subset_metadata.json          # index + statistics for the whole set

This is the structure expected by:
    python -m ReDesign.run_agent_crello --data_dir <output-dir> --output_dir ...
    python evaluation/eval_accuracy_baselines_crello.py --crello-subset <output-dir> ...
"""
from __future__ import annotations

import json
import sys
import traceback
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# =========================================================
# [Configuration]
# =========================================================
import argparse  # noqa: E402

# Paper-subset filtering thresholds (only applied with --filter paper)
THRESHOLD_TEXT_CHAR_MIN = 60
THRESHOLD_NON_TEXT_MIN = 5
THRESHOLD_NON_TEXT_MAX = 25

# [Crello Type Definitions]
# 0: SvgElement, 1: TextElement, 2: ImageElement, 3: ColoredBackground, 4: SvgMaskElement
TEXT_TYPE_CODE = 1
TYPE_CODE_TO_NAME = {
    0: "svgElement",
    1: "textElement",
    2: "imageElement",
    3: "coloredBackground",
    4: "svgMaskElement",
}
# unit_type mapping for evaluation compatibility
TYPE_CODE_TO_EVAL_TYPE = {
    0: "object",
    1: "text",
    2: "object",
    3: "background",
    4: "object",
}


# =========================================================
# [Helper] statistics
# =========================================================
def calculate_crello_semantics(rec: Dict[str, Any]) -> Dict[str, int]:
    """Compute total text-character count and non-text element count for a record."""
    types = rec.get("type", [])
    texts = rec.get("text", [])
    loop_len = len(types)
    text_char_sum = 0
    non_text_count = 0

    for i in range(loop_len):
        t_code = types[i]
        if t_code == TEXT_TYPE_CODE:
            if i < len(texts) and texts[i]:
                text_char_sum += len(str(texts[i]))
        else:
            non_text_count += 1

    return {
        "text_char_count": text_char_sum,
        "non_text_unit_count": non_text_count,
    }


# =========================================================
# [Core] element image decode + full-canvas placement
# =========================================================
def decode_element_image(img_obj) -> Optional[Image.Image]:
    """Decode a crello record image field into a PIL Image."""
    raw: Optional[bytes] = None
    if img_obj is None:
        return None
    if isinstance(img_obj, str):
        import base64
        raw = base64.b64decode(img_obj)
    elif isinstance(img_obj, bytes):
        raw = img_obj
    elif isinstance(img_obj, dict) and "bytes" in img_obj:
        raw = img_obj["bytes"]
    elif isinstance(img_obj, Image.Image):
        return img_obj.convert("RGBA")
    else:
        return None

    if raw is None:
        return None
    return Image.open(BytesIO(raw)).convert("RGBA")


def place_element_on_canvas(
    elem_img: Image.Image,
    canvas_w: int,
    canvas_h: int,
    width: int,
    height: int,
    left: float,
    top: float,
    angle: float,
    opacity: float,
) -> Tuple[Optional[Image.Image], Optional[List[int]]]:
    """
    Place an element onto the full canvas.
    Mirrors the GT loading logic used by the Crello evaluator.

    Returns:
        canvas (RGBA PIL Image, full canvas size)
        bbox [x1, y1, x2, y2] (alpha-based tight bbox)
    """
    # 1) Resize to declared width x height
    elem_img = elem_img.resize((width, height), Image.LANCZOS)

    # 2) apply opacity
    if opacity < 1.0:
        alpha_ch = elem_img.getchannel("A")
        alpha_ch = alpha_ch.point(lambda x: int(x * opacity))
        elem_img.putalpha(alpha_ch)

    # 3) apply rotation
    if angle != 0.0:
        elem_img = elem_img.rotate(-angle, expand=True, resample=Image.BICUBIC)

    # 4) center-based coordinates -> place on the full canvas
    cx = left + width / 2.0
    cy = top + height / 2.0
    cur_w, cur_h = elem_img.size
    x = int(cx - cur_w / 2.0)
    y = int(cy - cur_h / 2.0)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    canvas.alpha_composite(elem_img, dest=(x, y))

    # 5) Tight bbox from alpha
    arr = np.array(canvas, dtype=np.float32) / 255.0
    alpha = arr[..., 3]
    if not np.any(alpha > 0.01):
        return None, None

    rows = np.any(alpha > 0.01, axis=1)
    cols = np.any(alpha > 0.01, axis=0)
    y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    bbox = [x1, y1, x2 + 1, y2 + 1]

    return canvas, bbox


def composite_elements(
    element_canvases: List[Image.Image],
    canvas_w: int,
    canvas_h: int,
) -> Image.Image:
    """
    Composite all element canvases in z-order (list order) to build the reconstruction image.
    """
    composite = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    for elem_canvas in element_canvases:
        if elem_canvas is not None:
            composite = Image.alpha_composite(composite, elem_canvas)
    return composite


# =========================================================
# [Core] single-record processing
# =========================================================
def process_single_record(
    rec: Dict[str, Any],
    original_idx: int,
    record_id: str,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Process a single crello record:
      1) save each element image under elements/
      2) build & save composite.png
      3) save gt_metadata.json
    
    Returns:
        record_summary dict (for subset_metadata.json) or None on failure
    """
    record_dir = output_dir / record_id
    elements_dir = record_dir / "elements"
    elements_dir.mkdir(parents=True, exist_ok=True)

    canvas_w = int(rec["canvas_width"])
    canvas_h = int(rec["canvas_height"])

    images = rec.get("image", [])
    types = rec.get("type", [])
    texts = rec.get("text", [])
    widths = rec.get("width", [])
    heights = rec.get("height", [])
    lefts = rec.get("left", [])
    tops = rec.get("top", [])
    angles = rec.get("angle", [])
    opacities = rec.get("opacity", [])
    colors = rec.get("color", [])
    fonts = rec.get("font", [])
    font_sizes = rec.get("font_size", [])
    text_aligns = rec.get("text_align", [])
    line_heights = rec.get("line_height", [])
    letter_spacings = rec.get("letter_spacing", [])
    capitalizes = rec.get("capitalize", [])

    num_elements = len(types)

    # ---- Process each element ----
    unit_images_meta: List[Dict[str, Any]] = []
    element_canvases: List[Optional[Image.Image]] = []
    text_char_count = 0
    non_text_count = 0

    for i in range(num_elements):
        type_code = int(types[i])
        type_name = TYPE_CODE_TO_NAME.get(type_code, "unknown")
        eval_type = TYPE_CODE_TO_EVAL_TYPE.get(type_code, "object")

        w_elem = int(widths[i]) if i < len(widths) else 0
        h_elem = int(heights[i]) if i < len(heights) else 0
        left_val = float(lefts[i]) if i < len(lefts) else 0.0
        top_val = float(tops[i]) if i < len(tops) else 0.0
        angle_val = float(angles[i]) if i < len(angles) else 0.0
        opacity_val = float(opacities[i]) if i < len(opacities) else 1.0
        text_content = str(texts[i]) if (i < len(texts) and texts[i]) else ""
        color_val = colors[i] if i < len(colors) else []
        font_val = int(fonts[i]) if i < len(fonts) else 0
        font_size_val = float(font_sizes[i]) if i < len(font_sizes) else 0.0
        text_align_val = int(text_aligns[i]) if i < len(text_aligns) else 0
        line_height_val = float(line_heights[i]) if i < len(line_heights) else 0.0
        letter_spacing_val = float(letter_spacings[i]) if i < len(letter_spacings) else 0.0
        capitalize_val = bool(capitalizes[i]) if i < len(capitalizes) else False

        # statistics
        if type_code == TEXT_TYPE_CODE:
            text_char_count += len(text_content)
        else:
            non_text_count += 1

        # decode image
        img_obj = images[i] if i < len(images) else None
        elem_img = decode_element_image(img_obj)
        if elem_img is None or w_elem <= 0 or h_elem <= 0:
            element_canvases.append(None)
            unit_images_meta.append({
                "image_path": None,
                "unit_id": f"elem_{i}",
                "unit_type": eval_type,
                "type_name": type_name,
                "type_code": type_code,
                "z_index": i,
                "bbox": None,
                "width": w_elem,
                "height": h_elem,
                "left": left_val,
                "top": top_val,
                "angle": angle_val,
                "opacity": opacity_val,
                "text": text_content,
                "color": color_val,
                "font": font_val,
                "font_size": font_size_val,
                "text_align": text_align_val,
                "line_height": line_height_val,
                "letter_spacing": letter_spacing_val,
                "capitalize": capitalize_val,
                "valid": False,
            })
            continue

        # place on the full canvas
        canvas_img, bbox = place_element_on_canvas(
            elem_img, canvas_w, canvas_h,
            w_elem, h_elem, left_val, top_val,
            angle_val, opacity_val,
        )

        if canvas_img is None:
            element_canvases.append(None)
            unit_images_meta.append({
                "image_path": None,
                "unit_id": f"elem_{i}",
                "unit_type": eval_type,
                "type_name": type_name,
                "type_code": type_code,
                "z_index": i,
                "bbox": None,
                "width": w_elem,
                "height": h_elem,
                "left": left_val,
                "top": top_val,
                "angle": angle_val,
                "opacity": opacity_val,
                "text": text_content,
                "color": color_val,
                "font": font_val,
                "font_size": font_size_val,
                "text_align": text_align_val,
                "line_height": line_height_val,
                "letter_spacing": letter_spacing_val,
                "capitalize": capitalize_val,
                "valid": False,
            })
            continue

        # save
        elem_filename = f"element_{i:03d}.png"
        canvas_img.save(str(elements_dir / elem_filename), "PNG")
        element_canvases.append(canvas_img)

        # alpha-based area
        alpha_arr = np.array(canvas_img.getchannel("A"), dtype=np.float32) / 255.0
        area = float(np.sum(alpha_arr > 0.01))

        unit_images_meta.append({
            "image_path": elem_filename,
            "unit_id": f"elem_{i}",
            "unit_type": eval_type,
            "type_name": type_name,
            "type_code": type_code,
            "z_index": i,
            "bbox": bbox,
            "area": area,
            "width": w_elem,
            "height": h_elem,
            "left": left_val,
            "top": top_val,
            "angle": angle_val,
            "opacity": opacity_val,
            "text": text_content,
            "color": color_val,
            "font": font_val,
            "font_size": font_size_val,
            "text_align": text_align_val,
            "line_height": line_height_val,
            "letter_spacing": letter_spacing_val,
            "capitalize": capitalize_val,
            "valid": True,
        })

    # ---- build composite image ----
    composite_img = composite_elements(element_canvases, canvas_w, canvas_h)
    composite_path = record_dir / "composite.png"
    composite_img.save(str(composite_path), "PNG")

    # ---- GT Metadata JSON ----
    crello_id = rec.get("id", "")
    crello_format = rec.get("format", -1)
    crello_category = rec.get("category", -1)
    crello_group = rec.get("group", -1)
    crello_title = rec.get("title", "")
    crello_length = rec.get("length", num_elements)

    gt_metadata = {
        # canvas info
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        # record identity
        "record_id": record_id,
        "original_index": original_idx,
        "crello_template_id": crello_id,
        # crello classification meta
        "format": int(crello_format) if crello_format is not None else -1,
        "category": int(crello_category) if crello_category is not None else -1,
        "group": int(crello_group) if crello_group is not None else -1,
        "title": str(crello_title) if crello_title else "",
        "length": int(crello_length),
        # image paths (evaluation-compatible)
        "reconstructed_image_path": "composite.png",
        "unit_images_dir": "elements",
        # element info
        "unit_images": unit_images_meta,
        # statistics
        "statistics": {
            "text_char_count": text_char_count,
            "non_text_count": non_text_count,
            "total_elements": num_elements,
            "valid_elements": sum(1 for u in unit_images_meta if u["valid"]),
        },
    }

    meta_path = record_dir / "gt_metadata.json"
    meta_path.write_text(json.dumps(gt_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "record_id": record_id,
        "original_index": original_idx,
        "crello_template_id": crello_id,
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "total_elements": num_elements,
        "valid_elements": gt_metadata["statistics"]["valid_elements"],
        "text_char_count": text_char_count,
        "non_text_count": non_text_count,
    }


# =========================================================
# [Main]
# =========================================================
def main():

    ap = argparse.ArgumentParser(description="Render Crello Parquet records into agent/eval-ready directories")
    ap.add_argument("--parquet-glob", required=True,
                    help="Glob for the downloaded Crello test Parquet shards (e.g. 'crello_data/test-*.parquet')")
    ap.add_argument("--output-dir", required=True,
                    help="Output directory for the crello_test_*/ record directories")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N (post-filter) records (0 = all)")
    ap.add_argument("--filter", choices=["none", "paper"], default="none",
                    help="'none' keeps every record; 'paper' reproduces the paper subset "
                         f"(text_chars>{THRESHOLD_TEXT_CHAR_MIN}, "
                         f"{THRESHOLD_NON_TEXT_MIN}<=non_text<={THRESHOLD_NON_TEXT_MAX})")
    ap.add_argument("--cache-dir", default=None,
                    help="HuggingFace datasets cache dir (default: <output-dir>/.hf_cache)")
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / ".hf_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[Stage 0] Crello record preparation")
    print(f"  Parquet : {args.parquet_glob}")
    print(f"  Filter  : {args.filter}")
    print(f"  Output  : {out_dir}")
    print("=" * 60)

    # ---- Stage 1: Load Dataset ----
    print("\n[Stage 1] Loading Crello test dataset...")
    try:
        ds = load_dataset(
            "parquet",
            data_files={"test": args.parquet_glob},
            split="test",
            cache_dir=str(cache_dir),
        )
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        return
    print(f"  Loaded {len(ds)} records.")

    # ---- Stage 2: Select records ----
    print("\n[Stage 2] Selecting records...")
    selected_indices: List[int] = []
    for idx in tqdm(range(len(ds)), desc="Selecting", ncols=100):
        if args.filter == "none":
            selected_indices.append(idx)
            continue
        stats = calculate_crello_semantics(ds[idx])
        t_char = stats["text_char_count"]
        nt_unit = stats["non_text_unit_count"]
        if (t_char > THRESHOLD_TEXT_CHAR_MIN
                and THRESHOLD_NON_TEXT_MIN <= nt_unit <= THRESHOLD_NON_TEXT_MAX):
            selected_indices.append(idx)

    if args.limit and args.limit > 0:
        selected_indices = selected_indices[: args.limit]

    count = len(selected_indices)
    print(f"\n  Selected: {count} / {len(ds)} records")
    if count == 0:
        print("[Warn] 0 records selected. Adjust --filter / --limit.")
        return

    # ---- Stage 3: Process each record ----
    print(f"\n[Stage 3] Processing {count} records (image extraction + compositing)...")
    record_summaries: List[Dict[str, Any]] = []
    error_records: List[Dict[str, Any]] = []

    for original_idx in tqdm(selected_indices, desc="Extracting", ncols=100, unit="rec"):
        record_id = f"crello_test_{original_idx:04d}"
        try:
            summary = process_single_record(ds[original_idx], original_idx, record_id, out_dir)
            if summary is not None:
                record_summaries.append(summary)
            else:
                error_records.append({"record_id": record_id, "error": "process returned None"})
        except Exception as e:
            error_records.append({"record_id": record_id, "error": str(e), "traceback": traceback.format_exc()})
            tqdm.write(f"  [Error] {record_id}: {e}")

    # ---- Stage 4: Save global metadata ----
    print("\n[Stage 4] Saving subset metadata...")
    subset_metadata = {
        "base_dataset": args.parquet_glob,
        "filter": args.filter,
        "total_original": len(ds),
        "selected_count": count,
        "successfully_processed": len(record_summaries),
        "errors": len(error_records),
        "original_indices": selected_indices,
        "records": record_summaries,
        "error_details": error_records,
    }
    meta_path = out_dir / "subset_metadata.json"
    meta_path.write_text(json.dumps(subset_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("[Done] Crello record preparation complete")
    print(f"  Selected:     {count}")
    print(f"  Processed OK: {len(record_summaries)}")
    print(f"  Errors:       {len(error_records)}")
    print(f"  Output dir:   {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()