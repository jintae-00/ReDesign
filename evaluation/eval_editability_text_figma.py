#!/usr/bin/env python3
"""Text editability evaluation for multiple baseline models.

Content-based text matching pipeline:

Phase 1 — Extract pred text elements per model per episode
  agent/multi_tools/sparse_verif:  parse.json → filter type=="text" → [{bbox, content}]
  layered/qwen:                    render reconstruction → PaddleOCR on full image → [{bbox, content}]
  vtracer:                         output.svg → cairosvg rasterize → PaddleOCR → [{bbox, content}]

Phase 2 — Content Recognition (global sorted CER/WER per episode)
  Sort GT and pred texts by reading order → concatenate → CER/WER

Phase 3 — Content Modification (word replacement + CER/WER per episode)
  Select 5 words per episode from smaller-font GT texts, replace with random chars,
  apply same replacements to both GT and pred concatenated strings → CER/WER

Usage:
    # Replace <GPU_ID> with one of your own GPU ids (e.g. 0).
    python scripts/eval_text_edit_baselines.py \\
        --figma-data figma_data \\
        --agent-dir <AGENT_OUTPUT_DIR> \\
        --qwen-dir <QWEN_OUTPUT_DIR> \\
        --models vtracer layered qwen multi_tools sparse_verif agent \\
        --layered-dir <LAYERED_BASELINE_OUTPUT_DIR> \\
        --multi-tools-dir <MULTI_TOOLS_BASELINE_OUTPUT_DIR> \\
        --sparse-verif-dir <SPARSE_VERIF_BASELINE_OUTPUT_DIR> \\
        --vtracer-dir <VTRACER_BASELINE_OUTPUT_DIR> \\
        --output <OUTPUT_DIR> \\
        --seed 123 --num-workers 20 --ocr-gpu <GPU_ID> --ocr-pool-size 4 \\
        --n-mods 4 --min-pred-text-len 2

    The agent/qwen/baseline output dirs are produced by running the inference runners
    first (e.g. ``python -m ReDesign.run_agent_figma --data_dir figma_data \\
    --output_dir <AGENT_OUTPUT_DIR>``), and ``--figma-data`` should point at the
    downloaded ``figma_data`` dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import traceback
from pathlib import Path
from statistics import median as _median
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.baseline_model_configs import (
    MODEL_CONFIGS,
    add_baseline_dir_args,
    collect_gt_episodes,
    get_common_episodes,
    get_model_dir,
    scan_model_episodes,
    scan_model_episodes_multi,
    _resolve_multi_dirs,
)
import queue

import cv2

from evaluation.editability_utils.common_utils import (
    load_json,
    normalize_text,
    preprocess_text_content,
    save_json,
)

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

_print = lambda *a, **kw: print(*a, **kw, flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_TEXT_MODELS = ["vtracer", "layered", "qwen", "multi_tools", "sparse_verif", "agent"]
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


# ---------------------------------------------------------------------------
# LCS-based recall metrics (ignore FP / insertion penalty)
# ---------------------------------------------------------------------------

def _lcs_length(a, b) -> int:
    """Length of Longest Common Subsequence (space-optimized DP)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # Keep a as shorter for space efficiency
    if n > m:
        a, b = b, a
        n, m = m, n
    prev = [0] * (n + 1)
    for j in range(1, m + 1):
        curr = [0] * (n + 1)
        for i in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[i] = prev[i - 1] + 1
            else:
                curr[i] = max(prev[i], curr[i - 1])
        prev = curr
    return prev[n]


def compute_char_recall(gt: str, pred: str) -> float:
    """Character-level recall via LCS. Ignores insertions (FP).

    = LCS(gt, pred) / len(gt).  Higher = better.  1.0 = all GT chars found.
    """
    if not gt:
        return 1.0
    if not pred:
        return 0.0
    return _lcs_length(gt, pred) / len(gt)


def compute_word_recall(gt: str, pred: str) -> float:
    """Word-level recall via LCS. Ignores insertions (FP).

    = LCS(gt_words, pred_words) / len(gt_words).  Higher = better.
    """
    gw = gt.split()
    pw = pred.split()
    if not gw:
        return 1.0
    if not pw:
        return 0.0
    return _lcs_length(gw, pw) / len(gw)


# ---------------------------------------------------------------------------
# GPU-based PaddleOCR pool (loaded on a specific GPU, independent of CPU pool)
# ---------------------------------------------------------------------------

_GPU_OCR_QUEUE: Optional[queue.Queue] = None
_GPU_OCR_LOCK = Lock()


def _init_gpu_ocr_pool(gpu_id: int = 3, pool_size: int = 4) -> None:
    """Initialize GPU-based PaddleOCR pool on a specific GPU device."""
    global _GPU_OCR_QUEUE
    with _GPU_OCR_LOCK:
        if _GPU_OCR_QUEUE is not None:
            return
        from paddleocr import PaddleOCR
        _print(f"[OCR-GPU] Initializing {pool_size} PaddleOCR instances on GPU {gpu_id}...")
        q: queue.Queue = queue.Queue(maxsize=pool_size)
        for i in range(pool_size):
            ocr = PaddleOCR(
                text_detection_model_name='PP-OCRv5_server_det',
                text_det_box_thresh=0.3,
                text_det_unclip_ratio=2.0,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_angle_cls=False,
                device=f'gpu:{gpu_id}',
            )
            q.put(ocr)
            _print(f"[OCR-GPU] Instance {i+1}/{pool_size} ready.")
        _GPU_OCR_QUEUE = q
        _print(f"[OCR-GPU] Pool ready: {pool_size} instances on GPU {gpu_id}.")


def _run_gpu_ocr(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Run OCR using GPU pool. Returns [{id, text, box, score}]."""
    assert _GPU_OCR_QUEUE is not None, "GPU OCR pool not initialized"
    ocr = _GPU_OCR_QUEUE.get()
    try:
        results = ocr.ocr(img_bgr)
    finally:
        _GPU_OCR_QUEUE.put(ocr)

    rec_boxes = results[0].get('rec_polys', [])
    rec_texts = results[0].get('rec_texts', [])
    rec_scores = results[0].get('rec_scores', [])

    items = []
    for idx, (box, text, score) in enumerate(zip(rec_boxes, rec_texts, rec_scores)):
        try:
            box_py = np.asarray(box).tolist()
        except Exception:
            try:
                box_py = list(box)
            except Exception:
                box_py = []
        items.append({"id": idx, "text": text, "box": box_py, "score": float(score)})
    return items


def _quad_to_aabb(b) -> List[int]:
    """Convert polygon box to axis-aligned bounding box [x1, y1, x2, y2]."""
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
        return [int(x) for x in b]
    xs = [p[0] for p in b]
    ys = [p[1] for p in b]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


# ---------------------------------------------------------------------------
# GT text element extraction (from figma JSON)
# ---------------------------------------------------------------------------

def _build_gt_text_elements(gt_json_path: Path) -> List[Dict[str, Any]]:
    """Parse figma JSON and return all GT text elements.

    Returns list of {gt_id, text_content, font_size, bbox, word_count}.
    """
    try:
        frame = load_json(gt_json_path)
    except Exception:
        return []

    frame_ox = float(frame.get("render_x", 0) or 0)
    frame_oy = float(frame.get("render_y", 0) or 0)

    result: List[Dict[str, Any]] = []
    for unit in frame.get("unit_images", []):
        text_content = str(unit.get("text_content", "") or "").strip()
        if not text_content:
            continue
        text_content = preprocess_text_content(text_content)
        if not text_content:
            continue
        unit_id = str(unit.get("unit_id", ""))
        font_size = unit.get("font_size")
        rx = float(unit.get("render_x", 0) or 0) - frame_ox
        ry = float(unit.get("render_y", 0) or 0) - frame_oy
        rw = float(unit.get("render_width", 0) or 0)
        rh = float(unit.get("render_height", 0) or 0)
        bbox = [rx, ry, rx + rw, ry + rh]
        words = _WORD_RE.findall(text_content)
        result.append({
            "gt_id": f"gt_{unit_id}",
            "text_content": text_content,
            "font_size": float(font_size) if isinstance(font_size, (int, float)) else None,
            "bbox": bbox,
            "word_count": len(words),
        })
    return result


# ---------------------------------------------------------------------------
# Pred text extraction helpers
# ---------------------------------------------------------------------------

def _extract_pred_texts_agent(episode_dir: Path) -> List[Dict[str, Any]]:
    """Read parse.json → filter type=='text' → list of {content, bbox}."""
    parse_file = episode_dir / "parse.json"
    if not parse_file.exists():
        return []
    try:
        data = load_json(parse_file)
    except Exception:
        return []

    # parse.json can be {elements: [...]} or a flat list
    if isinstance(data, dict):
        elements = data.get("elements", [])
    elif isinstance(data, list):
        elements = data
    else:
        return []

    texts: List[Dict[str, Any]] = []
    for elem in elements:
        if elem.get("type") != "text":
            continue
        raw = str(elem.get("content", "") or "").strip()
        content = preprocess_text_content(raw)
        if not content:
            continue
        bbox = elem.get("bbox", [0, 0, 0, 0])
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            bbox = [0, 0, 0, 0]
        texts.append({
            "content": content,
            "bbox": [float(b) for b in bbox[:4]],
        })
    return texts


def _reconstruct_and_ocr(
    pred_dir: Path,
    ep_output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load pre-computed reconstructed.png → GPU OCR → save viz + pred_texts.

    qwen/layered episodes already contain a reconstructed.png produced by the
    baseline pipeline.  We read it directly (no CCA re-composite) and run OCR.

    Returns list of {content, bbox=[x1,y1,x2,y2], score}.
    Also saves:
      - reconstructed.png  (copy of source, for convenience)
      - ocr_det.png        (visualization with OCR bboxes + labels)
    """
    # --- Load source reconstructed image ---
    src_recon = pred_dir / "reconstructed.png"
    if not src_recon.exists():
        return []
    scene_rgb_bgr = cv2.imread(str(src_recon), cv2.IMREAD_COLOR)
    if scene_rgb_bgr is None:
        return []

    # Copy reconstructed image to output directory
    if ep_output_dir is not None:
        ep_output_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(src_recon), str(ep_output_dir / "reconstructed.png"))

    # --- Run GPU OCR ---
    ocr_items = _run_gpu_ocr(scene_rgb_bgr)

    # --- Build pred_texts list (same format as agent) ---
    pred_texts: List[Dict[str, Any]] = []
    boxes_for_viz, texts_for_viz, scores_for_viz = [], [], []

    for it in ocr_items:
        text = preprocess_text_content(str(it.get("text", "")))
        if not text:
            continue
        score = float(it.get("score", 0.0))
        if score <= 0.5:
            continue
        box = it.get("box", [])
        bbox = _quad_to_aabb(box) if box else [0, 0, 0, 0]
        pred_texts.append({
            "content": text,
            "bbox": [float(b) for b in bbox],
            "score": score,
        })
        boxes_for_viz.append(box)
        texts_for_viz.append(text)
        scores_for_viz.append(score)

    # --- Save OCR visualization ---
    if ep_output_dir is not None and boxes_for_viz:
        viz_img = scene_rgb_bgr.copy()
        for rbox, txt, sc in zip(boxes_for_viz, texts_for_viz, scores_for_viz):
            x1, y1, x2, y2 = _quad_to_aabb(rbox)
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{txt} ({sc:.2f})"
            cv2.putText(viz_img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(str(ep_output_dir / "ocr_det.png"), viz_img)

    return pred_texts


def _rasterize_svg_and_ocr(
    pred_dir: Path,
    ep_output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Rasterize output.svg → PNG → GPU OCR → pred texts.

    For omnisvg/vtracer models that produce output.svg per episode.
    Returns list of {content, bbox=[x1,y1,x2,y2], score}.
    """
    svg_path = pred_dir / "output.svg"
    if not svg_path.exists():
        return []

    # Rasterize SVG to PNG using cairosvg
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(url=str(svg_path))
    except Exception as e:
        _print(f"  [SVG] Failed to rasterize {svg_path}: {e}")
        return []

    # Decode PNG bytes to OpenCV image
    img_array = np.frombuffer(png_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        _print(f"  [SVG] Failed to decode rasterized PNG for {svg_path}")
        return []

    # Save rasterized image
    if ep_output_dir is not None:
        ep_output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ep_output_dir / "reconstructed.png"), img_bgr)

    # Run GPU OCR
    ocr_items = _run_gpu_ocr(img_bgr)

    # Build pred_texts list
    pred_texts: List[Dict[str, Any]] = []
    boxes_for_viz, texts_for_viz, scores_for_viz = [], [], []

    for it in ocr_items:
        text = preprocess_text_content(str(it.get("text", "")))
        if not text:
            continue
        score = float(it.get("score", 0.0))
        if score <= 0.5:
            continue
        box = it.get("box", [])
        bbox = _quad_to_aabb(box) if box else [0, 0, 0, 0]
        pred_texts.append({
            "content": text,
            "bbox": [float(b) for b in bbox],
            "score": score,
        })
        boxes_for_viz.append(box)
        texts_for_viz.append(text)
        scores_for_viz.append(score)

    # Save OCR visualization
    if ep_output_dir is not None and boxes_for_viz:
        viz_img = img_bgr.copy()
        for rbox, txt, sc in zip(boxes_for_viz, texts_for_viz, scores_for_viz):
            x1, y1, x2, y2 = _quad_to_aabb(rbox)
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{txt} ({sc:.2f})"
            cv2.putText(viz_img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(str(ep_output_dir / "ocr_det.png"), viz_img)

    return pred_texts


# ---------------------------------------------------------------------------
# Sort texts by reading order
# ---------------------------------------------------------------------------

def _reading_order_key(item: Dict[str, Any]) -> Tuple[float, float]:
    """Sort key for reading order: (y_center, x_center)."""
    bbox = item.get("bbox", [0, 0, 0, 0])
    y_center = (bbox[1] + bbox[3]) / 2
    x_center = (bbox[0] + bbox[2]) / 2
    return (y_center, x_center)


def _sorted_concat_text(text_items: List[Dict[str, Any]]) -> str:
    """Sort text items by reading order and concatenate contents."""
    sorted_items = sorted(text_items, key=_reading_order_key)
    return " ".join(normalize_text(it["content"]) for it in sorted_items if it.get("content"))


# ---------------------------------------------------------------------------
# Phase 1: Extract pred texts (all episodes for one model)
# ---------------------------------------------------------------------------

def extract_pred_texts_for_model(
    *,
    model_name: str,
    model_format: str,
    common_info: Dict[str, Dict[str, Any]],
    output_dir: Optional[Path] = None,
    min_text_len: int = 2,
    pbar=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract pred text elements for all episodes of one model.

    Returns {episode_id: [{content, bbox, ...}]}.
    Saves to output_dir/model_name/episodes/{eid}/pred_texts.json.
    For OCR models, also saves reconstructed.png and ocr_det.png.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}
    is_agent = (model_format == "agent")
    is_omnisvg = (model_format == "omnisvg")

    for eid in sorted(common_info.keys()):
        info = common_info[eid]
        ep_out_dir = (output_dir / model_name / "episodes" / eid) if output_dir else None

        # Check cache: pred_texts.json must exist.
        # For OCR/SVG models, also require reconstructed.png.
        pred_texts = None
        if ep_out_dir is not None:
            cache_file = ep_out_dir / "pred_texts.json"
            if cache_file.exists():
                cache_valid = True
                if not is_agent:
                    # OCR/SVG models must also have viz files
                    if not (ep_out_dir / "reconstructed.png").exists():
                        cache_valid = False
                if cache_valid:
                    try:
                        pred_texts = load_json(cache_file)
                    except Exception:
                        pred_texts = None

        # Extract if not cached
        if pred_texts is None:
            if is_agent:
                pred_texts = _extract_pred_texts_agent(info["model_dir"])
            elif is_omnisvg:
                pred_texts = _rasterize_svg_and_ocr(info["model_dir"], ep_out_dir)
            else:
                pred_texts = _reconstruct_and_ocr(info["model_dir"], ep_out_dir)

            # Save pred_texts.json
            if ep_out_dir is not None:
                ep_out_dir.mkdir(parents=True, exist_ok=True)
                save_json(ep_out_dir / "pred_texts.json", pred_texts)

        # Filter out short texts (separators like |, _, —)
        if min_text_len > 0:
            pred_texts = [t for t in pred_texts if len(t.get("content", "")) >= min_text_len]

        results[eid] = pred_texts

        if pbar:
            pbar.update(1)

    return results


# ---------------------------------------------------------------------------
# Phase 2: Content Recognition (LCS-based recall per episode)
# ---------------------------------------------------------------------------

def run_content_recognition(
    *,
    model_name: str,
    gt_texts_per_episode: Dict[str, List[Dict[str, Any]]],
    pred_texts_per_episode: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Compute per-episode recognition metrics via sorted text concatenation.

    Metrics per episode:
      char_recall / word_recall — LCS-based recall (FN-only, ignores FP)
    """
    rows: List[Dict[str, Any]] = []

    for eid in sorted(gt_texts_per_episode.keys()):
        gt_texts = gt_texts_per_episode[eid]
        pred_texts = pred_texts_per_episode.get(eid, [])

        if not gt_texts:
            continue

        # Sort by reading order and concatenate
        gt_items = [{"content": g["text_content"], "bbox": g["bbox"]} for g in gt_texts]
        gt_str = _sorted_concat_text(gt_items)
        pred_str = _sorted_concat_text(pred_texts)

        if not gt_str:
            continue

        c_recall = compute_char_recall(gt_str, pred_str)
        w_recall = compute_word_recall(gt_str, pred_str)

        rows.append({
            "episode_id": eid,
            "gt_text_count": len(gt_texts),
            "pred_text_count": len(pred_texts),
            "gt_str": gt_str,
            "pred_str": pred_str,
            "char_recall": c_recall,
            "word_recall": w_recall,
        })

    # Save detailed results
    checkpoint_path = output_dir / model_name / "content_recognition_results.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(checkpoint_path, rows)

    # Aggregate
    if rows:
        n = len(rows)
        avg_char_recall = sum(r["char_recall"] for r in rows) / n
        avg_word_recall = sum(r["word_recall"] for r in rows) / n
    else:
        avg_char_recall = avg_word_recall = float("nan")

    summary = {
        "count": len(rows),
        "avg_char_recall": avg_char_recall,
        "avg_word_recall": avg_word_recall,
    }
    save_json(output_dir / model_name / "content_recognition_summary.json", summary)

    return {"results": rows, "summary": summary}


# ---------------------------------------------------------------------------
# Phase 3: Content Modification (word replacement + LCS recall)
# ---------------------------------------------------------------------------

def _build_episode_modification_plan(
    gt_texts: List[Dict[str, Any]],
    episode_id: str,
    seed: int,
    n_mods: int = 5,
    min_word_len: int = 3,
) -> Dict[str, str]:
    """Select words from smaller-font GT texts and generate random replacements.

    Returns {original_word: replacement_word} mapping.
    All strings are normalized (lowercase, whitespace-collapsed).
    """
    rng = random.Random(f"{seed}_{episode_id}")

    # Compute median font size
    font_sizes = [float(g["font_size"]) for g in gt_texts if g.get("font_size") is not None]
    if not font_sizes:
        return {}
    med_fs = _median(font_sizes)

    # Pool: unique words from GT texts with font_size ≤ median, word length ≥ min_word_len
    pool: Set[str] = set()
    for g in gt_texts:
        fs = g.get("font_size")
        if fs is None or float(fs) > med_fs:
            continue
        normalized = normalize_text(g["text_content"])
        for word in normalized.split():
            if len(word) >= min_word_len:
                pool.add(word)

    if not pool:
        # Fallback: try all GT texts regardless of font size
        for g in gt_texts:
            normalized = normalize_text(g["text_content"])
            for word in normalized.split():
                if len(word) >= min_word_len:
                    pool.add(word)

    if not pool:
        return {}

    pool_list = sorted(pool)  # deterministic ordering
    selected = rng.sample(pool_list, min(n_mods, len(pool_list)))

    # Generate random replacement for each word (same length, lowercase letters)
    replacements: Dict[str, str] = {}
    for word in selected:
        replacement = "".join(rng.choices(string.ascii_lowercase, k=len(word)))
        # Ensure replacement differs from original
        while replacement == word:
            replacement = "".join(rng.choices(string.ascii_lowercase, k=len(word)))
        replacements[word] = replacement

    return replacements


def _apply_word_replacements(text_str: str, replacements: Dict[str, str]) -> str:
    """Whole-word find-and-replace on normalized text string."""
    words = text_str.split()
    return " ".join(replacements.get(w, w) for w in words)


def run_content_modification(
    *,
    model_name: str,
    gt_texts_per_episode: Dict[str, List[Dict[str, Any]]],
    pred_texts_per_episode: Dict[str, List[Dict[str, Any]]],
    seed: int,
    n_mods: int = 5,
    output_dir: Path,
) -> Dict[str, Any]:
    """Apply deterministic word replacements and measure LCS-based recall.

    Per episode:
      1. Build replacement plan from GT (same for all models)
      2. Apply replacements to both GT and pred concatenated strings
      3. LCS recall between modified GT and modified pred
    """
    rows: List[Dict[str, Any]] = []

    for eid in sorted(gt_texts_per_episode.keys()):
        gt_texts = gt_texts_per_episode[eid]
        pred_texts = pred_texts_per_episode.get(eid, [])

        if not gt_texts:
            continue

        gt_items = [{"content": g["text_content"], "bbox": g["bbox"]} for g in gt_texts]
        gt_str = _sorted_concat_text(gt_items)
        pred_str = _sorted_concat_text(pred_texts)

        if not gt_str:
            continue

        # Build modification plan (deterministic, same for all models)
        replacements = _build_episode_modification_plan(
            gt_texts, eid, seed, n_mods=n_mods,
        )

        if not replacements:
            continue

        # Apply replacements
        modified_gt = _apply_word_replacements(gt_str, replacements)
        modified_pred = _apply_word_replacements(pred_str, replacements)

        # Metrics
        mod_char_recall = compute_char_recall(modified_gt, modified_pred)
        mod_word_recall = compute_word_recall(modified_gt, modified_pred)

        # Also compute base recognition metrics for comparison
        base_char_recall = compute_char_recall(gt_str, pred_str)
        base_word_recall = compute_word_recall(gt_str, pred_str)

        # Count how many replacements actually applied to pred
        pred_words = set(pred_str.split())
        n_applied_to_pred = sum(1 for w in replacements if w in pred_words)

        rows.append({
            "episode_id": eid,
            "gt_text_count": len(gt_texts),
            "pred_text_count": len(pred_texts),
            "n_replacements": len(replacements),
            "n_applied_to_pred": n_applied_to_pred,
            "replacements": replacements,
            "gt_str": gt_str,
            "pred_str": pred_str,
            "modified_gt_str": modified_gt,
            "modified_pred_str": modified_pred,
            "base_char_recall": base_char_recall,
            "base_word_recall": base_word_recall,
            "mod_char_recall": mod_char_recall,
            "mod_word_recall": mod_word_recall,
        })

    # Save detailed results
    checkpoint_path = output_dir / model_name / "content_modification_results.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(checkpoint_path, rows)

    # Aggregate
    if rows:
        avg_mod_char_recall = sum(r["mod_char_recall"] for r in rows) / len(rows)
        avg_mod_word_recall = sum(r["mod_word_recall"] for r in rows) / len(rows)
        avg_base_char_recall = sum(r["base_char_recall"] for r in rows) / len(rows)
        avg_base_word_recall = sum(r["base_word_recall"] for r in rows) / len(rows)
        avg_n_applied = sum(r["n_applied_to_pred"] for r in rows) / len(rows)
        avg_n_replacements = sum(r["n_replacements"] for r in rows) / len(rows)
    else:
        avg_mod_char_recall = avg_mod_word_recall = float("nan")
        avg_base_char_recall = avg_base_word_recall = float("nan")
        avg_n_applied = avg_n_replacements = 0

    summary = {
        "count": len(rows),
        "avg_mod_char_recall": avg_mod_char_recall,
        "avg_mod_word_recall": avg_mod_word_recall,
        "avg_base_char_recall": avg_base_char_recall,
        "avg_base_word_recall": avg_base_word_recall,
        "avg_n_applied_to_pred": avg_n_applied,
        "avg_n_replacements": avg_n_replacements,
    }
    save_json(output_dir / model_name / "content_modification_summary.json", summary)

    return {"results": rows, "summary": summary}


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

def _build_cross_model_comparison(
    recognition_payloads: Dict[str, Dict[str, Any]],
    modification_payloads: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}

    comparison["content_recognition"] = {}
    for key in ["avg_char_recall", "avg_word_recall"]:
        comparison["content_recognition"][key] = {
            "preference": "higher",
            **{m: p.get("summary", {}).get(key) for m, p in recognition_payloads.items()},
        }

    comparison["content_modification"] = {}
    for key in ["avg_mod_char_recall", "avg_mod_word_recall",
                "avg_base_char_recall", "avg_base_word_recall",
                "avg_n_applied_to_pred"]:
        comparison["content_modification"][key] = {
            "preference": "higher",
            **{m: p.get("summary", {}).get(key) for m, p in modification_payloads.items()},
        }

    return comparison


# ---------------------------------------------------------------------------
# Visualization: per-episode image grids (GT + each model)
# ---------------------------------------------------------------------------

def _get_gt_image_path(common_info_entry: Dict[str, Any]) -> Optional[Path]:
    """Derive the GT reconstructed image path from a common_info entry.

    Works for both the merged/flat dataset (split_dir == figma_data root) and the
    legacy per-split layout. The canonical reconstruction lives inside the
    episode's unit_images dir (collision-free); flat convenience copies and the
    legacy filename are tried as fallbacks.
    """
    gt_json_path = Path(common_info_entry["gt_json_path"])
    split_dir = Path(common_info_entry["split_dir"])
    episode_id = gt_json_path.stem
    try:
        gt_json = load_json(gt_json_path)
    except Exception:
        gt_json = {}

    candidates = []
    # 1. canonical: <root>/<unit_images_dir>/<reconstructed_image_path>
    uid = gt_json.get("unit_images_dir")
    rp = gt_json.get("reconstructed_image_path")
    if uid and rp:
        candidates.append(split_dir / uid / rp)
    # 2. merged flat convenience copy keyed by episode id
    candidates.append(split_dir / "reconstructed_images" / f"{episode_id}.png")
    # 3. legacy per-split filename keyed by frame id
    frame_id = gt_json.get("frame_id", "")
    if frame_id:
        candidates.append(split_dir / "reconstructed_images" / f"_reconstructed_{frame_id.replace(':', '_')}.png")

    for c in candidates:
        if c.exists():
            return c
    return None


def _make_label_bar(text: str, width: int, bar_height: int = 40) -> np.ndarray:
    """Create a label bar image with centered text."""
    bar = np.ones((bar_height, width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = max(0, (width - tw) // 2)
    ty = (bar_height + th) // 2
    cv2.putText(bar, text, (tx, ty), font, font_scale, (0, 0, 0), thickness)
    return bar


def generate_episode_grids(
    *,
    models_order: List[str],
    model_infos: Dict[str, Dict[str, Any]],
    recognition_payloads: Dict[str, Dict[str, Any]],
    modification_payloads: Dict[str, Dict[str, Any]],
    output_dir: Path,
    seed: int,
    n_samples: int = 100,
) -> None:
    """Generate per-episode 1×(1+N_models) image grids for recognition & modification.

    Each grid: GT | model_1 | model_2 | ... with label + score annotation.
    """
    # Build per-episode score lookups
    recog_scores: Dict[str, Dict[str, Dict[str, float]]] = {}  # {model: {eid: {metric: val}}}
    mod_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m in models_order:
        recog_scores[m] = {}
        for row in recognition_payloads.get(m, {}).get("results", []):
            recog_scores[m][row["episode_id"]] = {
                "char_recall": row["char_recall"],
                "word_recall": row["word_recall"],
            }
        mod_scores[m] = {}
        for row in modification_payloads.get(m, {}).get("results", []):
            mod_scores[m][row["episode_id"]] = {
                "mod_char_recall": row["mod_char_recall"],
                "mod_word_recall": row["mod_word_recall"],
            }

    # Get common episode ids
    ref_model = models_order[0]
    all_eids = sorted(model_infos[ref_model]["common"].keys())

    # Sample episodes
    rng = random.Random(seed)
    sampled = rng.sample(all_eids, min(n_samples, len(all_eids)))
    sampled.sort()

    target_h = 512  # resize all images to this height
    bar_h = 40

    for task_name, scores_lookup, metric_keys in [
        ("recognition", recog_scores, ["char_recall", "word_recall"]),
        ("modification", mod_scores, ["mod_char_recall", "mod_word_recall"]),
    ]:
        viz_dir = output_dir / "visualization" / task_name
        viz_dir.mkdir(parents=True, exist_ok=True)

        for eid in sampled:
            # Load GT image
            info = model_infos[ref_model]["common"].get(eid)
            if info is None:
                continue
            gt_img_path = _get_gt_image_path(info)
            if gt_img_path is None:
                continue
            gt_img = cv2.imread(str(gt_img_path), cv2.IMREAD_COLOR)
            if gt_img is None:
                continue

            # Resize GT to target height
            scale = target_h / gt_img.shape[0]
            gt_resized = cv2.resize(gt_img, (int(gt_img.shape[1] * scale), target_h))
            col_w = gt_resized.shape[1]

            # Build columns: GT + each model
            columns = []

            # GT column
            gt_label = _make_label_bar("GT", col_w, bar_h)
            columns.append(np.vstack([gt_label, gt_resized]))

            # Model columns
            for m in models_order:
                m_info = model_infos[m]["common"].get(eid)
                if m_info is None:
                    # Blank column
                    blank = np.ones((target_h, col_w, 3), dtype=np.uint8) * 200
                    label = _make_label_bar(f"{m} (N/A)", col_w, bar_h)
                    columns.append(np.vstack([label, blank]))
                    continue

                recon_path = m_info["model_dir"] / "reconstructed.png"
                # For SVG models, reconstructed.png is in output_dir
                if not recon_path.exists():
                    alt_path = output_dir / m / "episodes" / eid / "reconstructed.png"
                    if alt_path.exists():
                        recon_path = alt_path
                if not recon_path.exists():
                    blank = np.ones((target_h, col_w, 3), dtype=np.uint8) * 200
                    label = _make_label_bar(f"{m} (no img)", col_w, bar_h)
                    columns.append(np.vstack([label, blank]))
                    continue

                m_img = cv2.imread(str(recon_path), cv2.IMREAD_COLOR)
                if m_img is None:
                    blank = np.ones((target_h, col_w, 3), dtype=np.uint8) * 200
                    label = _make_label_bar(f"{m} (err)", col_w, bar_h)
                    columns.append(np.vstack([label, blank]))
                    continue

                m_resized = cv2.resize(m_img, (col_w, target_h))

                # Build label with scores
                ep_scores = scores_lookup.get(m, {}).get(eid, {})
                if ep_scores:
                    vals = [f"{k}={v:.2f}" for k, v in ep_scores.items()]
                    label_text = f"{m}  {' '.join(vals)}"
                else:
                    label_text = m
                label = _make_label_bar(label_text, col_w, bar_h)
                columns.append(np.vstack([label, m_resized]))

            # Concatenate horizontally
            grid = np.hstack(columns)
            out_path = viz_dir / f"{eid}.png"
            cv2.imwrite(str(out_path), grid)

        _print(f"  [{task_name}] Saved {len(sampled)} episode grids to {viz_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text editability evaluation — content-based matching pipeline"
    )
    parser.add_argument("--figma-data", type=str, required=True,
                        help="Path to the downloaded figma_data dataset directory.")
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="Models to evaluate (e.g., agent multi_tools layered qwen)",
    )
    add_baseline_dir_args(parser)
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results "
                             "(default: ./eval_text_edit_baselines).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--n-mods", type=int, default=4,
                        help="Number of word replacements per episode for modification task")
    parser.add_argument("--min-pred-text-len", type=int, default=2,
                        help="Minimum pred text length (filters separator chars like | _ —)")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--ocr-gpu", type=int, default=0,
                        help="GPU id for PaddleOCR; set to your own (default: 0).")
    parser.add_argument("--ocr-pool-size", type=int, default=4,
                        help="Number of PaddleOCR instances on GPU (default: 4)")
    parser.add_argument("--no-tqdm", action="store_true", default=False)
    parser.add_argument("--exclude-episodes-file", type=str, default=None,
                        help="Text file with one episode ID per line to exclude")
    args = parser.parse_args()

    figma_data = Path(args.figma_data)
    output_dir = Path(args.output) if args.output else Path("eval_text_edit_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    for m in args.models:
        if m not in MODEL_CONFIGS:
            parser.error(f"Unknown model: {m}. Available: {list(MODEL_CONFIGS.keys())}")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    models_order = [m for m in ALL_TEXT_MODELS if m in args.models]
    for m in args.models:
        if m not in models_order:
            models_order.append(m)

    _print(
        f"[setup] models={models_order} seed={args.seed} "
        f"n_mods={args.n_mods} num_workers={args.num_workers} "
        f"ocr_gpu={args.ocr_gpu} ocr_pool_size={args.ocr_pool_size}"
    )

    # ===================================================================
    # Collect GT episodes and build per-model info
    # ===================================================================
    gt_map = collect_gt_episodes(figma_data)
    _print(f"[setup] GT episodes: {len(gt_map)}")

    model_infos: Dict[str, Dict[str, Any]] = {}
    for model_name in models_order:
        multi_dirs = _resolve_multi_dirs(args, model_name)
        if multi_dirs is not None:
            model_map = scan_model_episodes_multi(model_name, multi_dirs)
        else:
            base_dir = get_model_dir(args, model_name)
            model_map = scan_model_episodes(model_name, base_dir)

        model_format = MODEL_CONFIGS[model_name]["format"]
        common = get_common_episodes(gt_map, model_map)

        common_ids = sorted(common.keys())
        if args.max_episodes is not None:
            common_ids = common_ids[: args.max_episodes]
        common = {eid: common[eid] for eid in common_ids}

        model_infos[model_name] = {
            "common": common,
            "model_format": model_format,
        }
        _print(f"[setup] {model_name}: episodes={len(common)} format={model_format}")

    # Common episodes across all models
    common_episode_ids: Optional[Set[str]] = None
    for m in models_order:
        eids = set(model_infos[m]["common"].keys())
        if common_episode_ids is None:
            common_episode_ids = eids
        else:
            common_episode_ids = common_episode_ids & eids
    common_episode_ids_sorted = sorted(common_episode_ids) if common_episode_ids else []
    _print(f"[setup] Common episodes across all models: {len(common_episode_ids_sorted)}")

    # Apply episode exclusion list
    if args.exclude_episodes_file:
        exclude_path = Path(args.exclude_episodes_file)
        exclude_ids = set(
            line.strip() for line in exclude_path.read_text().splitlines() if line.strip()
        )
        before = len(common_episode_ids_sorted)
        common_episode_ids_sorted = [eid for eid in common_episode_ids_sorted if eid not in exclude_ids]
        _print(f"[setup] Excluded {before - len(common_episode_ids_sorted)} episodes "
               f"({len(common_episode_ids_sorted)} remaining) via {exclude_path.name}")

    # Restrict each model to common episodes
    for m in models_order:
        model_infos[m]["common"] = {
            eid: model_infos[m]["common"][eid]
            for eid in common_episode_ids_sorted
            if eid in model_infos[m]["common"]
        }

    # ===================================================================
    # Build GT text elements (shared across all models)
    # ===================================================================
    _print("\n" + "=" * 70)
    _print("BUILDING GT TEXT ELEMENTS")
    _print("=" * 70)

    gt_texts_per_episode: Dict[str, List[Dict[str, Any]]] = {}
    total_gt_texts = 0
    any_info = model_infos[models_order[0]]["common"]
    for eid in common_episode_ids_sorted:
        info = any_info[eid]
        gt_texts = _build_gt_text_elements(info["gt_json_path"])
        gt_texts_per_episode[eid] = gt_texts
        total_gt_texts += len(gt_texts)

    _print(f"[setup] Total GT text elements: {total_gt_texts} across {len(common_episode_ids_sorted)} episodes")

    # ===================================================================
    # Phase 1: Extract pred texts per model
    # ===================================================================
    _print("\n" + "=" * 70)
    _print("PHASE 1: PRED TEXT EXTRACTION")
    _print("=" * 70)

    has_ocr_format = any(
        model_infos[m]["model_format"] in ("qwen", "omnisvg") for m in models_order
    )
    if has_ocr_format:
        _init_gpu_ocr_pool(gpu_id=args.ocr_gpu, pool_size=args.ocr_pool_size)

    use_tqdm = bool(not args.no_tqdm and _tqdm is not None)
    tqdm_write = _tqdm.write if _tqdm is not None else _print

    all_pred_texts: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    n_eps = len(common_episode_ids_sorted)

    # Process models that need OCR (qwen-format) sequentially (they share OCR clients)
    # Process agent-format models in parallel (they just read JSON)
    agent_models = [m for m in models_order if model_infos[m]["model_format"] == "agent"]
    ocr_models = [m for m in models_order if model_infos[m]["model_format"] in ("qwen", "omnisvg")]

    # Agent models: fast, parallel
    for model_name in agent_models:
        info = model_infos[model_name]
        pbar = _tqdm(total=n_eps, desc=f"extract/{model_name}") if use_tqdm else None
        pred_texts = extract_pred_texts_for_model(
            model_name=model_name,
            model_format=info["model_format"],
            common_info=info["common"],
            output_dir=output_dir,
            min_text_len=0,
            pbar=pbar,
        )
        all_pred_texts[model_name] = pred_texts
        if pbar:
            pbar.close()
        n_total = sum(len(v) for v in pred_texts.values())
        tqdm_write(f"  {model_name}: {n_eps} episodes, {n_total} pred text elements")

    # OCR models: read source reconstructed.png → GPU OCR
    for model_name in ocr_models:
        info = model_infos[model_name]
        pbar = _tqdm(total=n_eps, desc=f"extract/{model_name}") if use_tqdm else None
        pred_texts = extract_pred_texts_for_model(
            model_name=model_name,
            model_format=info["model_format"],
            common_info=info["common"],
            output_dir=output_dir,
            min_text_len=0,
            pbar=pbar,
        )
        all_pred_texts[model_name] = pred_texts
        if pbar:
            pbar.close()
        n_total = sum(len(v) for v in pred_texts.values())
        tqdm_write(f"  {model_name}: {n_eps} episodes, {n_total} pred text elements")

    # ===================================================================
    # Phase 2: Content Recognition
    # ===================================================================
    _print("\n" + "=" * 70)
    _print("PHASE 2: CONTENT RECOGNITION (LCS-based recall)")
    _print("=" * 70)

    recognition_payloads: Dict[str, Dict[str, Any]] = {}
    for model_name in models_order:
        try:
            payload = run_content_recognition(
                model_name=model_name,
                gt_texts_per_episode=gt_texts_per_episode,
                pred_texts_per_episode=all_pred_texts[model_name],
                output_dir=output_dir,
            )
            recognition_payloads[model_name] = payload
            s = payload["summary"]
            tqdm_write(
                f"  {model_name}: episodes={s['count']} "
                f"char_recall={s['avg_char_recall']:.4f} word_recall={s['avg_word_recall']:.4f}"
            )
        except Exception as e:
            tqdm_write(f"  {model_name}: FAILED — {e}")
            traceback.print_exc()

    # ===================================================================
    # Phase 3: Content Modification
    # ===================================================================
    _print("\n" + "=" * 70)
    _print("PHASE 3: CONTENT MODIFICATION (word replacement + LCS recall)")
    _print("=" * 70)

    min_pred_text_len = args.min_pred_text_len
    modification_payloads: Dict[str, Dict[str, Any]] = {}
    for model_name in models_order:
        try:
            # Apply min_pred_text_len filter only for modification task
            mod_pred_texts = {}
            for eid, texts in all_pred_texts[model_name].items():
                if min_pred_text_len > 0:
                    mod_pred_texts[eid] = [t for t in texts if len(t.get("content", "")) >= min_pred_text_len]
                else:
                    mod_pred_texts[eid] = texts
            payload = run_content_modification(
                model_name=model_name,
                gt_texts_per_episode=gt_texts_per_episode,
                pred_texts_per_episode=mod_pred_texts,
                seed=args.seed,
                n_mods=args.n_mods,
                output_dir=output_dir,
            )
            modification_payloads[model_name] = payload
            s = payload["summary"]
            tqdm_write(
                f"  {model_name}: episodes={s['count']} "
                f"mod_C_Recall={s['avg_mod_char_recall']:.4f} mod_W_Recall={s['avg_mod_word_recall']:.4f} "
                f"applied={s['avg_n_applied_to_pred']:.1f}/{s['avg_n_replacements']:.1f}"
            )
        except Exception as e:
            tqdm_write(f"  {model_name}: FAILED — {e}")
            traceback.print_exc()

    # ===================================================================
    # Phase 4: Visualization (per-episode image grids)
    # ===================================================================
    _print("\n" + "=" * 70)
    _print("PHASE 4: VISUALIZATION (per-episode image grids)")
    _print("=" * 70)

    generate_episode_grids(
        models_order=models_order,
        model_infos=model_infos,
        recognition_payloads=recognition_payloads,
        modification_payloads=modification_payloads,
        output_dir=output_dir,
        seed=args.seed,
        n_samples=100,
    )

    # ===================================================================
    # Save cross-model comparison
    # ===================================================================
    comparison = _build_cross_model_comparison(recognition_payloads, modification_payloads)
    save_json(output_dir / "text_comparison_all_models.json", comparison)

    for model_name in models_order:
        overview: Dict[str, Any] = {}
        if model_name in recognition_payloads:
            overview["content_recognition"] = recognition_payloads[model_name]["summary"]
        if model_name in modification_payloads:
            overview["content_modification"] = modification_payloads[model_name]["summary"]
        save_json(output_dir / f"text_{model_name}_overview.json", overview)

    # ===================================================================
    # Final summary table
    # ===================================================================
    _print("\n" + "=" * 70)
    _print("FINAL SUMMARY")
    _print("=" * 70)

    _print(f"\n[Content Recognition] ({len(common_episode_ids_sorted)} episodes)")
    header = f"  {'Model':<15} {'Episodes':>8} {'C_Recall':>9} {'W_Recall':>9}"
    _print(header)
    _print("  " + "-" * (len(header) - 2))
    for m in models_order:
        s = recognition_payloads.get(m, {}).get("summary", {})
        count = s.get("count", 0)
        cr = s.get("avg_char_recall", float("nan"))
        wr = s.get("avg_word_recall", float("nan"))
        _print(f"  {m:<15} {count:>8} {cr:>9.4f} {wr:>9.4f}")

    _print(f"\n[Content Modification] ({args.n_mods} word replacements/episode)")
    header = (f"  {'Model':<15} {'Episodes':>8} "
              f"{'mC_Recall':>10} {'mW_Recall':>10} {'applied':>9}")
    _print(header)
    _print("  " + "-" * (len(header) - 2))
    for m in models_order:
        s = modification_payloads.get(m, {}).get("summary", {})
        count = s.get("count", 0)
        mcr = s.get("avg_mod_char_recall", float("nan"))
        mwr = s.get("avg_mod_word_recall", float("nan"))
        applied = s.get("avg_n_applied_to_pred", 0)
        n_rep = s.get("avg_n_replacements", 0)
        _print(f"  {m:<15} {count:>8} "
               f"{mcr:>10.4f} {mwr:>10.4f} {applied:>4.1f}/{n_rep:.1f}")

    # Per-model gap analysis (agent vs each baseline)
    ref_model = "agent"
    if ref_model in models_order:
        _print(f"\n[Per-Model Gap: agent vs each baseline]")

        # Recognition gaps
        ref_recog = recognition_payloads.get(ref_model, {}).get("summary", {})
        ref_cr = ref_recog.get("avg_char_recall", 0)
        ref_wr = ref_recog.get("avg_word_recall", 0)

        _print(f"\n  Recognition:")
        header = f"    {'Baseline':<15} {'C_Recall':>9} {'gap':>7} {'W_Recall':>9} {'gap':>7}"
        _print(header)
        _print("    " + "-" * (len(header) - 4))
        _print(f"    {'agent':<15} {ref_cr:>9.4f} {'(ref)':>7} {ref_wr:>9.4f} {'(ref)':>7}")
        for m in models_order:
            if m == ref_model:
                continue
            s = recognition_payloads.get(m, {}).get("summary", {})
            cr = s.get("avg_char_recall", 0)
            wr = s.get("avg_word_recall", 0)
            _print(f"    {m:<15} {cr:>9.4f} {ref_cr - cr:>+7.4f} {wr:>9.4f} {ref_wr - wr:>+7.4f}")

        # Modification gaps
        ref_mod = modification_payloads.get(ref_model, {}).get("summary", {})
        ref_mcr = ref_mod.get("avg_mod_char_recall", 0)
        ref_mwr = ref_mod.get("avg_mod_word_recall", 0)

        _print(f"\n  Modification:")
        header = f"    {'Baseline':<15} {'mC_Recall':>10} {'gap':>7} {'mW_Recall':>10} {'gap':>7}"
        _print(header)
        _print("    " + "-" * (len(header) - 4))
        _print(f"    {'agent':<15} {ref_mcr:>10.4f} {'(ref)':>7} {ref_mwr:>10.4f} {'(ref)':>7}")
        for m in models_order:
            if m == ref_model:
                continue
            s = modification_payloads.get(m, {}).get("summary", {})
            mcr = s.get("avg_mod_char_recall", 0)
            mwr = s.get("avg_mod_word_recall", 0)
            _print(f"    {m:<15} {mcr:>10.4f} {ref_mcr - mcr:>+7.4f} {ref_mwr - mwr:>10.4f} {ref_mwr - mwr:>+7.4f}")

    _print(f"\n[DONE] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
