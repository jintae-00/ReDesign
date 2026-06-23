#!/usr/bin/env python3
"""
run_multi_tools_crello.py - OCR + HiSAM + VLM + GDINO + SAM2 + LaMa on Crello Dataset

Same pipeline as run_multi_tools_figma.py but applied to the Crello test dataset (549 records).

Phase 1: OCR bbox -> HiSAM segmentation -> text RGBA -> LaMa inpaint (text union mask)
Phase 2: VLM Labeling -> GDINO detection -> SAM2 segmentation -> LaMa inpaint (repeat)

Output Format: parse.json + elements/ (Agent-compatible)
Evaluation: eval_accuracy_baselines_crello.py --multi-tools-dir <output_dir>

Processes EVERY crello_test_* record directory under <data_dir>.

Usage:
    python run_multi_tools_crello.py --data_dir crello_data/records --output_dir outputs/multi_tools_crello --gpu <GPU_IDS>
    python run_multi_tools_crello.py --data_dir crello_data/records --output_dir outputs/multi_tools_crello --gpu <GPU_IDS> --limit 10

    # Replace <GPU_IDS> with your own comma-separated GPU ids (e.g. 0 or 0,1).

Directory Structure:
    Input:
        <data_dir>/crello_test_XXXX/
            - composite.png

    Output:
        <output_dir>/episodes/{record_id}/
            - parse.json
            - history_tree.json
            - original_input.png
            - reconstructed.png
            - reconstructed_bordered.png
            - layers/layer_0000/layer_image.png
            - elements/
                ├── text_0000/
                │   ├── canvas_image.png  (RGBA, canvas size)
                │   ├── mask_canvas.png   (L, canvas size)
                │   └── metadata.json
                ├── obj_0001/
                │   ├── canvas_image.png
                │   ├── mask_canvas.png
                │   └── metadata.json
                └── ...
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import os
import gc
import io
import re
import logging
import multiprocessing as mp
import signal
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

MAX_OBJ_ITERATIONS = 4
ALPHA_THRESHOLD = 16
FRAME_TIMEOUT_SEC = 600  # 10 min timeout per frame

# GDINO detection
DINO_SCORE_MIN = 0.1

# SAM2 minimum mask area (pixels)
SAM2_MIN_MASK_PIXELS = 200

# Per-frame subprocess timeout (seconds)
FRAME_PROCESS_TIMEOUT_SEC = 660

# VLM labeling prompt
VLM_FRONT_ELEMS_PICK = r"""
You are the **Front-Most Elements Picker**.

Detect fully visible (not occluded) Objects in two phases (both visible):
- **PHASE A — THINK (VISIBLE)**: briefly reason step-by-step.
- **PHASE B — FINAL (VISIBLE)**: output exactly one JSON block at the very end:
  {"labels" : ["..."] }  OR  {"labels": []}

Guidelines:
1) **Z-ORDER IS IMPORTANT**: pick elements fully visible at the front-most layer. Discard occluded items.
2) Framing containers (background/panel/card/sheet) should be picked only when nothing else remains in front of the container.
3) Provide short, concrete label for each object consisting of 2 to 4 words depicting (color+category+shape).

Output the JSON only at the end.
"""


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_file: Path, name: str = "multi_tools_crello") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)

    return logger


# =============================================================================
# Path Resolution
# =============================================================================

# =============================================================================
# Record Data Loading (Crello)
# =============================================================================

@dataclass
class RecordInfo:
    record_id: str
    image_path: Path  # composite.png absolute path


def load_record_list(data_dir: Path) -> List[RecordInfo]:
    """Load every crello_test_* record directory under data_dir (split-agnostic).

    The input image for each record is ``<record_dir>/composite.png``.
    """
    records = []

    record_dirs = sorted([
        d for d in data_dir.iterdir()
        if (d.is_dir() or d.is_symlink()) and d.name.startswith("crello_test_")
    ])

    for record_dir in record_dirs:
        composite_path = record_dir / "composite.png"
        if composite_path.exists():
            records.append(RecordInfo(
                record_id=record_dir.name,
                image_path=composite_path.resolve(),
            ))
        else:
            print(f"[Warning] composite.png not found: {record_dir}")

    # Deduplicate by record_id and sort
    seen = set()
    unique = []
    for r in sorted(records, key=lambda x: x.record_id):
        if r.record_id not in seen:
            seen.add(r.record_id)
            unique.append(r)

    return unique


def is_record_completed(output_dir: Path, record_id: str) -> bool:
    return (output_dir / "episodes" / record_id / "parse.json").exists()


def is_record_failed(output_dir: Path, record_id: str) -> bool:
    return (output_dir / "episodes" / record_id / "_FAILED").exists()


def mark_record_failed(output_dir: Path, record_id: str, reason: str):
    fail_dir = output_dir / "episodes" / record_id
    fail_dir.mkdir(parents=True, exist_ok=True)
    with open(fail_dir / "_FAILED", 'w') as f:
        f.write(f"{datetime.now().isoformat()} | {reason}\n")


class _FrameTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _FrameTimeout("Frame processing timed out")


# =============================================================================
# Helper: quad_to_aabb
# =============================================================================

def _quad_to_aabb(b):
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
        return [int(x) for x in b]
    xs = [p[0] for p in b]
    ys = [p[1] for p in b]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


# =============================================================================
# VLM Labeling (Gemini 3 Flash)
# =============================================================================

def _resize_image_for_vlm(image_path: str, max_size: int = 1024) -> bytes:
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) > max_size:
                ratio = max_size / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        with open(image_path, "rb") as f:
            return f.read()


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                clean = match.strip()
                if not clean.startswith('{'):
                    start = clean.find('{')
                    end = clean.rfind('}')
                    if start >= 0 and end > start:
                        clean = clean[start:end+1]
                return json.loads(clean)
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def call_vlm_front_pick(image_path: str, logger: logging.Logger) -> List[str]:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    img_bytes = _resize_image_for_vlm(image_path)
    b64_str = base64.b64encode(img_bytes).decode("utf-8")

    llm = ChatOpenAI(
        model=os.environ.get("VLM_MODEL", "gemini-3-flash-preview"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=0,
        model_kwargs={"top_p": 1},
        max_retries=3,
        request_timeout=90,
    )

    try:
        resp = llm.invoke([HumanMessage(content=[
            {"type": "text", "text": VLM_FRONT_ELEMS_PICK},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
        ])])

        js = _extract_json_from_text(resp.content)
        labels = (js or {}).get("labels", [])
        logger.info(f"  VLM labels: {labels}")
        return labels

    except Exception as e:
        logger.warning(f"  VLM labeling failed: {e}")
        return []
    finally:
        del llm
        gc.collect()


# =============================================================================
# Element Save Helpers
# =============================================================================

def save_element_dir(
    episode_dir: Path,
    elem_id: str,
    canvas_rgba: np.ndarray,
    canvas_mask: np.ndarray,
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    elem_dir = episode_dir / "elements" / elem_id
    elem_dir.mkdir(parents=True, exist_ok=True)

    canvas_img_path = elem_dir / "canvas_image.png"
    Image.fromarray(canvas_rgba).save(canvas_img_path)

    mask_canvas_path = elem_dir / "mask_canvas.png"
    Image.fromarray(canvas_mask).save(mask_canvas_path)

    meta_path = elem_dir / "metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        "canvas_image_uri": str(canvas_img_path),
        "mask_canvas_uri": str(mask_canvas_path),
    }


# =============================================================================
# Reconstruction (Agent-compatible)
# =============================================================================

def create_reconstructions_agent(
    episode_dir: Path,
    elements: List[Dict[str, Any]],
    canvas_size: Tuple[int, int],
    logger: Optional[logging.Logger] = None,
) -> bool:
    try:
        W, H = canvas_size
        reconstructed = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        # Reverse order: background (bottom) -> objects -> text (top)
        for elem in reversed(elements):
            canvas_uri = elem.get("canvas_image_uri")
            if not canvas_uri or not Path(canvas_uri).exists():
                continue
            elem_img = Image.open(canvas_uri).convert("RGBA")
            if elem_img.size != (W, H):
                elem_img = elem_img.resize((W, H), Image.LANCZOS)
            reconstructed = Image.alpha_composite(reconstructed, elem_img)

        recon_path = episode_dir / "reconstructed.png"
        reconstructed.save(recon_path)

        # Bordered
        border_color = (255, 150, 200, 200)
        glow_color = (255, 180, 220, 100)
        border_width = 3
        glow_width = 5

        result = reconstructed.copy()
        glow_arr = np.zeros((H, W, 4), dtype=np.uint8)
        border_arr = np.zeros((H, W, 4), dtype=np.uint8)

        for elem in elements:
            canvas_uri = elem.get("canvas_image_uri")
            if not canvas_uri or not Path(canvas_uri).exists():
                continue
            elem_img = Image.open(canvas_uri).convert("RGBA")
            if elem_img.size != (W, H):
                elem_img = elem_img.resize((W, H), Image.LANCZOS)
            alpha = np.array(elem_img)[:, :, 3]
            binary_mask = (alpha > 128).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            for i in range(glow_width, 0, -1):
                a = int(glow_color[3] * (1 - i / (glow_width + 2)))
                cv2.drawContours(glow_arr, contours, -1, (*glow_color[:3], a), thickness=i * 2)
            cv2.drawContours(border_arr, contours, -1, border_color, thickness=border_width)

        glow_layer = Image.fromarray(glow_arr).filter(ImageFilter.GaussianBlur(radius=4))
        result = Image.alpha_composite(result, glow_layer)
        result = Image.alpha_composite(result, Image.fromarray(border_arr))
        result.save(episode_dir / "reconstructed_bordered.png")

        return True
    except Exception as e:
        if logger:
            logger.error(f"Reconstruction failed: {e}")
        return False


# =============================================================================
# Core: OCR + HiSAM + VLM + GDINO + SAM2 + LaMa Pipeline
# =============================================================================

class _CudaOOMError(RuntimeError):
    pass


def _check_cuda_oom(exc: Exception) -> bool:
    err_str = str(exc)
    return any(kw in err_str for kw in (
        "CUDA out of memory",
        "CUDA error",
        "Out of memory error on GPU",
        "ResourceExhaustedError",
    )) or isinstance(exc, MemoryError)


def run_multi_tools_pipeline(
    image_path: str,
    episode_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    import torch
    from baselines.tool_backends.tools.ocr_tool import run_ocr, unload_ocr
    from baselines.tool_backends.tools.hisam_tool import run_hisam_union, unload_hisam
    from baselines.tool_backends.tools.sam2_tool import run_sam2_union, reset_sam2_features, unload_sam2
    from baselines.tool_backends.tools.lama_tool import run_lama, unload_lama
    from baselines.tool_backends.tools.dino_tool import run_dino_batch_all, unload_dino

    with Image.open(image_path) as _tmp:
        W, H = _tmp.size
    del _tmp
    current_img_path = str(image_path)

    # Setup directories
    layers_dir = episode_dir / "layers" / "layer_0000"
    layers_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, layers_dir / "layer_image.png")
    shutil.copy2(image_path, episode_dir / "original_input.png")

    elements = []
    elem_counter = 0

    # ==================== PHASE 1: Text Extraction ====================
    logger.info("  Phase 1: Text extraction (OCR + HiSAM)")

    try:
        ocr_result = run_ocr(image_path)
    except Exception as e:
        if _check_cuda_oom(e):
            logger.error(f"  CUDA/GPU OOM in OCR: {e}")
            raise _CudaOOMError(f"OCR GPU OOM: {e}") from e
        raise
    text_boxes_raw = ocr_result.get("boxes", [])
    texts = ocr_result.get("texts", [])
    scores = ocr_result.get("scores", [])

    logger.info(f"  OCR found {len(text_boxes_raw)} text regions")

    if text_boxes_raw:
        text_boxes = [_quad_to_aabb(b) for b in text_boxes_raw]
        text_det_ids = [f"t_{i:03d}" for i in range(len(text_boxes))]

        try:
            hisam_result = run_hisam_union(
                image_path,
                boxes=text_boxes,
                det_ids=text_det_ids,
            )
        except Exception as e:
            if _check_cuda_oom(e):
                logger.error(f"  CUDA/GPU OOM in HiSAM: {e}")
                raise _CudaOOMError(f"HiSAM GPU OOM: {e}") from e
            raise

        canvas_rgb = np.array(Image.open(image_path).convert("RGB"))

        for det_id, box, text_content, score in zip(text_det_ids, text_boxes, texts, scores):
            mask_path = hisam_result["masks_by_id"].get(det_id)
            if not mask_path or not Path(mask_path).exists():
                continue

            mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_full is None or mask_full.max() == 0:
                continue

            mask_binary = (mask_full > 127).astype(np.uint8)

            canvas_rgba = np.zeros((H, W, 4), dtype=np.uint8)
            canvas_rgba[:, :, :3] = canvas_rgb
            canvas_rgba[:, :, 3] = mask_binary * 255
            canvas_mask = mask_binary * 255

            elem_id = f"text_{elem_counter:04d}"
            x1, y1, x2, y2 = box

            meta = {
                "id": elem_id,
                "type": "text",
                "det_id": det_id,
                "content": text_content,
                "bbox": [x1, y1, x2, y2],
                "ocr_score": float(score),
            }

            paths_saved = save_element_dir(episode_dir, elem_id, canvas_rgba, canvas_mask, meta)

            elements.append({
                "id": elem_id,
                "type": "text",
                "det_id": det_id,
                "content": text_content,
                "bbox": [x1, y1, x2, y2],
                "canvas_image_uri": paths_saved["canvas_image_uri"],
                "mask_canvas_uri": paths_saved["mask_canvas_uri"],
                "ocr_score": float(score),
                "source_layer_id": "layer_0000",
            })

            del mask_full, mask_binary, canvas_rgba, canvas_mask
            elem_counter += 1

        # Unload OCR + HiSAM BEFORE LaMa to free GPU memory
        unload_ocr()
        unload_hisam()
        del canvas_rgb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("  Unloaded OCR + HiSAM models before LaMa inpainting")

        # Phase 1.5: LaMa inpaint text region
        text_union_path = hisam_result.get("mask_union")
        if text_union_path and Path(text_union_path).exists():
            try:
                current_img_path = run_lama(current_img_path, text_union_path)
                logger.info(f"  LaMa text inpainting done")
            except Exception as e:
                if _check_cuda_oom(e):
                    logger.error(f"  CUDA OOM in Phase 1 LaMa: {e}")
                    raise _CudaOOMError(f"Phase 1 LaMa CUDA OOM: {e}") from e
                logger.warning(f"  Text LaMa failed (Phase 2 will use original image): {e}")

        # Unload LaMa after text inpainting
        unload_lama()

    else:
        # No text boxes -- still unload OCR
        unload_ocr()

    logger.info(f"  Phase 1 complete: {len(elements)} text elements extracted")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ==================== PHASE 2: Object Extraction ====================
    logger.info("  Phase 2: Object extraction (VLM Label -> GDINO -> SAM2 -> LaMa)")
    logger.info(f"  Phase 2 input image: {current_img_path}")

    iteration = 0
    while iteration < MAX_OBJ_ITERATIONS:
        logger.info(f"  Object iteration {iteration + 1}/{MAX_OBJ_ITERATIONS}")

        # Step 2a: VLM labeling
        labels = call_vlm_front_pick(current_img_path, logger)
        if not labels:
            logger.info(f"  VLM returned no labels at iteration {iteration}, stopping")
            break

        # Step 2b: GDINO detection
        try:
            dino_result = run_dino_batch_all(
                current_img_path,
                labels=labels,
                score_min=DINO_SCORE_MIN,
            )
        except Exception as e:
            if _check_cuda_oom(e):
                logger.error(f"  CUDA OOM in GDINO iter {iteration}: {e}")
                raise _CudaOOMError(f"GDINO CUDA OOM iter {iteration}: {e}") from e
            logger.warning(f"  GDINO failed at iteration {iteration}: {e}")
            break

        dino_boxes = dino_result.get("boxes", [])
        dino_labels = dino_result.get("labels", [])
        dino_confs = dino_result.get("confs", [])

        if not dino_boxes:
            logger.info(f"  GDINO returned no bbox at iteration {iteration}, stopping")
            break

        logger.info(f"  GDINO detected {len(dino_boxes)} objects: {dino_labels}")

        dino_det_ids = [f"o_{elem_counter + i:03d}" for i in range(len(dino_boxes))]

        # Step 2c: SAM2 segmentation
        try:
            sam2_result = run_sam2_union(
                current_img_path,
                boxes=dino_boxes,
                det_ids=dino_det_ids,
            )
        except Exception as e:
            if _check_cuda_oom(e):
                logger.error(f"  CUDA OOM in SAM2 iter {iteration}: {e}")
                raise _CudaOOMError(f"SAM2 CUDA OOM iter {iteration}: {e}") from e
            logger.warning(f"  SAM2 failed at iteration {iteration}: {e}")
            break

        union_mask_path = sam2_result.get("mask_union")
        if union_mask_path:
            union_check = cv2.imread(union_mask_path, cv2.IMREAD_GRAYSCALE)
            if union_check is None or union_check.max() == 0:
                del union_check
                logger.info(f"  Empty SAM2 union mask at iteration {iteration}, stopping")
                break
            union_mask_pixels = int((union_check > 127).sum())
            del union_check
            if union_mask_pixels < SAM2_MIN_MASK_PIXELS:
                logger.info(f"  SAM2 union mask too small ({union_mask_pixels} px < {SAM2_MIN_MASK_PIXELS}), stopping")
                break

        # Load current image for RGB extraction
        current_img = cv2.imread(current_img_path)
        if current_img is None:
            break
        current_rgb = current_img[:, :, ::-1]  # BGR -> RGB
        cur_H, cur_W = current_img.shape[:2]

        new_elements_count = 0
        for det_id, box, label in zip(dino_det_ids, dino_boxes, dino_labels):
            mask_path = sam2_result["masks_by_id"].get(det_id)
            if not mask_path or not Path(mask_path).exists():
                continue

            mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_full is None or mask_full.max() == 0:
                continue

            mask_binary = (mask_full > 127).astype(np.uint8)

            # Canvas size RGBA
            canvas_rgba = np.zeros((H, W, 4), dtype=np.uint8)
            rgb_h, rgb_w = min(cur_H, H), min(cur_W, W)
            canvas_rgba[:rgb_h, :rgb_w, :3] = current_rgb[:rgb_h, :rgb_w]
            canvas_rgba[:, :, 3] = mask_binary * 255

            canvas_mask = mask_binary * 255

            elem_id = f"obj_{elem_counter:04d}"
            x1, y1, x2, y2 = [int(v) for v in box]

            meta = {
                "id": elem_id,
                "type": "object",
                "det_id": det_id,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "iteration": iteration,
            }

            paths_saved = save_element_dir(episode_dir, elem_id, canvas_rgba, canvas_mask, meta)

            elements.append({
                "id": elem_id,
                "type": "object",
                "det_id": det_id,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "canvas_image_uri": paths_saved["canvas_image_uri"],
                "mask_canvas_uri": paths_saved["mask_canvas_uri"],
                "iteration": iteration,
                "source_layer_id": "layer_0000",
            })

            del mask_full, mask_binary, canvas_rgba, canvas_mask
            elem_counter += 1
            new_elements_count += 1

        del current_img, current_rgb
        if new_elements_count == 0:
            logger.info(f"  No valid segments at iteration {iteration}, stopping")
            break

        logger.info(f"  Found {new_elements_count} objects at iteration {iteration}")

        # Step 2d: LaMa inpaint
        if union_mask_path and Path(union_mask_path).exists():
            try:
                current_img_path = run_lama(current_img_path, union_mask_path)
            except Exception as e:
                if _check_cuda_oom(e):
                    logger.error(f"  CUDA OOM in LaMa iter {iteration}: {e}")
                    raise _CudaOOMError(f"LaMa CUDA OOM iter {iteration}: {e}") from e
                logger.warning(f"  LaMa failed at iteration {iteration}: {e}")
                break

        iteration += 1

        # Clean up SAM2 predictor features
        reset_sam2_features()
        del dino_result, sam2_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ==================== PHASE 3: Background ====================
    bg_id = f"bg_{elem_counter:04d}"
    bg_dir = episode_dir / "elements" / bg_id
    bg_dir.mkdir(parents=True, exist_ok=True)

    bg_img = Image.open(current_img_path).convert("RGB")
    bg_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    bg_arr = np.array(bg_img)
    if bg_arr.shape[:2] == (H, W):
        bg_rgba[:, :, :3] = bg_arr
    else:
        bg_rgba[:, :, :3] = cv2.resize(bg_arr, (W, H))
    bg_rgba[:, :, 3] = 255
    bg_mask = np.full((H, W), 255, dtype=np.uint8)

    bg_meta = {"id": bg_id, "type": "background", "bbox": [0, 0, W, H]}
    bg_paths = save_element_dir(episode_dir, bg_id, bg_rgba, bg_mask, bg_meta)
    del bg_img, bg_rgba, bg_arr, bg_mask

    elements.append({
        "id": bg_id,
        "type": "background",
        "bbox": [0, 0, W, H],
        "canvas_image_uri": bg_paths["canvas_image_uri"],
        "mask_canvas_uri": bg_paths["mask_canvas_uri"],
        "source_layer_id": "layer_0000",
    })

    # Unload all remaining models
    unload_sam2()
    unload_dino()
    unload_lama()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "canvas_size": [W, H],
        "elements": elements,
        "total_obj_iterations": iteration,
        "method": "vlm_gdino_sam2",
    }


# =============================================================================
# Build history_tree.json (Agent-compatible)
# =============================================================================

def build_minimal_history_tree(
    elements: List[Dict[str, Any]],
    root_image_path: str,
) -> Dict[str, Any]:
    tree = {
        "layer_0000": {
            "layer_id": "layer_0000",
            "parent_id": None,
            "depth": 0,
            "image_path": root_image_path,
            "action_type": "Root",
            "children_ids": [],
        }
    }

    for elem in elements:
        elem_id = elem["id"]
        elem_type = elem.get("type", "object")

        if elem_type == "text":
            action_type = "Finalize_Text"
        elif elem_type == "background":
            action_type = "Finalize_Obj"
        else:
            action_type = "Finalize_Obj"

        tree["layer_0000"]["children_ids"].append(elem_id)

        tree[elem_id] = {
            "layer_id": elem_id,
            "parent_id": "layer_0000",
            "depth": 1,
            "image_path": elem.get("canvas_image_uri", ""),
            "action_type": action_type,
            "children_ids": [],
        }

    return tree


# =============================================================================
# Per-Frame Subprocess Worker
# =============================================================================

def _frame_worker_fn(
    gpu_id: int,
    record_id: str,
    image_path: str,
    output_dir: Path,
    result_queue: mp.Queue,
):
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log_file = output_dir / f"gpu{gpu_id}.log"
    logger = setup_logging(log_file, name=f"gpu{gpu_id}_{record_id}")
    logger.info(f"[pid={os.getpid()}] Processing {record_id} on GPU {gpu_id}")

    # Internal timeout as safety net
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(FRAME_TIMEOUT_SEC)

    start_time = time.time()

    try:
        episode_dir = output_dir / "episodes" / record_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        pipeline_result = run_multi_tools_pipeline(
            image_path,
            episode_dir,
            logger,
        )

        elements = pipeline_result["elements"]
        W, H = pipeline_result["canvas_size"]

        # Save parse.json
        parse_data = {
            "episode_id": record_id,
            "root_image": str(episode_dir / "layers" / "layer_0000" / "layer_image.png"),
            "elements": elements,
        }
        with open(episode_dir / "parse.json", 'w', encoding='utf-8') as f:
            json.dump(parse_data, f, indent=2, ensure_ascii=False)

        # Save history_tree.json
        history_tree = build_minimal_history_tree(
            elements,
            str(episode_dir / "layers" / "layer_0000" / "layer_image.png"),
        )
        with open(episode_dir / "history_tree.json", 'w', encoding='utf-8') as f:
            json.dump(history_tree, f, indent=2, ensure_ascii=False)

        # Create reconstructions
        create_reconstructions_agent(episode_dir, elements, (W, H), logger)

        elapsed = time.time() - start_time

        result_queue.put({
            "gpu_id": gpu_id,
            "record_id": record_id,
            "success": True,
            "num_elements": len(elements),
            "iterations": pipeline_result["total_obj_iterations"],
            "elapsed": elapsed,
        })

        logger.info(
            f"  Done: {record_id} | {len(elements)} elements | "
            f"{pipeline_result['total_obj_iterations']} obj iters | {elapsed:.1f}s"
        )

    except (_CudaOOMError, _FrameTimeout, Exception) as e:
        elapsed = time.time() - start_time
        err_str = str(e)

        if isinstance(e, _CudaOOMError) or _check_cuda_oom(e):
            partial_parse = output_dir / "episodes" / record_id / "parse.json"
            if partial_parse.exists():
                partial_parse.unlink()
            mark_record_failed(output_dir, record_id, f"GPU OOM: {err_str[:200]}")
            logger.error(f"  GPU OOM: {record_id}: {err_str[:200]}")
        elif isinstance(e, _FrameTimeout):
            mark_record_failed(output_dir, record_id, f"Timeout after {elapsed:.0f}s")
            logger.error(f"  TIMEOUT: {record_id} after {elapsed:.0f}s")
        else:
            logger.error(f"  Failed: {record_id}: {e}\n{traceback.format_exc()}")

        result_queue.put({
            "gpu_id": gpu_id,
            "record_id": record_id,
            "success": False,
            "error": err_str,
            "elapsed": elapsed,
        })

    finally:
        signal.alarm(0)


# =============================================================================
# Main Runner
# =============================================================================

def run_multi_tools_crello(
    data_dir: Path,
    output_dir: Path,
    gpu_ids: List[int],
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip_completed: bool = True,
    retry_failed: bool = False,
    workers_per_gpu: int = 1,
    gpu_workers_map: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "multi_tools_crello_run.log"
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("Multi-Tools Pipeline (OCR + HiSAM + VLM + GDINO + SAM2 + LaMa) — Crello Dataset")
    logger.info("=" * 70)

    logger.info(f"data_dir: {data_dir}")
    logger.info(f"Output: {output_dir}")

    records = load_record_list(data_dir)
    logger.info(f"Found {len(records)} total records")

    # If retry_failed, clean up _FAILED markers and incomplete episode dirs
    retried = 0
    if retry_failed:
        episodes_dir = output_dir / "episodes"
        if episodes_dir.exists():
            for fail_marker in episodes_dir.glob("*/_FAILED"):
                frame_dir = fail_marker.parent
                shutil.rmtree(frame_dir, ignore_errors=True)
                retried += 1
        if retried:
            logger.info(f"Cleaned up {retried} previously failed episodes for retry")

    records_to_process = []
    skipped = 0
    skipped_failed = 0
    for rec in records:
        if skip_completed and is_record_completed(output_dir, rec.record_id):
            skipped += 1
            continue
        if is_record_failed(output_dir, rec.record_id):
            skipped_failed += 1
            continue
        records_to_process.append(rec)

    if limit:
        records_to_process = records_to_process[:limit]

    logger.info(f"Will process {len(records_to_process)} records (skipped {skipped} completed, {skipped_failed} failed, {retried} retrying)")

    if dry_run:
        for r in records_to_process[:20]:
            logger.info(f"  {r.record_id}")
        return {"dry_run": True}

    # Per-frame subprocess architecture
    result_queue = mp.Queue()

    # Build slot list: supports per-GPU worker counts via gpu_workers_map
    slots = []
    if gpu_workers_map:
        for gid, n_workers in sorted(gpu_workers_map.items()):
            for w in range(n_workers):
                slots.append((gid, f"{gid}_{w}"))
    else:
        for gid in gpu_ids:
            for w in range(workers_per_gpu):
                slots.append((gid, f"{gid}_{w}"))

    num_slots = len(slots)
    logger.info(f"GPUs: {[s[0] for s in slots]} | total slots: {num_slots}")

    total = len(records_to_process)
    frame_idx = 0
    completed = 0
    received_records = set()
    active = {}  # slot_key -> (Process, record_id, start_time)

    results = {"success": 0, "failed": 0, "errors": []}
    pbar = tqdm(total=total, desc="MultiTools Crello", unit="record", ncols=100)

    logger.info(f"Subprocess-per-frame mode: {num_slots} slots across GPUs {list(dict.fromkeys(s[0] for s in slots))}")

    while completed < total:
        # 1. Spawn new subprocesses for idle slots
        for gpu_id, slot_key in slots:
            if slot_key not in active and frame_idx < total:
                rec = records_to_process[frame_idx]
                frame_idx += 1
                p = mp.Process(
                    target=_frame_worker_fn,
                    args=(gpu_id, rec.record_id, str(rec.image_path),
                          output_dir, result_queue),
                )
                p.start()
                active[slot_key] = (p, rec.record_id, time.time())
                logger.debug(f"  Spawned pid={p.pid} for {rec.record_id} on GPU {gpu_id} slot {slot_key}")

        # 2. Collect results (non-blocking drain)
        while True:
            try:
                r = result_queue.get_nowait()
            except Exception:
                break
            rid = r["record_id"]
            if rid in received_records:
                continue
            received_records.add(rid)
            completed += 1
            if r["success"]:
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(r)
            pbar.update(1)
            pbar.set_postfix_str(f"ok={results['success']} fail={results['failed']}", refresh=True)

        # 3. Reap finished / timed-out subprocesses
        for slot_key in list(active):
            proc, rid, start_t = active[slot_key]
            elapsed = time.time() - start_t

            if not proc.is_alive():
                proc.join(timeout=5)
                del active[slot_key]
                # If process crashed without sending a result
                if rid not in received_records:
                    received_records.add(rid)
                    completed += 1
                    results["failed"] += 1
                    exit_code = proc.exitcode
                    mark_record_failed(output_dir, rid,
                                       f"Process crashed (exit code {exit_code})")
                    logger.error(f"  Process crashed for {rid} (exit code {exit_code})")
                    pbar.update(1)
                    pbar.set_postfix_str(f"ok={results['success']} fail={results['failed']}", refresh=True)

            elif elapsed > FRAME_PROCESS_TIMEOUT_SEC:
                logger.warning(f"  Killing timed-out process for {rid} (pid={proc.pid}, {elapsed:.0f}s)")
                proc.kill()
                proc.join(timeout=10)
                del active[slot_key]
                if rid not in received_records:
                    received_records.add(rid)
                    completed += 1
                    results["failed"] += 1
                    mark_record_failed(output_dir, rid, f"Killed after {elapsed:.0f}s timeout")
                    pbar.update(1)
                    pbar.set_postfix_str(f"ok={results['success']} fail={results['failed']}", refresh=True)

        time.sleep(0.5)  # avoid busy loop

    pbar.close()

    # Clean up any remaining active processes
    for slot_key, (proc, rid, _) in active.items():
        proc.kill()
        proc.join(timeout=10)

    logger.info(f"DONE: {results['success']} success, {results['failed']} failed")
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Tools Pipeline (OCR + HiSAM + VLM + GDINO + SAM2 + LaMa) on Crello Dataset"
    )
    parser.add_argument("--data_dir", "-i", type=str, required=True,
                        help="Path to the Crello dataset directory (containing crello_test_*/ record dirs)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Path to the output directory (an episodes/ subfolder is created here)")
    parser.add_argument("--gpu", type=str, default="0",
                        help="comma-separated GPU ids (set to your own; e.g. 0 or 0,1)")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Uniform number of workers per GPU (ignored if --gpu_workers is set)")
    parser.add_argument("--gpu_workers", type=str, default=None,
                        help="Per-GPU worker counts as 'gpu_id:count' pairs, comma-separated "
                             "with user-specific GPU ids, e.g. '0:2,1:3,2:3,3:3'")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_skip", action="store_true")
    parser.add_argument("--retry_failed", action="store_true")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]

    # Parse per-GPU worker map
    gpu_workers_map = None
    if args.gpu_workers:
        gpu_workers_map = {}
        for spec in args.gpu_workers.split(","):
            gid_str, cnt_str = spec.split(":")
            gpu_workers_map[int(gid_str.strip())] = int(cnt_str.strip())

    run_multi_tools_crello(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        gpu_ids=gpu_ids,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_completed=not args.no_skip,
        retry_failed=args.retry_failed,
        workers_per_gpu=args.workers_per_gpu,
        gpu_workers_map=gpu_workers_map,
    )


if __name__ == "__main__":
    main()
