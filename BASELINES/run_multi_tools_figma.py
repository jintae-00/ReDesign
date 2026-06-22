#!/usr/bin/env python3
"""
run_baseline2_figma.py - Baseline 2: OCR + HiSAM + VLM Labeling + GDINO + SAM2 + LaMa

Phase 1: OCR bbox -> HiSAM segmentation -> save text RGBA -> LaMa inpaint (remove text union mask)
Phase 2: VLM Labeling -> GDINO detection -> SAM2 segmentation -> LaMa inpaint (iterative)

Output Format: parse.json + elements/ (Agent-compatible)
Evaluation: evaluation.editability_utils → extract_agent_elements()

Usage:
    python run_baseline2_figma.py --gpu 0
    python run_baseline2_figma.py --gpu 0,1,2,3 --limit 10

Directory Structure:
    Output:
        baseline2_experiment/split_0/episodes/{frame_id}/
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

FIGMA_DATA_BASE = "figma_data/process/subset"
BASELINE2_EXPERIMENT_BASE = "baseline_muilti_tools_experiment"

# All dataset splits: (prefix, num_splits) — dino80 (435) + dino90 (474) = 909 frames
ALL_DATASET_SPLITS = [
    ("dino80_obj_5_60_char_25_split_", 4),   # splits 0-3
    ("dino90_obj_5_25_char_50_split_", 5),   # splits 0-4
]

MAX_OBJ_ITERATIONS = 4
ALPHA_THRESHOLD = 16
FRAME_TIMEOUT_SEC = 600  # 10 min timeout per frame

# GDINO detection
DINO_SCORE_MIN = 0.1

# SAM2 minimum mask area (pixels) — stop if union mask smaller than this
SAM2_MIN_MASK_PIXELS = 200

# Per-frame subprocess timeout (seconds) — process is killed if exceeded
FRAME_PROCESS_TIMEOUT_SEC = 660  # slightly longer than FRAME_TIMEOUT_SEC for graceful handling

# VLM labeling prompt (same as REDESIGN/prompts.py VLM_FRONT_ELEMS_PICK)
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

def setup_logging(log_file: Path, name: str = "baseline2") -> logging.Logger:
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

def get_src_root() -> Path:
    current = Path(__file__).resolve().parent
    if current.name == "src":
        return current
    elif (current / "src").exists():
        return current / "src"
    else:
        return current


def get_all_split_paths(src_root: Path) -> Dict[str, Any]:
    all_valid_frames_dirs = []
    all_split_dirs = []
    for prefix, num_splits in ALL_DATASET_SPLITS:
        for i in range(num_splits):
            split_name = f"{prefix}{i}"
            split_dir = src_root / FIGMA_DATA_BASE / split_name
            all_split_dirs.append(split_dir)
            all_valid_frames_dirs.append(split_dir / "valid_frames")
    return {
        "split_dirs": all_split_dirs,
        "valid_frames_dirs": all_valid_frames_dirs,
        "output_dir": src_root / BASELINE2_EXPERIMENT_BASE / "all_splits",
    }


# =============================================================================
# Frame Data Loading
# =============================================================================

@dataclass
class FrameInfo:
    frame_id: str
    json_path: Path
    image_path: Path


def load_frame_list(paths: Dict[str, Any]) -> List[FrameInfo]:
    frames = []
    for split_dir, vf_dir in zip(paths["split_dirs"], paths["valid_frames_dirs"]):
        if not vf_dir.exists():
            continue
        json_files = sorted(vf_dir.glob("*.json"))
        for json_path in json_files:
            frame_id = json_path.stem
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                rel_path = json_data.get("reconstructed_image_path")
                unit_images_dir = json_data.get("unit_images_dir")
                if rel_path and unit_images_dir:
                    image_path = split_dir / unit_images_dir / rel_path
                    if image_path.exists():
                        frames.append(FrameInfo(frame_id=frame_id, json_path=json_path, image_path=image_path))
            except Exception:
                continue
    return frames


def is_frame_completed(output_dir: Path, frame_id: str) -> bool:
    return (output_dir / "episodes" / frame_id / "parse.json").exists()


def is_frame_failed(output_dir: Path, frame_id: str) -> bool:
    return (output_dir / "episodes" / frame_id / "_FAILED").exists()


def mark_frame_failed(output_dir: Path, frame_id: str, reason: str):
    fail_dir = output_dir / "episodes" / frame_id
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
    """Resize image for VLM to reduce network overhead."""
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
    """Extract JSON object from LLM response text."""
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
    """
    Call Gemini 3 Flash VLM to label front-most elements.
    Same logic as REDESIGN/nodes/vlm_front_pick.py.
    Returns list of labels for GDINO detection.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    img_bytes = _resize_image_for_vlm(image_path)
    b64_str = base64.b64encode(img_bytes).decode("utf-8")

    llm = ChatOpenAI(
        model="gemini-3-flash-preview",
        base_url="https://gateway.letsur.ai/v1",
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
    """Save element files to episode_dir/elements/{elem_id}/"""
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

        # Reverse order: background (bottom) → objects → text (top)
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
    """Raised when CUDA OOM is caught mid-pipeline to propagate failure."""
    pass


def _check_cuda_oom(exc: Exception) -> bool:
    """Check if an exception is a CUDA/GPU out of memory error (PyTorch or PaddlePaddle)."""
    err_str = str(exc)
    return any(kw in err_str for kw in (
        "CUDA out of memory",
        "CUDA error",
        "Out of memory error on GPU",    # PaddlePaddle ResourceExhaustedError
        "ResourceExhaustedError",          # PaddlePaddle
    )) or isinstance(exc, MemoryError)


def run_baseline2_pipeline(
    image_path: str,
    episode_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Phase 1: OCR -> HiSAM -> save text RGBA -> LaMa inpaint (remove text union mask)
    Phase 2: VLM Labeling -> GDINO -> SAM2 -> LaMa (objects, iterative)
    """
    import torch
    from BASELINES.tool_backends.tools.ocr_tool import run_ocr, unload_ocr
    from BASELINES.tool_backends.tools.hisam_tool import run_hisam_union, unload_hisam
    from BASELINES.tool_backends.tools.sam2_tool import run_sam2_union, reset_sam2_features, unload_sam2
    from BASELINES.tool_backends.tools.lama_tool import run_lama, unload_lama
    from BASELINES.tool_backends.tools.dino_tool import run_dino_batch_all, unload_dino

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

    # ==================== PHASE 1: Text Extraction (single pass) ====================
    logger.info("  Phase 1: Text extraction (OCR + HiSAM)")

    try:
        ocr_result = run_ocr(image_path)
    except Exception as e:
        if _check_cuda_oom(e):
            logger.error(f"  CUDA/GPU OOM in OCR -- aborting frame: {e}")
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
                logger.error(f"  CUDA/GPU OOM in HiSAM -- aborting frame: {e}")
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

        # Phase 1.5: LaMa inpaint text region -> text-free image for Phase 2
        text_union_path = hisam_result.get("mask_union")
        if text_union_path and Path(text_union_path).exists():
            try:
                current_img_path = run_lama(current_img_path, text_union_path)
                logger.info(f"  LaMa text inpainting done -> text-free image ready for Phase 2")
            except Exception as e:
                if _check_cuda_oom(e):
                    logger.error(f"  CUDA OOM in Phase 1 LaMa -- aborting frame: {e}")
                    raise _CudaOOMError(f"Phase 1 LaMa CUDA OOM: {e}") from e
                logger.warning(f"  Text LaMa failed (Phase 2 will use original image): {e}")

        # Unload LaMa after text inpainting -- will be reloaded in Phase 2 if needed
        unload_lama()

    else:
        # No text boxes -- still unload OCR (HiSAM was never loaded)
        unload_ocr()

    logger.info(f"  Phase 1 complete: {len(elements)} text elements extracted")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ==================== PHASE 2: Object Extraction (iterative) ====================
    # Uses text-inpainted image -> VLM Labeling -> GDINO -> SAM2 -> LaMa -> repeat
    logger.info("  Phase 2: Object extraction (VLM Label -> GDINO -> SAM2 -> LaMa)")
    logger.info(f"  Phase 2 input image: {current_img_path}")

    iteration = 0
    while iteration < MAX_OBJ_ITERATIONS:
        logger.info(f"  Object iteration {iteration + 1}/{MAX_OBJ_ITERATIONS}")

        # Step 2a: VLM labeling - get front-most element labels
        labels = call_vlm_front_pick(current_img_path, logger)
        if not labels:
            logger.info(f"  VLM returned no labels at iteration {iteration}, stopping")
            break

        # Step 2b: GDINO detection using VLM labels
        # STOP condition 1: GDINO returns no bbox
        try:
            dino_result = run_dino_batch_all(
                current_img_path,
                labels=labels,
                score_min=DINO_SCORE_MIN,
            )
        except Exception as e:
            if _check_cuda_oom(e):
                logger.error(f"  CUDA OOM in GDINO iter {iteration} -- aborting frame: {e}")
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

        # Step 2c: SAM2 segmentation using GDINO boxes
        # STOP condition 2: SAM2 union mask is empty or too small
        try:
            sam2_result = run_sam2_union(
                current_img_path,
                boxes=dino_boxes,
                det_ids=dino_det_ids,
            )
        except Exception as e:
            if _check_cuda_oom(e):
                logger.error(f"  CUDA OOM in SAM2 iter {iteration} -- aborting frame: {e}")
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
                    logger.error(f"  CUDA OOM in LaMa iter {iteration} -- aborting frame: {e}")
                    raise _CudaOOMError(f"LaMa CUDA OOM iter {iteration}: {e}") from e
                logger.warning(f"  LaMa failed at iteration {iteration}: {e}")
                break

        iteration += 1

        # Clean up SAM2 predictor features (no local reference to avoid holding model)
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

    # Unload all remaining models to free GPU memory for next frame
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
    """
    Build the minimal history_tree.json required by Agent evaluation.
    All elements are registered flatly as children of the root.
    """
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
# Per-Frame Subprocess Worker (fresh process per frame -> GPU memory fully reset)
# =============================================================================

def _frame_worker_fn(
    gpu_id: int,
    frame_id: str,
    image_path: str,
    output_dir: Path,
    result_queue: mp.Queue,
):
    """
    Fresh subprocess per frame.
    All GPU memory is freed when this process exits -- no accumulation possible.
    """
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log_file = output_dir / f"gpu{gpu_id}.log"
    logger = setup_logging(log_file, name=f"gpu{gpu_id}_{frame_id}")
    logger.info(f"[pid={os.getpid()}] Processing {frame_id} on GPU {gpu_id}")

    # Internal timeout as safety net (slightly shorter than external kill timeout)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(FRAME_TIMEOUT_SEC)

    start_time = time.time()

    try:
        episode_dir = output_dir / "episodes" / frame_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        pipeline_result = run_baseline2_pipeline(
            image_path,
            episode_dir,
            logger,
        )

        elements = pipeline_result["elements"]
        W, H = pipeline_result["canvas_size"]

        # Save parse.json
        parse_data = {
            "episode_id": frame_id,
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
            "frame_id": frame_id,
            "success": True,
            "num_elements": len(elements),
            "iterations": pipeline_result["total_obj_iterations"],
            "elapsed": elapsed,
        })

        logger.info(
            f"  Done: {frame_id} | {len(elements)} elements | "
            f"{pipeline_result['total_obj_iterations']} obj iters | {elapsed:.1f}s"
        )

    except (_CudaOOMError, _FrameTimeout, Exception) as e:
        elapsed = time.time() - start_time
        err_str = str(e)

        if isinstance(e, _CudaOOMError) or _check_cuda_oom(e):
            partial_parse = output_dir / "episodes" / frame_id / "parse.json"
            if partial_parse.exists():
                partial_parse.unlink()
            mark_frame_failed(output_dir, frame_id, f"GPU OOM: {err_str[:200]}")
            logger.error(f"  GPU OOM: {frame_id}: {err_str[:200]}")
        elif isinstance(e, _FrameTimeout):
            mark_frame_failed(output_dir, frame_id, f"Timeout after {elapsed:.0f}s")
            logger.error(f"  TIMEOUT: {frame_id} after {elapsed:.0f}s")
        else:
            logger.error(f"  Failed: {frame_id}: {e}\n{traceback.format_exc()}")

        result_queue.put({
            "gpu_id": gpu_id,
            "frame_id": frame_id,
            "success": False,
            "error": err_str,
            "elapsed": elapsed,
        })

    finally:
        signal.alarm(0)
    # Process exits here -> CUDA context destroyed -> GPU memory fully freed


# =============================================================================
# Main Runner
# =============================================================================

def run_baseline2(
    gpu_ids: List[int],
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip_completed: bool = True,
    retry_failed: bool = False,
    src_root: Optional[Path] = None,
    workers_per_gpu: int = 1,
) -> Dict[str, Any]:
    if src_root is None:
        src_root = get_src_root()

    paths = get_all_split_paths(src_root)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    log_file = paths["output_dir"] / "baseline2_run.log"
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("Baseline 2: OCR + HiSAM + VLM Label + GDINO + SAM2 + LaMa")
    logger.info("=" * 70)

    num_slots = len(gpu_ids) * workers_per_gpu
    logger.info(f"GPUs: {gpu_ids} | workers_per_gpu: {workers_per_gpu} | total slots: {num_slots}")
    logger.info(f"Output: {paths['output_dir']}")

    frames = load_frame_list(paths)
    logger.info(f"Found {len(frames)} total frames")

    # If retry_failed, clean up _FAILED markers and incomplete episode dirs
    retried = 0
    if retry_failed:
        episodes_dir = paths["output_dir"] / "episodes"
        if episodes_dir.exists():
            for fail_marker in episodes_dir.glob("*/_FAILED"):
                frame_dir = fail_marker.parent
                shutil.rmtree(frame_dir, ignore_errors=True)
                retried += 1
        if retried:
            logger.info(f"Cleaned up {retried} previously failed episodes for retry")

    frames_to_process = []
    skipped = 0
    skipped_failed = 0
    for frame in frames:
        if skip_completed and is_frame_completed(paths["output_dir"], frame.frame_id):
            skipped += 1
            continue
        if is_frame_failed(paths["output_dir"], frame.frame_id):
            skipped_failed += 1
            continue
        frames_to_process.append(frame)

    if limit:
        frames_to_process = frames_to_process[:limit]

    logger.info(f"Will process {len(frames_to_process)} frames (skipped {skipped} completed, {skipped_failed} failed, {retried} retrying)")

    if dry_run:
        for f in frames_to_process[:20]:
            logger.info(f"  {f.frame_id}")
        return {"dry_run": True}

    # -- Per-frame subprocess architecture --------------------------
    # Each frame runs in a fresh subprocess -> process exit frees ALL GPU memory.
    # One active subprocess per GPU slot at a time.
    result_queue = mp.Queue()

    # Build slot list: (gpu_id, slot_key)
    # e.g. gpu_ids=[1,2], workers_per_gpu=2 -> slots=[(1,"1_0"),(1,"1_1"),(2,"2_0"),(2,"2_1")]
    slots = []
    for gid in gpu_ids:
        for w in range(workers_per_gpu):
            slots.append((gid, f"{gid}_{w}"))

    total = len(frames_to_process)
    frame_idx = 0
    completed = 0
    received_frames = set()  # frame_ids whose results we've collected
    active = {}  # slot_key -> (Process, frame_id, start_time)

    results = {"success": 0, "failed": 0, "errors": []}
    pbar = tqdm(total=total, desc="Baseline2", unit="frame", ncols=100)

    logger.info(f"Subprocess-per-frame mode: {len(slots)} slots across GPUs {gpu_ids}")

    while completed < total:
        # 1. Spawn new subprocesses for idle slots
        for gpu_id, slot_key in slots:
            if slot_key not in active and frame_idx < total:
                frame = frames_to_process[frame_idx]
                frame_idx += 1
                p = mp.Process(
                    target=_frame_worker_fn,
                    args=(gpu_id, frame.frame_id, str(frame.image_path),
                          paths["output_dir"], result_queue),
                )
                p.start()
                active[slot_key] = (p, frame.frame_id, time.time())
                logger.debug(f"  Spawned pid={p.pid} for {frame.frame_id} on GPU {gpu_id} slot {slot_key}")

        # 2. Collect results (non-blocking drain)
        while True:
            try:
                r = result_queue.get_nowait()
            except Exception:
                break
            fid = r["frame_id"]
            if fid in received_frames:
                continue
            received_frames.add(fid)
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
            proc, fid, start_t = active[slot_key]
            elapsed = time.time() - start_t

            if not proc.is_alive():
                proc.join(timeout=5)
                del active[slot_key]
                # If process crashed without sending a result
                if fid not in received_frames:
                    received_frames.add(fid)
                    completed += 1
                    results["failed"] += 1
                    exit_code = proc.exitcode
                    mark_frame_failed(paths["output_dir"], fid,
                                      f"Process crashed (exit code {exit_code})")
                    logger.error(f"  Process crashed for {fid} (exit code {exit_code})")
                    pbar.update(1)
                    pbar.set_postfix_str(f"ok={results['success']} fail={results['failed']}", refresh=True)

            elif elapsed > FRAME_PROCESS_TIMEOUT_SEC:
                logger.warning(f"  Killing timed-out process for {fid} (pid={proc.pid}, {elapsed:.0f}s)")
                proc.kill()
                proc.join(timeout=10)
                del active[slot_key]
                if fid not in received_frames:
                    received_frames.add(fid)
                    completed += 1
                    results["failed"] += 1
                    mark_frame_failed(paths["output_dir"], fid, f"Killed after {elapsed:.0f}s timeout")
                    pbar.update(1)
                    pbar.set_postfix_str(f"ok={results['success']} fail={results['failed']}", refresh=True)

        time.sleep(0.5)  # avoid busy loop

    pbar.close()

    # Clean up any remaining active processes
    for slot_key, (proc, fid, _) in active.items():
        proc.kill()
        proc.join(timeout=10)

    logger.info(f"DONE: {results['success']} success, {results['failed']} failed")
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline 2: OCR + HiSAM + VLM + GDINO + SAM2 + LaMa")
    parser.add_argument("--gpu", type=str, default="0",
                        help="Comma-separated GPU ids to use (user-specific, e.g. '0,1,2,3')")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Number of workers per GPU (default: 1)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_skip", action="store_true")
    parser.add_argument("--retry_failed", action="store_true", help="Clean up and retry previously failed (CUDA OOM etc.) episodes")
    parser.add_argument("--src_root", type=str, default=None)
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]
    src_root = Path(args.src_root) if args.src_root else None

    run_baseline2(
        gpu_ids=gpu_ids,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_completed=not args.no_skip,
        retry_failed=args.retry_failed,
        src_root=src_root,
        workers_per_gpu=args.workers_per_gpu,
    )


if __name__ == "__main__":
    main()
