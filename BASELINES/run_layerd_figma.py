#!/usr/bin/env python3
"""
run_baseline1_figma.py - Baseline 1: LayerD + LaMa Iterative Extraction

Iteratively extracts the front layer with LayerD (no CCA applied; saved as RGBA
directly) and inpaints with LaMa, producing the same output format as the Qwen baseline.

Output Format: metadata.json + layer_*.png (Qwen-compatible)
Evaluation: evaluation_figma.py -> extract_qwen_elements_cca()

Processes EVERY episode found under <data_dir>/valid_frames/.

Usage:
    python run_layerd_figma.py --data_dir figma_data --output_dir outputs/layerd_figma --gpu <GPU_IDS>
    python run_layerd_figma.py --data_dir figma_data --output_dir outputs/layerd_figma --gpu <GPU_IDS> --limit 10
    python run_layerd_figma.py --data_dir figma_data --output_dir outputs/layerd_figma --gpu <GPU_IDS> --workers_per_gpu 1

    # Replace <GPU_IDS> with your own comma-separated GPU ids (e.g. 0 or 0,1).

Directory Structure:
    Input:
        <data_dir>/valid_frames/<frame_id>.json
        <data_dir>/unit_images/...        (GT layers + reconstruction)

    Output:
        <output_dir>/episodes/{frame_id}/
            - input.png
            - layer_00.png  (RGBA, canvas size)
            - layer_01.png
            - ...
            - metadata.json
            - reconstructed.png
            - reconstructed_bordered.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import os
import gc
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

MAX_LAYERD_ITERATIONS = 4
ALPHA_THRESHOLD_LAYERD = 16
CCA_MIN_AREA = 100
CCA_ALPHA_THRESHOLD = 10
INPUT_IMAGE_NAME = "input.png"
FRAME_TIMEOUT_SEC = 600  # 10 min timeout per frame to avoid hanging
MAX_IMAGE_LONG_SIDE = 1536  # Resize images larger than this to prevent OOM


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_file: Path, name: str = "baseline1") -> logging.Logger:
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
# Frame Data Loading
# =============================================================================

@dataclass
class FrameInfo:
    frame_id: str
    json_path: Path
    image_path: Path


def load_frame_list(data_dir: Path) -> List[FrameInfo]:
    """Load every valid_frames/*.json under data_dir (split-agnostic).

    The GT reconstruction (input image) resolves to
    ``data_dir / <unit_images_dir> / <reconstructed_image_path>`` from each JSON.
    """
    frames = []
    vf_dir = data_dir / "valid_frames"
    if not vf_dir.exists():
        print(f"[Warning] Not found: {vf_dir}")
        return frames
    json_files = sorted(vf_dir.glob("*.json"))
    for json_path in json_files:
        frame_id = json_path.stem
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            rel_path = json_data.get("reconstructed_image_path")
            unit_images_dir = json_data.get("unit_images_dir")
            if rel_path and unit_images_dir:
                image_path = data_dir / unit_images_dir / rel_path
                if image_path.exists():
                    frames.append(FrameInfo(
                        frame_id=frame_id,
                        json_path=json_path,
                        image_path=image_path,
                    ))
        except Exception as e:
            print(f"[Warning] Failed to load {json_path}: {e}")
    return frames


def is_frame_completed(output_dir: Path, frame_id: str) -> bool:
    return (output_dir / frame_id / "metadata.json").exists()


def is_frame_failed(output_dir: Path, frame_id: str) -> bool:
    """Check if a frame was previously marked as permanently failed."""
    return (output_dir / frame_id / "_FAILED").exists()


def mark_frame_failed(output_dir: Path, frame_id: str, reason: str):
    """Mark a frame as permanently failed so it won't be retried."""
    fail_dir = output_dir / frame_id
    fail_dir.mkdir(parents=True, exist_ok=True)
    with open(fail_dir / "_FAILED", 'w') as f:
        f.write(f"{datetime.now().isoformat()} | {reason}\n")


class _FrameTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _FrameTimeout("Frame processing timed out")


# =============================================================================
# CCA (Connected Component Analysis) - Standalone
# =============================================================================

def _clean_transparent_pixels(arr: np.ndarray, alpha_threshold: int = 10) -> np.ndarray:
    result = arr.copy()
    transparent = result[:, :, 3] < alpha_threshold
    result[transparent, 0] = 0
    result[transparent, 1] = 0
    result[transparent, 2] = 0
    return result


def run_cca_standalone(
    rgba_arr: np.ndarray,
    min_area: int = CCA_MIN_AREA,
    alpha_threshold: int = CCA_ALPHA_THRESHOLD,
) -> List[np.ndarray]:
    """
    Split components from an RGBA array via CCA and return a list of
    canvas-size RGBA arrays.
    """
    from scipy import ndimage

    alpha = rgba_arr[:, :, 3]
    binary = (alpha > alpha_threshold).astype(np.uint8)
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_features = ndimage.label(binary, structure=structure)

    components = []
    for comp_idx in range(1, num_features + 1):
        comp_mask = (labeled == comp_idx).astype(np.uint8)
        area = int(comp_mask.sum())
        if area < min_area:
            continue

        layer_arr = rgba_arr.copy()
        layer_arr[:, :, 3] = np.where(comp_mask > 0, layer_arr[:, :, 3], 0)
        layer_arr = _clean_transparent_pixels(layer_arr, alpha_threshold=alpha_threshold)
        components.append(layer_arr)

    # Sort by area (largest first)
    components.sort(key=lambda x: (x[:, :, 3] > 0).sum(), reverse=True)
    return components


# =============================================================================
# Reconstruction (Qwen-compatible)
# =============================================================================

def create_reconstructions(frame_output_dir: Path, logger: Optional[logging.Logger] = None) -> bool:
    try:
        layer_paths = sorted(frame_output_dir.glob("layer_*.png"))
        if not layer_paths:
            return False

        base_layer = Image.open(layer_paths[0]).convert("RGBA")
        canvas_w, canvas_h = base_layer.size

        reconstructed = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        layers_data = []

        # Reverse order: background (last layer, bottom) → front layers (top)
        for p in reversed(layer_paths):
            img = Image.open(p).convert("RGBA")
            if img.size != (canvas_w, canvas_h):
                img = img.resize((canvas_w, canvas_h), Image.LANCZOS)
            reconstructed = Image.alpha_composite(reconstructed, img)
            layers_data.append((p, img))

        reconstructed_path = frame_output_dir / "reconstructed.png"
        reconstructed.save(reconstructed_path)

        # Bordered reconstruction
        border_color = (255, 150, 200, 200)
        glow_color = (255, 180, 220, 100)
        border_width = 3
        glow_width = 5

        result = reconstructed.copy()
        glow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        border_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        glow_arr_base = np.array(glow_layer)
        border_arr_base = np.array(border_layer)

        for _, layer_img in layers_data:
            elem_arr = np.array(layer_img)
            alpha = elem_arr[:, :, 3]
            binary_mask = (alpha > 128).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            for i in range(glow_width, 0, -1):
                alpha_val = int(glow_color[3] * (1 - i / (glow_width + 2)))
                cv2.drawContours(glow_arr_base, contours, -1, (*glow_color[:3], alpha_val), thickness=i * 2)
            cv2.drawContours(border_arr_base, contours, -1, border_color, thickness=border_width)

        glow_layer = Image.fromarray(glow_arr_base)
        border_layer = Image.fromarray(border_arr_base)
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=4))
        result = Image.alpha_composite(result, glow_layer)
        result = Image.alpha_composite(result, border_layer)

        bordered_path = frame_output_dir / "reconstructed_bordered.png"
        result.save(bordered_path)
        return True

    except Exception as e:
        if logger:
            logger.error(f"Failed to create reconstructions: {e}")
        return False


# =============================================================================
# Core: LayerD + LaMa Pipeline
# =============================================================================

def _maybe_resize_for_processing(image_path: str, max_long_side: int, logger: logging.Logger) -> Tuple[str, float]:
    """
    If image exceeds max_long_side, save a resized copy and return (new_path, scale).
    Otherwise return (original_path, 1.0).
    """
    img = Image.open(image_path)
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return image_path, 1.0

    scale = max_long_side / long_side
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info(f"  Resizing {w}x{h} → {new_w}x{new_h} (scale={scale:.3f}) for GPU processing")
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    resized.save(tmp.name)
    return tmp.name, scale


def run_layerd_lama_pipeline(
    image_path: str,
    frame_output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    LayerD + LaMa iterative extraction pipeline.
    Uses the standalone tools under BASELINES/tool_backends (no ToolGPUManager).
    Large images are resized down for GPU processing, then masks/layers
    are scaled back to original resolution.
    """
    import torch
    from BASELINES.tool_backends.tools.layerd_tool import run_layerd_front
    from BASELINES.tool_backends.tools.lama_tool import run_lama

    canvas = Image.open(image_path).convert("RGBA")
    W, H = canvas.size  # original size

    # Resize for GPU processing if needed
    proc_img_path, scale = _maybe_resize_for_processing(image_path, MAX_IMAGE_LONG_SIDE, logger)
    need_upscale = scale < 1.0
    if need_upscale:
        proc_canvas = Image.open(proc_img_path)
        pW, pH = proc_canvas.size
    else:
        pW, pH = W, H

    current_img_path = proc_img_path

    layers = []
    iteration = 0

    while iteration < MAX_LAYERD_ITERATIONS:
        logger.info(f"  LayerD iteration {iteration + 1}/{MAX_LAYERD_ITERATIONS}")

        # 1. LayerD front extraction (at processing resolution)
        try:
            result = run_layerd_front(current_img_path)
        except Exception as e:
            logger.warning(f"  LayerD failed at iteration {iteration}: {e}")
            break

        front_rgb_path = result["front_rgb"]
        front_mask_path = result["front_mask"]

        # 2. Check if mask is empty
        mask = cv2.imread(front_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.max() == 0:
            logger.info(f"  Empty mask at iteration {iteration}, stopping")
            break

        # 3. Build RGBA — upscale to original size if needed
        front_bgr = cv2.imread(front_rgb_path, cv2.IMREAD_COLOR)
        if front_bgr is None:
            logger.warning(f"  Failed to read front_rgb at iteration {iteration}")
            break

        if need_upscale:
            front_bgr = cv2.resize(front_bgr, (W, H), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        front_rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
        front_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        front_rgba[:, :, :3] = front_rgb
        front_rgba[:, :, 3] = mask

        # 4. Save front layer RGBA directly (no CCA)
        layer_idx = len(layers)
        layer_path = frame_output_dir / f"layer_{layer_idx:02d}.png"
        Image.fromarray(front_rgba).save(layer_path)

        # Compute bbox
        alpha_ch = front_rgba[:, :, 3]
        rows = np.any(alpha_ch > 0, axis=1)
        cols = np.any(alpha_ch > 0, axis=0)
        if rows.any() and cols.any():
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            bbox = [int(x1), int(y1), int(x2) + 1, int(y2) + 1]
        else:
            bbox = [0, 0, W, H]

        layers.append({
            "path": str(layer_path),
            "bbox": bbox,
            "z_order": layer_idx,
            "iteration": iteration,
            "area": int((alpha_ch > 0).sum()),
        })

        # 6. LaMa inpaint (at processing resolution)
        try:
            inpainted_path = run_lama(current_img_path, front_mask_path)
            current_img_path = inpainted_path
        except Exception as e:
            logger.warning(f"  LaMa failed at iteration {iteration}: {e}")
            break

        iteration += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 7. Save background (residual) as last layer — upscale if needed
    bg_idx = len(layers)
    bg_path = frame_output_dir / f"layer_{bg_idx:02d}.png"

    bg_img = Image.open(current_img_path).convert("RGB")
    bg_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    bg_arr = np.array(bg_img)
    if bg_arr.shape[:2] == (H, W):
        bg_rgba[:, :, :3] = bg_arr
    else:
        bg_resized = cv2.resize(bg_arr, (W, H))
        bg_rgba[:, :, :3] = bg_resized
    bg_rgba[:, :, 3] = 255
    Image.fromarray(bg_rgba).save(bg_path)

    layers.append({
        "path": str(bg_path),
        "bbox": [0, 0, W, H],
        "z_order": bg_idx,
        "iteration": iteration,
        "type": "background",
    })

    return {
        "total_iterations": iteration,
        "num_layers": len(layers),
        "layers": layers,
        "canvas_size": [W, H],
    }


# =============================================================================
# Worker Process
# =============================================================================

def worker_process(
    worker_id: int,
    gpu_id: int,
    frame_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: Path,
):
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    worker_logger = setup_logging(
        output_dir / f"worker_{worker_id}.log",
        name=f"baseline1_w{worker_id}"
    )
    worker_logger.info(f"Worker {worker_id} started on GPU {gpu_id}")

    # Set up SIGALRM handler for per-frame timeout
    signal.signal(signal.SIGALRM, _timeout_handler)

    while True:
        try:
            item = frame_queue.get(timeout=1.0)
        except:
            if frame_queue.empty():
                break
            continue

        if item is None:
            break

        frame_id, image_path = item
        start_time = time.time()

        # Start per-frame timeout alarm
        signal.alarm(FRAME_TIMEOUT_SEC)

        try:
            frame_output_dir = output_dir / frame_id
            frame_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy input image
            input_dest = frame_output_dir / INPUT_IMAGE_NAME
            shutil.copy2(image_path, input_dest)

            # Run pipeline
            pipeline_result = run_layerd_lama_pipeline(
                str(image_path),
                frame_output_dir,
                worker_logger,
            )

            # Save metadata.json (Qwen-compatible)
            elapsed = time.time() - start_time
            original_size = Image.open(image_path).size

            metadata = {
                "frame_id": frame_id,
                "source_image": str(image_path),
                "input_image": str(input_dest),
                "original_size": list(original_size),
                "num_layers": pipeline_result["num_layers"],
                "layer_paths": [l["path"] for l in pipeline_result["layers"]],
                "gpu_id": gpu_id,
                "worker_id": worker_id,
                "processing_time_sec": elapsed,
                "timestamp": datetime.now().isoformat(),
                "method": "layerd_lama",
                "max_iterations": MAX_LAYERD_ITERATIONS,
                "total_iterations": pipeline_result["total_iterations"],
            }

            metadata_path = frame_output_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Create reconstructions
            create_reconstructions(frame_output_dir, worker_logger)

            result_queue.put({
                "worker_id": worker_id,
                "frame_id": frame_id,
                "success": True,
                "num_layers": pipeline_result["num_layers"],
                "iterations": pipeline_result["total_iterations"],
                "elapsed": elapsed,
            })

            worker_logger.info(
                f"  Done: {frame_id} | {pipeline_result['num_layers']} layers | "
                f"{pipeline_result['total_iterations']} iters | {elapsed:.1f}s"
            )

        except _FrameTimeout:
            elapsed = time.time() - start_time
            mark_frame_failed(output_dir, frame_id, f"Timeout after {elapsed:.0f}s")
            result_queue.put({
                "worker_id": worker_id,
                "frame_id": frame_id,
                "success": False,
                "error": f"Timeout after {elapsed:.0f}s",
                "trace": "",
            })
            worker_logger.error(f"  TIMEOUT: {frame_id} after {elapsed:.0f}s — marked as failed, skipping")

        except Exception as e:
            elapsed = time.time() - start_time
            err_str = str(e)
            # Mark as permanently failed if it's a CUDA OOM or similar unrecoverable error
            if "CUDA out of memory" in err_str or "CUDA error" in err_str:
                mark_frame_failed(output_dir, frame_id, f"CUDA error: {err_str[:200]}")
                worker_logger.error(f"  CUDA FAIL (permanent): {frame_id}: {err_str[:200]}")
            result_queue.put({
                "worker_id": worker_id,
                "frame_id": frame_id,
                "success": False,
                "error": err_str,
                "trace": traceback.format_exc(),
            })
            worker_logger.error(f"  Failed: {frame_id}: {e}")

        finally:
            # Cancel any pending alarm
            signal.alarm(0)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker_logger.info(f"Worker {worker_id} shutdown complete")


# =============================================================================
# Main Runner
# =============================================================================

def run_baseline1(
    data_dir: Path,
    output_dir: Path,
    gpu_ids: List[int],
    workers_per_gpu: int = 1,
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip_completed: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "baseline1_run.log"
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("Baseline 1: LayerD + LaMa Iterative Extraction")
    logger.info("=" * 70)
    logger.info(f"data_dir: {data_dir}")
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Workers per GPU: {workers_per_gpu}")
    total_workers = len(gpu_ids) * workers_per_gpu
    logger.info(f"Total workers: {total_workers}")
    logger.info(f"Output: {output_dir}")

    if workers_per_gpu < 1:
        raise ValueError(f"workers_per_gpu must be >= 1, got {workers_per_gpu}")

    # Load frames
    frames = load_frame_list(data_dir)
    logger.info(f"Found {len(frames)} total frames")

    # Filter completed and permanently failed frames
    frames_to_process = []
    skipped = 0
    skipped_failed = 0
    for frame in frames:
        if skip_completed and is_frame_completed(output_dir, frame.frame_id):
            skipped += 1
            continue
        if is_frame_failed(output_dir, frame.frame_id):
            skipped_failed += 1
            continue
        frames_to_process.append(frame)

    if limit:
        frames_to_process = frames_to_process[:limit]

    logger.info(f"Will process {len(frames_to_process)} frames (skipped {skipped} completed, {skipped_failed} failed)")

    if dry_run:
        logger.info("[DRY RUN] Would process:")
        for f in frames_to_process[:20]:
            logger.info(f"  {f.frame_id}")
        return {"dry_run": True}

    # Queue setup
    frame_queue = mp.Queue()
    result_queue = mp.Queue()

    for frame in frames_to_process:
        frame_queue.put((frame.frame_id, str(frame.image_path)))

    for _ in range(total_workers):
        frame_queue.put(None)

    # Start workers (N per GPU)
    workers = []
    worker_id = 0
    for gpu_id in gpu_ids:
        for _ in range(workers_per_gpu):
            p = mp.Process(
                target=worker_process,
                args=(worker_id, gpu_id, frame_queue, result_queue, output_dir),
            )
            p.start()
            workers.append(p)
            worker_id += 1

    # Collect results with tqdm progress
    total = len(frames_to_process)
    results = {"success": 0, "failed": 0, "errors": []}
    pbar = tqdm(total=total, desc="Baseline1", unit="frame", ncols=100)

    while pbar.n < total:
        try:
            r = result_queue.get(timeout=2.0)
        except:
            # Check if all workers dead but results not fully collected
            if all(not p.is_alive() for p in workers):
                break
            continue
        if r["success"]:
            results["success"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(r)
        pbar.update(1)
        pbar.set_postfix_str(f"ok={results['success']} fail={results['failed']}", refresh=True)

    pbar.close()

    # Wait for workers to fully exit
    for p in workers:
        p.join(timeout=10)

    logger.info("=" * 70)
    logger.info(f"DONE: {results['success']} success, {results['failed']} failed")
    logger.info("=" * 70)

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Baseline 1: LayerD + LaMa Iterative Extraction"
    )
    parser.add_argument("--data_dir", "-i", type=str, required=True,
                        help="Path to the Figma dataset directory (containing valid_frames/, unit_images/)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Path to the output directory")
    parser.add_argument("--gpu", type=str, default="0",
                        help="comma-separated GPU ids (set to your own; e.g. 0 or 0,1)")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Number of workers per GPU")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of frames to process")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_skip", action="store_true")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]

    run_baseline1(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_completed=not args.no_skip,
    )


if __name__ == "__main__":
    main()
