#!/usr/bin/env python3
"""
BASELINES/run_qwen_figma.py - Qwen Image Layered baseline experiment runner (Figma)

Runs the Qwen Image Layered model as a baseline over a whole Figma dataset directory.
The provided qwen_gpus are divided into groups of qwen_pair_size for parallel processing.
Processes EVERY episode found under <data_dir>/valid_frames/.

Note: Replace the GPU id placeholder <QWEN_GPU_IDS> with your own comma-separated GPU ids.

Usage:
    # Basic usage
    python run_qwen_figma.py --data_dir figma_data --output_dir outputs/qwen_figma --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2

    # Split the GPUs into two pairs to process in parallel
    # e.g. with --qwen_gpus a,b,c,d and --qwen_pair_size 2:
    # -> pair (a,b): frame 0, 2, 4, ...
    # -> pair (c,d): frame 1, 3, 5, ...

    # Dry run (preview without executing)
    python run_qwen_figma.py --data_dir figma_data --output_dir outputs/qwen_figma --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --dry_run

    # For testing (first 10 frames only)
    python run_qwen_figma.py --data_dir figma_data --output_dir outputs/qwen_figma --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --limit 10

Directory Structure:
    Input:
        <data_dir>/valid_frames/*.json
        <data_dir>/unit_images/...

    Output:
        <output_dir>/{frame_id}/
            - input.png          # Original input image
            - layer_00.png
            - layer_01.png
            - ...
            - metadata.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import os
import logging
import multiprocessing as mp
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import traceback
import cv2
import numpy as np
from PIL import Image, ImageFilter


# =============================================================================
# Configuration
# =============================================================================

# Qwen default parameters
DEFAULT_NUM_LAYERS = 4
DEFAULT_SEED = 777
DEFAULT_RESOLUTION = 640
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_ALPHA_THRESHOLD = 0

# Input image filename
INPUT_IMAGE_NAME = "input.png"


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Path, name: str = "qwen_baseline") -> logging.Logger:
    """Setup logger for this run."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console)
    
    # File handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# Path Resolution
# =============================================================================

# =============================================================================
# GPU Configuration
# =============================================================================

def parse_gpu_list(gpu_str: Optional[str]) -> List[int]:
    """Parse a GPU list string (e.g. '2,3,4,5' -> [2, 3, 4, 5])."""
    if not gpu_str:
        return []
    try:
        return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]
    except ValueError:
        return []


def create_gpu_pairs(gpu_ids: List[int], pair_size: int) -> List[Tuple[int, ...]]:
    """Split a list of GPU IDs into tuples of size pair_size."""
    if not gpu_ids or pair_size <= 0:
        return []

    pairs = []
    for i in range(0, len(gpu_ids), pair_size):
        pair = tuple(gpu_ids[i:i + pair_size])
        if len(pair) == pair_size:  # Use complete pairs only
            pairs.append(pair)
    
    return pairs


# =============================================================================
# Frame Data Loading
# =============================================================================

@dataclass
class FrameInfo:
    """Data class holding frame information."""
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
            continue

    return frames


def is_frame_completed(output_dir: Path, frame_id: str) -> bool:
    """Check whether a frame has already been processed."""
    metadata_path = output_dir / frame_id / "metadata.json"
    return metadata_path.exists()


def is_input_image_missing(output_dir: Path, frame_id: str) -> bool:
    """Check whether a completed frame is missing its input image."""
    input_image_path = output_dir / frame_id / INPUT_IMAGE_NAME
    return not input_image_path.exists()


def create_reconstructions(
    frame_output_dir: Path, 
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Generate reconstructed.png and reconstructed_bordered.png from the saved layer images.
    Adapts the logic of REDESIGN/reconstruction.py to the Qwen Layer structure.
    """
    try:
        # 1. Find and sort layer images
        layer_paths = sorted(frame_output_dir.glob("layer_*.png"))
        if not layer_paths:
            return False

        # 2. Prepare canvas (size based on the first layer)
        base_layer = Image.open(layer_paths[0]).convert("RGBA")
        canvas_w, canvas_h = base_layer.size

        # ---------------------------------------------------------
        # A. Vanilla Reconstruction (reconstructed.png)
        # ---------------------------------------------------------
        reconstructed = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        layers_data = [] # Store (image_path, pil_image) tuples

        for p in layer_paths:
            img = Image.open(p).convert("RGBA")
            if img.size != (canvas_w, canvas_h):
                img = img.resize((canvas_w, canvas_h), Image.LANCZOS)
            reconstructed = Image.alpha_composite(reconstructed, img)
            layers_data.append((p, img))
            
        reconstructed_path = frame_output_dir / "reconstructed.png"
        reconstructed.save(reconstructed_path)
        
        # ---------------------------------------------------------
        # B. Bordered Reconstruction (reconstructed_bordered.png)
        # ---------------------------------------------------------
        # Settings (same as reconstruction.py)
        border_color = (255, 150, 200, 200)  # Light Pink
        glow_color = (255, 180, 220, 100)    # Soft Pink
        border_width = 3
        glow_width = 5

        # Create a copy
        result = reconstructed.copy()

        # Glow Layer (for the blur effect)
        glow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        # Border Layer (for the sharp outline)
        border_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        glow_arr_base = np.array(glow_layer)
        border_arr_base = np.array(border_layer)

        for _, layer_img in layers_data:
            # Extract the alpha channel
            elem_arr = np.array(layer_img)
            alpha = elem_arr[:, :, 3]

            # Build a binary mask (threshold 128)
            binary_mask = (alpha > 128).astype(np.uint8) * 255

            # Find contours with OpenCV
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            # Draw the glow (multiple overlapping passes for a gradient effect)
            for i in range(glow_width, 0, -1):
                alpha_val = int(glow_color[3] * (1 - i / (glow_width + 2)))
                cv2.drawContours(
                    glow_arr_base, 
                    contours, 
                    -1, 
                    (*glow_color[:3], alpha_val),
                    thickness=i * 2
                )
            
            # Draw the border
            cv2.drawContours(
                border_arr_base,
                contours,
                -1,
                border_color,
                thickness=border_width
            )

        # Convert arrays back to images
        glow_layer = Image.fromarray(glow_arr_base)
        border_layer = Image.fromarray(border_arr_base)

        # Apply blur to the glow
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=4))

        # Composite: Base -> Glow -> Border
        result = Image.alpha_composite(result, glow_layer)
        result = Image.alpha_composite(result, border_layer)
        
        bordered_path = frame_output_dir / "reconstructed_bordered.png"
        result.save(bordered_path)
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to create reconstructions for {frame_output_dir.name}: {e}")
        else:
            print(f"Failed to create reconstructions: {e}")
        return False


# =============================================================================
# Input Image Backfill
# =============================================================================


def backfill_missing_files(
    frames: List[FrameInfo],
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, int]:
    """
    Backfill missing files (input image, reconstruction) for completed frames.
    """
    stats = {"input_image": 0, "reconstruction": 0}

    for frame in frames:
        frame_output_dir = output_dir / frame.frame_id

        # Check whether the frame is completed (metadata present)
        if not is_frame_completed(output_dir, frame.frame_id):
            continue

        # 1. Backfill the input image
        input_image_path = frame_output_dir / INPUT_IMAGE_NAME
        if not input_image_path.exists():
            try:
                shutil.copy2(frame.image_path, input_image_path)
                stats["input_image"] += 1
                logger.debug(f"Backfilled input image for {frame.frame_id}")
            except Exception as e:
                logger.warning(f"Failed to backfill input image for {frame.frame_id}: {e}")

        # 2. Backfill the reconstruction images
        recon_path = frame_output_dir / "reconstructed.png"
        border_path = frame_output_dir / "reconstructed_bordered.png"

        if not recon_path.exists() or not border_path.exists():
            # Check whether layer files exist
            if list(frame_output_dir.glob("layer_*.png")):
                success = create_reconstructions(frame_output_dir, logger)
                if success:
                    stats["reconstruction"] += 1
                    logger.debug(f"Backfilled reconstructions for {frame.frame_id}")
    
    return stats


# =============================================================================
# Qwen Worker Process
# =============================================================================

def worker_process(
    worker_id: int,
    gpu_pair: Tuple[int, ...],
    frame_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: Path,
    qwen_params: Dict[str, Any],
):
    """
    Worker process that runs the Qwen model on a single GPU pair.
    """
    import os
    import gc
    import torch
    import tempfile
    import shutil
    from PIL import Image
    import numpy as np

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_pair))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    offload_dir = tempfile.mkdtemp(prefix=f"qwen_baseline_{worker_id}_")
    pipeline = None
    
    try:
        # Load pipeline
        print(f"[Worker {worker_id}] Loading Qwen pipeline on GPUs {gpu_pair}...")
        from diffusers import QwenImageLayeredPipeline
        
        pipeline = QwenImageLayeredPipeline.from_pretrained(
            "Qwen/Qwen-Image-Layered",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            offload_folder=offload_dir,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )
        print(f"[Worker {worker_id}] Pipeline loaded successfully")

        # Frame processing loop
        while True:
            try:
                item = frame_queue.get(timeout=1.0)
            except:
                # Check whether the queue is empty
                if frame_queue.empty():
                    break
                continue

            if item is None:  # Termination signal
                break

            frame_id, image_path = item
            start_time = time.time()

            try:
                # Create the output directory
                frame_output_dir = output_dir / frame_id
                frame_output_dir.mkdir(parents=True, exist_ok=True)

                # Load the image
                image = Image.open(image_path).convert("RGBA")
                original_size = image.size

                # Save the input image (copy of the original)
                input_image_dest = frame_output_dir / INPUT_IMAGE_NAME
                shutil.copy2(image_path, input_image_dest)

                # Qwen inference
                inputs = {
                    "image": image,
                    "generator": torch.Generator(device="cpu").manual_seed(qwen_params["seed"]),
                    "num_inference_steps": qwen_params["num_inference_steps"],
                    "layers": qwen_params["num_layers"],
                    "resolution": qwen_params["resolution"],
                    "true_cfg_scale": qwen_params["true_cfg_scale"],
                    "cfg_normalize": True,
                    "use_en_prompt": True,
                }
                
                with torch.inference_mode():
                    output = pipeline(**inputs)
                    output_images = output.images[0]
                
                # Save layers
                layer_paths = []
                for i, layer_img in enumerate(output_images):
                    layer_img = layer_img.convert("RGBA")
                    if layer_img.size != original_size:
                        layer_img = layer_img.resize(original_size, Image.LANCZOS)

                    # Filter out semi-transparent pixels
                    arr = np.array(layer_img)
                    alpha_threshold = qwen_params["alpha_threshold"]
                    mask = arr[:, :, 3] < alpha_threshold
                    arr[mask] = [0, 0, 0, 0]
                    
                    layer_path = frame_output_dir / f"layer_{i:02d}.png"
                    Image.fromarray(arr).save(layer_path)
                    layer_paths.append(str(layer_path))
                
                # Save metadata
                elapsed = time.time() - start_time
                metadata = {
                    "frame_id": frame_id,
                    "source_image": str(image_path),
                    "input_image": str(input_image_dest),
                    "original_size": list(original_size),
                    "num_layers": len(layer_paths),
                    "layer_paths": layer_paths,
                    "gpu_pair": list(gpu_pair),
                    "worker_id": worker_id,
                    "processing_time_sec": elapsed,
                    "timestamp": datetime.now().isoformat(),
                    "qwen_params": qwen_params,
                }
                
                metadata_path = frame_output_dir / "metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                create_reconstructions(frame_output_dir)
                
                result_queue.put({
                    "worker_id": worker_id,
                    "frame_id": frame_id,
                    "success": True,
                    "num_layers": len(layer_paths),
                    "elapsed": elapsed,
                })
                
            except Exception as e:
                result_queue.put({
                    "worker_id": worker_id,
                    "frame_id": frame_id,
                    "success": False,
                    "error": str(e),
                    "trace": traceback.format_exc(),
                })
            
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        traceback.print_exc()
    
    finally:
        if pipeline is not None:
            del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        if os.path.exists(offload_dir):
            shutil.rmtree(offload_dir, ignore_errors=True)
        
        print(f"[Worker {worker_id}] Shutdown complete")


# =============================================================================
# Main Runner
# =============================================================================

def run_qwen_baseline(
    data_dir: Path,
    output_dir: Path,
    qwen_gpus: List[int],
    qwen_pair_size: int,
    num_layers: int = DEFAULT_NUM_LAYERS,
    seed: int = DEFAULT_SEED,
    resolution: int = DEFAULT_RESOLUTION,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    true_cfg_scale: float = DEFAULT_TRUE_CFG_SCALE,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip_completed: bool = True,
) -> Dict[str, Any]:
    """
    Run the Qwen Image Layered baseline experiment over a whole Figma dataset directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging setup
    log_file = output_dir / "qwen_baseline.log"
    logger = setup_logging(log_file)

    # Create GPU pairs
    gpu_pairs = create_gpu_pairs(qwen_gpus, qwen_pair_size)
    if not gpu_pairs:
        raise ValueError(f"Cannot create GPU pairs from {qwen_gpus} with pair_size {qwen_pair_size}")

    logger.info("=" * 70)
    logger.info(f"Qwen Image Layered Baseline (Figma)")
    logger.info("=" * 70)
    logger.info(f"data_dir: {data_dir}")
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"GPU pairs: {gpu_pairs}")
    logger.info(f"num_layers: {num_layers}, resolution: {resolution}")
    logger.info(f"dry_run: {dry_run}, limit: {limit}")

    # Load the frame list
    frames = load_frame_list(data_dir)
    logger.info(f"Found {len(frames)} valid frames")
    
    if not frames:
        logger.warning("No valid frames found!")
        return {"error": "No valid frames found"}
    
    # Backfill input images missing from completed frames
    logger.info("Checking for missing input images in completed frames...")


    backfill_stats = backfill_missing_files(frames, output_dir, logger)
    if backfill_stats["input_image"] > 0 or backfill_stats["reconstruction"] > 0:
        logger.info(f"Backfilled: {backfill_stats['input_image']} inputs, {backfill_stats['reconstruction']} reconstructions")
    else:
        logger.info("No missing files found in completed frames")




    
    # Filter out completed frames
    if skip_completed:
        pending_frames = [f for f in frames if not is_frame_completed(output_dir, f.frame_id)]
        completed_count = len(frames) - len(pending_frames)
        logger.info(f"Already completed: {completed_count}, Pending: {len(pending_frames)}")
    else:
        pending_frames = frames

    # Apply limit
    if limit:
        pending_frames = pending_frames[:limit]
        logger.info(f"Limited to {len(pending_frames)} frames")
    
    if dry_run:
        logger.info("\n[DRY RUN] Would process these frames:")
        for i, frame in enumerate(pending_frames[:20]):
            logger.info(f"  [{i+1:3d}] {frame.frame_id}")
        if len(pending_frames) > 20:
            logger.info(f"  ... and {len(pending_frames) - 20} more")
        
        logger.info(f"\nGPU pair assignments:")
        for i, pair in enumerate(gpu_pairs):
            assigned = len([f for j, f in enumerate(pending_frames) if j % len(gpu_pairs) == i])
            logger.info(f"  Worker {i} (GPUs {pair}): {assigned} frames")
        
        return {
            "dry_run": True,
            "total_frames": len(pending_frames),
            "gpu_pairs": [list(p) for p in gpu_pairs],
            "backfilled_input_images": backfill_stats["input_image"],
        }
    
    # Exit if there are no frames to process
    if not pending_frames:
        logger.info("No pending frames to process. All done!")
        return {
            "processed": [],
            "failed": [],
            "backfilled_stats": backfill_stats,
            "message": "All frames already completed",
        }
    
    # Qwen parameters
    qwen_params = {
        "num_layers": num_layers,
        "seed": seed,
        "resolution": resolution,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "alpha_threshold": alpha_threshold,
    }

    # Multiprocessing setup
    mp.set_start_method("spawn", force=True)

    frame_queue = mp.Queue()
    result_queue = mp.Queue()

    # Add frames to the queue
    for frame in pending_frames:
        frame_queue.put((frame.frame_id, str(frame.image_path)))

    # Add termination signals
    for _ in gpu_pairs:
        frame_queue.put(None)

    # Start worker processes
    workers = []
    for i, pair in enumerate(gpu_pairs):
        p = mp.Process(
            target=worker_process,
            args=(i, pair, frame_queue, result_queue, output_dir, qwen_params),
            daemon=True,
        )
        p.start()
        workers.append(p)
        logger.info(f"Started worker {i} on GPUs {pair}")
    
    # Collect results
    results = {
        "start_time": datetime.now().isoformat(),
        "gpu_pairs": [list(p) for p in gpu_pairs],
        "qwen_params": qwen_params,
        "backfilled_stats": backfill_stats,
        "processed": [],
        "failed": [],
    }
    
    total = len(pending_frames)
    processed = 0
    
    try:
        while processed < total:
            try:
                result = result_queue.get(timeout=600)  # 10-minute timeout
            except:
                # Check worker status
                alive = sum(1 for w in workers if w.is_alive())
                if alive == 0:
                    logger.warning("All workers have stopped")
                    break
                continue
            
            processed += 1
            
            if result["success"]:
                logger.info(f"[{processed}/{total}] ✓ {result['frame_id']} "
                          f"({result['num_layers']} layers, {result['elapsed']:.1f}s)")
                results["processed"].append({
                    "frame_id": result["frame_id"],
                    "num_layers": result["num_layers"],
                    "elapsed": result["elapsed"],
                })
            else:
                logger.error(f"[{processed}/{total}] ✗ {result['frame_id']}: {result['error']}")
                results["failed"].append({
                    "frame_id": result["frame_id"],
                    "error": result["error"],
                })
            
            # Save intermediate results
            if processed % 10 == 0:
                results["end_time"] = datetime.now().isoformat()
                results_file = output_dir / f"qwen_baseline_results.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
    
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
    
    finally:
        # Wait for workers to finish
        for w in workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()

    # Save final results
    results["end_time"] = datetime.now().isoformat()
    results_file = output_dir / f"qwen_baseline_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total processed: {len(results['processed'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    logger.info(f"Backfilled: {backfill_stats['input_image']} inputs, {backfill_stats['reconstruction']} reconstructions")
    
    if results["failed"]:
        logger.info("\nFailed frames:")
        for f in results["failed"][:10]:
            logger.info(f"  - {f['frame_id']}: {f['error'][:80]}")
        if len(results["failed"]) > 10:
            logger.info(f"  ... and {len(results['failed']) - 10} more")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen Image Layered Baseline Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: Replace the GPU id placeholder <QWEN_GPU_IDS> below with your own
comma-separated GPU ids (e.g. 0,1,2,3).

Examples:
    # Basic run: use the GPUs in pairs of 2
    python run_qwen_figma.py --data_dir figma_data --output_dir outputs/qwen_figma --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2

    # RTX 3090: use pairs of 3 GPUs
    python run_qwen_figma.py --data_dir figma_data --output_dir outputs/qwen_figma --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 3

    # Dry run (preview without executing)
    python run_qwen_figma.py --data_dir figma_data --output_dir outputs/qwen_figma --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --dry_run

    # For testing (first 10 frames only)
    python run_qwen_figma.py --data_dir figma_data --output_dir outputs/qwen_figma --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --limit 10

    # Adjust the number of layers
    python run_qwen_figma.py --data_dir figma_data --output_dir outputs/qwen_figma --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --num_layers 6
        """
    )

    parser.add_argument(
        "--data_dir", "-i",
        type=str,
        required=True,
        help="Path to the Figma dataset directory (containing valid_frames/, unit_images/)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Path to the output directory"
    )
    parser.add_argument(
        "--qwen_gpus",
        type=str,
        required=True,
        help="GPU IDs for the Qwen model, comma-separated and user-specific (e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--qwen_pair_size",
        type=int,
        required=True,
        help="Number of GPUs per Qwen pair (e.g., 2 for A6000, 3 for RTX3090)"
    )

    # Qwen parameters
    parser.add_argument(
        "--num_layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help=f"Number of layers to generate (default: {DEFAULT_NUM_LAYERS})"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Resolution for Qwen (default: {DEFAULT_RESOLUTION})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    
    # Execution options
    parser.add_argument(
        "--dry_run", "-d",
        action="store_true",
        help="Show what would be done without actually running"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of frames to process (for testing)"
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Don't skip already completed frames (re-process all)"
    )

    args = parser.parse_args()

    # Parse GPUs
    qwen_gpus = parse_gpu_list(args.qwen_gpus)
    if not qwen_gpus:
        print(f"Error: Invalid qwen_gpus: {args.qwen_gpus}")
        sys.exit(1)

    try:
        results = run_qwen_baseline(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            qwen_gpus=qwen_gpus,
            qwen_pair_size=args.qwen_pair_size,
            num_layers=args.num_layers,
            resolution=args.resolution,
            seed=args.seed,
            dry_run=args.dry_run,
            limit=args.limit,
            skip_completed=not args.no_skip,
        )

        if not args.dry_run:
            print(f"\nResults saved to: {Path(args.output_dir) / 'qwen_baseline_results.json'}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()