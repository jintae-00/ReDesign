#!/usr/bin/env python3
"""
baselines/run_qwen_crello.py - Qwen Image Layered baseline runner for the Crello test set

Processes EVERY crello_test_* record directory under <data_dir> in parallel with one
worker per GPU pair, and shows a unified tqdm progress bar.

Note: Replace the GPU id placeholder <QWEN_GPU_IDS> below with your own
comma-separated GPU ids.

Usage:
    # Basic: split the GPUs into pairs of 2
    python run_qwen_crello.py --data_dir crello_data/records --output_dir outputs/qwen_crello --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2

    # Testing (10 records only)
    python run_qwen_crello.py --data_dir crello_data/records --output_dir outputs/qwen_crello --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --limit 10

    # Dry run
    python run_qwen_crello.py --data_dir crello_data/records --output_dir outputs/qwen_crello --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --dry_run

Directory Structure:
    Input:
        <data_dir>/crello_test_XXXX/
            - composite.png

    Output:
        <output_dir>/{record_id}/
            - input.png
            - layer_00.png, layer_01.png, ...
            - reconstructed.png
            - reconstructed_bordered.png
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

INPUT_IMAGE_NAME = "input.png"


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Path, name: str = "qwen_crello_baseline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console)

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
    if not gpu_str:
        return []
    try:
        return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]
    except ValueError:
        return []


def create_gpu_pairs(gpu_ids: List[int], pair_size: int) -> List[Tuple[int, ...]]:
    if not gpu_ids or pair_size <= 0:
        return []
    pairs = []
    for i in range(0, len(gpu_ids), pair_size):
        pair = tuple(gpu_ids[i:i + pair_size])
        if len(pair) == pair_size:
            pairs.append(pair)
    return pairs


# =============================================================================
# Record Data Loading (corresponds to FrameInfo in the Figma runner)
# =============================================================================

@dataclass
class RecordInfo:
    """Crello record information."""
    record_id: str
    image_path: Path  # Absolute path to composite.png


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

    # Sort by record_id and deduplicate (avoids symlink overlaps)
    seen = set()
    unique = []
    for r in sorted(records, key=lambda x: x.record_id):
        if r.record_id not in seen:
            seen.add(r.record_id)
            unique.append(r)

    return unique


def is_record_completed(output_dir: Path, record_id: str) -> bool:
    return (output_dir / record_id / "metadata.json").exists()


# =============================================================================
# Reconstruction (same as the original)
# =============================================================================

def create_reconstructions(
    frame_output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> bool:
    try:
        layer_paths = sorted(frame_output_dir.glob("layer_*.png"))
        if not layer_paths:
            return False

        base_layer = Image.open(layer_paths[0]).convert("RGBA")
        canvas_w, canvas_h = base_layer.size

        # A. Vanilla Reconstruction
        reconstructed = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        layers_data = []
        for p in layer_paths:
            img = Image.open(p).convert("RGBA")
            if img.size != (canvas_w, canvas_h):
                img = img.resize((canvas_w, canvas_h), Image.LANCZOS)
            reconstructed = Image.alpha_composite(reconstructed, img)
            layers_data.append((p, img))

        reconstructed_path = frame_output_dir / "reconstructed.png"
        reconstructed.save(reconstructed_path)

        # B. Bordered Reconstruction
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
            logger.error(f"Failed to create reconstructions for {frame_output_dir.name}: {e}")
        return False


# =============================================================================
# Backfill (fill in missing files for completed records)
# =============================================================================

def backfill_missing_files(
    records: List[RecordInfo],
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, int]:
    stats = {"input_image": 0, "reconstruction": 0}

    for rec in records:
        rec_output_dir = output_dir / rec.record_id
        if not is_record_completed(output_dir, rec.record_id):
            continue

        # Input image
        input_image_path = rec_output_dir / INPUT_IMAGE_NAME
        if not input_image_path.exists():
            try:
                shutil.copy2(rec.image_path, input_image_path)
                stats["input_image"] += 1
            except Exception as e:
                logger.warning(f"Failed to backfill input for {rec.record_id}: {e}")

        # Reconstruction
        recon_path = rec_output_dir / "reconstructed.png"
        border_path = rec_output_dir / "reconstructed_bordered.png"
        if not recon_path.exists() or not border_path.exists():
            if list(rec_output_dir.glob("layer_*.png")):
                if create_reconstructions(rec_output_dir, logger):
                    stats["reconstruction"] += 1

    return stats


# =============================================================================
# Qwen Worker Process
# =============================================================================

def worker_process(
    worker_id: int,
    gpu_pair: Tuple[int, ...],
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: Path,
    qwen_params: Dict[str, Any],
):
    """Worker process that runs the Qwen model on a single GPU pair."""
    import gc
    import torch
    import tempfile

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_pair))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    offload_dir = tempfile.mkdtemp(prefix=f"qwen_crello_{worker_id}_")
    pipeline = None

    try:
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

        while True:
            try:
                item = task_queue.get(timeout=1.0)
            except:
                if task_queue.empty():
                    break
                continue

            if item is None:
                break

            record_id, image_path = item
            start_time = time.time()

            try:
                rec_output_dir = output_dir / record_id
                rec_output_dir.mkdir(parents=True, exist_ok=True)

                image = Image.open(image_path).convert("RGBA")
                original_size = image.size

                # Save the input image
                input_dest = rec_output_dir / INPUT_IMAGE_NAME
                shutil.copy2(image_path, input_dest)

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
                layer_paths_saved = []
                for i, layer_img in enumerate(output_images):
                    layer_img = layer_img.convert("RGBA")
                    if layer_img.size != original_size:
                        layer_img = layer_img.resize(original_size, Image.LANCZOS)

                    arr = np.array(layer_img)
                    alpha_threshold = qwen_params["alpha_threshold"]
                    mask = arr[:, :, 3] < alpha_threshold
                    arr[mask] = [0, 0, 0, 0]

                    layer_path = rec_output_dir / f"layer_{i:02d}.png"
                    Image.fromarray(arr).save(layer_path)
                    layer_paths_saved.append(str(layer_path))

                elapsed = time.time() - start_time
                metadata = {
                    "record_id": record_id,
                    "source_image": str(image_path),
                    "input_image": str(input_dest),
                    "original_size": list(original_size),
                    "num_layers": len(layer_paths_saved),
                    "layer_paths": layer_paths_saved,
                    "gpu_pair": list(gpu_pair),
                    "worker_id": worker_id,
                    "processing_time_sec": elapsed,
                    "timestamp": datetime.now().isoformat(),
                    "qwen_params": qwen_params,
                }

                with open(rec_output_dir / "metadata.json", 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                create_reconstructions(rec_output_dir)

                result_queue.put({
                    "worker_id": worker_id,
                    "record_id": record_id,
                    "success": True,
                    "num_layers": len(layer_paths_saved),
                    "elapsed": elapsed,
                })

            except Exception as e:
                result_queue.put({
                    "worker_id": worker_id,
                    "record_id": record_id,
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
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

        if os.path.exists(offload_dir):
            shutil.rmtree(offload_dir, ignore_errors=True)

        print(f"[Worker {worker_id}] Shutdown complete")


# =============================================================================
# Main Runner
# =============================================================================

def run_qwen_crello_baseline(
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
    """Qwen Image Layered baseline — run over a whole Crello test dataset directory."""
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "qwen_crello_baseline.log"
    logger = setup_logging(log_file)

    # GPU pair
    gpu_pairs = create_gpu_pairs(qwen_gpus, qwen_pair_size)
    if not gpu_pairs:
        raise ValueError(f"Cannot create GPU pairs from {qwen_gpus} with pair_size {qwen_pair_size}")

    logger.info("=" * 70)
    logger.info(f"Qwen Crello Baseline")
    logger.info("=" * 70)
    logger.info(f"data_dir: {data_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"GPU pairs: {gpu_pairs} ({len(gpu_pairs)} workers)")

    # ---- Load all records ----
    all_records = load_record_list(data_dir)
    logger.info(f"Total records: {len(all_records)}")

    if not all_records:
        logger.warning("No records found!")
        return {"error": "No records found"}

    # Backfill
    logger.info("Checking for missing files in completed records...")
    backfill_stats = backfill_missing_files(all_records, output_dir, logger)
    if backfill_stats["input_image"] > 0 or backfill_stats["reconstruction"] > 0:
        logger.info(f"Backfilled: {backfill_stats['input_image']} inputs, "
                     f"{backfill_stats['reconstruction']} reconstructions")

    # Filter out completed records
    if skip_completed:
        pending = [r for r in all_records if not is_record_completed(output_dir, r.record_id)]
        completed_count = len(all_records) - len(pending)
        logger.info(f"Already completed: {completed_count}, Pending: {len(pending)}")
    else:
        pending = list(all_records)

    if limit:
        pending = pending[:limit]
        logger.info(f"Limited to {len(pending)} records")

    if dry_run:
        logger.info("\n[DRY RUN] Would process these records:")
        for i, rec in enumerate(pending[:20]):
            logger.info(f"  [{i+1:3d}] {rec.record_id}")
        if len(pending) > 20:
            logger.info(f"  ... and {len(pending) - 20} more")
        logger.info(f"\nGPU pair assignments:")
        for i, pair in enumerate(gpu_pairs):
            assigned = len([r for j, r in enumerate(pending) if j % len(gpu_pairs) == i])
            logger.info(f"  Worker {i} (GPUs {pair}): {assigned} records")
        return {"dry_run": True, "total_records": len(pending), "gpu_pairs": [list(p) for p in gpu_pairs]}

    if not pending:
        logger.info("No pending records. All done!")
        return {"message": "All records already completed", "backfill_stats": backfill_stats}

    # ---- Qwen parameters ----
    qwen_params = {
        "num_layers": num_layers,
        "seed": seed,
        "resolution": resolution,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "alpha_threshold": alpha_threshold,
    }

    # ---- Multiprocessing ----
    mp.set_start_method("spawn", force=True)

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for rec in pending:
        task_queue.put((rec.record_id, str(rec.image_path)))
    for _ in gpu_pairs:
        task_queue.put(None)  # Termination signal

    workers = []
    for i, pair in enumerate(gpu_pairs):
        p = mp.Process(
            target=worker_process,
            args=(i, pair, task_queue, result_queue, output_dir, qwen_params),
            daemon=True,
        )
        p.start()
        workers.append(p)
        logger.info(f"Started worker {i} on GPUs {pair}")

    # ---- Collect results (unified tqdm) ----
    results = {
        "data_dir": str(data_dir),
        "start_time": datetime.now().isoformat(),
        "gpu_pairs": [list(p) for p in gpu_pairs],
        "qwen_params": qwen_params,
        "backfill_stats": backfill_stats,
        "processed": [],
        "failed": [],
    }

    total = len(pending)
    pbar = tqdm(total=total, desc="Qwen Crello Baseline", ncols=120, unit="rec")

    try:
        processed = 0
        while processed < total:
            try:
                result = result_queue.get(timeout=600)
            except:
                alive = sum(1 for w in workers if w.is_alive())
                if alive == 0:
                    logger.warning("All workers have stopped")
                    break
                continue

            processed += 1
            pbar.update(1)

            if result["success"]:
                elapsed_str = f"{result['elapsed']:.1f}s"
                pbar.set_postfix_str(
                    f"✓ {result['record_id']} ({result['num_layers']}L, {elapsed_str})"
                )
                logger.debug(f"[{processed}/{total}] ✓ {result['record_id']} "
                             f"({result['num_layers']} layers, {elapsed_str})")
                results["processed"].append({
                    "record_id": result["record_id"],
                    "num_layers": result["num_layers"],
                    "elapsed": result["elapsed"],
                })
            else:
                pbar.set_postfix_str(f"✗ {result['record_id']}")
                logger.error(f"[{processed}/{total}] ✗ {result['record_id']}: {result['error']}")
                results["failed"].append({
                    "record_id": result["record_id"],
                    "error": result["error"],
                })

            # Intermediate save
            if processed % 10 == 0:
                results["end_time"] = datetime.now().isoformat()
                with open(output_dir / "qwen_crello_results.json", 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
    finally:
        pbar.close()
        for w in workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()

    # Final save
    results["end_time"] = datetime.now().isoformat()
    results_file = output_dir / "qwen_crello_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total processed: {len(results['processed'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    logger.info(f"Backfilled: {backfill_stats['input_image']} inputs, "
                f"{backfill_stats['reconstruction']} reconstructions")

    if results["failed"]:
        logger.info("\nFailed records:")
        for f in results["failed"][:10]:
            logger.info(f"  - {f['record_id']}: {f['error'][:80]}")
        if len(results["failed"]) > 10:
            logger.info(f"  ... and {len(results['failed']) - 10} more")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen Image Layered Baseline on Crello Test Set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: Replace the GPU id placeholder <QWEN_GPU_IDS> below with your own
comma-separated GPU ids (e.g. 0,1,2,3,4,5,6,7).

Examples:
    # Split GPUs into pairs of 2
    python run_qwen_crello.py --data_dir crello_data/records --output_dir outputs/qwen_crello --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2

    # Testing (10 records only)
    python run_qwen_crello.py --data_dir crello_data/records --output_dir outputs/qwen_crello --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --limit 10

    # Dry run
    python run_qwen_crello.py --data_dir crello_data/records --output_dir outputs/qwen_crello --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --dry_run
        """
    )

    parser.add_argument("--data_dir", "-i", type=str, required=True,
                        help="Path to the Crello dataset directory (containing crello_test_*/ record dirs)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Path to the output directory")
    parser.add_argument("--qwen_gpus", type=str, required=True,
                        help="GPU IDs for the Qwen model, comma-separated and user-specific (e.g., '0,1,2,3,4,5,6,7')")
    parser.add_argument("--qwen_pair_size", type=int, required=True,
                        help="GPUs per Qwen pair (e.g., 2 for A6000)")

    # Qwen parameters
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # Execution options
    parser.add_argument("--dry_run", "-d", action="store_true")
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--no_skip", action="store_true")

    args = parser.parse_args()

    qwen_gpus = parse_gpu_list(args.qwen_gpus)
    if not qwen_gpus:
        print(f"Error: Invalid qwen_gpus: {args.qwen_gpus}")
        sys.exit(1)

    try:
        results = run_qwen_crello_baseline(
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
            print(f"\nResults saved to: {Path(args.output_dir) / 'qwen_crello_results.json'}")

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