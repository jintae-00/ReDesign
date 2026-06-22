#!/usr/bin/env python3
"""
run_vtracer_baseline.py - VTracer Image-to-SVG Baseline (Figma + Crello unified)

Converts images to SVG with VTracer for use as a baseline.
Saves output.svg; evaluation rasterizes each individual SVG element to perform GT matching.
CPU only -- no GPU required. Runs in parallel across multiple workers.

Each dataset takes a whole data directory (no splits):
    Figma  : --figma_data_dir (dir with valid_frames/ + unit_images/)
    Crello : --crello_data_dir (dir with crello_test_*/ each containing composite.png)

Usage:
    # Figma only
    python run_vtracer_baseline.py --dataset figma --figma_data_dir figma_data \
        --figma_output_dir outputs/vtracer_figma --workers 32
    # Crello only
    python run_vtracer_baseline.py --dataset crello --crello_data_dir crello_data/records \
        --crello_output_dir outputs/vtracer_crello --workers 32
    # Both, run sequentially
    python run_vtracer_baseline.py --dataset all \
        --figma_data_dir figma_data --figma_output_dir outputs/vtracer_figma \
        --crello_data_dir crello_data/records --crello_output_dir outputs/vtracer_crello --workers 32
    # Limit the number of items
    python run_vtracer_baseline.py --dataset all --workers 32 --limit 10 --dry_run \
        --figma_data_dir figma_data --figma_output_dir outputs/vtracer_figma \
        --crello_data_dir crello_data/records --crello_output_dir outputs/vtracer_crello
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm import tqdm

import vtracer

# =============================================================================
# Configuration
# =============================================================================

# --- VTracer parameters (user-specified) ---
VTRACER_PARAMS = dict(
    colormode="color",
    hierarchical="stacked",
    mode="spline",            # Curve Fitting = Spline
    filter_speckle=8,         # Filter Speckle (Cleaner)
    color_precision=7,        # Color Precision (More accurate)
    layer_difference=64,      # Gradient Step (Less layers)
    corner_threshold=60,      # Corner Threshold (Smoother)
    length_threshold=4,       # Segment Length (More coarse)
    splice_threshold=45,      # Splice Threshold (Less accurate)
)


# =============================================================================
# Data Item
# =============================================================================

@dataclass
class DataItem:
    item_id: str
    image_path: Path
    dataset: str   # "figma" or "crello"


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_file: Path, name: str = "vtracer") -> logging.Logger:
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
# Figma Data Loading
# =============================================================================

def load_figma_items(data_dir: Path) -> List[DataItem]:
    """Load every valid_frames/*.json under data_dir (split-agnostic).

    The GT reconstruction (input image) resolves to
    ``data_dir / <unit_images_dir> / <reconstructed_image_path>`` from each JSON.
    """
    items = []
    vf_dir = data_dir / "valid_frames"
    if not vf_dir.exists():
        print(f"[Warning] Not found: {vf_dir}")
        return items
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
                    items.append(DataItem(
                        item_id=frame_id,
                        image_path=image_path,
                        dataset="figma",
                    ))
        except Exception as e:
            print(f"[Warning] Failed to load {json_path}: {e}")
    return items


# =============================================================================
# Crello Data Loading
# =============================================================================

def load_crello_items(data_dir: Path) -> List[DataItem]:
    """Load every crello_test_* record directory under data_dir (split-agnostic).

    The input image for each record is ``<record_dir>/composite.png``.
    """
    items = []
    seen = set()
    record_dirs = sorted([
        d for d in data_dir.iterdir()
        if (d.is_dir() or d.is_symlink()) and d.name.startswith("crello_test_")
    ])
    for record_dir in record_dirs:
        record_id = record_dir.name
        if record_id in seen:
            continue
        seen.add(record_id)
        composite_path = record_dir / "composite.png"
        if composite_path.exists():
            items.append(DataItem(
                item_id=record_id,
                image_path=composite_path.resolve(),
                dataset="crello",
            ))
    return items


# =============================================================================
# Completion Check
# =============================================================================

def is_completed(output_dir: Path, item_id: str) -> bool:
    return (output_dir / item_id / "output.svg").exists()


def is_failed(output_dir: Path, item_id: str) -> bool:
    return (output_dir / item_id / "_FAILED").exists()


# =============================================================================
# Worker Process
# =============================================================================

def worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    figma_output_dir: Path,
    crello_output_dir: Path,
):
    while True:
        try:
            item = task_queue.get(timeout=1.0)
        except:
            if task_queue.empty():
                break
            continue

        if item is None:
            break

        item_id, image_path_str, dataset = item
        image_path = Path(image_path_str)
        output_dir = figma_output_dir if dataset == "figma" else crello_output_dir
        start_time = time.time()

        try:
            item_output_dir = output_dir / item_id
            item_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy input image
            input_dest = item_output_dir / "input.png"
            shutil.copy2(image_path, input_dest)

            # Get canvas size
            original_img = Image.open(image_path)
            canvas_w, canvas_h = original_img.size

            # Run vtracer
            svg_path = item_output_dir / "output.svg"
            vtracer.convert_image_to_svg_py(
                str(image_path),
                str(svg_path),
                **VTRACER_PARAMS,
            )

            elapsed = time.time() - start_time

            # Count paths in SVG
            path_count = 0
            if svg_path.exists():
                svg_content = svg_path.read_text(encoding='utf-8')
                path_count = svg_content.count('<path')

            # Save metadata
            metadata = {
                "item_id": item_id,
                "dataset": dataset,
                "source_image": str(image_path),
                "canvas_size": [canvas_w, canvas_h],
                "svg_path_count": path_count,
                "processing_time_sec": elapsed,
                "timestamp": datetime.now().isoformat(),
                "method": "vtracer",
                "vtracer_params": VTRACER_PARAMS,
            }
            metadata_path = item_output_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            result_queue.put({
                "worker_id": worker_id,
                "item_id": item_id,
                "dataset": dataset,
                "success": True,
                "path_count": path_count,
                "elapsed": elapsed,
            })

        except Exception as e:
            elapsed = time.time() - start_time
            # Mark failed
            fail_dir = output_dir / item_id
            fail_dir.mkdir(parents=True, exist_ok=True)
            with open(fail_dir / "_FAILED", 'w') as f:
                f.write(str(e))

            result_queue.put({
                "worker_id": worker_id,
                "item_id": item_id,
                "dataset": dataset,
                "success": False,
                "error": str(e),
            })


# =============================================================================
# Single-Dataset Processing (with its own tqdm)
# =============================================================================

def _process_dataset(
    items: List[DataItem],
    dataset_name: str,
    output_dir: Path,
    num_workers: int,
    logger: logging.Logger,
) -> Dict[str, int]:
    """Process one dataset with dedicated workers and tqdm bar."""
    if not items:
        logger.info(f"[{dataset_name}] No items to process")
        return {"success": 0, "failed": 0}

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(items)
    logger.info(f"[{dataset_name}] Processing {total} items with {num_workers} workers")

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for item in items:
        task_queue.put((item.item_id, str(item.image_path), item.dataset))

    for _ in range(num_workers):
        task_queue.put(None)

    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(i, task_queue, result_queue, output_dir, output_dir),
        )
        p.start()
        workers.append(p)

    success_count = 0
    fail_count = 0
    pbar = tqdm(total=total, desc=f"VTracer [{dataset_name}]", unit="item")

    completed = 0
    while completed < total:
        try:
            result = result_queue.get(timeout=120)
            completed += 1
            if result.get("success"):
                success_count += 1
            else:
                fail_count += 1
            pbar.set_postfix(ok=success_count, fail=fail_count)
            pbar.update(1)
        except:
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                break
            logger.warning(f"[{dataset_name}] Timeout waiting for results ({alive} workers alive)")

    pbar.close()

    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()

    logger.info(f"[{dataset_name}] Done: {success_count} success, {fail_count} failed / {total}")
    return {"success": success_count, "failed": fail_count}


# =============================================================================
# Main Runner
# =============================================================================

def run_vtracer_baseline(
    datasets: List[str],
    figma_data_dir: Optional[Path] = None,
    figma_output_dir: Optional[Path] = None,
    crello_data_dir: Optional[Path] = None,
    crello_output_dir: Optional[Path] = None,
    num_workers: int = 32,
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip_completed: bool = True,
) -> Dict[str, Any]:
    # Pick a log directory from whichever output dir is configured for the run.
    log_base = figma_output_dir if "figma" in datasets and figma_output_dir else crello_output_dir
    log_base.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_base.parent / "vtracer_run.log")

    logger.info("=" * 70)
    logger.info("VTracer Image-to-SVG Baseline")
    logger.info("=" * 70)
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"VTracer params: {VTRACER_PARAMS}")

    # Load and filter items per dataset
    dataset_items: Dict[str, List[DataItem]] = {}

    if "figma" in datasets:
        figma_items = load_figma_items(figma_data_dir)
        logger.info(f"Figma frames found: {len(figma_items)}")
        if skip_completed:
            before = len(figma_items)
            figma_items = [
                it for it in figma_items
                if not is_completed(figma_output_dir, it.item_id)
                and not is_failed(figma_output_dir, it.item_id)
            ]
            logger.info(f"Figma: skipping {before - len(figma_items)} completed/failed")
        if limit:
            figma_items = figma_items[:limit]
        dataset_items["figma"] = figma_items

    if "crello" in datasets:
        crello_items = load_crello_items(crello_data_dir)
        logger.info(f"Crello records found: {len(crello_items)}")
        if skip_completed:
            before = len(crello_items)
            crello_items = [
                it for it in crello_items
                if not is_completed(crello_output_dir, it.item_id)
                and not is_failed(crello_output_dir, it.item_id)
            ]
            logger.info(f"Crello: skipping {before - len(crello_items)} completed/failed")
        if limit:
            crello_items = crello_items[:limit]
        dataset_items["crello"] = crello_items

    total_items = sum(len(v) for v in dataset_items.values())
    logger.info(f"Total items to process: {total_items}")

    if dry_run:
        for ds_name, ds_items in dataset_items.items():
            logger.info(f"[DRY RUN] {ds_name}: {len(ds_items)} items")
            for item in ds_items[:10]:
                logger.info(f"  {item.item_id}")
            if len(ds_items) > 10:
                logger.info(f"  ... and {len(ds_items) - 10} more")
        return {"dry_run": True, "total": total_items}

    if total_items == 0:
        logger.info("No items to process")
        return {"total": 0, "success": 0, "failed": 0}

    # Process each dataset sequentially with its own tqdm
    total_success = 0
    total_failed = 0

    if "figma" in dataset_items and dataset_items["figma"]:
        result = _process_dataset(
            dataset_items["figma"], "figma", figma_output_dir, num_workers, logger,
        )
        total_success += result["success"]
        total_failed += result["failed"]

    if "crello" in dataset_items and dataset_items["crello"]:
        result = _process_dataset(
            dataset_items["crello"], "crello", crello_output_dir, num_workers, logger,
        )
        total_success += result["success"]
        total_failed += result["failed"]

    logger.info("=" * 70)
    logger.info(f"All done: {total_success} success, {total_failed} failed / {total_items}")
    logger.info("=" * 70)

    return {"total": total_items, "success": total_success, "failed": total_failed}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="VTracer Baseline (Figma + Crello)")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["figma", "crello", "all"],
                        help="Which dataset(s) to process (default: all)")
    parser.add_argument("--figma_data_dir", type=str, default=None,
                        help="Path to the Figma dataset directory (containing valid_frames/, unit_images/)")
    parser.add_argument("--figma_output_dir", type=str, default=None,
                        help="Path to the Figma output directory")
    parser.add_argument("--crello_data_dir", type=str, default=None,
                        help="Path to the Crello dataset directory (containing crello_test_*/ record dirs)")
    parser.add_argument("--crello_output_dir", type=str, default=None,
                        help="Path to the Crello output directory")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel workers (default: 32)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of items to process")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print what would be processed")
    parser.add_argument("--no_skip", action="store_true",
                        help="Do not skip completed items")

    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["figma", "crello"]
    else:
        datasets = [args.dataset]

    if "figma" in datasets and (not args.figma_data_dir or not args.figma_output_dir):
        parser.error("--figma_data_dir and --figma_output_dir are required for the figma dataset")
    if "crello" in datasets and (not args.crello_data_dir or not args.crello_output_dir):
        parser.error("--crello_data_dir and --crello_output_dir are required for the crello dataset")

    run_vtracer_baseline(
        datasets=datasets,
        figma_data_dir=Path(args.figma_data_dir) if args.figma_data_dir else None,
        figma_output_dir=Path(args.figma_output_dir) if args.figma_output_dir else None,
        crello_data_dir=Path(args.crello_data_dir) if args.crello_data_dir else None,
        crello_output_dir=Path(args.crello_output_dir) if args.crello_output_dir else None,
        num_workers=args.workers,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_completed=not args.no_skip,
    )
