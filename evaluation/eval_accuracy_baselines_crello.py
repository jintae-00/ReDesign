#!/usr/bin/env python3
"""Accuracy evaluation for baseline models on Crello dataset.

Evaluates visual quality, layout, and composite fidelity for baseline models
using the same metrics as evaluation_crello.py. Results are saved in a
timestamped subfolder.

Usage:
    python scripts/eval_accuracy_baselines_crello.py \
        --crello-subset ./crello_subset \
        --models vtracer agent qwen \
        --vtracer-dir ./baseline_vtracer_experiment/crello \
        --agent-dir ./crello_experiment_agent_0206 \
        --qwen-dir ./crello_experiment_qwen_0206 \
        --output ./eval_crello_baselines_accuracy \
        --num-workers 12 --gpu-workers "0:3,1:3,2:3,3:3" --no-viz
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from multiprocessing import Event, Manager, Process
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Model configuration for Crello
# ---------------------------------------------------------------------------

CRELLO_MODEL_CONFIGS = {
    "vtracer": {"format": "omnisvg", "description": "VTracer image-to-SVG baseline"},
    "agent": {"format": "agent", "description": "Main agent (parse.json + elements/)"},
    "qwen": {"format": "qwen", "description": "Qwen baseline (layer_XX.png / CCA)"},
    "layered": {"format": "qwen", "description": "LayerD baseline (layer_XX.png)"},
    "multi_tools": {"format": "agent", "description": "Multi-tools pipeline"},
}


def _json_safe_default(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return str(v)
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Episode scanning for Crello
# ---------------------------------------------------------------------------

def scan_crello_gt(crello_subset_dir: Path) -> Dict[str, Path]:
    """Scan crello_subset/ for GT records with gt_metadata.json."""
    gt_map: Dict[str, Path] = {}
    for record_dir in sorted(crello_subset_dir.iterdir()):
        if not record_dir.name.startswith("crello_test_"):
            continue
        if not record_dir.is_dir():
            continue
        if (record_dir / "gt_metadata.json").exists():
            gt_map[record_dir.name] = record_dir
    return gt_map


def scan_crello_vtracer(base_dir: Path) -> Dict[str, Path]:
    """Scan VTracer crello experiment for output.svg files."""
    found: Dict[str, Path] = {}
    if not base_dir.exists():
        return found
    for ep_dir in sorted(base_dir.iterdir()):
        if ep_dir.is_dir() and (ep_dir / "output.svg").exists():
            found[ep_dir.name] = ep_dir
    return found


def scan_crello_agent(base_dir: Path) -> Dict[str, Path]:
    """Scan agent experiment (split_*/episodes/ layout)."""
    found: Dict[str, Path] = {}
    for split_dir in sorted(base_dir.glob("split_*")):
        episodes_dir = split_dir / "episodes"
        if not episodes_dir.exists():
            continue
        for ep_dir in sorted(episodes_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "parse.json").exists():
                if ep_dir.name not in found:
                    found[ep_dir.name] = ep_dir
    # Also check direct episodes/ layout
    episodes_dir = base_dir / "episodes"
    if episodes_dir.exists():
        for ep_dir in sorted(episodes_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "parse.json").exists():
                if ep_dir.name not in found:
                    found[ep_dir.name] = ep_dir
    # Also check all_splits/episodes/ layout (e.g., crello multi_tools)
    all_splits_episodes = base_dir / "all_splits" / "episodes"
    if all_splits_episodes.exists():
        for ep_dir in sorted(all_splits_episodes.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "parse.json").exists():
                if ep_dir.name not in found:
                    found[ep_dir.name] = ep_dir
    return found


def scan_crello_qwen(base_dir: Path) -> Dict[str, Path]:
    """Scan qwen experiment (flat or split layout)."""
    found: Dict[str, Path] = {}
    if not base_dir.exists():
        return found
    for ep_dir in sorted(base_dir.iterdir()):
        if ep_dir.is_dir() and ep_dir.name.startswith("crello_test_"):
            if (ep_dir / "layer_00.png").exists():
                found[ep_dir.name] = ep_dir
    # Split layout
    for split_dir in sorted(base_dir.glob("split_*")):
        if not split_dir.is_dir():
            continue
        for ep_dir in sorted(split_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "layer_00.png").exists():
                if ep_dir.name not in found:
                    found[ep_dir.name] = ep_dir
    # all_splits layout
    all_splits = base_dir / "all_splits"
    if all_splits.exists():
        for ep_dir in sorted(all_splits.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "layer_00.png").exists():
                if ep_dir.name not in found:
                    found[ep_dir.name] = ep_dir
    return found


CRELLO_SCAN_FUNCTIONS = {
    "vtracer": scan_crello_vtracer,
    "agent": scan_crello_agent,
    "qwen": scan_crello_qwen,
    "layered": scan_crello_qwen,  # same format
    "multi_tools": scan_crello_agent,  # same format
}


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def worker_process(
    worker_id: int,
    gpu_id: int,
    task_queue,
    args_dict: Dict,
    progress_queue=None,
):
    """Worker that evaluates baseline models by pulling tasks from a shared queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch

    import evaluation.crello_metrics as _ec
    from evaluation.crello_metrics import (
        MetricModels,
        evaluate_episode,
        extract_agent_elements,
        extract_gt_elements,
        extract_omnisvg_elements,
        extract_qwen_elements_cca,
    )

    # For SVG-based models (omnisvg), relax matching threshold so that
    # more GT-pred pairs are matched.
    SVG_DUMMY_COST = 1.0
    _original_dummy_cost = _ec.DUMMY_COST

    output_dir = Path(args_dict["output"])
    log_file_path = output_dir / f"worker_{worker_id}_gpu{gpu_id}.log"

    class SimpleLogger:
        def __init__(self, path):
            self._f = open(path, "w")
        def info(self, msg): self._f.write(f"[INFO] {msg}\n"); self._f.flush()
        def warn(self, msg): self._f.write(f"[WARN] {msg}\n"); self._f.flush()
        def error(self, msg): self._f.write(f"[ERROR] {msg}\n"); self._f.flush()
        def progress(self, cur, total, eid, extra=""): pass
        def close(self): self._f.close()

    logger = SimpleLogger(log_file_path)
    logger.info(f"Worker {worker_id} started on GPU {gpu_id} (dynamic queue)")

    metric_models = MetricModels("cuda:0", logger=logger)
    logger.info("Metric models loaded")

    use_optimal = args_dict.get("matching", "optimal") == "optimal"
    models_to_eval = args_dict["models_to_eval"]
    results = {m: [] for m in models_to_eval}

    task_count = 0
    empty_retries = 0
    while True:
        try:
            task = task_queue.get(timeout=2)
            empty_retries = 0
        except Empty:
            empty_retries += 1
            if empty_retries >= 3:
                break
            continue
        task_count += 1
        episode_id = task["episode_id"]
        record_dir = Path(task["record_dir"])
        logger.info(f"[task {task_count}] Episode {episode_id}")

        try:
            # Extract GT elements (Crello: from record_dir)
            gt_elements, canvas_size, gt_recon_img = extract_gt_elements(
                record_dir, logger=logger
            )
            if not gt_elements:
                logger.warn(f"[{episode_id}] No GT elements, skipping")
                continue

            episode_model_results = {}
            for model_name in models_to_eval:
                model_dir_str = task.get(f"model_dir_{model_name}")
                if model_dir_str is None:
                    continue

                model_dir = Path(model_dir_str)
                model_format = CRELLO_MODEL_CONFIGS[model_name]["format"]

                # Check if already done (resume support)
                episode_out_dir = output_dir / episode_id
                if (episode_out_dir / model_name / "metrics.json").exists():
                    logger.info(f"[{episode_id}][{model_name}] Already done, skipping")
                    continue

                # Extract pred elements
                if model_format == "qwen":
                    pred_elements = extract_qwen_elements_cca(
                        model_dir, canvas_size, logger=logger
                    )
                elif model_format == "omnisvg":
                    pred_elements = extract_omnisvg_elements(
                        model_dir, canvas_size, logger=logger,
                        render_scale=0.5,
                    )
                else:
                    pred_elements = extract_agent_elements(
                        model_dir, canvas_size,
                        apply_alpha_correction=True,
                        text_refinement=True,
                        logger=logger,
                    )

                if not pred_elements:
                    logger.warn(f"[{episode_id}][{model_name}] No pred elements")
                    continue

                # Relax matching threshold for SVG-based models
                if model_format == "omnisvg":
                    _ec.DUMMY_COST = SVG_DUMMY_COST

                try:
                    with torch.no_grad():
                        res = evaluate_episode(
                            episode_id, gt_elements, pred_elements, canvas_size,
                            model_name, output_dir, metric_models,
                            save_visualization=not args_dict.get("no_viz", True),
                            gt_recon_img=gt_recon_img,
                            use_optimal_matching=use_optimal,
                            logger=logger,
                        )
                finally:
                    _ec.DUMMY_COST = _original_dummy_cost

                # Save metrics.json
                method_dir = output_dir / episode_id / model_name
                method_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = method_dir / "metrics.json"
                if not metrics_path.exists():
                    with open(metrics_path, "w", encoding="utf-8") as _f:
                        json.dump(res, _f, indent=2, default=str)

                results[model_name].append(res)
                episode_model_results[model_name] = res
                logger.info(
                    f"[{episode_id}][{model_name}] Done: "
                    f"matched={res['counts']['matched_pairs']} "
                    f"FN={res['counts']['fn']} FP={res['counts']['fp']}"
                )

                del pred_elements

            if progress_queue is not None:
                if episode_model_results:
                    progress_queue.put({
                        "status": "done_episode",
                        "episode_id": episode_id,
                        "model_results": episode_model_results,
                    })
                else:
                    progress_queue.put({
                        "status": "skipped_episode",
                        "episode_id": episode_id,
                    })

            del gt_elements, gt_recon_img

        except Exception as e:
            logger.error(f"[{episode_id}] Error: {e}")
            import traceback
            logger.error(traceback.format_exc())

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Worker {worker_id} done ({task_count} tasks): " + ", ".join(
        f"{m}={len(results[m])}" for m in models_to_eval
    ))
    logger.close()
    return results


def worker_wrapper(worker_id, gpu_id, task_queue, args_dict, result_queue, progress_queue):
    try:
        results = worker_process(worker_id, gpu_id, task_queue, args_dict, progress_queue)
        result_queue.put({"worker_id": worker_id, "results": results})
    except Exception as e:
        import traceback
        print(f"Worker {worker_id} CRASHED: {e}")
        traceback.print_exc()
        result_queue.put({"worker_id": worker_id, "results": {}})


# ---------------------------------------------------------------------------
# Progress monitor
# ---------------------------------------------------------------------------

def _safe_mean(vals, is_psnr=False):
    clean = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return 0.0
    if is_psnr:
        finite = [v for v in clean if not (isinstance(v, float) and math.isinf(v))]
        if finite:
            mx = max(finite)
            clean = [v if not (isinstance(v, float) and math.isinf(v)) else mx for v in clean]
        else:
            return 0.0
    return sum(clean) / len(clean)


def progress_monitor(progress_queue, stop_event, total_tasks, model_names):
    """Monitor process that prints real-time cumulative averages."""
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_tasks, unit="ep", dynamic_ncols=True, position=0, leave=True)

    def init_stats():
        return {
            "vq_inter": {"l1": [], "l2": [], "psnr": []},
            "vq_union": {"l1": [], "l2": [], "psnr": []},
            "vq_gt": {"l1": [], "l2": [], "psnr": []},
            "lay_soft": {"iou": 0, "pq": 0, "sq": 0, "rq": 0},
            "lay_bin": {"iou": 0, "pq": 0, "sq": 0, "rq": 0},
            "comp": {"l1": [], "psnr": [], "ssim": 0, "lpips": 0, "dino": 0},
            "count": 0,
            "comp_count": 0,
            "comp_skipped": 0,
        }

    stats = {m: init_stats() for m in model_names}
    total_processed = 0
    total_skipped = 0

    def _fmt_vq(s, key, met):
        return _safe_mean(s[key][met], is_psnr=(met == "psnr"))

    def _fmt_comp(s, met):
        v = s["comp_count"]
        if v == 0:
            return 0.0
        if met in ["l1", "psnr"]:
            return _safe_mean(s["comp"][met], is_psnr=(met == "psnr"))
        return s["comp"][met] / v

    def _print_table():
        if total_processed == 0:
            return
        lines = [
            "\n" + "=" * 120,
            f" [CUMULATIVE SUMMARY]  Processed: {total_processed}/{total_tasks}  (skipped: {total_skipped})",
            "-" * 120,
        ]
        header = f" {'Category / Metric':<28}"
        for m in model_names:
            header += f" | {m.upper():<30}"
        lines.append(header)
        lines.append("-" * 120)

        lines.append(" [Visual - Intersection]")
        row = f"  L1 / L2 / PSNR             "
        for m in model_names:
            s = stats[m]
            if s["count"] > 0:
                row += f" | L1:{_fmt_vq(s,'vq_inter','l1'):.4f} L2:{_fmt_vq(s,'vq_inter','l2'):.4f} PSNR:{_fmt_vq(s,'vq_inter','psnr'):.2f}"
            else:
                row += f" | {'N/A':>30}"
        lines.append(row)

        lines.append(" [Visual - Union]")
        row = f"  L1 / L2 / PSNR             "
        for m in model_names:
            s = stats[m]
            if s["count"] > 0:
                row += f" | L1:{_fmt_vq(s,'vq_union','l1'):.4f} L2:{_fmt_vq(s,'vq_union','l2'):.4f} PSNR:{_fmt_vq(s,'vq_union','psnr'):.2f}"
            else:
                row += f" | {'N/A':>30}"
        lines.append(row)

        lines.append(" [Visual - GT Region]")
        row = f"  L1 / L2 / PSNR             "
        for m in model_names:
            s = stats[m]
            if s["count"] > 0 and s["vq_gt"]["l1"]:
                row += f" | L1:{_fmt_vq(s,'vq_gt','l1'):.4f} L2:{_fmt_vq(s,'vq_gt','l2'):.4f} PSNR:{_fmt_vq(s,'vq_gt','psnr'):.2f}"
            else:
                row += f" | {'N/A':>30}"
        lines.append(row)

        lines.append(" [Layout - Soft]")
        row = f"  IoU / PQ / SQ / RQ         "
        for m in model_names:
            s = stats[m]
            v = s["count"]
            if v > 0:
                row += f" | I:{s['lay_soft']['iou']/v:.4f} P:{s['lay_soft']['pq']/v:.4f} S:{s['lay_soft']['sq']/v:.4f} R:{s['lay_soft']['rq']/v:.4f}"
            else:
                row += f" | {'N/A':>30}"
        lines.append(row)

        lines.append(" [Layout - Binary]")
        row = f"  IoU / PQ / SQ / RQ         "
        for m in model_names:
            s = stats[m]
            v = s["count"]
            if v > 0:
                row += f" | I:{s['lay_bin']['iou']/v:.4f} P:{s['lay_bin']['pq']/v:.4f} S:{s['lay_bin']['sq']/v:.4f} R:{s['lay_bin']['rq']/v:.4f}"
            else:
                row += f" | {'N/A':>30}"
        lines.append(row)

        lines.append(" [Composite]")
        row = f"  L1 / PSNR / SSIM           "
        for m in model_names:
            s = stats[m]
            if s["comp_count"] > 0:
                row += f" | L1:{_fmt_comp(s,'l1'):.4f} PSNR:{_fmt_comp(s,'psnr'):.2f} SSIM:{_fmt_comp(s,'ssim'):.4f}"
            else:
                row += f" | {'N/A':>30}"
        lines.append(row)

        row = f"  LPIPS / DINO               "
        for m in model_names:
            s = stats[m]
            if s["comp_count"] > 0:
                row += f" | LP:{_fmt_comp(s,'lpips'):.4f} DN:{_fmt_comp(s,'dino'):.4f}"
            else:
                row += f" | {'N/A':>30}"
        lines.append(row)

        row = f"  Episodes (comp/skip/total) "
        for m in model_names:
            s = stats[m]
            row += f" | {s['comp_count']}/{s['comp_skipped']}/{s['count']:>20}"
        lines.append(row)
        lines.append("=" * 120)

        writer = tqdm.write if tqdm is not None else print
        writer("\n".join(lines))

    while not stop_event.is_set() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=0.5)
            if update.get("status") == "skipped_episode":
                total_skipped += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"skip={total_skipped} done={total_processed}")
                continue
            if update.get("status") == "done_episode":
                total_processed += 1
                if pbar is not None:
                    pbar.update(1)

                model_results = update.get("model_results", {})
                for m_key, res in model_results.items():
                    if m_key not in stats:
                        continue
                    s = stats[m_key]
                    s["count"] += 1

                    em_dual = res.get("element_metrics_dual")
                    if em_dual:
                        vq_dual = em_dual.get("visual_quality", {})
                        for reg, target in [("intersection_region", "vq_inter"), ("union_region", "vq_union"), ("gt_region", "vq_gt")]:
                            region_data = vq_dual.get(reg, {}).get("simple_avg", {})
                            for met in ["l1", "l2", "psnr"]:
                                val = region_data.get(met)
                                if val is not None:
                                    s[target][met].append(val)

                        iou_dual = em_dual.get("iou", {}).get("simple_avg", {})
                        pq_dual = res.get("panoptic_quality_dual", {})
                        for style, target in [("soft", "lay_soft"), ("binary", "lay_bin")]:
                            iou_val = iou_dual.get(style)
                            if iou_val is not None:
                                s[target]["iou"] += iou_val
                            pq_data = pq_dual.get(style, {})
                            for met in ["pq", "sq", "rq"]:
                                val = pq_data.get(met)
                                if val is not None:
                                    s[target][met] += val
                    else:
                        em = res.get("element_metrics", {})
                        simple_avg = em.get("simple_avg", {})
                        for met in ["l1", "l2", "psnr"]:
                            val = simple_avg.get(met)
                            if val is not None:
                                s["vq_inter"][met].append(val)

                        pq = res.get("panoptic_quality", {})
                        for met in ["pq", "sq", "rq"]:
                            val = pq.get(met)
                            if val is not None:
                                s["lay_soft"][met] += val
                        iou_val = simple_avg.get("iou")
                        if iou_val is not None:
                            s["lay_soft"]["iou"] += iou_val

                    # Composite (skip if composite_skipped or non-text <= 5)
                    if res.get("counts", {}).get("composite_skipped", False):
                        s["comp_skipped"] += 1
                    else:
                        comp = res.get("composite_metrics") or {}
                        if comp:
                            s["comp_count"] += 1
                            for met in ["l1", "psnr"]:
                                val = comp.get(met)
                                if val is not None:
                                    s["comp"][met].append(val)
                            for met in ["ssim", "lpips", "dino"]:
                                val = comp.get(met)
                                if val is not None:
                                    s["comp"][met] += val

                _print_table()

        except Empty:
            continue

    if pbar is not None:
        pbar.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Accuracy evaluation for baseline models on Crello dataset"
    )
    parser.add_argument("--crello-subset", type=str, required=True,
                        help="Path to crello_subset/ with GT records")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        choices=list(CRELLO_MODEL_CONFIGS.keys()),
                        help="Models to evaluate")
    parser.add_argument("--vtracer-dir", type=str, default=None,
                        help="VTracer crello experiment directory")
    parser.add_argument("--agent-dir", type=str, default=None,
                        help="Agent crello experiment directory")
    parser.add_argument("--qwen-dir", type=str, default=None,
                        help="Qwen crello experiment directory")
    parser.add_argument("--layered-dir", type=str, default=None,
                        help="LayerD crello experiment directory")
    parser.add_argument("--multi-tools-dir", type=str, default=None,
                        help="Multi-tools crello experiment directory")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--resume-dir", type=str, default=None,
                        help="Resume from a previous output directory")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--gpu-workers", type=str, default=None,
                        help="Per-GPU worker counts, e.g. '0:3,1:3,2:3,3:3'")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--no-viz", action="store_true", default=True)
    parser.add_argument("--matching", type=str, default="optimal",
                        choices=["optimal", "legacy"])

    args = parser.parse_args()

    # Output directory
    if args.resume_dir:
        output_dir = Path(args.resume_dir)
        if not output_dir.exists():
            print(f"[ERROR] Resume directory does not exist: {output_dir}")
            sys.exit(1)
        print(f"Resuming from: {output_dir}")
    else:
        if not args.output:
            parser.error("--output is required unless --resume-dir is set.")
        timestamp = datetime.now().strftime("%H%M%S")
        output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    crello_subset_dir = Path(args.crello_subset)

    # Build worker→GPU mapping
    if args.gpu_workers:
        worker_gpu_ids = []
        for spec in args.gpu_workers.split(","):
            gpu_str, count_str = spec.split(":")
            gpu_id = int(gpu_str.strip())
            count = int(count_str.strip())
            worker_gpu_ids.extend([gpu_id] * count)
        num_workers = len(worker_gpu_ids)
    else:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        num_workers = min(args.num_workers, len(gpu_ids))
        worker_gpu_ids = gpu_ids[:num_workers]

    print("=" * 80)
    print("CRELLO BASELINE ACCURACY EVALUATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")
    print(f"Models to evaluate: {args.models}")
    print(f"Workers: {num_workers}, GPU mapping: {worker_gpu_ids}")
    print("=" * 80)

    # 1. Collect GT records
    gt_map = scan_crello_gt(crello_subset_dir)
    print(f"\nGT records: {len(gt_map)}")

    # 2. Scan episodes for each model, build task list
    dir_map = {
        "vtracer": args.vtracer_dir,
        "agent": args.agent_dir,
        "qwen": args.qwen_dir,
        "layered": args.layered_dir,
        "multi_tools": args.multi_tools_dir,
    }

    all_episode_tasks: Dict[str, Dict] = {}

    for model_name in args.models:
        model_dir_arg = dir_map.get(model_name)
        if model_dir_arg is None:
            print(f"  [WARNING] No directory specified for {model_name}, skipping")
            continue

        model_dir = Path(model_dir_arg)
        scan_fn = CRELLO_SCAN_FUNCTIONS[model_name]
        model_map = scan_fn(model_dir)

        common = set(gt_map.keys()) & set(model_map.keys())
        print(f"  {model_name}: {len(model_map)} episodes, {len(common)} common with GT")

        for eid in sorted(common):
            if eid not in all_episode_tasks:
                all_episode_tasks[eid] = {
                    "episode_id": eid,
                    "record_dir": str(gt_map[eid]),
                }
            all_episode_tasks[eid][f"model_dir_{model_name}"] = str(model_map[eid])

    print(f"\nTotal unique episodes to evaluate: {len(all_episode_tasks)}")

    task_list = list(all_episode_tasks.values())
    task_list.sort(key=lambda t: t["episode_id"])
    if args.max_episodes:
        task_list = task_list[:args.max_episodes]
        print(f"Capped to {len(task_list)} episodes")

    if not task_list:
        print("No episodes to evaluate!")
        return

    # 3. Distribute tasks to workers via shared queue (dynamic dispatch)
    print(f"\nTask distribution: {len(task_list)} tasks -> {num_workers} workers (dynamic queue)")
    for i in range(num_workers):
        print(f"  Worker {i} (GPU {worker_gpu_ids[i]})")

    args_dict = {
        "output": str(output_dir),
        "no_viz": args.no_viz,
        "matching": args.matching,
        "models_to_eval": args.models,
    }

    # 4. Run workers with progress monitoring
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    progress_queue_obj = manager.Queue()
    stop_event = Event()

    # Fill the shared task queue
    for task in task_list:
        task_queue.put(task)

    monitor = Process(
        target=progress_monitor,
        args=(progress_queue_obj, stop_event, len(task_list), args.models),
    )
    monitor.start()

    start_time = time.time()
    processes = []
    for i in range(num_workers):
        p = Process(
            target=worker_wrapper,
            args=(i, worker_gpu_ids[i], task_queue, args_dict,
                  result_queue, progress_queue_obj),
        )
        p.start()
        processes.append(p)
        print(f"  Worker {i} started (PID: {p.pid}, GPU: {worker_gpu_ids[i]})")

    for p in processes:
        p.join()

    stop_event.set()
    monitor.join(timeout=10)

    elapsed_time = time.time() - start_time
    print(f"\nAll workers completed in {elapsed_time:.2f}s")

    # 5. Collect results
    all_results: Dict[str, List] = {m: [] for m in args.models}
    while not result_queue.empty():
        wr = result_queue.get()
        for m in args.models:
            model_results = wr.get("results", {}).get(m, [])
            all_results[m].extend(model_results)
            if model_results:
                print(f"  Worker {wr['worker_id']}: {len(model_results)} {m}")

    # 6. Aggregate and save
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)

    from evaluation.crello_metrics import aggregate_results

    all_summaries: Dict[str, Any] = {}
    for model_name in args.models:
        results = all_results.get(model_name, [])
        if not results:
            continue
        summary = aggregate_results(results)
        all_summaries[model_name] = summary

        print(f"\n{model_name.upper()} ({len(results)} episodes)")
        print("-" * 40)
        s = summary["element_metrics"]["simple_avg"]
        print(f"  Element: L1={s.get('l1', 0):.4f}, IoU={s.get('iou', 0):.4f}")
        pq = summary["panoptic_quality"]
        print(f"  PQ={pq['pq']:.4f}, SQ={pq['sq']:.4f}, RQ={pq['rq']:.4f}")
        comp = summary.get("composite_metrics") or {}
        n_comp = comp.get("num_episodes", len(results))
        n_skip = comp.get("num_episodes_skipped", 0)
        print(f"  Composite ({n_comp} eps, {n_skip} skipped): L1={comp.get('l1', float('nan')):.4f}, PSNR={comp.get('psnr', float('nan')):.2f}, SSIM={comp.get('ssim', float('nan')):.4f}")
        print(f"  LPIPS={comp.get('lpips', float('nan')):.4f}, DINO={comp.get('dino', float('nan')):.4f}")

    # Save unified summary
    unified = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": "crello",
            "models_evaluated": args.models,
            "matching_algorithm": args.matching,
            "num_workers": num_workers,
            "worker_gpu_ids": worker_gpu_ids,
            "elapsed_time_seconds": elapsed_time,
        },
        "results": {
            "all_episodes": all_summaries,
        },
        "per_model_episode_counts": {
            m: len(all_results.get(m, [])) for m in args.models
        },
        "per_episode_details": {},
    }

    for model_name in args.models:
        results = all_results.get(model_name, [])
        if results:
            unified["per_episode_details"][model_name] = [
                {
                    "episode_id": r.get("episode_id", ""),
                    "background_l1": r.get("background_l1", 0.0),
                    "element_metrics": r.get("element_metrics", {}),
                    "panoptic_quality": r.get("panoptic_quality", {}),
                    "composite_metrics": r.get("composite_metrics") or {},
                    "counts": r.get("counts", {}),
                }
                for r in results
            ]

    summary_path = output_dir / "evaluation_unified_summary.json"
    with open(summary_path, "w") as f:
        json.dump(unified, f, indent=2, default=_json_safe_default)

    print(f"\nSaved unified summary to {summary_path}")
    print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.1f}min)")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
