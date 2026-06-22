#!/usr/bin/env python3
"""Accuracy evaluation for baseline models.

Evaluates visual quality, layout, and composite fidelity for baseline models
(layered, multi_tools, sparse_verif) using the same metrics as evaluation_figma.py.
Results are saved in a timestamped subfolder. Existing agent/qwen results
can be copied in for unified comparison.

Usage:
    python scripts/eval_accuracy_baselines.py \
        --figma-data ./figma_data \
        --exp-pairs \
            ./figma_agent_experiment_0131:./figma_qwen_experiment_0131:dino90_obj_5_25_char_50 \
            ./figma_agent_experiment_0208:./figma_qwen_experiment_0208:dino80_obj_5_60_char_25 \
        --models layered multi_tools sparse_verif \
        --layered-dir ./baseline_layerd_experiment \
        --multi-tools-dir ./baseline_muilti_tools_experiment \
        --sparse-verif-dir ./baseline_sparse_verification_agent_experiment \
        --existing-results ./eval_figma_rgba_0210_match \
        --output ./eval_baselines_accuracy \
        --num-workers 4 --gpu-ids 4,5,6,7 --no-viz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import textwrap
import time
from datetime import datetime
from multiprocessing import Event, Manager, Process
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Ensure project root is on path
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


def _json_safe_default(obj):
    """JSON serializer for numpy/nan/inf types."""
    import math
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

    import gc
    import torch

    import evaluation.figma_metrics as _ef
    from evaluation.figma_metrics import (
        MetricModels,
        WorkerLogger,
        evaluate_episode,
        extract_agent_elements,
        extract_gt_elements,
        extract_omnisvg_elements,
        extract_qwen_elements_cca,
    )

    # For SVG-based models (omnisvg), relax matching threshold so that
    # more GT-pred pairs are matched.  The default DUMMY_COST=0.4 is
    # too strict for VTracer-style element boundaries.
    SVG_DUMMY_COST = 1.0
    _original_dummy_cost = _ef.DUMMY_COST

    output_dir = Path(args_dict["output"])
    log_file_path = output_dir / f"worker_{worker_id}_gpu{gpu_id}.log"

    # Create a simple logger
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

    # Load metric models
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
        logger.info(f"[task {task_count}] Episode {episode_id}")

        try:
            # Extract GT elements
            gt_elements, canvas_size, gt_recon_img = extract_gt_elements(
                task["gt_json_path"], task["split_dir"], logger=logger
            )
            if not gt_elements:
                logger.warn(f"[{episode_id}] No GT elements, skipping")
                if progress_queue is not None:
                    progress_queue.put({
                        "status": "skipped_episode",
                        "episode_id": episode_id,
                    })
                continue

            # Evaluate each model
            episode_model_results = {}
            for model_name in models_to_eval:
                model_dir_str = task.get(f"model_dir_{model_name}")
                if model_dir_str is None:
                    continue

                model_dir = Path(model_dir_str)
                model_format = MODEL_CONFIGS[model_name]["format"]

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
                    _ef.DUMMY_COST = SVG_DUMMY_COST

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
                    _ef.DUMMY_COST = _original_dummy_cost

                # Always save metrics.json (even when visualization is off)
                method_dir = output_dir / episode_id / model_name
                method_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = method_dir / "metrics.json"
                if not metrics_path.exists():
                    import json as _json
                    with open(metrics_path, "w", encoding="utf-8") as _f:
                        _json.dump(res, _f, indent=2, default=str)

                results[model_name].append(res)
                episode_model_results[model_name] = res
                logger.info(
                    f"[{episode_id}][{model_name}] Done: "
                    f"matched={res['counts']['matched_pairs']} "
                    f"FN={res['counts']['fn']} FP={res['counts']['fp']}"
                )

                # Release pred elements immediately
                del pred_elements

            # Send progress update for this episode
            if progress_queue is not None:
                if episode_model_results:
                    progress_queue.put({
                        "status": "done_episode",
                        "episode_id": episode_id,
                        "model_results": episode_model_results,
                    })
                else:
                    # All models were skipped (resume) — count for progress
                    # but don't re-accumulate stats (already preloaded)
                    progress_queue.put({
                        "status": "resumed_episode",
                        "episode_id": episode_id,
                    })

            # Release GT data for this episode
            del gt_elements, gt_recon_img

        except Exception as e:
            logger.error(f"[{episode_id}] Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if progress_queue is not None:
                progress_queue.put({
                    "status": "error_episode",
                    "episode_id": episode_id,
                })

        # Free GPU memory after each episode
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Worker {worker_id} done ({task_count} tasks): " + ", ".join(
        f"{m}={len(results[m])}" for m in models_to_eval
    ))
    logger.close()
    return results


def worker_wrapper(worker_id, gpu_id, task_queue, args_dict, result_queue, progress_queue):
    """Wrapper that puts results in the queue."""
    try:
        results = worker_process(worker_id, gpu_id, task_queue, args_dict, progress_queue)
        result_queue.put({"worker_id": worker_id, "results": results})
    except Exception as e:
        import traceback
        print(f"Worker {worker_id} CRASHED: {e}")
        traceback.print_exc()
        result_queue.put({"worker_id": worker_id, "results": {}})


# ---------------------------------------------------------------------------
# Progress monitor process
# ---------------------------------------------------------------------------

def _safe_mean(vals, is_psnr=False):
    """NaN-safe mean. For PSNR, replaces inf with max finite value."""
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


def progress_monitor(progress_queue, stop_event, total_tasks, model_names,
                     preloaded_results=None):
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
    total_errors = 0

    def _accumulate(model_results):
        """Accumulate one episode's model results into stats."""
        for m_key, res in model_results.items():
            if m_key not in stats:
                continue
            s = stats[m_key]
            s["count"] += 1

            # Visual Quality (intersection, union)
            em_dual = res.get("element_metrics_dual")
            if em_dual:
                vq_dual = em_dual.get("visual_quality", {})
                for reg, target in [("intersection_region", "vq_inter"), ("union_region", "vq_union"), ("gt_region", "vq_gt")]:
                    region_data = vq_dual.get(reg, {}).get("simple_avg", {})
                    for met in ["l1", "l2", "psnr"]:
                        val = region_data.get(met)
                        if val is not None:
                            s[target][met].append(val)

                # Layout (soft/binary)
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
                # Fallback: use element_metrics (non-dual format)
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

    # Pre-populate stats from preloaded results (resume mode)
    if preloaded_results:
        # Group by episode_id to count unique episodes
        per_episode: Dict[str, Dict] = {}
        for m_key, results_list in preloaded_results.items():
            for res in results_list:
                eid = res.get("episode_id", "")
                if eid not in per_episode:
                    per_episode[eid] = {}
                per_episode[eid][m_key] = res
        for _eid, model_results in per_episode.items():
            _accumulate(model_results)
            total_processed += 1
        preloaded_count = len(per_episode)
        if pbar is not None:
            pbar.n = preloaded_count
            pbar.refresh()
        writer = tqdm.write if tqdm is not None else print
        writer(f"[Resume] Pre-loaded {preloaded_count} episodes from disk")

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
            f" [CUMULATIVE SUMMARY]  Processed: {total_processed}/{total_tasks}  (skipped: {total_skipped}, errors: {total_errors})",
            "-" * 120,
        ]

        # Build header
        header = f" {'Category / Metric':<28}"
        for m in model_names:
            header += f" | {m.upper():<30}"
        lines.append(header)
        lines.append("-" * 120)

        # Visual Quality - Intersection
        lines.append(" [Visual - Intersection]")
        row = f"  L1 / L2 / PSNR             "
        for m in model_names:
            s = stats[m]
            if s["count"] > 0:
                row += f" | L1:{_fmt_vq(s,'vq_inter','l1'):.4f} L2:{_fmt_vq(s,'vq_inter','l2'):.4f} PSNR:{_fmt_vq(s,'vq_inter','psnr'):.2f}"
            else:
                row += f" | {'N/A':>30}"
        lines.append(row)

        # Visual Quality - Union
        lines.append(" [Visual - Union]")
        row = f"  L1 / L2 / PSNR             "
        for m in model_names:
            s = stats[m]
            if s["count"] > 0:
                row += f" | L1:{_fmt_vq(s,'vq_union','l1'):.4f} L2:{_fmt_vq(s,'vq_union','l2'):.4f} PSNR:{_fmt_vq(s,'vq_union','psnr'):.2f}"
            else:
                row += f" | {'N/A':>30}"
        lines.append(row)

        # Visual Quality - GT Region
        lines.append(" [Visual - GT Region]")
        row = f"  L1 / L2 / PSNR             "
        for m in model_names:
            s = stats[m]
            if s["count"] > 0 and s["vq_gt"]["l1"]:
                row += f" | L1:{_fmt_vq(s,'vq_gt','l1'):.4f} L2:{_fmt_vq(s,'vq_gt','l2'):.4f} PSNR:{_fmt_vq(s,'vq_gt','psnr'):.2f}"
            else:
                row += f" | {'N/A':>30}"
        lines.append(row)

        # Layout - Soft
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

        # Layout - Binary
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

        # Composite
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

        # Episode counts per model
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
            status = update.get("status")
            if status == "skipped_episode":
                total_skipped += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"skip={total_skipped} done={total_processed} err={total_errors}")
                continue
            if status == "resumed_episode":
                # Already counted in preloaded stats; do NOT advance pbar
                continue
            if status == "error_episode":
                total_errors += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"skip={total_skipped} done={total_processed} err={total_errors}")
                continue
            if status == "done_episode":
                total_processed += 1
                if pbar is not None:
                    pbar.update(1)

                model_results = update.get("model_results", {})
                _accumulate(model_results)
                _print_table()

        except Empty:
            continue

    if pbar is not None:
        pbar.close()


# ---------------------------------------------------------------------------
# Copy existing results
# ---------------------------------------------------------------------------

def copy_existing_results(
    existing_dir: Path,
    output_dir: Path,
    episode_ids: Set[str],
    models: List[str] = ("agent", "qwen"),
) -> Dict[str, int]:
    """Copy existing agent/qwen metrics.json into the new output directory."""
    counts = {m: 0 for m in models}
    for eid in sorted(episode_ids):
        for model in models:
            src = existing_dir / eid / model / "metrics.json"
            if src.exists():
                dst = output_dir / eid / model / "metrics.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                counts[model] += 1
    return counts


def load_existing_per_episode_results(
    existing_dir: Path,
    episode_ids: Set[str],
    models: List[str] = ("agent", "qwen"),
) -> Dict[str, List[Dict]]:
    """Load existing per-episode metrics.json for aggregation."""
    results: Dict[str, List[Dict]] = {m: [] for m in models}
    for eid in sorted(episode_ids):
        for model in models:
            src = existing_dir / eid / model / "metrics.json"
            if src.exists():
                try:
                    with open(src) as f:
                        data = json.load(f)
                    data["episode_id"] = eid
                    data["method"] = model
                    results[model].append(data)
                except Exception:
                    pass
    return results


def _load_metrics_from_output_dir(
    output_dir: Path,
    model_names: List[str],
) -> Dict[str, List[Dict]]:
    """Load all existing metrics.json from output directory for resume."""
    results: Dict[str, List[Dict]] = {m: [] for m in model_names}
    if not output_dir.exists():
        return results
    for eid_dir in sorted(output_dir.iterdir()):
        if not eid_dir.is_dir() or eid_dir.name.startswith("worker_"):
            continue
        eid = eid_dir.name
        for model_name in model_names:
            metrics_path = eid_dir / model_name / "metrics.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        data = json.load(f)
                    data["episode_id"] = eid
                    data["method"] = model_name
                    results[model_name].append(data)
                except Exception:
                    pass
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Accuracy evaluation for baseline models (extends evaluation_figma.py)"
    )
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True,
                        help="Format: agent_dir:qwen_dir:gt_subset_prefix")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Baseline models to evaluate")
    add_baseline_dir_args(parser)
    parser.add_argument("--existing-results", type=str, default=None,
                        help="Path to existing agent/qwen evaluation results to copy")
    parser.add_argument("--output", type=str, default=None,
                        help="Output root directory. Required unless --resume-dir is set.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--gpu-workers", type=str, default=None,
                        help="Per-GPU worker counts, e.g. '0:1,1:2,2:1,3:4'. "
                             "Overrides --num-workers and --gpu-ids.")
    parser.add_argument("--resume-dir", type=str, default=None,
                        help="Resume from a previous output directory "
                             "(skips already-completed episodes). "
                             "If set, no new HHMMSS subfolder is created.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--no-viz", action="store_true", default=True,
                        help="Skip visualization generation (default: True)")
    parser.add_argument("--matching", type=str, default="optimal",
                        choices=["optimal", "legacy"])
    parser.add_argument("--text-refinement-mode", type=str, default="hybrid",
                        choices=["kill", "correct", "hybrid"])

    args = parser.parse_args()

    # Output directory: resume or new timestamped
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

    figma_data_dir = Path(args.figma_data)

    # Build worker→GPU mapping
    if args.gpu_workers:
        # Per-GPU worker counts: "0:1,1:2,2:1,3:4"
        worker_gpu_ids = []
        for spec in args.gpu_workers.split(","):
            gpu_str, count_str = spec.split(":")
            gpu_id = int(gpu_str.strip())
            count = int(count_str.strip())
            worker_gpu_ids.extend([gpu_id] * count)
        num_workers = len(worker_gpu_ids)
        gpu_ids = sorted(set(worker_gpu_ids))
    else:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        num_workers = min(args.num_workers, len(gpu_ids))
        worker_gpu_ids = gpu_ids[:num_workers]

    print("=" * 80)
    print("BASELINE ACCURACY EVALUATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")
    print(f"Models to evaluate: {args.models}")
    print(f"Workers: {num_workers}, GPU mapping: {worker_gpu_ids}")
    print("=" * 80)

    # 1. Collect GT episodes
    gt_map = collect_gt_episodes(figma_data_dir, args.exp_pairs)
    print(f"\nGT episodes: {len(gt_map)}")

    # 2. Scan episodes for all models (including agent/qwen if requested)
    all_episode_tasks: Dict[str, Dict] = {}

    for model_name in args.models:
        # Agent/Qwen may span multiple experiment directories
        multi_dirs = _resolve_multi_dirs(args, model_name)
        if multi_dirs is not None:
            model_map = scan_model_episodes_multi(model_name, multi_dirs)
        else:
            base_dir = get_model_dir(args, model_name)
            model_map = scan_model_episodes(model_name, base_dir)
        common = get_common_episodes(gt_map, model_map)
        print(f"  {model_name}: {len(model_map)} episodes, {len(common)} common with GT")

        for eid, info in common.items():
            if eid not in all_episode_tasks:
                all_episode_tasks[eid] = {
                    "episode_id": eid,
                    "gt_json_path": info["gt_json_path"],
                    "split_name": info["split_name"],
                    "split_dir": info["split_dir"],
                }
            all_episode_tasks[eid][f"model_dir_{model_name}"] = str(info["model_dir"])

    print(f"\nTotal unique episodes to evaluate: {len(all_episode_tasks)}")

    # Apply max-episodes cap
    task_list = list(all_episode_tasks.values())
    task_list.sort(key=lambda t: t["episode_id"])
    if args.max_episodes:
        task_list = task_list[:args.max_episodes]
        print(f"Capped to {len(task_list)} episodes")

    if not task_list:
        print("No episodes to evaluate!")
        return

    # 3. Copy existing agent/qwen results if provided
    all_episode_ids = {t["episode_id"] for t in task_list}
    if args.existing_results:
        existing_dir = Path(args.existing_results)
        print(f"\nCopying existing results from {existing_dir}...")
        copy_counts = copy_existing_results(existing_dir, output_dir, all_episode_ids)
        for m, c in copy_counts.items():
            print(f"  {m}: {c} episodes copied")

    # 4. Distribute tasks to workers via shared queue (dynamic dispatch)
    print(f"\nTask distribution: {len(task_list)} tasks -> {num_workers} workers (dynamic queue)")
    for i in range(num_workers):
        print(f"  Worker {i} (GPU {worker_gpu_ids[i]})")

    args_dict = {
        "output": str(output_dir),
        "no_viz": args.no_viz,
        "matching": args.matching,
        "models_to_eval": args.models,
    }

    # 5. Run workers with progress monitoring
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    progress_queue = manager.Queue()
    stop_event = Event()

    # Fill the shared task queue
    for task in task_list:
        task_queue.put(task)

    # Load preloaded results for resume mode
    preloaded_results = None
    if args.resume_dir:
        print("Loading existing metrics from disk for resume...")
        preloaded_results = _load_metrics_from_output_dir(output_dir, args.models)
        for m in args.models:
            print(f"  {m}: {len(preloaded_results.get(m, []))} episodes loaded")

    # Start progress monitor
    monitor = Process(
        target=progress_monitor,
        args=(progress_queue, stop_event, len(task_list), args.models,
              preloaded_results),
    )
    monitor.start()

    start_time = time.time()
    processes = []
    for i in range(num_workers):
        p = Process(
            target=worker_wrapper,
            args=(i, worker_gpu_ids[i], task_queue, args_dict, result_queue, progress_queue),
        )
        p.start()
        processes.append(p)
        print(f"  Worker {i} started (PID: {p.pid}, GPU: {worker_gpu_ids[i]})")

    for p in processes:
        p.join()

    # Stop progress monitor
    stop_event.set()
    monitor.join(timeout=10)

    elapsed_time = time.time() - start_time
    print(f"\nAll workers completed in {elapsed_time:.2f}s")

    # 6. Collect results — load ALL metrics from disk (includes both
    #    pre-existing and newly evaluated episodes)
    all_results: Dict[str, List] = _load_metrics_from_output_dir(output_dir, args.models)

    # Also print per-worker stats from this run
    while not result_queue.empty():
        wr = result_queue.get()
        for m in args.models:
            model_results = wr.get("results", {}).get(m, [])
            if model_results:
                print(f"  Worker {wr['worker_id']}: {len(model_results)} {m}")
    for m in args.models:
        print(f"  Total {m} results (disk): {len(all_results.get(m, []))}")

    # 7. Load existing agent/qwen results for unified summary (only if not evaluated fresh)
    if args.existing_results:
        models_needing_existing = [m for m in ("agent", "qwen") if m not in args.models]
        if models_needing_existing:
            existing = load_existing_per_episode_results(
                Path(args.existing_results), all_episode_ids, models=models_needing_existing
            )
            for m in models_needing_existing:
                if existing[m]:
                    all_results[m] = existing[m]
                    print(f"  Loaded {len(existing[m])} existing {m} results")

    # 8. Aggregate and save unified summary
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)

    from evaluation.figma_metrics import aggregate_results

    # Include all evaluated models + any pre-loaded ones
    all_model_names = list(args.models)
    for m in ("agent", "qwen"):
        if m not in all_model_names and m in all_results and all_results[m]:
            all_model_names.append(m)

    all_summaries: Dict[str, Any] = {}
    for model_name in all_model_names:
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
        print(f"  Composite ({n_comp} eps): L1={comp.get('l1', float('nan')):.4f}, PSNR={comp.get('psnr', float('nan')):.2f}, SSIM={comp.get('ssim', float('nan')):.4f}")
        print(f"  LPIPS={comp.get('lpips', float('nan')):.4f}, DINO={comp.get('dino', float('nan')):.4f}")

    # Save unified summary
    unified = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "exp_pairs": args.exp_pairs,
            "models_evaluated": args.models,
            "matching_algorithm": args.matching,
            "num_workers": num_workers,
            "worker_gpu_ids": worker_gpu_ids,
            "elapsed_time_seconds": elapsed_time,
            "existing_results_source": args.existing_results,
        },
        "results": {
            "all_episodes": all_summaries,
        },
        "per_model_episode_counts": {
            m: len(all_results.get(m, [])) for m in all_model_names
        },
        "per_episode_details": {},
    }

    for model_name in all_model_names:
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
    main()
