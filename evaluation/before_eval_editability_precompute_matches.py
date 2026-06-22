#!/usr/bin/env python3
"""Pre-compute GT-to-pred matches for baseline models.

Produces match JSONs in the same format as evaluation/editability_utils/match_runner.py,
saving to {output}/{model_name}/episodes/{episode_id}.json.

Usage:
    python scripts/precompute_baseline_matches.py \
        --figma-data figma_data \
        --model layered --model-dir <LAYERED_BASELINE_OUTPUT_DIR> \
        --output <MATCH_ROOT_DIR> \
        --num-workers 4

    The agent/qwen/baseline output dirs are produced by running the inference runners
    first (e.g. ``python -m REDESIGN.run_agent_figma --data_dir figma_data \
    --output_dir <AGENT_OUTPUT_DIR>``), and ``--figma-data`` should point at the
    downloaded ``figma_data`` dataset.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.baseline_model_configs import (
    MODEL_CONFIGS,
    add_baseline_dir_args,
    collect_gt_episodes,
    get_common_episodes,
    get_model_dir,
    scan_model_episodes,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class BaselineTask:
    episode_id: str
    split_name: str
    split_dir: Path
    gt_json_path: Path
    pred_dir: Path
    model_format: str  # "agent", "qwen", or "omnisvg"


def _load_elements(task: BaselineTask) -> Tuple[List[Dict], List[Dict], Tuple[int, int]]:
    """Load GT and pred elements for a baseline task."""
    import pickle
    from evaluation.figma_metrics import (
        extract_agent_elements,
        extract_gt_elements,
        extract_omnisvg_elements,
        extract_qwen_elements_cca,
    )
    from evaluation.editability_utils.loaders import _attach_gt_metadata

    gt_elements, canvas_size, _ = extract_gt_elements(
        task.gt_json_path, task.split_dir, logger=None
    )
    _attach_gt_metadata(gt_elements, task.gt_json_path)

    if task.model_format == "qwen":
        pred_elements = extract_qwen_elements_cca(
            task.pred_dir, canvas_size, logger=None
        )
        for elem in pred_elements:
            elem.setdefault("meta", {})
    elif task.model_format == "omnisvg":
        # Render pred at 1/2 resolution WITHOUT upscale — matching at lower res (4x faster).
        # 0.5 scale preserves enough detail for accurate matching while being fast.
        # (0.25 was too aggressive — caused excessive MIN_ELEMENT_AREA filtering.)
        _MATCH_SCALE = 0.5
        gt_bboxes = [tuple(g.get("bbox", [0, 0, 1, 1])) for g in gt_elements]
        pred_elements = extract_omnisvg_elements(
            task.pred_dir, canvas_size, logger=None,
            filter_bboxes=gt_bboxes,
            render_scale=_MATCH_SCALE,
            skip_upscale=True,
        )
        for elem in pred_elements:
            elem.setdefault("meta", {})

        # Downscale GT to match pred resolution
        from PIL import Image as _PILImage
        import cv2 as _cv2
        W, H = canvas_size
        sW, sH = max(1, int(W * _MATCH_SCALE)), max(1, int(H * _MATCH_SCALE))
        canvas_size = (sW, sH)
        for elem in gt_elements:
            if "image" in elem and elem["image"] is not None:
                elem["image"] = elem["image"].resize((sW, sH), _PILImage.NEAREST)
            if "mask" in elem and elem["mask"] is not None:
                elem["mask"] = _cv2.resize(elem["mask"], (sW, sH), interpolation=_cv2.INTER_NEAREST)
            bbox = elem.get("bbox", [0, 0, W, H])
            elem["bbox"] = [
                int(bbox[0] * _MATCH_SCALE),
                int(bbox[1] * _MATCH_SCALE),
                max(1, int(bbox[2] * _MATCH_SCALE)),
                max(1, int(bbox[3] * _MATCH_SCALE)),
            ]
    else:
        pred_elements = extract_agent_elements(
            task.pred_dir, canvas_size,
            apply_alpha_correction=True,
            text_refinement=True,
            logger=None,
        )
        from evaluation.editability_utils.loaders import _attach_agent_metadata
        _attach_agent_metadata(pred_elements, task.pred_dir)

    return gt_elements, pred_elements, canvas_size


class _EpisodeTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _EpisodeTimeout("Episode matching timed out")


def _match_single_episode(
    task: BaselineTask,
    per_episode_dir: Path,
    cfg: Any,
    timeout: int = 0,
) -> Dict[str, Any]:
    """Run matching for a single episode and save result."""
    import signal
    from evaluation.editability_utils.matching_core import greedy_match_gt_to_pred
    from evaluation.editability_utils.common_utils import save_json

    # Set per-episode timeout (SIGALRM, works in forked subprocesses)
    if timeout > 0:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)

    try:
        return _match_single_episode_inner(task, per_episode_dir, cfg)
    except _EpisodeTimeout:
        return {
            "episode_id": task.episode_id,
            "split": task.split_name,
            "counts": {"gt": 0, "pred": 0, "matches": 0},
            "timing_sec": {"load": 0, "match": 0, "total": 0},
            "match_stats": {},
            "path": "",
            "status": "timeout",
        }
    except Exception as e:
        # Catch ALL exceptions inside the worker so the future never
        # propagates an opaque error to the main process.
        import traceback
        tb = traceback.format_exc()
        err_msg = f"[WORKER ERROR] {task.episode_id}: {e}\n{tb}"
        # Write to a per-episode error log so it's never lost
        err_log = per_episode_dir / f"{task.episode_id}.error.log"
        try:
            err_log.write_text(err_msg)
        except Exception:
            pass
        print(err_msg, flush=True)
        return {
            "episode_id": task.episode_id,
            "split": task.split_name,
            "counts": {"gt": 0, "pred": 0, "matches": 0},
            "timing_sec": {"load": 0, "match": 0, "total": 0},
            "match_stats": {},
            "path": "",
            "status": "error",
            "error": str(e),
        }
    finally:
        if timeout > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def _match_single_episode_inner(
    task: BaselineTask,
    per_episode_dir: Path,
    cfg: Any,
) -> Dict[str, Any]:
    """Core matching logic (separated for timeout wrapper)."""
    from evaluation.editability_utils.matching_core import greedy_match_gt_to_pred
    from evaluation.editability_utils.common_utils import save_json

    ep_t0 = time.time()
    load_t0 = time.time()
    gt_elements, pred_elements, canvas_size = _load_elements(task)
    load_dt = time.time() - load_t0

    match_t0 = time.time()
    matches, match_stats = greedy_match_gt_to_pred(
        gt_elements, pred_elements, canvas_size, cfg, progress_cb=None
    )
    match_dt = time.time() - match_t0

    episode_payload = {
        "episode_id": task.episode_id,
        "split": task.split_name,
        "canvas_size": list(canvas_size),
        "counts": {
            "gt": len(gt_elements),
            "pred": len(pred_elements),
            "matches": len(matches),
        },
        "timing_sec": {
            "load": float(load_dt),
            "match": float(match_dt),
            "total": float(time.time() - ep_t0),
        },
        "match_stats": match_stats,
        "matches": matches,
    }

    episode_path = per_episode_dir / f"{task.episode_id}.json"
    save_json(episode_path, episode_payload)
    episode_payload["timing_sec"]["save"] = float(time.time() - ep_t0) - load_dt - match_dt
    episode_payload["timing_sec"]["total"] = float(time.time() - ep_t0)

    return {
        "episode_id": task.episode_id,
        "split": task.split_name,
        "counts": episode_payload["counts"],
        "timing_sec": episode_payload["timing_sec"],
        "match_stats": match_stats,
        "path": str(episode_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute GT-to-pred matches for a baseline model"
    )
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Root directory of the baseline model experiment")
    parser.add_argument("--output", type=str, required=True,
                        help="Match root directory (e.g., editability_matches/merge_sweep_fast/merge_max)")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--lambda-l1", type=float, default=0.5)
    parser.add_argument("--lambda-iou", type=float, default=0.5)
    parser.add_argument("--max-merge-n", type=int, default=None,
                        help="Max greedy merge length (None = unlimited)")
    parser.add_argument("--min-gt-overlap", type=float, default=0.05,
                        help="Candidate filter: bbox intersection / GT bbox area threshold")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Max candidate preds per GT element (top-K by bbox overlap, None = unlimited)")
    parser.add_argument("--min-cost-improve", type=float, default=0.01,
                        help="Greedy merge early stop threshold")
    parser.add_argument("--l1-mode", type=str, default="rgba",
                        choices=["rgb", "rgba"])
    parser.add_argument("--per-episode-timeout", type=int, default=0,
                        help="Per-episode timeout in seconds (0 = no timeout). "
                             "Episodes exceeding this are skipped.")
    parser.add_argument("--trace-episodes", action="store_true")

    args = parser.parse_args()

    model_name = args.model
    model_format = MODEL_CONFIGS[model_name]["format"]
    figma_data_dir = Path(args.figma_data)
    model_base_dir = Path(args.model_dir)
    output_root = Path(args.output)

    per_episode_dir = output_root / model_name / "episodes"
    per_episode_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"BASELINE MATCH PRECOMPUTATION: {model_name}")
    print("=" * 80)
    print(f"Model format: {model_format}")
    print(f"Model dir: {model_base_dir}")
    print(f"Output: {per_episode_dir}")
    print(f"Workers: {args.num_workers}")

    # 1. Collect GT episodes
    gt_map = collect_gt_episodes(figma_data_dir)
    print(f"GT episodes: {len(gt_map)}")

    # 2. Scan model episodes
    model_map = scan_model_episodes(model_name, model_base_dir)
    print(f"Model episodes: {len(model_map)}")

    # 3. Intersect
    common = get_common_episodes(gt_map, model_map)
    print(f"Common episodes: {len(common)}")

    # Build tasks
    tasks: List[BaselineTask] = []
    skipped = 0
    for eid in sorted(common.keys()):
        info = common[eid]
        if args.skip_existing and (per_episode_dir / f"{eid}.json").exists():
            skipped += 1
            continue
        tasks.append(BaselineTask(
            episode_id=eid,
            split_name=info["split_name"],
            split_dir=info["split_dir"],
            gt_json_path=info["gt_json_path"],
            pred_dir=info["model_dir"],
            model_format=model_format,
        ))

    if args.max_episodes:
        tasks = tasks[:args.max_episodes]

    print(f"Skipped existing: {skipped}")
    print(f"Tasks to run: {len(tasks)}")

    if not tasks:
        print("Nothing to do!")
        # Save summary
        from evaluation.editability_utils.common_utils import save_json
        save_json(output_root / model_name / "summary.json", {
            "model": model_name,
            "num_episodes": len(common),
            "num_skipped_existing": skipped,
            "num_to_run": 0,
        })
        return

    # Create match config
    from evaluation.editability_utils.matching_core import MatchConfig
    cfg = MatchConfig(
        lambda_l1=args.lambda_l1,
        lambda_iou=args.lambda_iou,
        max_merge_n=args.max_merge_n,
        min_gt_overlap=args.min_gt_overlap,
        min_cost_improve=args.min_cost_improve,
        l1_mode=args.l1_mode,
        max_candidates=args.max_candidates,
    )

    t0 = time.time()
    episodes_out: List[Dict[str, Any]] = []

    # Cumulative stats tracker
    cum_gt = 0
    cum_pred = 0
    cum_matches = 0
    cum_match_time = 0.0

    def _update_cum_stats(result):
        nonlocal cum_gt, cum_pred, cum_matches, cum_match_time
        cum_gt += result["counts"]["gt"]
        cum_pred += result["counts"]["pred"]
        cum_matches += result["counts"]["matches"]
        cum_match_time += result["timing_sec"]["match"]

    def _cum_postfix():
        n = len(episodes_out)
        if n == 0:
            return {}
        return {
            "avg_gt": f"{cum_gt/n:.1f}",
            "avg_pred": f"{cum_pred/n:.1f}",
            "avg_match": f"{cum_matches/n:.1f}",
            "avg_t": f"{cum_match_time/n:.2f}s",
        }

    if args.num_workers <= 1:
        # Single-worker mode
        iterable = tasks
        progress = None
        if tqdm is not None:
            progress = tqdm(tasks, desc=f"matching:{model_name}", unit="ep", dynamic_ncols=True)
            iterable = progress

        timeouts = 0
        for i, task in enumerate(iterable):
            try:
                result = _match_single_episode(task, per_episode_dir, cfg, timeout=args.per_episode_timeout)
                if result.get("status") == "timeout":
                    timeouts += 1
                    msg = f"[TIMEOUT] {task.episode_id} (>{args.per_episode_timeout}s)"
                    if tqdm is not None:
                        tqdm.write(msg)
                    else:
                        print(msg)
                    continue
                episodes_out.append(result)
                _update_cum_stats(result)
                if progress is not None:
                    progress.set_postfix({**_cum_postfix(), "tout": timeouts})
                if args.trace_episodes:
                    msg = (
                        f"[{model_name}] {i+1}/{len(tasks)} {task.episode_id} "
                        f"gt={result['counts']['gt']} pred={result['counts']['pred']} "
                        f"matches={result['counts']['matches']} "
                        f"match={result['timing_sec']['match']:.2f}s"
                    )
                    if tqdm is not None:
                        tqdm.write(msg)
                    else:
                        print(msg)
            except Exception as e:
                print(f"[ERROR] {task.episode_id}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Multi-worker mode – use fork (matching is CPU-only, no CUDA)
        # spawn causes each child to re-import evaluation_figma (torch, lpips, …)
        # which is extremely slow and can hang.
        mp_method = "fork"
        print(f"Using {args.num_workers} workers ({mp_method})")
        progress = None
        if tqdm is not None:
            progress = tqdm(total=len(tasks), desc=f"matching:{model_name}", unit="ep", dynamic_ncols=True)

        ctx = mp.get_context(mp_method)
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=ctx,
        ) as ex:
            futures = {}
            for task in tasks:
                fut = ex.submit(_match_single_episode, task, per_episode_dir, cfg, timeout=args.per_episode_timeout)
                futures[fut] = task

            pending = set(futures.keys())
            errors = 0
            timeouts = 0
            while pending:
                finished, pending = wait(pending, timeout=5.0, return_when=FIRST_COMPLETED)
                for fut in finished:
                    task = futures[fut]
                    try:
                        result = fut.result()
                        status = result.get("status")
                        if status == "timeout":
                            timeouts += 1
                            if tqdm is not None:
                                tqdm.write(f"[TIMEOUT] {task.episode_id} (>{args.per_episode_timeout}s)")
                            else:
                                print(f"[TIMEOUT] {task.episode_id}")
                        elif status == "error":
                            errors += 1
                            err_detail = result.get("error", "unknown")
                            msg = f"[ERROR] {task.episode_id}: {err_detail} (see {per_episode_dir / f'{task.episode_id}.error.log'})"
                            if tqdm is not None:
                                tqdm.write(msg)
                            else:
                                print(msg)
                        else:
                            episodes_out.append(result)
                            _update_cum_stats(result)
                    except Exception as e:
                        errors += 1
                        print(f"\n[ERROR] {task.episode_id}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix({**_cum_postfix(), "err": errors, "tout": timeouts})

        if progress is not None:
            progress.close()

    elapsed = time.time() - t0
    tout_msg = f" (timeouts: {timeouts})" if timeouts else ""
    print(f"\nCompleted {len(episodes_out)}/{len(tasks)} episodes in {elapsed:.1f}s{tout_msg}")

    # Save summary
    from evaluation.editability_utils.common_utils import save_json
    summary = {
        "model": model_name,
        "model_format": model_format,
        "num_episodes": len(common),
        "num_skipped_existing": skipped,
        "num_to_run": len(tasks),
        "num_completed": len(episodes_out),
        "elapsed_seconds": elapsed,
        "config": {
            "lambda_l1": cfg.lambda_l1,
            "lambda_iou": cfg.lambda_iou,
            "max_merge_n": cfg.max_merge_n,
            "max_candidates": cfg.max_candidates,
            "l1_mode": cfg.l1_mode,
            "num_workers": args.num_workers,
        },
        "episodes": episodes_out,
    }
    save_json(output_root / model_name / "summary.json", summary)
    print(f"Saved summary to {output_root / model_name / 'summary.json'}")


if __name__ == "__main__":
    main()
