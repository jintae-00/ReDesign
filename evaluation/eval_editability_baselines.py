#!/usr/bin/env python3
"""Editability evaluation for baseline models — parallel execution.

Runs 3 baselines (layered, multi_tools, sparse_verif) IN PARALLEL per subtask
and shows running averages for ALL 5 models (+ pre-loaded agent/qwen results).

Prerequisites:
    1. Pre-computed matches at match_root/{model_name}/episodes/
    2. Original agent/qwen editability results at --original-run-dir

Usage:
    # Replace <GPU_ID> with one of your own GPU ids (e.g. 0).
    CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/eval_editability_baselines.py \
        --figma-data figma_data \
        --agent-dir <AGENT_OUTPUT_DIR> \
        --qwen-dir <QWEN_OUTPUT_DIR> \
        --match-root <MATCH_ROOT_DIR> \
        --models layered multi_tools sparse_verif \
        --layered-dir <LAYERED_BASELINE_OUTPUT_DIR> \
        --multi-tools-dir <MULTI_TOOLS_BASELINE_OUTPUT_DIR> \
        --sparse-verif-dir <SPARSE_VERIF_BASELINE_OUTPUT_DIR> \
        --original-run-dir <ORIGINAL_EDITABILITY_RESULTS_DIR> \
        --output <OUTPUT_DIR> \
        --seed 42 --selection-seed 42 --per-episode-elements 2 \
        --num-workers 32 --no-save-triplet-viz

    The agent/qwen/baseline output dirs are produced by running the inference runners
    first (e.g. ``python -m ReDesign.run_agent_figma --data_dir figma_data \
    --output_dir <AGENT_OUTPUT_DIR>``), and ``--figma-data`` should point at the
    downloaded ``figma_data`` dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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


# ---------------------------------------------------------------------------
# Baseline EpisodeTask and EpisodeCache
# ---------------------------------------------------------------------------

@dataclass
class BaselineEpisodeTask:
    """Lightweight episode task for baselines."""
    episode_id: str
    split_name: str
    split_dir: Path
    gt_json_path: Path
    pred_dir: Path
    model_format: str  # "agent", "qwen", or "omnisvg"


def _load_baseline_episode_elements(
    task: BaselineEpisodeTask,
    needed_pred_ids: Optional[Set[str]] = None,
) -> Tuple[List[Dict], List[Dict], Tuple[int, int]]:
    """Load GT + pred elements for a baseline episode.

    For omnisvg format, ``needed_pred_ids`` enables selective rendering:
    matched elements are rendered at full resolution, all others at 0.25 scale
    (upscaled to canvas_size).  This gives ~10x speedup for VTracer episodes
    with 700+ SVG paths while keeping ROI metrics accurate.
    """
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
        gt_bboxes = [tuple(g.get("bbox", [0, 0, 1, 1])) for g in gt_elements]
        # Full-res rendering for fairness (all models evaluated at same resolution).
        # filter_bboxes skips paths not overlapping GT (safe — they never affect metrics).
        pred_elements = extract_omnisvg_elements(
            task.pred_dir, canvas_size, logger=None,
            filter_bboxes=gt_bboxes,
        )
        for elem in pred_elements:
            elem.setdefault("meta", {})
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


class BaselineEpisodeCache:
    """LRU cache for baseline episode elements, compatible with EpisodeCache interface."""

    def __init__(
        self,
        task_map: Dict[str, BaselineEpisodeTask],
        max_items: Optional[int] = None,
        max_loaders: Optional[int] = None,
    ):
        self.task_map = task_map
        self.model = "baseline"
        # Per-episode needed pred IDs for selective rendering (set by caller).
        self.needed_pred_ids_map: Dict[str, Set[str]] = {}

        if max_items is None:
            try:
                max_items = int(os.environ.get("EDITABILITY_CACHE_EPISODES", "8"))
            except Exception:
                max_items = 8
        self.max_items = None if max_items is None or int(max_items) <= 0 else int(max_items)

        if max_loaders is None:
            try:
                max_loaders = int(os.environ.get("EDITABILITY_MAX_EP_LOADERS", "2"))
            except Exception:
                max_loaders = 2
        self.max_loaders = max(1, int(max_loaders))
        self._load_sem = Semaphore(self.max_loaders)
        self._cache: OrderedDict[str, Tuple[List, List, Tuple[int, int]]] = OrderedDict()
        self._cache_lock = Lock()
        # Per-episode locks: prevent two threads from loading the same episode
        self._ep_locks: Dict[str, Lock] = {}
        self._ep_locks_guard = Lock()

    def _get_ep_lock(self, episode_id: str) -> Lock:
        with self._ep_locks_guard:
            if episode_id not in self._ep_locks:
                self._ep_locks[episode_id] = Lock()
            return self._ep_locks[episode_id]

    def get(self, episode_id: str):
        import time as _time
        import threading as _thr

        # Fast path: already cached
        with self._cache_lock:
            item = self._cache.get(episode_id)
            if item is not None:
                self._cache.move_to_end(episode_id)
                return item

        # Per-episode lock: only one thread loads a given episode
        ep_lock = self._get_ep_lock(episode_id)
        with ep_lock:
            # Re-check cache (another thread may have loaded while we waited)
            with self._cache_lock:
                item = self._cache.get(episode_id)
                if item is not None:
                    self._cache.move_to_end(episode_id)
                    return item

            _t0 = _time.monotonic()
            _tid = _thr.current_thread().name
            print(f"  [cache] {_tid} loading {episode_id} (waiting for sem ...)", flush=True)

            needed_ids = self.needed_pred_ids_map.get(episode_id)
            with self._load_sem:
                print(f"  [cache] {_tid} loading {episode_id} (sem acquired, rendering ...)", flush=True)
                loaded = _load_baseline_episode_elements(
                    self.task_map[episode_id],
                    needed_pred_ids=needed_ids,
                )
                _elapsed = _time.monotonic() - _t0
                gt_elems, pred_elems, cs = loaded
                print(
                    f"  [cache] {_tid} loaded {episode_id}: "
                    f"gt={len(gt_elems)} pred={len(pred_elems)} "
                    f"canvas={cs} {_elapsed:.1f}s",
                    flush=True,
                )

            with self._cache_lock:
                self._cache[episode_id] = loaded
                self._cache.move_to_end(episode_id)
                if self.max_items is not None:
                    while len(self._cache) > self.max_items:
                        self._cache.popitem(last=False)
                return loaded


# ---------------------------------------------------------------------------
# Cache pre-warming for z_order (parallel episode loading)
# ---------------------------------------------------------------------------

def _prewarm_cache_parallel(
    cache: BaselineEpisodeCache,
    episode_ids: List[str],
    max_workers: int = 8,
) -> None:
    """Pre-load episodes into cache in parallel using ThreadPoolExecutor."""
    from concurrent.futures import as_completed as _as_completed

    with cache._cache_lock:
        to_load = [eid for eid in episode_ids if eid not in cache._cache]
    if not to_load:
        print(f"  [prewarm] all {len(episode_ids)} episodes already cached")
        return
    max_workers = max(1, min(max_workers, len(to_load)))
    print(f"  [prewarm] loading {len(to_load)} episodes with {max_workers} workers ...")

    # Temporarily increase semaphore to allow parallel loading
    old_sem = cache._load_sem
    cache._load_sem = Semaphore(max_workers)

    def _load_one(eid: str) -> str:
        cache.get(eid)
        return eid

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_load_one, eid): eid for eid in to_load}
        done = 0
        for fut in _as_completed(futs):
            fut.result()  # propagate exceptions
            done += 1
            if done % 50 == 0 or done == len(to_load):
                print(f"  [prewarm] {done}/{len(to_load)} loaded")

    cache._load_sem = old_sem
    print(f"  [prewarm] done, cache size={len(cache._cache)}")


# ---------------------------------------------------------------------------
# Run atomic subtask for a baseline model
# ---------------------------------------------------------------------------

def _make_worst_case_result(
    episode_id: str,
    gt_index: int,
    task_type: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a worst-case result for a (episode, gt_index) with no matching."""
    worst = {
        "l1": 1.0, "l2": 1.0, "psnr": 0.0, "ssim": 0.0,
        "lpips": 1.0, "dino": 0.0,
        "full_l1": 1.0, "full_l2": 1.0, "full_psnr": 0.0,
        "full_ssim": 0.0, "full_lpips": 1.0, "full_dino": 0.0,
    }
    return {
        "episode_id": episode_id,
        "gt_index": gt_index,
        "pred_indices": [],
        "task_type": task_type,
        "params": params,
        "applied_edit": "worst_case_no_match",
        "metrics": worst,
        "metrics_full": worst,
    }


def run_baseline_atomic_subtask(
    *,
    task_type: str,
    param_grid: Sequence[Dict[str, Any]],
    model_name: str,
    match_root: Path,
    output_dir: Path,
    task_map: Dict[str, BaselineEpisodeTask],
    cache: BaselineEpisodeCache,
    seed: int,
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    max_tasks: Optional[int] = None,
    log_every: int = 0,
    save_pair_viz: bool = False,
    pair_viz_max: Optional[int] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    build_log_every: int = 0,
    include_iou: bool = False,
    roi_mode: str = "source",
    roi_dilation_ratio: float = 0.08,
    agent_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run an atomic subtask for a baseline model using pre-computed matches."""
    from evaluation.editability_utils.subtasks.common import (
        aggregate_results,
        evaluate_image_edit_candidates,
        load_match_payloads,
        sample_candidates,
        summarize_capacity,
    )
    from evaluation.editability_utils.subtasks.atomic._shared import (
        _build_base_candidates,
    )

    subtask_name = f"atomic_{task_type}"
    print(f"[{model_name}][{subtask_name}] init")

    payloads = load_match_payloads(match_root, model_name)
    payloads = [p for p in payloads if p["episode_id"] in task_map]
    print(f"[{model_name}][{subtask_name}] payloads={len(payloads)} episodes={len(task_map)}")

    # Pre-scan payloads to collect needed pred element IDs per episode.
    # This enables selective rendering: matched elements at full-res,
    # all others at low-res (scene-only).
    if hasattr(cache, "needed_pred_ids_map") and not cache.needed_pred_ids_map:
        for p in payloads:
            eid = p["episode_id"]
            ids: Set[str] = set()
            for m in p.get("matches", []):
                for pid in m.get("selected_pred_ids", []):
                    ids.add(str(pid))
                bsid = m.get("best_single_pred_id")
                if bsid:
                    ids.add(str(bsid))
            if ids:
                cache.needed_pred_ids_map[eid] = ids
        n_eps = len(cache.needed_pred_ids_map)
        n_ids = sum(len(v) for v in cache.needed_pred_ids_map.values())
        print(f"[{model_name}][{subtask_name}] selective rendering: "
              f"{n_ids} needed pred IDs across {n_eps} episodes")

    old_cache_max = cache.max_items

    old_model = cache.model
    cache.model = model_name
    candidates = _build_base_candidates(
        payloads,
        cache,
        task_type=task_type,
        param_grid=param_grid,
        seed=seed,
        subset_keys=subset_keys,
        build_log_every=build_log_every,
        model=model_name,
    )
    cache.model = old_model

    capacity = summarize_capacity(candidates)
    sampled = sample_candidates(
        candidates,
        max_tasks=max_tasks,
        seed=seed,
        balance_text_non_text=True,
        cache=cache,
    )

    expected_count = len(agent_results) if agent_results else len(sampled)

    print(
        f"[{model_name}][{subtask_name}] "
        f"episodes={len(task_map)} candidates={len(candidates)} "
        f"sampled={len(sampled)} expected={expected_count}"
    )

    # --- Increase loader concurrency for evaluation ---
    old_sem = cache._load_sem
    eval_loaders = max(cache.max_loaders, min(num_workers, 8))
    cache._load_sem = Semaphore(eval_loaders)

    results = evaluate_image_edit_candidates(
        sampled,
        cache,
        include_iou=include_iou,
        include_edge_sharpness=False,
        include_lpips=True,
        roi_mode=roi_mode,
        roi_dilation_ratio=roi_dilation_ratio,
        progress_prefix=f"[{model_name}][{subtask_name}]",
        log_every=log_every,
        save_pair_viz_dir=(output_dir / model_name / subtask_name / "element_pairs") if save_pair_viz else None,
        pair_viz_max=pair_viz_max,
        reference_metrics=None,
        reference_label=None,
        model_label=model_name,
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        checkpoint_path=output_dir / model_name / f"{subtask_name}_results.json",
        resume=True,
    )

    # --- Restore semaphore and cache size ---
    cache._load_sem = old_sem
    cache.max_items = old_cache_max

    # Pad with worst-case for missing (episode_id, gt_index) pairs
    if agent_results and len(results) < expected_count:
        existing_keys = set()
        for r in results:
            existing_keys.add((r["episode_id"], r["gt_index"]))
        n_padded = 0
        for ref in agent_results:
            key = (ref["episode_id"], ref["gt_index"])
            if key not in existing_keys:
                results.append(_make_worst_case_result(
                    ref["episode_id"], ref["gt_index"],
                    task_type, ref.get("params", {}),
                ))
                n_padded += 1
        if n_padded > 0:
            print(f"[{model_name}][{subtask_name}] padded {n_padded} worst-case "
                  f"results (total: {len(results)}/{expected_count})")

    summary = aggregate_results(results)
    print(f"[{model_name}][{subtask_name}] done results={len(results)}")

    from evaluation.editability_utils.subtasks.common import save_subtask_outputs
    save_subtask_outputs(
        output_dir=output_dir,
        model=model_name,
        subtask_name=subtask_name,
        capacity=capacity,
        sampled_count=len(sampled),
        results=results,
        summary=summary,
    )

    return {
        "capacity": capacity,
        "sampled_count": len(sampled),
        "results": results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Subtask definitions (param grids matching atomic subtasks)
# ---------------------------------------------------------------------------

def _get_subtask_config():
    subtasks = []

    subtasks.append({
        "name": "delete",
        "param_grid": [{}],
        "seed_offset": 0,
        "include_iou": False,
        "roi_mode": "source",
        "roi_dilation_ratio": 0.0,
    })

    trans_params = []
    for frac in (0.7, 0.85, 0.95):
        for sx in (-1, 1):
            for sy in (-1, 1):
                trans_params.append({
                    "aggressive": True,
                    "aggressive_fraction": float(frac),
                    "x_sign": int(sx),
                    "y_sign": int(sy),
                })
    subtasks.append({
        "name": "transition",
        "param_grid": trans_params,
        "seed_offset": 1,
        "include_iou": True,
        "roi_mode": "transition_dual",
        "roi_dilation_ratio": 0.0,
    })

    rot_params = [{"angle_deg": float(a)} for a in (-75.0, -55.0, -35.0, 35.0, 55.0, 75.0)]
    subtasks.append({
        "name": "rotation",
        "param_grid": rot_params,
        "seed_offset": 2,
        "include_iou": True,
        "roi_mode": "target",
        "roi_dilation_ratio": 0.0,
    })

    opacity_params = [{"min_alpha_delta": int(v)} for v in (110, 140, 180)]
    subtasks.append({
        "name": "opacity",
        "param_grid": opacity_params,
        "seed_offset": 3,
        "include_iou": False,
        "roi_mode": "source",
        "roi_dilation_ratio": 0.0,
    })

    subtasks.append({
        "name": "z_order",
        "param_grid": [{"direction": "front"}, {"direction": "back"}],
        "seed_offset": 4,
        "include_iou": False,
        "roi_mode": "source",
        "roi_dilation_ratio": 0.0,
    })

    recolor_params = []
    for h in (-40.0, -20.0, 20.0, 40.0):
        for s in (0.8, 1.2):
            recolor_params.append({"hue_shift_deg": float(h), "sat_mul": float(s), "val_mul": 1.0})
    subtasks.append({
        "name": "recolor",
        "param_grid": recolor_params,
        "seed_offset": 5,
        "include_iou": False,
        "roi_mode": "source",
        "roi_dilation_ratio": 0.0,
    })

    return subtasks


# ---------------------------------------------------------------------------
# Copy existing results
# ---------------------------------------------------------------------------

def copy_existing_editability_results(
    original_dir: Path,
    output_dir: Path,
    models: Sequence[str] = ("agent", "qwen"),
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for model in models:
        src_dir = original_dir / model
        dst_dir = output_dir / model
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            file_count = sum(1 for _ in dst_dir.rglob("*.json"))
            counts[model] = file_count
        else:
            counts[model] = 0

    for fname in [
        "atomic_selected_subset.json",
        "atomic_comparison_qwen_vs_agent.json",
    ]:
        src = original_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # Only copy model-specific overviews for models that aren't being re-evaluated
    for model in models:
        fname = f"atomic_{model}_overview.json"
        src = original_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    return counts


# ---------------------------------------------------------------------------
# Live monitoring helpers (5-model running averages)
# ---------------------------------------------------------------------------

ALL_MODELS = ["multi_tools", "layered", "qwen", "sparse_verif", "simple_verif", "agent"]

PRIORITY_METRICS = [
    "l1", "l2", "psnr", "ssim", "lpips", "dino",
    "full_l1", "full_l2", "full_psnr", "full_ssim", "full_lpips", "full_dino",
]


def _psnr_inf_fallback() -> float:
    raw = os.environ.get("EDITABILITY_PSNR_INF_FALLBACK", "100.0")
    try:
        v = float(raw)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return 100.0


def _safe_read_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        with open(path) as f:
            x = json.load(f)
        if isinstance(x, list):
            return [r for r in x if isinstance(r, dict)]
    except Exception:
        pass
    return []


def _row_key(row: Dict[str, Any], subtask_name: str) -> str:
    params = row.get("params", {})
    if not isinstance(params, dict):
        params = {}
    params_key = json.dumps(params, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return (
        f"{str(row.get('episode_id', ''))}"
        f"::gt{int(row.get('gt_index', -1))}"
        f"::task={str(row.get('task_type', subtask_name))}"
        f"::params={params_key}"
    )


def _compute_mean(rows: List[Dict[str, Any]], keys: Set[str], metric: str) -> Optional[float]:
    """Compute mean of a metric across rows whose key is in `keys`."""
    vals: List[float] = []
    inf_count = 0
    is_psnr = "psnr" in str(metric).lower()
    for r in rows:
        rk = r.get("_key")
        if rk not in keys:
            continue
        mm = r.get("metrics", {})
        if not isinstance(mm, dict):
            continue
        v = mm.get(metric)
        if isinstance(v, (int, float)) and (v == v):
            fv = float(v)
            if is_psnr:
                if math.isfinite(fv):
                    vals.append(fv)
                elif fv > 0:
                    inf_count += 1
            elif math.isfinite(fv):
                vals.append(fv)
    if is_psnr and inf_count > 0:
        rep = max(vals) if vals else _psnr_inf_fallback()
        vals.extend([float(rep)] * int(inf_count))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _five_model_running_msg(
    model_rows: Dict[str, List[Dict[str, Any]]],
    subtask_name: str,
) -> Tuple[int, str]:
    """Compute running averages on the intersection of keys across ALL models that have data."""
    # Build key-sets per model
    model_keys: Dict[str, Set[str]] = {}
    for m, rows in model_rows.items():
        ks = set()
        for r in rows:
            if "_key" not in r:
                r["_key"] = _row_key(r, subtask_name)
            ks.add(r["_key"])
        model_keys[m] = ks

    # Intersection of keys from ALL models with data
    non_empty = [ks for ks in model_keys.values() if ks]
    if len(non_empty) < 2:
        total_rows = {m: len(rows) for m, rows in model_rows.items() if rows}
        return 0, (
            f"[live-5model][atomic_{subtask_name}] common=0 "
            + " ".join(f"{m}={n}" for m, n in total_rows.items())
        )

    common_keys = set.intersection(*non_empty)
    n_common = len(common_keys)
    if n_common <= 0:
        total_rows = {m: len(rows) for m, rows in model_rows.items() if rows}
        return 0, (
            f"[live-5model][atomic_{subtask_name}] common=0 "
            + " ".join(f"{m}={n}" for m, n in total_rows.items())
        )

    # Compute means per model on common keys (top-6 priority metrics)
    tokens_per_model: Dict[str, List[str]] = {m: [] for m in ALL_MODELS}
    for metric in PRIORITY_METRICS[:6]:
        for m in ALL_MODELS:
            rows = model_rows.get(m, [])
            if not rows:
                continue
            val = _compute_mean(rows, common_keys, metric)
            if val is not None:
                tokens_per_model[m].append(f"{metric}={val:.4f}")

    parts = [f"[live-5model][atomic_{subtask_name}] common={n_common}"]
    for m in ALL_MODELS:
        toks = tokens_per_model.get(m, [])
        if toks:
            parts.append(f"{m}({' '.join(toks)})")
        else:
            n = len(model_rows.get(m, []))
            if n > 0:
                parts.append(f"{m}(rows={n})")

    return n_common, " ".join(parts)


# ---------------------------------------------------------------------------
# Parallel subtask runner for 3 baselines
# ---------------------------------------------------------------------------

def _run_single_model_subtask(
    model_name: str,
    subtask_cfg: Dict[str, Any],
    args,
    match_root: Path,
    output_dir: Path,
    task_map: Dict[str, BaselineEpisodeTask],
    cache: BaselineEpisodeCache,
    selected_keys: Set[Tuple[str, int]],
    agent_results: Optional[List[Dict[str, Any]]],
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Wrapper to run a subtask for one model."""
    sub_name = subtask_cfg["name"]
    sub_seed = args.seed + subtask_cfg["seed_offset"]
    if num_workers is None:
        num_workers = max(1, args.num_workers // len(args.models))

    return run_baseline_atomic_subtask(
        task_type=sub_name,
        param_grid=subtask_cfg["param_grid"],
        model_name=model_name,
        match_root=match_root,
        output_dir=output_dir,
        task_map=task_map,
        cache=cache,
        seed=sub_seed,
        subset_keys=selected_keys,
        log_every=args.log_every,
        save_pair_viz=args.save_pair_viz,
        num_workers=num_workers,
        show_tqdm=not args.no_tqdm,
        build_log_every=args.build_log_every,
        include_iou=subtask_cfg.get("include_iou", False),
        roi_mode=subtask_cfg.get("roi_mode", "source"),
        roi_dilation_ratio=subtask_cfg.get("roi_dilation_ratio", 0.08),
        agent_results=agent_results,
    )


def _run_subtask_sequential(
    subtask_cfg: Dict[str, Any],
    args,
    match_root: Path,
    output_dir: Path,
    model_infos: Dict[str, Dict[str, Any]],
    selected_keys: Set[Tuple[str, int]],
    agent_results_by_subtask: Dict[str, List[Dict[str, Any]]],
    log_every: int = 25,
) -> Dict[str, Dict[str, Any]]:
    """Run one subtask across all baseline models SEQUENTIALLY.

    Avoids GPU contention between models and gives each model full num_workers.
    """
    sub_name = subtask_cfg["name"]
    subtask_name = f"atomic_{sub_name}"
    models = list(model_infos.keys())

    print(f"\n[sequential][{subtask_name}] running {len(models)} models: {models}")

    agent_results = agent_results_by_subtask.get(sub_name)
    results: Dict[str, Dict[str, Any]] = {}

    for model_name in models:
        info = model_infos[model_name]
        print(f"\n[sequential][{subtask_name}] >>> {model_name}")
        t0 = time.time()
        results[model_name] = _run_single_model_subtask(
            model_name=model_name,
            subtask_cfg=subtask_cfg,
            args=args,
            match_root=match_root,
            output_dir=output_dir,
            task_map=info["task_map"],
            cache=info["cache"],
            selected_keys=selected_keys,
            agent_results=agent_results,
            num_workers=args.num_workers,  # full workers for the single model
        )
        elapsed = time.time() - t0
        n_results = len(results[model_name].get("results", []))
        print(
            f"[sequential][{subtask_name}] <<< {model_name} done "
            f"results={n_results} elapsed={elapsed:.1f}s"
        )

    # Final 5-model comparison
    ckpt_paths: Dict[str, Path] = {}
    for m in ALL_MODELS:
        ckpt_paths[m] = output_dir / m / f"{subtask_name}_results.json"

    all_rows_final: Dict[str, List[Dict[str, Any]]] = {}
    for m in ALL_MODELS:
        if m in results:
            rows = results[m].get("results", [])
            for r in rows:
                if "_key" not in r:
                    r["_key"] = _row_key(r, sub_name)
            all_rows_final[m] = rows
        else:
            rows = _safe_read_rows(ckpt_paths[m])
            if rows:
                for r in rows:
                    if "_key" not in r:
                        r["_key"] = _row_key(r, sub_name)
                all_rows_final[m] = rows

    n_common, msg = _five_model_running_msg(all_rows_final, sub_name)
    if n_common > 0:
        print(f"[FINAL] {msg}")

    return results


def _run_subtask_parallel(
    subtask_cfg: Dict[str, Any],
    args,
    match_root: Path,
    output_dir: Path,
    model_infos: Dict[str, Dict[str, Any]],
    selected_keys: Set[Tuple[str, int]],
    agent_results_by_subtask: Dict[str, List[Dict[str, Any]]],
    log_every: int = 25,
) -> Dict[str, Dict[str, Any]]:
    """Run one subtask across all baseline models in parallel, monitoring progress."""
    sub_name = subtask_cfg["name"]
    subtask_name = f"atomic_{sub_name}"
    models = list(model_infos.keys())
    n_models = len(models)

    print(f"\n[parallel][{subtask_name}] starting {n_models} models in parallel: {models}")

    # Agent results for this subtask
    agent_results = agent_results_by_subtask.get(sub_name)

    # Checkpoint paths: all 5 models
    ckpt_paths: Dict[str, Path] = {}
    for m in ALL_MODELS:
        ckpt_paths[m] = output_dir / m / f"{subtask_name}_results.json"

    # Pre-load agent/qwen rows (already completed)
    preloaded_rows: Dict[str, List[Dict[str, Any]]] = {}
    for m in ("agent", "qwen"):
        rows = _safe_read_rows(ckpt_paths[m])
        if rows:
            for r in rows:
                r["_key"] = _row_key(r, sub_name)
            preloaded_rows[m] = rows

    done_notified: Dict[str, bool] = {m: False for m in models}
    last_logged = -1
    last_row_counts: Dict[str, int] = {m: -1 for m in models}
    last_live_log_ts = 0.0
    poll_sec = 1.0
    log_step = max(1, int(log_every))

    with ThreadPoolExecutor(max_workers=n_models) as ex:
        futures = {}
        for model_name in models:
            info = model_infos[model_name]
            fut = ex.submit(
                _run_single_model_subtask,
                model_name=model_name,
                subtask_cfg=subtask_cfg,
                args=args,
                match_root=match_root,
                output_dir=output_dir,
                task_map=info["task_map"],
                cache=info["cache"],
                selected_keys=selected_keys,
                agent_results=agent_results,
            )
            futures[model_name] = fut

        while not all(f.done() for f in futures.values()):
            # Check for completions and errors
            for m, fut in futures.items():
                if fut.done():
                    exc = fut.exception()
                    if exc is not None:
                        raise RuntimeError(
                            f"[parallel][{subtask_name}] {m} worker failed"
                        ) from exc
                    if not done_notified[m]:
                        remaining = [m2 for m2 in models if not futures[m2].done()]
                        print(f"[parallel][{subtask_name}] {m} done; waiting: {remaining}")
                        done_notified[m] = True

            # Read checkpoint files for all 5 models
            all_rows: Dict[str, List[Dict[str, Any]]] = {}
            for m in ALL_MODELS:
                if m in preloaded_rows:
                    all_rows[m] = preloaded_rows[m]
                else:
                    rows = _safe_read_rows(ckpt_paths[m])
                    if rows:
                        for r in rows:
                            if "_key" not in r:
                                r["_key"] = _row_key(r, sub_name)
                        all_rows[m] = rows

            n_common, msg = _five_model_running_msg(all_rows, sub_name)

            if n_common > 0 and n_common != last_logged and (
                last_logged < 0 or n_common >= last_logged + log_step
            ):
                print(msg)
                last_logged = n_common
            elif n_common <= 0:
                now = time.time()
                row_counts = {m: len(all_rows.get(m, [])) for m in models}
                if (
                    row_counts != last_row_counts
                    or (now - last_live_log_ts) >= 20.0
                ):
                    counts_str = " ".join(f"{m}={row_counts[m]}" for m in models)
                    preloaded_str = " ".join(
                        f"{m}={len(preloaded_rows[m])}"
                        for m in ("agent", "qwen") if m in preloaded_rows
                    )
                    print(
                        f"[live-5model][atomic_{sub_name}] common=0 "
                        f"baselines({counts_str}) existing({preloaded_str})"
                    )
                    last_row_counts = row_counts.copy()
                    last_live_log_ts = now

            time.sleep(poll_sec)

        # Collect results
        results: Dict[str, Dict[str, Any]] = {}
        for m, fut in futures.items():
            results[m] = fut.result()

    # Final running message
    all_rows_final: Dict[str, List[Dict[str, Any]]] = {}
    for m in ALL_MODELS:
        if m in results:
            rows = results[m].get("results", [])
            for r in rows:
                if "_key" not in r:
                    r["_key"] = _row_key(r, sub_name)
            all_rows_final[m] = rows
        elif m in preloaded_rows:
            all_rows_final[m] = preloaded_rows[m]
        else:
            rows = _safe_read_rows(ckpt_paths[m])
            if rows:
                for r in rows:
                    if "_key" not in r:
                        r["_key"] = _row_key(r, sub_name)
                all_rows_final[m] = rows

    n_common, msg = _five_model_running_msg(all_rows_final, sub_name)
    if n_common > 0:
        print(f"[FINAL] {msg}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Editability evaluation for baseline models (parallel)"
    )
    parser.add_argument("--figma-data", type=str, required=True,
                        help="Path to the downloaded figma_data dataset directory.")
    parser.add_argument("--match-root", type=str, required=True,
                        help="Directory of pre-computed matches produced by "
                             "before_eval_editability_precompute_matches.py "
                             "(contains {model}/episodes/{episode_id}.json).")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Baseline models to evaluate.")
    add_baseline_dir_args(parser)
    parser.add_argument("--original-run-dir", type=str, required=True,
                        help="Path to original agent/qwen editability results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output root directory. Required unless --resume-dir is set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--per-episode-elements", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-matching-cost", type=float, default=0.9)
    parser.add_argument("--min-matching-iou", type=float, default=-1.0)
    parser.add_argument("--min-cross-model-iou", type=float, default=-1.0)
    parser.add_argument("--min-gt-opaque-pixels", type=int, default=400)
    parser.add_argument("--opaque-alpha-threshold", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--cache-episodes", type=int, default=8)
    parser.add_argument("--max-episode-loaders", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--build-log-every", type=int, default=50)
    parser.add_argument("--save-pair-viz", action="store_true", default=False)
    parser.add_argument("--no-save-triplet-viz", action="store_true", default=False)
    parser.add_argument("--no-copy-existing", action="store_true", default=False,
                        help="Skip copying agent/qwen results from original-run-dir")
    parser.add_argument("--metric-bg-mode", type=str, default="best_of_black_white",
                        choices=["premultiplied", "best_of_black_white"])
    parser.add_argument("--no-tqdm", action="store_true", default=False)
    parser.add_argument("--sequential", action="store_true", default=False,
                        help="Run baseline models sequentially (avoids GPU contention)")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Write checkpoint every N results (default: 25, set 1 for every result)")
    parser.add_argument("--disable-evalfigma-postprocess", action="store_true",
                        help="(Deprecated) Postprocess is always disabled to avoid double-application.")
    parser.add_argument("--disable-evalfigma-text-refinement", action="store_true",
                        help="(Deprecated) Text refinement is always disabled at eval-time.")
    parser.add_argument("--resume-dir", type=str, default=None)

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

    match_root = Path(args.match_root)
    original_run_dir = Path(args.original_run_dir)
    figma_data_dir = Path(args.figma_data)

    # Set environment variables
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ["EDITABILITY_CACHE_EPISODES"] = str(max(1, args.cache_episodes))
    os.environ["EDITABILITY_MAX_EP_LOADERS"] = str(max(1, args.max_episode_loaders))
    os.environ["EDITABILITY_MIN_GT_OPAQUE_PIXELS"] = str(max(0, args.min_gt_opaque_pixels))
    os.environ["EDITABILITY_OPAQUE_ALPHA_THRESHOLD"] = str(max(0, min(255, args.opaque_alpha_threshold)))
    os.environ["EDITABILITY_STRICT_GT_OPAQUE_CHECK"] = "0"
    os.environ["EDITABILITY_MAX_MATCHING_COST"] = str(float(args.max_matching_cost))
    os.environ["EDITABILITY_MIN_MATCHING_IOU"] = str(float(args.min_matching_iou))
    # Postprocess (clean_alpha_noise + text_refinement) is already applied once
    # during element extraction (extract_agent_elements / extract_qwen_elements_cca).
    # Disable eval-time postprocess to prevent double-application.
    os.environ["EDITABILITY_USE_EVALFIGMA_POSTPROC"] = "0"
    os.environ["EDITABILITY_USE_EVALFIGMA_TEXT_REFINEMENT"] = "0"
    os.environ["EDITABILITY_METRIC_BG_MODE"] = args.metric_bg_mode.strip().lower()
    os.environ["EDITABILITY_CHECKPOINT_INTERVAL"] = str(max(1, args.checkpoint_every))

    mode_str = "SEQUENTIAL" if args.sequential else "PARALLEL"
    print("=" * 80)
    print(f"BASELINE EDITABILITY EVALUATION ({mode_str})")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")
    print(f"Models: {args.models}")
    print(f"Match root: {match_root}")
    print(f"Original run: {original_run_dir}")
    if args.sequential:
        print(f"Workers: {args.num_workers} (full per model, sequential)")
    else:
        print(f"Workers: {args.num_workers} (split across {len(args.models)} models)")
    print(f"Checkpoint every: {args.checkpoint_every} results")
    print("=" * 80)

    # 1. Load selected keys from original run
    subset_path = original_run_dir / "atomic_selected_subset.json"
    if not subset_path.exists():
        print(f"[ERROR] Cannot find {subset_path}")
        sys.exit(1)

    with open(subset_path) as f:
        original_subset = json.load(f)

    selected_keys: Set[Tuple[str, int]] = set()
    for item in original_subset.get("keys", []):
        eid = item["episode_id"]
        gt_idx = int(item["gt_index"])
        selected_keys.add((eid, gt_idx))

    selected_episodes = {eid for eid, _ in selected_keys}
    print(f"\nLoaded {len(selected_keys)} selected (episode_id, gt_index) pairs "
          f"across {len(selected_episodes)} episodes")

    # 2. Copy existing agent/qwen results (only for models NOT being re-evaluated)
    from evaluation.editability_utils.common_utils import save_json
    models_to_copy = [m for m in ("agent", "qwen") if m not in args.models]
    if models_to_copy and not args.no_copy_existing:
        print(f"\nCopying existing results for: {models_to_copy}")
        copy_counts = copy_existing_editability_results(
            original_run_dir, output_dir, models=models_to_copy
        )
        for m, c in copy_counts.items():
            print(f"  {m}: {c} files copied")
    else:
        print("\nAll 5 models will be evaluated fresh — no results copied.")

    save_json(output_dir / "atomic_selected_subset.json", original_subset)

    # 3. Collect GT episodes
    gt_map = collect_gt_episodes(figma_data_dir)
    print(f"\nGT episodes: {len(gt_map)}")

    # 4. Pre-load agent results for reference and task-count alignment
    # Only load if agent is not being re-evaluated (backward compat)
    subtask_configs = _get_subtask_config()
    agent_results_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    if "agent" not in args.models:
        agent_dir = output_dir / "agent"
        for subtask_cfg in subtask_configs:
            sub_name = subtask_cfg["name"]
            agent_path = agent_dir / f"atomic_{sub_name}_results.json"
            if agent_path.exists():
                try:
                    with open(agent_path) as _f:
                        agent_results_by_subtask[sub_name] = json.load(_f)
                except Exception:
                    agent_results_by_subtask[sub_name] = []
        if agent_results_by_subtask:
            print(f"Loaded agent reference results: "
                  + ", ".join(f"{k}={len(v)}" for k, v in agent_results_by_subtask.items()))

    # 5. Build per-model task maps and caches
    model_infos: Dict[str, Dict[str, Any]] = {}
    for model_name in args.models:
        # Agent/Qwen may span multiple experiment directories
        multi_dirs = _resolve_multi_dirs(args, model_name)
        if multi_dirs is not None:
            model_map = scan_model_episodes_multi(model_name, multi_dirs)
        else:
            base_dir = get_model_dir(args, model_name)
            model_map = scan_model_episodes(model_name, base_dir)
        model_format = MODEL_CONFIGS[model_name]["format"]

        common = get_common_episodes(gt_map, model_map)

        filtered_common = {
            eid: info for eid, info in common.items()
            if eid in selected_episodes
        }
        print(f"  {model_name}: scanned={len(model_map)} common={len(common)} "
              f"selected={len(filtered_common)}")

        if not filtered_common:
            print(f"  [WARNING] No episodes for {model_name}, skipping")
            continue

        # Verify matches exist
        match_dir = match_root / model_name / "episodes"
        if not match_dir.exists():
            print(f"  [ERROR] Match directory not found: {match_dir}")
            continue
        match_count = len(list(match_dir.glob("*.json")))
        print(f"  {model_name}: match_files={match_count}")

        task_map: Dict[str, BaselineEpisodeTask] = {}
        for eid, info in filtered_common.items():
            task_map[eid] = BaselineEpisodeTask(
                episode_id=eid,
                split_name=info["split_name"],
                split_dir=info["split_dir"],
                gt_json_path=info["gt_json_path"],
                pred_dir=info["model_dir"],
                model_format=model_format,
            )

        cache = BaselineEpisodeCache(
            task_map,
            max_items=args.cache_episodes,
            max_loaders=args.max_episode_loaders,
        )

        model_infos[model_name] = {
            "task_map": task_map,
            "cache": cache,
        }

    if not model_infos:
        print("[ERROR] No models to evaluate")
        sys.exit(1)

    # 6. Run subtasks
    run_fn = _run_subtask_sequential if args.sequential else _run_subtask_parallel
    all_model_results: Dict[str, Dict[str, Any]] = {}

    for subtask_cfg in subtask_configs:
        sub_name = subtask_cfg["name"]

        subtask_results = run_fn(
            subtask_cfg=subtask_cfg,
            args=args,
            match_root=match_root,
            output_dir=output_dir,
            model_infos=model_infos,
            selected_keys=selected_keys,
            agent_results_by_subtask=agent_results_by_subtask,
            log_every=args.log_every,
        )

        for model_name, payload in subtask_results.items():
            if model_name not in all_model_results:
                all_model_results[model_name] = {}
            all_model_results[model_name][sub_name] = payload

    # 7. Save per-model overviews
    for model_name, payloads in all_model_results.items():
        save_json(
            output_dir / f"atomic_{model_name}_overview.json",
            payloads,
        )

    # 8. Generate cross-model comparison
    print(f"\n{'='*80}")
    print("CROSS-MODEL COMPARISON (ALL 5 MODELS)")
    print(f"{'='*80}")

    # Load agent/qwen overviews
    for existing_model in ("agent", "qwen"):
        overview_path = output_dir / f"atomic_{existing_model}_overview.json"
        if overview_path.exists() and existing_model not in all_model_results:
            try:
                with open(overview_path) as f:
                    all_model_results[existing_model] = json.load(f)
            except Exception:
                pass

    comparison: Dict[str, Any] = {"models": list(all_model_results.keys())}
    subtask_names = [cfg["name"] for cfg in subtask_configs]

    for sub_name in subtask_names:
        sub_comparison = {}
        for m in sorted(all_model_results.keys()):
            m_data = all_model_results[m].get(sub_name, {})
            m_summary = m_data.get("summary", {})
            sub_comparison[m] = m_summary
        comparison[sub_name] = sub_comparison

    save_json(output_dir / "atomic_comparison_all_models.json", comparison)

    # Print final summary table
    base_metrics = ["l1", "l2", "psnr", "ssim", "lpips", "dino"]
    roi_prefixes = [("union", ""), ("gt", "gt_"), ("pred", "pred_")]
    display_models = [m for m in ALL_MODELS if m in all_model_results]

    for sub_name in subtask_names:
        print(f"\n  [{sub_name}]")
        header = f"    {'ROI / Metric':<18}"
        for m in display_models:
            header += f" | {m:<15}"
        print(header)
        print("    " + "-" * (18 + 18 * len(display_models)))

        for roi_label, prefix in roi_prefixes:
            print(f"    [{roi_label}]")
            for mk in base_metrics:
                metric_key = f"{prefix}{mk}" if prefix else mk
                row = f"      {mk:<16}"
                for m in display_models:
                    sub_data = all_model_results[m].get(sub_name, {})
                    summary = sub_data.get("summary", {})
                    val = summary.get(metric_key)
                    if val is not None:
                        row += f" | {float(val):>13.4f} "
                    else:
                        row += f" | {'N/A':>13} "
                print(row)

    print(f"\nResults saved to {output_dir}")
    print(f"Completed at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
