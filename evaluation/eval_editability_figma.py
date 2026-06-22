#!/usr/bin/env python3
"""Re-evaluate editability with AGGRESSIVE edit parameters.

Runs ALL 7 models × 6 subtasks with aggressive edit params to ensure
the edit magnitude is large enough that the no-edit floor is clearly
worse than any model's performance.

Uses the same subset_keys as the original run (153816) for consistency.
Saves per-episode per-model metrics for later visualization.

Usage:
    # Replace <GPU_ID> with one of your own GPU ids (e.g. 0).
    # First 3 subtasks on one GPU
    CUDA_VISIBLE_DEVICES=<GPU_ID> conda run --no-capture-output -n agent_qwen_layerd \
        python -u scripts/eval_editability_aggressive.py \
        --subtasks delete opacity recolor --num-workers 16

    # Last 3 subtasks on another GPU
    CUDA_VISIBLE_DEVICES=<GPU_ID> conda run --no-capture-output -n agent_qwen_layerd \
        python -u scripts/eval_editability_aggressive.py \
        --subtasks rotation transition z_order --num-workers 16

Input directories are configured via the REDESIGN_* environment variables documented
in the Configuration section below; point REDESIGN_FIGMA_DATA at the downloaded
``figma_data`` dataset and the agent/qwen/baseline output dirs at the inference
runner outputs (e.g. ``python -m REDESIGN.run_agent_figma --data_dir figma_data \
--output_dir <AGENT_OUTPUT_DIR>``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Semaphore
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.baseline_model_configs import (
    MODEL_CONFIGS,
    collect_gt_episodes,
    get_common_episodes,
    scan_model_episodes,
    scan_model_episodes_multi,
)
from evaluation.eval_editability_baselines import (
    BaselineEpisodeTask,
    BaselineEpisodeCache,
    run_baseline_atomic_subtask,
)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# All paths below can be overridden via environment variables so the script is
# portable across machines. Defaults assume the recommended deployment layout:
#   figma_data/                      <- merged 909-episode dataset (HuggingFace)
#   outputs/<model>_agent/           <- inference outputs (run_agent_figma.py / baselines)
#   outputs/editability_matches/     <- precompute output (before_eval_editability_precompute_matches.py)
#   outputs/eval_editability_figma/  <- this script's output
# See evaluation/README.md for the full editability-evaluation pipeline.
# ---------------------------------------------------------------------------

def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default

OUTPUTS_BASE = _env_path("REDESIGN_OUTPUTS", BASE_DIR / "outputs")

FIGMA_DATA_DIR = _env_path("REDESIGN_FIGMA_DATA", BASE_DIR / "figma_data")
MATCH_ROOT = _env_path("REDESIGN_MATCH_ROOT", OUTPUTS_BASE / "editability_matches")

# Subset-selection file: ships with the repo for paper-reproducible episode
# selection; override REDESIGN_SUBSET_FILE to use your own.
SUBSET_FILE = _env_path("REDESIGN_SUBSET_FILE", BASE_DIR / "evaluation" / "assets" / "atomic_selected_subset.json")

# Output directory
OUTPUT_DIR = _env_path("REDESIGN_EDIT_OUTPUT", OUTPUTS_BASE / "eval_editability_figma")

# exp_pairs (agent_dir:qwen_dir:gt_prefix). With the merged dataset, GT discovery
# ignores gt_prefix (all 909 episodes are collected); the dirs are still used to
# locate agent/qwen inference outputs. Override via REDESIGN_EXP_PAIRS (space-sep).
_default_exp_pairs = [
    f"{OUTPUTS_BASE / 'figma_agent'}:{OUTPUTS_BASE / 'figma_qwen'}:merged",
]
EXP_PAIRS = os.environ.get("REDESIGN_EXP_PAIRS", " ".join(_default_exp_pairs)).split()

# All 7 models to evaluate
ALL_MODELS = [
    "agent", "qwen", "layered", "multi_tools",
    "sparse_verif", "simple_verif", "vtracer",
]

# Model directory mapping (single-dir models). Override with REDESIGN_<MODEL>_DIR.
MODEL_DIRS = {
    "layered":      _env_path("REDESIGN_LAYERED_DIR",      OUTPUTS_BASE / "baseline_layered"),
    "multi_tools":  _env_path("REDESIGN_MULTI_TOOLS_DIR",  OUTPUTS_BASE / "baseline_multi_tools"),
    "sparse_verif": _env_path("REDESIGN_SPARSE_VERIF_DIR", OUTPUTS_BASE / "baseline_sparse_verif"),
    "simple_verif": _env_path("REDESIGN_SIMPLE_VERIF_DIR", OUTPUTS_BASE / "baseline_simple_verif"),
    "vtracer":      _env_path("REDESIGN_VTRACER_DIR",      OUTPUTS_BASE / "baseline_vtracer"),
}

# Multi-dir models (agent, qwen) — first field of each exp-pair is the agent dir,
# second is the qwen dir.
AGENT_DIRS = [Path(p.split(":")[0]) for p in EXP_PAIRS]
QWEN_DIRS = [Path(p.split(":")[1]) for p in EXP_PAIRS]

SUBTASK_NAMES = ["delete", "opacity", "recolor", "rotation", "transition", "z_order"]

SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# Aggressive Parameter Grids
# ═══════════════════════════════════════════════════════════════════════════

def _get_aggressive_subtask_configs() -> List[Dict[str, Any]]:
    """Return subtask configs with AGGRESSIVE edit parameters."""
    configs = []

    # 1. DELETE — no params needed, already removes element entirely
    configs.append({
        "name": "delete",
        "param_grid": [{}],
        "seed_offset": 0,
        "include_iou": False,
        "roi_mode": "source",
        "roi_dilation_ratio": 0.0,
    })

    # 2. OPACITY — near-full to full transparency
    #    Original: min_alpha_delta = [110, 140, 180]
    #    Aggressive: push to near-total/total transparency
    opacity_params = [{"min_alpha_delta": int(v)} for v in (220, 245, 255)]
    configs.append({
        "name": "opacity",
        "param_grid": opacity_params,
        "seed_offset": 3,
        "include_iou": False,
        "roi_mode": "source",
        "roi_dilation_ratio": 0.0,
    })

    # 3. RECOLOR — extreme color changes
    #    Original: hue=[-40,-20,20,40], sat=[0.8,1.2], val=1.0
    #    Aggressive: huge hue shift + desaturate/super-saturate + darken/brighten
    #    sat=0.0 → complete desaturation (gray), val=0.3 → very dark
    recolor_params = []
    for h in (-120.0, 120.0):
        for s in (0.0, 2.5):
            for v in (0.4, 1.8):
                recolor_params.append({
                    "hue_shift_deg": float(h),
                    "sat_mul": float(s),
                    "val_mul": float(v),
                })
    configs.append({
        "name": "recolor",
        "param_grid": recolor_params,
        "seed_offset": 5,
        "include_iou": False,
        "roi_mode": "source",
        "roi_dilation_ratio": 0.0,
    })

    # 4. ROTATION — extreme angles
    #    Original: angles = [-75, -55, -35, 35, 55, 75]
    #    Aggressive: near-180° rotations for dramatic change
    rot_params = [{"angle_deg": float(a)} for a in (-150.0, -120.0, -90.0, 90.0, 120.0, 150.0)]
    configs.append({
        "name": "rotation",
        "param_grid": rot_params,
        "seed_offset": 2,
        "include_iou": True,
        "roi_mode": "target",
        "roi_dilation_ratio": 0.0,
    })

    # 5. TRANSITION — push elements further off-canvas
    #    Original: fractions = [0.7, 0.85, 0.95]
    #    Aggressive: up to 100% displacement
    trans_params = []
    for frac in (0.85, 0.95, 1.0):
        for sx in (-1, 1):
            for sy in (-1, 1):
                trans_params.append({
                    "aggressive": True,
                    "aggressive_fraction": float(frac),
                    "x_sign": int(sx),
                    "y_sign": int(sy),
                })
    configs.append({
        "name": "transition",
        "param_grid": trans_params,
        "seed_offset": 1,
        "include_iou": True,
        "roi_mode": "transition_dual",
        "roi_dilation_ratio": 0.0,
    })

    # 6. Z_ORDER — already binary (front/back), no change needed
    configs.append({
        "name": "z_order",
        "param_grid": [{"direction": "front"}, {"direction": "back"}],
        "seed_offset": 4,
        "include_iou": False,
        "roi_mode": "source",
        "roi_dilation_ratio": 0.0,
    })

    return configs


# ═══════════════════════════════════════════════════════════════════════════
# Model Setup
# ═══════════════════════════════════════════════════════════════════════════

def setup_model(
    model_name: str,
    gt_map: Dict[str, Any],
    selected_episodes: Set[str],
    cache_episodes: int = 8,
    max_loaders: int = 2,
) -> Optional[Dict[str, Any]]:
    """Set up task_map and cache for a model."""
    model_format = MODEL_CONFIGS[model_name]["format"]

    # Scan model episodes
    if model_name == "agent":
        model_map = scan_model_episodes_multi("agent", AGENT_DIRS)
    elif model_name == "qwen":
        model_map = scan_model_episodes_multi("qwen", QWEN_DIRS)
    else:
        base_dir = MODEL_DIRS.get(model_name)
        if base_dir is None or not base_dir.exists():
            print(f"  [WARNING] No directory for {model_name}, skipping")
            return None
        model_map = scan_model_episodes(model_name, base_dir)

    common = get_common_episodes(gt_map, model_map)
    filtered = {eid: info for eid, info in common.items() if eid in selected_episodes}

    # Check matches exist
    match_dir = MATCH_ROOT / model_name / "episodes"
    if not match_dir.exists():
        print(f"  [ERROR] No match dir: {match_dir}")
        return None
    match_count = len(list(match_dir.glob("*.json")))

    print(f"  {model_name}: scanned={len(model_map)} common={len(common)} "
          f"selected={len(filtered)} matches={match_count}", flush=True)

    if not filtered:
        print(f"  [WARNING] No episodes for {model_name}", flush=True)
        return None

    task_map: Dict[str, BaselineEpisodeTask] = {}
    for eid, info in filtered.items():
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
        max_items=cache_episodes,
        max_loaders=max_loaders,
    )

    return {"task_map": task_map, "cache": cache}


# ═══════════════════════════════════════════════════════════════════════════
# Per-Episode Metric Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_per_episode_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Extract per-episode metrics from results.

    Returns: {episode_id -> {gt_index -> {metric_key: value}}}
    """
    per_ep: Dict[str, Dict[int, Dict[str, float]]] = {}
    for r in results:
        eid = r["episode_id"]
        gt_idx = r["gt_index"]
        metrics = r.get("metrics", {})
        if eid not in per_ep:
            per_ep[eid] = {}
        # Keep numeric metrics only
        clean_metrics = {}
        for k, v in metrics.items():
            try:
                fv = float(v)
                if not (fv != fv):  # skip NaN
                    clean_metrics[k] = fv
            except (TypeError, ValueError):
                pass
        per_ep[eid][gt_idx] = clean_metrics
    return per_ep


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Aggressive editability re-evaluation")
    parser.add_argument("--subtasks", type=str, nargs="+", default=None,
                        choices=SUBTASK_NAMES,
                        help="Subtasks to evaluate (default: all)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models to evaluate (default: all 7)")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--cache-episodes", type=int, default=50)
    parser.add_argument("--max-loaders", type=int, default=6)
    parser.add_argument("--model-first", action="store_true", default=True,
                        help="Iterate model-first for better cache utilization (default)")
    parser.add_argument("--subtask-first", action="store_true", default=False,
                        help="Iterate subtask-first instead of model-first")
    parser.add_argument("--sequential", action="store_true",
                        help="Run models sequentially (avoids GPU contention)")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--build-log-every", type=int, default=100)
    args = parser.parse_args()

    subtasks_to_run = args.subtasks or SUBTASK_NAMES
    models_to_run = args.models or ALL_MODELS

    # Set env variables
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ["EDITABILITY_CACHE_EPISODES"] = str(max(1, args.cache_episodes))
    os.environ["EDITABILITY_MAX_EP_LOADERS"] = str(max(1, args.max_loaders))
    os.environ["EDITABILITY_MIN_GT_OPAQUE_PIXELS"] = "400"
    os.environ["EDITABILITY_OPAQUE_ALPHA_THRESHOLD"] = "250"
    os.environ["EDITABILITY_STRICT_GT_OPAQUE_CHECK"] = "0"
    os.environ["EDITABILITY_MAX_MATCHING_COST"] = "0.9"
    os.environ["EDITABILITY_MIN_MATCHING_IOU"] = "-1.0"
    os.environ["EDITABILITY_USE_EVALFIGMA_POSTPROC"] = "0"
    os.environ["EDITABILITY_USE_EVALFIGMA_TEXT_REFINEMENT"] = "0"
    os.environ["EDITABILITY_METRIC_BG_MODE"] = "premultiplied"
    os.environ["EDITABILITY_CHECKPOINT_INTERVAL"] = "25"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("AGGRESSIVE EDITABILITY RE-EVALUATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print(f"Models: {models_to_run}", flush=True)
    print(f"Subtasks: {subtasks_to_run}", flush=True)
    print(f"Workers: {args.num_workers}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load selected keys from original run
    subset_path = SUBSET_FILE
    if not subset_path.exists():
        print(f"[ERROR] {subset_path} not found")
        sys.exit(1)

    with open(subset_path) as f:
        original_subset = json.load(f)

    selected_keys: Set[Tuple[str, int]] = set()
    for item in original_subset.get("keys", []):
        selected_keys.add((item["episode_id"], int(item["gt_index"])))
    selected_episodes = {eid for eid, _ in selected_keys}
    print(f"\nSelected: {len(selected_keys)} pairs across {len(selected_episodes)} episodes",
          flush=True)

    # Save subset to output
    from evaluation.editability_utils.common_utils import save_json
    save_json(OUTPUT_DIR / "atomic_selected_subset.json", original_subset)

    # 2. Collect GT episodes
    gt_map = collect_gt_episodes(FIGMA_DATA_DIR, EXP_PAIRS)
    print(f"GT episodes: {len(gt_map)}", flush=True)

    # 3. Set up models
    print(f"\nSetting up {len(models_to_run)} models...", flush=True)
    model_infos: Dict[str, Dict[str, Any]] = {}
    for model_name in models_to_run:
        info = setup_model(
            model_name, gt_map, selected_episodes,
            cache_episodes=args.cache_episodes,
            max_loaders=args.max_loaders,
        )
        if info is not None:
            model_infos[model_name] = info

    if not model_infos:
        print("[ERROR] No models available")
        sys.exit(1)

    print(f"\nReady: {list(model_infos.keys())}", flush=True)

    # 4. Get subtask configs
    all_configs = _get_aggressive_subtask_configs()
    config_map = {c["name"]: c for c in all_configs}

    # 5. Run evaluations
    # Per-episode metrics collector: {subtask -> {model -> {eid -> {gt_idx -> metrics}}}}
    all_per_episode: Dict[str, Dict[str, Any]] = {}

    # Determine iteration order: model-first (default) or subtask-first
    iterate_model_first = not args.subtask_first

    if iterate_model_first:
        # MODEL-FIRST: For each model, run all subtasks.
        # This keeps the episode cache warm across subtasks for the same model.
        print(f"\nIteration order: MODEL-FIRST (better cache utilization)", flush=True)

        # Always run agent first for padding reference
        model_order = list(model_infos.keys())
        if "agent" in model_order:
            model_order.remove("agent")
            model_order.insert(0, "agent")

        # Pre-collect agent results per subtask for padding
        agent_results_by_subtask: Dict[str, List[Dict[str, Any]]] = {}

        for model_name in model_order:
            info = model_infos[model_name]
            model_t0 = time.time()
            print(f"\n{'='*60}", flush=True)
            print(f"MODEL: {model_name} ({len(subtasks_to_run)} subtasks)", flush=True)
            print(f"{'='*60}", flush=True)

            for sub_name in subtasks_to_run:
                cfg = config_map.get(sub_name)
                if cfg is None:
                    continue

                subtask_label = f"atomic_{sub_name}"
                sub_seed = SEED + cfg["seed_offset"]
                t0 = time.time()

                print(f"\n>>> {model_name} × {subtask_label}", flush=True)

                # Use agent results for padding (if not agent itself)
                ref_results = None
                if model_name != "agent":
                    ref_results = agent_results_by_subtask.get(sub_name)
                    if ref_results is None:
                        # Try loading from checkpoint
                        agent_ckpt = OUTPUT_DIR / "agent" / f"{subtask_label}_results.json"
                        if agent_ckpt.exists():
                            try:
                                with open(agent_ckpt) as f:
                                    ref_results = json.load(f)
                            except Exception:
                                pass

                result = run_baseline_atomic_subtask(
                    task_type=sub_name,
                    param_grid=cfg["param_grid"],
                    model_name=model_name,
                    match_root=MATCH_ROOT,
                    output_dir=OUTPUT_DIR,
                    task_map=info["task_map"],
                    cache=info["cache"],
                    seed=sub_seed,
                    subset_keys=selected_keys,
                    log_every=args.log_every,
                    num_workers=args.num_workers,
                    show_tqdm=True,
                    build_log_every=args.build_log_every,
                    include_iou=cfg.get("include_iou", False),
                    roi_mode=cfg.get("roi_mode", "source"),
                    roi_dilation_ratio=cfg.get("roi_dilation_ratio", 0.08),
                    agent_results=ref_results,
                )

                elapsed = time.time() - t0
                n_results = len(result.get("results", []))

                # Save agent results for padding reference
                if model_name == "agent":
                    agent_results_by_subtask[sub_name] = result.get("results", [])

                # Extract per-episode metrics
                per_ep = extract_per_episode_metrics(result.get("results", []))
                if sub_name not in all_per_episode:
                    all_per_episode[sub_name] = {}
                all_per_episode[sub_name][model_name] = per_ep

                # Print summary
                summary = result.get("summary", {})
                key_metrics = ["l1", "ssim", "lpips", "dino"]
                metric_str = " ".join(
                    f"{k}={summary.get(k, 'N/A'):.4f}" if isinstance(summary.get(k), (int, float))
                    else f"{k}=N/A"
                    for k in key_metrics
                )
                print(f"<<< {model_name} × {subtask_label}: {n_results} results "
                      f"in {elapsed:.1f}s | {metric_str}", flush=True)

            model_elapsed = time.time() - model_t0
            print(f"\n=== {model_name} ALL SUBTASKS DONE in {model_elapsed:.0f}s ===", flush=True)

        # Save per-episode metrics per subtask
        for sub_name in subtasks_to_run:
            if sub_name in all_per_episode:
                save_json(
                    OUTPUT_DIR / f"per_episode_{sub_name}.json",
                    all_per_episode[sub_name],
                )

    else:
        # SUBTASK-FIRST: For each subtask, run all models.
        print(f"\nIteration order: SUBTASK-FIRST", flush=True)

        for sub_name in subtasks_to_run:
            cfg = config_map.get(sub_name)
            if cfg is None:
                print(f"[WARNING] Unknown subtask: {sub_name}", flush=True)
                continue

            subtask_label = f"atomic_{sub_name}"
            sub_seed = SEED + cfg["seed_offset"]

            print(f"\n{'='*60}", flush=True)
            print(f"SUBTASK: {subtask_label} (aggressive params)", flush=True)
            print(f"Param grid size: {len(cfg['param_grid'])}", flush=True)
            print(f"{'='*60}", flush=True)

            per_episode_this_subtask: Dict[str, Any] = {}

            # Load agent results first (for padding reference)
            agent_results_for_ref: Optional[List[Dict[str, Any]]] = None
            if "agent" not in model_infos:
                agent_ckpt = OUTPUT_DIR / "agent" / f"{subtask_label}_results.json"
                if agent_ckpt.exists():
                    try:
                        with open(agent_ckpt) as f:
                            agent_results_for_ref = json.load(f)
                    except Exception:
                        pass

            model_order = list(model_infos.keys())
            if "agent" in model_order:
                model_order.remove("agent")
                model_order.insert(0, "agent")

            for model_name in model_order:
                info = model_infos[model_name]
                t0 = time.time()
                print(f"\n>>> {model_name} × {subtask_label}", flush=True)

                ref_results = agent_results_for_ref
                if model_name == "agent":
                    ref_results = None

                result = run_baseline_atomic_subtask(
                    task_type=sub_name,
                    param_grid=cfg["param_grid"],
                    model_name=model_name,
                    match_root=MATCH_ROOT,
                    output_dir=OUTPUT_DIR,
                    task_map=info["task_map"],
                    cache=info["cache"],
                    seed=sub_seed,
                    subset_keys=selected_keys,
                    log_every=args.log_every,
                    num_workers=args.num_workers,
                    show_tqdm=True,
                    build_log_every=args.build_log_every,
                    include_iou=cfg.get("include_iou", False),
                    roi_mode=cfg.get("roi_mode", "source"),
                    roi_dilation_ratio=cfg.get("roi_dilation_ratio", 0.08),
                    agent_results=ref_results,
                )

                elapsed = time.time() - t0
                n_results = len(result.get("results", []))

                per_ep = extract_per_episode_metrics(result.get("results", []))
                per_episode_this_subtask[model_name] = per_ep

                if model_name == "agent" and agent_results_for_ref is None:
                    agent_results_for_ref = result.get("results", [])

                summary = result.get("summary", {})
                key_metrics = ["l1", "ssim", "lpips", "dino"]
                metric_str = " ".join(
                    f"{k}={summary.get(k, 'N/A'):.4f}" if isinstance(summary.get(k), (int, float))
                    else f"{k}=N/A"
                    for k in key_metrics
                )
                print(f"<<< {model_name} × {subtask_label}: {n_results} results "
                      f"in {elapsed:.1f}s | {metric_str}", flush=True)

            all_per_episode[sub_name] = per_episode_this_subtask
            save_json(
                OUTPUT_DIR / f"per_episode_{sub_name}.json",
                per_episode_this_subtask,
            )

    # Print cross-model comparison for all subtasks
    print(f"\n{'='*60}", flush=True)
    print("CROSS-MODEL COMPARISON (ALL SUBTASKS)", flush=True)
    print(f"{'='*60}", flush=True)
    for sub_name in subtasks_to_run:
        subtask_label = f"atomic_{sub_name}"
        print(f"\n  [{sub_name}]", flush=True)
        header = f"  {'model':<15} | {'count':>6} | {'l1':>8} | {'ssim':>8} | {'lpips':>8} | {'dino':>8}"
        print(header, flush=True)
        print("  " + "-" * 70, flush=True)
        for model_name in model_infos:
            summary_path = OUTPUT_DIR / model_name / f"{subtask_label}_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    s = json.load(f).get("summary", {})
                count = s.get("total", "?")
                row = f"  {model_name:<15} | {count:>6}"
                for mk in ["l1", "ssim", "lpips", "dino"]:
                    v = s.get(mk)
                    row += f" | {float(v):>8.4f}" if v is not None else f" | {'N/A':>8}"
                print(row, flush=True)

    # 6. Save merged overview
    print(f"\n{'='*80}", flush=True)
    print("SAVING MERGED RESULTS", flush=True)
    print(f"{'='*80}", flush=True)

    # Save per-model overviews
    for model_name in model_infos:
        overview = {}
        for sub_name in subtasks_to_run:
            summary_path = OUTPUT_DIR / model_name / f"atomic_{sub_name}_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    overview[sub_name] = json.load(f)
        save_json(OUTPUT_DIR / f"atomic_{model_name}_overview.json", overview)

    # Save all per-episode metrics merged
    save_json(OUTPUT_DIR / "all_per_episode_metrics.json", all_per_episode)

    print(f"\nAll done! Output: {OUTPUT_DIR}", flush=True)
    print(f"Completed: {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()
