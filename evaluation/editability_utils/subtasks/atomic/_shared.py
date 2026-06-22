#!/usr/bin/env python3
"""Shared runner for atomic subtasks."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..common import (
    Candidate,
    EpisodeCache,
    aggregate_results,
    build_task_map,
    evaluate_image_edit_candidates,
    passes_matching_cost_filter,
    passes_matching_iou_filter,
    passes_gt_opaque_filter_for_match,
    load_match_payloads,
    sample_candidates,
    save_subtask_outputs,
    summarize_capacity,
)


def _select_param_variants_for_pair(
    task_type: str,
    param_grid: Sequence[Dict[str, Any]],
    episode_id: str,
    gt_index: int,
    pred_indices: Sequence[int],
    seed: int,
) -> List[Dict[str, Any]]:
    all_params = [dict(p) for p in param_grid]
    if not all_params:
        return []

    if task_type in {"transition", "rotation"}:
        k = 2
    else:
        k = 1
    k = max(1, min(k, len(all_params)))
    if len(all_params) <= k:
        return all_params

    # Keep param sampling model-agnostic so different baselines (agent/qwen/others)
    # receive the same edit params for the same (episode_id, gt_index, task_type).
    key = f"{int(seed)}::{str(task_type)}::{str(episode_id)}::gt{int(gt_index)}"
    rng = random.Random(int(hashlib.md5(key.encode("utf-8")).hexdigest()[:12], 16))
    idxs = list(range(len(all_params)))
    rng.shuffle(idxs)
    picked = idxs[:k]
    return [all_params[i] for i in picked]


def _build_base_candidates(
    payloads: List[Dict[str, Any]],
    cache: EpisodeCache,
    task_type: str,
    param_grid: Sequence[Dict[str, Any]],
    seed: int,
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    build_log_every: int = 0,
    model: str = "",
    min_gt_opaque_pixels: Optional[int] = None,
    opaque_alpha_threshold: Optional[int] = None,
    matching_cost_threshold: Optional[float] = None,
) -> List[Candidate]:
    out: List[Candidate] = []
    need_z_scene = task_type == "z_order"
    total_payloads = len(payloads)
    opaque_ok_cache: Dict[Tuple[str, int], bool] = {}
    for pi, payload in enumerate(payloads, start=1):
        eid = payload["episode_id"]
        gt_elements = None
        pred_elements = None
        # For z_order building, use lightweight pred count from payload metadata
        # instead of loading the full episode (avoids expensive rendering).
        _z_pred_count: int = 0
        if need_z_scene:
            counts = payload.get("counts")
            if isinstance(counts, dict) and counts.get("pred", 0) > 0:
                _z_pred_count = int(counts["pred"])
            else:
                # Fallback: load full episode only when metadata is missing
                gt_elements, pred_elements, _ = cache.get(eid)
                _z_pred_count = len(pred_elements) if pred_elements else 0
        for m in payload.get("matches", []):
            gt_idx = m.get("gt_index")
            pred_indices = [int(x) for x in m.get("selected_pred_indices", [])]
            pred_ids = m.get("selected_pred_ids")  # element IDs for remapping
            if gt_idx is None or not pred_indices:
                continue
            gt_idx = int(gt_idx)
            if not passes_matching_cost_filter(m, max_cost=matching_cost_threshold):
                continue
            if not passes_matching_iou_filter(m):
                continue
            if subset_keys is not None and (eid, gt_idx) not in subset_keys:
                continue
            if not passes_gt_opaque_filter_for_match(
                cache,
                eid,
                gt_idx,
                m,
                min_opaque_pixels=min_gt_opaque_pixels,
                alpha_threshold=opaque_alpha_threshold,
                memo=opaque_ok_cache,
            ):
                continue

            if task_type == "z_order":
                # z-order swap must be defined on a single matched pred element.
                best_single = m.get("best_single_pred_index")
                best_single_id = m.get("best_single_pred_id")
                if best_single is not None:
                    pred_indices = [int(best_single)]
                    if best_single_id:
                        pred_ids = [best_single_id]
                else:
                    pred_indices = [pred_indices[0]]
                    if pred_ids:
                        pred_ids = [pred_ids[0]]
                if _z_pred_count <= 0:
                    continue
                if pred_indices[0] < 0 or pred_indices[0] >= _z_pred_count:
                    # Keep z-order candidate generation robust to malformed best_single.
                    # Prefer first valid selected pred index; fall back to 0th pred.
                    fallback = []
                    for x in m.get("selected_pred_indices", []):
                        try:
                            xi = int(x)
                        except Exception:
                            continue
                        if 0 <= xi < _z_pred_count:
                            fallback.append(xi)
                    pred_indices = [fallback[0]] if fallback else [0]

            valid_params: List[Dict[str, Any]]
            if task_type == "z_order":
                # For full cross-model alignment, keep z-order param candidates
                # model-agnostic (no model-side feasibility filtering).
                valid_params = [dict(p) for p in param_grid]
            else:
                valid_params = []
                for p in param_grid:
                    valid_params.append(dict(p))

            chosen_params = _select_param_variants_for_pair(
                task_type=task_type,
                param_grid=valid_params,
                episode_id=eid,
                gt_index=gt_idx,
                pred_indices=pred_indices,
                seed=seed,
            )
            for p in chosen_params:
                out.append(
                    Candidate(
                        episode_id=eid,
                        gt_index=gt_idx,
                        pred_indices=[int(x) for x in pred_indices],
                        task_type=task_type,
                        params=dict(p),
                        gt_is_text=(str(m.get("gt_type", "")).lower() == "text"),
                        pred_ids=list(pred_ids) if pred_ids else None,
                    )
                )
        if build_log_every > 0 and (pi == 1 or pi % build_log_every == 0 or pi == total_payloads):
            prefix = f"[{model}][atomic_{task_type}]" if model else f"[atomic_{task_type}]"
            print(f"{prefix} build candidates payload {pi}/{total_payloads} -> {len(out)}")
    return out


def run_atomic_subtask(
    *,
    task_type: str,
    param_grid: Sequence[Dict[str, Any]],
    figma_data: Path,
    exp_pairs: Sequence[str],
    model: str,
    match_root: Path,
    output_dir: Path,
    seed: int,
    max_tasks: Optional[int],
    max_episodes: Optional[int],
    include_iou: bool,
    roi_mode: str,
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    roi_dilation_ratio: float = 0.08,
    log_every: int = 25,
    save_pair_viz: bool = False,
    pair_viz_max: Optional[int] = None,
    reference_results: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    build_log_every: int = 0,
    min_gt_opaque_pixels: Optional[int] = None,
    opaque_alpha_threshold: Optional[int] = None,
    matching_cost_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    subtask_name = f"atomic_{task_type}"
    print(f"[{model}][{subtask_name}] init")
    task_map = build_task_map(figma_data, exp_pairs, model=model, max_episodes=max_episodes)
    payloads = load_match_payloads(match_root, model)
    payloads = [p for p in payloads if p["episode_id"] in task_map]
    print(f"[{model}][{subtask_name}] payloads={len(payloads)} episodes={len(task_map)}")

    cache = EpisodeCache(task_map, model)
    candidates = _build_base_candidates(
        payloads,
        cache,
        task_type=task_type,
        param_grid=param_grid,
        seed=seed,
        subset_keys=subset_keys,
        build_log_every=build_log_every,
        model=model,
        min_gt_opaque_pixels=min_gt_opaque_pixels,
        opaque_alpha_threshold=opaque_alpha_threshold,
        matching_cost_threshold=matching_cost_threshold,
    )
    capacity = summarize_capacity(candidates)

    sampled = sample_candidates(
        candidates,
        max_tasks=max_tasks,
        seed=seed,
        balance_text_non_text=True,
        cache=cache,
    )
    print(
        f"[{model}][{subtask_name}] "
        f"episodes={len(task_map)} candidates={len(candidates)} sampled={len(sampled)}"
    )

    results = evaluate_image_edit_candidates(
        sampled,
        cache,
        include_iou=include_iou,
        include_edge_sharpness=False,
        include_lpips=True,
        roi_mode=roi_mode,
        roi_dilation_ratio=roi_dilation_ratio,
        progress_prefix=f"[{model}][{subtask_name}]",
        log_every=log_every,
        save_pair_viz_dir=(output_dir / model / subtask_name / "element_pairs") if save_pair_viz else None,
        pair_viz_max=pair_viz_max,
        reference_metrics=[r.get("metrics", {}) for r in (reference_results or [])],
        reference_label="qwen",
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        checkpoint_path=output_dir / model / f"{subtask_name}_results.json",
        resume=True,
    )
    summary = aggregate_results(results)
    print(f"[{model}][{subtask_name}] done results={len(results)}")

    save_subtask_outputs(
        output_dir=output_dir,
        model=model,
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
