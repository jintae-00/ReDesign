#!/usr/bin/env python3
"""Text style combo subtask: scaling + bold + recolor."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from ...common_utils import (
    compute_region_metrics,
    load_json,
    save_json,
    sample_with_seed_balanced_by_key,
    thread_map,
)
from ...task_common import apply_edit_to_scene, render_scene_rgba, union_mask_from_indices
from ..common import (
    Candidate,
    EpisodeCache,
    aggregate_results,
    candidate_key,
    result_row_key,
    result_row_sort_key,
    summarize_capacity,
)
from ._shared import collect_text_pairs


def _apply_combo(elements, target_indices, canvas_size, params):
    out = elements
    edits = [
        {"task_type": "super_scaling", "scale": params.get("scale", 1.8)},
        {"task_type": "text_bold", "strength": params.get("strength", 3)},
        {
            "task_type": "recolor",
            "hue_shift_deg": params.get("hue_shift_deg", 60.0),
            "sat_mul": params.get("sat_mul", 1.4),
            "val_mul": params.get("val_mul", 1.0),
        },
    ]
    for e in edits:
        out = apply_edit_to_scene(out, target_indices, canvas_size, e)
    return out


def run(
    figma_data: Path,
    exp_pairs: Sequence[str],
    model: str,
    match_root: Path,
    output_dir: Path,
    seed: int = 123,
    max_tasks: Optional[int] = None,
    max_episodes: Optional[int] = None,
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    log_every: int = 0,
    save_pair_viz: bool = False,
    pair_viz_max: Optional[int] = None,
    reference_results: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    build_log_every: int = 0,
) -> Dict:
    _, pairs, cache = collect_text_pairs(
        figma_data=figma_data,
        exp_pairs=exp_pairs,
        model=model,
        match_root=match_root,
        max_episodes=max_episodes,
        subset_keys=subset_keys,
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        build_log_every=build_log_every,
        need_pred_text=False,
    )

    params_grid = [
        {"scale": 1.8, "strength": 2, "hue_shift_deg": 90.0, "sat_mul": 0.6, "val_mul": 1.0},
        {"scale": 2.4, "strength": 4, "hue_shift_deg": -90.0, "sat_mul": 1.8, "val_mul": 1.0},
        {"scale": 3.2, "strength": 3, "hue_shift_deg": 55.0, "sat_mul": 1.6, "val_mul": 1.0},
        {"scale": 4.0, "strength": 4, "hue_shift_deg": -55.0, "sat_mul": 0.7, "val_mul": 1.0},
    ]

    candidates: List[Candidate] = []
    for p in pairs:
        pred_indices = p.get("pred_indices") or [p["pred_index"]]
        if not pred_indices:
            continue
        for params in params_grid:
            candidates.append(
                Candidate(
                    episode_id=p["episode_id"],
                    gt_index=int(p["gt_index"]),
                    pred_indices=[int(x) for x in pred_indices],
                    task_type="text_style_combo",
                    params=params,
                )
            )

    sampled = sample_with_seed_balanced_by_key(
        candidates, key_fn=lambda c: c.episode_id, max_count=max_tasks, seed=seed
    )
    sampled = sorted(
        sampled,
        key=lambda c: (str(c.episode_id), int(c.gt_index), tuple(int(x) for x in c.pred_indices)),
    )
    checkpoint_path = output_dir / model / "style_combo_results.json"
    existing_by_key: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path.exists():
        try:
            loaded = load_json(checkpoint_path)
            if isinstance(loaded, list):
                for row in loaded:
                    if isinstance(row, dict):
                        k = result_row_key(row)
                        if k:
                            existing_by_key[k] = row
        except Exception:
            existing_by_key = {}
    allowed_keys = {candidate_key(c) for c in candidates}
    if allowed_keys:
        existing_by_key = {k: v for k, v in existing_by_key.items() if k in allowed_keys}

    sampled = [c for c in sampled if candidate_key(c) not in existing_by_key]
    pair_viz_dir = (output_dir / model / "style_combo" / "element_pairs") if save_pair_viz else None
    if pair_viz_dir is not None:
        pair_viz_dir.mkdir(parents=True, exist_ok=True)

    run_lock = Lock()
    run_done = len(existing_by_key)
    total_target = len(existing_by_key) + len(sampled)
    run_sum: Dict[str, float] = {}
    run_cnt: Dict[str, int] = {}
    for row in existing_by_key.values():
        mm = row.get("metrics", {})
        if not isinstance(mm, dict):
            continue
        for k, v in mm.items():
            if isinstance(v, (int, float)) and (v == v):
                run_sum[k] = run_sum.get(k, 0.0) + float(v)
                run_cnt[k] = run_cnt.get(k, 0) + 1

    def _join_pred_indices(xs: Sequence[int]) -> str:
        if len(xs) <= 6:
            return "-".join(str(int(x)) for x in xs)
        return "-".join(str(int(x)) for x in xs[:6]) + f"-n{len(xs)}"

    def _render_pair_panel(
        gt_before: np.ndarray,
        gt_after: np.ndarray,
        pred_before: np.ndarray,
        pred_after: np.ndarray,
    ) -> np.ndarray:
        pad = 4
        h, _ = gt_before.shape[:2]
        spacer_v = np.full((h, pad, 4), 255, dtype=np.uint8)
        top = np.concatenate([gt_before, spacer_v, gt_after], axis=1)
        bot = np.concatenate([pred_before, spacer_v, pred_after], axis=1)
        spacer_h = np.full((pad, top.shape[1], 4), 255, dtype=np.uint8)
        return np.concatenate([top, spacer_h, bot], axis=0)

    def _eval_one(job: Tuple[int, Candidate]) -> Dict:
        nonlocal run_done
        i, c = job
        gt_elements, pred_elements, canvas_size = cache.get(c.episode_id)
        need_viz = pair_viz_dir is not None and (pair_viz_max is None or i <= pair_viz_max)
        gt_before_rgba = render_scene_rgba(gt_elements, canvas_size) if need_viz else None
        pred_before_rgba = render_scene_rgba(pred_elements, canvas_size) if need_viz else None
        gt_edit = _apply_combo(gt_elements, [c.gt_index], canvas_size, c.params)
        pred_edit = _apply_combo(pred_elements, c.pred_indices, canvas_size, c.params)

        gt_rgba = render_scene_rgba(gt_edit, canvas_size)
        pred_rgba = render_scene_rgba(pred_edit, canvas_size)

        gt_src = union_mask_from_indices(gt_elements, [c.gt_index])
        pred_src = union_mask_from_indices(pred_elements, c.pred_indices)
        roi = gt_src | pred_src
        metrics = compute_region_metrics(
            gt_rgba,
            pred_rgba,
            roi,
            include_iou=True,
            include_edge_sharpness=True,
            include_lpips=True,
            include_dino=True,
        )
        full_roi = np.ones((gt_rgba.shape[0], gt_rgba.shape[1]), dtype=bool)
        metrics_full = compute_region_metrics(
            gt_rgba,
            pred_rgba,
            full_roi,
            include_iou=False,
            include_edge_sharpness=False,
            include_lpips=True,
            include_dino=True,
        )
        for k in ("l1", "l2", "psnr", "ssim", "lpips", "dino"):
            if k in metrics_full:
                metrics[f"full_{k}"] = metrics_full[k]

        row = {
            "episode_id": c.episode_id,
            "gt_index": c.gt_index,
            "pred_indices": c.pred_indices,
            "task_type": c.task_type,
            "params": c.params,
            "applied_edit": {
                "task_type": "text_style_combo",
                "edits": [
                    {"task_type": "super_scaling", "scale": c.params.get("scale", 1.8)},
                    {"task_type": "text_bold", "strength": c.params.get("strength", 3)},
                    {
                        "task_type": "recolor",
                        "hue_shift_deg": c.params.get("hue_shift_deg", 60.0),
                        "sat_mul": c.params.get("sat_mul", 1.4),
                        "val_mul": c.params.get("val_mul", 1.0),
                    },
                ],
            },
            "metrics": metrics,
            "metrics_full": {k: v for k, v in metrics_full.items() if k in {"l1", "l2", "psnr", "ssim", "lpips", "dino"}},
        }
        if need_viz and pair_viz_dir is not None and gt_before_rgba is not None and pred_before_rgba is not None:
            pair_name = (
                f"{i:05d}__{c.episode_id}"
                f"__gt{int(c.gt_index)}"
                f"__pred{_join_pred_indices(c.pred_indices)}"
            )
            pair_dir = pair_viz_dir / pair_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            panel = _render_pair_panel(gt_before_rgba, gt_rgba, pred_before_rgba, pred_rgba)
            Image.fromarray(panel, "RGBA").save(pair_dir / "panel.png")
            save_json(pair_dir / "meta.json", row)

        with run_lock:
            run_done += 1
            if log_every > 0:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and (v == v):
                        run_sum[k] = run_sum.get(k, 0.0) + float(v)
                        run_cnt[k] = run_cnt.get(k, 0) + 1
                if run_done == 1 or run_done % log_every == 0 or run_done == total_target:
                    xs = []
                    for key in [
                        "l1",
                        "l2",
                        "psnr",
                        "ssim",
                        "lpips",
                        "dino",
                        "iou",
                        "full_l1",
                        "full_l2",
                        "full_psnr",
                        "full_ssim",
                        "full_lpips",
                        "full_dino",
                    ]:
                        c_cnt = run_cnt.get(key, 0)
                        if c_cnt > 0:
                            xs.append(f"{key}={run_sum[key] / c_cnt:.4f}")
                    msg = f"[{model}][style_combo] running_avg {run_done}/{total_target} agent({' '.join(xs)})"
                    if reference_results:
                        n = min(run_done, len(reference_results))
                        ref_sum: Dict[str, float] = {}
                        ref_cnt: Dict[str, int] = {}
                        for rr in reference_results[:n]:
                            mm = rr.get("metrics", {}) if isinstance(rr, dict) else {}
                            if not isinstance(mm, dict):
                                continue
                            for k, v in mm.items():
                                if isinstance(v, (int, float)) and (v == v):
                                    ref_sum[k] = ref_sum.get(k, 0.0) + float(v)
                                    ref_cnt[k] = ref_cnt.get(k, 0) + 1
                        ref_xs = []
                        for key in [
                            "l1",
                            "l2",
                            "psnr",
                            "ssim",
                            "lpips",
                            "dino",
                            "iou",
                            "full_l1",
                            "full_l2",
                            "full_psnr",
                            "full_ssim",
                            "full_lpips",
                            "full_dino",
                        ]:
                            c_cnt = ref_cnt.get(key, 0)
                            if c_cnt > 0:
                                ref_xs.append(f"{key}={ref_sum[key] / c_cnt:.4f}")
                        msg += f" qwen_prefix({' '.join(ref_xs)})"
                    print(msg)
            existing_by_key[result_row_key(row)] = row
            save_json(checkpoint_path, sorted(existing_by_key.values(), key=result_row_sort_key))
        return row

    if sampled:
        _ = thread_map(
            list(enumerate(sampled, start=1)),
            _eval_one,
            num_workers=max(1, int(num_workers)),
            desc=f"[{model}][style_combo]",
            show_tqdm=show_tqdm,
        )
    else:
        print(f"[{model}][style_combo] resume: no pending items")

    results = sorted(existing_by_key.values(), key=result_row_sort_key)

    return {
        "capacity": summarize_capacity(candidates),
        "sampled_count": len(results),
        "results": results,
        "summary": aggregate_results(results),
    }
