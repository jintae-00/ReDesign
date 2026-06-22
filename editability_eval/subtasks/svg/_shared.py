#!/usr/bin/env python3
"""Shared runner for SVG subtasks."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from ...common_utils import (
    bbox_from_mask,
    compute_region_metrics,
    dilate_mask,
    element_to_rgba,
    load_json,
    relative_dilation_radius,
    render_prompt_overlay_rgba,
    save_json,
    rgba_to_element_like,
    thread_map,
)
from ...nanobanana_bridge import build_recolor_instruction, run_nanobanana_on_rgba
from ...task_common import apply_edit_to_scene, render_scene_rgba, union_mask_from_indices
from ..common import (
    Candidate,
    EpisodeCache,
    aggregate_results,
    build_task_map,
    candidate_key,
    evaluate_image_edit_candidates,
    passes_matching_cost_filter,
    passes_matching_iou_filter,
    passes_gt_opaque_filter_for_match,
    load_match_payloads,
    result_row_key,
    result_row_sort_key,
    sample_candidates,
    save_subtask_outputs,
    summarize_capacity,
)


def _build_candidates(
    payloads: List[Dict[str, Any]],
    cache: EpisodeCache,
    task_type: str,
    param_grid: Sequence[Dict[str, Any]],
    require_stroke: bool,
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    build_log_every: int = 0,
    model: str = "",
    min_gt_opaque_pixels: Optional[int] = None,
    opaque_alpha_threshold: Optional[int] = None,
    matching_cost_threshold: Optional[float] = None,
) -> List[Candidate]:
    out: List[Candidate] = []
    total_payloads = len(payloads)
    gt_unit_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    opaque_ok_cache: Dict[Tuple[str, int], bool] = {}

    def _unit_map_for_episode(eid: str) -> Dict[str, Dict[str, Any]]:
        if eid in gt_unit_cache:
            return gt_unit_cache[eid]
        task = cache.task_map.get(eid)
        if task is None:
            gt_unit_cache[eid] = {}
            return gt_unit_cache[eid]
        frame = load_json(task.gt_json_path)
        d: Dict[str, Dict[str, Any]] = {}
        for unit in frame.get("unit_images", []):
            unit_id = str(unit.get("unit_id", ""))
            if unit_id:
                d[f"gt_{unit_id}"] = unit
        gt_unit_cache[eid] = d
        return d

    def _is_rectangle_unit(unit: Dict[str, Any]) -> bool:
        node_type = str(unit.get("node_type", "")).upper()
        if node_type == "RECTANGLE":
            return True
        radii = unit.get("rectangle_corner_radii") or []
        if isinstance(radii, list) and len(radii) == 4:
            return True
        raw = unit.get("raw_node_data", {})
        fg = raw.get("fillGeometry") if isinstance(raw, dict) else None
        if isinstance(fg, list) and len(fg) > 0 and node_type in {"FRAME", "VECTOR", "INSTANCE", "COMPONENT"}:
            return True
        return False

    def _has_stroke_unit(unit: Dict[str, Any]) -> bool:
        strokes = unit.get("strokes_raw")
        stroke_w = unit.get("stroke_weight")
        return bool(strokes) or (stroke_w is not None and float(stroke_w) > 0)

    for pi, payload in enumerate(payloads, start=1):
        eid = payload["episode_id"]
        unit_map = _unit_map_for_episode(eid)

        for m in payload.get("matches", []):
            gt_idx = m.get("gt_index")
            pred_indices = m.get("selected_pred_indices", [])
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

            gt_id = str(m.get("gt_id", "") or "")
            unit = unit_map.get(gt_id, {})
            # Only aspect-ratio relies on rectangle-like geometry assumptions.
            if task_type in {"aspect_ratio"} and (not _is_rectangle_unit(unit)):
                continue
            if require_stroke and (not _has_stroke_unit(unit)):
                continue

            for p in param_grid:
                out.append(
                    Candidate(
                        episode_id=eid,
                        gt_index=gt_idx,
                        pred_indices=[int(x) for x in pred_indices],
                        task_type=task_type,
                        params=dict(p),
                    )
                )
        if build_log_every > 0 and (pi == 1 or pi % build_log_every == 0 or pi == total_payloads):
            prefix = f"[{model}][svg_{task_type}]" if model else f"[svg_{task_type}]"
            print(f"{prefix} build candidates payload {pi}/{total_payloads} -> {len(out)}")

    return out


def _is_svg_pred_element(elem: Dict[str, Any], model: str) -> bool:
    if model != "agent":
        return False
    parsed = elem.get("meta", {}).get("parsed", {})
    return bool(str(parsed.get("svg_uri", "") or "").strip())


def _evaluate_svg_recolor_candidates(
    candidates: List[Candidate],
    cache: EpisodeCache,
    model: str,
    include_iou: bool,
    include_edge_sharpness: bool,
    include_lpips: bool,
    roi_mode: str,
    roi_dilation_ratio: float,
    use_nanobanana_for_image_recolor: bool,
    require_nanobanana_for_image_recolor: bool,
    nanobanana_retries: int,
    max_nanobanana_calls: Optional[int] = None,
    log_every: int = 0,
    save_pair_viz_dir: Optional[Path] = None,
    pair_viz_max: Optional[int] = None,
    reference_metrics: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
) -> List[Dict[str, Any]]:
    if save_pair_viz_dir is not None:
        save_pair_viz_dir.mkdir(parents=True, exist_ok=True)

    existing_by_key: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path is not None and resume and checkpoint_path.exists():
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

    run_lock = Lock()
    run_done = len(existing_by_key)
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

    nanobanana_budget = None if max_nanobanana_calls is None else max(0, int(max_nanobanana_calls))
    nanobanana_calls_used = 0
    nanobanana_budget_warned = False
    if nanobanana_budget is not None:
        for row in existing_by_key.values():
            metas = row.get("nanobanana", [])
            if not isinstance(metas, list):
                continue
            for meta in metas:
                if not isinstance(meta, dict):
                    continue
                att = meta.get("attempts", 0)
                try:
                    nanobanana_calls_used += max(0, int(att))
                except Exception:
                    pass
        nanobanana_calls_used = min(nanobanana_calls_used, nanobanana_budget)

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

    def _eval_one(job: Tuple[int, Candidate]) -> Dict[str, Any]:
        nonlocal run_done, nanobanana_calls_used, nanobanana_budget_warned
        i, cand = job
        gt_elements, pred_elements, canvas_size = cache.get(cand.episode_id)
        edit = {"task_type": "recolor"}
        edit.update(cand.params)

        need_viz = save_pair_viz_dir is not None and (pair_viz_max is None or i <= pair_viz_max)
        gt_before_rgba = render_scene_rgba(gt_elements, canvas_size) if need_viz else None
        pred_before_rgba = render_scene_rgba(pred_elements, canvas_size) if need_viz else None

        gt_edit_scene = apply_edit_to_scene(gt_elements, [cand.gt_index], canvas_size, edit)
        pred_edit_scene = [dict(e) for e in pred_elements]
        nanobanana_instruction = build_recolor_instruction(
            hue_shift_deg=float(edit.get("hue_shift_deg", 0.0)),
            sat_mul=float(edit.get("sat_mul", 1.0)),
            val_mul=float(edit.get("val_mul", 1.0)),
        )

        nanobanana_meta: List[Dict[str, Any]] = []
        applied_pred_indices: List[int] = []
        for pidx in cand.pred_indices:
            if pidx < 0 or pidx >= len(pred_edit_scene):
                continue
            elem = pred_edit_scene[pidx]

            if use_nanobanana_for_image_recolor and (not _is_svg_pred_element(elem, model=model)):
                reserved_calls = 0
                if nanobanana_budget is not None:
                    with run_lock:
                        remaining = int(nanobanana_budget) - int(nanobanana_calls_used)
                        if remaining > 0:
                            reserved_calls = min(max(1, int(nanobanana_retries)), remaining)
                            nanobanana_calls_used += int(reserved_calls)
                        else:
                            if not nanobanana_budget_warned:
                                print(
                                    f"[{model}][svg_recolor] nanobanana budget exhausted "
                                    f"({nanobanana_budget} calls). Remaining edits use fallback/no-op."
                                )
                                nanobanana_budget_warned = True
                if nanobanana_budget is not None and reserved_calls <= 0:
                    nb_info = {"ok": False, "skipped": "budget_exhausted", "pred_index": int(pidx)}
                    nanobanana_meta.append(nb_info)
                    # Budget exhaustion behaves like nanobanana disabled: deterministic fallback.
                    pred_edit_scene = apply_edit_to_scene(pred_edit_scene, [pidx], canvas_size, edit)
                    applied_pred_indices.append(int(pidx))
                    continue

                rgba = element_to_rgba(elem, canvas_size)
                nb_rgba, nb_info = run_nanobanana_on_rgba(
                    rgba,
                    nanobanana_instruction,
                    retries=int(reserved_calls) if reserved_calls > 0 else nanobanana_retries,
                )
                if reserved_calls > 0 and nanobanana_budget is not None:
                    try:
                        used = int(nb_info.get("attempts", 0))
                    except Exception:
                        used = 0
                    used = max(0, min(int(reserved_calls), int(used)))
                    refund = int(reserved_calls) - int(used)
                    if refund > 0:
                        with run_lock:
                            nanobanana_calls_used = max(0, int(nanobanana_calls_used) - int(refund))
                nb_info["pred_index"] = int(pidx)
                nanobanana_meta.append(nb_info)
                if nb_rgba is not None:
                    if nb_rgba.shape[:2] != rgba.shape[:2]:
                        nb_rgba = np.array(
                            Image.fromarray(np.clip(nb_rgba, 0, 255).astype(np.uint8), "RGBA").resize(
                                (rgba.shape[1], rgba.shape[0]),
                                Image.LANCZOS,
                            ),
                            dtype=np.uint8,
                        )
                    # Keep recolor constrained to the original parsed element support.
                    # This avoids generative spill outside matched element regions.
                    safe_rgba = rgba.copy()
                    fg = rgba[..., 3] > 0
                    if bool(fg.any()):
                        safe_rgba[fg, :3] = nb_rgba[fg, :3]
                    safe_rgba[..., 3] = rgba[..., 3]
                    pred_edit_scene[pidx] = rgba_to_element_like(elem, safe_rgba)
                    applied_pred_indices.append(int(pidx))
                elif require_nanobanana_for_image_recolor:
                    # Keep original when required nanobanana edit fails.
                    continue
                else:
                    pred_edit_scene = apply_edit_to_scene(pred_edit_scene, [pidx], canvas_size, edit)
                    applied_pred_indices.append(int(pidx))
            else:
                # Deterministic SVG path edit.
                pred_edit_scene = apply_edit_to_scene(pred_edit_scene, [pidx], canvas_size, edit)
                applied_pred_indices.append(int(pidx))

        gt_rgba = render_scene_rgba(gt_edit_scene, canvas_size)
        pred_rgba = render_scene_rgba(pred_edit_scene, canvas_size)

        # IMPORTANT: source/target ROI uses GT∪Pred regions.
        gt_src = union_mask_from_indices(gt_elements, [cand.gt_index])
        gt_dst = union_mask_from_indices(gt_edit_scene, [cand.gt_index])
        pred_src = union_mask_from_indices(pred_elements, cand.pred_indices)
        pred_dst = union_mask_from_indices(pred_edit_scene, cand.pred_indices)
        src = gt_src | pred_src
        dst = gt_dst | pred_dst
        if roi_mode == "source":
            roi = src
        elif roi_mode == "target":
            roi = dst
        else:
            roi = src | dst

        if roi_dilation_ratio > 0 and bool(roi.any()):
            r = relative_dilation_radius(bbox_from_mask(roi), ratio=float(roi_dilation_ratio))
            roi = dilate_mask(roi, r)

        metrics = compute_region_metrics(
            gt_rgba,
            pred_rgba,
            roi,
            include_iou=include_iou,
            include_edge_sharpness=include_edge_sharpness,
            include_lpips=include_lpips,
            include_dino=include_lpips,
        )
        full_roi = np.ones((gt_rgba.shape[0], gt_rgba.shape[1]), dtype=bool)
        metrics_full = compute_region_metrics(
            gt_rgba,
            pred_rgba,
            full_roi,
            include_iou=False,
            include_edge_sharpness=False,
            include_lpips=include_lpips,
            include_dino=include_lpips,
        )
        for k in ("l1", "l2", "psnr", "ssim", "lpips", "dino"):
            if k in metrics_full:
                metrics[f"full_{k}"] = metrics_full[k]

        row = {
            "episode_id": cand.episode_id,
            "gt_index": cand.gt_index,
            "pred_indices": cand.pred_indices,
            "task_type": cand.task_type,
            "params": cand.params,
            "applied_edit": edit,
            "metrics": metrics,
            "metrics_full": {k: v for k, v in metrics_full.items() if k in {"l1", "l2", "psnr", "ssim", "lpips", "dino"}},
            "applied_pred_indices": applied_pred_indices,
            "nanobanana": nanobanana_meta,
            "nanobanana_prompt": (nanobanana_instruction if nanobanana_meta else None),
        }
        if need_viz and save_pair_viz_dir is not None and gt_before_rgba is not None and pred_before_rgba is not None:
            pair_name = (
                f"{i:05d}__{cand.episode_id}"
                f"__gt{int(cand.gt_index)}"
                f"__pred{_join_pred_indices(cand.pred_indices)}"
            )
            pair_dir = save_pair_viz_dir / pair_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            panel = _render_pair_panel(gt_before_rgba, gt_rgba, pred_before_rgba, pred_rgba)
            Image.fromarray(panel, "RGBA").save(pair_dir / "panel_raw.png")
            if nanobanana_meta:
                panel_with_prompt = render_prompt_overlay_rgba(
                    panel,
                    str(nanobanana_instruction or ""),
                    title="Nanobanana Prompt (colorization)",
                )
                Image.fromarray(panel_with_prompt, "RGBA").save(pair_dir / "panel.png")
            else:
                Image.fromarray(panel, "RGBA").save(pair_dir / "panel.png")
            Image.fromarray(gt_before_rgba, "RGBA").save(pair_dir / "gt_before.png")
            Image.fromarray(gt_rgba, "RGBA").save(pair_dir / "gt_after.png")
            Image.fromarray(pred_before_rgba, "RGBA").save(pair_dir / "pred_before.png")
            Image.fromarray(pred_rgba, "RGBA").save(pair_dir / "pred_after.png")

            roi_vis = np.zeros((roi.shape[0], roi.shape[1], 4), dtype=np.uint8)
            roi_vis[..., 1] = 255
            roi_vis[..., 3] = (roi.astype(np.uint8) * 200)
            Image.fromarray(roi_vis, "RGBA").save(pair_dir / "roi.png")

            src_vis = np.zeros((src.shape[0], src.shape[1], 4), dtype=np.uint8)
            src_vis[..., 2] = 255
            src_vis[..., 3] = (src.astype(np.uint8) * 180)
            Image.fromarray(src_vis, "RGBA").save(pair_dir / "roi_source.png")

            dst_vis = np.zeros((dst.shape[0], dst.shape[1], 4), dtype=np.uint8)
            dst_vis[..., 0] = 255
            dst_vis[..., 3] = (dst.astype(np.uint8) * 180)
            Image.fromarray(dst_vis, "RGBA").save(pair_dir / "roi_target.png")
            save_json(pair_dir / "meta.json", row)

        if log_every > 0:
            with run_lock:
                run_done += 1
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and (v == v):
                        run_sum[k] = run_sum.get(k, 0.0) + float(v)
                        run_cnt[k] = run_cnt.get(k, 0) + 1
                if run_done == 1 or run_done % log_every == 0 or run_done == len(candidates):
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
                    msg = (
                        f"[{model}][svg_recolor] running_avg {run_done}/{len(candidates)} "
                        f"agent({' '.join(xs)})"
                    )
                    if reference_metrics:
                        n = min(run_done, len(reference_metrics))
                        ref_sum: Dict[str, float] = {}
                        ref_cnt: Dict[str, int] = {}
                        for rr in reference_metrics[:n]:
                            if not isinstance(rr, dict):
                                continue
                            for k, v in rr.items():
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
                if checkpoint_path is not None:
                    save_json(checkpoint_path, sorted(existing_by_key.values(), key=result_row_sort_key))
        else:
            with run_lock:
                existing_by_key[result_row_key(row)] = row
                if checkpoint_path is not None:
                    save_json(checkpoint_path, sorted(existing_by_key.values(), key=result_row_sort_key))
        return row

    ordered_candidates = sorted(
        candidates,
        key=lambda c: (str(c.episode_id), int(c.gt_index), tuple(int(x) for x in c.pred_indices)),
    )
    pending = [(i, c) for i, c in enumerate(ordered_candidates, start=1) if candidate_key(c) not in existing_by_key]
    if pending:
        _ = thread_map(
            pending,
            _eval_one,
            num_workers=max(1, int(num_workers)),
            desc=f"[{model}][svg_recolor]",
            show_tqdm=show_tqdm,
        )
    elif checkpoint_path is not None and resume:
        print(f"[{model}][svg_recolor] resume: no pending items")

    rows = sorted(existing_by_key.values(), key=result_row_sort_key)
    if checkpoint_path is not None:
        save_json(checkpoint_path, rows)
    return rows


def run_svg_subtask(
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
    include_edge_sharpness: bool,
    include_lpips: bool,
    roi_mode: str,
    require_stroke: bool,
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    roi_dilation_ratio: float = 0.08,
    use_nanobanana_for_image_recolor: bool = False,
    require_nanobanana_for_image_recolor: bool = True,
    nanobanana_retries: int = 2,
    max_nanobanana_calls: Optional[int] = None,
    log_every: int = 0,
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
    task_map = build_task_map(figma_data, exp_pairs, model=model, max_episodes=max_episodes)
    payloads = load_match_payloads(match_root, model)
    payloads = [p for p in payloads if p["episode_id"] in task_map]

    cache = EpisodeCache(task_map, model)
    candidates = _build_candidates(
        payloads,
        cache,
        task_type,
        param_grid,
        require_stroke=require_stroke,
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
        balance_text_non_text=False,
        cache=cache,
    )
    print(
        f"[{model}][svg_{task_type}] episodes={len(task_map)} "
        f"candidates={len(candidates)} sampled={len(sampled)}"
    )

    if task_type == "recolor":
        results = _evaluate_svg_recolor_candidates(
            sampled,
            cache,
            model=model,
            include_iou=include_iou,
            include_edge_sharpness=include_edge_sharpness,
            include_lpips=include_lpips,
            roi_mode=roi_mode,
            roi_dilation_ratio=roi_dilation_ratio,
            use_nanobanana_for_image_recolor=use_nanobanana_for_image_recolor,
            require_nanobanana_for_image_recolor=require_nanobanana_for_image_recolor,
            nanobanana_retries=nanobanana_retries,
            max_nanobanana_calls=max_nanobanana_calls,
            log_every=log_every,
            save_pair_viz_dir=(output_dir / model / f"svg_{task_type}" / "element_pairs") if save_pair_viz else None,
            pair_viz_max=pair_viz_max,
            reference_metrics=[r.get("metrics", {}) for r in (reference_results or [])],
            num_workers=num_workers,
            show_tqdm=show_tqdm,
            checkpoint_path=output_dir / model / f"svg_{task_type}_results.json",
            resume=True,
        )
    else:
        results = evaluate_image_edit_candidates(
            sampled,
            cache,
            include_iou=include_iou,
            include_edge_sharpness=include_edge_sharpness,
            include_lpips=include_lpips,
            roi_mode=roi_mode,
            roi_dilation_ratio=roi_dilation_ratio,
            progress_prefix=f"[{model}][svg_{task_type}]",
            log_every=log_every,
            save_pair_viz_dir=(output_dir / model / f"svg_{task_type}" / "element_pairs") if save_pair_viz else None,
            pair_viz_max=pair_viz_max,
            reference_metrics=[r.get("metrics", {}) for r in (reference_results or [])],
            reference_label="qwen",
            num_workers=num_workers,
            show_tqdm=show_tqdm,
            checkpoint_path=output_dir / model / f"svg_{task_type}_results.json",
            resume=True,
        )
    summary = aggregate_results(results)
    print(f"[{model}][svg_{task_type}] done results={len(results)}")

    save_subtask_outputs(
        output_dir=output_dir,
        model=model,
        subtask_name=f"svg_{task_type}",
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
