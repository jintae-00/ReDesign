#!/usr/bin/env python3
"""GT-centric greedy matching for editability tasks."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common_utils import element_to_rgba


@dataclass
class MatchConfig:
    lambda_l1: float = 0.7
    lambda_iou: float = 0.3
    # None means no hard cap on greedy merge length.
    max_merge_n: Optional[int] = None
    min_gt_overlap: float = 0.0
    min_cost_improve: float = 1e-4
    l1_mode: str = "rgba"  # rgb | rgba (default: include alpha to avoid transparent/background ambiguity)
    # Max candidates per GT element (None = unlimited). Top-K by bbox overlap.
    max_candidates: Optional[int] = None

    def __post_init__(self) -> None:
        mode = str(self.l1_mode).strip().lower()
        if mode not in {"rgb", "rgba"}:
            raise ValueError(f"Invalid l1_mode={self.l1_mode!r}. Expected one of: rgb, rgba")
        self.l1_mode = mode
        if self.max_merge_n is not None:
            cap = int(self.max_merge_n)
            # Non-positive cap is treated as unlimited merge for convenience in CLI usage.
            self.max_merge_n = None if cap <= 0 else cap


def _bbox_overlap_over_gt_ratio(gt_bbox: Sequence[int], pred_bbox: Sequence[int]) -> float:
    """Intersection area normalized by GT bbox area."""
    ax1, ay1, ax2, ay2 = [int(v) for v in gt_bbox]
    bx1, by1, bx2, by2 = [int(v) for v in pred_bbox]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    gt_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return float(inter / gt_area)


def _pred_union_rgba_and_mask(preds: List[Dict[str, Any]], canvas_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    from evaluation_figma import composite_elements_transparent

    if not preds:
        h, w = canvas_size[1], canvas_size[0]
        return np.zeros((h, w, 4), dtype=np.uint8), np.zeros((h, w), dtype=np.float32)
    pred_img, pred_mask = composite_elements_transparent(preds, canvas_size)
    return np.array(pred_img.convert("RGBA"), dtype=np.uint8), pred_mask.astype(np.float32)


def _cost_from_buffers(
    gt_rgba: np.ndarray,
    gt_mask: np.ndarray,
    pred_rgba: np.ndarray,
    pred_mask: np.ndarray,
    lambda_l1: float,
    lambda_iou: float,
    l1_mode: str = "rgba",
) -> Dict[str, float]:
    gt_bin = gt_mask > 0
    pred_bin = pred_mask > 0
    union = gt_bin | pred_bin
    inter = gt_bin & pred_bin
    union_count = int(union.sum())
    inter_count = int(inter.sum())
    iou = float(inter_count / union_count) if union_count > 0 else 0.0

    if gt_bin.sum() <= 0:
        l1 = 1.0
    elif inter_count <= 0:
        # Explicit rule: no overlap => maximal penalty.
        l1 = 1.0
    elif union_count > 0:
        gt_rgb = gt_rgba[..., :3].astype(np.float32) / 255.0
        pr_rgb = pred_rgba[..., :3].astype(np.float32) / 255.0
        if l1_mode == "rgb":
            l1 = float(np.mean(np.abs(gt_rgb[union] - pr_rgb[union])))
        else:
            gt_alpha = gt_rgba[..., 3].astype(np.float32) / 255.0
            pr_alpha = pred_rgba[..., 3].astype(np.float32) / 255.0
            rgb_diff = np.abs(gt_rgb[union] - pr_rgb[union]).mean(axis=1)
            alpha_diff = np.abs(gt_alpha[union] - pr_alpha[union])
            pixel_l1 = np.maximum(rgb_diff, alpha_diff)
            # Pixels where only one of GT or Pred has alpha get the maximum penalty.
            alpha_miss = ((gt_alpha[union] > 1e-6) & (pr_alpha[union] <= 1e-6)) | (
                (pr_alpha[union] > 1e-6) & (gt_alpha[union] <= 1e-6)
            )
            pixel_l1[alpha_miss] = 1.0
            l1 = float(pixel_l1.mean())
    else:
        # Degenerate fallback.
        l1 = 1.0

    cost = float(lambda_l1 * l1 + lambda_iou * (1.0 - iou))
    return {
        "cost": float(cost),
        "l1": float(l1),
        "iou": float(iou),
        "score": float(1.0 - cost),
    }


def _cost_from_prepared(
    gt_rgb: np.ndarray,
    gt_alpha: np.ndarray,
    gt_mask: np.ndarray,
    pred_rgb: np.ndarray,
    pred_alpha: np.ndarray,
    pred_mask: np.ndarray,
    lambda_l1: float,
    lambda_iou: float,
    l1_mode: str = "rgba",
) -> Dict[str, float]:
    inter = gt_mask & pred_mask
    union = gt_mask | pred_mask
    union_count = int(union.sum())
    inter_count = int(inter.sum())
    iou = float(inter_count / union_count) if union_count > 0 else 0.0
    gt_count = int(gt_mask.sum())
    if gt_count <= 0:
        l1 = 1.0
    elif inter_count <= 0:
        # Explicit rule: no overlap => maximal penalty.
        l1 = 1.0
    elif union_count > 0:
        if l1_mode == "rgb":
            l1 = float(np.abs(gt_rgb[union] - pred_rgb[union]).mean())
        else:
            rgb_diff = np.abs(gt_rgb[union] - pred_rgb[union]).mean(axis=1)
            alpha_diff = np.abs(gt_alpha[union] - pred_alpha[union])
            pixel_l1 = np.maximum(rgb_diff, alpha_diff)
            # Pixels where only one of GT or Pred has alpha get the maximum penalty.
            alpha_miss = ((gt_alpha[union] > 1e-6) & (pred_alpha[union] <= 1e-6)) | (
                (pred_alpha[union] > 1e-6) & (gt_alpha[union] <= 1e-6)
            )
            pixel_l1[alpha_miss] = 1.0
            l1 = float(pixel_l1.mean())
    else:
        # Degenerate fallback.
        l1 = 1.0

    cost = float(lambda_l1 * l1 + lambda_iou * (1.0 - iou))
    return {
        "cost": float(cost),
        "l1": float(l1),
        "iou": float(iou),
        "score": float(1.0 - cost),
    }


def compute_set_cost(
    gt_elem: Dict[str, Any],
    pred_indices: Sequence[int],
    pred_elements: List[Dict[str, Any]],
    canvas_size: Tuple[int, int],
    lambda_l1: float,
    lambda_iou: float,
    l1_mode: str = "rgba",
) -> Dict[str, float]:
    gt_rgba = element_to_rgba(gt_elem, canvas_size)
    gt_mask = gt_elem["mask"]

    if pred_indices:
        preds = [pred_elements[i] for i in pred_indices]
        pred_rgba, pred_mask = _pred_union_rgba_and_mask(preds, canvas_size)
    else:
        h, w = canvas_size[1], canvas_size[0]
        pred_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        pred_mask = np.zeros((h, w), dtype=np.float32)

    return _cost_from_buffers(
        gt_rgba=gt_rgba,
        gt_mask=gt_mask,
        pred_rgba=pred_rgba,
        pred_mask=pred_mask,
        lambda_l1=lambda_l1,
        lambda_iou=lambda_iou,
        l1_mode=l1_mode,
    )


def _candidate_pred_indices(
    gt_elem: Dict[str, Any],
    pred_elements: List[Dict[str, Any]],
    min_gt_overlap: float,
    max_candidates: Optional[int] = None,
) -> List[int]:
    gt_bbox = gt_elem.get("bbox", [0, 0, 1, 1])
    scored: List[Tuple[int, float]] = []
    for i, pred in enumerate(pred_elements):
        ov = _bbox_overlap_over_gt_ratio(gt_bbox, pred.get("bbox", [0, 0, 1, 1]))
        if ov >= min_gt_overlap:
            scored.append((i, ov))
    if max_candidates is not None and len(scored) > max_candidates:
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:max_candidates]
    return [idx for idx, _ in scored]


def greedy_match_gt_to_pred(
    gt_elements: List[Dict[str, Any]],
    pred_elements: List[Dict[str, Any]],
    canvas_size: Tuple[int, int],
    cfg: MatchConfig,
    progress_cb: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    h, w = canvas_size[1], canvas_size[0]
    empty_mask = np.zeros((h, w), dtype=np.float32)
    empty_rgb01 = np.zeros((h, w, 3), dtype=np.float32)
    empty_alpha01 = np.zeros((h, w), dtype=np.float32)
    empty_mask_bin = np.zeros((h, w), dtype=bool)

    prep_t0 = time.time()
    # Lazy pred cache: only rasterize when first accessed (critical for VTracer with 700+ paths)
    _pred_rgba_raw: Dict[int, np.ndarray] = {}
    _pred_rgb01: Dict[int, np.ndarray] = {}
    _pred_alpha01: Dict[int, np.ndarray] = {}
    _pred_mask_bin: Dict[int, np.ndarray] = {}

    def _ensure_pred_cached(idx: int) -> None:
        if idx not in _pred_rgba_raw:
            rgba = element_to_rgba(pred_elements[idx], canvas_size)
            _pred_rgba_raw[idx] = rgba
            _pred_rgb01[idx] = rgba[..., :3].astype(np.float32) / 255.0
            _pred_alpha01[idx] = rgba[..., 3].astype(np.float32) / 255.0
            _pred_mask_bin[idx] = pred_elements[idx].get("mask", empty_mask) > 0

    pred_cache_prepare_sec = time.time() - prep_t0

    total_cost_evals = 0
    total_candidates = 0
    max_candidates = 0
    max_gt_time = 0.0
    slowest_gt_index = -1
    total_candidate_filter_sec = 0.0
    total_eval_sec = 0.0
    total_union_render_sec = 0.0
    total_gt_prepare_sec = 0.0
    total_union_cache_hit = 0
    total_union_cache_miss = 0
    total_union_render_calls = 0
    total_eval_empty = 0
    total_eval_single = 0
    total_eval_multi = 0

    for g_idx, gt in enumerate(gt_elements):
        gt_t0 = time.time()
        union_cache: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        gt_prep_t0 = time.time()
        gt_rgba = element_to_rgba(gt, canvas_size)
        gt_rgb01 = gt_rgba[..., :3].astype(np.float32) / 255.0
        gt_alpha01 = gt_rgba[..., 3].astype(np.float32) / 255.0
        gt_mask_bin = gt.get("mask", empty_mask) > 0
        total_gt_prepare_sec += time.time() - gt_prep_t0

        cand_t0 = time.time()
        candidate_indices = _candidate_pred_indices(gt, pred_elements, cfg.min_gt_overlap, cfg.max_candidates)
        cand_dt = time.time() - cand_t0
        total_candidate_filter_sec += cand_dt
        total_candidates += len(candidate_indices)
        max_candidates = max(max_candidates, len(candidate_indices))
        gt_cost_evals = 0
        gt_eval_sec = 0.0
        gt_union_render_sec = 0.0
        gt_union_cache_hit = 0
        gt_union_cache_miss = 0
        gt_union_render_calls = 0

        def eval_set(indices: Sequence[int]) -> Dict[str, float]:
            nonlocal total_cost_evals
            nonlocal total_eval_sec
            nonlocal total_union_render_sec
            nonlocal total_union_cache_hit
            nonlocal total_union_cache_miss
            nonlocal total_union_render_calls
            nonlocal total_eval_empty
            nonlocal total_eval_single
            nonlocal total_eval_multi
            nonlocal gt_cost_evals
            nonlocal gt_eval_sec
            nonlocal gt_union_render_sec
            nonlocal gt_union_cache_hit
            nonlocal gt_union_cache_miss
            nonlocal gt_union_render_calls

            eval_t0 = time.time()
            total_cost_evals += 1
            gt_cost_evals += 1

            if not indices:
                total_eval_empty += 1
                pred_rgb01, pred_alpha01, pred_mask_bin = empty_rgb01, empty_alpha01, empty_mask_bin
            elif len(indices) == 1:
                total_eval_single += 1
                idx = indices[0]
                _ensure_pred_cached(idx)
                pred_rgb01 = _pred_rgb01[idx]
                pred_alpha01 = _pred_alpha01[idx]
                pred_mask_bin = _pred_mask_bin[idx]
            else:
                total_eval_multi += 1
                key = tuple(sorted(indices))
                cached = union_cache.get(key)
                if cached is None:
                    total_union_cache_miss += 1
                    gt_union_cache_miss += 1
                    union_t0 = time.time()
                    preds = [pred_elements[i] for i in key]
                    pred_rgba, pred_mask = _pred_union_rgba_and_mask(preds, canvas_size)
                    cached = (
                        pred_rgba[..., :3].astype(np.float32) / 255.0,
                        pred_rgba[..., 3].astype(np.float32) / 255.0,
                        pred_mask > 0,
                    )
                    union_cache[key] = cached
                    union_dt = time.time() - union_t0
                    total_union_render_sec += union_dt
                    gt_union_render_sec += union_dt
                    total_union_render_calls += 1
                    gt_union_render_calls += 1
                else:
                    total_union_cache_hit += 1
                    gt_union_cache_hit += 1
                pred_rgb01, pred_alpha01, pred_mask_bin = cached

            out = _cost_from_prepared(
                gt_rgb=gt_rgb01,
                gt_alpha=gt_alpha01,
                gt_mask=gt_mask_bin,
                pred_rgb=pred_rgb01,
                pred_alpha=pred_alpha01,
                pred_mask=pred_mask_bin,
                lambda_l1=cfg.lambda_l1,
                lambda_iou=cfg.lambda_iou,
                l1_mode=cfg.l1_mode,
            )
            eval_dt = time.time() - eval_t0
            total_eval_sec += eval_dt
            gt_eval_sec += eval_dt
            return out

        # Best single match (for Text Content Recognition flow).
        best_single_idx: Optional[int] = None
        best_single_cost = float("inf")
        best_single_metrics = {"cost": 1.0, "l1": 1.0, "iou": 0.0, "score": 0.0}
        for p_idx in candidate_indices:
            m = eval_set([p_idx])
            if m["cost"] < best_single_cost:
                best_single_cost = m["cost"]
                best_single_idx = p_idx
                best_single_metrics = m

        selected: List[int] = []
        merge_trace: List[Dict[str, Any]] = []
        current = eval_set(selected)

        # GT-centric greedy merge: add one pred at a time while cost improves.
        remaining = set(candidate_indices)
        while remaining and (cfg.max_merge_n is None or len(selected) < cfg.max_merge_n):
            trial_best_idx: Optional[int] = None
            trial_best_metrics: Optional[Dict[str, float]] = None
            for p_idx in remaining:
                trial = eval_set(selected + [p_idx])
                if trial_best_metrics is None or trial["cost"] < trial_best_metrics["cost"]:
                    trial_best_idx = p_idx
                    trial_best_metrics = trial

            if trial_best_idx is None or trial_best_metrics is None:
                break

            if current["cost"] - trial_best_metrics["cost"] < cfg.min_cost_improve:
                break

            selected.append(trial_best_idx)
            current = trial_best_metrics
            remaining.remove(trial_best_idx)
            merge_trace.append(
                {
                    "step": int(len(selected)),
                    "added_pred_index": int(trial_best_idx),
                    "added_pred_id": pred_elements[trial_best_idx].get("id"),
                    "metrics": dict(current),
                }
            )

        selected_ids = [pred_elements[i]["id"] for i in selected]
        best_single_id = pred_elements[best_single_idx]["id"] if best_single_idx is not None else None

        matches.append(
            {
                "gt_index": g_idx,
                "gt_id": gt.get("id"),
                "gt_type": gt.get("type", "object"),
                "gt_area": float(gt.get("area", 0.0)),
                "selected_pred_indices": selected,
                "selected_pred_ids": selected_ids,
                "best_single_pred_index": best_single_idx,
                "best_single_pred_id": best_single_id,
                "merged_metrics": current,
                "best_single_metrics": best_single_metrics,
                "merge_trace": merge_trace,
                "num_candidates": len(candidate_indices),
            }
        )
        gt_dt = time.time() - gt_t0
        if gt_dt > max_gt_time:
            max_gt_time = gt_dt
            slowest_gt_index = g_idx
        if progress_cb is not None:
            progress_cb(
                g_idx + 1,
                len(gt_elements),
                {
                    "avg_candidates": float(total_candidates / max(1, g_idx + 1)),
                    "cost_evals": int(total_cost_evals),
                    "gt_elapsed_sec": float(gt_dt),
                    "gt_candidates": int(len(candidate_indices)),
                    "gt_cost_evals": int(gt_cost_evals),
                    "gt_eval_sec": float(gt_eval_sec),
                    "gt_candidate_filter_sec": float(cand_dt),
                    "gt_union_render_sec": float(gt_union_render_sec),
                    "gt_union_render_calls": int(gt_union_render_calls),
                    "gt_union_cache_hit": int(gt_union_cache_hit),
                    "gt_union_cache_miss": int(gt_union_cache_miss),
                },
            )

    n_gt = len(gt_elements)
    stats = {
        "num_gt": n_gt,
        "num_pred": len(pred_elements),
        "total_cost_evals": total_cost_evals,
        "avg_cost_evals_per_gt": float(total_cost_evals / max(1, n_gt)),
        "avg_candidates_per_gt": float(total_candidates / max(1, n_gt)),
        "max_candidates_per_gt": int(max_candidates),
        "slowest_gt_index": int(slowest_gt_index),
        "slowest_gt_sec": float(max_gt_time),
        "pred_cache_prepare_sec": float(pred_cache_prepare_sec),
        "total_gt_prepare_sec": float(total_gt_prepare_sec),
        "total_candidate_filter_sec": float(total_candidate_filter_sec),
        "total_eval_sec": float(total_eval_sec),
        "total_union_render_sec": float(total_union_render_sec),
        "avg_eval_sec_per_cost_eval": float(total_eval_sec / max(1, total_cost_evals)),
        "avg_union_render_sec_per_call": float(total_union_render_sec / max(1, total_union_render_calls)),
        "union_render_calls": int(total_union_render_calls),
        "union_cache_hit": int(total_union_cache_hit),
        "union_cache_miss": int(total_union_cache_miss),
        "eval_count_empty": int(total_eval_empty),
        "eval_count_single": int(total_eval_single),
        "eval_count_multi": int(total_eval_multi),
    }
    return matches, stats
