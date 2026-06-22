#!/usr/bin/env python3
"""Shared helpers for per-subtask editability runners."""

from __future__ import annotations

import importlib
import json
import random
import hashlib
import os
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..common_utils import (
    bbox_from_mask,
    compute_region_metrics,
    dilate_mask,
    load_json,
    relative_dilation_radius,
    sample_with_seed_balanced_by_key,
    save_json,
    thread_map,
)
from ..loaders import EpisodeTask, collect_episode_tasks, load_episode_elements
from ..task_common import apply_edit_to_scene, render_scene_rgba, union_mask_from_indices


@dataclass
class Candidate:
    episode_id: str
    gt_index: int
    pred_indices: List[int]
    task_type: str
    params: Dict[str, Any]
    gt_is_text: Optional[bool] = None
    pred_ids: Optional[List[str]] = None  # element IDs for index remapping


class EpisodeCache:
    def __init__(
        self,
        task_map: Dict[str, EpisodeTask],
        model: str,
        max_items: Optional[int] = None,
        max_loaders: Optional[int] = None,
    ):
        self.task_map = task_map
        self.model = model
        if max_items is None:
            raw = os.environ.get("EDITABILITY_CACHE_EPISODES", "8")
            try:
                max_items = int(raw)
            except Exception:
                max_items = 8
        self.max_items = None if max_items is None or int(max_items) <= 0 else int(max_items)
        if max_loaders is None:
            raw_loaders = os.environ.get("EDITABILITY_MAX_EP_LOADERS", "2")
            try:
                max_loaders = int(raw_loaders)
            except Exception:
                max_loaders = 2
        self.max_loaders = max(1, int(max_loaders))
        self._load_sem = Semaphore(self.max_loaders)
        self._cache: "OrderedDict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Tuple[int, int]]]" = OrderedDict()
        self._cache_lock = Lock()

    def get(self, episode_id: str):
        with self._cache_lock:
            item = self._cache.get(episode_id)
            if item is not None:
                self._cache.move_to_end(episode_id)
                return item

        # Bound concurrent episode loads to avoid OOM spikes when many workers miss cache at once.
        with self._load_sem:
            loaded = load_episode_elements(self.task_map[episode_id], model=self.model)

        with self._cache_lock:
            item2 = self._cache.get(episode_id)
            if item2 is not None:
                self._cache.move_to_end(episode_id)
                return item2
            self._cache[episode_id] = loaded
            self._cache.move_to_end(episode_id)
            if self.max_items is not None:
                while len(self._cache) > self.max_items:
                    self._cache.popitem(last=False)
            return loaded


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except Exception:
        return float(default)


_EVALFIGMA_POSTPROC_FNS: Optional[Tuple[Optional[Callable[..., Any]], Optional[Callable[..., Any]]]] = None
_EVALFIGMA_POSTPROC_LOCK = Lock()


def _load_evalfigma_postprocess_fns() -> Tuple[Optional[Callable[..., Any]], Optional[Callable[..., Any]]]:
    global _EVALFIGMA_POSTPROC_FNS
    if _EVALFIGMA_POSTPROC_FNS is not None:
        return _EVALFIGMA_POSTPROC_FNS
    with _EVALFIGMA_POSTPROC_LOCK:
        if _EVALFIGMA_POSTPROC_FNS is not None:
            return _EVALFIGMA_POSTPROC_FNS
        clean_fn: Optional[Callable[..., Any]] = None
        text_refine_fn: Optional[Callable[..., Any]] = None
        try:
            mod = importlib.import_module("evaluation.figma_metrics")
            c = getattr(mod, "clean_alpha_noise", None)
            if callable(c):
                clean_fn = c
            t = getattr(mod, "apply_soft_kmeans_refinement", None)
            if callable(t):
                text_refine_fn = t
        except Exception:
            clean_fn = None
            text_refine_fn = None
        _EVALFIGMA_POSTPROC_FNS = (clean_fn, text_refine_fn)
    return _EVALFIGMA_POSTPROC_FNS


def _materialize_element_from_rgba(elem: Dict[str, Any], rgba_img: Image.Image, canvas_size: Tuple[int, int]) -> Dict[str, Any]:
    w, h = int(canvas_size[0]), int(canvas_size[1])
    img = rgba_img.convert("RGBA")
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)

    alpha_u8 = np.array(img.getchannel("A"), dtype=np.uint8)
    alpha = alpha_u8.astype(np.float32) / 255.0
    mask_bin = alpha > 0
    x1, y1, x2, y2 = bbox_from_mask(mask_bin)

    out = dict(elem)
    out["image"] = img
    out["mask"] = alpha
    out["area"] = float(mask_bin.sum())
    out["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
    return out


def _apply_evalfigma_postprocess_to_scene(
    scene: List[Dict[str, Any]],
    target_indices: Sequence[int],
    canvas_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    if _env_int("EDITABILITY_USE_EVALFIGMA_POSTPROC", 0) <= 0:
        return scene

    clean_alpha_noise, text_refine_fn = _load_evalfigma_postprocess_fns()
    if clean_alpha_noise is None and text_refine_fn is None:
        return scene

    use_text_refine = _env_int("EDITABILITY_USE_EVALFIGMA_TEXT_REFINEMENT", 1) > 0
    out = list(scene)
    unique_targets = sorted({int(i) for i in target_indices if 0 <= int(i) < len(scene)})
    for idx in unique_targets:
        elem = scene[idx]
        img = elem.get("image")
        if img is None:
            continue
        try:
            rgba = img.convert("RGBA")
        except Exception:
            continue

        if clean_alpha_noise is not None:
            try:
                rgba = clean_alpha_noise(rgba)
            except Exception:
                pass

        updated = _materialize_element_from_rgba(elem, rgba, canvas_size)
        if use_text_refine and text_refine_fn is not None and str(updated.get("type", "")).lower() == "text":
            try:
                refined = text_refine_fn(updated)
                if isinstance(refined, dict):
                    updated = refined
            except Exception:
                pass
            img2 = updated.get("image")
            if img2 is not None:
                try:
                    rgba2 = img2.convert("RGBA")
                    if clean_alpha_noise is not None:
                        rgba2 = clean_alpha_noise(rgba2)
                    updated = _materialize_element_from_rgba(updated, rgba2, canvas_size)
                except Exception:
                    pass
        out[idx] = updated
    return out


def _psnr_inf_fallback() -> float:
    raw = os.environ.get("EDITABILITY_PSNR_INF_FALLBACK", "100.0")
    try:
        v = float(raw)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return 100.0


def _is_psnr_metric(metric_key: str) -> bool:
    return "psnr" in str(metric_key).lower()


def _normalize_metric_for_stats(
    metric_key: str,
    value: Any,
    *,
    psnr_max: Dict[str, float],
) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    fv = float(value)
    if fv != fv:
        return None

    if _is_psnr_metric(metric_key):
        if math.isfinite(fv):
            prev = psnr_max.get(metric_key)
            if prev is None or fv > prev:
                psnr_max[metric_key] = fv
            return fv
        if fv > 0:
            rep = float(psnr_max.get(metric_key, _psnr_inf_fallback()))
            prev = psnr_max.get(metric_key)
            if prev is None or rep > prev:
                psnr_max[metric_key] = rep
            return rep
        return None

    if math.isfinite(fv):
        return fv
    return None


def gt_opaque_pixel_count(gt_elem: Dict[str, Any], alpha_threshold: Optional[int] = None) -> int:
    if alpha_threshold is None:
        alpha_threshold = _env_int("EDITABILITY_OPAQUE_ALPHA_THRESHOLD", 250)
    thr = max(0, min(255, int(alpha_threshold)))

    img = gt_elem.get("image")
    if img is not None:
        try:
            alpha = np.array(img.convert("RGBA").getchannel("A"), dtype=np.uint8)
            return int((alpha >= thr).sum())
        except Exception:
            pass

    # Fallback for missing image payloads.
    mask = gt_elem.get("mask")
    if isinstance(mask, np.ndarray):
        # Approximate opaque pixels from mask when alpha is unavailable.
        return int((mask >= 0.98).sum())
    return 0


def is_gt_opaque_enough(
    cache: EpisodeCache,
    episode_id: str,
    gt_index: int,
    *,
    min_opaque_pixels: Optional[int] = None,
    alpha_threshold: Optional[int] = None,
    memo: Optional[Dict[Tuple[str, int], bool]] = None,
) -> bool:
    if min_opaque_pixels is None:
        min_opaque_pixels = _env_int("EDITABILITY_MIN_GT_OPAQUE_PIXELS", 1000)
    min_px = max(0, int(min_opaque_pixels))
    if min_px <= 0:
        return True

    key = (str(episode_id), int(gt_index))
    if memo is not None and key in memo:
        return bool(memo[key])

    ok = False
    try:
        gt_elements, _, _ = cache.get(str(episode_id))
        gi = int(gt_index)
        if 0 <= gi < len(gt_elements):
            px = gt_opaque_pixel_count(gt_elements[gi], alpha_threshold=alpha_threshold)
            ok = px >= min_px
    except Exception:
        ok = False

    if memo is not None:
        memo[key] = bool(ok)
    return bool(ok)


def passes_gt_opaque_filter_for_match(
    cache: EpisodeCache,
    episode_id: str,
    gt_index: int,
    match: Dict[str, Any],
    *,
    min_opaque_pixels: Optional[int] = None,
    alpha_threshold: Optional[int] = None,
    memo: Optional[Dict[Tuple[str, int], bool]] = None,
) -> bool:
    """
    Fast-path by payload `gt_area` (already precomputed during matching).
    If strict mode is enabled, fallback to exact alpha>=threshold counting.
    """
    if min_opaque_pixels is None:
        min_opaque_pixels = _env_int("EDITABILITY_MIN_GT_OPAQUE_PIXELS", 1000)
    min_px = max(0, int(min_opaque_pixels))
    if min_px <= 0:
        return True

    strict = _env_int("EDITABILITY_STRICT_GT_OPAQUE_CHECK", 0) > 0
    gt_area = match.get("gt_area")
    try:
        area = float(gt_area)
        if area < float(min_px):
            return False
        if not strict:
            return True
    except Exception:
        # No valid payload area -> strict exact check below.
        pass

    return is_gt_opaque_enough(
        cache,
        episode_id,
        gt_index,
        min_opaque_pixels=min_px,
        alpha_threshold=alpha_threshold,
        memo=memo,
    )


def extract_match_cost(match: Dict[str, Any]) -> Optional[float]:
    for key in ("best_single_metrics", "merged_metrics"):
        mm = match.get(key)
        if isinstance(mm, dict):
            v = mm.get("cost")
            if isinstance(v, (int, float)) and (float(v) == float(v)) and math.isfinite(float(v)):
                return float(v)
    return None


def extract_match_iou(match: Dict[str, Any]) -> Optional[float]:
    for key in ("merged_metrics", "best_single_metrics"):
        mm = match.get(key)
        if isinstance(mm, dict):
            v = mm.get("iou")
            if isinstance(v, (int, float)) and (float(v) == float(v)) and math.isfinite(float(v)):
                return float(v)
    return None


def passes_matching_cost_filter(match: Dict[str, Any], max_cost: Optional[float] = None) -> bool:
    if max_cost is None:
        max_cost = _env_float("EDITABILITY_MAX_MATCHING_COST", 0.5)
    try:
        thr = float(max_cost)
    except Exception:
        thr = 0.5
    if not math.isfinite(thr) or thr < 0:
        return True

    cost = extract_match_cost(match)
    if cost is None:
        return False
    return float(cost) <= float(thr)


def passes_matching_iou_filter(match: Dict[str, Any], min_iou: Optional[float] = None) -> bool:
    if min_iou is None:
        min_iou = _env_float("EDITABILITY_MIN_MATCHING_IOU", -1.0)
    try:
        thr = float(min_iou)
    except Exception:
        thr = -1.0
    if not math.isfinite(thr) or thr < 0:
        return True

    iou = extract_match_iou(match)
    if iou is None:
        return False
    return float(iou) >= float(thr)


def load_match_payloads(match_root: Path, model: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    bad_files = 0
    epi_dir = (match_root / model / "episodes")
    for p in sorted(epi_dir.glob("*.json")):
        try:
            payload = load_json(p)
        except Exception as e:
            bad_files += 1
            print(f"[warn][{model}] skip malformed match payload: {p.name} ({type(e).__name__})")
            continue
        if not isinstance(payload, dict):
            bad_files += 1
            print(f"[warn][{model}] skip non-dict match payload: {p.name}")
            continue
        if "episode_id" not in payload:
            bad_files += 1
            print(f"[warn][{model}] skip payload missing episode_id: {p.name}")
            continue
        out.append(payload)
    if bad_files > 0:
        print(
            f"[warn][{model}] ignored {bad_files} malformed payload(s) under {epi_dir}"
        )
    return out


def build_task_map(
    figma_data: Path,
    exp_pairs: Sequence[str],
    model: str,
    max_episodes: Optional[int] = None,
) -> Dict[str, EpisodeTask]:
    tasks = collect_episode_tasks(figma_data, exp_pairs, model=model, max_episodes=max_episodes)
    return {t.episode_id: t for t in tasks}


def is_text_gt(gt_elem: Dict[str, Any]) -> bool:
    gt_type = str(gt_elem.get("type", "")).lower()
    if gt_type == "text":
        return True
    unit = gt_elem.get("meta", {}).get("gt_unit", {})
    return str(unit.get("unit_type", "")).lower() == "text"


def is_rectangle_path_gt(gt_elem: Dict[str, Any]) -> bool:
    unit = gt_elem.get("meta", {}).get("gt_unit", {})
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


def has_stroke_gt(gt_elem: Dict[str, Any]) -> bool:
    unit = gt_elem.get("meta", {}).get("gt_unit", {})
    strokes = unit.get("strokes_raw")
    stroke_w = unit.get("stroke_weight")
    return bool(strokes) or (stroke_w is not None and float(stroke_w) > 0)


def _balanced_pick_text_nontext(
    cands: List[Candidate],
    cache: EpisodeCache,
    max_tasks: Optional[int],
    seed: int,
) -> List[Candidate]:
    if max_tasks is None:
        return cands

    text_c: List[Candidate] = []
    non_c: List[Candidate] = []
    for c in cands:
        c_is_text = c.gt_is_text
        if c_is_text is None:
            gt_elements, _, _ = cache.get(c.episode_id)
            gt = gt_elements[c.gt_index]
            c_is_text = is_text_gt(gt)
        if c_is_text:
            text_c.append(c)
        else:
            non_c.append(c)

    # Half-half target.
    half = max_tasks // 2
    text_s = sample_with_seed_balanced_by_key(
        text_c, key_fn=lambda c: c.episode_id, max_count=half, seed=seed
    )
    non_s = sample_with_seed_balanced_by_key(
        non_c, key_fn=lambda c: c.episode_id, max_count=max_tasks - len(text_s), seed=seed + 1
    )

    merged = text_s + non_s
    merged = sample_with_seed_balanced_by_key(
        merged, key_fn=lambda c: c.episode_id, max_count=max_tasks, seed=seed + 2
    )
    return merged


def _sample_diverse_by_param_and_episode(
    cands: List[Candidate],
    max_tasks: Optional[int],
    seed: int,
) -> List[Candidate]:
    if max_tasks is None or max_tasks >= len(cands):
        return cands

    def _param_key(c: Candidate) -> str:
        return f"{c.task_type}::{json.dumps(c.params, sort_keys=True, ensure_ascii=True)}"

    buckets: Dict[str, List[Candidate]] = {}
    for c in cands:
        buckets.setdefault(_param_key(c), []).append(c)

    keys = list(buckets.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    for k in list(keys):
        h = hashlib.md5(k.encode("utf-8")).hexdigest()
        kseed = int(h[:8], 16)
        buckets[k] = sample_with_seed_balanced_by_key(
            buckets[k], key_fn=lambda c: c.episode_id, max_count=None, seed=seed + (kseed % 997)
        )

    out: List[Candidate] = []
    while len(out) < max_tasks and keys:
        next_keys: List[str] = []
        for k in keys:
            b = buckets.get(k, [])
            if not b:
                continue
            out.append(b.pop())
            if len(out) >= max_tasks:
                break
            if b:
                next_keys.append(k)
        keys = next_keys
    return out


def sample_candidates(
    cands: List[Candidate],
    max_tasks: Optional[int],
    seed: int,
    balance_text_non_text: bool,
    cache: EpisodeCache,
) -> List[Candidate]:
    base = cands
    if balance_text_non_text:
        base = _balanced_pick_text_nontext(cands, cache, max_tasks, seed)
    if max_tasks is None:
        return base
    return _sample_diverse_by_param_and_episode(base, max_tasks=max_tasks, seed=seed)


def _to_stable_json(x: Any) -> str:
    try:
        return json.dumps(x, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        return json.dumps(str(x), ensure_ascii=True)


def candidate_row_key(
    episode_id: str,
    gt_index: int,
    pred_indices: Sequence[int],
    task_type: str,
    params: Optional[Dict[str, Any]],
) -> str:
    pred_key = ",".join(str(int(v)) for v in pred_indices)
    return (
        f"{str(episode_id)}"
        f"::gt{int(gt_index)}"
        f"::pred[{pred_key}]"
        f"::task={str(task_type)}"
        f"::params={_to_stable_json(params or {})}"
    )


def candidate_key(cand: Candidate) -> str:
    return candidate_row_key(
        episode_id=str(cand.episode_id),
        gt_index=int(cand.gt_index),
        pred_indices=[int(x) for x in cand.pred_indices],
        task_type=str(cand.task_type),
        params=dict(cand.params),
    )


def result_row_key(row: Dict[str, Any]) -> str:
    pred_indices = row.get("pred_indices")
    if isinstance(pred_indices, list):
        pred = [int(x) for x in pred_indices]
    else:
        p_idx = row.get("pred_index")
        pred = [int(p_idx)] if isinstance(p_idx, (int, float)) else []
    return candidate_row_key(
        episode_id=str(row.get("episode_id", "")),
        gt_index=int(row.get("gt_index", -1)),
        pred_indices=pred,
        task_type=str(row.get("task_type", "")),
        params=row.get("params") if isinstance(row.get("params"), dict) else {},
    )


def result_row_sort_key(row: Dict[str, Any]) -> Tuple[str, int, Tuple[int, ...], str, str]:
    pred_indices = row.get("pred_indices")
    if isinstance(pred_indices, list):
        pred = tuple(int(x) for x in pred_indices)
    else:
        p_idx = row.get("pred_index")
        pred = (int(p_idx),) if isinstance(p_idx, (int, float)) else tuple()
    params = row.get("params") if isinstance(row.get("params"), dict) else {}
    return (
        str(row.get("episode_id", "")),
        int(row.get("gt_index", -1)),
        pred,
        str(row.get("task_type", "")),
        _to_stable_json(params),
    )


def evaluate_image_edit_candidates(
    candidates: List[Candidate],
    cache: EpisodeCache,
    include_iou: bool = True,
    include_edge_sharpness: bool = False,
    include_lpips: bool = False,
    roi_mode: str = "source_target",
    roi_dilation_ratio: float = 0.08,
    progress_prefix: Optional[str] = None,
    log_every: int = 0,
    save_pair_viz_dir: Optional[Path] = None,
    pair_viz_max: Optional[int] = None,
    reference_metrics: Optional[List[Dict[str, Any]]] = None,
    reference_label: str = "qwen",
    model_label: str = "agent",
    num_workers: int = 1,
    show_tqdm: bool = True,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
) -> List[Dict[str, Any]]:
    total = len(candidates)

    if save_pair_viz_dir is not None:
        save_pair_viz_dir.mkdir(parents=True, exist_ok=True)

    existing_by_key: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path is not None and resume and checkpoint_path.exists():
        try:
            loaded = load_json(checkpoint_path)
            if isinstance(loaded, list):
                for row in loaded:
                    if not isinstance(row, dict):
                        continue
                    k = result_row_key(row)
                    if k:
                        existing_by_key[k] = row
        except Exception:
            # Ignore malformed checkpoints and continue fresh.
            existing_by_key = {}

    allowed_keys = {candidate_key(c) for c in candidates}
    if allowed_keys:
        existing_by_key = {k: v for k, v in existing_by_key.items() if k in allowed_keys}

    def _is_num(v: Any) -> bool:
        return isinstance(v, (int, float)) and (float(v) == float(v))

    run_lock = Lock()
    run_done = len(existing_by_key)
    run_sum: Dict[str, float] = {}
    run_cnt: Dict[str, int] = {}
    run_psnr_max: Dict[str, float] = {}

    def _accumulate(metrics: Dict[str, Any]) -> None:
        for k, v in metrics.items():
            vv = _normalize_metric_for_stats(k, v, psnr_max=run_psnr_max)
            if vv is not None:
                run_sum[k] = run_sum.get(k, 0.0) + float(vv)
                run_cnt[k] = run_cnt.get(k, 0) + 1

    for row in existing_by_key.values():
        mm = row.get("metrics", {})
        if isinstance(mm, dict):
            _accumulate(mm)

    _ckpt_interval = max(1, _env_int("EDITABILITY_CHECKPOINT_INTERVAL", 1))
    _ckpt_counter = [0]

    def _persist_checkpoint_locked(*, force: bool = False) -> None:
        if checkpoint_path is None:
            return
        _ckpt_counter[0] += 1
        if not force and _ckpt_counter[0] % _ckpt_interval != 0:
            return
        payload = sorted(existing_by_key.values(), key=result_row_sort_key)
        save_json(checkpoint_path, payload)

    def _build_metric_tokens(
        sums: Dict[str, float],
        cnts: Dict[str, int],
        keys: Sequence[str],
        *,
        metric_prefix: str = "",
    ) -> List[str]:
        xs: List[str] = []
        for k in keys:
            kk = f"{metric_prefix}{k}"
            c = cnts.get(kk, 0)
            if c > 0:
                xs.append(f"{k}={sums[kk] / c:.4f}")
        return xs

    def _build_running_msg(
        prefix: str,
        done: int,
        sums: Dict[str, float],
        cnts: Dict[str, int],
        *,
        model_label: str,
    ) -> str:
        transition_triplet = (
            any(k.startswith("source_") for k in sums.keys())
            and any(k.startswith("target_") for k in sums.keys())
            and any(k.startswith("avg_") for k in sums.keys())
        )
        base_keys = [
            "l1",
            "l2",
            "psnr",
            "ssim",
            "lpips",
            "dino",
            "iou",
            "edge_sharpness_gt",
            "edge_sharpness_pred",
        ]
        full_keys = ["full_l1", "full_l2", "full_psnr", "full_ssim", "full_lpips", "full_dino"]
        if transition_triplet:
            src = _build_metric_tokens(sums, cnts, base_keys, metric_prefix="source_")
            tgt = _build_metric_tokens(sums, cnts, base_keys, metric_prefix="target_")
            avg = _build_metric_tokens(sums, cnts, base_keys, metric_prefix="avg_")
            full = _build_metric_tokens(sums, cnts, full_keys)
            if not src:
                src = _build_metric_tokens(sums, cnts, sorted(sums.keys())[:6], metric_prefix="source_")
            if not tgt:
                tgt = _build_metric_tokens(sums, cnts, sorted(sums.keys())[:6], metric_prefix="target_")
            if not avg:
                avg = _build_metric_tokens(sums, cnts, sorted(sums.keys())[:6], metric_prefix="avg_")
            if not full:
                full = _build_metric_tokens(sums, cnts, sorted(sums.keys())[:6], metric_prefix="full_")
            return (
                f"{prefix} running_avg {done}/{total} {model_label}("
                f"source({' '.join(src)}) "
                f"target({' '.join(tgt)}) "
                f"avg({' '.join(avg)}) "
                f"full({' '.join(full)})"
                ")"
            )

        xs = _build_metric_tokens(sums, cnts, base_keys + full_keys)
        if not xs:
            for k in sorted(sums.keys())[:6]:
                c = cnts.get(k, 0)
                if c > 0:
                    xs.append(f"{k}={sums[k] / c:.4f}")
        return f"{prefix} running_avg {done}/{total} {model_label}({' '.join(xs)})"

    def _running_log(metrics: Dict[str, Any]) -> None:
        nonlocal run_done
        if log_every <= 0:
            return
        with run_lock:
            run_done += 1
            _accumulate(metrics)
            if not (run_done == 1 or run_done % log_every == 0 or run_done == total):
                return
            prefix = progress_prefix or "[eval]"
            msg = _build_running_msg(prefix, run_done, run_sum, run_cnt, model_label=model_label)

            if reference_metrics:
                n = min(run_done, len(reference_metrics))
                ref_sum: Dict[str, float] = {}
                ref_cnt: Dict[str, int] = {}
                ref_psnr_max: Dict[str, float] = {}
                for rr in reference_metrics[:n]:
                    if not isinstance(rr, dict):
                        continue
                    for k, v in rr.items():
                        vv = _normalize_metric_for_stats(k, v, psnr_max=ref_psnr_max)
                        if vv is not None:
                            ref_sum[k] = ref_sum.get(k, 0.0) + float(vv)
                            ref_cnt[k] = ref_cnt.get(k, 0) + 1
                ref_msg = _build_running_msg(prefix, run_done, ref_sum, ref_cnt, model_label=f"{reference_label}_prefix")
                cut = f"{prefix} running_avg {run_done}/{total} "
                if ref_msg.startswith(cut):
                    ref_msg = ref_msg[len(cut):]
                msg += f" {ref_msg}"

            print(msg)

    def _join_pred_indices(xs: Sequence[int]) -> str:
        if len(xs) <= 6:
            return "-".join(str(int(x)) for x in xs)
        head = "-".join(str(int(x)) for x in xs[:6])
        return f"{head}-n{len(xs)}"

    def _render_pair_panel(
        gt_before: np.ndarray,
        gt_after: np.ndarray,
        pred_before: np.ndarray,
        pred_after: np.ndarray,
    ) -> np.ndarray:
        pad = 4
        h, w = gt_before.shape[:2]
        spacer_v = np.full((h, pad, 4), 255, dtype=np.uint8)
        spacer_h = np.full((pad, w * 2 + pad, 4), 255, dtype=np.uint8)
        top = np.concatenate([gt_before, spacer_v, gt_after], axis=1)
        bot = np.concatenate([pred_before, spacer_v, pred_after], axis=1)
        return np.concatenate([top, spacer_h, bot], axis=0)

    def _eval_one(job: Tuple[int, Candidate]) -> Dict[str, Any]:
        i, cand = job
        gt_elements, pred_elements, canvas_size = cache.get(cand.episode_id)

        # Remap pred_indices using pred_ids if available (handles index mismatch
        # when matching was done at a different render_scale than evaluation).
        # Use local variable to avoid mutating cand (checkpoint keys use original indices).
        eval_pred_indices = cand.pred_indices
        if cand.pred_ids:
            id_to_idx = {e.get("id"): idx for idx, e in enumerate(pred_elements)}
            remapped = [id_to_idx[pid] for pid in cand.pred_ids if pid in id_to_idx]
            if remapped:
                eval_pred_indices = remapped

        edit = {"task_type": cand.task_type}
        edit.update(cand.params)

        pred_eval_scene = _apply_evalfigma_postprocess_to_scene(pred_elements, eval_pred_indices, canvas_size)

        need_viz = save_pair_viz_dir is not None and (pair_viz_max is None or i <= pair_viz_max)
        gt_before_rgba = None
        pred_before_rgba = None
        if need_viz:
            gt_before_rgba = render_scene_rgba(gt_elements, canvas_size)
            pred_before_rgba = render_scene_rgba(pred_eval_scene, canvas_size)

        gt_edit_scene = apply_edit_to_scene(gt_elements, [cand.gt_index], canvas_size, edit)
        pred_edit_scene = apply_edit_to_scene(pred_eval_scene, eval_pred_indices, canvas_size, edit)

        gt_rgba = render_scene_rgba(gt_edit_scene, canvas_size)
        pred_rgba = render_scene_rgba(pred_edit_scene, canvas_size)

        # Source/target masks: GT-only, Pred-only, and Union.
        gt_src = union_mask_from_indices(gt_elements, [cand.gt_index])
        gt_dst = union_mask_from_indices(gt_edit_scene, [cand.gt_index])
        pred_src = union_mask_from_indices(pred_eval_scene, eval_pred_indices)
        pred_dst = union_mask_from_indices(pred_edit_scene, eval_pred_indices)
        src_union = gt_src | pred_src
        dst_union = gt_dst | pred_dst

        metrics_source: Optional[Dict[str, Any]] = None
        metrics_target: Optional[Dict[str, Any]] = None
        metrics_avg: Optional[Dict[str, Any]] = None

        def _compute_roi_metrics(roi_mask, *, heavy=True):
            """Compute ROI metrics. heavy=False skips LPIPS/DINO for secondary ROIs."""
            return compute_region_metrics(
                gt_rgba, pred_rgba, roi_mask,
                include_iou=include_iou,
                include_edge_sharpness=include_edge_sharpness,
                include_lpips=include_lpips if heavy else False,
                include_dino=include_lpips if heavy else False,
            )

        def _avg_two_metrics(m1, m2):
            avg = {}
            for k in sorted(set(m1.keys()) | set(m2.keys())):
                sv, tv = m1.get(k), m2.get(k)
                if _is_num(sv) and _is_num(tv):
                    avg[k] = float((float(sv) + float(tv)) * 0.5)
                elif _is_num(sv):
                    avg[k] = float(sv)
                elif _is_num(tv):
                    avg[k] = float(tv)
                else:
                    avg[k] = float("nan")
            return avg

        if roi_mode == "transition_dual":
            # Union ROI (default)
            metrics_source = _compute_roi_metrics(src_union)
            metrics_target = _compute_roi_metrics(dst_union)
            metrics_avg = _avg_two_metrics(metrics_source, metrics_target)

            metrics = dict(metrics_avg)
            for k, v in metrics_source.items():
                metrics[f"source_{k}"] = v
            for k, v in metrics_target.items():
                metrics[f"target_{k}"] = v
            for k, v in metrics_avg.items():
                metrics[f"avg_{k}"] = v

            # GT-only ROI (lightweight: skip LPIPS/DINO)
            gt_source_m = _compute_roi_metrics(gt_src, heavy=False)
            gt_target_m = _compute_roi_metrics(gt_dst, heavy=False)
            gt_avg_m = _avg_two_metrics(gt_source_m, gt_target_m)
            for k, v in gt_avg_m.items():
                metrics[f"gt_{k}"] = v
            for k, v in gt_source_m.items():
                metrics[f"gt_source_{k}"] = v
            for k, v in gt_target_m.items():
                metrics[f"gt_target_{k}"] = v

            # Pred-only ROI (lightweight: skip LPIPS/DINO)
            pred_source_m = _compute_roi_metrics(pred_src, heavy=False)
            pred_target_m = _compute_roi_metrics(pred_dst, heavy=False)
            pred_avg_m = _avg_two_metrics(pred_source_m, pred_target_m)
            for k, v in pred_avg_m.items():
                metrics[f"pred_{k}"] = v
            for k, v in pred_source_m.items():
                metrics[f"pred_source_{k}"] = v
            for k, v in pred_target_m.items():
                metrics[f"pred_target_{k}"] = v

            roi = src_union | dst_union
        else:
            # Determine union ROI based on roi_mode
            if roi_mode == "source":
                roi_union = src_union
                roi_gt = gt_src
                roi_pred = pred_src
            elif roi_mode == "target":
                roi_union = dst_union
                roi_gt = gt_dst
                roi_pred = pred_dst
            else:
                roi_union = (src_union | dst_union)
                roi_gt = (gt_src | gt_dst)
                roi_pred = (pred_src | pred_dst)

            # Union ROI metrics (default, backward-compatible)
            metrics = _compute_roi_metrics(roi_union)

            # GT-only ROI metrics (lightweight: skip LPIPS/DINO)
            gt_metrics = _compute_roi_metrics(roi_gt, heavy=False)
            for k, v in gt_metrics.items():
                metrics[f"gt_{k}"] = v

            # Pred-only ROI metrics (lightweight: skip LPIPS/DINO)
            pred_metrics = _compute_roi_metrics(roi_pred, heavy=False)
            for k, v in pred_metrics.items():
                metrics[f"pred_{k}"] = v

            roi = roi_union

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

        result_row = {
            "episode_id": cand.episode_id,
            "gt_index": cand.gt_index,
            "pred_indices": cand.pred_indices,
            "task_type": cand.task_type,
            "params": cand.params,
            "applied_edit": edit,
            "metrics": metrics,
            "metrics_full": {k: v for k, v in metrics_full.items() if k in {"l1", "l2", "psnr", "ssim", "lpips", "dino"}},
        }
        if metrics_source is not None:
            result_row["metrics_source"] = metrics_source
        if metrics_target is not None:
            result_row["metrics_target"] = metrics_target
        if metrics_avg is not None:
            result_row["metrics_avg"] = metrics_avg
        _running_log(metrics)
        with run_lock:
            existing_by_key[result_row_key(result_row)] = result_row
            if checkpoint_path is not None:
                _persist_checkpoint_locked()

        if need_viz and gt_before_rgba is not None and pred_before_rgba is not None and save_pair_viz_dir is not None:
            pair_name = (
                f"{i:05d}__{cand.episode_id}"
                f"__gt{int(cand.gt_index)}"
                f"__pred{_join_pred_indices(cand.pred_indices)}"
            )
            pair_dir = save_pair_viz_dir / pair_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            panel = _render_pair_panel(gt_before_rgba, gt_rgba, pred_before_rgba, pred_rgba)
            Image.fromarray(panel, "RGBA").save(pair_dir / "panel.png")
            Image.fromarray(gt_before_rgba, "RGBA").save(pair_dir / "gt_before.png")
            Image.fromarray(gt_rgba, "RGBA").save(pair_dir / "gt_after.png")
            Image.fromarray(pred_before_rgba, "RGBA").save(pair_dir / "pred_before.png")
            Image.fromarray(pred_rgba, "RGBA").save(pair_dir / "pred_after.png")

            roi_vis = np.zeros((roi.shape[0], roi.shape[1], 4), dtype=np.uint8)
            roi_vis[..., 1] = 255
            roi_vis[..., 3] = (roi.astype(np.uint8) * 200)
            Image.fromarray(roi_vis, "RGBA").save(pair_dir / "roi.png")
            save_json(
                pair_dir / "meta.json",
                {
                    "episode_id": cand.episode_id,
                    "gt_index": cand.gt_index,
                    "pred_indices": cand.pred_indices,
                    "task_type": cand.task_type,
                    "params": cand.params,
                    "applied_edit": edit,
                    "metrics": metrics,
                    "metrics_full": {k: v for k, v in metrics_full.items() if k in {"l1", "l2", "psnr", "ssim", "lpips", "dino"}},
                },
            )
        return result_row

    ordered_candidates = sorted(
        candidates,
        key=lambda c: (str(c.episode_id), int(c.gt_index), tuple(int(x) for x in c.pred_indices)),
    )
    jobs = [(i, c) for i, c in enumerate(ordered_candidates, start=1) if candidate_key(c) not in existing_by_key]
    desc = progress_prefix or "[eval]"
    if jobs:
        _ = thread_map(
            jobs,
            _eval_one,
            num_workers=max(1, int(num_workers)),
            desc=desc,
            show_tqdm=show_tqdm,
        )
    elif checkpoint_path is not None and resume:
        prefix = progress_prefix or "[eval]"
        print(f"{prefix} resume: all {total} tasks already computed")

    final_rows = sorted(existing_by_key.values(), key=result_row_sort_key)
    if checkpoint_path is not None:
        save_json(checkpoint_path, final_rows)
    return final_rows


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, List[Dict[str, float]]] = {}
    for r in results:
        by_type.setdefault(r["task_type"], []).append(r["metrics"])

    out: Dict[str, Any] = {"total": len(results), "by_task_type": {}}
    for t, vals in by_type.items():
        keys = sorted({k for d in vals for k in d.keys()})
        mean = {}
        for k in keys:
            xs: List[float] = []
            inf_count = 0
            for d in vals:
                if k not in d:
                    continue
                v = d[k]
                if not isinstance(v, (int, float)):
                    continue
                fv = float(v)
                if fv != fv:
                    continue
                if _is_psnr_metric(k):
                    if math.isfinite(fv):
                        xs.append(fv)
                    elif fv > 0:
                        inf_count += 1
                elif math.isfinite(fv):
                    xs.append(fv)
            if _is_psnr_metric(k) and inf_count > 0:
                rep = max(xs) if xs else _psnr_inf_fallback()
                xs.extend([float(rep)] * int(inf_count))
            mean[k] = float(sum(xs) / len(xs)) if xs else float("nan")
        out["by_task_type"][t] = {"count": len(vals), "mean": mean}
    return out


def summarize_capacity(candidates: List[Candidate]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"total": len(candidates), "by_task_type": {}}
    for c in candidates:
        out["by_task_type"][c.task_type] = out["by_task_type"].get(c.task_type, 0) + 1
    return out


def compare_two_models(
    qwen_summary: Dict[str, Any],
    agent_summary: Dict[str, Any],
    metric_preferences: Dict[str, str],
) -> Dict[str, Any]:
    """metric_preferences: metric -> 'lower' or 'higher'"""
    out: Dict[str, Any] = {"by_task_type": {}}

    q_tasks = qwen_summary.get("by_task_type", {})
    a_tasks = agent_summary.get("by_task_type", {})
    all_task_names = sorted(set(q_tasks.keys()) | set(a_tasks.keys()))

    for t in all_task_names:
        q_mean = q_tasks.get(t, {}).get("mean", {})
        a_mean = a_tasks.get(t, {}).get("mean", {})
        metrics = sorted(set(q_mean.keys()) | set(a_mean.keys()))

        out["by_task_type"][t] = {"metrics": {}}
        for m in metrics:
            q = q_mean.get(m)
            a = a_mean.get(m)
            pref = metric_preferences.get(m, "lower")
            winner = None
            if isinstance(q, (int, float)) and isinstance(a, (int, float)):
                if q == q and a == a:
                    if pref == "lower":
                        winner = "agent" if a < q else ("qwen" if q < a else "tie")
                    else:
                        winner = "agent" if a > q else ("qwen" if q > a else "tie")
            out["by_task_type"][t]["metrics"][m] = {
                "qwen": q,
                "agent": a,
                "preference": pref,
                "winner": winner,
            }

    return out


def save_subtask_outputs(
    output_dir: Path,
    model: str,
    subtask_name: str,
    capacity: Dict[str, Any],
    sampled_count: int,
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    model_dir = output_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    save_json(model_dir / f"{subtask_name}_capacity.json", capacity)
    save_json(model_dir / f"{subtask_name}_results.json", results)
    wrapped_summary = {
        "model": model,
        "subtask": subtask_name,
        "capacity": capacity,
        "sampled_count": sampled_count,
        "summary": summary,
    }
    save_json(model_dir / f"{subtask_name}_summary.json", wrapped_summary)
