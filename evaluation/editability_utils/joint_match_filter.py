#!/usr/bin/env python3
"""Joint Qwen/Agent match filtering helpers for editability runners."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .common_utils import load_json, save_json
from .loaders import collect_episode_tasks, load_episode_elements
from .task_common import union_mask_from_indices

PairKey = Tuple[str, int]


def _finite_float(v: Any) -> Optional[float]:
    if not isinstance(v, (int, float)):
        return None
    fv = float(v)
    if fv != fv or not math.isfinite(fv):
        return None
    return fv


def extract_match_cost(match: Dict[str, Any]) -> Optional[float]:
    for key in ("best_single_metrics", "merged_metrics"):
        mm = match.get(key)
        if isinstance(mm, dict):
            fv = _finite_float(mm.get("cost"))
            if fv is not None:
                return fv
    return None


def extract_match_iou(match: Dict[str, Any]) -> Optional[float]:
    for key in ("merged_metrics", "best_single_metrics"):
        mm = match.get(key)
        if isinstance(mm, dict):
            fv = _finite_float(mm.get("iou"))
            if fv is not None:
                return fv
    return None


def extract_selected_pred_indices(match: Dict[str, Any]) -> List[int]:
    raw = match.get("selected_pred_indices", [])
    out: List[int] = []
    if isinstance(raw, list):
        for x in raw:
            try:
                xi = int(x)
            except Exception:
                continue
            if xi >= 0:
                out.append(xi)
    if out:
        return out
    b = match.get("best_single_pred_index")
    try:
        bi = int(b)
    except Exception:
        bi = -1
    return [bi] if bi >= 0 else []


def load_match_index(match_root: Path, model: str) -> Dict[PairKey, Dict[str, Any]]:
    out: Dict[PairKey, Dict[str, Any]] = {}
    epi_dir = match_root / model / "episodes"
    for p in sorted(epi_dir.glob("*.json")):
        try:
            payload = load_json(p)
        except Exception:
            continue
        eid = str(payload.get("episode_id", ""))
        if not eid:
            continue
        matches = payload.get("matches", [])
        if not isinstance(matches, list):
            continue
        for m in matches:
            if not isinstance(m, dict):
                continue
            gi = m.get("gt_index")
            if gi is None:
                continue
            try:
                gt_idx = int(gi)
            except Exception:
                continue
            out[(eid, gt_idx)] = m
    return out


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a).astype(bool)
    bb = np.asarray(b).astype(bool)
    if aa.shape != bb.shape:
        return 0.0
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    if union <= 0:
        return 0.0
    return float(inter / union)


def _parse_cached_iou(raw: Any) -> Dict[PairKey, float]:
    out: Dict[PairKey, float] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        if "::gt" not in k:
            continue
        eid, rhs = k.rsplit("::gt", 1)
        try:
            gt_idx = int(rhs)
        except Exception:
            continue
        fv = _finite_float(v)
        if fv is None:
            continue
        out[(eid, gt_idx)] = fv
    return out


def _to_cached_iou_dict(iou_map: Dict[PairKey, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for (eid, gt_idx), v in iou_map.items():
        fv = _finite_float(v)
        if fv is None:
            continue
        out[f"{eid}::gt{gt_idx}"] = fv
    return out


def compute_cross_model_pred_iou(
    *,
    figma_data: Path,
    exp_pairs: Sequence[str],
    keys: Set[PairKey],
    qwen_matches: Dict[PairKey, Dict[str, Any]],
    agent_matches: Dict[PairKey, Dict[str, Any]],
    cache_path: Optional[Path] = None,
    build_log_every: int = 0,
) -> Dict[PairKey, float]:
    cached: Dict[PairKey, float] = {}
    if cache_path is not None and cache_path.exists():
        try:
            cached = _parse_cached_iou(load_json(cache_path))
        except Exception:
            cached = {}

    pending = sorted(k for k in keys if k not in cached)
    if not pending:
        return {k: cached[k] for k in keys if k in cached}

    q_task_map = {t.episode_id: t for t in collect_episode_tasks(figma_data, exp_pairs, model="qwen")}
    a_task_map = {t.episode_id: t for t in collect_episode_tasks(figma_data, exp_pairs, model="agent")}

    by_episode: Dict[str, List[PairKey]] = defaultdict(list)
    for key in pending:
        by_episode[key[0]].append(key)

    done_eps = 0
    total_eps = len(by_episode)
    missing_task_eps = 0
    load_fail_eps = 0
    for eid in sorted(by_episode.keys()):
        done_eps += 1
        if eid not in q_task_map or eid not in a_task_map:
            missing_task_eps += 1
            continue
        try:
            _, q_pred, _ = load_episode_elements(q_task_map[eid], model="qwen")
            _, a_pred, _ = load_episode_elements(a_task_map[eid], model="agent")
        except Exception:
            load_fail_eps += 1
            continue

        for key in by_episode[eid]:
            q_match = qwen_matches.get(key)
            a_match = agent_matches.get(key)
            if not isinstance(q_match, dict) or not isinstance(a_match, dict):
                continue
            q_idx = extract_selected_pred_indices(q_match)
            a_idx = extract_selected_pred_indices(a_match)
            if not q_idx or not a_idx:
                continue
            q_mask = union_mask_from_indices(q_pred, q_idx)
            a_mask = union_mask_from_indices(a_pred, a_idx)
            cached[key] = _binary_iou(q_mask, a_mask)

        if build_log_every > 0 and (done_eps == 1 or done_eps % build_log_every == 0 or done_eps == total_eps):
            print(
                f"[joint-filter] cross-model IoU episode {done_eps}/{total_eps} "
                f"cache={len(cached)} missing_task_eps={missing_task_eps} load_fail_eps={load_fail_eps}"
            )
        if cache_path is not None and done_eps % 20 == 0:
            try:
                save_json(cache_path, _to_cached_iou_dict(cached))
            except Exception:
                pass

    if cache_path is not None:
        try:
            save_json(cache_path, _to_cached_iou_dict(cached))
        except Exception:
            pass
    return {k: cached[k] for k in keys if k in cached}


def apply_joint_subset_filter(
    *,
    match_root: Path,
    subset_keys: Optional[Set[PairKey]],
    max_matching_cost: float,
    min_matching_iou: float,
    min_cross_model_iou: float,
    figma_data: Optional[Path] = None,
    exp_pairs: Optional[Sequence[str]] = None,
    cross_iou_cache_path: Optional[Path] = None,
    build_log_every: int = 0,
) -> Tuple[Optional[Set[PairKey]], Dict[str, Any]]:
    qwen_idx = load_match_index(match_root, "qwen")
    agent_idx = load_match_index(match_root, "agent")
    q_keys = set(qwen_idx.keys())
    a_keys = set(agent_idx.keys())
    shared = q_keys & a_keys

    cost_enabled = bool(math.isfinite(float(max_matching_cost)) and float(max_matching_cost) >= 0.0)
    iou_enabled = bool(math.isfinite(float(min_matching_iou)) and float(min_matching_iou) >= 0.0)
    cross_enabled = bool(math.isfinite(float(min_cross_model_iou)) and float(min_cross_model_iou) >= 0.0)

    active = set(shared)
    q_cost_pass = set(q_keys)
    a_cost_pass = set(a_keys)
    if cost_enabled:
        thr = float(max_matching_cost)
        q_cost_pass = set()
        for k, m in qwen_idx.items():
            c = extract_match_cost(m)
            if c is not None and float(c) <= thr:
                q_cost_pass.add(k)
        a_cost_pass = set()
        for k, m in agent_idx.items():
            c = extract_match_cost(m)
            if c is not None and float(c) <= thr:
                a_cost_pass.add(k)
        active &= q_cost_pass & a_cost_pass

    q_iou_pass = set(q_keys)
    a_iou_pass = set(a_keys)
    if iou_enabled:
        thr = float(min_matching_iou)
        q_iou_pass = set()
        for k, m in qwen_idx.items():
            v = extract_match_iou(m)
            if v is not None and float(v) >= thr:
                q_iou_pass.add(k)
        a_iou_pass = set()
        for k, m in agent_idx.items():
            v = extract_match_iou(m)
            if v is not None and float(v) >= thr:
                a_iou_pass.add(k)
        active &= q_iou_pass & a_iou_pass

    if subset_keys is not None:
        active &= set(subset_keys)

    cross_iou_map: Dict[PairKey, float] = {}
    cross_pass: Set[PairKey] = set(active)
    if cross_enabled:
        if figma_data is None or exp_pairs is None:
            raise ValueError("figma_data and exp_pairs are required when min_cross_model_iou is enabled")
        cross_iou_map = compute_cross_model_pred_iou(
            figma_data=figma_data,
            exp_pairs=exp_pairs,
            keys=set(active),
            qwen_matches=qwen_idx,
            agent_matches=agent_idx,
            cache_path=cross_iou_cache_path,
            build_log_every=build_log_every,
        )
        if active and (not cross_iou_map):
            raise RuntimeError(
                "min_cross_model_iou is enabled but no cross-model IoU could be computed. "
                "Check --figma-data/--exp-pairs and runtime dependencies."
            )
        thr = float(min_cross_model_iou)
        cross_pass = {k for k, v in cross_iou_map.items() if float(v) >= thr}
        active &= cross_pass

    if subset_keys is None and (not cost_enabled) and (not iou_enabled) and (not cross_enabled):
        final_subset: Optional[Set[PairKey]] = None
    else:
        final_subset = active

    stats: Dict[str, Any] = {
        "qwen_total": len(q_keys),
        "agent_total": len(a_keys),
        "shared_total": len(shared),
        "cost_enabled": cost_enabled,
        "iou_enabled": iou_enabled,
        "cross_enabled": cross_enabled,
        "qwen_cost_pass": len(q_cost_pass),
        "agent_cost_pass": len(a_cost_pass),
        "qwen_iou_pass": len(q_iou_pass),
        "agent_iou_pass": len(a_iou_pass),
        "cross_iou_known": len(cross_iou_map),
        "cross_iou_pass": len(cross_pass) if cross_enabled else len(active),
        "final_subset": len(active) if final_subset is not None else None,
    }
    return final_subset, stats
