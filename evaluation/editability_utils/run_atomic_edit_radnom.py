#!/usr/bin/env python3
"""Run atomic subtasks with per-episode preselection shared across all subtasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from .common_utils import load_json, save_json
from .joint_match_filter import apply_joint_subset_filter, extract_selected_pred_indices, load_match_index
from .loaders import collect_episode_tasks, load_episode_elements
from .run_atomic_edit import _compact_compare_line, _metric_preferences, _run_subtask_paired
from .subset_manifest import load_subset_keys
from .subtasks.atomic import delete, opacity, recolor, rotation, transition, z_order
from .subtasks.common import compare_two_models

PairKey = Tuple[str, int]


def _pairkey_to_str(key: PairKey) -> str:
    return f"{str(key[0])}::gt{int(key[1])}"


def _pairkey_from_str(s: str) -> Optional[PairKey]:
    if not isinstance(s, str) or "::gt" not in s:
        return None
    eid, rhs = s.rsplit("::gt", 1)
    try:
        gi = int(rhs)
    except Exception:
        return None
    return (str(eid), int(gi))


def _load_foreground_cache(path: Path) -> Dict[PairKey, bool]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except Exception:
        return {}

    data = payload.get("triplet_full_foreground") if isinstance(payload, dict) else payload
    out: Dict[PairKey, bool] = {}
    if not isinstance(data, dict):
        return out
    for k, v in data.items():
        pk = _pairkey_from_str(k)
        if pk is None:
            continue
        out[pk] = bool(v)
    return out


def _save_foreground_cache(path: Path, values: Dict[PairKey, bool]) -> None:
    payload = {
        "version": 1,
        "triplet_full_foreground": {
            _pairkey_to_str(k): bool(v)
            for k, v in sorted(values.items(), key=lambda kv: (str(kv[0][0]), int(kv[0][1])))
        },
    }
    save_json(path, payload)


def _selected_pred_ids_from_match(match: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    raw = match.get("selected_pred_ids")
    if isinstance(raw, list):
        for x in raw:
            sx = str(x)
            if sx:
                out.append(sx)
    if out:
        return out
    b = match.get("best_single_pred_id")
    if b is not None:
        sb = str(b)
        if sb:
            return [sb]
    return []


_QWEN_ID_RE = re.compile(r"^qwen_L(\d+)_C(\d+)$")


def _qwen_z_from_pred_id(pred_id: str) -> Optional[int]:
    m = _QWEN_ID_RE.match(str(pred_id))
    if not m:
        return None
    try:
        layer = int(m.group(1))
        comp = int(m.group(2))
    except Exception:
        return None
    return int(layer * 10000 + comp)


def _compute_agent_layer_order(history_tree: Dict[str, Any], root_id: str = "layer_0000") -> List[str]:
    z_order: List[str] = []
    visited: Set[str] = set()

    def dfs(layer_id: str) -> None:
        if layer_id in visited:
            return
        visited.add(layer_id)

        node = history_tree.get(layer_id)
        if not isinstance(node, dict):
            return

        action_type = node.get("action_type")
        children = node.get("children_ids") or []
        real_children = [str(c) for c in children if isinstance(c, str) and not str(c).startswith("_temp_")]

        if action_type in ["Finalize_Text", "Finalize_Obj"]:
            z_order.append(layer_id)
            return
        if action_type == "Discard":
            return
        for child_id in real_children:
            dfs(child_id)

    dfs(root_id)
    return z_order


def _load_agent_pred_z_map(agent_episode_dir: Path) -> Dict[str, int]:
    parse_path = agent_episode_dir / "parse.json"
    hist_path = agent_episode_dir / "history_tree.json"
    if (not parse_path.exists()) or (not hist_path.exists()):
        return {}
    try:
        parse_data = load_json(parse_path)
        history_tree = load_json(hist_path)
    except Exception:
        return {}
    if not isinstance(parse_data, dict) or not isinstance(history_tree, dict):
        return {}

    layer_order = _compute_agent_layer_order(history_tree)
    layer_to_z = {str(layer_id): int(idx) for idx, layer_id in enumerate(layer_order)}

    out: Dict[str, int] = {}
    elems = parse_data.get("elements")
    if not isinstance(elems, list):
        return out
    for idx, elem in enumerate(elems):
        if not isinstance(elem, dict):
            continue
        core_id = str(elem.get("id", ""))
        if not core_id:
            continue
        source_layer = str(elem.get("source_layer_id", ""))
        z = int(layer_to_z.get(source_layer, idx))
        out[f"agent_{core_id}"] = z
        out[core_id] = z
    return out


def _load_gt_z_meta(gt_json_path: Path) -> Tuple[Dict[str, int], Optional[int]]:
    try:
        frame = load_json(gt_json_path)
    except Exception:
        return {}, None
    if not isinstance(frame, dict):
        return {}, None

    zmap: Dict[str, int] = {}
    zvals: List[int] = []
    if frame.get("expanded_background_path"):
        zmap["gt_background"] = -1
        zvals.append(-1)

    units = frame.get("unit_images")
    if isinstance(units, list):
        for idx, u in enumerate(units):
            if not isinstance(u, dict):
                continue
            uid = u.get("unit_id")
            if uid is None:
                continue
            z = int(u.get("z_index", idx))
            zmap[f"gt_{uid}"] = z
            zvals.append(z)
    min_z = min(zvals) if zvals else None
    return zmap, min_z


def _episode_seed(seed: int, episode_id: str) -> int:
    token = f"{int(seed)}::{str(episode_id)}"
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def _sample_keys_per_episode(
    keys: Set[PairKey],
    *,
    per_episode: int,
    seed: int,
) -> Set[PairKey]:
    if per_episode <= 0:
        return set()

    by_episode: Dict[str, List[int]] = defaultdict(list)
    for eid, gt_idx in sorted(keys):
        by_episode[str(eid)].append(int(gt_idx))

    selected: Set[PairKey] = set()
    for eid in sorted(by_episode.keys()):
        candidates = sorted(set(by_episode[eid]))
        if not candidates:
            continue

        rng = random.Random(_episode_seed(seed, eid))
        rng.shuffle(candidates)
        keep_n = min(int(per_episode), len(candidates))
        for gt_idx in candidates[:keep_n]:
            selected.add((eid, int(gt_idx)))
    return selected


def _build_base_selection_keys(
    *,
    match_root: Path,
    figma_data: Path,
    exp_pairs: Sequence[str],
    subset_keys: Optional[Set[PairKey]],
    max_matching_cost: float,
    min_matching_iou: float,
    min_cross_model_iou: float,
    build_log_every: int,
) -> Tuple[Set[PairKey], Dict[str, Any]]:
    filtered_keys, joint_stats = apply_joint_subset_filter(
        match_root=match_root,
        figma_data=figma_data,
        exp_pairs=exp_pairs,
        max_matching_cost=float(max_matching_cost),
        min_matching_iou=float(min_matching_iou),
        min_cross_model_iou=float(min_cross_model_iou),
        subset_keys=subset_keys,
        cross_iou_cache_path=match_root / "_cache" / "qwen_agent_pred_iou.json",
        build_log_every=max(0, int(build_log_every)),
    )

    if filtered_keys is not None:
        return set(filtered_keys), joint_stats

    qwen_idx = load_match_index(match_root, "qwen")
    agent_idx = load_match_index(match_root, "agent")
    shared = set(qwen_idx.keys()) & set(agent_idx.keys())
    return shared, joint_stats


def _apply_max_episodes_cap(
    keys: Set[PairKey],
    *,
    figma_data: Path,
    exp_pairs: Sequence[str],
    max_episodes: Optional[int],
) -> Set[PairKey]:
    if max_episodes is None:
        return set(keys)

    q_tasks = collect_episode_tasks(figma_data, exp_pairs, model="qwen", max_episodes=max_episodes)
    a_tasks = collect_episode_tasks(figma_data, exp_pairs, model="agent", max_episodes=max_episodes)
    allowed = {t.episode_id for t in q_tasks} & {t.episode_id for t in a_tasks}
    return {(eid, gt_idx) for eid, gt_idx in keys if eid in allowed}


def _mask_bool_from_element(elem: Dict[str, Any], canvas_size: Tuple[int, int]) -> np.ndarray:
    w, h = int(canvas_size[0]), int(canvas_size[1])
    mask = elem.get("mask")
    if isinstance(mask, np.ndarray):
        m = np.array(mask, copy=False)
        if m.shape != (h, w):
            try:
                if m.dtype != np.uint8:
                    if float(np.max(m)) <= 1.0:
                        m = np.clip(m * 255.0, 0, 255).astype(np.uint8)
                    else:
                        m = np.clip(m, 0, 255).astype(np.uint8)
                pil = Image.fromarray(m, "L")
                m = np.array(pil.resize((w, h), Image.BILINEAR), dtype=np.uint8)
            except Exception:
                m = np.zeros((h, w), dtype=np.uint8)
        return m > 0

    img = elem.get("image")
    if img is None:
        return np.zeros((h, w), dtype=bool)
    try:
        rgba = img.convert("RGBA")
        if rgba.size != (w, h):
            rgba = rgba.resize((w, h), Image.LANCZOS)
        return np.array(rgba.getchannel("A"), dtype=np.uint8) > 0
    except Exception:
        return np.zeros((h, w), dtype=bool)


def _build_full_foreground_checker(
    masks: List[np.ndarray],
    z_order: List[int],
):
    n = len(masks)
    order = sorted(range(n), key=lambda i: (int(z_order[i]), int(i)))
    memo: Dict[int, bool] = {}

    def _check(idx_raw: int) -> bool:
        idx = int(idx_raw)
        if idx < 0 or idx >= n:
            return False
        if idx in memo:
            return bool(memo[idx])

        tgt = masks[idx]
        if not bool(tgt.any()):
            memo[idx] = False
            return False

        z_t = int(z_order[idx])
        behind = np.zeros_like(tgt, dtype=bool)
        for j in order:
            if int(z_order[j]) >= z_t:
                break
            behind |= masks[j]
            if bool(np.all(behind[tgt])):
                memo[idx] = True
                return True
        memo[idx] = False
        return False

    return _check


def _sample_full_foreground_keys_per_episode(
    keys: Set[PairKey],
    *,
    per_episode: int,
    seed: int,
    match_root: Path,
    figma_data: Path,
    exp_pairs: Sequence[str],
    cache_path: Optional[Path] = None,
    use_existing_cache: bool = True,
    build_log_every: int = 0,
) -> Tuple[Set[PairKey], Dict[str, Any]]:
    selected: Set[PairKey] = set()
    fg_cache: Dict[PairKey, bool] = (
        _load_foreground_cache(cache_path) if (cache_path is not None and use_existing_cache) else {}
    )
    cache_dirty = False
    stats: Dict[str, Any] = {
        "enabled": True,
        "mode": "sample_after_foreground_check",
        "input_pairs": len(keys),
        "input_episodes": len({eid for eid, _ in keys}),
        "per_episode_elements": int(per_episode),
        "evaluated_pairs": 0,
        "skipped_after_quota": 0,
        "cache_hits": 0,
        "cache_miss": 0,
        "drop_cached_not_full_foreground": 0,
        "drop_missing_task": 0,
        "drop_load_fail": 0,
        "drop_missing_match": 0,
        "drop_gt_not_full_foreground": 0,
        "drop_qwen_not_full_foreground": 0,
        "drop_agent_not_full_foreground": 0,
        "used_existing_cache": bool(use_existing_cache),
    }
    if per_episode <= 0 or not keys:
        stats["selected_pairs"] = 0
        stats["selected_episodes"] = 0
        return selected, stats

    q_idx = load_match_index(match_root, "qwen")
    a_idx = load_match_index(match_root, "agent")
    q_tasks = {t.episode_id: t for t in collect_episode_tasks(figma_data, exp_pairs, model="qwen")}
    a_tasks = {t.episode_id: t for t in collect_episode_tasks(figma_data, exp_pairs, model="agent")}

    by_episode: Dict[str, List[int]] = defaultdict(list)
    for eid, gt_idx in sorted(keys):
        by_episode[str(eid)].append(int(gt_idx))

    done_eps = 0
    total_eps = len(by_episode)
    for eid in sorted(by_episode.keys()):
        done_eps += 1
        candidates = sorted(set(int(x) for x in by_episode[eid]))
        if not candidates:
            continue
        rng = random.Random(_episode_seed(seed, eid))
        rng.shuffle(candidates)
        quota = min(int(per_episode), len(candidates))

        if eid not in q_tasks or eid not in a_tasks:
            stats["drop_missing_task"] += len(candidates)
            continue

        loaded = False
        load_failed = False
        gt_check = None
        q_check = None
        a_check = None

        picked = 0
        for ci, gt_idx in enumerate(candidates):
            if picked >= quota:
                stats["skipped_after_quota"] += len(candidates) - ci
                break

            key = (eid, int(gt_idx))
            cached = fg_cache.get(key)
            if cached is not None:
                stats["cache_hits"] += 1
                if not bool(cached):
                    stats["drop_cached_not_full_foreground"] += 1
                    continue
            else:
                stats["cache_miss"] += 1
                stats["evaluated_pairs"] += 1

                q_match = q_idx.get(key)
                a_match = a_idx.get(key)
                if not isinstance(q_match, dict) or not isinstance(a_match, dict):
                    stats["drop_missing_match"] += 1
                    continue

                if load_failed:
                    stats["drop_load_fail"] += 1
                    continue

                if not loaded:
                    try:
                        q_gt, q_pred, q_canvas = load_episode_elements(q_tasks[eid], model="qwen")
                        a_gt, a_pred, a_canvas = load_episode_elements(a_tasks[eid], model="agent")
                    except Exception:
                        load_failed = True
                        stats["drop_load_fail"] += 1
                        continue

                    if len(q_gt) >= len(a_gt):
                        gt_elements = q_gt
                        gt_canvas = q_canvas
                    else:
                        gt_elements = a_gt
                        gt_canvas = a_canvas

                    gt_masks = [_mask_bool_from_element(e, gt_canvas) for e in gt_elements]
                    q_masks = [_mask_bool_from_element(e, q_canvas) for e in q_pred]
                    a_masks = [_mask_bool_from_element(e, a_canvas) for e in a_pred]

                    gt_check = _build_full_foreground_checker(
                        gt_masks,
                        [int(e.get("z_index", i)) for i, e in enumerate(gt_elements)],
                    )
                    q_check = _build_full_foreground_checker(
                        q_masks,
                        [int(e.get("z_index", i)) for i, e in enumerate(q_pred)],
                    )
                    a_check = _build_full_foreground_checker(
                        a_masks,
                        [int(e.get("z_index", i)) for i, e in enumerate(a_pred)],
                    )
                    loaded = True

                q_pred_idx = extract_selected_pred_indices(q_match)
                a_pred_idx = extract_selected_pred_indices(a_match)

                ok = True
                if gt_check is None or not gt_check(gt_idx):
                    stats["drop_gt_not_full_foreground"] += 1
                    ok = False
                elif q_check is None or not all(q_check(int(i)) for i in q_pred_idx):
                    stats["drop_qwen_not_full_foreground"] += 1
                    ok = False
                elif a_check is None or not all(a_check(int(i)) for i in a_pred_idx):
                    stats["drop_agent_not_full_foreground"] += 1
                    ok = False

                fg_cache[key] = bool(ok)
                cache_dirty = True
                if not ok:
                    continue

            selected.add(key)
            picked += 1

        if build_log_every > 0 and (done_eps == 1 or done_eps % build_log_every == 0 or done_eps == total_eps):
            print(
                "[foreground-filter] "
                f"episode {done_eps}/{total_eps} selected={len(selected)} evaluated={stats['evaluated_pairs']} "
                f"skip_after_quota={stats['skipped_after_quota']} "
                f"cache_hits={stats['cache_hits']} cache_miss={stats['cache_miss']} "
                f"drop_missing_task={stats['drop_missing_task']} drop_load_fail={stats['drop_load_fail']} "
                f"drop_missing_match={stats['drop_missing_match']} "
                f"drop_gt={stats['drop_gt_not_full_foreground']} "
                f"drop_qwen={stats['drop_qwen_not_full_foreground']} "
                f"drop_agent={stats['drop_agent_not_full_foreground']}"
            )

        if cache_path is not None and cache_dirty and done_eps % 20 == 0:
            try:
                _save_foreground_cache(cache_path, fg_cache)
                cache_dirty = False
            except Exception:
                pass

    stats["selected_pairs"] = len(selected)
    stats["selected_episodes"] = len({eid for eid, _ in selected})
    if cache_path is not None:
        stats["cache_path"] = str(cache_path)
        stats["cache_size"] = len(fg_cache)
        if cache_dirty:
            try:
                _save_foreground_cache(cache_path, fg_cache)
            except Exception:
                pass
    return selected, stats


def _sample_front_zorder_keys_per_episode(
    keys: Set[PairKey],
    *,
    per_episode: int,
    seed: int,
    match_root: Path,
    figma_data: Path,
    exp_pairs: Sequence[str],
    build_log_every: int = 0,
) -> Tuple[Set[PairKey], Dict[str, Any]]:
    selected: Set[PairKey] = set()
    stats: Dict[str, Any] = {
        "enabled": True,
        "mode": "z_order_front_only",
        "input_pairs": len(keys),
        "input_episodes": len({eid for eid, _ in keys}),
        "per_episode_elements": int(per_episode),
        "evaluated_pairs": 0,
        "skipped_after_quota": 0,
        "drop_missing_task": 0,
        "drop_missing_match": 0,
        "drop_gt_backmost": 0,
        "drop_qwen_backmost": 0,
        "drop_agent_backmost": 0,
    }
    if per_episode <= 0 or not keys:
        stats["selected_pairs"] = 0
        stats["selected_episodes"] = 0
        return selected, stats

    q_idx = load_match_index(match_root, "qwen")
    a_idx = load_match_index(match_root, "agent")
    q_tasks = {t.episode_id: t for t in collect_episode_tasks(figma_data, exp_pairs, model="qwen")}
    a_tasks = {t.episode_id: t for t in collect_episode_tasks(figma_data, exp_pairs, model="agent")}

    by_episode: Dict[str, List[int]] = defaultdict(list)
    for eid, gt_idx in sorted(keys):
        by_episode[str(eid)].append(int(gt_idx))

    gt_meta_cache: Dict[str, Tuple[Dict[str, int], Optional[int]]] = {}
    agent_z_cache: Dict[str, Tuple[Dict[str, int], Optional[int]]] = {}
    qwen_min_z_cache: Dict[str, Optional[int]] = {}

    done_eps = 0
    total_eps = len(by_episode)
    for eid in sorted(by_episode.keys()):
        done_eps += 1
        candidates = sorted(set(int(x) for x in by_episode[eid]))
        if not candidates:
            continue
        rng = random.Random(_episode_seed(seed, eid))
        rng.shuffle(candidates)
        quota = min(int(per_episode), len(candidates))

        q_task = q_tasks.get(eid)
        a_task = a_tasks.get(eid)
        if q_task is None or a_task is None:
            stats["drop_missing_task"] += len(candidates)
            continue

        if eid not in gt_meta_cache:
            gt_meta_cache[eid] = _load_gt_z_meta(q_task.gt_json_path)
        gt_zmap, gt_min_z = gt_meta_cache[eid]

        if eid not in agent_z_cache:
            if a_task.agent_episode_dir is not None:
                az = _load_agent_pred_z_map(a_task.agent_episode_dir)
            else:
                az = {}
            agent_z_cache[eid] = (az, (min(az.values()) if az else None))
        agent_z_map, agent_min_z = agent_z_cache[eid]

        if eid not in qwen_min_z_cache:
            q_z_vals: List[int] = []
            for gt_idx in candidates:
                m = q_idx.get((eid, int(gt_idx)))
                if not isinstance(m, dict):
                    continue
                for pid in _selected_pred_ids_from_match(m):
                    z = _qwen_z_from_pred_id(pid)
                    if z is not None:
                        q_z_vals.append(int(z))
            qwen_min_z_cache[eid] = min(q_z_vals) if q_z_vals else None
        qwen_min_z = qwen_min_z_cache[eid]

        picked = 0
        for ci, gt_idx in enumerate(candidates):
            if picked >= quota:
                stats["skipped_after_quota"] += len(candidates) - ci
                break

            stats["evaluated_pairs"] += 1
            key = (eid, int(gt_idx))
            q_match = q_idx.get(key)
            a_match = a_idx.get(key)
            if not isinstance(q_match, dict) or not isinstance(a_match, dict):
                stats["drop_missing_match"] += 1
                continue

            gt_id = str(q_match.get("gt_id", ""))
            gt_z = gt_zmap.get(gt_id)
            if gt_z is None:
                gt_z = gt_zmap.get(str(a_match.get("gt_id", "")))
            if gt_z is None or gt_min_z is None or int(gt_z) <= int(gt_min_z):
                stats["drop_gt_backmost"] += 1
                continue

            q_ids = _selected_pred_ids_from_match(q_match)
            if qwen_min_z is None or not q_ids:
                stats["drop_qwen_backmost"] += 1
                continue
            q_ok = True
            for pid in q_ids:
                z = _qwen_z_from_pred_id(pid)
                if z is None or int(z) <= int(qwen_min_z):
                    q_ok = False
                    break
            if not q_ok:
                stats["drop_qwen_backmost"] += 1
                continue

            a_ids = _selected_pred_ids_from_match(a_match)
            if agent_min_z is None or not a_ids:
                stats["drop_agent_backmost"] += 1
                continue
            a_ok = True
            for pid in a_ids:
                z = agent_z_map.get(str(pid))
                if z is None or int(z) <= int(agent_min_z):
                    a_ok = False
                    break
            if not a_ok:
                stats["drop_agent_backmost"] += 1
                continue

            selected.add(key)
            picked += 1

        if build_log_every > 0 and (done_eps == 1 or done_eps % build_log_every == 0 or done_eps == total_eps):
            print(
                "[foreground-filter-z] "
                f"episode {done_eps}/{total_eps} selected={len(selected)} evaluated={stats['evaluated_pairs']} "
                f"skip_after_quota={stats['skipped_after_quota']} "
                f"drop_missing_task={stats['drop_missing_task']} "
                f"drop_missing_match={stats['drop_missing_match']} "
                f"drop_gt_back={stats['drop_gt_backmost']} "
                f"drop_q_back={stats['drop_qwen_backmost']} "
                f"drop_a_back={stats['drop_agent_backmost']}"
            )

    stats["selected_pairs"] = len(selected)
    stats["selected_episodes"] = len({eid for eid, _ in selected})
    return selected, stats


def _stable_params_json(row: Dict[str, Any]) -> str:
    params = row.get("params", {})
    if not isinstance(params, dict):
        params = {}
    return json.dumps(params, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _triplet_task_key(row: Dict[str, Any], default_task: str) -> str:
    return (
        f"{str(row.get('episode_id', ''))}"
        f"::gt{int(row.get('gt_index', -1))}"
        f"::task={str(row.get('task_type', default_task))}"
        f"::params={_stable_params_json(row)}"
    )


def _build_pair_meta_index(pair_root: Path, default_task: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not pair_root.exists():
        return out
    for d in sorted(pair_root.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = load_json(meta_path)
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        k = _triplet_task_key(meta, default_task=default_task)
        if k not in out:
            out[k] = d
    return out


def _load_result_rows_checkpoint(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return []
    if not raw:
        return []

    def _as_rows(x: Any) -> List[Dict[str, Any]]:
        if isinstance(x, list):
            return [r for r in x if isinstance(r, dict)]
        if isinstance(x, dict):
            rows = x.get("results", [])
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        return []

    try:
        data = json.loads(raw)
        return _as_rows(data)
    except json.JSONDecodeError:
        # Live checkpoint can be partially written; parse complete top-level objects only.
        text = raw.lstrip()
        if not text.startswith("["):
            return []
        out: List[Dict[str, Any]] = []
        dec = json.JSONDecoder()
        i = raw.find("[") + 1
        n = len(raw)
        while i < n:
            while i < n and raw[i] in " \t\r\n,":
                i += 1
            if i >= n or raw[i] == "]":
                break
            try:
                obj, j = dec.raw_decode(raw, i)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                out.append(obj)
            i = j
        return out


def _fit_rgba_on_white(img: Image.Image, width: int, height: int) -> Image.Image:
    rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    scale = min(width / float(rgb.width), height / float(rgb.height))
    nw = max(1, int(round(rgb.width * scale)))
    nh = max(1, int(round(rgb.height * scale)))
    resized = rgb.resize((nw, nh), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (width, height), color=(255, 255, 255))
    out.paste(resized, ((width - nw) // 2, (height - nh) // 2))
    return out


def _load_cell_or_placeholder(
    img_path: Optional[Path],
    width: int,
    height: int,
    title: str,
) -> Image.Image:
    if img_path is not None and img_path.exists():
        try:
            return _fit_rgba_on_white(Image.open(img_path), width, height)
        except Exception:
            pass
    ph = Image.new("RGB", (width, height), color=(236, 236, 236))
    return ph


def _render_triplet_2x3(
    out_path: Path,
    *,
    subtask: str,
    row_q: Dict[str, Any],
    row_a: Dict[str, Any],
    q_dir: Optional[Path],
    a_dir: Optional[Path],
    cell_w: int,
    cell_h: int,
) -> None:
    from PIL import ImageDraw, ImageFont

    q_gt_before = (q_dir / "gt_before.png") if q_dir is not None else None
    q_gt_after = (q_dir / "gt_after.png") if q_dir is not None else None
    q_pred_before = (q_dir / "pred_before.png") if q_dir is not None else None
    q_pred_after = (q_dir / "pred_after.png") if q_dir is not None else None
    a_gt_before = (a_dir / "gt_before.png") if a_dir is not None else None
    a_gt_after = (a_dir / "gt_after.png") if a_dir is not None else None
    a_pred_before = (a_dir / "pred_before.png") if a_dir is not None else None
    a_pred_after = (a_dir / "pred_after.png") if a_dir is not None else None

    gt_before = q_gt_before if (q_gt_before and q_gt_before.exists()) else a_gt_before
    gt_after = q_gt_after if (q_gt_after and q_gt_after.exists()) else a_gt_after

    cells = [
        _load_cell_or_placeholder(gt_before, cell_w, cell_h, "GT before"),
        _load_cell_or_placeholder(q_pred_before, cell_w, cell_h, "Qwen before"),
        _load_cell_or_placeholder(a_pred_before, cell_w, cell_h, "Agent before"),
        _load_cell_or_placeholder(gt_after, cell_w, cell_h, "GT after"),
        _load_cell_or_placeholder(q_pred_after, cell_w, cell_h, "Qwen after"),
        _load_cell_or_placeholder(a_pred_after, cell_w, cell_h, "Agent after"),
    ]

    margin = 16
    gap = 10
    header_h = 54
    label_h = 18
    canvas_w = margin * 2 + cell_w * 3 + gap * 2
    canvas_h = margin * 2 + header_h + label_h + cell_h * 2 + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    eid = str(row_q.get("episode_id", ""))
    gt_idx = int(row_q.get("gt_index", -1))
    q_l1 = row_q.get("metrics", {}).get("l1") if isinstance(row_q.get("metrics"), dict) else None
    q_l2 = row_q.get("metrics", {}).get("l2") if isinstance(row_q.get("metrics"), dict) else None
    a_l1 = row_a.get("metrics", {}).get("l1") if isinstance(row_a.get("metrics"), dict) else None
    a_l2 = row_a.get("metrics", {}).get("l2") if isinstance(row_a.get("metrics"), dict) else None
    draw.text(
        (margin, margin),
        f"subtask={subtask} | episode={eid} | gt={gt_idx}",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text(
        (margin, margin + 18),
        f"l1 q={q_l1} a={a_l1} | l2 q={q_l2} a={a_l2}",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text(
        (margin, margin + 36),
        f"params={_stable_params_json(row_q)}",
        fill=(0, 0, 0),
        font=font,
    )

    labels = ["GT", "Qwen", "Agent"]
    y0 = margin + header_h
    for c in range(3):
        x = margin + c * (cell_w + gap)
        draw.text((x + 4, y0), labels[c], fill=(0, 0, 0), font=font)

    y1 = y0 + label_h
    y2 = y1 + cell_h + gap
    for c in range(3):
        x = margin + c * (cell_w + gap)
        top_img = cells[c]
        bot_img = cells[3 + c]
        canvas.paste(top_img, (x, y1))
        canvas.paste(bot_img, (x, y2))
        draw.rectangle([x, y1, x + cell_w - 1, y1 + cell_h - 1], outline=(90, 90, 90), width=2)
        draw.rectangle([x, y2, x + cell_w - 1, y2 + cell_h - 1], outline=(90, 90, 90), width=2)
        draw.text((x + 4, y1 + 4), "Before", fill=(30, 30, 30), font=font)
        draw.text((x + 4, y2 + 4), "After", fill=(30, 30, 30), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _save_triplet_visualizations_for_subtask(
    *,
    output_dir: Path,
    subtask_name: str,
    q_payload: Dict[str, Any],
    a_payload: Dict[str, Any],
    cell_w: int,
    cell_h: int,
    max_cases: Optional[int],
) -> Dict[str, Any]:
    task_name = f"atomic_{subtask_name}"
    q_rows = q_payload.get("results", []) if isinstance(q_payload, dict) else []
    a_rows = a_payload.get("results", []) if isinstance(a_payload, dict) else []
    if not isinstance(q_rows, list):
        q_rows = []
    if not isinstance(a_rows, list):
        a_rows = []
    q_rows = [x for x in q_rows if isinstance(x, dict)]
    a_rows = [x for x in a_rows if isinstance(x, dict)]

    q_map = {_triplet_task_key(r, default_task=subtask_name): r for r in q_rows}
    a_map = {_triplet_task_key(r, default_task=subtask_name): r for r in a_rows}
    keys = sorted(set(q_map.keys()) & set(a_map.keys()))
    if max_cases is not None:
        keys = keys[: max(0, int(max_cases))]

    q_pair_idx = _build_pair_meta_index(
        output_dir / "qwen" / task_name / "element_pairs",
        default_task=subtask_name,
    )
    a_pair_idx = _build_pair_meta_index(
        output_dir / "agent" / task_name / "element_pairs",
        default_task=subtask_name,
    )

    viz_root = output_dir / "triplet_pair_viz" / task_name
    created = 0
    missing_q_pair = 0
    missing_a_pair = 0

    for rank, k in enumerate(keys, start=1):
        row_q = q_map[k]
        row_a = a_map[k]
        q_dir = q_pair_idx.get(k)
        a_dir = a_pair_idx.get(k)
        if q_dir is None:
            missing_q_pair += 1
        if a_dir is None:
            missing_a_pair += 1
        eid = str(row_q.get("episode_id", ""))
        gt_idx = int(row_q.get("gt_index", -1))
        out_name = f"{rank:05d}__{eid}__gt{gt_idx}.png"
        _render_triplet_2x3(
            viz_root / out_name,
            subtask=subtask_name,
            row_q=row_q,
            row_a=row_a,
            q_dir=q_dir,
            a_dir=a_dir,
            cell_w=int(cell_w),
            cell_h=int(cell_h),
        )
        created += 1

    return {
        "subtask": subtask_name,
        "task_name": task_name,
        "aligned_cases": len(set(q_map.keys()) & set(a_map.keys())),
        "requested_cases": len(keys),
        "created": created,
        "missing_q_pair_dirs": missing_q_pair,
        "missing_a_pair_dirs": missing_a_pair,
        "viz_dir": str(viz_root),
    }


def _save_triplet_visualizations_incremental(
    *,
    output_dir: Path,
    subtask_name: str,
    q_rows: List[Dict[str, Any]],
    a_rows: List[Dict[str, Any]],
    written_keys: Set[str],
    cell_w: int,
    cell_h: int,
    max_cases: Optional[int],
) -> Dict[str, Any]:
    task_name = f"atomic_{subtask_name}"
    q_map = {_triplet_task_key(r, default_task=subtask_name): r for r in q_rows if isinstance(r, dict)}
    a_map = {_triplet_task_key(r, default_task=subtask_name): r for r in a_rows if isinstance(r, dict)}
    keys = sorted(set(q_map.keys()) & set(a_map.keys()))
    if max_cases is not None:
        keys = keys[: max(0, int(max_cases))]

    q_pair_idx = _build_pair_meta_index(
        output_dir / "qwen" / task_name / "element_pairs",
        default_task=subtask_name,
    )
    a_pair_idx = _build_pair_meta_index(
        output_dir / "agent" / task_name / "element_pairs",
        default_task=subtask_name,
    )

    viz_root = output_dir / "triplet_pair_viz" / task_name
    created_now = 0
    pending_missing_pair = 0

    for k in keys:
        if k in written_keys:
            continue
        q_dir = q_pair_idx.get(k)
        a_dir = a_pair_idx.get(k)
        if q_dir is None or a_dir is None:
            pending_missing_pair += 1
            continue

        row_q = q_map[k]
        row_a = a_map[k]
        eid = str(row_q.get("episode_id", ""))
        gt_idx = int(row_q.get("gt_index", -1))
        key_hash = hashlib.sha1(k.encode("utf-8")).hexdigest()[:10]
        out_name = f"{eid}__gt{gt_idx}__{key_hash}.png"
        _render_triplet_2x3(
            viz_root / out_name,
            subtask=subtask_name,
            row_q=row_q,
            row_a=row_a,
            q_dir=q_dir,
            a_dir=a_dir,
            cell_w=int(cell_w),
            cell_h=int(cell_h),
        )
        written_keys.add(k)
        created_now += 1

    return {
        "subtask": subtask_name,
        "task_name": task_name,
        "aligned_cases": len(keys),
        "created_now": created_now,
        "created_total": len(written_keys),
        "pending_missing_pair_dirs": pending_missing_pair,
        "viz_dir": str(viz_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run atomic editability subtasks (Qwen vs Agent) with per-episode preselection"
    )
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--match-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--selection-seed", type=int, default=None, help="Seed for per-episode element preselection; defaults to --seed")
    parser.add_argument("--per-episode-elements", type=int, default=2, help="Number of (episode_id, gt_index) pairs sampled per episode before running subtasks")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--subset-manifest", type=str, default=None, help="Category subset manifest json from build_category_subsets.py")
    parser.add_argument("--max-matching-cost", type=float, default=0.5, help="Only evaluate matched pairs with matching cost <= this threshold (best_single/merged cost). Negative disables this filter.")
    parser.add_argument("--min-matching-iou", type=float, default=-1.0, help="Only evaluate matched pairs with matching IoU >= this threshold (merged_metrics.iou). Negative disables this filter.")
    parser.add_argument("--min-cross-model-iou", type=float, default=-1.0, help="Only keep GT triplets where IoU(Qwen matched pred, Agent matched pred) >= threshold. Negative disables.")
    parser.add_argument("--log-every", type=int, default=25, help="Print progress every N evaluated tasks per subtask")
    parser.add_argument("--save-pair-viz", action="store_true", help="Save edited task element-pair visualizations")
    parser.add_argument("--pair-viz-max-per-subtask", type=int, default=None, help="Optional cap for saved pair visualizations per model/subtask")
    parser.add_argument(
        "--save-triplet-viz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save 2x3 triplet visualizations (GT/Qwen/Agent x Before/After) for each aligned case.",
    )
    parser.add_argument("--triplet-viz-cell-width", type=int, default=520, help="Cell width for triplet 2x3 visualization.")
    parser.add_argument("--triplet-viz-cell-height", type=int, default=300, help="Cell height for triplet 2x3 visualization.")
    parser.add_argument("--triplet-viz-max-per-subtask", type=int, default=None, help="Optional cap for generated triplet visualizations per subtask.")
    parser.add_argument("--num-workers", type=int, default=1, help="Thread workers used inside each subtask evaluation")
    parser.add_argument("--min-gt-opaque-pixels", type=int, default=1500, help="Only evaluate matched pairs whose GT element has at least this many opaque(alpha>=threshold) pixels")
    parser.add_argument("--opaque-alpha-threshold", type=int, default=250, help="Alpha threshold (0-255) used to count opaque pixels")
    parser.add_argument("--strict-gt-opaque-filter", action="store_true", help="Use exact alpha>=threshold counting from GT images (slower). Default uses fast payload gt_area filter.")
    parser.add_argument("--cache-episodes", type=int, default=8, help="LRU cache size (episodes) for loaded GT/pred elements")
    parser.add_argument("--max-episode-loaders", type=int, default=2, help="Maximum concurrent episode loads when cache misses")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--foreground-filter-mode", type=str, default="z_order", choices=["z_order", "strict"], help="Foreground filter mode: z_order (fast, metadata-only) or strict (pixel-wise backing check).")
    parser.add_argument("--disable-full-foreground-filter", action="store_true", help="Disable strict full-foreground filtering (all GT/Qwen/Agent target pixels must have non-target backing behind).")
    parser.add_argument("--no-foreground-cache", action="store_true", help="Disable foreground triplet cache. When enabled (default), strict foreground pass/fail is reused from JSON cache to avoid repeated image loads.")
    parser.add_argument("--rebuild-foreground-cache", action="store_true", help="Ignore existing foreground cache and rebuild it during this run.")
    parser.add_argument("--foreground-cache-path", type=str, default=None, help="Optional cache JSON path for strict foreground triplet pass/fail.")
    parser.add_argument("--disable-evalfigma-postprocess", action="store_true", help="Disable evaluation_figma-style preprocess (alpha clean + optional text refinement) on pred targets before edit.")
    parser.add_argument("--disable-evalfigma-text-refinement", action="store_true", help="Disable text refinement step inside evaluation_figma-style pred preprocess.")
    parser.add_argument(
        "--metric-bg-mode",
        type=str,
        default="best_of_black_white",
        choices=["premultiplied", "best_of_black_white"],
        help=(
            "Metric RGB background mode for compute_region_metrics: "
            "premultiplied(black-case only) or best_of_black_white."
        ),
    )
    parser.add_argument("--build-log-every", type=int, default=50, help="Print candidate-build progress every N payloads")
    args = parser.parse_args()

    figma_data = Path(args.figma_data)
    match_root = Path(args.match_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ["EDITABILITY_CACHE_EPISODES"] = str(max(1, int(args.cache_episodes)))
    os.environ["EDITABILITY_MAX_EP_LOADERS"] = str(max(1, int(args.max_episode_loaders)))
    os.environ["EDITABILITY_MIN_GT_OPAQUE_PIXELS"] = str(max(0, int(args.min_gt_opaque_pixels)))
    os.environ["EDITABILITY_OPAQUE_ALPHA_THRESHOLD"] = str(max(0, min(255, int(args.opaque_alpha_threshold))))
    os.environ["EDITABILITY_STRICT_GT_OPAQUE_CHECK"] = "1" if args.strict_gt_opaque_filter else "0"
    os.environ["EDITABILITY_MAX_MATCHING_COST"] = str(float(args.max_matching_cost))
    os.environ["EDITABILITY_MIN_MATCHING_IOU"] = str(float(args.min_matching_iou))
    os.environ["EDITABILITY_USE_EVALFIGMA_POSTPROC"] = "0" if args.disable_evalfigma_postprocess else "1"
    os.environ["EDITABILITY_USE_EVALFIGMA_TEXT_REFINEMENT"] = "0" if args.disable_evalfigma_text_refinement else "1"
    os.environ["EDITABILITY_METRIC_BG_MODE"] = str(args.metric_bg_mode).strip().lower()

    if args.save_triplet_viz and not args.save_pair_viz:
        print("[setup] --save-triplet-viz enabled: forcing --save-pair-viz to ensure per-case source images.")
        args.save_pair_viz = True
    if args.save_triplet_viz and args.pair_viz_max_per_subtask is not None:
        print("[setup] --save-triplet-viz enabled: overriding --pair-viz-max-per-subtask=None for full triplet coverage.")
        args.pair_viz_max_per_subtask = None

    user_subset = load_subset_keys(Path(args.subset_manifest), "atomic") if args.subset_manifest else None
    if user_subset is not None:
        print(f"[setup] loaded atomic subset keys: {len(user_subset)}")
    else:
        print("[setup] no subset-manifest provided; using all matched atomic candidates")

    base_keys, joint_stats = _build_base_selection_keys(
        match_root=match_root,
        figma_data=figma_data,
        exp_pairs=args.exp_pairs,
        subset_keys=user_subset,
        max_matching_cost=float(args.max_matching_cost),
        min_matching_iou=float(args.min_matching_iou),
        min_cross_model_iou=float(args.min_cross_model_iou),
        build_log_every=max(0, int(args.build_log_every)),
    )
    base_keys = _apply_max_episodes_cap(
        base_keys,
        figma_data=figma_data,
        exp_pairs=args.exp_pairs,
        max_episodes=args.max_episodes,
    )
    sel_seed = int(args.seed if args.selection_seed is None else args.selection_seed)
    per_episode = max(0, int(args.per_episode_elements))
    fg_cache_path = None if args.no_foreground_cache else Path(args.foreground_cache_path) if args.foreground_cache_path else (match_root / "_cache" / "full_foreground_triplet_flags.json")
    foreground_stats: Dict[str, Any]
    if not args.disable_full_foreground_filter:
        if str(args.foreground_filter_mode) == "strict":
            selected_keys, foreground_stats = _sample_full_foreground_keys_per_episode(
                base_keys,
                per_episode=per_episode,
                seed=sel_seed,
                match_root=match_root,
                figma_data=figma_data,
                exp_pairs=args.exp_pairs,
                cache_path=fg_cache_path,
                use_existing_cache=not args.rebuild_foreground_cache,
                build_log_every=max(0, int(args.build_log_every)),
            )
        else:
            selected_keys, foreground_stats = _sample_front_zorder_keys_per_episode(
                base_keys,
                per_episode=per_episode,
                seed=sel_seed,
                match_root=match_root,
                figma_data=figma_data,
                exp_pairs=args.exp_pairs,
                build_log_every=max(0, int(args.build_log_every)),
            )
    else:
        selected_keys = _sample_keys_per_episode(
            base_keys,
            per_episode=per_episode,
            seed=sel_seed,
        )
        foreground_stats = {
            "enabled": False,
            "mode": "disabled",
            "input_pairs": len(base_keys),
            "input_episodes": len({eid for eid, _ in base_keys}),
            "per_episode_elements": per_episode,
            "selected_pairs": len(selected_keys),
            "selected_episodes": len({eid for eid, _ in selected_keys}),
        }
    print(
        "[setup] joint filters "
        f"cost={float(args.max_matching_cost):.4f} "
        f"min_iou={float(args.min_matching_iou):.4f} "
        f"min_cross_iou={float(args.min_cross_model_iou):.4f} "
        f"shared={joint_stats.get('shared_total')} "
        f"q_cost={joint_stats.get('qwen_cost_pass')} a_cost={joint_stats.get('agent_cost_pass')} "
        f"q_iou={joint_stats.get('qwen_iou_pass')} a_iou={joint_stats.get('agent_iou_pass')} "
        f"cross_known={joint_stats.get('cross_iou_known')} cross_pass={joint_stats.get('cross_iou_pass')} "
        f"final_subset={joint_stats.get('final_subset')}"
    )
    print(
        "[setup] full_foreground_filter "
        f"enabled={not args.disable_full_foreground_filter} "
        f"mode={args.foreground_filter_mode} "
        f"input_pairs={foreground_stats.get('input_pairs')} "
        f"selected_pairs={foreground_stats.get('selected_pairs')} "
        f"evaluated_pairs={foreground_stats.get('evaluated_pairs')} "
        f"cache_hits={foreground_stats.get('cache_hits')} "
        f"cache_miss={foreground_stats.get('cache_miss')} "
        f"cache_rebuild={args.rebuild_foreground_cache} "
        f"cache={str(fg_cache_path) if fg_cache_path is not None else 'disabled'}"
    )

    selected_episodes = {eid for eid, _ in selected_keys}
    print(
        f"[selection][atomic] base_pairs={len(base_keys)} base_episodes={len({eid for eid, _ in base_keys})} "
        f"per_episode={int(args.per_episode_elements)} selection_seed={sel_seed} "
        f"selected_pairs={len(selected_keys)} selected_episodes={len(selected_episodes)}"
    )
    save_json(
        output_dir / "atomic_selected_subset.json",
        {
            "selection_seed": sel_seed,
            "per_episode_elements": int(args.per_episode_elements),
            "base_pairs": len(base_keys),
            "selected_pairs": len(selected_keys),
            "selected_episodes": len(selected_episodes),
            "keys": [{"episode_id": eid, "gt_index": gt_idx} for eid, gt_idx in sorted(selected_keys)],
            "joint_stats": joint_stats,
            "foreground_filter_stats": foreground_stats,
        },
    )

    print(
        f"[setup] save_pair_viz={args.save_pair_viz} "
        f"pair_viz_max_per_subtask={args.pair_viz_max_per_subtask} "
        f"save_triplet_viz={args.save_triplet_viz} "
        f"triplet_viz_cell={args.triplet_viz_cell_width}x{args.triplet_viz_cell_height} "
        f"triplet_viz_max_per_subtask={args.triplet_viz_max_per_subtask} "
        f"log_every={args.log_every} "
        f"num_workers={args.num_workers} "
        f"max_matching_cost={os.environ.get('EDITABILITY_MAX_MATCHING_COST')} "
        f"min_matching_iou={os.environ.get('EDITABILITY_MIN_MATCHING_IOU')} "
        f"min_gt_opaque_pixels={os.environ.get('EDITABILITY_MIN_GT_OPAQUE_PIXELS')} "
        f"opaque_alpha_threshold={os.environ.get('EDITABILITY_OPAQUE_ALPHA_THRESHOLD')} "
        f"strict_gt_opaque_filter={os.environ.get('EDITABILITY_STRICT_GT_OPAQUE_CHECK')} "
        f"evalfigma_postprocess={os.environ.get('EDITABILITY_USE_EVALFIGMA_POSTPROC')} "
        f"evalfigma_text_refinement={os.environ.get('EDITABILITY_USE_EVALFIGMA_TEXT_REFINEMENT')} "
        f"metric_bg_mode={os.environ.get('EDITABILITY_METRIC_BG_MODE')} "
        f"cache_episodes={os.environ.get('EDITABILITY_CACHE_EPISODES')} "
        f"max_episode_loaders={os.environ.get('EDITABILITY_MAX_EP_LOADERS')} "
        f"tqdm={not args.no_tqdm} "
        f"build_log_every={args.build_log_every}"
    )

    subtasks = [
        ("delete", delete.run, 0),
        ("transition", transition.run, 1),
        ("rotation", rotation.run, 2),
        ("opacity", opacity.run, 3),
        ("z_order", z_order.run, 4),
        ("recolor", recolor.run, 5),
    ]

    qwen: Dict[str, Any] = {}
    agent: Dict[str, Any] = {}
    comparison: Dict[str, Any] = {}
    triplet_viz_summary: Dict[str, Any] = {}

    for subtask_name, run_fn, seed_offset in subtasks:
        sub_seed = int(args.seed) + int(seed_offset)
        print(f"[subtask][atomic][{subtask_name}] start qwen+agent")
        triplet_stop_event: Optional[threading.Event] = None
        triplet_thread: Optional[threading.Thread] = None
        triplet_written_keys: Set[str] = set()
        triplet_live_state: Dict[str, Any] = {
            "subtask": subtask_name,
            "task_name": f"atomic_{subtask_name}",
            "aligned_cases": 0,
            "created_now": 0,
            "created_total": 0,
            "pending_missing_pair_dirs": 0,
            "viz_dir": str(output_dir / "triplet_pair_viz" / f"atomic_{subtask_name}"),
        }

        if args.save_triplet_viz:
            q_ckpt_path = output_dir / "qwen" / f"atomic_{subtask_name}_results.json"
            a_ckpt_path = output_dir / "agent" / f"atomic_{subtask_name}_results.json"
            poll_sec = 2.0
            triplet_stop_event = threading.Event()

            def _triplet_live_worker() -> None:
                while not triplet_stop_event.is_set():
                    try:
                        q_rows_live = _load_result_rows_checkpoint(q_ckpt_path)
                        a_rows_live = _load_result_rows_checkpoint(a_ckpt_path)
                        info_live = _save_triplet_visualizations_incremental(
                            output_dir=output_dir,
                            subtask_name=subtask_name,
                            q_rows=q_rows_live,
                            a_rows=a_rows_live,
                            written_keys=triplet_written_keys,
                            cell_w=int(args.triplet_viz_cell_width),
                            cell_h=int(args.triplet_viz_cell_height),
                            max_cases=args.triplet_viz_max_per_subtask,
                        )
                        if int(info_live.get("created_now", 0)) > 0:
                            print(
                                "[triplet-viz-live] "
                                f"subtask={subtask_name} +{int(info_live.get('created_now', 0))} "
                                f"total={int(info_live.get('created_total', 0))} "
                                f"aligned={int(info_live.get('aligned_cases', 0))}"
                            )
                        triplet_live_state.update(info_live)
                    except Exception:
                        pass
                    triplet_stop_event.wait(poll_sec)

            triplet_thread = threading.Thread(
                target=_triplet_live_worker,
                name=f"triplet-viz-{subtask_name}",
                daemon=True,
            )
            triplet_thread.start()

        try:
            q_payload, a_payload = _run_subtask_paired(
                subtask_name=subtask_name,
                run_fn=run_fn,
                sub_seed=sub_seed,
                figma_data=figma_data,
                exp_pairs=args.exp_pairs,
                match_root=match_root,
                output_dir=output_dir,
                max_tasks_per_subtask=None,
                max_episodes=args.max_episodes,
                subset_keys=selected_keys,
                log_every=max(1, int(args.log_every)),
                save_pair_viz=args.save_pair_viz,
                pair_viz_max_per_subtask=args.pair_viz_max_per_subtask,
                num_workers=max(1, int(args.num_workers)),
                show_tqdm=not args.no_tqdm,
                build_log_every=max(0, int(args.build_log_every)),
            )
        finally:
            if triplet_stop_event is not None:
                triplet_stop_event.set()
            if triplet_thread is not None:
                triplet_thread.join(timeout=5.0)

        qwen[subtask_name] = q_payload
        agent[subtask_name] = a_payload
        comparison[subtask_name] = compare_two_models(
            qwen_summary=q_payload.get("summary", {}),
            agent_summary=a_payload.get("summary", {}),
            metric_preferences=_metric_preferences(),
        )
        print(
            f"[live-compare][atomic][{subtask_name}] "
            + _compact_compare_line(
                comparison[subtask_name],
                metric_order=[
                    "l1",
                    "l2",
                    "psnr",
                    "ssim",
                    "lpips",
                    "dino",
                    "full_l1",
                    "full_l2",
                    "full_psnr",
                    "full_ssim",
                    "full_lpips",
                    "full_dino",
                    "source_l1",
                    "target_l1",
                    "avg_l1",
                    "source_ssim",
                    "target_ssim",
                    "avg_ssim",
                    "source_lpips",
                    "target_lpips",
                    "avg_lpips",
                ],
            )
        )

        save_json(output_dir / "atomic_comparison_qwen_vs_agent.json", comparison)
        save_json(output_dir / "atomic_qwen_overview.json", qwen)
        save_json(output_dir / "atomic_agent_overview.json", agent)

        if args.save_triplet_viz:
            # Final flush after subtask completion (captures rows produced near the end).
            q_rows_done = q_payload.get("results", []) if isinstance(q_payload, dict) else []
            a_rows_done = a_payload.get("results", []) if isinstance(a_payload, dict) else []
            triplet_info = _save_triplet_visualizations_incremental(
                output_dir=output_dir,
                subtask_name=subtask_name,
                q_rows=q_rows_done if isinstance(q_rows_done, list) else [],
                a_rows=a_rows_done if isinstance(a_rows_done, list) else [],
                written_keys=triplet_written_keys,
                cell_w=int(args.triplet_viz_cell_width),
                cell_h=int(args.triplet_viz_cell_height),
                max_cases=args.triplet_viz_max_per_subtask,
            )
            triplet_live_state.update(triplet_info)
            triplet_info = dict(triplet_live_state)
            triplet_viz_summary[subtask_name] = triplet_info
            print(
                "[triplet-viz] "
                f"subtask={subtask_name} "
                f"aligned={triplet_info.get('aligned_cases')} "
                f"created_total={triplet_info.get('created_total')} "
                f"pending_missing_pair_dirs={triplet_info.get('pending_missing_pair_dirs')} "
                f"dir={triplet_info.get('viz_dir')}"
            )
            save_json(output_dir / "atomic_triplet_viz_summary.json", triplet_viz_summary)

    print("[DONE] atomic subtasks completed for qwen+agent (episode-wise preselection)")


if __name__ == "__main__":
    main()
