#!/usr/bin/env python3
"""Build Atomic/Text/SVG subset manifest from precomputed matching outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common_utils import load_json, sample_with_seed_balanced_by_key, save_json
from .subtasks.common import (
    build_task_map,
    load_match_payloads,
)


def _index_matches(payloads: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for payload in payloads:
        eid = str(payload.get("episode_id", ""))
        if not eid:
            continue
        d = out.setdefault(eid, {})
        for m in payload.get("matches", []):
            gt_idx = m.get("gt_index")
            if gt_idx is None:
                continue
            d[int(gt_idx)] = m
    return out


def _safe_text(unit: Dict[str, Any]) -> str:
    return str(unit.get("text_content", "") or "").strip()


def _safe_float(d: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


def _base_item(
    eid: str,
    gt_idx: int,
    gt_id: str,
    gt_type: str,
    gt_area: float,
    agent_match: Dict[str, Any],
    qwen_match: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "task_key": f"{eid}::{gt_idx}",
        "episode_id": eid,
        "gt_index": gt_idx,
        "gt_id": gt_id,
        "gt_type": gt_type,
        "gt_area": float(gt_area),
        "agent": {
            "pred_indices": [int(x) for x in agent_match.get("selected_pred_indices", [])],
            "pred_ids": [str(x) for x in agent_match.get("selected_pred_ids", [])],
            "merged_metrics": dict(agent_match.get("merged_metrics", {})),
            "best_single_pred_index": agent_match.get("best_single_pred_index"),
            "best_single_pred_id": agent_match.get("best_single_pred_id"),
            "best_single_metrics": dict(agent_match.get("best_single_metrics", {})),
        },
        "qwen": {
            "pred_indices": [int(x) for x in (qwen_match or {}).get("selected_pred_indices", [])],
            "pred_ids": [str(x) for x in (qwen_match or {}).get("selected_pred_ids", [])],
            "merged_metrics": dict((qwen_match or {}).get("merged_metrics", {})),
            "best_single_pred_index": (qwen_match or {}).get("best_single_pred_index"),
            "best_single_pred_id": (qwen_match or {}).get("best_single_pred_id"),
            "best_single_metrics": dict((qwen_match or {}).get("best_single_metrics", {})),
        },
    }


def _unit_by_gt_id(task: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    frame = load_json(task.gt_json_path)
    for unit in frame.get("unit_images", []):
        unit_id = str(unit.get("unit_id", ""))
        if unit_id:
            out[f"gt_{unit_id}"] = unit
    return out


def _agent_elem_by_id(task: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if task.agent_episode_dir is None:
        return out
    parse_data = load_json(task.agent_episode_dir / "parse.json")
    for elem in parse_data.get("elements", []):
        eid = str(elem.get("id", ""))
        if eid:
            out[eid] = elem
    return out


def _is_text(gt_type: str, unit: Dict[str, Any]) -> bool:
    if str(gt_type).lower() == "text":
        return True
    return str(unit.get("unit_type", "")).lower() == "text"


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


def build_subsets(
    figma_data: Path,
    exp_pairs: Sequence[str],
    match_root: Path,
    max_episodes: Optional[int],
    atomic_iou_min: float,
    atomic_l1_max: float,
    min_gt_opaque_pixels: int,
    require_qwen_pair: bool,
    require_text_content: bool,
    sample_atomic: Optional[int] = None,
    sample_text: Optional[int] = None,
    sample_svg: Optional[int] = None,
    sample_seed: int = 123,
) -> Dict[str, Any]:
    task_map = build_task_map(figma_data, exp_pairs, model="agent", max_episodes=max_episodes)

    agent_payloads = [p for p in load_match_payloads(match_root, "agent") if p["episode_id"] in task_map]
    qwen_payloads = [p for p in load_match_payloads(match_root, "qwen") if p["episode_id"] in task_map]
    qwen_idx = _index_matches(qwen_payloads)
    gt_unit_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    agent_elem_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    atomic_items: List[Dict[str, Any]] = []
    text_items: List[Dict[str, Any]] = []
    svg_items: List[Dict[str, Any]] = []

    for payload in agent_payloads:
        eid = payload["episode_id"]
        task = task_map[eid]
        if eid not in gt_unit_cache:
            gt_unit_cache[eid] = _unit_by_gt_id(task)
        if eid not in agent_elem_cache:
            agent_elem_cache[eid] = _agent_elem_by_id(task)
        unit_map = gt_unit_cache[eid]
        agent_elem_map = agent_elem_cache[eid]
        qmap = qwen_idx.get(eid, {})

        for m in payload.get("matches", []):
            gt_idx_raw = m.get("gt_index")
            if gt_idx_raw is None:
                continue
            gt_idx = int(gt_idx_raw)
            pred_indices = [int(x) for x in m.get("selected_pred_indices", [])]
            if not pred_indices:
                continue

            gt_id = str(m.get("gt_id", ""))
            gt_type = str(m.get("gt_type", ""))
            gt_area = _safe_float(m, "gt_area", 0.0)
            unit = unit_map.get(gt_id, {})
            if gt_area < float(max(0, int(min_gt_opaque_pixels))):
                continue

            qm = qmap.get(gt_idx)
            has_qwen_pair = bool(qm and qm.get("selected_pred_indices"))
            if require_qwen_pair and not has_qwen_pair:
                continue

            base = _base_item(eid, gt_idx, gt_id, gt_type, gt_area, m, qm)

            # Atomic subset: keep full visible matching set (no extra IOU/L1 subset filtering).
            atomic_items.append(base)

            # Text subset.
            if _is_text(gt_type, unit):
                gt_text = _safe_text(unit)
                if (not require_text_content) or gt_text:
                    item = dict(base)
                    item["gt_text"] = gt_text
                    item["gt_text_len"] = len(gt_text)
                    text_items.append(item)

            # SVG subset.
            if len(pred_indices) == 1 and _is_rectangle_unit(unit):
                pred_ids = [str(x) for x in m.get("selected_pred_ids", [])]
                if len(pred_ids) != 1:
                    continue
                pred_id = pred_ids[0]
                core_id = pred_id[len("agent_") :] if pred_id.startswith("agent_") else pred_id
                parsed = agent_elem_map.get(core_id, {})
                svg_uri = str(parsed.get("svg_uri", "") or "").strip()
                if svg_uri:
                    item = dict(base)
                    item["agent_svg_uri"] = svg_uri
                    item["gt_has_stroke"] = bool(_has_stroke_unit(unit))
                    svg_items.append(item)

    # Stable ordering for reproducible sampling.
    def _sort_key(x: Dict[str, Any]) -> Tuple[str, int]:
        return str(x["episode_id"]), int(x["gt_index"])

    atomic_items.sort(key=_sort_key)
    text_items.sort(key=_sort_key)
    svg_items.sort(key=_sort_key)

    def _diverse_sample(items: List[Dict[str, Any]], cap: Optional[int], seed: int) -> List[Dict[str, Any]]:
        return sample_with_seed_balanced_by_key(
            items,
            key_fn=lambda x: (str(x.get("episode_id", "")), str(x.get("gt_type", ""))),
            max_count=cap,
            seed=seed,
        )

    atomic_items = _diverse_sample(atomic_items, sample_atomic, sample_seed)
    text_items = _diverse_sample(text_items, sample_text, sample_seed + 1)
    svg_items = _diverse_sample(svg_items, sample_svg, sample_seed + 2)

    def _cat_payload(items: List[Dict[str, Any]], sample_cap: Optional[int]) -> Dict[str, Any]:
        return {
            "count": len(items),
            "sample_cap": sample_cap,
            "keys": [str(it["task_key"]) for it in items],
            "items": items,
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "atomic_iou_min": float(atomic_iou_min),  # deprecated, retained for backward compatibility
            "atomic_l1_max": float(atomic_l1_max),  # deprecated, retained for backward compatibility
            "min_gt_opaque_pixels": int(min_gt_opaque_pixels),
            "require_qwen_pair": bool(require_qwen_pair),
            "require_text_content": bool(require_text_content),
            "max_episodes": max_episodes,
            "sample_atomic": sample_atomic,
            "sample_text": sample_text,
            "sample_svg": sample_svg,
            "sample_seed": int(sample_seed),
        },
        "counts": {
            "episodes_in_scope": len(task_map),
            "agent_episode_payloads": len(agent_payloads),
            "qwen_episode_payloads": len(qwen_payloads),
        },
        "categories": {
            "atomic": _cat_payload(atomic_items, sample_atomic),
            "text": _cat_payload(text_items, sample_text),
            "svg": _cat_payload(svg_items, sample_svg),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Atomic/Text/SVG subset manifest from matches")
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--match-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Output json manifest path")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--atomic-iou-min", type=float, default=0.0, help="Deprecated. Atomic subset no longer filters by IOU.")
    parser.add_argument("--atomic-l1-max", type=float, default=1.0, help="Deprecated. Atomic subset no longer filters by L1.")
    parser.add_argument("--min-gt-opaque-pixels", type=int, default=1500, help="Keep only visible matches with gt_area >= threshold")
    parser.add_argument("--allow-missing-qwen-pair", action="store_true", help="Do not require qwen matched pair for subset rows")
    parser.add_argument("--allow-empty-text-content", action="store_true", help="Include text GT rows even if text_content is empty")
    parser.add_argument("--sample-atomic", type=int, default=None, help="Optional cap for sampled atomic subset size")
    parser.add_argument("--sample-text", type=int, default=None, help="Optional cap for sampled text subset size")
    parser.add_argument("--sample-svg", type=int, default=None, help="Optional cap for sampled svg subset size")
    parser.add_argument("--sample-seed", type=int, default=123, help="Seed used for optional subset sampling")
    args = parser.parse_args()

    payload = build_subsets(
        figma_data=Path(args.figma_data),
        exp_pairs=args.exp_pairs,
        match_root=Path(args.match_root),
        max_episodes=args.max_episodes,
        atomic_iou_min=args.atomic_iou_min,
        atomic_l1_max=args.atomic_l1_max,
        min_gt_opaque_pixels=args.min_gt_opaque_pixels,
        require_qwen_pair=not args.allow_missing_qwen_pair,
        require_text_content=not args.allow_empty_text_content,
        sample_atomic=args.sample_atomic,
        sample_text=args.sample_text,
        sample_svg=args.sample_svg,
        sample_seed=args.sample_seed,
    )
    out_path = Path(args.output)
    save_json(out_path, payload)

    cats = payload.get("categories", {})
    print(
        "[DONE] subset manifest "
        f"atomic={cats.get('atomic', {}).get('count', 0)} "
        f"text={cats.get('text', {}).get('count', 0)} "
        f"svg={cats.get('svg', {}).get('count', 0)} "
        f"-> {out_path}"
    )


if __name__ == "__main__":
    main()
