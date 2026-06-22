#!/usr/bin/env python3
"""SVG/image editability tasks (super scaling, stroke, aspect ratio)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..common_utils import compute_region_metrics, load_json, save_json, sample_with_seed
from ..loaders import EpisodeTask, collect_episode_tasks, load_episode_elements
from ..task_common import apply_edit_to_scene, render_scene_rgba, union_mask_from_indices


def _load_match_payloads(match_dir: Path) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for p in sorted((match_dir / "episodes").glob("*.json")):
        payloads.append(load_json(p))
    return payloads


def build_svg_candidates(payloads: List[Dict[str, Any]], gt_meta_by_episode: Dict[str, Dict[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for payload in payloads:
        eid = payload["episode_id"]
        meta_map = gt_meta_by_episode.get(eid, {})
        for m in payload.get("matches", []):
            gt_idx = m.get("gt_index")
            pred_indices = m.get("selected_pred_indices", [])
            if gt_idx is None or not pred_indices:
                continue

            gt_meta = meta_map.get(gt_idx, {})
            node_type = str(gt_meta.get("node_type", "")).upper()
            has_stroke = bool(gt_meta.get("strokes_raw")) or float(gt_meta.get("stroke_weight") or 0) > 0
            is_rect = node_type in {"RECTANGLE", "FRAME", "INSTANCE", "COMPONENT", "VECTOR", ""}

            # Super scaling
            for s in (1.8, 3.0):
                candidates.append(
                    {
                        "episode_id": eid,
                        "task_type": "super_scaling",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "scale": s,
                    }
                )

            # Stroke
            if has_stroke:
                candidates.append(
                    {
                        "episode_id": eid,
                        "task_type": "stroke",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "stroke_width": 2,
                        "stroke_rgb": [0, 0, 0],
                    }
                )

            if is_rect:
                # Aspect ratio
                candidates.append(
                    {
                        "episode_id": eid,
                        "task_type": "aspect_ratio",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "scale_x": 1.2,
                        "scale_y": 0.85,
                    }
                )

    return candidates


def evaluate_svg_candidates(
    candidates: List[Dict[str, Any]],
    task_map: Dict[str, EpisodeTask],
    model: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    cache: Dict[str, Any] = {}

    for c in candidates:
        eid = c["episode_id"]
        if eid not in cache:
            cache[eid] = load_episode_elements(task_map[eid], model=model)
        gt_elements, pred_elements, canvas_size = cache[eid]

        gt_idx = c["gt_index"]
        pred_indices = c["pred_indices"]

        gt_edit_scene = apply_edit_to_scene(gt_elements, [gt_idx], canvas_size, c)
        pred_edit_scene = apply_edit_to_scene(pred_elements, pred_indices, canvas_size, c)

        gt_rgba = render_scene_rgba(gt_edit_scene, canvas_size)
        pred_rgba = render_scene_rgba(pred_edit_scene, canvas_size)

        roi = union_mask_from_indices(gt_elements, [gt_idx]) | union_mask_from_indices(gt_edit_scene, [gt_idx])
        # Stroke uses edge-focused ROI approximation.
        if c["task_type"] == "stroke":
            from ..common_utils import edge_mask_from_alpha

            roi = edge_mask_from_alpha((roi * 255).astype("uint8"), radius=2)

        metrics = compute_region_metrics(
            gt_rgba,
            pred_rgba,
            roi,
            include_iou=True,
            include_edge_sharpness=True,
        )

        results.append(
            {
                "episode_id": eid,
                "task_type": c["task_type"],
                "gt_index": gt_idx,
                "pred_indices": pred_indices,
                "metrics": metrics,
            }
        )

    return results


def aggregate_svg(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, float]]] = {}
    for r in results:
        by_task.setdefault(r["task_type"], []).append(r["metrics"])

    out: Dict[str, Any] = {"total": len(results), "by_task_type": {}}
    for k, vals in by_task.items():
        keys = sorted({mk for d in vals for mk in d.keys()})
        mean = {}
        for mk in keys:
            xs = [float(d[mk]) for d in vals if mk in d and d[mk] == d[mk]]
            mean[mk] = float(sum(xs) / len(xs)) if xs else float("nan")
        out["by_task_type"][k] = {"count": len(vals), "mean": mean}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SVG/image editability evaluator")
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--model", type=str, choices=["agent", "qwen"], required=True)
    parser.add_argument("--match-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    tasks = collect_episode_tasks(Path(args.figma_data), args.exp_pairs, model=args.model, max_episodes=args.max_episodes)
    task_map = {t.episode_id: t for t in tasks}

    payloads = _load_match_payloads(Path(args.match_root) / args.model)
    payloads = [p for p in payloads if p["episode_id"] in task_map]

    # Preload GT metadata map once for candidate validity checks.
    gt_meta_by_episode: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for p in payloads:
        eid = p["episode_id"]
        gt_elements, _, _ = load_episode_elements(task_map[eid], model=args.model)
        gt_meta_by_episode[eid] = {
            i: e.get("meta", {}).get("gt_unit", {}) for i, e in enumerate(gt_elements)
        }

    candidates = build_svg_candidates(payloads, gt_meta_by_episode)
    sampled = sample_with_seed(candidates, args.max_tasks, args.seed)
    results = evaluate_svg_candidates(sampled, task_map, model=args.model)
    summary = aggregate_svg(results)

    out_dir = Path(args.output)
    save_json(out_dir / f"svg_{args.model}_results.json", results)
    save_json(out_dir / f"svg_{args.model}_summary.json", summary)

    print(
        f"[DONE] model={args.model} svg_candidates={len(candidates)} sampled={len(sampled)} results={len(results)}"
    )


if __name__ == "__main__":
    main()
