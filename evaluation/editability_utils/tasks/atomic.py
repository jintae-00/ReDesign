#!/usr/bin/env python3
"""Atomic editability tasks (non-text + text shared geometric/color edits)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..common_utils import compute_region_metrics, load_json, save_json, sample_with_seed
from ..loaders import EpisodeTask, collect_episode_tasks, load_episode_elements
from ..task_common import apply_edit_to_scene, render_scene_rgba, union_mask_from_indices


def build_atomic_candidates(
    episode_match_payloads: List[Dict[str, Any]],
    canvas_size_by_episode: Dict[str, Sequence[int]],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for payload in episode_match_payloads:
        episode_id = payload["episode_id"]
        canvas_w, canvas_h = canvas_size_by_episode.get(episode_id, [1920, 1080])

        for m in payload.get("matches", []):
            gt_idx = m.get("gt_index")
            pred_indices = m.get("selected_pred_indices", [])
            if gt_idx is None or not pred_indices:
                continue

            # Delete
            candidates.append(
                {
                    "episode_id": episode_id,
                    "task_type": "delete",
                    "gt_index": gt_idx,
                    "pred_indices": pred_indices,
                }
            )

            # Transition: 4 deterministic directions.
            shift = max(4, int(min(canvas_w, canvas_h) * 0.03))
            for dx, dy in [(shift, 0), (-shift, 0), (0, shift), (0, -shift)]:
                candidates.append(
                    {
                        "episode_id": episode_id,
                        "task_type": "transition",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "dx": dx,
                        "dy": dy,
                    }
                )

            # Rotation
            for angle in (-20.0, 20.0):
                candidates.append(
                    {
                        "episode_id": episode_id,
                        "task_type": "rotation",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "angle_deg": angle,
                    }
                )

            # Opacity
            for opacity_factor in (0.5, 1.5):
                candidates.append(
                    {
                        "episode_id": episode_id,
                        "task_type": "opacity",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "opacity_factor": opacity_factor,
                    }
                )

            # Z-order
            for direction in ("front", "back"):
                candidates.append(
                    {
                        "episode_id": episode_id,
                        "task_type": "z_order",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "direction": direction,
                    }
                )

            # Recolor
            for hue_shift in (-30.0, 30.0):
                candidates.append(
                    {
                        "episode_id": episode_id,
                        "task_type": "recolor",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "hue_shift_deg": hue_shift,
                        "sat_mul": 1.1,
                        "val_mul": 1.0,
                    }
                )

    return candidates


def evaluate_atomic_candidates(
    candidates: List[Dict[str, Any]],
    task_map: Dict[str, EpisodeTask],
    model: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # Lazy cache per episode.
    episode_cache: Dict[str, Any] = {}

    for cand in candidates:
        episode_id = cand["episode_id"]
        if episode_id not in episode_cache:
            task = task_map[episode_id]
            gt_elements, pred_elements, canvas_size = load_episode_elements(task, model=model)
            episode_cache[episode_id] = (gt_elements, pred_elements, canvas_size)

        gt_elements, pred_elements, canvas_size = episode_cache[episode_id]
        gt_idx = cand["gt_index"]
        pred_indices = cand["pred_indices"]

        gt_before = render_scene_rgba(gt_elements, canvas_size)
        pred_before = render_scene_rgba(pred_elements, canvas_size)

        gt_edited_scene = apply_edit_to_scene(gt_elements, [gt_idx], canvas_size, cand)
        pred_edited_scene = apply_edit_to_scene(pred_elements, pred_indices, canvas_size, cand)

        gt_after = render_scene_rgba(gt_edited_scene, canvas_size)
        pred_after = render_scene_rgba(pred_edited_scene, canvas_size)

        # ROI: source + target GT mask (dilated) around edited region.
        src = union_mask_from_indices(gt_elements, [gt_idx])
        dst = union_mask_from_indices(gt_edited_scene, [gt_idx])
        roi = (src | dst)

        include_iou = cand["task_type"] in {"transition", "rotation"}
        metrics = compute_region_metrics(gt_after, pred_after, roi, include_iou=include_iou)

        results.append(
            {
                "episode_id": episode_id,
                "task_type": cand["task_type"],
                "gt_index": gt_idx,
                "pred_indices": pred_indices,
                "edit": {k: v for k, v in cand.items() if k not in {"episode_id", "gt_index", "pred_indices"}},
                "metrics": metrics,
            }
        )

    return results


def aggregate_atomic(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"total": len(results), "by_task_type": {}}
    per_task: Dict[str, List[Dict[str, float]]] = {}
    for r in results:
        per_task.setdefault(r["task_type"], []).append(r["metrics"])

    for task_type, vals in per_task.items():
        keys = sorted({k for d in vals for k in d.keys()})
        agg = {}
        for k in keys:
            xs = [float(d[k]) for d in vals if k in d and d[k] == d[k]]
            agg[k] = float(sum(xs) / len(xs)) if xs else float("nan")
        out["by_task_type"][task_type] = {"count": len(vals), "mean": agg}
    return out


def _load_match_payloads(match_dir: Path) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for p in sorted((match_dir / "episodes").glob("*.json")):
        payloads.append(load_json(p))
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomic editability evaluator")
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--model", type=str, choices=["agent", "qwen"], required=True)
    parser.add_argument("--match-root", type=str, required=True, help="Root that contains <model>/episodes/*.json")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    tasks = collect_episode_tasks(Path(args.figma_data), args.exp_pairs, model=args.model, max_episodes=args.max_episodes)
    task_map = {t.episode_id: t for t in tasks}

    payloads = _load_match_payloads(Path(args.match_root) / args.model)
    payloads = [p for p in payloads if p["episode_id"] in task_map]

    canvas_map = {p["episode_id"]: p.get("canvas_size", [1920, 1080]) for p in payloads}
    candidates = build_atomic_candidates(payloads, canvas_map)
    sampled = sample_with_seed(candidates, args.max_tasks, args.seed)

    results = evaluate_atomic_candidates(sampled, task_map, model=args.model)
    summary = aggregate_atomic(results)

    out_dir = Path(args.output)
    save_json(out_dir / f"atomic_{args.model}_results.json", results)
    save_json(out_dir / f"atomic_{args.model}_summary.json", summary)

    print(
        f"[DONE] model={args.model} candidates={len(candidates)} sampled={len(sampled)} "
        f"results={len(results)}"
    )


if __name__ == "__main__":
    main()
