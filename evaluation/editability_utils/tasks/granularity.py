#!/usr/bin/env python3
"""Granularity metrics: Fragmentation, GT Coverage, Non-GT Contamination."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..common_utils import compute_iou, load_json, save_json
from ..loaders import EpisodeTask, collect_episode_tasks, load_episode_elements
from ..task_common import union_mask_from_indices


def _load_match_payloads(match_dir: Path) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for p in sorted((match_dir / "episodes").glob("*.json")):
        payloads.append(load_json(p))
    return payloads


def evaluate_granularity(
    payloads: List[Dict[str, Any]],
    task_map: Dict[str, EpisodeTask],
    model: str,
    coverage_tau: float = 0.5,
) -> Dict[str, Any]:
    per_gt: List[Dict[str, Any]] = []
    cache: Dict[str, Any] = {}

    for payload in payloads:
        eid = payload["episode_id"]
        if eid not in task_map:
            continue
        if eid not in cache:
            cache[eid] = load_episode_elements(task_map[eid], model=model)
        gt_elements, pred_elements, _ = cache[eid]

        for m in payload.get("matches", []):
            gt_idx = m.get("gt_index")
            pred_indices = m.get("selected_pred_indices", [])
            if gt_idx is None or gt_idx >= len(gt_elements):
                continue

            gt_mask = gt_elements[gt_idx]["mask"] > 0
            pred_mask = union_mask_from_indices(pred_elements, pred_indices)

            fragmentation = len(pred_indices)
            coverage = compute_iou(gt_mask.astype(np.float32), pred_mask.astype(np.float32))

            pred_area = float(pred_mask.sum())
            contam_area = float((pred_mask & (~gt_mask)).sum())
            contamination = contam_area / (pred_area + 1e-6)

            per_gt.append(
                {
                    "episode_id": eid,
                    "gt_index": gt_idx,
                    "gt_id": gt_elements[gt_idx].get("id"),
                    "fragmentation": fragmentation,
                    "gt_coverage": coverage,
                    "non_gt_contamination": contamination,
                    "is_edit_covered": 1 if coverage > coverage_tau else 0,
                }
            )

    if not per_gt:
        return {
            "count": 0,
            "fragmentation_mean": float("nan"),
            "gt_coverage_mean": float("nan"),
            "non_gt_contamination_mean": float("nan"),
            "edit_coverage_rate": float("nan"),
            "per_gt": [],
        }

    frag = np.array([x["fragmentation"] for x in per_gt], dtype=np.float32)
    cov = np.array([x["gt_coverage"] for x in per_gt], dtype=np.float32)
    cont = np.array([x["non_gt_contamination"] for x in per_gt], dtype=np.float32)
    edit_cov = np.array([x["is_edit_covered"] for x in per_gt], dtype=np.float32)

    return {
        "count": int(len(per_gt)),
        "fragmentation_mean": float(frag.mean()),
        "gt_coverage_mean": float(cov.mean()),
        "non_gt_contamination_mean": float(cont.mean()),
        "edit_coverage_rate": float(edit_cov.mean()),
        "per_gt": per_gt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Granularity metric evaluator")
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--model", type=str, choices=["agent", "qwen"], required=True)
    parser.add_argument("--match-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--coverage-tau", type=float, default=0.5)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    tasks = collect_episode_tasks(Path(args.figma_data), args.exp_pairs, model=args.model, max_episodes=args.max_episodes)
    task_map = {t.episode_id: t for t in tasks}

    payloads = _load_match_payloads(Path(args.match_root) / args.model)
    payloads = [p for p in payloads if p["episode_id"] in task_map]

    summary = evaluate_granularity(payloads, task_map, model=args.model, coverage_tau=args.coverage_tau)

    out_dir = Path(args.output)
    save_json(out_dir / f"granularity_{args.model}_summary.json", summary)
    print(f"[DONE] model={args.model} granularity_count={summary['count']}")


if __name__ == "__main__":
    main()
