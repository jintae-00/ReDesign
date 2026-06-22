#!/usr/bin/env python3
"""Orchestrate per-task editability evaluation from precomputed matches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from evaluation.editability_utils.common_utils import load_json, save_json
from evaluation.editability_utils.loaders import collect_episode_tasks, load_episode_elements
from evaluation.editability_utils.task_sampling import summarize_task_capacity
from evaluation.editability_utils.tasks.atomic import (
    aggregate_atomic,
    build_atomic_candidates,
    evaluate_atomic_candidates,
)
from evaluation.editability_utils.tasks.granularity import evaluate_granularity
from evaluation.editability_utils.tasks.svg import aggregate_svg, build_svg_candidates, evaluate_svg_candidates
from evaluation.editability_utils.tasks.text import (
    aggregate_content_recognition,
    aggregate_style_edits,
    build_style_edit_candidates,
    evaluate_content_recognition,
    evaluate_style_edits,
)


def _load_payloads(model_match_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted((model_match_dir / "episodes").glob("*.json")):
        out.append(load_json(p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run editability task evaluations")
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--model", type=str, choices=["agent", "qwen"], required=True)
    parser.add_argument("--match-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-atomic", type=int, default=None)
    parser.add_argument("--max-text-style", type=int, default=None)
    parser.add_argument("--max-svg", type=int, default=None)
    args = parser.parse_args()

    tasks = collect_episode_tasks(Path(args.figma_data), args.exp_pairs, model=args.model, max_episodes=args.max_episodes)
    task_map = {t.episode_id: t for t in tasks}

    payloads = _load_payloads(Path(args.match_root) / args.model)
    payloads = [p for p in payloads if p["episode_id"] in task_map]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Capacity report (requirement [3]).
    canvas_map = {p["episode_id"]: p.get("canvas_size", [1920, 1080]) for p in payloads}
    atomic_candidates = build_atomic_candidates(payloads, canvas_map)
    text_style_candidates = build_style_edit_candidates(payloads)

    gt_meta_by_episode: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for p in payloads:
        eid = p["episode_id"]
        gt_elements, _, _ = load_episode_elements(task_map[eid], model=args.model)
        gt_meta_by_episode[eid] = {i: e.get("meta", {}).get("gt_unit", {}) for i, e in enumerate(gt_elements)}

    svg_candidates = build_svg_candidates(payloads, gt_meta_by_episode)

    capacity = {
        "atomic": summarize_task_capacity(atomic_candidates),
        "text_style": summarize_task_capacity(text_style_candidates),
        "svg": summarize_task_capacity(svg_candidates),
        "note": "Candidates are generated after matching, then shuffled by seed and sampled by max-* options.",
    }
    save_json(out_dir / f"capacity_{args.model}.json", capacity)

    # Text content recognition (full, no task sampling).
    content_results = evaluate_content_recognition(payloads, task_map, model=args.model)
    save_json(out_dir / f"text_content_recognition_{args.model}.json", content_results)
    save_json(out_dir / f"text_content_recognition_{args.model}_summary.json", aggregate_content_recognition(content_results))

    # Atomic sampled evaluation.
    sampled_atomic = atomic_candidates
    if args.max_atomic is not None:
        from .common_utils import sample_with_seed

        sampled_atomic = sample_with_seed(atomic_candidates, args.max_atomic, args.seed)
    atomic_results = evaluate_atomic_candidates(sampled_atomic, task_map, model=args.model)
    save_json(out_dir / f"atomic_{args.model}_results.json", atomic_results)
    save_json(out_dir / f"atomic_{args.model}_summary.json", aggregate_atomic(atomic_results))

    # Text style sampled evaluation.
    sampled_text_style = text_style_candidates
    if args.max_text_style is not None:
        from .common_utils import sample_with_seed

        sampled_text_style = sample_with_seed(text_style_candidates, args.max_text_style, args.seed)
    style_results = evaluate_style_edits(sampled_text_style, task_map, model=args.model)
    save_json(out_dir / f"text_style_edit_{args.model}.json", style_results)
    save_json(out_dir / f"text_style_edit_{args.model}_summary.json", aggregate_style_edits(style_results))

    # SVG sampled evaluation.
    sampled_svg = svg_candidates
    if args.max_svg is not None:
        from .common_utils import sample_with_seed

        sampled_svg = sample_with_seed(svg_candidates, args.max_svg, args.seed)
    svg_results = evaluate_svg_candidates(sampled_svg, task_map, model=args.model)
    save_json(out_dir / f"svg_{args.model}_results.json", svg_results)
    save_json(out_dir / f"svg_{args.model}_summary.json", aggregate_svg(svg_results))

    # Granularity (always full).
    granularity_summary = evaluate_granularity(payloads, task_map, model=args.model, coverage_tau=0.5)
    save_json(out_dir / f"granularity_{args.model}_summary.json", granularity_summary)

    print(
        f"[DONE] model={args.model} "
        f"content={len(content_results)} atomic={len(atomic_results)} text_style={len(style_results)} svg={len(svg_results)}"
    )


if __name__ == "__main__":
    main()
