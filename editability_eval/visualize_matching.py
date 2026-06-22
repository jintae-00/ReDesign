#!/usr/bin/env python3
"""Visualize per-episode matching pairs with L1/cost metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from .common_utils import load_json
from .loaders import EpisodeTask, collect_episode_tasks, load_episode_elements
from .matching_visuals import save_match_visualizations


def _filter_tasks(tasks: Sequence[EpisodeTask], episode_ids: Sequence[str]) -> List[EpisodeTask]:
    if not episode_ids:
        return list(tasks)
    episode_set = set(episode_ids)
    return [t for t in tasks if t.episode_id in episode_set]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GT:model matching pairs per episode")
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--match-root", type=str, required=True, help="Root with {qwen|agent}/episodes/*.json")
    parser.add_argument("--output", type=str, required=True, help="Directory to save visualization PNGs")
    parser.add_argument("--model", type=str, choices=["qwen", "agent"], required=True)
    parser.add_argument("--episode-ids", type=str, nargs="*", default=None, help="Optional explicit episode ids")
    parser.add_argument("--max-episodes", type=int, default=10, help="Max episodes when --episode-ids not set")
    parser.add_argument("--max-rows", type=int, default=80, help="Max GT rows per episode image")
    parser.add_argument("--panel-width", type=int, default=220, help="Per-panel width in pixels")
    args = parser.parse_args()

    tasks = collect_episode_tasks(
        figma_data_dir=Path(args.figma_data),
        exp_pairs=args.exp_pairs,
        model=args.model,
        max_episodes=None,
    )
    requested_episode_ids = args.episode_ids or []
    tasks = _filter_tasks(tasks, requested_episode_ids)
    if not requested_episode_ids and args.max_episodes is not None:
        tasks = tasks[: max(0, int(args.max_episodes))]

    if not tasks:
        print("[visualize] no tasks to render")
        return

    match_dir = Path(args.match_root) / args.model / "episodes"
    out_dir = Path(args.output) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for i, task in enumerate(tasks, start=1):
        match_path = match_dir / f"{task.episode_id}.json"
        if not match_path.exists():
            print(f"[skip] {task.episode_id}: missing {match_path}")
            continue

        payload = load_json(match_path)
        gt_elements, pred_elements, canvas_size = load_episode_elements(task, model=args.model)
        out_path = out_dir / f"{task.episode_id}.png"
        save_match_visualizations(
            episode_id=task.episode_id,
            split_name=task.split_name,
            payload=payload,
            gt_elements=gt_elements,
            pred_elements=pred_elements,
            canvas_size=canvas_size,
            episode_out_path=out_path,
            pair_out_dir=None,
            max_rows=args.max_rows,
            panel_width=args.panel_width,
        )
        rendered += 1
        print(f"[{i}/{len(tasks)}] rendered {task.episode_id} -> {out_path}")

    print(f"[DONE] rendered={rendered} model={args.model} out={out_dir}")


if __name__ == "__main__":
    main()
