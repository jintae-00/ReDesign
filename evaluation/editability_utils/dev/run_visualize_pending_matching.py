#!/usr/bin/env python3
"""Render side-by-side pending visualizations for both qwen and agent."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from evaluation.editability_utils.common_utils import load_json
from evaluation.editability_utils.loaders import EpisodeTask, collect_episode_tasks, load_episode_elements
from evaluation.editability_utils.matching_visuals import save_match_visualizations


def _episode_png_path(viz_root: Path, episode_id: str, model: str) -> Path:
    return viz_root / episode_id / model / "episode.png"


def _pair_dir_path(viz_root: Path, episode_id: str, model: str) -> Path:
    return viz_root / episode_id / model / "pairs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pending matched episodes for both qwen and agent"
    )
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--match-root", type=str, required=True, help="Root with {qwen|agent}/episodes/*.json")
    parser.add_argument("--viz-root", type=str, required=True, help="Visualization output root")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap after auto-limit min(parsed, 50)")
    parser.add_argument("--max-rows", type=int, default=80, help="Max GT rows rendered per episode")
    parser.add_argument("--panel-width", type=int, default=220, help="Per-panel width")
    parser.add_argument("--save-pairs", action="store_true", help="Also save per-GT pair images")
    parser.add_argument("--force", action="store_true", help="Render even if qwen+agent episode png already exist")
    args = parser.parse_args()

    figma_data_dir = Path(args.figma_data)
    match_root = Path(args.match_root)
    viz_root = Path(args.viz_root)
    viz_root.mkdir(parents=True, exist_ok=True)

    qwen_tasks = collect_episode_tasks(
        figma_data_dir=figma_data_dir,
        exp_pairs=args.exp_pairs,
        model="qwen",
        max_episodes=None,
    )
    agent_tasks = collect_episode_tasks(
        figma_data_dir=figma_data_dir,
        exp_pairs=args.exp_pairs,
        model="agent",
        max_episodes=None,
    )
    qwen_task_map: Dict[str, EpisodeTask] = {t.episode_id: t for t in qwen_tasks}
    agent_task_map: Dict[str, EpisodeTask] = {t.episode_id: t for t in agent_tasks}

    qwen_match_files = {p.stem for p in (match_root / "qwen" / "episodes").glob("*.json")}
    agent_match_files = {p.stem for p in (match_root / "agent" / "episodes").glob("*.json")}

    common_parsed = sorted(
        qwen_match_files & agent_match_files & set(qwen_task_map.keys()) & set(agent_task_map.keys())
    )
    auto_limit = min(len(common_parsed), 50)
    target_count = auto_limit
    if args.max_episodes is not None:
        target_count = min(target_count, max(0, int(args.max_episodes)))
    target_episodes: List[str] = common_parsed[:target_count]

    pending: List[str] = []
    skipped_existing_viz = 0
    for eid in target_episodes:
        q_out = _episode_png_path(viz_root, eid, "qwen")
        a_out = _episode_png_path(viz_root, eid, "agent")
        if (not args.force) and q_out.exists() and a_out.exists():
            skipped_existing_viz += 1
            continue
        pending.append(eid)

    print("[pending-viz] models=qwen+agent", flush=True)
    print(f"  common_parsed={len(common_parsed)}", flush=True)
    print(f"  auto_limit=min(common_parsed,50)={auto_limit}", flush=True)
    if args.max_episodes is not None:
        print(f"  max_episodes_override={args.max_episodes}", flush=True)
    print(f"  target_before_skip={len(target_episodes)}", flush=True)
    print(f"  skipped_existing_viz={skipped_existing_viz}", flush=True)
    print(f"  to_render={len(pending)}", flush=True)

    rendered = 0
    for i, eid in enumerate(pending, start=1):
        q_task = qwen_task_map[eid]
        a_task = agent_task_map[eid]

        q_payload = load_json(match_root / "qwen" / "episodes" / f"{eid}.json")
        a_payload = load_json(match_root / "agent" / "episodes" / f"{eid}.json")

        q_gt, q_pred, q_size = load_episode_elements(q_task, model="qwen")
        a_gt, a_pred, a_size = load_episode_elements(a_task, model="agent")
        if q_size != a_size:
            raise ValueError(f"Canvas size mismatch for episode={eid}: qwen={q_size}, agent={a_size}")

        q_episode_out = _episode_png_path(viz_root, eid, "qwen")
        a_episode_out = _episode_png_path(viz_root, eid, "agent")
        q_pair_out = _pair_dir_path(viz_root, eid, "qwen") if args.save_pairs else None
        a_pair_out = _pair_dir_path(viz_root, eid, "agent") if args.save_pairs else None

        save_match_visualizations(
            episode_id=eid,
            split_name=q_task.split_name,
            payload=q_payload,
            gt_elements=q_gt,
            pred_elements=q_pred,
            canvas_size=q_size,
            episode_out_path=q_episode_out,
            pair_out_dir=q_pair_out,
            parsed_layers_src_dir=q_task.qwen_episode_dir,
            max_rows=args.max_rows,
            panel_width=args.panel_width,
        )
        save_match_visualizations(
            episode_id=eid,
            split_name=a_task.split_name,
            payload=a_payload,
            gt_elements=a_gt,
            pred_elements=a_pred,
            canvas_size=a_size,
            episode_out_path=a_episode_out,
            pair_out_dir=a_pair_out,
            parsed_layers_src_dir=None,
            max_rows=args.max_rows,
            panel_width=args.panel_width,
        )

        rendered += 1
        print(f"[{i}/{len(pending)}] rendered {eid} (qwen+agent)", flush=True)

    print(f"[DONE] rendered={rendered} out={viz_root}", flush=True)


if __name__ == "__main__":
    main()
