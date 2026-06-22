#!/usr/bin/env python3
"""Precompute GT:Qwen and GT:Agent matching pairs in one command."""

from __future__ import annotations

import argparse
from pathlib import Path

from .match_runner import run_matching
from .matching_core import MatchConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute matching pairs for all models")
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--lambda-l1", type=float, default=0.7)
    parser.add_argument("--lambda-iou", type=float, default=0.3, help="Weight for IoU term in matching cost: lambda_iou * (1 - iou).")
    parser.add_argument(
        "--l1-mode",
        type=str,
        default="rgba",
        choices=["rgb", "rgba"],
        help="L1 channel mode on GT+Pred union region. 'rgba' is robust to transparent/background ambiguity.",
    )
    parser.add_argument(
        "--max-merge-n",
        type=int,
        default=None,
        help="Max greedy merge size per GT. Default: unlimited. Set <=0 for unlimited.",
    )
    parser.add_argument("--min-gt-overlap", type=float, default=0.0, help="Candidate filter: bbox intersection area / GT bbox area threshold.")
    parser.add_argument("--min-cost-improve", type=float, default=1e-4)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--trace-episodes", action="store_true", help="Print per-episode start/end timing logs")
    parser.add_argument("--detailed-logs", action="store_true", help="Enable detailed per-GT/per-episode matching logs")
    parser.add_argument("--slow-episode-sec", type=float, default=30.0, help="Always log episodes slower than this")
    parser.add_argument("--gt-progress-every", type=int, default=25, help="Print GT-level progress every N GTs")
    parser.add_argument("--gt-progress-sec", type=float, default=10.0, help="Print GT-level progress at least every N sec")
    parser.add_argument("--num-workers", type=int, default=1, help="Episode-level parallel workers")
    parser.add_argument(
        "--mp-start-method",
        type=str,
        choices=["spawn", "fork", "forkserver"],
        default="spawn",
        help="Multiprocessing start method for episode workers. 'fork' can reduce startup overhead on Linux.",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=8,
        help="ProcessPool worker recycle period. Set <=0 to disable recycle for max throughput.",
    )
    parser.add_argument("--visualize-during-matching", action="store_true", help="Save visualization while matching runs")
    parser.add_argument("--viz-save-pairs", action="store_true", help="Also save per-GT pair images")
    parser.add_argument("--viz-output", type=str, default=None, help="Visualization output root (default: <output>/viz)")
    parser.add_argument("--viz-max-rows", type=int, default=80, help="Max GT rows/pairs rendered per episode")
    parser.add_argument("--viz-panel-width", type=int, default=220, help="Per-panel width for viz images")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip existing episode json outputs")
    args = parser.parse_args()

    cfg = MatchConfig(
        lambda_l1=args.lambda_l1,
        lambda_iou=args.lambda_iou,
        l1_mode=args.l1_mode,
        max_merge_n=args.max_merge_n,
        min_gt_overlap=args.min_gt_overlap,
        min_cost_improve=args.min_cost_improve,
    )

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    viz_root = None
    if args.visualize_during_matching:
        viz_root = Path(args.viz_output) if args.viz_output else (out_root / "viz")
        viz_root.mkdir(parents=True, exist_ok=True)
    viz_pair_root = None
    if args.visualize_during_matching and args.viz_save_pairs:
        viz_pair_root = viz_root

    print("[START] precompute matching", flush=True)
    print(f"  figma_data={args.figma_data}", flush=True)
    print(f"  output={out_root}", flush=True)
    print(f"  exp_pairs={len(args.exp_pairs)}", flush=True)
    print(f"  num_workers={args.num_workers}", flush=True)
    print(f"  mp_start_method={args.mp_start_method}", flush=True)
    print(
        f"  max_tasks_per_child={(None if int(args.max_tasks_per_child) <= 0 else int(args.max_tasks_per_child))}",
        flush=True,
    )
    print("  backend=CPU(numpy)", flush=True)
    print(f"  cost=lambda_l1*l1_union_region + lambda_iou*(1-iou) (mode={args.l1_mode})", flush=True)
    print(f"  lambda_l1={args.lambda_l1} lambda_iou={args.lambda_iou}", flush=True)
    print(f"  detailed_logs={args.detailed_logs}", flush=True)
    print(f"  max_merge_n={cfg.max_merge_n if cfg.max_merge_n is not None else 'unlimited'}", flush=True)
    print(f"  resume={not args.no_resume}", flush=True)
    print(f"  visualize_during_matching={args.visualize_during_matching}", flush=True)
    if args.visualize_during_matching:
        print(f"  viz_output={viz_root}", flush=True)
        print(f"  viz_save_pairs={args.viz_save_pairs}", flush=True)
        print(f"  viz_max_rows={args.viz_max_rows}", flush=True)
        print(f"  viz_panel_width={args.viz_panel_width}", flush=True)

    # q = run_matching(
    #     figma_data=Path(args.figma_data),
    #     exp_pairs=args.exp_pairs,
    #     model="qwen",
    #     output_dir=out_root,
    #     max_episodes=args.max_episodes,
    #     cfg=cfg,
    #     show_progress=not args.no_progress,
    #     progress_desc="precompute:qwen",
    #     trace_episodes=args.trace_episodes,
    #     detailed_logs=args.detailed_logs,
    #     slow_episode_sec=args.slow_episode_sec,
    #     gt_progress_every=args.gt_progress_every,
    #     gt_progress_sec=args.gt_progress_sec,
    #     num_workers=args.num_workers,
    #     visualize_episode_dir=viz_root,
    #     visualize_pair_dir=viz_pair_root,
    #     viz_max_rows=args.viz_max_rows,
    #     viz_panel_width=args.viz_panel_width,
    #     skip_existing=not args.no_resume,
    # )
    a = run_matching(
        figma_data=Path(args.figma_data),
        exp_pairs=args.exp_pairs,
        model="agent",
        output_dir=out_root,
        max_episodes=args.max_episodes,
        cfg=cfg,
        show_progress=not args.no_progress,
        progress_desc="precompute:agent",
        trace_episodes=args.trace_episodes,
        detailed_logs=args.detailed_logs,
        slow_episode_sec=args.slow_episode_sec,
        gt_progress_every=args.gt_progress_every,
        gt_progress_sec=args.gt_progress_sec,
        num_workers=args.num_workers,
        mp_start_method=args.mp_start_method,
        max_tasks_per_child=(None if int(args.max_tasks_per_child) <= 0 else int(args.max_tasks_per_child)),
        visualize_episode_dir=viz_root,
        visualize_pair_dir=viz_pair_root,
        viz_max_rows=args.viz_max_rows,
        viz_panel_width=args.viz_panel_width,
        skip_existing=not args.no_resume,
    )

    print(f"[DONE] agent episodes={a['num_episodes']}")


if __name__ == "__main__":
    main()
