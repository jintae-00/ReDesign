#!/usr/bin/env python3
"""Save GT:Qwen greedy match pairs for editability tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from .match_runner import run_matching
from .matching_core import MatchConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="GT:Qwen matching saver (editability)")
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
    parser.add_argument("--num-workers", type=int, default=1, help="Episode-level parallel workers")
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

    summary = run_matching(
        figma_data=Path(args.figma_data),
        exp_pairs=args.exp_pairs,
        model="qwen",
        output_dir=Path(args.output),
        max_episodes=args.max_episodes,
        cfg=cfg,
        num_workers=args.num_workers,
        skip_existing=not args.no_resume,
    )
    print(f"[DONE] qwen matching episodes={summary['num_episodes']} -> {Path(args.output) / 'qwen'}")


if __name__ == "__main__":
    main()
