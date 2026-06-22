#!/usr/bin/env python3
"""Precompute GT:Qwen/Agent matching pairs with optional merge sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

from .common_utils import load_json, save_json
from .match_runner import run_matching
from .matching_core import MatchConfig


def _merge_label(max_merge_n: Optional[int]) -> str:
    return f"merge_{int(max_merge_n)}" if max_merge_n is not None else "merge_max"


def _parse_merge_value(raw: str) -> Optional[int]:
    tok = str(raw).strip().lower()
    if tok in {"max", "unlimited", "none", "all", "inf"}:
        return None
    iv = int(tok)
    return None if iv <= 0 else int(iv)


def _normalize_merge_values(raw_values: Sequence[Optional[int]]) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for raw in raw_values:
        if raw is None:
            v: Optional[int] = None
        else:
            iv = int(raw)
            v = None if iv <= 0 else iv
        if v not in out:
            out.append(v)
    return out


def _dedupe_strs(xs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def _choose_base_merge(merge_values: Sequence[Optional[int]]) -> Optional[int]:
    if any(v is None for v in merge_values):
        return None
    finite = [int(v) for v in merge_values if v is not None]
    if not finite:
        return None
    return max(finite)


def _run_editability_tasks(
    *,
    figma_data: Path,
    exp_pairs: Sequence[str],
    match_root: Path,
    output_root: Path,
    tasks: Sequence[str],
    max_episodes: Optional[int],
    seed: int,
    num_workers: int,
) -> None:
    module_by_task = {
        "atomic": "editability_eval.run_atomic_edit",
        "text": "editability_eval.run_text_edit",
        "svg": "editability_eval.run_svg_edit",
    }
    for task_name in tasks:
        module_name = module_by_task[task_name]
        task_output = output_root / task_name
        task_output.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            module_name,
            "--figma-data",
            str(figma_data),
            "--exp-pairs",
            *list(exp_pairs),
            "--match-root",
            str(match_root),
            "--output",
            str(task_output),
            "--seed",
            str(int(seed)),
            "--num-workers",
            str(max(1, int(num_workers))),
        ]
        if max_episodes is not None:
            cmd.extend(["--max-episodes", str(int(max_episodes))])
        print(f"[RUN] editability task={task_name} output={task_output}", flush=True)
        subprocess.run(cmd, check=True)


def _trim_match_for_merge_cap(match: Dict[str, Any], max_merge_n: Optional[int]) -> Dict[str, Any]:
    if max_merge_n is None:
        return dict(match)
    cap = int(max_merge_n)
    if cap <= 0:
        return dict(match)

    out = dict(match)
    selected_indices = out.get("selected_pred_indices", [])
    selected_ids = out.get("selected_pred_ids", [])
    if not isinstance(selected_indices, list) or not isinstance(selected_ids, list):
        return out

    if len(selected_indices) <= cap:
        return out

    trace = out.get("merge_trace")
    if not isinstance(trace, list) or len(trace) < cap:
        raise ValueError(
            "Cannot materialize lower merge cap because merge_trace is missing/short. "
            "Re-run base merge with --no-resume or use a fresh --output directory."
        )

    step = trace[cap - 1]
    if not isinstance(step, dict):
        raise ValueError("Invalid merge_trace entry format.")
    metrics = step.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Invalid merge_trace metrics format.")

    out["selected_pred_indices"] = selected_indices[:cap]
    out["selected_pred_ids"] = selected_ids[:cap]
    out["merged_metrics"] = dict(metrics)
    return out


def _materialize_merge_variant(
    *,
    src_root: Path,
    dst_root: Path,
    model: str,
    max_merge_n: Optional[int],
    skip_existing: bool,
    src_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    src_ep_dir = src_root / model / "episodes"
    dst_ep_dir = dst_root / model / "episodes"
    dst_ep_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    src_eps = sorted(src_ep_dir.glob("*.json"))
    for src_path in src_eps:
        dst_path = dst_ep_dir / src_path.name
        if skip_existing and dst_path.exists():
            skipped += 1
            continue
        payload = load_json(src_path)
        matches = payload.get("matches", [])
        if isinstance(matches, list):
            payload["matches"] = [
                _trim_match_for_merge_cap(m, max_merge_n) if isinstance(m, dict) else m for m in matches
            ]
        save_json(dst_path, payload)
        written += 1

    if src_summary is None:
        src_summary_path = src_root / model / "summary.json"
        if src_summary_path.exists():
            loaded = load_json(src_summary_path)
            if isinstance(loaded, dict):
                src_summary = loaded

    if isinstance(src_summary, dict):
        out_summary = json.loads(json.dumps(src_summary))
        cfg = out_summary.get("config", {})
        if not isinstance(cfg, dict):
            cfg = {}
        cfg["max_merge_n"] = max_merge_n
        out_summary["config"] = cfg
        out_summary["materialized_from"] = str(src_root)
        out_summary["materialized_written"] = int(written)
        out_summary["materialized_skipped"] = int(skipped)
        save_json(dst_root / model / "summary.json", out_summary)

    return {
        "episodes_total": len(src_eps),
        "episodes_written": int(written),
        "episodes_skipped": int(skipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute matching pairs for one or more models")
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", choices=["qwen", "agent"], default=["qwen"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--lambda-l1", type=float, default=0.5)
    parser.add_argument("--lambda-iou", type=float, default=0.5, help="Weight for IoU term in matching cost: lambda_iou * (1 - iou).")
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
    parser.add_argument(
        "--max-merge-values",
        type=_parse_merge_value,
        nargs="+",
        default=None,
        help=(
            "Sweep merge caps. Example: --max-merge-values 4 8 16 32 max. "
            "Use 'max' (or <=0) for unlimited merge."
        ),
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
    parser.add_argument("--run-editability", action="store_true", help="After matching, run atomic/text/svg editability for each merge setting.")
    parser.add_argument("--editability-tasks", type=str, nargs="+", choices=["atomic", "text", "svg"], default=["atomic", "text", "svg"])
    parser.add_argument("--editability-output", type=str, default=None, help="Editability output root (default: <output>/editability)")
    parser.add_argument("--eval-seed", type=int, default=123, help="Seed passed to editability task runners")
    parser.add_argument("--eval-num-workers", type=int, default=None, help="num-workers passed to editability task runners (default: --num-workers)")
    args = parser.parse_args()

    if args.max_merge_values is not None and args.max_merge_n is not None:
        parser.error("Use either --max-merge-n or --max-merge-values, not both.")

    merge_values = _normalize_merge_values(args.max_merge_values if args.max_merge_values is not None else [args.max_merge_n])
    models = _dedupe_strs(args.models)
    tasks = _dedupe_strs(args.editability_tasks)

    if args.run_editability and set(models) != {"qwen", "agent"}:
        parser.error("--run-editability requires --models qwen agent because edit runners compare both models.")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    editability_root = Path(args.editability_output) if args.editability_output else (out_root / "editability")
    figma_data = Path(args.figma_data)

    print("[START] precompute matching", flush=True)
    print(f"  figma_data={args.figma_data}", flush=True)
    print(f"  output={out_root}", flush=True)
    print(f"  models={models}", flush=True)
    print(f"  exp_pairs={len(args.exp_pairs)}", flush=True)
    print(f"  num_workers={args.num_workers}", flush=True)
    print(f"  mp_start_method={args.mp_start_method}", flush=True)
    print(
        f"  max_tasks_per_child={(None if int(args.max_tasks_per_child) <= 0 else int(args.max_tasks_per_child))}",
        flush=True,
    )
    print(f"  merge_values={[_merge_label(v) for v in merge_values]}", flush=True)
    print("  backend=CPU(numpy)", flush=True)
    print(f"  cost=lambda_l1*l1_union_region + lambda_iou*(1-iou) (mode={args.l1_mode})", flush=True)
    print(f"  lambda_l1={args.lambda_l1} lambda_iou={args.lambda_iou}", flush=True)
    print(f"  detailed_logs={args.detailed_logs}", flush=True)
    print(f"  resume={not args.no_resume}", flush=True)
    print(f"  visualize_during_matching={args.visualize_during_matching}", flush=True)
    if args.run_editability:
        print(f"  run_editability=True tasks={tasks} editability_output={editability_root}", flush=True)

    multi_merge = len(merge_values) > 1
    fast_sweep_mode = multi_merge
    if args.visualize_during_matching and multi_merge:
        fast_sweep_mode = False
        print("[INFO] visualization is enabled; running full compute per merge to keep merge-specific viz outputs", flush=True)

    run_records: List[Dict[str, object]] = []

    if not fast_sweep_mode:
        for merge_n in merge_values:
            cfg = MatchConfig(
                lambda_l1=args.lambda_l1,
                lambda_iou=args.lambda_iou,
                l1_mode=args.l1_mode,
                max_merge_n=merge_n,
                min_gt_overlap=args.min_gt_overlap,
                min_cost_improve=args.min_cost_improve,
            )
            merge_label = _merge_label(cfg.max_merge_n)
            run_out_root = (out_root / merge_label) if multi_merge else out_root
            run_out_root.mkdir(parents=True, exist_ok=True)

            viz_root = None
            viz_pair_root = None
            if args.visualize_during_matching:
                if args.viz_output:
                    base_viz_root = Path(args.viz_output)
                    viz_root = (base_viz_root / merge_label) if multi_merge else base_viz_root
                else:
                    viz_root = run_out_root / "viz"
                viz_root.mkdir(parents=True, exist_ok=True)
                if args.viz_save_pairs:
                    viz_pair_root = viz_root

            print(f"[RUN] merge={merge_label} output={run_out_root}", flush=True)
            print(f"  max_merge_n={cfg.max_merge_n if cfg.max_merge_n is not None else 'unlimited'}", flush=True)

            record: Dict[str, object] = {
                "merge_label": merge_label,
                "max_merge_n": cfg.max_merge_n,
                "match_root": str(run_out_root),
                "matching": {},
            }

            for model_name in models:
                summary = run_matching(
                    figma_data=figma_data,
                    exp_pairs=args.exp_pairs,
                    model=model_name,
                    output_dir=run_out_root,
                    max_episodes=args.max_episodes,
                    cfg=cfg,
                    show_progress=not args.no_progress,
                    progress_desc=f"precompute:{model_name}:{merge_label}",
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
                cast_matching = record.get("matching")
                if isinstance(cast_matching, dict):
                    cast_matching[model_name] = summary
                print(f"[DONE] merge={merge_label} model={model_name} episodes={summary['num_episodes']}", flush=True)

            if args.run_editability:
                eval_workers = args.eval_num_workers if args.eval_num_workers is not None else args.num_workers
                eval_case_root = editability_root / merge_label
                eval_case_root.mkdir(parents=True, exist_ok=True)
                _run_editability_tasks(
                    figma_data=figma_data,
                    exp_pairs=args.exp_pairs,
                    match_root=run_out_root,
                    output_root=eval_case_root,
                    tasks=tasks,
                    max_episodes=args.max_episodes,
                    seed=int(args.eval_seed),
                    num_workers=int(eval_workers),
                )
                record["editability_root"] = str(eval_case_root)
                record["editability_tasks"] = list(tasks)
                print(f"[DONE] merge={merge_label} editability_root={eval_case_root}", flush=True)

            run_records.append(record)
    else:
        base_merge_n = _choose_base_merge(merge_values)
        base_label = _merge_label(base_merge_n)
        base_root = out_root / base_label
        base_root.mkdir(parents=True, exist_ok=True)

        base_cfg = MatchConfig(
            lambda_l1=args.lambda_l1,
            lambda_iou=args.lambda_iou,
            l1_mode=args.l1_mode,
            max_merge_n=base_merge_n,
            min_gt_overlap=args.min_gt_overlap,
            min_cost_improve=args.min_cost_improve,
        )

        print(
            "[SWEEP] single-pass mode: run once at "
            f"{base_label}, then materialize other merges by truncating merge trace",
            flush=True,
        )

        base_summaries: Dict[str, Dict[str, Any]] = {}
        for model_name in models:
            summary = run_matching(
                figma_data=figma_data,
                exp_pairs=args.exp_pairs,
                model=model_name,
                output_dir=base_root,
                max_episodes=args.max_episodes,
                cfg=base_cfg,
                show_progress=not args.no_progress,
                progress_desc=f"precompute:{model_name}:{base_label}",
                trace_episodes=args.trace_episodes,
                detailed_logs=args.detailed_logs,
                slow_episode_sec=args.slow_episode_sec,
                gt_progress_every=args.gt_progress_every,
                gt_progress_sec=args.gt_progress_sec,
                num_workers=args.num_workers,
                mp_start_method=args.mp_start_method,
                max_tasks_per_child=(None if int(args.max_tasks_per_child) <= 0 else int(args.max_tasks_per_child)),
                visualize_episode_dir=None,
                visualize_pair_dir=None,
                viz_max_rows=args.viz_max_rows,
                viz_panel_width=args.viz_panel_width,
                skip_existing=not args.no_resume,
            )
            base_summaries[model_name] = summary
            print(f"[DONE] base merge={base_label} model={model_name} episodes={summary['num_episodes']}", flush=True)

        for merge_n in merge_values:
            merge_label = _merge_label(merge_n)
            merge_root = out_root / merge_label
            merge_root.mkdir(parents=True, exist_ok=True)

            record: Dict[str, object] = {
                "merge_label": merge_label,
                "max_merge_n": merge_n,
                "match_root": str(merge_root),
                "matching": {},
            }

            if merge_n == base_merge_n:
                for model_name in models:
                    cast_matching = record.get("matching")
                    if isinstance(cast_matching, dict):
                        cast_matching[model_name] = base_summaries.get(model_name, {})
                print(f"[SKIP] merge={merge_label} uses base outputs directly", flush=True)
            else:
                for model_name in models:
                    stats = _materialize_merge_variant(
                        src_root=base_root,
                        dst_root=merge_root,
                        model=model_name,
                        max_merge_n=merge_n,
                        skip_existing=not args.no_resume,
                        src_summary=base_summaries.get(model_name),
                    )
                    cast_matching = record.get("matching")
                    if isinstance(cast_matching, dict):
                        cast_matching[model_name] = {
                            "materialized": True,
                            "from_merge": base_label,
                            "episodes_total": stats["episodes_total"],
                            "episodes_written": stats["episodes_written"],
                            "episodes_skipped": stats["episodes_skipped"],
                        }
                print(f"[DONE] merge={merge_label} materialized from {base_label}", flush=True)

            if args.run_editability:
                eval_workers = args.eval_num_workers if args.eval_num_workers is not None else args.num_workers
                eval_case_root = editability_root / merge_label
                eval_case_root.mkdir(parents=True, exist_ok=True)
                _run_editability_tasks(
                    figma_data=figma_data,
                    exp_pairs=args.exp_pairs,
                    match_root=merge_root,
                    output_root=eval_case_root,
                    tasks=tasks,
                    max_episodes=args.max_episodes,
                    seed=int(args.eval_seed),
                    num_workers=int(eval_workers),
                )
                record["editability_root"] = str(eval_case_root)
                record["editability_tasks"] = list(tasks)
                print(f"[DONE] merge={merge_label} editability_root={eval_case_root}", flush=True)

            run_records.append(record)

    summary_path = out_root / "precompute_sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"runs": run_records}, f, ensure_ascii=False, indent=2)
    print(f"[DONE] wrote summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
