#!/usr/bin/env python3
"""Generate 2x3 triplet visualizations from existing atomic result outputs."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common_utils import load_json, save_json
from .run_atomic_edit_radnom import _build_pair_meta_index, _save_triplet_visualizations_for_subtask, _triplet_task_key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Backfill triplet visualizations (GT/Qwen/Agent x Before/After) "
            "from existing atomic overview/result files."
        )
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Atomic run output directory (contains atomic_qwen_overview.json / atomic_agent_overview.json).",
    )
    p.add_argument(
        "--subtasks",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset (e.g., delete transition rotation opacity z_order recolor).",
    )
    p.add_argument("--cell-width", type=int, default=520)
    p.add_argument("--cell-height", type=int, default=300)
    p.add_argument("--max-per-subtask", type=int, default=None)
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "pair", "reconstruct"],
        help=(
            "pair: use saved element_pairs images only. "
            "reconstruct: render from scene reconstruction. "
            "auto: use pair when all subtasks have pair dirs, otherwise reconstruct."
        ),
    )
    p.add_argument("--figma-data", type=Path, default=None, help="Required for reconstruct backend.")
    p.add_argument("--exp-pairs", type=str, nargs="+", default=None, help="Required for reconstruct backend.")
    p.add_argument("--match-root", type=Path, default=None, help="Optional match root to rank/filter by GT area.")
    p.add_argument("--reconstruct-log-every", type=int, default=1, help="Progress log interval passed to reconstruct backend.")
    p.add_argument("--reconstruct-num-workers", type=int, default=1, help="Parallel workers passed to reconstruct backend.")
    p.add_argument(
        "--reconstruct-parallel-backend",
        type=str,
        default="process",
        choices=["thread", "process"],
        help="Parallel backend passed to reconstruct backend.",
    )
    p.add_argument(
        "--selection-mode",
        type=str,
        default="lexicographic",
        choices=["lexicographic", "largest_gt_area", "largest_gt_area_then_l1_gap"],
        help="Case ordering strategy before truncating to --max-per-subtask.",
    )
    p.add_argument("--min-gt-area", type=float, default=0.0, help="Optional lower-bound filter on GT area.")
    p.add_argument("--start-rank", type=int, default=0, help="Skip first N ranked cases per subtask.")
    p.add_argument(
        "--agent-win-metric",
        type=str,
        choices=["l1", "l2", "source_l1", "target_l1", "avg_l1", "source_l2", "target_l2", "avg_l2"],
        default=None,
        help=(
            "Optional filter: keep only cases where (qwen_metric - agent_metric) "
            ">= --min-agent-win-gap."
        ),
    )
    p.add_argument(
        "--min-agent-win-gap",
        type=float,
        default=0.1,
        help="Minimum required (qwen_metric - agent_metric) when --agent-win-metric is set.",
    )
    p.add_argument(
        "--recolor-prioritize-gt-delta",
        action="store_true",
        help="For recolor subtask, prioritize cases with larger GT recolor delta.",
    )
    p.add_argument(
        "--recolor-min-gt-delta",
        type=float,
        default=0.0,
        help="For recolor subtask, optional minimum GT recolor delta filter.",
    )
    p.add_argument(
        "--recolor-delta-max-candidates",
        type=int,
        default=0,
        help="For recolor delta scoring, score only top-N pre-ranked candidates (0 means all).",
    )
    p.add_argument(
        "--recolor-delta-num-workers",
        type=int,
        default=0,
        help="Workers for recolor delta scoring. 0 means use reconstruct workers.",
    )
    return p.parse_args()


def _norm_subtask(s: str) -> str:
    x = str(s).strip()
    if x.startswith("atomic_"):
        x = x[len("atomic_") :]
    return x


def _to_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _build_gt_area_index(match_root: Optional[Path]) -> Dict[Tuple[str, int], float]:
    out: Dict[Tuple[str, int], float] = {}
    if match_root is None:
        return out
    epi_dir = match_root / "qwen" / "episodes"
    if not epi_dir.exists():
        return out
    for p in sorted(epi_dir.glob("*.json")):
        try:
            payload = load_json(p)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        eid = str(payload.get("episode_id", ""))
        if not eid:
            continue
        matches = payload.get("matches", [])
        if not isinstance(matches, list):
            continue
        for m in matches:
            if not isinstance(m, dict):
                continue
            try:
                gt_idx = int(m.get("gt_index"))
                area = float(m.get("gt_area"))
            except Exception:
                continue
            if math.isfinite(area):
                out[(eid, gt_idx)] = area
    return out


def _score_key(
    k: str,
    *,
    q_map: Dict[str, Dict[str, Any]],
    a_map: Dict[str, Dict[str, Any]],
    gt_area_idx: Dict[Tuple[str, int], float],
    selection_mode: str,
) -> Tuple[float, float]:
    rq = q_map[k]
    ra = a_map[k]
    eid = str(rq.get("episode_id", ""))
    gt_idx = int(rq.get("gt_index", -1))
    area = float(gt_area_idx.get((eid, gt_idx), -1.0))
    q_l1 = _to_float((rq.get("metrics", {}) if isinstance(rq.get("metrics"), dict) else {}).get("l1"))
    a_l1 = _to_float((ra.get("metrics", {}) if isinstance(ra.get("metrics"), dict) else {}).get("l1"))
    gap = abs(a_l1 - q_l1) if math.isfinite(q_l1) and math.isfinite(a_l1) else 0.0
    if selection_mode == "largest_gt_area_then_l1_gap":
        return (area, gap)
    if selection_mode == "largest_gt_area":
        return (area, 0.0)
    return (0.0, 0.0)


def _run_reconstruct_backend(args: argparse.Namespace, subtasks: List[str]) -> None:
    if args.figma_data is None or not args.exp_pairs:
        raise ValueError("--backend reconstruct/auto requires --figma-data and --exp-pairs.")
    cmd = [
        sys.executable,
        "-m",
        "editability_eval.generate_atomic_triplet_viz_reconstruct",
        "--output-dir",
        str(args.output_dir),
        "--figma-data",
        str(args.figma_data),
        "--exp-pairs",
        *list(args.exp_pairs),
        "--subtasks",
        *[_norm_subtask(s) for s in subtasks],
        "--cell-width",
        str(int(args.cell_width)),
        "--cell-height",
        str(int(args.cell_height)),
        "--selection-mode",
        str(args.selection_mode),
        "--min-gt-area",
        str(float(args.min_gt_area)),
        "--start-rank",
        str(int(args.start_rank)),
    ]
    if args.max_per_subtask is not None:
        cmd += ["--max-per-subtask", str(int(args.max_per_subtask))]
    cmd += ["--num-workers", str(max(1, int(args.reconstruct_num_workers)))]
    cmd += ["--parallel-backend", str(args.reconstruct_parallel_backend)]
    cmd += ["--log-every", str(max(1, int(args.reconstruct_log_every)))]
    if args.match_root is not None:
        cmd += ["--match-root", str(args.match_root)]
    if args.agent_win_metric:
        cmd += ["--agent-win-metric", str(args.agent_win_metric), "--min-agent-win-gap", str(float(args.min_agent_win_gap))]
    if bool(args.recolor_prioritize_gt_delta):
        cmd += ["--recolor-prioritize-gt-delta"]
    if float(args.recolor_min_gt_delta) > 0.0:
        cmd += ["--recolor-min-gt-delta", str(float(args.recolor_min_gt_delta))]
    if int(args.recolor_delta_max_candidates) > 0:
        cmd += ["--recolor-delta-max-candidates", str(int(args.recolor_delta_max_candidates))]
    if int(args.recolor_delta_num_workers) > 0:
        cmd += ["--recolor-delta-num-workers", str(int(args.recolor_delta_num_workers))]
    print("[triplet-viz] backend=reconstruct cmd=" + " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    q_path = out_dir / "atomic_qwen_overview.json"
    a_path = out_dir / "atomic_agent_overview.json"
    if not q_path.exists() or not a_path.exists():
        raise FileNotFoundError(
            f"Missing overview files under {out_dir}. "
            "Need both atomic_qwen_overview.json and atomic_agent_overview.json."
        )

    q_over = load_json(q_path)
    a_over = load_json(a_path)
    if not isinstance(q_over, dict) or not isinstance(a_over, dict):
        raise ValueError("Overview json format invalid.")

    all_subtasks = sorted(set(q_over.keys()) & set(a_over.keys()))
    if args.subtasks:
        want = {_norm_subtask(x) for x in args.subtasks}
        subtasks = [s for s in all_subtasks if _norm_subtask(s) in want]
    else:
        subtasks = all_subtasks

    if str(args.backend) == "reconstruct":
        _run_reconstruct_backend(args, subtasks)
        return
    if str(args.backend) == "auto":
        needs_reconstruct = False
        for s in subtasks:
            s_norm = _norm_subtask(s)
            q_root = out_dir / "qwen" / f"atomic_{s_norm}" / "element_pairs"
            a_root = out_dir / "agent" / f"atomic_{s_norm}" / "element_pairs"
            q_count = len([d for d in q_root.iterdir() if d.is_dir()]) if q_root.exists() else 0
            a_count = len([d for d in a_root.iterdir() if d.is_dir()]) if a_root.exists() else 0
            if q_count == 0 or a_count == 0:
                needs_reconstruct = True
                break
        if needs_reconstruct:
            print("[triplet-viz] backend=auto detected missing pair dirs; switching to reconstruct backend.")
            _run_reconstruct_backend(args, subtasks)
            return

    gt_area_idx = _build_gt_area_index(args.match_root)

    summary: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "subtasks": subtasks,
        "cell_width": int(args.cell_width),
        "cell_height": int(args.cell_height),
        "max_per_subtask": args.max_per_subtask,
        "agent_win_metric": str(args.agent_win_metric) if args.agent_win_metric else None,
        "min_agent_win_gap": float(args.min_agent_win_gap),
        "recolor_prioritize_gt_delta": bool(args.recolor_prioritize_gt_delta),
        "recolor_min_gt_delta": float(args.recolor_min_gt_delta),
        "recolor_delta_max_candidates": int(args.recolor_delta_max_candidates),
        "recolor_delta_num_workers": int(args.recolor_delta_num_workers),
        "results": {},
    }

    for s in subtasks:
        q_payload = q_over.get(s, {})
        a_payload = a_over.get(s, {})
        q_rows = q_payload.get("results", []) if isinstance(q_payload, dict) else []
        a_rows = a_payload.get("results", []) if isinstance(a_payload, dict) else []
        if not isinstance(q_rows, list):
            q_rows = []
        if not isinstance(a_rows, list):
            a_rows = []
        q_rows = [r for r in q_rows if isinstance(r, dict)]
        a_rows = [r for r in a_rows if isinstance(r, dict)]

        aligned_before = 0
        filtered_aligned = 0
        s_norm = _norm_subtask(s)
        q_pair_idx = _build_pair_meta_index(out_dir / "qwen" / f"atomic_{s_norm}" / "element_pairs", default_task=s_norm)
        a_pair_idx = _build_pair_meta_index(out_dir / "agent" / f"atomic_{s_norm}" / "element_pairs", default_task=s_norm)

        q_map = {_triplet_task_key(r, default_task=s_norm): r for r in q_rows}
        a_map = {_triplet_task_key(r, default_task=s_norm): r for r in a_rows}
        keys = sorted(set(q_map.keys()) & set(a_map.keys()))
        aligned_before = len(keys)

        if args.agent_win_metric:
            metric = str(args.agent_win_metric)
            min_gap = float(args.min_agent_win_gap)
            kept_keys: List[str] = []
            for k in keys:
                q_metrics = q_map[k].get("metrics", {})
                a_metrics = a_map[k].get("metrics", {})
                if not isinstance(q_metrics, dict) or not isinstance(a_metrics, dict):
                    continue
                qv = _to_float(q_metrics.get(metric))
                av = _to_float(a_metrics.get(metric))
                if not (math.isfinite(qv) and math.isfinite(av)):
                    continue
                if (qv - av) >= min_gap:
                    kept_keys.append(k)
            keys = kept_keys

        if str(args.selection_mode) != "lexicographic":
            keys = sorted(
                keys,
                key=lambda k: _score_key(
                    k,
                    q_map=q_map,
                    a_map=a_map,
                    gt_area_idx=gt_area_idx,
                    selection_mode=str(args.selection_mode),
                ),
                reverse=True,
            )

        if float(args.min_gt_area) > 0.0:
            keys = [
                k
                for k in keys
                if float(gt_area_idx.get((str(q_map[k].get("episode_id", "")), int(q_map[k].get("gt_index", -1))), -1.0))
                >= float(args.min_gt_area)
            ]

        if int(args.start_rank) > 0:
            keys = keys[int(args.start_rank) :]

        # Pair backend never renders placeholders: keep only keys with both qwen/agent pair dirs.
        keys = [k for k in keys if k in q_pair_idx and k in a_pair_idx]
        filtered_aligned = len(keys)

        q_rows = [q_map[k] for k in keys]
        a_rows = [a_map[k] for k in keys]

        q_payload_use = dict(q_payload) if isinstance(q_payload, dict) else {}
        a_payload_use = dict(a_payload) if isinstance(a_payload, dict) else {}
        q_payload_use["results"] = q_rows
        a_payload_use["results"] = a_rows

        # Avoid stale placeholders from previous runs.
        viz_dir = out_dir / "triplet_pair_viz" / f"atomic_{s_norm}"
        if viz_dir.exists():
            shutil.rmtree(viz_dir)

        info = _save_triplet_visualizations_for_subtask(
            output_dir=out_dir,
            subtask_name=s_norm,
            q_payload=q_payload_use,
            a_payload=a_payload_use,
            cell_w=int(args.cell_width),
            cell_h=int(args.cell_height),
            max_cases=args.max_per_subtask,
        )
        info["aligned_before_filter"] = int(aligned_before)
        info["aligned_after_filter"] = int(filtered_aligned)
        summary["results"][s] = info
        print(
            f"[triplet-viz] subtask={s} aligned={info.get('aligned_cases')} "
            f"before_filter={info.get('aligned_before_filter')} "
            f"after_filter={info.get('aligned_after_filter')} "
            f"selection_mode={args.selection_mode} "
            f"requested={info.get('requested_cases')} created={info.get('created')} "
            f"missing_q={info.get('missing_q_pair_dirs')} missing_a={info.get('missing_a_pair_dirs')} "
            f"dir={info.get('viz_dir')}"
        )

    save_json(out_dir / "atomic_triplet_viz_summary.json", summary)
    print(f"Saved summary: {out_dir / 'atomic_triplet_viz_summary.json'}")


if __name__ == "__main__":
    main()
