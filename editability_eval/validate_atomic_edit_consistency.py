#!/usr/bin/env python3
"""Validate atomic edit consistency across GT/Qwen/Agent result rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common_utils import element_to_rgba, load_json, save_json
from .loaders import EpisodeTask, collect_episode_tasks, load_episode_elements
from .task_common import apply_edit_to_scene


def _stable_params_json(row: Dict[str, Any]) -> str:
    params = row.get("params", {})
    if not isinstance(params, dict):
        params = {}
    return json.dumps(params, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _row_key(row: Dict[str, Any], default_task: str) -> str:
    return (
        f"{str(row.get('episode_id', ''))}"
        f"::gt{int(row.get('gt_index', -1))}"
        f"::task={str(row.get('task_type', default_task))}"
        f"::params={_stable_params_json(row)}"
    )


def _rows_from_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, dict):
        rows = data.get("results", [])
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _is_grayscale_ratio(rgba: np.ndarray) -> Tuple[float, int]:
    """Return (ratio, opaque_pixels) where RGB channels are nearly equal."""
    arr = np.asarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        return 0.0, 0
    alpha = arr[..., 3] > 0
    n = int(alpha.sum())
    if n <= 0:
        return 0.0, 0
    rgb = arr[..., :3][alpha].astype(np.int16)
    delta = rgb.max(axis=1) - rgb.min(axis=1)
    gray = int((delta <= 1).sum())
    return float(gray / max(1, n)), n


class EpisodeGTCache:
    def __init__(self, task_map: Dict[str, EpisodeTask]) -> None:
        self.task_map = task_map
        self.cache: Dict[str, Tuple[List[Dict[str, Any]], Tuple[int, int]]] = {}

    def get(self, episode_id: str) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
        hit = self.cache.get(episode_id)
        if hit is not None:
            return hit
        task = self.task_map[episode_id]
        gt, _, canvas = load_episode_elements(task, model="qwen")
        out = (gt, tuple(canvas))
        self.cache[episode_id] = out
        return out


def _edit_from_row(row: Dict[str, Any], default_task: str) -> Dict[str, Any]:
    task_type = str(row.get("task_type", default_task))
    params = row.get("params", {})
    if not isinstance(params, dict):
        params = {}
    edit = {"task_type": task_type}
    edit.update(params)
    return edit


def _summarize(
    *,
    output_dir: Path,
    figma_data: Path,
    exp_pairs: Sequence[str],
    subtasks: Sequence[str],
) -> Dict[str, Any]:
    q_tasks = collect_episode_tasks(figma_data, exp_pairs, model="qwen")
    q_task_map = {t.episode_id: t for t in q_tasks}
    gt_cache = EpisodeGTCache(q_task_map)

    report: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "figma_data": str(figma_data),
        "exp_pairs": list(exp_pairs),
        "alignment_key": "episode_id + gt_index + task_type + params",
        "subtasks": {},
    }

    for subtask in subtasks:
        q_rows = _rows_from_file(output_dir / "qwen" / f"atomic_{subtask}_results.json")
        a_rows = _rows_from_file(output_dir / "agent" / f"atomic_{subtask}_results.json")
        q_map = {_row_key(r, subtask): r for r in q_rows}
        a_map = {_row_key(r, subtask): r for r in a_rows}
        aligned = sorted(set(q_map.keys()) & set(a_map.keys()))
        stats: Dict[str, int] = defaultdict(int)
        first_examples: Dict[str, Optional[Any]] = {
            "task_mismatch": None,
            "params_mismatch": None,
            "edit_mismatch": None,
            "gt_no_change": None,
        }
        recolor_no_change_samples: List[Dict[str, Any]] = []

        for k in aligned:
            rq = q_map[k]
            ra = a_map[k]
            tq = str(rq.get("task_type", subtask))
            ta = str(ra.get("task_type", subtask))
            pq = _stable_params_json(rq)
            pa = _stable_params_json(ra)
            if tq != ta:
                stats["task_mismatch"] += 1
                if first_examples["task_mismatch"] is None:
                    first_examples["task_mismatch"] = {"key": k, "q_task": tq, "a_task": ta}
                continue
            if pq != pa:
                stats["params_mismatch"] += 1
                if first_examples["params_mismatch"] is None:
                    first_examples["params_mismatch"] = {"key": k, "q_params": pq, "a_params": pa}
                continue

            edit_q = _edit_from_row(rq, subtask)
            edit_a = _edit_from_row(ra, subtask)
            if edit_q != edit_a:
                stats["edit_mismatch"] += 1
                if first_examples["edit_mismatch"] is None:
                    first_examples["edit_mismatch"] = {"key": k, "q_edit": edit_q, "a_edit": edit_a}
                continue

            eid = str(rq.get("episode_id", ""))
            gt_idx = int(rq.get("gt_index", -1))
            if eid not in q_task_map:
                stats["missing_episode"] += 1
                continue
            try:
                gt_elements, canvas = gt_cache.get(eid)
            except Exception:
                stats["episode_load_fail"] += 1
                continue
            if not (0 <= gt_idx < len(gt_elements)):
                stats["gt_idx_oob"] += 1
                continue

            stats["verified_rows"] += 1
            try:
                gt_after = apply_edit_to_scene(gt_elements, [gt_idx], canvas, edit_q)
                if subtask == "z_order":
                    before_z = int(gt_elements[gt_idx].get("z_index", gt_idx))
                    after_z = int(gt_after[gt_idx].get("z_index", gt_idx))
                    changed = before_z != after_z
                else:
                    before_rgba = element_to_rgba(gt_elements[gt_idx], canvas)
                    after_rgba = element_to_rgba(gt_after[gt_idx], canvas)
                    changed = not np.array_equal(before_rgba, after_rgba)
                if changed:
                    stats["gt_changed"] += 1
                else:
                    stats["gt_no_change"] += 1
                    if first_examples["gt_no_change"] is None:
                        first_examples["gt_no_change"] = {"key": k, "task": subtask, "params": edit_q}
                    if subtask == "recolor" and len(recolor_no_change_samples) < 5:
                        gray_ratio, opaque_pixels = _is_grayscale_ratio(before_rgba)
                        recolor_no_change_samples.append(
                            {
                                "key": k,
                                "params": dict(edit_q),
                                "grayscale_ratio": float(gray_ratio),
                                "opaque_pixels": int(opaque_pixels),
                            }
                        )
            except Exception:
                stats["gt_apply_fail"] += 1

        report["subtasks"][subtask] = {
            "qwen_rows": len(q_rows),
            "agent_rows": len(a_rows),
            "aligned_rows": len(aligned),
            "qwen_only": len(set(q_map.keys()) - set(a_map.keys())),
            "agent_only": len(set(a_map.keys()) - set(q_map.keys())),
            "stats": dict(stats),
            "first_examples": first_examples,
            "recolor_no_change_samples": recolor_no_change_samples,
        }

    return report


def _to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Atomic Edit Consistency Validation",
        "",
        f"- output_dir: `{report.get('output_dir', '')}`",
        f"- alignment_key: `{report.get('alignment_key', '')}`",
        "",
    ]
    subtasks = report.get("subtasks", {})
    for subtask in ("delete", "transition", "rotation", "opacity", "z_order", "recolor"):
        item = subtasks.get(subtask)
        if not isinstance(item, dict):
            continue
        stats = item.get("stats", {}) if isinstance(item.get("stats"), dict) else {}
        lines.append(f"## atomic_{subtask}")
        lines.append(
            f"- rows: qwen={item.get('qwen_rows', 0)} agent={item.get('agent_rows', 0)} "
            f"aligned={item.get('aligned_rows', 0)} qwen_only={item.get('qwen_only', 0)} agent_only={item.get('agent_only', 0)}"
        )
        lines.append(
            f"- key/edit mismatch: task={stats.get('task_mismatch', 0)} "
            f"params={stats.get('params_mismatch', 0)} edit={stats.get('edit_mismatch', 0)}"
        )
        lines.append(
            f"- GT replay: verified={stats.get('verified_rows', 0)} changed={stats.get('gt_changed', 0)} "
            f"no_change={stats.get('gt_no_change', 0)} apply_fail={stats.get('gt_apply_fail', 0)}"
        )
        if subtask == "recolor":
            samples = item.get("recolor_no_change_samples", [])
            if isinstance(samples, list) and samples:
                lines.append("- recolor no-change sample stats (grayscale_ratio):")
                for s in samples:
                    lines.append(
                        f"  - key={s.get('key')} grayscale_ratio={s.get('grayscale_ratio')} opaque_pixels={s.get('opaque_pixels')}"
                    )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate whether same atomic edit is applied across GT/Qwen/Agent.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--figma-data", type=Path, required=True)
    p.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    p.add_argument(
        "--subtasks",
        type=str,
        nargs="+",
        default=["delete", "transition", "rotation", "opacity", "z_order", "recolor"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = _summarize(
        output_dir=args.output_dir,
        figma_data=args.figma_data,
        exp_pairs=args.exp_pairs,
        subtasks=[str(x).strip().replace("atomic_", "") for x in args.subtasks],
    )
    out_json = args.output_dir / "atomic_edit_consistency_validation.json"
    out_md = args.output_dir / "atomic_edit_consistency_validation.md"
    save_json(out_json, report)
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"saved {out_json}")
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
