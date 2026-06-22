#!/usr/bin/env python3
"""Analyze atomic edit results by IoU bins (Qwen vs Agent)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def _to_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isfinite(f):
            return f
    return None


def _task_key(row: Dict[str, Any]) -> str:
    params = row.get("params", {})
    if not isinstance(params, dict):
        params = {}
    params_key = json.dumps(params, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return (
        f"{str(row.get('episode_id', ''))}"
        f"::gt{int(row.get('gt_index', -1))}"
        f"::task={str(row.get('task_type', ''))}"
        f"::params={params_key}"
    )


def _extract_iou(row: Dict[str, Any]) -> Optional[float]:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    return _to_float(metrics.get("iou"))


def _build_edges(start: float, end: float, step: float) -> List[float]:
    vals: List[float] = []
    x = float(start)
    while x < float(end) - 1e-12:
        vals.append(round(x, 10))
        x += float(step)
    vals.append(round(float(end), 10))
    vals.append(1.0000001)  # include upper bound
    return vals


def _bin_label(lo: float, hi: float) -> str:
    if hi > 1.0:
        return f"[{lo:.1f},1.0]"
    return f"[{lo:.1f},{hi:.1f})"


def _pick_bin_value(mode: str, q_iou: float, a_iou: float) -> float:
    if mode == "qwen":
        return q_iou
    if mode == "agent":
        return a_iou
    if mode == "min":
        return min(q_iou, a_iou)
    if mode == "max":
        return max(q_iou, a_iou)
    return (q_iou + a_iou) / 2.0  # avg


def _list_common_subtasks(merge_dir: Path) -> List[str]:
    q_dir = merge_dir / "qwen"
    a_dir = merge_dir / "agent"
    q_files = {p.name for p in q_dir.glob("atomic_*_results.json")}
    a_files = {p.name for p in a_dir.glob("atomic_*_results.json")}
    names = sorted(q_files & a_files)
    out: List[str] = []
    for name in names:
        stem = name.removeprefix("atomic_").removesuffix("_results.json")
        out.append(stem)
    return out


def _analyze_one(
    merge_dir: Path,
    subtask: str,
    edges: Sequence[float],
    bin_by: str,
) -> Dict[str, Any]:
    q_path = merge_dir / "qwen" / f"atomic_{subtask}_results.json"
    a_path = merge_dir / "agent" / f"atomic_{subtask}_results.json"
    q_rows = _load_rows(q_path)
    a_rows = _load_rows(a_path)

    q_map = {_task_key(r): r for r in q_rows}
    a_map = {_task_key(r): r for r in a_rows}
    keys = sorted(set(q_map.keys()) & set(a_map.keys()))

    bins: List[Dict[str, Any]] = []
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        bins.append(
            {
                "label": _bin_label(lo, hi),
                "lo": lo,
                "hi": hi,
                "n": 0,
                "qwen_win": 0,
                "agent_win": 0,
                "tie": 0,
                "qwen_iou_sum": 0.0,
                "agent_iou_sum": 0.0,
            }
        )

    no_iou = 0
    below_start = 0
    for k in keys:
        q_iou = _extract_iou(q_map[k])
        a_iou = _extract_iou(a_map[k])
        if q_iou is None or a_iou is None:
            no_iou += 1
            continue
        v = _pick_bin_value(bin_by, q_iou, a_iou)
        if v < float(edges[0]):
            below_start += 1
            continue
        chosen = None
        for b in bins:
            if v >= b["lo"] and (v < b["hi"] or b["hi"] > 1.0):
                chosen = b
                break
        if chosen is None:
            continue
        chosen["n"] += 1
        chosen["qwen_iou_sum"] += q_iou
        chosen["agent_iou_sum"] += a_iou
        if a_iou > q_iou:
            chosen["agent_win"] += 1
        elif q_iou > a_iou:
            chosen["qwen_win"] += 1
        else:
            chosen["tie"] += 1

    for b in bins:
        n = int(b["n"])
        if n > 0:
            b["qwen_iou_mean"] = float(b["qwen_iou_sum"] / n)
            b["agent_iou_mean"] = float(b["agent_iou_sum"] / n)
            b["agent_win_rate"] = float(b["agent_win"] / n)
            b["qwen_win_rate"] = float(b["qwen_win"] / n)
        else:
            b["qwen_iou_mean"] = None
            b["agent_iou_mean"] = None
            b["agent_win_rate"] = None
            b["qwen_win_rate"] = None
        del b["qwen_iou_sum"]
        del b["agent_iou_sum"]

    return {
        "subtask": subtask,
        "qwen_rows": len(q_rows),
        "agent_rows": len(a_rows),
        "aligned_pairs": len(keys),
        "pairs_without_iou": int(no_iou),
        "pairs_below_start": int(below_start),
        "bins": bins,
    }


def _print_report(merge: str, item: Dict[str, Any]) -> None:
    print(
        f"merge={merge} subtask={item['subtask']} "
        f"aligned={item['aligned_pairs']} no_iou={item['pairs_without_iou']} below_start={item['pairs_below_start']}"
    )
    if item["aligned_pairs"] <= 0:
        print("  (no aligned pairs yet)")
        return
    if all(int(b["n"]) == 0 for b in item["bins"]):
        print("  (no samples in requested iou bins yet)")
        return
    for b in item["bins"]:
        n = int(b["n"])
        if n <= 0:
            continue
        print(
            "  "
            f"{b['label']}: n={n} "
            f"agent_win={b['agent_win']} qwen_win={b['qwen_win']} tie={b['tie']} "
            f"agent_win_rate={b['agent_win_rate']:.3f} "
            f"iou_mean(qwen={b['qwen_iou_mean']:.4f}, agent={b['agent_iou_mean']:.4f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze atomic results by IoU bins (Qwen vs Agent).")
    parser.add_argument("--root", type=str, default="editability_results/atomic_merge_sweep")
    parser.add_argument("--merges", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--subtasks", type=str, nargs="+", default=None, help="Optional subset: delete transition rotation opacity z_order recolor")
    parser.add_argument("--bin-start", type=float, default=0.1)
    parser.add_argument("--bin-end", type=float, default=0.7)
    parser.add_argument("--bin-step", type=float, default=0.1)
    parser.add_argument("--bin-by", type=str, choices=["avg", "qwen", "agent", "min", "max"], default="avg")
    parser.add_argument("--json-out", type=str, default=None, help="Optional output path for machine-readable report")
    args = parser.parse_args()

    edges = _build_edges(args.bin_start, args.bin_end, args.bin_step)
    root = Path(args.root)
    report: Dict[str, Any] = {
        "root": str(root),
        "bin_by": args.bin_by,
        "bin_spec": {"start": args.bin_start, "end": args.bin_end, "step": args.bin_step},
        "merges": {},
    }

    for m in args.merges:
        merge_name = f"merge_{int(m)}"
        merge_dir = root / merge_name
        subtask_names = _list_common_subtasks(merge_dir)
        if args.subtasks:
            allow = set(args.subtasks)
            subtask_names = [s for s in subtask_names if s in allow]
        merge_items: List[Dict[str, Any]] = []
        for subtask in subtask_names:
            item = _analyze_one(merge_dir, subtask, edges, args.bin_by)
            merge_items.append(item)
            _print_report(merge_name, item)
        if not merge_items:
            print(f"merge={merge_name}: no common atomic_*_results.json yet")
        report["merges"][merge_name] = merge_items

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved json report: {out_path}")


if __name__ == "__main__":
    main()

