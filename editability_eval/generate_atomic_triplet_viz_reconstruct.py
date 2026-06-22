#!/usr/bin/env python3
"""Generate 2x3 atomic triplet visualizations by reconstructing scenes from result rows."""

from __future__ import annotations

import argparse
import os
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common_utils import load_json, save_json
from .loaders import collect_episode_tasks, load_episode_elements
from .subtasks.common import _apply_evalfigma_postprocess_to_scene
from .task_common import apply_edit_to_scene, element_to_rgba, render_scene_rgba, union_mask_from_indices


def _stable_params_json(row: Dict[str, Any]) -> str:
    params = row.get("params", {})
    if not isinstance(params, dict):
        params = {}
    return json.dumps(params, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _triplet_key(row: Dict[str, Any], default_task: str) -> str:
    return (
        f"{str(row.get('episode_id', ''))}"
        f"::gt{int(row.get('gt_index', -1))}"
        f"::task={str(row.get('task_type', default_task))}"
        f"::params={_stable_params_json(row)}"
    )


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v == v:
            return v
    except Exception:
        pass
    return float(default)


def _norm_subtask(s: str) -> str:
    x = str(s).strip()
    if x.startswith("atomic_"):
        x = x[len("atomic_") :]
    return x


def _rows_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        xs = payload.get("results", [])
    elif isinstance(payload, list):
        xs = payload
    else:
        xs = []
    if not isinstance(xs, list):
        return []
    return [x for x in xs if isinstance(x, dict)]


def _indices_from_row(row: Dict[str, Any]) -> List[int]:
    xs = row.get("pred_indices")
    if isinstance(xs, list):
        out: List[int] = []
        for x in xs:
            try:
                xi = int(x)
            except Exception:
                continue
            if xi >= 0:
                out.append(xi)
        if out:
            return out
    p = row.get("pred_index")
    try:
        pi = int(p)
    except Exception:
        pi = -1
    return [pi] if pi >= 0 else []


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
            except Exception:
                continue
            area = _to_float(m.get("gt_area"), default=-1.0)
            if area < 0:
                continue
            out[(eid, gt_idx)] = float(area)
    return out


def _fit_rgba_on_white(rgba: np.ndarray, width: int, height: int) -> Image.Image:
    img = Image.fromarray(np.asarray(rgba, dtype=np.uint8), "RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, img).convert("RGB")
    scale = min(width / float(rgb.width), height / float(rgb.height))
    nw = max(1, int(round(rgb.width * scale)))
    nh = max(1, int(round(rgb.height * scale)))
    resized = rgb.resize((nw, nh), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (width, height), color=(255, 255, 255))
    out.paste(resized, ((width - nw) // 2, (height - nh) // 2))
    return out


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not bool(m.any()):
        return None
    ys, xs = np.where(m)
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)


def _draw_bbox_rgba(
    rgba: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    *,
    color: Tuple[int, int, int, int] = (0, 255, 0, 255),
    width: int = 4,
) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.uint8)
    if bbox is None:
        return arr
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return arr
    img = Image.fromarray(arr, "RGBA")
    draw = ImageDraw.Draw(img)
    x2i = max(x1, x2 - 1)
    y2i = max(y1, y2 - 1)
    w = max(1, int(width))
    for t in range(w):
        xa = x1 + t
        ya = y1 + t
        xb = x2i - t
        yb = y2i - t
        if xb <= xa or yb <= ya:
            break
        draw.rectangle([xa, ya, xb, yb], outline=color)
    return np.asarray(img, dtype=np.uint8)


def _compute_gt_recolor_delta(
    renderer: "_EpisodeRendererCache",
    row_q: Dict[str, Any],
    default_task: str,
) -> float:
    eid = str(row_q.get("episode_id", ""))
    gt_idx = int(row_q.get("gt_index", -1))
    if not eid or gt_idx < 0:
        return float("nan")
    data = renderer.get(eid)
    gt = data["gt"]
    canvas = data["canvas"]
    if not (0 <= gt_idx < len(gt)):
        return float("nan")

    edit = {"task_type": str(row_q.get("task_type", default_task))}
    params = row_q.get("params", {})
    if isinstance(params, dict):
        edit.update(params)

    before_rgba = element_to_rgba(gt[gt_idx], canvas)
    gt_after_scene = apply_edit_to_scene(gt, [gt_idx], canvas, edit)
    after_rgba = element_to_rgba(gt_after_scene[gt_idx], canvas)
    before_a = before_rgba[..., 3] > 0
    after_a = after_rgba[..., 3] > 0
    mask = before_a | after_a
    if not bool(mask.any()):
        return 0.0
    diff = np.abs(after_rgba[..., :3].astype(np.float32) - before_rgba[..., :3].astype(np.float32))
    score = float(np.mean(diff[mask]))
    return score


def _render_triplet(
    out_path: Path,
    *,
    subtask: str,
    row_q: Dict[str, Any],
    row_a: Dict[str, Any],
    gt_before: np.ndarray,
    q_before: np.ndarray,
    a_before: np.ndarray,
    gt_after: np.ndarray,
    q_after: np.ndarray,
    a_after: np.ndarray,
    cell_w: int,
    cell_h: int,
) -> None:
    cells = [
        _fit_rgba_on_white(gt_before, cell_w, cell_h),
        _fit_rgba_on_white(q_before, cell_w, cell_h),
        _fit_rgba_on_white(a_before, cell_w, cell_h),
        _fit_rgba_on_white(gt_after, cell_w, cell_h),
        _fit_rgba_on_white(q_after, cell_w, cell_h),
        _fit_rgba_on_white(a_after, cell_w, cell_h),
    ]

    margin = 16
    gap = 10
    header_h = 54
    label_h = 18
    canvas_w = margin * 2 + cell_w * 3 + gap * 2
    canvas_h = margin * 2 + header_h + label_h + cell_h * 2 + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    eid = str(row_q.get("episode_id", ""))
    gt_idx = int(row_q.get("gt_index", -1))
    q_l1 = row_q.get("metrics", {}).get("l1") if isinstance(row_q.get("metrics"), dict) else None
    q_l2 = row_q.get("metrics", {}).get("l2") if isinstance(row_q.get("metrics"), dict) else None
    a_l1 = row_a.get("metrics", {}).get("l1") if isinstance(row_a.get("metrics"), dict) else None
    a_l2 = row_a.get("metrics", {}).get("l2") if isinstance(row_a.get("metrics"), dict) else None
    draw.text((margin, margin), f"subtask={subtask} | episode={eid} | gt={gt_idx}", fill=(0, 0, 0), font=font)
    draw.text((margin, margin + 18), f"l1 q={q_l1} a={a_l1} | l2 q={q_l2} a={a_l2}", fill=(0, 0, 0), font=font)
    draw.text((margin, margin + 36), f"params={_stable_params_json(row_q)}", fill=(0, 0, 0), font=font)

    labels = ["GT", "Qwen", "Agent"]
    y0 = margin + header_h
    for c in range(3):
        x = margin + c * (cell_w + gap)
        draw.text((x + 4, y0), labels[c], fill=(0, 0, 0), font=font)

    y1 = y0 + label_h
    y2 = y1 + cell_h + gap
    for c in range(3):
        x = margin + c * (cell_w + gap)
        canvas.paste(cells[c], (x, y1))
        canvas.paste(cells[3 + c], (x, y2))
        draw.rectangle([x, y1, x + cell_w - 1, y1 + cell_h - 1], outline=(90, 90, 90), width=2)
        draw.rectangle([x, y2, x + cell_w - 1, y2 + cell_h - 1], outline=(90, 90, 90), width=2)
        draw.text((x + 4, y1 + 4), "Before", fill=(30, 30, 30), font=font)
        draw.text((x + 4, y2 + 4), "After", fill=(30, 30, 30), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


class _EpisodeRendererCache:
    def __init__(
        self,
        *,
        figma_data: Path,
        exp_pairs: Sequence[str],
    ) -> None:
        q_tasks = collect_episode_tasks(figma_data, exp_pairs, model="qwen")
        a_tasks = collect_episode_tasks(figma_data, exp_pairs, model="agent")
        self._q_task_map = {t.episode_id: t for t in q_tasks}
        self._a_task_map = {t.episode_id: t for t in a_tasks}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def get(self, episode_id: str) -> Dict[str, Any]:
        with self._lock:
            cached = self._cache.get(episode_id)
            if cached is not None:
                return cached
        qt = self._q_task_map.get(episode_id)
        at = self._a_task_map.get(episode_id)
        if qt is None or at is None:
            raise KeyError(f"missing episode task for {episode_id}")
        q_gt, q_pred, q_canvas = load_episode_elements(qt, model="qwen")
        a_gt, a_pred, a_canvas = load_episode_elements(at, model="agent")
        if tuple(q_canvas) != tuple(a_canvas):
            canvas = tuple(q_canvas)
        else:
            canvas = tuple(q_canvas)
        # Prefer GT with more elements; this mirrors other paths in run_atomic_edit_radnom.
        gt = q_gt if len(q_gt) >= len(a_gt) else a_gt
        obj = {
            "gt": gt,
            "q_pred": q_pred,
            "a_pred": a_pred,
            "canvas": canvas,
        }
        with self._lock:
            self._cache[episode_id] = obj
        return obj


_PROC_CTX: Dict[str, Any] = {}


def _render_one_case(
    *,
    renderer: _EpisodeRendererCache,
    subtask: str,
    q_map: Dict[str, Dict[str, Any]],
    a_map: Dict[str, Dict[str, Any]],
    cell_w: int,
    cell_h: int,
    viz_root: Path,
    rank: int,
    key: str,
) -> Tuple[bool, int, str, str]:
    rq = q_map[key]
    ra = a_map[key]
    eid = str(rq.get("episode_id", ""))
    try:
        gt_idx = int(rq.get("gt_index", -1))
        data = renderer.get(eid)
        gt = data["gt"]
        q_pred = data["q_pred"]
        a_pred = data["a_pred"]
        canvas = data["canvas"]

        edit = {"task_type": str(rq.get("task_type", subtask))}
        params = rq.get("params", {})
        if isinstance(params, dict):
            edit.update(params)

        q_idx = _indices_from_row(rq)
        a_idx = _indices_from_row(ra)

        q_eval_scene = _apply_evalfigma_postprocess_to_scene(q_pred, q_idx, canvas)
        a_eval_scene = _apply_evalfigma_postprocess_to_scene(a_pred, a_idx, canvas)

        gt_before = render_scene_rgba(gt, canvas)
        q_before = render_scene_rgba(q_eval_scene, canvas)
        a_before = render_scene_rgba(a_eval_scene, canvas)

        gt_after_scene = apply_edit_to_scene(gt, [gt_idx], canvas, edit)
        q_after_scene = apply_edit_to_scene(q_eval_scene, q_idx, canvas, edit)
        a_after_scene = apply_edit_to_scene(a_eval_scene, a_idx, canvas, edit)

        gt_after = render_scene_rgba(gt_after_scene, canvas)
        q_after = render_scene_rgba(q_after_scene, canvas)
        a_after = render_scene_rgba(a_after_scene, canvas)

        h = int(canvas[1]) if len(canvas) > 1 else int(gt_before.shape[0])
        w = int(canvas[0]) if len(canvas) > 0 else int(gt_before.shape[1])

        def _safe_union(scene: List[Dict[str, Any]], idxs: List[int]) -> np.ndarray:
            if scene:
                try:
                    return np.asarray(union_mask_from_indices(scene, idxs), dtype=bool)
                except Exception:
                    pass
            return np.zeros((h, w), dtype=bool)

        gt_before_mask = _safe_union(gt, [gt_idx])
        gt_after_mask = _safe_union(gt_after_scene, [gt_idx])
        if not bool(gt_after_mask.any()) and bool(gt_before_mask.any()):
            gt_after_mask = gt_before_mask

        q_before_mask = _safe_union(q_eval_scene, q_idx)
        q_after_mask = _safe_union(q_after_scene, q_idx)
        if not bool(q_after_mask.any()) and bool(q_before_mask.any()):
            q_after_mask = q_before_mask

        a_before_mask = _safe_union(a_eval_scene, a_idx)
        a_after_mask = _safe_union(a_after_scene, a_idx)
        if not bool(a_after_mask.any()) and bool(a_before_mask.any()):
            a_after_mask = a_before_mask

        gt_before = _draw_bbox_rgba(gt_before, _bbox_from_mask(gt_before_mask))
        gt_after = _draw_bbox_rgba(gt_after, _bbox_from_mask(gt_after_mask))
        q_before = _draw_bbox_rgba(q_before, _bbox_from_mask(q_before_mask))
        q_after = _draw_bbox_rgba(q_after, _bbox_from_mask(q_after_mask))
        a_before = _draw_bbox_rgba(a_before, _bbox_from_mask(a_before_mask))
        a_after = _draw_bbox_rgba(a_after, _bbox_from_mask(a_after_mask))

        key_hash = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        out_name = f"{rank:05d}__{eid}__gt{gt_idx}__{key_hash}.png"
        _render_triplet(
            viz_root / out_name,
            subtask=_norm_subtask(subtask),
            row_q=rq,
            row_a=ra,
            gt_before=gt_before,
            q_before=q_before,
            a_before=a_before,
            gt_after=gt_after,
            q_after=q_after,
            a_after=a_after,
            cell_w=cell_w,
            cell_h=cell_h,
        )
        return (True, rank, eid, "")
    except Exception as e:
        return (False, rank, eid, f"{type(e).__name__}: {e}")


def _init_process_ctx(
    figma_data: str,
    exp_pairs: List[str],
    subtask: str,
    q_rows: List[Dict[str, Any]],
    a_rows: List[Dict[str, Any]],
    cell_w: int,
    cell_h: int,
    viz_root: str,
) -> None:
    global _PROC_CTX
    renderer = _EpisodeRendererCache(figma_data=Path(figma_data), exp_pairs=exp_pairs)
    q_map = {_triplet_key(r, default_task=subtask): r for r in q_rows if isinstance(r, dict)}
    a_map = {_triplet_key(r, default_task=subtask): r for r in a_rows if isinstance(r, dict)}
    _PROC_CTX = {
        "renderer": renderer,
        "subtask": subtask,
        "q_map": q_map,
        "a_map": a_map,
        "cell_w": int(cell_w),
        "cell_h": int(cell_h),
        "viz_root": str(viz_root),
    }


def _process_render_one(job: Tuple[int, str]) -> Tuple[bool, int, str, str]:
    rank, key = job
    ctx = _PROC_CTX
    return _render_one_case(
        renderer=ctx["renderer"],
        subtask=ctx["subtask"],
        q_map=ctx["q_map"],
        a_map=ctx["a_map"],
        cell_w=int(ctx["cell_w"]),
        cell_h=int(ctx["cell_h"]),
        viz_root=Path(ctx["viz_root"]),
        rank=rank,
        key=key,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate atomic 2x3 triplet visualizations from result rows.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--figma-data", type=Path, required=True)
    p.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    p.add_argument(
        "--subtasks",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset (e.g., delete transition rotation opacity z_order recolor).",
    )
    p.add_argument("--cell-width", type=int, default=520)
    p.add_argument("--cell-height", type=int, default=300)
    p.add_argument("--max-per-subtask", type=int, default=10)
    p.add_argument("--log-every", type=int, default=1, help="Progress log interval per subtask (cases).")
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers used per subtask for case rendering.",
    )
    p.add_argument(
        "--parallel-backend",
        type=str,
        default="process",
        choices=["thread", "process"],
        help="Parallel backend for per-case rendering when --num-workers > 1.",
    )
    p.add_argument(
        "--selection-mode",
        type=str,
        default="lexicographic",
        choices=["lexicographic", "largest_gt_area", "largest_gt_area_then_l1_gap"],
        help=(
            "Case selection order. "
            "largest_gt_area_then_l1_gap sorts by GT area desc, then |agent_l1-qwen_l1| desc."
        ),
    )
    p.add_argument(
        "--min-gt-area",
        type=float,
        default=0.0,
        help="Optional lower-bound filter on GT area (from qwen match payload).",
    )
    p.add_argument(
        "--start-rank",
        type=int,
        default=0,
        help="Skip first N ranked cases per subtask after ordering/filtering.",
    )
    p.add_argument(
        "--match-root",
        type=Path,
        default=None,
        help="Match root used to read GT area (e.g., editability_matches/.../merge_max).",
    )
    p.add_argument(
        "--agent-win-metric",
        type=str,
        choices=["l1", "l2", "source_l1", "target_l1", "avg_l1", "source_l2", "target_l2", "avg_l2"],
        default=None,
        help=(
            "Optional filter: keep only cases where (qwen_metric - agent_metric) >= --min-agent-win-gap "
            "(i.e., agent is better on lower-is-better metric)."
        ),
    )
    p.add_argument(
        "--min-agent-win-gap",
        type=float,
        default=0.0,
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
        help="Workers for recolor delta scoring (thread pool). 0 means use --num-workers.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import cv2  # type: ignore

        cv2.setNumThreads(1)
    except Exception:
        pass

    out_dir = Path(args.output_dir)
    q_over = load_json(out_dir / "atomic_qwen_overview.json")
    a_over = load_json(out_dir / "atomic_agent_overview.json")
    if not isinstance(q_over, dict) or not isinstance(a_over, dict):
        raise ValueError("invalid overview files")

    all_subtasks = sorted(set(q_over.keys()) & set(a_over.keys()))
    if args.subtasks:
        want = {_norm_subtask(x) for x in args.subtasks}
        subtasks = [s for s in all_subtasks if _norm_subtask(s) in want]
    else:
        subtasks = all_subtasks

    renderer = _EpisodeRendererCache(figma_data=Path(args.figma_data), exp_pairs=args.exp_pairs)
    gt_area_idx = _build_gt_area_index(args.match_root if args.match_root is not None else None)
    log_every = max(1, int(args.log_every))
    num_workers = max(1, int(args.num_workers))
    if int(args.num_workers) <= 0:
        num_workers = max(1, (os.cpu_count() or 8))

    summary: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "figma_data": str(args.figma_data),
        "exp_pairs": list(args.exp_pairs),
        "subtasks": subtasks,
        "cell_width": int(args.cell_width),
        "cell_height": int(args.cell_height),
        "max_per_subtask": int(args.max_per_subtask),
        "selection_mode": str(args.selection_mode),
        "min_gt_area": float(args.min_gt_area),
        "start_rank": int(args.start_rank),
        "match_root": str(args.match_root) if args.match_root is not None else None,
        "agent_win_metric": str(args.agent_win_metric) if args.agent_win_metric else None,
        "min_agent_win_gap": float(args.min_agent_win_gap),
        "recolor_prioritize_gt_delta": bool(args.recolor_prioritize_gt_delta),
        "recolor_min_gt_delta": float(args.recolor_min_gt_delta),
        "recolor_delta_max_candidates": int(args.recolor_delta_max_candidates),
        "recolor_delta_num_workers": int(args.recolor_delta_num_workers),
        "num_workers": int(num_workers),
        "parallel_backend": str(args.parallel_backend),
        "results": {},
    }

    for s in subtasks:
        q_rows = _rows_from_payload(q_over.get(s, {}))
        a_rows = _rows_from_payload(a_over.get(s, {}))
        q_map = {_triplet_key(r, default_task=s): r for r in q_rows}
        a_map = {_triplet_key(r, default_task=s): r for r in a_rows}
        raw_keys = sorted(set(q_map.keys()) & set(a_map.keys()))
        print(
            f"[triplet-viz-reconstruct] start subtask={_norm_subtask(s)} "
            f"aligned={len(raw_keys)}",
            flush=True,
        )

        min_gt_area = float(args.min_gt_area)

        def _score(k: str) -> Tuple[float, float]:
            rq = q_map[k]
            ra = a_map[k]
            eid = str(rq.get("episode_id", ""))
            gt_idx = int(rq.get("gt_index", -1))
            area = float(gt_area_idx.get((eid, gt_idx), -1.0))
            q_l1 = _to_float((rq.get("metrics", {}) if isinstance(rq.get("metrics"), dict) else {}).get("l1"), default=0.0)
            a_l1 = _to_float((ra.get("metrics", {}) if isinstance(ra.get("metrics"), dict) else {}).get("l1"), default=0.0)
            gap = abs(a_l1 - q_l1)
            if str(args.selection_mode) == "largest_gt_area_then_l1_gap":
                return (area, gap)
            if str(args.selection_mode) == "largest_gt_area":
                return (area, 0.0)
            return (0.0, 0.0)

        keys = list(raw_keys)
        if str(args.selection_mode) != "lexicographic":
            keys = sorted(keys, key=_score, reverse=True)
        if args.agent_win_metric is not None:
            m = str(args.agent_win_metric)
            min_gap = float(args.min_agent_win_gap)
            kept: List[str] = []
            for k in keys:
                rq = q_map[k]
                ra = a_map[k]
                q_metrics = rq.get("metrics", {}) if isinstance(rq.get("metrics"), dict) else {}
                a_metrics = ra.get("metrics", {}) if isinstance(ra.get("metrics"), dict) else {}
                qv = _to_float(q_metrics.get(m), default=float("nan"))
                av = _to_float(a_metrics.get(m), default=float("nan"))
                if not (qv == qv and av == av):
                    continue
                if (float(qv) - float(av)) >= min_gap:
                    kept.append(k)
            keys = kept

        if min_gt_area > 0.0:
            kept: List[str] = []
            for k in keys:
                rq = q_map[k]
                eid = str(rq.get("episode_id", ""))
                gt_idx = int(rq.get("gt_index", -1))
                area = float(gt_area_idx.get((eid, gt_idx), -1.0))
                if area >= min_gt_area:
                    kept.append(k)
            keys = kept

        recolor_gt_delta: Dict[str, float] = {}
        if _norm_subtask(s) == "recolor" and bool(args.recolor_prioritize_gt_delta):
            max_delta_cands = max(0, int(args.recolor_delta_max_candidates))
            if max_delta_cands > 0 and len(keys) > max_delta_cands:
                keys = keys[:max_delta_cands]
                print(
                    f"[triplet-viz-reconstruct] recolor-delta-cap subtask={_norm_subtask(s)} "
                    f"cap={max_delta_cands}",
                    flush=True,
                )
            delta_workers = int(args.recolor_delta_num_workers)
            if delta_workers <= 0:
                delta_workers = num_workers
            delta_workers = max(1, int(delta_workers))
            print(
                f"[triplet-viz-reconstruct] recolor-delta-score subtask={_norm_subtask(s)} "
                f"candidates={len(keys)} workers={delta_workers}",
                flush=True,
            )
            if delta_workers <= 1 or len(keys) <= 1:
                for di, k in enumerate(keys, start=1):
                    rq = q_map[k]
                    try:
                        recolor_gt_delta[k] = _compute_gt_recolor_delta(renderer, rq, s)
                    except Exception:
                        recolor_gt_delta[k] = float("nan")
                    if di % log_every == 0 or di == len(keys):
                        print(
                            f"[triplet-viz-reconstruct] recolor-delta-progress subtask={_norm_subtask(s)} "
                            f"{di}/{len(keys)}",
                            flush=True,
                        )
            else:
                done_delta = 0
                with ThreadPoolExecutor(max_workers=delta_workers) as ex:
                    futs = {
                        ex.submit(_compute_gt_recolor_delta, renderer, q_map[k], s): k
                        for k in keys
                    }
                    for fut in as_completed(futs):
                        k = futs[fut]
                        try:
                            recolor_gt_delta[k] = float(fut.result())
                        except Exception:
                            recolor_gt_delta[k] = float("nan")
                        done_delta += 1
                        if done_delta % log_every == 0 or done_delta == len(keys):
                            print(
                                f"[triplet-viz-reconstruct] recolor-delta-progress subtask={_norm_subtask(s)} "
                                f"{done_delta}/{len(keys)}",
                                flush=True,
                            )
            min_delta = float(args.recolor_min_gt_delta)
            if min_delta > 0.0:
                keys = [k for k in keys if _to_float(recolor_gt_delta.get(k), default=float("nan")) >= min_delta]
            score_cache = {k: _score(k) for k in keys}
            keys = sorted(
                keys,
                key=lambda k: (
                    _to_float(recolor_gt_delta.get(k), default=-1.0),
                    score_cache.get(k, (0.0, 0.0))[0],
                    score_cache.get(k, (0.0, 0.0))[1],
                ),
                reverse=True,
            )
        if int(args.start_rank) > 0:
            keys = keys[int(args.start_rank) :]
        if args.max_per_subtask is not None:
            keys = keys[: max(0, int(args.max_per_subtask))]

        viz_root = out_dir / "triplet_pair_viz_reconstructed" / f"atomic_{_norm_subtask(s)}"
        created = 0
        failed = 0
        first_error: Optional[str] = None
        total_jobs = len(keys)
        use_process_pool = num_workers > 1 and str(args.parallel_backend) == "process"
        episode_ids = sorted({str(q_map[k].get("episode_id", "")) for k in keys if str(q_map[k].get("episode_id", ""))})
        if (not use_process_pool) and episode_ids:
            print(
                f"[triplet-viz-reconstruct] preload subtask={_norm_subtask(s)} episodes={len(episode_ids)}",
                flush=True,
            )
            for epi_i, eid in enumerate(episode_ids, start=1):
                try:
                    renderer.get(eid)
                except Exception as e:
                    if first_error is None:
                        first_error = f"{type(e).__name__}: {e}"
                    failed += 1
                if epi_i % log_every == 0 or epi_i == len(episode_ids):
                    print(
                        f"[triplet-viz-reconstruct] preload-progress subtask={_norm_subtask(s)} "
                        f"{epi_i}/{len(episode_ids)}",
                        flush=True,
                    )

        print(
            f"[triplet-viz-reconstruct] render subtask={_norm_subtask(s)} "
            f"jobs={total_jobs} workers={num_workers} backend={'process' if use_process_pool else ('thread' if num_workers > 1 else 'single')}",
            flush=True,
        )

        done = 0
        jobs = [(rank, k) for rank, k in enumerate(keys, start=1)]

        def _run_thread_jobs(pending_jobs: List[Tuple[int, str]]) -> None:
            nonlocal done, created, failed, first_error
            with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
                futs = {
                    ex.submit(
                        _render_one_case,
                        renderer=renderer,
                        subtask=s,
                        q_map=q_map,
                        a_map=a_map,
                        cell_w=int(args.cell_width),
                        cell_h=int(args.cell_height),
                        viz_root=viz_root,
                        rank=rank,
                        key=k,
                    ): (rank, k)
                    for rank, k in pending_jobs
                }
                for fut in as_completed(futs):
                    ok, rank_done, eid_done, err = fut.result()
                    done += 1
                    if ok:
                        created += 1
                    else:
                        failed += 1
                        if first_error is None:
                            first_error = err
                            print(
                                f"[triplet-viz-reconstruct] first_error subtask={_norm_subtask(s)} "
                                f"rank={rank_done}/{total_jobs} episode={eid_done} err={first_error}",
                                flush=True,
                            )
                    if done % log_every == 0 or done == total_jobs:
                        print(
                            f"[triplet-viz-reconstruct] progress subtask={_norm_subtask(s)} "
                            f"{done}/{total_jobs} created={created} failed={failed}",
                            flush=True,
                        )

        if num_workers <= 1:
            for rank, k in jobs:
                ok, rank_done, eid_done, err = _render_one_case(
                    renderer=renderer,
                    subtask=s,
                    q_map=q_map,
                    a_map=a_map,
                    cell_w=int(args.cell_width),
                    cell_h=int(args.cell_height),
                    viz_root=viz_root,
                    rank=rank,
                    key=k,
                )
                done += 1
                if ok:
                    created += 1
                else:
                    failed += 1
                    if first_error is None:
                        first_error = err
                        print(
                            f"[triplet-viz-reconstruct] first_error subtask={_norm_subtask(s)} "
                            f"rank={rank_done}/{total_jobs} episode={eid_done} err={first_error}",
                            flush=True,
                        )
                if done % log_every == 0 or done == total_jobs:
                    print(
                        f"[triplet-viz-reconstruct] progress subtask={_norm_subtask(s)} "
                        f"{done}/{total_jobs} created={created} failed={failed}",
                        flush=True,
                    )
        elif use_process_pool:
            selected_q_rows = [q_map[k] for k in keys]
            selected_a_rows = [a_map[k] for k in keys]
            process_error: Optional[str] = None
            try:
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_init_process_ctx,
                    initargs=(
                        str(args.figma_data),
                        list(args.exp_pairs),
                        str(s),
                        selected_q_rows,
                        selected_a_rows,
                        int(args.cell_width),
                        int(args.cell_height),
                        str(viz_root),
                    ),
                ) as ex:
                    futs = {ex.submit(_process_render_one, job): job for job in jobs}
                    for fut in as_completed(futs):
                        ok, rank_done, eid_done, err = fut.result()
                        done += 1
                        if ok:
                            created += 1
                        else:
                            failed += 1
                            if first_error is None:
                                first_error = err
                                print(
                                    f"[triplet-viz-reconstruct] first_error subtask={_norm_subtask(s)} "
                                    f"rank={rank_done}/{total_jobs} episode={eid_done} err={first_error}",
                                    flush=True,
                                )
                        if done % log_every == 0 or done == total_jobs:
                            print(
                                f"[triplet-viz-reconstruct] progress subtask={_norm_subtask(s)} "
                                f"{done}/{total_jobs} created={created} failed={failed}",
                                flush=True,
                            )
            except Exception as e:
                process_error = f"{type(e).__name__}: {e}"
            if process_error is not None:
                print(
                    f"[triplet-viz-reconstruct] process backend failed; fallback=thread err={process_error}",
                    flush=True,
                )
                if done > 0:
                    raise RuntimeError(
                        f"Process backend failed after partial completion: {process_error}"
                    )
                _run_thread_jobs(jobs)
        else:
            _run_thread_jobs(jobs)

        if total_jobs == 0:
            print(
                f"[triplet-viz-reconstruct] progress subtask={_norm_subtask(s)} "
                f"0/0 created=0 failed=0",
                flush=True,
            )
        elif done % log_every != 0 and done != total_jobs:
            print(
                f"[triplet-viz-reconstruct] progress subtask={_norm_subtask(s)} "
                f"{done}/{total_jobs} created={created} failed={failed}",
                flush=True,
            )

        info = {
            "subtask": _norm_subtask(s),
            "aligned_cases": len(raw_keys),
            "requested_cases": len(keys),
            "created": int(created),
            "failed": int(failed),
            "first_error": first_error,
            "viz_dir": str(viz_root),
            "recolor_prioritize_gt_delta": bool(args.recolor_prioritize_gt_delta) if _norm_subtask(s) == "recolor" else False,
            "recolor_min_gt_delta": float(args.recolor_min_gt_delta) if _norm_subtask(s) == "recolor" else 0.0,
        }
        summary["results"][s] = info
        print(
            f"[triplet-viz-reconstruct] subtask={_norm_subtask(s)} "
            f"aligned={info['aligned_cases']} requested={info['requested_cases']} "
            f"created={info['created']} failed={info['failed']} "
            f"first_error={info['first_error']} dir={info['viz_dir']}"
            ,
            flush=True,
        )

    out_summary = out_dir / "atomic_triplet_viz_reconstructed_summary.json"
    save_json(out_summary, summary)
    print(f"Saved summary: {out_summary}", flush=True)


if __name__ == "__main__":
    main()
