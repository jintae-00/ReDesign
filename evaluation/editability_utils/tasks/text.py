#!/usr/bin/env python3
"""Text editability tasks.

Key requirement reflected:
- First use GT:model matching result.
- For each GT text element, use the best-score matched element.
- If matched element has text content (Agent), use it directly.
- Otherwise run OCR on the matched individual element image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..common_utils import (
    compute_cer,
    compute_region_metrics,
    compute_wer,
    load_json,
    run_ocr_on_rgba,
    save_json,
    sample_with_seed,
)
from ..loaders import EpisodeTask, collect_episode_tasks, load_episode_elements
from ..task_common import apply_edit_to_scene, render_scene_rgba, union_mask_from_indices


def _load_match_payloads(match_dir: Path) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for p in sorted((match_dir / "episodes").glob("*.json")):
        payloads.append(load_json(p))
    return payloads


def evaluate_content_recognition(
    payloads: List[Dict[str, Any]],
    task_map: Dict[str, EpisodeTask],
    model: str,
) -> List[Dict[str, Any]]:
    """Evaluate CER/WER with per-element OCR fallback after matching."""
    results: List[Dict[str, Any]] = []
    cache: Dict[str, Any] = {}

    for payload in payloads:
        eid = payload["episode_id"]
        if eid not in task_map:
            continue
        if eid not in cache:
            cache[eid] = load_episode_elements(task_map[eid], model=model)
        gt_elements, pred_elements, canvas_size = cache[eid]

        for m in payload.get("matches", []):
            gt_idx = m.get("gt_index")
            if gt_idx is None or gt_idx >= len(gt_elements):
                continue

            gt_elem = gt_elements[gt_idx]
            gt_text = ""
            gt_meta = gt_elem.get("meta", {}).get("gt_unit", {})
            gt_type = str(gt_elem.get("type", ""))
            if gt_type == "text" or gt_meta.get("unit_type") == "text":
                gt_text = str(gt_meta.get("text_content", "") or "")
            if not gt_text:
                continue

            p_idx = m.get("best_single_pred_index")
            if p_idx is None:
                selected = m.get("selected_pred_indices", [])
                if not selected:
                    continue
                p_idx = selected[0]

            if p_idx >= len(pred_elements):
                continue

            pred_elem = pred_elements[p_idx]
            parsed_text = ""
            parsed = pred_elem.get("meta", {}).get("parsed", {})
            if model == "agent":
                parsed_text = str(parsed.get("content", "") or "").strip()

            if parsed_text:
                pred_text = parsed_text
                text_source = "parsed_content"
            else:
                pred_rgba = pred_elem["image"].convert("RGBA")
                if pred_rgba.size != tuple(canvas_size):
                    pred_rgba = pred_rgba.resize(tuple(canvas_size))
                pred_text = run_ocr_on_rgba(np.array(pred_rgba, dtype=np.uint8))
                text_source = "ocr_on_matched_element"

            cer = compute_cer(gt_text, pred_text)
            wer = compute_wer(gt_text, pred_text)

            results.append(
                {
                    "episode_id": eid,
                    "gt_index": gt_idx,
                    "gt_id": gt_elem.get("id"),
                    "pred_index": p_idx,
                    "pred_id": pred_elem.get("id"),
                    "gt_text": gt_text,
                    "pred_text": pred_text,
                    "text_source": text_source,
                    "matching_score": m.get("best_single_metrics", {}).get("score", 0.0),
                    "cer": cer,
                    "wer": wer,
                }
            )

    return results


def aggregate_content_recognition(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"count": 0, "cer": float("nan"), "wer": float("nan")}
    cer_vals = [r["cer"] for r in results]
    wer_vals = [r["wer"] for r in results]
    return {
        "count": len(results),
        "cer": float(sum(cer_vals) / len(cer_vals)),
        "wer": float(sum(wer_vals) / len(wer_vals)),
    }


def build_style_edit_candidates(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for payload in payloads:
        eid = payload["episode_id"]
        for m in payload.get("matches", []):
            gt_type = m.get("gt_type", "")
            if gt_type != "text":
                continue
            gt_idx = m.get("gt_index")
            pred_indices = m.get("selected_pred_indices", [])
            if gt_idx is None or not pred_indices:
                continue

            candidates.extend(
                [
                    {
                        "episode_id": eid,
                        "task_type": "super_scaling",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "scale": 1.2,
                    },
                    {
                        "episode_id": eid,
                        "task_type": "text_bold",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "strength": 1,
                    },
                    {
                        "episode_id": eid,
                        "task_type": "text_italic",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "shear": 0.15,
                    },
                    {
                        "episode_id": eid,
                        "task_type": "recolor",
                        "gt_index": gt_idx,
                        "pred_indices": pred_indices,
                        "hue_shift_deg": 25.0,
                        "sat_mul": 1.1,
                        "val_mul": 1.0,
                    },
                ]
            )
    return candidates


def evaluate_style_edits(
    candidates: List[Dict[str, Any]],
    task_map: Dict[str, EpisodeTask],
    model: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    cache: Dict[str, Any] = {}

    for c in candidates:
        eid = c["episode_id"]
        if eid not in cache:
            cache[eid] = load_episode_elements(task_map[eid], model=model)
        gt_elements, pred_elements, canvas_size = cache[eid]

        gt_idx = c["gt_index"]
        pred_indices = c["pred_indices"]

        gt_edited = apply_edit_to_scene(gt_elements, [gt_idx], canvas_size, c)
        pred_edited = apply_edit_to_scene(pred_elements, pred_indices, canvas_size, c)

        gt_rgba = render_scene_rgba(gt_edited, canvas_size)
        pred_rgba = render_scene_rgba(pred_edited, canvas_size)

        roi = union_mask_from_indices(gt_elements, [gt_idx]) | union_mask_from_indices(gt_edited, [gt_idx])
        metrics = compute_region_metrics(gt_rgba, pred_rgba, roi, include_iou=True, include_edge_sharpness=True)

        results.append(
            {
                "episode_id": eid,
                "task_type": c["task_type"],
                "gt_index": gt_idx,
                "pred_indices": pred_indices,
                "metrics": metrics,
            }
        )

    return results


def aggregate_style_edits(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, float]]] = {}
    for r in results:
        by_task.setdefault(r["task_type"], []).append(r["metrics"])

    out: Dict[str, Any] = {"total": len(results), "by_task_type": {}}
    for k, vals in by_task.items():
        keys = sorted({mk for d in vals for mk in d.keys()})
        mean = {}
        for mk in keys:
            xs = [float(d[mk]) for d in vals if mk in d and d[mk] == d[mk]]
            mean[mk] = float(sum(xs) / len(xs)) if xs else float("nan")
        out["by_task_type"][k] = {"count": len(vals), "mean": mean}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Text editability evaluator")
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--model", type=str, choices=["agent", "qwen"], required=True)
    parser.add_argument("--match-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-style-tasks", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    tasks = collect_episode_tasks(Path(args.figma_data), args.exp_pairs, model=args.model, max_episodes=args.max_episodes)
    task_map = {t.episode_id: t for t in tasks}

    payloads = _load_match_payloads(Path(args.match_root) / args.model)
    payloads = [p for p in payloads if p["episode_id"] in task_map]

    recog = evaluate_content_recognition(payloads, task_map, model=args.model)
    recog_summary = aggregate_content_recognition(recog)

    style_candidates = build_style_edit_candidates(payloads)
    style_sampled = sample_with_seed(style_candidates, args.max_style_tasks, args.seed)
    style_results = evaluate_style_edits(style_sampled, task_map, model=args.model)
    style_summary = aggregate_style_edits(style_results)

    out_dir = Path(args.output)
    save_json(out_dir / f"text_content_recognition_{args.model}.json", recog)
    save_json(out_dir / f"text_content_recognition_{args.model}_summary.json", recog_summary)
    save_json(out_dir / f"text_style_edit_{args.model}.json", style_results)
    save_json(out_dir / f"text_style_edit_{args.model}_summary.json", style_summary)

    print(
        f"[DONE] model={args.model} content_pairs={len(recog)} style_candidates={len(style_candidates)} "
        f"style_sampled={len(style_sampled)}"
    )


if __name__ == "__main__":
    main()
