#!/usr/bin/env python3
"""Text content recognition subtask (CER/WER).

Rule:
- Use GT:model matching first.
- Use best single matched pred element.
- If pred has parsed content (agent), use it.
- Else OCR on matched single element image.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

from ...common_utils import (
    composite_rgba_on_background,
    compute_cer,
    compute_wer,
    load_json,
    preprocess_text_content,
    run_ocr_on_rgba_black_white_best,
    save_json,
    warmup_ocr_clients,
)
from ...task_common import render_scene_rgba
from ..common import (
    EpisodeCache,
    build_task_map,
    extract_match_cost,
    load_match_payloads,
    passes_gt_opaque_filter_for_match,
    passes_matching_cost_filter,
    passes_matching_iou_filter,
)
from ._shared import (
    aggregate_per_episode_text_metric,
)


_GT_UNIT_MAP_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}
_GT_UNIT_MAP_LOCK = Lock()


def run(
    figma_data: Path,
    exp_pairs: Sequence[str],
    model: str,
    match_root: Path,
    output_dir: Optional[Path] = None,
    seed: int = 123,
    max_tasks: Optional[int] = None,
    max_episodes: Optional[int] = None,
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    build_log_every: int = 0,
    log_every: int = 0,
    save_pair_viz: bool = False,
    pair_viz_max: Optional[int] = None,
    reference_results: Optional[List[Dict[str, Any]]] = None,
    resume: bool = True,
) -> Dict:
    # Recognition must evaluate all matched text pairs; max_tasks is ignored.
    _ = (seed, max_tasks)
    task_map = build_task_map(figma_data, exp_pairs, model=model, max_episodes=max_episodes)
    payloads = load_match_payloads(match_root, model)
    payloads = [p for p in payloads if p["episode_id"] in task_map]
    cache = EpisodeCache(task_map, model)

    def _pair_key(pp: Dict[str, Any]) -> Tuple[str, int, int]:
        return (
            str(pp.get("episode_id", "")),
            int(pp.get("gt_index", -1)),
            int(pp.get("pred_index", -1)),
        )

    def _unit_map_for_episode(eid: str) -> Dict[str, Dict[str, Any]]:
        task = task_map.get(eid)
        if task is None:
            return {}
        cache_key = str(task.gt_json_path)
        with _GT_UNIT_MAP_LOCK:
            cached = _GT_UNIT_MAP_CACHE.get(cache_key)
            if cached is not None:
                return cached
            frame = load_json(task.gt_json_path)
            out: Dict[str, Dict[str, Any]] = {}
            for unit in frame.get("unit_images", []):
                unit_id = str(unit.get("unit_id", ""))
                if unit_id:
                    out[f"gt_{unit_id}"] = unit
            _GT_UNIT_MAP_CACHE[cache_key] = out
            return out

    def _build_refs(*, log_progress: bool) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        opaque_ok_cache: Dict[Tuple[str, int], bool] = {}
        total_payloads = len(payloads)
        for pi, payload in enumerate(payloads, start=1):
            eid = payload["episode_id"]
            unit_map: Optional[Dict[str, Dict[str, Any]]] = None
            for m in payload.get("matches", []):
                gt_idx = m.get("gt_index")
                if gt_idx is None:
                    continue
                gt_idx = int(gt_idx)
                if not passes_matching_cost_filter(m):
                    continue
                if not passes_matching_iou_filter(m):
                    continue
                if subset_keys is not None and (eid, gt_idx) not in subset_keys:
                    continue
                if not passes_gt_opaque_filter_for_match(cache, eid, gt_idx, m, memo=opaque_ok_cache):
                    continue
                gt_type = str(m.get("gt_type", "") or "").lower()
                if gt_type != "text":
                    continue
                gt_id = str(m.get("gt_id", "") or "")
                if unit_map is None:
                    unit_map = _unit_map_for_episode(eid)
                unit = unit_map.get(gt_id, {})
                gt_text = preprocess_text_content(unit.get("text_content", ""))
                if not gt_text:
                    continue

                p_idx = m.get("best_single_pred_index")
                if p_idx is None:
                    selected = m.get("selected_pred_indices", [])
                    if not selected:
                        continue
                    p_idx = selected[0]
                p_idx = int(p_idx)
                if p_idx < 0:
                    continue

                refs.append(
                    {
                        "episode_id": eid,
                        "gt_index": gt_idx,
                        "pred_index": p_idx,
                        "pred_indices": [int(x) for x in m.get("selected_pred_indices", [])],
                        "gt_id": gt_id,
                        "gt_text": gt_text,
                        "match_score": m.get("best_single_metrics", {}).get("score", 0.0),
                        "match_cost": extract_match_cost(m),
                    }
                )
            if log_progress and build_log_every > 0 and (pi == 1 or pi % build_log_every == 0 or pi == total_payloads):
                print(f"[{model}][text_pairs] build payload {pi}/{total_payloads}")
        return refs

    refs = _build_refs(log_progress=True)
    all_keys: Set[Tuple[str, int, int]] = {_pair_key(rr) for rr in refs}

    checkpoint_path: Optional[Path] = None
    existing_by_key: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    if output_dir is not None:
        checkpoint_path = output_dir / model / "content_recognition_results.json"
        if resume and checkpoint_path.exists():
            try:
                loaded = load_json(checkpoint_path)
                if isinstance(loaded, list):
                    for row in loaded:
                        if isinstance(row, dict):
                            existing_by_key[_pair_key(row)] = row
            except Exception:
                existing_by_key = {}
    if all_keys:
        existing_by_key = {k: v for k, v in existing_by_key.items() if k in all_keys}
    # Recompute stale qwen checkpoints created by old single-element OCR path.
    if model != "agent":
        existing_by_key = {
            k: v
            for k, v in existing_by_key.items()
            if str(v.get("ocr_pipeline", "")) == "black_white_solid_v1"
        }

    run_cer = 0.0
    run_wer = 0.0
    for row in existing_by_key.values():
        c = row.get("cer")
        w = row.get("wer")
        if isinstance(c, (int, float)) and (c == c):
            run_cer += float(c)
        if isinstance(w, (int, float)) and (w == w):
            run_wer += float(w)

    total_target = len(all_keys)
    run_done = len(existing_by_key)
    initial_done = run_done
    use_tqdm = bool(show_tqdm and _tqdm is not None)
    pbar = _tqdm(total=total_target, desc=f"[{model}][content_recognition]") if use_tqdm else None
    if pbar is not None and run_done > 0:
        pbar.update(run_done)

    if model != "agent" and total_target > run_done:
        try:
            if build_log_every > 0:
                print(f"[{model}][content_recognition] warmup ocr start")
            warmup_ocr_clients()
            if build_log_every > 0:
                print(f"[{model}][content_recognition] warmup ocr done")
        except Exception:
            pass

    pair_viz_dir: Optional[Path] = None
    if output_dir is not None and save_pair_viz:
        pair_viz_dir = output_dir / model / "content_recognition" / "element_pairs"
        pair_viz_dir.mkdir(parents=True, exist_ok=True)

    def _crop_bbox(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, int, int]:
        roi = (a[..., 3] > 0) | (b[..., 3] > 0)
        ys, xs = np.where(roi)
        if ys.size == 0:
            h, w = a.shape[:2]
            return 0, 0, w, h
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        pad = 2
        h, w = a.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return x1, y1, x2, y2

    def _save_pair_viz(row: Dict[str, Any], idx: int) -> None:
        if pair_viz_dir is None:
            return
        if pair_viz_max is not None and idx > int(pair_viz_max):
            return
        try:
            eid = str(row.get("episode_id", ""))
            gt_idx = int(row.get("gt_index", -1))
            pred_indices = [int(x) for x in (row.get("pred_indices") or [row.get("pred_index", -1)])]
            gt_elements, pred_elements, canvas_size = cache.get(eid)
            if not (0 <= gt_idx < len(gt_elements)):
                return

            gt_elem = gt_elements[gt_idx]
            gt_rgba = np.array(gt_elem["image"].convert("RGBA"), dtype=np.uint8)
            if gt_rgba.shape[1] != canvas_size[0] or gt_rgba.shape[0] != canvas_size[1]:
                gt_rgba = np.array(Image.fromarray(gt_rgba, "RGBA").resize(canvas_size), dtype=np.uint8)

            valid_pred = [i for i in pred_indices if 0 <= i < len(pred_elements)]
            pred_scene = render_scene_rgba([pred_elements[i] for i in valid_pred], canvas_size) if valid_pred else np.zeros_like(gt_rgba)

            x1, y1, x2, y2 = _crop_bbox(gt_rgba, pred_scene)
            gt_crop = gt_rgba[y1:y2, x1:x2]
            pred_crop = pred_scene[y1:y2, x1:x2]
            pred_crop_viz = pred_crop
            bg = row.get("ocr_background_rgb")
            if isinstance(bg, (list, tuple)) and len(bg) == 3:
                try:
                    pred_crop_viz = composite_rgba_on_background(
                        pred_crop,
                        (int(bg[0]), int(bg[1]), int(bg[2])),
                    )
                except Exception:
                    pred_crop_viz = pred_crop
            h = max(gt_crop.shape[0], pred_crop.shape[0])
            if gt_crop.shape[0] != h:
                pad = np.zeros((h - gt_crop.shape[0], gt_crop.shape[1], 4), dtype=np.uint8)
                gt_crop = np.concatenate([gt_crop, pad], axis=0)
            if pred_crop_viz.shape[0] != h:
                pad = np.zeros((h - pred_crop_viz.shape[0], pred_crop_viz.shape[1], 4), dtype=np.uint8)
                pred_crop_viz = np.concatenate([pred_crop_viz, pad], axis=0)
            spacer = np.full((h, 4, 4), 255, dtype=np.uint8)
            panel = np.concatenate([gt_crop, spacer, pred_crop_viz], axis=1)

            pair_name = f"{idx:05d}__{eid}__gt{gt_idx}__pred{'-'.join(str(i) for i in valid_pred[:8])}"
            pair_dir = pair_viz_dir / pair_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(panel, "RGBA").save(pair_dir / "panel.png")
            Image.fromarray(gt_crop, "RGBA").save(pair_dir / "gt_elem.png")
            Image.fromarray(pred_crop_viz, "RGBA").save(pair_dir / "pred_group.png")
            save_json(pair_dir / "meta.json", row)
        except Exception:
            return

    def _ocr_text_from_group(
        pred_elements: Sequence[Dict[str, Any]],
        canvas_size: Tuple[int, int],
        indices: Sequence[int],
        gt_text: str,
    ) -> Tuple[str, Dict[str, Any]]:
        valid = [int(i) for i in indices if 0 <= int(i) < len(pred_elements)]
        if not valid:
            return "", {}
        scene = render_scene_rgba([pred_elements[i] for i in valid], canvas_size)
        alpha = scene[..., 3] > 0
        if not bool(alpha.any()):
            return "", {}
        ys, xs = np.where(alpha)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        crop = scene[y1:y2, x1:x2]
        if crop.size == 0:
            return "", {}
        ocr_meta = run_ocr_on_rgba_black_white_best(crop, gt_text=gt_text)
        return preprocess_text_content(ocr_meta.get("best_text", "")), ocr_meta

    def _eval_ref(ref: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(ref)
        eid = str(ref["episode_id"])
        pred_id: Any = None
        pred_text = ""
        text_source = "missing_pred_index"
        ocr_meta: Optional[Dict[str, Any]] = None
        try:
            _, pred_elements, canvas_size = cache.get(eid)
            p_idx = int(ref["pred_index"])
            pred_indices = [int(x) for x in (ref.get("pred_indices") or [p_idx])]
            valid_pred = [i for i in pred_indices if 0 <= i < len(pred_elements)]
            if valid_pred:
                pred_id = [pred_elements[i].get("id") for i in valid_pred]
                if model == "agent":
                    parsed = pred_elements[valid_pred[0]].get("meta", {}).get("parsed", {})
                    content = preprocess_text_content(parsed.get("content", "")) if isinstance(parsed, dict) else ""
                    if content:
                        pred_text = content
                        text_source = "parsed_content"
                    else:
                        pred_text, ocr_meta = _ocr_text_from_group(
                            pred_elements,
                            canvas_size,
                            valid_pred,
                            gt_text=str(ref.get("gt_text", "")),
                        )
                        text_source = "ocr_on_matched_group_bw_bg"
                else:
                    pred_text, ocr_meta = _ocr_text_from_group(
                        pred_elements,
                        canvas_size,
                        valid_pred,
                        gt_text=str(ref.get("gt_text", "")),
                    )
                    text_source = "ocr_on_matched_group_bw_bg"
        except Exception:
            pred_id = None
            pred_text = ""
            text_source = "pred_text_error"
            ocr_meta = None

        out["pred_id"] = pred_id
        out["pred_text"] = preprocess_text_content(pred_text)
        out["text_source"] = text_source
        cer_selected = compute_cer(out["gt_text"], out["pred_text"])
        wer_selected = compute_wer(out["gt_text"], out["pred_text"])
        out["cer_selected_text"] = cer_selected
        out["wer_selected_text"] = wer_selected
        out["ocr_pipeline"] = "black_white_solid_v1"
        if isinstance(ocr_meta, dict) and ocr_meta:
            out["ocr_background_rgb"] = ocr_meta.get("best_background_rgb")
            out["ocr_candidates"] = ocr_meta.get("candidates")
            cmin = ocr_meta.get("min_cer")
            wmin = ocr_meta.get("min_wer")
            out["cer"] = float(cmin) if isinstance(cmin, (int, float)) and (float(cmin) == float(cmin)) else cer_selected
            out["wer"] = float(wmin) if isinstance(wmin, (int, float)) and (float(wmin) == float(wmin)) else wer_selected
        else:
            out["ocr_background_rgb"] = None
            out["ocr_candidates"] = None
            out["cer"] = cer_selected
            out["wer"] = wer_selected
        return out

    inflight: Dict[Future, Tuple[str, int, int]] = {}
    inflight_keys: Set[Tuple[str, int, int]] = set()
    max_inflight = max(1, int(num_workers)) * 2
    if build_log_every > 0:
        print(f"[{model}][content_recognition] workers={max(1, int(num_workers))} inflight={max_inflight}")

    def _persist() -> None:
        if checkpoint_path is not None:
            save_json(checkpoint_path, sorted(existing_by_key.values(), key=_pair_key))

    def _consume_done(done_set: Set[Future]) -> None:
        nonlocal run_done, run_cer, run_wer
        for fut in done_set:
            key = inflight.pop(fut)
            inflight_keys.discard(key)
            row = fut.result()
            existing_by_key[key] = row
            run_done += 1
            if pbar is not None:
                pbar.update(1)
            run_cer += float(row.get("cer", 0.0))
            run_wer += float(row.get("wer", 0.0))
            _persist()
            _save_pair_viz(row, run_done)
            if log_every > 0 and (run_done == 1 or run_done % log_every == 0 or run_done == total_target):
                msg = (
                    f"[{model}][content_recognition] running_avg {run_done}/{total_target} "
                    f"{model}(cer={run_cer / run_done:.4f} wer={run_wer / run_done:.4f})"
                )
                if reference_results:
                    n = min(run_done, len(reference_results))
                    if n > 0:
                        qcer = [
                            float(reference_results[j].get("cer"))
                            for j in range(n)
                            if isinstance(reference_results[j].get("cer"), (int, float))
                        ]
                        qwer = [
                            float(reference_results[j].get("wer"))
                            for j in range(n)
                            if isinstance(reference_results[j].get("wer"), (int, float))
                        ]
                        if qcer and qwer:
                            msg += f" qwen_prefix(cer={sum(qcer) / len(qcer):.4f} wer={sum(qwer) / len(qwer):.4f})"
                print(msg)

    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
        for ref in refs:
            key = _pair_key(ref)
            if key in existing_by_key or key in inflight_keys:
                continue
            fut = ex.submit(_eval_ref, ref)
            inflight[fut] = key
            inflight_keys.add(key)
            while len(inflight) >= max_inflight:
                done, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
                _consume_done(done)
        while inflight:
            done, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
            _consume_done(done)
    if pbar is not None:
        pbar.close()

    if all_keys:
        existing_by_key = {k: v for k, v in existing_by_key.items() if k in all_keys}

    if checkpoint_path is not None and resume and (total_target - initial_done <= 0):
        print(f"[{model}][content_recognition] resume: no pending items")

    pairs = sorted(existing_by_key.values(), key=_pair_key)
    if checkpoint_path is not None:
        save_json(checkpoint_path, pairs)

    cer_summary = aggregate_per_episode_text_metric(pairs, "cer")
    wer_summary = aggregate_per_episode_text_metric(pairs, "wer")

    summary = {
        "count": len(pairs),
        "cer_overall_episode_avg": cer_summary["overall_episode_avg"],
        "wer_overall_episode_avg": wer_summary["overall_episode_avg"],
        "cer_per_episode": cer_summary["per_episode"],
        "wer_per_episode": wer_summary["per_episode"],
    }

    return {
        "capacity": {"total": len(pairs), "by_task_type": {"text_content_recognition": len(pairs)}},
        "sampled_count": len(pairs),
        "results": pairs,
        "summary": summary,
    }
