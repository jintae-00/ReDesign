#!/usr/bin/env python3
"""Shared helpers for text subtasks."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from ...common_utils import (
    composite_rgba_on_background,
    compute_cer,
    compute_wer,
    load_json,
    preprocess_text_content,
    render_prompt_overlay_rgba,
    run_ocr_on_rgba_black_white_best,
    save_json,
    sample_with_seed_balanced_by_key,
    warmup_ocr_clients,
)
from ...common_utils import thread_map
from ...nanobanana_bridge import build_text_edit_instruction, run_nanobanana_on_rgba
from ...task_common import apply_edit_to_scene, render_scene_rgba, union_mask_from_indices
from ..common import (
    Candidate,
    EpisodeCache,
    build_task_map,
    extract_match_cost,
    evaluate_image_edit_candidates,
    passes_matching_cost_filter,
    passes_matching_iou_filter,
    passes_gt_opaque_filter_for_match,
    is_text_gt,
    load_match_payloads,
)


_GT_UNIT_MAP_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}
_GT_UNIT_MAP_LOCK = Lock()


def _gt_text_from_elem(gt_elem: Dict[str, Any]) -> str:
    unit = gt_elem.get("meta", {}).get("gt_unit", {})
    return preprocess_text_content(unit.get("text_content", ""))


def _pred_text_from_elem(
    pred_elem: Dict[str, Any],
    model: str,
    canvas_size: Tuple[int, int],
    gt_text: str,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    parsed = pred_elem.get("meta", {}).get("parsed", {})
    content = preprocess_text_content(parsed.get("content", ""))
    if content:
        return content, "parsed_content", None

    rgba = pred_elem["image"].convert("RGBA")
    if rgba.size != canvas_size:
        rgba = rgba.resize(canvas_size)
    ocr_meta = run_ocr_on_rgba_black_white_best(
        np.array(rgba, dtype=np.uint8),
        gt_text=str(gt_text),
    )
    text = preprocess_text_content(ocr_meta.get("best_text", ""))
    return text, "ocr_on_matched_element_bw_bg", ocr_meta


def collect_text_pairs(
    figma_data: Path,
    exp_pairs: Sequence[str],
    model: str,
    match_root: Path,
    max_episodes: Optional[int] = None,
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    build_log_every: int = 0,
    need_pred_text: bool = True,
    max_pairs: Optional[int] = None,
    sample_seed: int = 123,
    matching_cost_threshold: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], EpisodeCache]:
    task_map = build_task_map(figma_data, exp_pairs, model=model, max_episodes=max_episodes)
    payloads = load_match_payloads(match_root, model)
    payloads = [p for p in payloads if p["episode_id"] in task_map]

    cache = EpisodeCache(task_map, model)
    refs: List[Dict[str, Any]] = []
    opaque_ok_cache: Dict[Tuple[str, int], bool] = {}

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

    total_payloads = len(payloads)
    for pi, payload in enumerate(payloads, start=1):
        eid = payload["episode_id"]
        unit_map: Optional[Dict[str, Dict[str, Any]]] = None

        for m in payload.get("matches", []):
            gt_idx = m.get("gt_index")
            if gt_idx is None:
                continue
            gt_idx = int(gt_idx)
            if not passes_matching_cost_filter(m, max_cost=matching_cost_threshold):
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
                    "gt_index": int(gt_idx),
                    "pred_index": p_idx,
                    "pred_indices": [int(x) for x in m.get("selected_pred_indices", [])],
                    "gt_id": gt_id,
                    "gt_text": gt_text,
                    "match_score": m.get("best_single_metrics", {}).get("score", 0.0),
                    "match_cost": extract_match_cost(m),
                }
            )
        if build_log_every > 0 and (pi == 1 or pi % build_log_every == 0 or pi == total_payloads):
            print(f"[{model}][text_pairs] build payload {pi}/{total_payloads} -> refs={len(refs)}")

    if max_pairs is not None:
        refs = sample_with_seed_balanced_by_key(
            refs,
            key_fn=lambda r: str(r.get("episode_id", "")),
            max_count=max(0, int(max_pairs)),
            seed=int(sample_seed),
        )
        if build_log_every > 0:
            print(f"[{model}][text_pairs] sampled refs={len(refs)} (max_pairs={int(max_pairs)})")

    # Qwen/image baseline heavily depends on OCR. Warm up once to avoid
    # concurrent model downloads/initializations from worker threads.
    if need_pred_text and refs and model != "agent":
        try:
            warmup_ocr_clients()
        except Exception:
            pass

    if not need_pred_text:
        pairs = [
            {
                **ref,
                "pred_id": None,
                "pred_text": "",
                "text_source": "not_required",
            }
            for ref in refs
        ]
        if build_log_every > 0:
            print(f"[{model}][text_pairs] built pairs={len(pairs)} (pred_text skipped)")
        return task_map, pairs, cache

    def _build_pair(ref: Dict[str, Any]) -> Dict[str, Any]:
        eid = str(ref["episode_id"])
        _, pred_elements, canvas_size = cache.get(eid)
        p_idx = int(ref["pred_index"])
        if p_idx < 0 or p_idx >= len(pred_elements):
            return {
                **ref,
                "pred_id": None,
                "pred_text": "",
                "text_source": "missing_pred_index",
            }
        pred_elem = pred_elements[p_idx]
        pred_text, text_source, ocr_meta = _pred_text_from_elem(
            pred_elem,
            model=model,
            canvas_size=canvas_size,
            gt_text=str(ref.get("gt_text", "")),
        )
        out = {
            **ref,
            "pred_id": pred_elem.get("id"),
            "pred_text": pred_text,
            "text_source": text_source,
        }
        if isinstance(ocr_meta, dict):
            out["ocr_pipeline"] = "black_white_solid_v1"
            out["ocr_background_rgb"] = ocr_meta.get("best_background_rgb")
            out["ocr_candidates"] = ocr_meta.get("candidates")
        return out

    pairs = thread_map(
        refs,
        _build_pair,
        num_workers=max(1, int(num_workers)),
        desc=f"[{model}][text_pairs]",
        show_tqdm=show_tqdm,
    )
    if build_log_every > 0:
        print(f"[{model}][text_pairs] built pairs={len(pairs)}")
    return task_map, pairs, cache


def aggregate_per_episode_text_metric(pairs: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    per_ep: Dict[str, List[float]] = {}
    for p in pairs:
        per_ep.setdefault(p["episode_id"], []).append(float(p[key]))

    ep_scores = {ep: (sum(vals) / len(vals) if vals else float("nan")) for ep, vals in per_ep.items()}
    finite_vals = [v for v in ep_scores.values() if v == v]
    overall = float(sum(finite_vals) / len(finite_vals)) if finite_vals else float("nan")
    return {"per_episode": ep_scores, "overall_episode_avg": overall}


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _extract_words_with_spans(text: str) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for m in _WORD_RE.finditer(text):
        out.append((m.group(0), m.start(), m.end()))
    return out


def _random_mixed_case_word(length: int, rng: random.Random) -> str:
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(letters) for _ in range(max(1, int(length))))


def build_content_edit_plan(text: str, rng: random.Random) -> List[Dict[str, Any]]:
    words = _extract_words_with_spans(text)
    if not words:
        return []
    valid = [i for i, (w, _, _) in enumerate(words) if len(w) >= 2]
    if not valid:
        return []
    max_pick = min(4, len(valid))
    k = rng.randint(1, max_pick)
    picked = sorted(rng.sample(valid, k))

    plan: List[Dict[str, Any]] = []
    for wi in picked:
        word = words[wi][0]
        repl = _random_mixed_case_word(len(word), rng)
        plan.append({"word_index": wi, "op": "char_randomize", "value": repl, "src": word})
    return plan


def _apply_plan_on_text(text: str, plan: List[Dict[str, Any]]) -> Tuple[str, List[int]]:
    words = _extract_words_with_spans(text)
    if not words or not plan:
        return text, []

    by_idx = {int(x["word_index"]): str(x["value"]) for x in plan}
    chunks: List[str] = []
    cursor = 0
    changed: List[int] = []
    for i, (_, s, e) in enumerate(words):
        chunks.append(text[cursor:s])
        if i in by_idx:
            chunks.append(by_idx[i])
            changed.append(i)
        else:
            chunks.append(text[s:e])
        cursor = e
    chunks.append(text[cursor:])
    return "".join(chunks), changed


def _stable_rng(seed: int, episode_id: str, gt_index: int) -> random.Random:
    key = f"{seed}:{episode_id}:{gt_index}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return random.Random(int(h[:12], 16))


def _word_subset_text(text: str, indices: List[int]) -> str:
    words = [w for w, _, _ in _extract_words_with_spans(text)]
    return " ".join([words[i] for i in indices if 0 <= i < len(words)])


def evaluate_content_modification_pairs(
    pairs: List[Dict[str, Any]],
    cache: EpisodeCache,
    model: str,
    seed: int,
    use_nanobanana_for_image: bool = True,
    require_nanobanana_for_image: bool = True,
    nanobanana_retries: int = 2,
    max_nanobanana_calls: Optional[int] = None,
    log_every: int = 0,
    save_pair_viz_dir: Optional[Path] = None,
    pair_viz_max: Optional[int] = None,
    reference_results: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
) -> List[Dict[str, Any]]:
    if save_pair_viz_dir is not None:
        save_pair_viz_dir.mkdir(parents=True, exist_ok=True)

    def _pair_key_from_dict(p: Dict[str, Any]) -> str:
        return (
            f"{str(p.get('episode_id', ''))}"
            f"::gt{int(p.get('gt_index', -1))}"
            f"::pred{int(p.get('pred_index', -1))}"
            f"::seed{int(p.get('seed', seed))}"
        )

    def _row_sort_key(r: Dict[str, Any]) -> Tuple[str, int, int, int]:
        return (
            str(r.get("episode_id", "")),
            int(r.get("gt_index", -1)),
            int(r.get("pred_index", -1)),
            int(r.get("seed", seed)),
        )

    existing_by_key: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path is not None and resume and checkpoint_path.exists():
        try:
            loaded = load_json(checkpoint_path)
            if isinstance(loaded, list):
                for row in loaded:
                    if isinstance(row, dict):
                        existing_by_key[_pair_key_from_dict(row)] = row
        except Exception:
            existing_by_key = {}
    allowed_keys = {
        _pair_key_from_dict({**p, "seed": int(seed)})
        for p in pairs
    }
    if allowed_keys:
        existing_by_key = {k: v for k, v in existing_by_key.items() if k in allowed_keys}
    existing_by_key = {
        k: v
        for k, v in existing_by_key.items()
        if str(v.get("ocr_pipeline", "")) == "black_white_solid_v1"
    }

    run_lock = Lock()
    run_done = len(existing_by_key)
    run_sum_cer = 0.0
    run_sum_wer = 0.0
    run_sum_edited = 0.0
    run_sum_untouched = 0.0
    for row in existing_by_key.values():
        cfull = row.get("cer")
        wfull = row.get("wer")
        ce = row.get("cer_edited_word")
        cu = row.get("cer_untouched_words")
        if isinstance(cfull, (int, float)) and (cfull == cfull):
            run_sum_cer += float(cfull)
        if isinstance(wfull, (int, float)) and (wfull == wfull):
            run_sum_wer += float(wfull)
        if isinstance(ce, (int, float)) and (ce == ce):
            run_sum_edited += float(ce)
        if isinstance(cu, (int, float)) and (cu == cu):
            run_sum_untouched += float(cu)

    nanobanana_budget = None if max_nanobanana_calls is None else max(0, int(max_nanobanana_calls))
    nanobanana_calls_used = 0
    nanobanana_budget_warned = False
    if nanobanana_budget is not None:
        for row in existing_by_key.values():
            nb = row.get("nanobanana", {})
            if isinstance(nb, dict):
                att = nb.get("attempts", 0)
                try:
                    nanobanana_calls_used += max(0, int(att))
                except Exception:
                    pass
        nanobanana_calls_used = min(nanobanana_calls_used, nanobanana_budget)

    pending_pairs = [p for p in pairs if _pair_key_from_dict({**p, "seed": int(seed)}) not in existing_by_key]
    total_target = len(existing_by_key) + len(pending_pairs)
    print(
        f"[{model}][content_mod] total={total_target} "
        f"resume_done={len(existing_by_key)} pending={len(pending_pairs)} "
        f"workers={max(1, int(num_workers))} tqdm={bool(show_tqdm)}"
    )

    def _join_pred_indices(xs: Sequence[int]) -> str:
        if len(xs) <= 6:
            return "-".join(str(int(x)) for x in xs)
        return "-".join(str(int(x)) for x in xs[:6]) + f"-n{len(xs)}"

    def _render_pred_panel(
        pred_before: np.ndarray,
        pred_after: np.ndarray,
    ) -> np.ndarray:
        pad = 4
        h = max(pred_before.shape[0], pred_after.shape[0])
        a = pred_before
        b = pred_after
        if a.shape[0] != h:
            p = np.zeros((h - a.shape[0], a.shape[1], 4), dtype=np.uint8)
            a = np.concatenate([a, p], axis=0)
        if b.shape[0] != h:
            p = np.zeros((h - b.shape[0], b.shape[1], 4), dtype=np.uint8)
            b = np.concatenate([b, p], axis=0)
        spacer = np.full((h, pad, 4), 255, dtype=np.uint8)
        return np.concatenate([a, spacer, b], axis=1)

    def _eval_one(job: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        nonlocal run_done, run_sum_cer, run_sum_wer, run_sum_edited, run_sum_untouched
        nonlocal nanobanana_calls_used, nanobanana_budget_warned
        i, p = job
        rng = _stable_rng(seed, str(p["episode_id"]), int(p["gt_index"]))
        plan = build_content_edit_plan(str(p["gt_text"]), rng)
        gt_edit, tgt_idx = _apply_plan_on_text(str(p["gt_text"]), plan)
        changed_pairs = [(str(x.get("src", "")), str(x.get("value", ""))) for x in plan]

        pred_edit = ""
        pred_edit_source = "rule_based_on_pred_text"
        nanobanana_info: Dict[str, Any] = {}
        pred_before_rgba: Optional[np.ndarray] = None
        pred_after_rgba: Optional[np.ndarray] = None
        source_ocr_meta: Optional[Dict[str, Any]] = None
        edit_ocr_meta: Optional[Dict[str, Any]] = None
        nanobanana_instruction: Optional[str] = None

        image_baseline = str(p.get("text_source", "")) != "parsed_content"
        need_nanobanana = bool(use_nanobanana_for_image and image_baseline)
        reserved_calls = 0
        if need_nanobanana and nanobanana_budget is not None:
            with run_lock:
                remaining = int(nanobanana_budget) - int(nanobanana_calls_used)
                if remaining > 0:
                    reserved_calls = min(max(1, int(nanobanana_retries)), remaining)
                    nanobanana_calls_used += int(reserved_calls)
                else:
                    if not nanobanana_budget_warned:
                        print(
                            f"[{model}][content_mod] nanobanana budget exhausted "
                            f"({nanobanana_budget} calls). Remaining pairs use fallback/no-op."
                        )
                        nanobanana_budget_warned = True
            if reserved_calls <= 0:
                need_nanobanana = False
                nanobanana_info = {"ok": False, "skipped": "budget_exhausted"}
                pred_edit_source = "nanobanana_budget_exhausted"

        if need_nanobanana:
            try:
                _, pred_elements, canvas_size = cache.get(str(p["episode_id"]))
                pred_idx = int(p["pred_index"])
                if 0 <= pred_idx < len(pred_elements):
                    pred_elem = pred_elements[pred_idx]
                    pred_rgba = pred_elem["image"].convert("RGBA")
                    if pred_rgba.size != canvas_size:
                        pred_rgba = pred_rgba.resize(canvas_size)
                    pred_before_rgba_raw = np.array(pred_rgba, dtype=np.uint8)
                    source_ocr_meta = run_ocr_on_rgba_black_white_best(
                        pred_before_rgba_raw,
                        gt_text=str(p.get("gt_text", "")),
                    )
                    source_bg = source_ocr_meta.get("best_background_rgb") if isinstance(source_ocr_meta, dict) else None
                    source_bg_rgb = None
                    if isinstance(source_bg, (list, tuple)) and len(source_bg) == 3:
                        source_bg_rgb = (int(source_bg[0]), int(source_bg[1]), int(source_bg[2]))
                    if source_bg_rgb is None:
                        source_bg_rgb = (255, 255, 255)
                    pred_before_rgba = composite_rgba_on_background(pred_before_rgba_raw, source_bg_rgb)
                    instruction = build_text_edit_instruction(
                        gt_edit,
                        source_text=str(p.get("pred_text", "")),
                        changed_words=changed_pairs,
                    )
                    nanobanana_instruction = instruction
                    edited_rgba, nb_meta = run_nanobanana_on_rgba(
                        pred_before_rgba,
                        instruction,
                        retries=int(reserved_calls) if reserved_calls > 0 else nanobanana_retries,
                    )
                    nanobanana_info = nb_meta
                    if isinstance(nanobanana_info, dict):
                        nanobanana_info["source_ocr_background_rgb"] = [
                            int(source_bg_rgb[0]),
                            int(source_bg_rgb[1]),
                            int(source_bg_rgb[2]),
                        ]
                    if edited_rgba is not None:
                        pred_after_rgba = edited_rgba
                        edit_ocr_meta = run_ocr_on_rgba_black_white_best(
                            edited_rgba,
                            gt_text=gt_edit,
                        )
                        pred_edit = preprocess_text_content(edit_ocr_meta.get("best_text", ""))
                        pred_edit_source = "nanobanana_ocr_bw_bg"
                    else:
                        pred_edit_source = "nanobanana_failed"
            except Exception as e:
                nanobanana_info = {"ok": False, "error": str(e)}
                pred_edit_source = "nanobanana_exception"
            finally:
                if reserved_calls > 0 and nanobanana_budget is not None:
                    try:
                        used = int((nanobanana_info or {}).get("attempts", 0))
                    except Exception:
                        used = 0
                    used = max(0, min(int(reserved_calls), int(used)))
                    refund = int(reserved_calls) - int(used)
                    if refund > 0:
                        with run_lock:
                            nanobanana_calls_used = max(0, int(nanobanana_calls_used) - int(refund))

        if (not pred_edit) and (not (need_nanobanana and require_nanobanana_for_image)):
            pred_edit, _ = _apply_plan_on_text(str(p["pred_text"]), plan)
            pred_edit = preprocess_text_content(pred_edit)
        elif (not pred_edit) and need_nanobanana and require_nanobanana_for_image:
            pred_edit = ""
            pred_edit_source = "nanobanana_required_failed"

        cer_selected_text = compute_cer(gt_edit, pred_edit)
        wer_selected_text = compute_wer(gt_edit, pred_edit)
        cer_full = cer_selected_text
        wer_full = wer_selected_text
        if isinstance(edit_ocr_meta, dict):
            cmin = edit_ocr_meta.get("min_cer")
            wmin = edit_ocr_meta.get("min_wer")
            if isinstance(cmin, (int, float)) and (float(cmin) == float(cmin)):
                cer_full = float(cmin)
            if isinstance(wmin, (int, float)) and (float(wmin) == float(wmin)):
                wer_full = float(wmin)

        # edited words CER (focus region)
        gt_t = _word_subset_text(gt_edit, tgt_idx)
        pd_t = _word_subset_text(pred_edit, tgt_idx)
        cer_edited = compute_cer(gt_t, pd_t)

        # untouched words CER (preservation region)
        n_words_gt = len(_extract_words_with_spans(gt_edit))
        untouched_idx = [i for i in range(n_words_gt) if i not in set(tgt_idx)]
        gt_u = _word_subset_text(gt_edit, untouched_idx)
        pd_u = _word_subset_text(pred_edit, untouched_idx)
        cer_untouched = compute_cer(gt_u, pd_u)

        row = {
            **p,
            "seed": int(seed),
            "edit_plan": plan,
            "target_word_indices": tgt_idx,
            "gt_text_edited": gt_edit,
            "pred_text_edited": pred_edit,
            "pred_text_edit_source": pred_edit_source,
            "nanobanana": nanobanana_info,
            "nanobanana_prompt": nanobanana_instruction,
            "ocr_pipeline": "black_white_solid_v1",
            "ocr_background_rgb": (
                edit_ocr_meta.get("best_background_rgb")
                if isinstance(edit_ocr_meta, dict)
                else None
            ),
            "ocr_candidates": (
                edit_ocr_meta.get("candidates")
                if isinstance(edit_ocr_meta, dict)
                else None
            ),
            "source_ocr_background_rgb": (
                source_ocr_meta.get("best_background_rgb")
                if isinstance(source_ocr_meta, dict)
                else None
            ),
            "source_ocr_candidates": (
                source_ocr_meta.get("candidates")
                if isinstance(source_ocr_meta, dict)
                else None
            ),
            "cer_selected_text": cer_selected_text,
            "wer_selected_text": wer_selected_text,
            "cer": cer_full,
            "wer": wer_full,
            "cer_edited_word": cer_edited,
            "cer_untouched_words": cer_untouched,
        }

        if save_pair_viz_dir is not None and (pair_viz_max is None or i <= pair_viz_max):
            pair_name = (
                f"{i:05d}__{p['episode_id']}"
                f"__gt{int(p['gt_index'])}"
                f"__pred{_join_pred_indices(p.get('pred_indices') or [p['pred_index']])}"
            )
            pair_dir = save_pair_viz_dir / pair_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            if pred_before_rgba is not None:
                Image.fromarray(pred_before_rgba, "RGBA").save(pair_dir / "pred_before.png")
            if pred_after_rgba is not None:
                Image.fromarray(pred_after_rgba, "RGBA").save(pair_dir / "pred_after.png")
            if pred_before_rgba is not None and pred_after_rgba is not None:
                panel = _render_pred_panel(pred_before_rgba, pred_after_rgba)
                prompt_text = str(nanobanana_instruction or "")
                panel_with_prompt = render_prompt_overlay_rgba(
                    panel,
                    prompt_text,
                    title="Nanobanana Prompt (text modification)",
                )
                Image.fromarray(panel_with_prompt, "RGBA").save(pair_dir / "panel.png")
            save_json(pair_dir / "meta.json", row)

        with run_lock:
            run_done += 1
            if log_every > 0:
                run_sum_cer += float(cer_full)
                run_sum_wer += float(wer_full)
                run_sum_edited += float(cer_edited)
                run_sum_untouched += float(cer_untouched)
                if run_done == 1 or run_done % log_every == 0 or run_done == total_target:
                    msg = (
                        f"[{model}][content_mod] running_avg {run_done}/{total_target} "
                        f"agent(cer={run_sum_cer / run_done:.4f} "
                        f"wer={run_sum_wer / run_done:.4f} "
                        f"cer_edited_word={run_sum_edited / run_done:.4f} "
                        f"cer_untouched_words={run_sum_untouched / run_done:.4f})"
                    )
                    if reference_results:
                        n = min(run_done, len(reference_results))
                        if n > 0:
                            q0 = [
                                float(reference_results[j].get("cer"))
                                for j in range(n)
                                if isinstance(reference_results[j].get("cer"), (int, float))
                            ]
                            q00 = [
                                float(reference_results[j].get("wer"))
                                for j in range(n)
                                if isinstance(reference_results[j].get("wer"), (int, float))
                            ]
                            q1 = [
                                float(reference_results[j].get("cer_edited_word"))
                                for j in range(n)
                                if isinstance(reference_results[j].get("cer_edited_word"), (int, float))
                            ]
                            q2 = [
                                float(reference_results[j].get("cer_untouched_words"))
                                for j in range(n)
                                if isinstance(reference_results[j].get("cer_untouched_words"), (int, float))
                            ]
                            if q0 and q00 and q1 and q2:
                                msg += (
                                    f" qwen_prefix(cer={sum(q0) / len(q0):.4f} "
                                    f"wer={sum(q00) / len(q00):.4f} "
                                    f"cer_edited_word={sum(q1) / len(q1):.4f} "
                                    f"cer_untouched_words={sum(q2) / len(q2):.4f})"
                                )
                    print(msg)
            existing_by_key[_pair_key_from_dict(row)] = row
            if checkpoint_path is not None:
                save_json(checkpoint_path, sorted(existing_by_key.values(), key=_row_sort_key))
        return row

    ordered_pairs = sorted(
        pending_pairs,
        key=lambda p: (
            str(p.get("episode_id", "")),
            int(p.get("gt_index", -1)),
            int(p.get("pred_index", -1)),
        ),
    )
    if ordered_pairs:
        _ = thread_map(
            list(enumerate(ordered_pairs, start=1)),
            _eval_one,
            num_workers=max(1, int(num_workers)),
            desc=f"[{model}][content_mod]",
            show_tqdm=show_tqdm,
        )
    elif checkpoint_path is not None and resume:
        print(f"[{model}][content_mod] resume: no pending items")

    results = sorted(existing_by_key.values(), key=_row_sort_key)
    if checkpoint_path is not None:
        save_json(checkpoint_path, results)
    return results


def aggregate_content_modification(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "count": 0,
            "cer": float("nan"),
            "wer": float("nan"),
            "cer_edited_word": float("nan"),
            "cer_untouched_words": float("nan"),
        }
    cer_vals = [float(r["cer"]) for r in results if isinstance(r.get("cer"), (int, float))]
    wer_vals = [float(r["wer"]) for r in results if isinstance(r.get("wer"), (int, float))]
    a = [r["cer_edited_word"] for r in results]
    b = [r["cer_untouched_words"] for r in results]
    return {
        "count": len(results),
        "cer": float(sum(cer_vals) / len(cer_vals)) if cer_vals else float("nan"),
        "wer": float(sum(wer_vals) / len(wer_vals)) if wer_vals else float("nan"),
        "cer_edited_word": float(sum(a) / len(a)),
        "cer_untouched_words": float(sum(b) / len(b)),
    }


def build_style_candidates(
    pairs: List[Dict[str, Any]],
    task_type: str,
    params_grid: Sequence[Dict[str, Any]],
) -> List[Candidate]:
    out: List[Candidate] = []
    for p in pairs:
        pred_indices = p.get("pred_indices") or [p["pred_index"]]
        if not pred_indices:
            continue
        for params in params_grid:
            out.append(
                Candidate(
                    episode_id=p["episode_id"],
                    gt_index=int(p["gt_index"]),
                    pred_indices=[int(x) for x in pred_indices],
                    task_type=task_type,
                    params=dict(params),
                )
            )
    return out


def evaluate_style_subtask(
    candidates: List[Candidate],
    cache: EpisodeCache,
    seed: int,
    max_tasks: Optional[int],
    progress_prefix: Optional[str] = None,
    log_every: int = 0,
    save_pair_viz_dir: Optional[Path] = None,
    pair_viz_max: Optional[int] = None,
    reference_results: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
) -> List[Dict[str, Any]]:
    sampled: List[Candidate]
    if max_tasks is None or max_tasks >= len(candidates):
        sampled = sample_with_seed_balanced_by_key(
            candidates, key_fn=lambda c: c.episode_id, max_count=max_tasks, seed=seed
        )
    else:
        buckets: Dict[str, List[Candidate]] = {}
        for c in candidates:
            key = json.dumps(c.params, sort_keys=True, ensure_ascii=True)
            buckets.setdefault(key, []).append(c)
        keys = list(buckets.keys())
        rng = random.Random(seed)
        rng.shuffle(keys)
        for k in keys:
            buckets[k] = sample_with_seed_balanced_by_key(
                buckets[k], key_fn=lambda c: c.episode_id, max_count=None, seed=seed + int(hashlib.md5(k.encode("utf-8")).hexdigest()[:8], 16) % 997
            )
        sampled = []
        while len(sampled) < max_tasks and keys:
            next_keys: List[str] = []
            for k in keys:
                b = buckets.get(k, [])
                if not b:
                    continue
                sampled.append(b.pop())
                if len(sampled) >= max_tasks:
                    break
                if b:
                    next_keys.append(k)
            keys = next_keys
    return evaluate_image_edit_candidates(
        sampled,
        cache,
        include_iou=True,
        include_edge_sharpness=True,
        include_lpips=True,
        roi_mode="source",
        roi_dilation_ratio=0.0,
        progress_prefix=progress_prefix or f"[{cache.model}][text_style]",
        log_every=log_every,
        save_pair_viz_dir=save_pair_viz_dir,
        pair_viz_max=pair_viz_max,
        reference_metrics=[r.get("metrics", {}) for r in (reference_results or [])],
        reference_label="qwen",
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
