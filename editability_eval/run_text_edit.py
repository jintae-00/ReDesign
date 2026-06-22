#!/usr/bin/env python3
"""Run all text subtasks and compare Qwen vs Agent."""

from __future__ import annotations

import argparse
import json
import os
import time
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .common_utils import load_json, save_json
from .joint_match_filter import apply_joint_subset_filter
from .subset_manifest import load_subset_keys
from .subtasks.common import compare_two_models
from .subtasks.text import (
    content_modification,
    content_recognition,
    style_bold,
    style_combo,
    style_italic,
    style_recolor,
    style_scaling,
)


def _style_metric_pref() -> Dict[str, str]:
    return {
        "l1": "lower",
        "l2": "lower",
        "lpips": "lower",
        "dino": "higher",
        "psnr": "higher",
        "ssim": "higher",
        "full_l1": "lower",
        "full_l2": "lower",
        "full_lpips": "lower",
        "full_dino": "higher",
        "full_psnr": "higher",
        "full_ssim": "higher",
        "iou": "higher",
        "edge_sharpness_gt": "higher",
        "edge_sharpness_pred": "higher",
    }


def _compare_scalar(q: Dict[str, Any], a: Dict[str, Any], key: str, pref: str) -> Dict[str, Any]:
    qv = q.get(key)
    av = a.get(key)
    winner = None
    if isinstance(qv, (int, float)) and isinstance(av, (int, float)) and qv == qv and av == av:
        if pref == "lower":
            winner = "agent" if av < qv else ("qwen" if qv < av else "tie")
        else:
            winner = "agent" if av > qv else ("qwen" if qv > av else "tie")
    return {"qwen": qv, "agent": av, "preference": pref, "winner": winner}


def _run_subtask_for_model(
    *,
    model: str,
    subtask_name: str,
    run_fn,
    seed: int,
    figma_data: Path,
    exp_pairs: List[str],
    match_root: Path,
    output_dir: Path,
    max_tasks_per_subtask: int | None,
    max_episodes: int | None,
    subset_keys: Optional[Set[Tuple[str, int]]],
    use_nanobanana_for_image_content_mod: bool,
    require_nanobanana_for_image_content_mod: bool,
    nanobanana_retries: int,
    content_mod_max_llm_calls: Optional[int],
    log_every: int,
    save_pair_viz: bool,
    pair_viz_max_per_subtask: Optional[int],
    num_workers: int,
    show_tqdm: bool,
    build_log_every: int,
    reference_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if subtask_name == "content_recognition":
        return run_fn(
            figma_data=figma_data,
            exp_pairs=exp_pairs,
            model=model,
            match_root=match_root,
            output_dir=output_dir,
            seed=seed,
            max_tasks=None,
            max_episodes=max_episodes,
            subset_keys=subset_keys,
            num_workers=num_workers,
            show_tqdm=show_tqdm,
            build_log_every=build_log_every,
            log_every=log_every,
            save_pair_viz=save_pair_viz,
            pair_viz_max=pair_viz_max_per_subtask,
            reference_results=reference_results,
            resume=True,
        )
    if subtask_name == "content_modification":
        return run_fn(
            figma_data=figma_data,
            exp_pairs=exp_pairs,
            model=model,
            match_root=match_root,
            output_dir=output_dir,
            seed=seed,
            max_tasks=max_tasks_per_subtask,
            max_episodes=max_episodes,
            subset_keys=subset_keys,
            use_nanobanana_for_image=use_nanobanana_for_image_content_mod,
            require_nanobanana_for_image=require_nanobanana_for_image_content_mod,
            nanobanana_retries=nanobanana_retries,
            max_nanobanana_calls=content_mod_max_llm_calls,
            log_every=log_every,
            save_pair_viz=save_pair_viz,
            pair_viz_max=pair_viz_max_per_subtask,
            reference_results=reference_results,
            num_workers=num_workers,
            show_tqdm=show_tqdm,
            build_log_every=build_log_every,
        )
    return run_fn(
        figma_data=figma_data,
        exp_pairs=exp_pairs,
        model=model,
        match_root=match_root,
        output_dir=output_dir,
        seed=seed,
        max_tasks=max_tasks_per_subtask,
        max_episodes=max_episodes,
        subset_keys=subset_keys,
        log_every=log_every,
        save_pair_viz=save_pair_viz,
        pair_viz_max=pair_viz_max_per_subtask,
        reference_results=reference_results,
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        build_log_every=build_log_every,
    )


def _safe_read_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        x = load_json(path)
        if isinstance(x, list):
            return [r for r in x if isinstance(r, dict)]
    except Exception:
        pass
    return []


def _psnr_inf_fallback() -> float:
    raw = os.environ.get("EDITABILITY_PSNR_INF_FALLBACK", "100.0")
    try:
        v = float(raw)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return 100.0


def _paired_key_text(row: Dict[str, Any], subtask_name: str) -> str:
    eid = str(row.get("episode_id", ""))
    gt_idx = int(row.get("gt_index", -1))
    if subtask_name == "content_modification":
        return f"{eid}::gt{gt_idx}::seed{int(row.get('seed', -1))}"
    if subtask_name == "content_recognition":
        return f"{eid}::gt{gt_idx}"
    params = row.get("params", {})
    if not isinstance(params, dict):
        params = {}
    params_key = json.dumps(params, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return (
        f"{eid}"
        f"::gt{gt_idx}"
        f"::task={str(row.get('task_type', subtask_name))}"
        f"::params={params_key}"
    )


def _paired_running_msg_text(rows_q: List[Dict[str, Any]], rows_a: List[Dict[str, Any]], subtask_name: str) -> Tuple[int, str]:
    q_map = {_paired_key_text(r, subtask_name): r for r in rows_q}
    a_map = {_paired_key_text(r, subtask_name): r for r in rows_a}
    keys = sorted(set(q_map.keys()) & set(a_map.keys()))
    done = len(keys)
    if done <= 0:
        return 0, f"[live-paired][text][{subtask_name}] aligned_pairs=0"

    if subtask_name == "content_recognition":
        metric_keys = ["cer", "wer"]

        def _get(row: Dict[str, Any], k: str) -> Any:
            return row.get(k)

    elif subtask_name == "content_modification":
        metric_keys = ["cer", "wer", "cer_edited_word", "cer_untouched_words"]

        def _get(row: Dict[str, Any], k: str) -> Any:
            return row.get(k)

    else:
        metric_keys = [
            "l1",
            "l2",
            "psnr",
            "ssim",
            "lpips",
            "dino",
            "iou",
            "full_l1",
            "full_l2",
            "full_psnr",
            "full_ssim",
            "full_lpips",
            "full_dino",
        ]

        def _get(row: Dict[str, Any], k: str) -> Any:
            mm = row.get("metrics", {})
            if not isinstance(mm, dict):
                return None
            return mm.get(k)

    def _mean(model_map: Dict[str, Dict[str, Any]], metric: str) -> Optional[float]:
        vals: List[float] = []
        inf_count = 0
        is_psnr = "psnr" in str(metric).lower()
        for k in keys:
            v = _get(model_map[k], metric)
            if isinstance(v, (int, float)) and (v == v):
                fv = float(v)
                if is_psnr:
                    if math.isfinite(fv):
                        vals.append(fv)
                    elif fv > 0:
                        inf_count += 1
                elif math.isfinite(fv):
                    vals.append(fv)
        if is_psnr and inf_count > 0:
            rep = max(vals) if vals else _psnr_inf_fallback()
            vals.extend([float(rep)] * int(inf_count))
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    q_tokens: List[str] = []
    a_tokens: List[str] = []
    for m in metric_keys:
        qv = _mean(q_map, m)
        av = _mean(a_map, m)
        if qv is None or av is None:
            continue
        q_tokens.append(f"{m}={qv:.4f}")
        a_tokens.append(f"{m}={av:.4f}")

    return (
        done,
        f"[live-paired][text][{subtask_name}] aligned_pairs={done} "
        f"qwen({' '.join(q_tokens)}) agent({' '.join(a_tokens)})",
    )


def _run_subtask_paired(
    *,
    subtask_name: str,
    run_fn,
    sub_seed: int,
    figma_data: Path,
    exp_pairs: List[str],
    match_root: Path,
    output_dir: Path,
    max_tasks_per_subtask: int | None,
    max_episodes: int | None,
    subset_keys: Optional[Set[Tuple[str, int]]],
    use_nanobanana_for_image_content_mod: bool,
    require_nanobanana_for_image_content_mod: bool,
    nanobanana_retries: int,
    content_mod_max_llm_calls: Optional[int],
    log_every: int,
    save_pair_viz: bool,
    pair_viz_max_per_subtask: Optional[int],
    num_workers: int,
    show_tqdm: bool,
    build_log_every: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    total_workers = max(1, int(num_workers))
    q_workers = max(1, total_workers // 2)
    a_workers = max(1, total_workers - q_workers)
    print(
        f"[subtask][text][{subtask_name}] paired workers "
        f"total={total_workers} qwen={q_workers} agent={a_workers}"
    )

    q_ckpt = output_dir / "qwen" / f"{subtask_name}_results.json"
    a_ckpt = output_dir / "agent" / f"{subtask_name}_results.json"
    last_logged = -1
    last_q_rows = -1
    last_a_rows = -1
    last_live_log_ts = 0.0
    poll_sec = 1.0
    log_step = max(1, int(log_every))
    q_done_notified = False
    a_done_notified = False

    with ThreadPoolExecutor(max_workers=2) as ex:
        fq = ex.submit(
            _run_subtask_for_model,
            model="qwen",
            subtask_name=subtask_name,
            run_fn=run_fn,
            seed=sub_seed,
            figma_data=figma_data,
            exp_pairs=exp_pairs,
            match_root=match_root,
            output_dir=output_dir,
            max_tasks_per_subtask=max_tasks_per_subtask,
            max_episodes=max_episodes,
            subset_keys=subset_keys,
            use_nanobanana_for_image_content_mod=use_nanobanana_for_image_content_mod,
            require_nanobanana_for_image_content_mod=require_nanobanana_for_image_content_mod,
            nanobanana_retries=nanobanana_retries,
            content_mod_max_llm_calls=content_mod_max_llm_calls,
            log_every=0,
            save_pair_viz=save_pair_viz,
            pair_viz_max_per_subtask=pair_viz_max_per_subtask,
            num_workers=q_workers,
            show_tqdm=show_tqdm,
            build_log_every=build_log_every,
            reference_results=None,
        )
        fa = ex.submit(
            _run_subtask_for_model,
            model="agent",
            subtask_name=subtask_name,
            run_fn=run_fn,
            seed=sub_seed,
            figma_data=figma_data,
            exp_pairs=exp_pairs,
            match_root=match_root,
            output_dir=output_dir,
            max_tasks_per_subtask=max_tasks_per_subtask,
            max_episodes=max_episodes,
            subset_keys=subset_keys,
            use_nanobanana_for_image_content_mod=use_nanobanana_for_image_content_mod,
            require_nanobanana_for_image_content_mod=require_nanobanana_for_image_content_mod,
            nanobanana_retries=nanobanana_retries,
            content_mod_max_llm_calls=content_mod_max_llm_calls,
            log_every=0,
            save_pair_viz=save_pair_viz,
            pair_viz_max_per_subtask=pair_viz_max_per_subtask,
            num_workers=a_workers,
            show_tqdm=show_tqdm,
            build_log_every=build_log_every,
            reference_results=None,
        )

        while not (fq.done() and fa.done()):
            if fq.done():
                exc = fq.exception()
                if exc is not None:
                    raise RuntimeError(f"[subtask][text][{subtask_name}] qwen worker failed") from exc
                if not q_done_notified:
                    print(f"[subtask][text][{subtask_name}] qwen worker done; waiting agent")
                    q_done_notified = True
            if fa.done():
                exc = fa.exception()
                if exc is not None:
                    raise RuntimeError(f"[subtask][text][{subtask_name}] agent worker failed") from exc
                if not a_done_notified:
                    print(f"[subtask][text][{subtask_name}] agent worker done; waiting qwen")
                    a_done_notified = True
            q_rows = _safe_read_rows(q_ckpt)
            a_rows = _safe_read_rows(a_ckpt)
            done, msg = _paired_running_msg_text(q_rows, a_rows, subtask_name)
            if done > 0 and done != last_logged and (last_logged < 0 or done >= last_logged + log_step):
                print(msg)
                last_logged = done
            if done <= 0:
                qn = len(q_rows)
                an = len(a_rows)
                now = time.time()
                if (
                    qn != last_q_rows
                    or an != last_a_rows
                    or (now - last_live_log_ts) >= 20.0
                ):
                    print(
                        f"[live-paired][text][{subtask_name}] "
                        f"aligned_pairs=0 qwen_rows={qn} agent_rows={an}"
                    )
                    last_q_rows = qn
                    last_a_rows = an
                    last_live_log_ts = now
            time.sleep(poll_sec)

        q_payload = fq.result()
        a_payload = fa.result()

    done, msg = _paired_running_msg_text(
        q_payload.get("results", []),
        a_payload.get("results", []),
        subtask_name,
    )
    if done > 0:
        print(msg)
    return q_payload, a_payload


def _subtask_comparison(subtask_name: str, q_payload: Dict[str, Any], a_payload: Dict[str, Any]) -> Dict[str, Any]:
    if subtask_name == "content_recognition":
        return {
            "cer_overall_episode_avg": _compare_scalar(
                q_payload.get("summary", {}),
                a_payload.get("summary", {}),
                "cer_overall_episode_avg",
                "lower",
            ),
            "wer_overall_episode_avg": _compare_scalar(
                q_payload.get("summary", {}),
                a_payload.get("summary", {}),
                "wer_overall_episode_avg",
                "lower",
            ),
        }
    if subtask_name == "content_modification":
        return {
            "cer": _compare_scalar(
                q_payload.get("summary", {}),
                a_payload.get("summary", {}),
                "cer",
                "lower",
            ),
            "wer": _compare_scalar(
                q_payload.get("summary", {}),
                a_payload.get("summary", {}),
                "wer",
                "lower",
            ),
            "cer_edited_word": _compare_scalar(
                q_payload.get("summary", {}),
                a_payload.get("summary", {}),
                "cer_edited_word",
                "lower",
            ),
            "cer_untouched_words": _compare_scalar(
                q_payload.get("summary", {}),
                a_payload.get("summary", {}),
                "cer_untouched_words",
                "lower",
            ),
        }
    return compare_two_models(
        qwen_summary=q_payload.get("summary", {}),
        agent_summary=a_payload.get("summary", {}),
        metric_preferences=_style_metric_pref(),
    )


def _save_text_subtask_outputs(output_dir: Path, model: str, subtask_name: str, payload: Dict[str, Any]) -> None:
    model_dir = output_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    save_json(model_dir / f"{subtask_name}_summary.json", payload.get("summary", {}))
    save_json(model_dir / f"{subtask_name}_capacity.json", payload.get("capacity", {}))
    save_json(model_dir / f"{subtask_name}_results.json", payload.get("results", []))


def _fmt_metric_value(v: Any) -> str:
    if isinstance(v, (int, float)):
        fv = float(v)
        if fv != fv:
            return "nan"
        if not math.isfinite(fv):
            return "inf" if fv > 0 else "-inf"
        return f"{fv:.4f}"
    return str(v)


def _compact_text_compare_line(compare_payload: Dict[str, Any], metric_order: Sequence[str], max_items: int = 8) -> str:
    by_task = compare_payload.get("by_task_type")
    if isinstance(by_task, dict) and by_task:
        task_name = sorted(by_task.keys())[0]
        metrics = by_task.get(task_name, {}).get("metrics", {})
        if not isinstance(metrics, dict):
            return f"task={task_name} (no-metrics)"
        ordered: List[str] = []
        seen: Set[str] = set()
        for m in list(metric_order) + sorted(metrics.keys()):
            if m in seen or m not in metrics:
                continue
            seen.add(m)
            item = metrics.get(m, {})
            if not isinstance(item, dict):
                continue
            qv = item.get("qwen")
            av = item.get("agent")
            winner = str(item.get("winner", "n/a"))
            ordered.append(f"{m}[Q={_fmt_metric_value(qv)} A={_fmt_metric_value(av)} W={winner}]")
            if len(ordered) >= max(1, int(max_items)):
                break
        return f"task={task_name} " + " ".join(ordered)

    ordered2: List[str] = []
    seen2: Set[str] = set()
    for m in list(metric_order) + sorted(compare_payload.keys()):
        if m in seen2 or m not in compare_payload:
            continue
        seen2.add(m)
        item = compare_payload.get(m, {})
        if not isinstance(item, dict):
            continue
        qv = item.get("qwen")
        av = item.get("agent")
        winner = str(item.get("winner", "n/a"))
        ordered2.append(f"{m}[Q={_fmt_metric_value(qv)} A={_fmt_metric_value(av)} W={winner}]")
        if len(ordered2) >= max(1, int(max_items)):
            break
    if ordered2:
        return " ".join(ordered2)
    return str(compare_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text editability subtasks (Qwen vs Agent)")
    parser.add_argument("--figma-data", type=str, required=True)
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True)
    parser.add_argument("--match-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tasks-per-subtask", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--subset-manifest", type=str, default=None, help="Category subset manifest json from build_category_subsets.py")
    parser.add_argument("--max-matching-cost", type=float, default=0.25, help="Only evaluate matched pairs with matching cost <= this threshold (best_single/merged cost). Negative disables this filter.")
    parser.add_argument("--min-matching-iou", type=float, default=-1.0, help="Only evaluate matched pairs with matching IoU >= this threshold (merged_metrics.iou). Negative disables this filter.")
    parser.add_argument("--min-cross-model-iou", type=float, default=-1.0, help="Only keep GT triplets where IoU(Qwen matched pred, Agent matched pred) >= threshold. Negative disables.")
    parser.add_argument("--no-nanobanana-content-mod", action="store_true", help="Disable nanobanana call for image-based text content modification")
    parser.add_argument("--allow-content-mod-fallback", action="store_true", help="Allow rule-based fallback if nanobanana image edit fails")
    parser.add_argument("--nanobanana-retries", type=int, default=2)
    parser.add_argument("--content-mod-max-llm-calls", type=int, default=None, help="Max nanobanana API attempts for content_modification per model (default: unlimited)")
    parser.add_argument("--log-every", type=int, default=25, help="Print running metric averages every N tasks")
    parser.add_argument("--save-pair-viz", action="store_true", help="Save edited task element-pair visualizations")
    parser.add_argument("--pair-viz-max-per-subtask", type=int, default=None, help="Optional cap for saved pair visualizations per model/subtask")
    parser.add_argument("--num-workers", type=int, default=1, help="Thread workers used inside each subtask evaluation")
    parser.add_argument("--ocr-workers", type=int, default=1, help="Paddle OCR client pool size per process (1-4)")
    parser.add_argument("--min-gt-opaque-pixels", type=int, default=1500, help="Only evaluate matched pairs whose GT element has at least this many opaque(alpha>=threshold) pixels")
    parser.add_argument("--opaque-alpha-threshold", type=int, default=250, help="Alpha threshold (0-255) used to count opaque pixels")
    parser.add_argument("--strict-gt-opaque-filter", action="store_true", help="Use exact alpha>=threshold counting from GT images (slower). Default uses fast payload gt_area filter.")
    parser.add_argument("--cache-episodes", type=int, default=8, help="LRU cache size (episodes) for loaded GT/pred elements")
    parser.add_argument("--max-episode-loaders", type=int, default=2, help="Maximum concurrent episode loads when cache misses")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--build-log-every", type=int, default=50, help="Print pair/candidate build progress every N payloads")
    args = parser.parse_args()

    figma_data = Path(args.figma_data)
    match_root = Path(args.match_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ["EDITABILITY_CACHE_EPISODES"] = str(max(1, int(args.cache_episodes)))
    os.environ["EDITABILITY_MAX_EP_LOADERS"] = str(max(1, int(args.max_episode_loaders)))
    os.environ["EDITABILITY_OCR_WORKERS"] = str(max(1, min(4, int(args.ocr_workers))))
    os.environ["EDITABILITY_MIN_GT_OPAQUE_PIXELS"] = str(max(0, int(args.min_gt_opaque_pixels)))
    os.environ["EDITABILITY_OPAQUE_ALPHA_THRESHOLD"] = str(max(0, min(255, int(args.opaque_alpha_threshold))))
    os.environ["EDITABILITY_STRICT_GT_OPAQUE_CHECK"] = "1" if args.strict_gt_opaque_filter else "0"
    os.environ["EDITABILITY_MAX_MATCHING_COST"] = str(float(args.max_matching_cost))
    os.environ["EDITABILITY_MIN_MATCHING_IOU"] = str(float(args.min_matching_iou))
    if args.max_tasks_per_subtask is not None:
        print(
            "[setup] max_tasks_per_subtask is ignored to keep per-element-pair coverage. "
            "All eligible candidates will be evaluated."
        )
        args.max_tasks_per_subtask = None
    subset_keys = load_subset_keys(Path(args.subset_manifest), "text") if args.subset_manifest else None
    if subset_keys is not None:
        print(f"[setup] loaded text subset keys: {len(subset_keys)}")
    subset_keys, joint_stats = apply_joint_subset_filter(
        match_root=match_root,
        figma_data=figma_data,
        exp_pairs=args.exp_pairs,
        max_matching_cost=float(args.max_matching_cost),
        min_matching_iou=float(args.min_matching_iou),
        min_cross_model_iou=float(args.min_cross_model_iou),
        subset_keys=subset_keys,
        cross_iou_cache_path=match_root / "_cache" / "qwen_agent_pred_iou.json",
        build_log_every=max(0, int(args.build_log_every)),
    )
    print(
        "[setup] joint filters "
        f"cost={float(args.max_matching_cost):.4f} "
        f"min_iou={float(args.min_matching_iou):.4f} "
        f"min_cross_iou={float(args.min_cross_model_iou):.4f} "
        f"shared={joint_stats.get('shared_total')} "
        f"q_cost={joint_stats.get('qwen_cost_pass')} a_cost={joint_stats.get('agent_cost_pass')} "
        f"q_iou={joint_stats.get('qwen_iou_pass')} a_iou={joint_stats.get('agent_iou_pass')} "
        f"cross_known={joint_stats.get('cross_iou_known')} cross_pass={joint_stats.get('cross_iou_pass')} "
        f"final_subset={joint_stats.get('final_subset')}"
    )
    print(
        f"[setup] num_workers={args.num_workers} "
        f"ocr_workers={os.environ.get('EDITABILITY_OCR_WORKERS')} "
        f"max_matching_cost={os.environ.get('EDITABILITY_MAX_MATCHING_COST')} "
        f"min_matching_iou={os.environ.get('EDITABILITY_MIN_MATCHING_IOU')} "
        f"min_gt_opaque_pixels={os.environ.get('EDITABILITY_MIN_GT_OPAQUE_PIXELS')} "
        f"opaque_alpha_threshold={os.environ.get('EDITABILITY_OPAQUE_ALPHA_THRESHOLD')} "
        f"strict_gt_opaque_filter={os.environ.get('EDITABILITY_STRICT_GT_OPAQUE_CHECK')} "
        f"cache_episodes={os.environ.get('EDITABILITY_CACHE_EPISODES')} "
        f"max_episode_loaders={os.environ.get('EDITABILITY_MAX_EP_LOADERS')} "
        f"tqdm={not args.no_tqdm} "
        f"nanobanana_content_mod={not args.no_nanobanana_content_mod} "
        f"require_nanobanana_content_mod={not args.allow_content_mod_fallback} "
        f"content_mod_max_llm_calls={args.content_mod_max_llm_calls} "
        f"save_pair_viz={args.save_pair_viz} "
        f"pair_viz_max_per_subtask={args.pair_viz_max_per_subtask} "
        f"log_every={args.log_every} "
        f"build_log_every={args.build_log_every}"
    )

    subtasks = [
        ("content_recognition", content_recognition.run, 0),
        ("content_modification", content_modification.run, 1),
        ("style_scaling", style_scaling.run, 2),
        ("style_bold", style_bold.run, 3),
        ("style_italic", style_italic.run, 4),
        ("style_recolor", style_recolor.run, 5),
        ("style_combo", style_combo.run, 6),
    ]

    qwen: Dict[str, Any] = {}
    agent: Dict[str, Any] = {}
    comparison: Dict[str, Any] = {}

    for subtask_name, run_fn, seed_offset in subtasks:
        sub_seed = int(args.seed) + int(seed_offset)
        print(f"[subtask][text][{subtask_name}] start qwen+agent")
        q_payload, a_payload = _run_subtask_paired(
            subtask_name=subtask_name,
            run_fn=run_fn,
            sub_seed=sub_seed,
            figma_data=figma_data,
            exp_pairs=args.exp_pairs,
            match_root=match_root,
            output_dir=output_dir,
            max_tasks_per_subtask=args.max_tasks_per_subtask,
            max_episodes=args.max_episodes,
            subset_keys=subset_keys,
            use_nanobanana_for_image_content_mod=not args.no_nanobanana_content_mod,
            require_nanobanana_for_image_content_mod=not args.allow_content_mod_fallback,
            nanobanana_retries=args.nanobanana_retries,
            content_mod_max_llm_calls=args.content_mod_max_llm_calls,
            log_every=max(1, int(args.log_every)),
            save_pair_viz=args.save_pair_viz,
            pair_viz_max_per_subtask=args.pair_viz_max_per_subtask,
            num_workers=max(1, int(args.num_workers)),
            show_tqdm=not args.no_tqdm,
            build_log_every=max(0, int(args.build_log_every)),
        )

        qwen[subtask_name] = q_payload
        agent[subtask_name] = a_payload
        _save_text_subtask_outputs(output_dir, "qwen", subtask_name, q_payload)
        _save_text_subtask_outputs(output_dir, "agent", subtask_name, a_payload)
        comparison[subtask_name] = _subtask_comparison(subtask_name, q_payload, a_payload)
        print(
            f"[live-compare][text][{subtask_name}] "
            + _compact_text_compare_line(
                comparison[subtask_name],
                metric_order=[
                    "cer_overall_episode_avg",
                    "wer_overall_episode_avg",
                    "cer_edited_word",
                    "cer_untouched_words",
                    "l1",
                    "l2",
                    "psnr",
                    "ssim",
                    "lpips",
                    "dino",
                    "full_l1",
                    "full_l2",
                    "full_psnr",
                    "full_ssim",
                    "full_lpips",
                    "full_dino",
                    "iou",
                ],
            )
        )

        save_json(output_dir / "text_comparison_qwen_vs_agent.json", comparison)
        save_json(output_dir / "text_qwen_overview.json", qwen)
        save_json(output_dir / "text_agent_overview.json", agent)

    print("[DONE] text subtasks completed for qwen+agent")


if __name__ == "__main__":
    main()
