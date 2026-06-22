#!/usr/bin/env python3
"""Runner utilities for GT:model matching and saving match pairs."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .common_utils import save_json
from .loaders import EpisodeTask, collect_episode_tasks, load_episode_elements
from .matching_core import MatchConfig, greedy_match_gt_to_pred
from .matching_visuals import save_match_visualizations

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _print_match_breakdown(model: str, episode_id: str, stats: Dict[str, Any]) -> None:
    total_eval = float(stats.get("total_eval_sec", 0.0))
    total_union = float(stats.get("total_union_render_sec", 0.0))
    total_filter = float(stats.get("total_candidate_filter_sec", 0.0))
    total_prep = float(stats.get("total_gt_prepare_sec", 0.0))
    cache_hit = int(stats.get("union_cache_hit", 0))
    cache_miss = int(stats.get("union_cache_miss", 0))
    cache_total = cache_hit + cache_miss
    hit_ratio = (100.0 * cache_hit / cache_total) if cache_total > 0 else 0.0
    print(
        f"[{model}] ep={episode_id} breakdown | "
        f"pred_cache={float(stats.get('pred_cache_prepare_sec', 0.0)):.2f}s "
        f"gt_prep={total_prep:.2f}s cand_filter={total_filter:.2f}s "
        f"eval_total={total_eval:.2f}s union_render={total_union:.2f}s "
        f"union_cache_hit={cache_hit}/{cache_total}({hit_ratio:.1f}%) "
        f"eval_empty/single/multi={int(stats.get('eval_count_empty', 0))}/"
        f"{int(stats.get('eval_count_single', 0))}/{int(stats.get('eval_count_multi', 0))}",
        flush=True,
    )


def _make_gt_progress_cb(
    model: str,
    episode_id: str,
    show_progress: bool,
    iterable: Any,
    trace_episodes: bool,
    detailed_logs: bool,
    gt_progress_every: int,
    gt_progress_sec: float,
) -> Any:
    last_gt_log_t = time.time()

    def _on_gt_progress(done: int, total: int, info: Dict[str, Any]) -> None:
        nonlocal last_gt_log_t
        if show_progress and tqdm is not None and hasattr(iterable, "set_postfix"):
            iterable.set_postfix(
                {
                    "episode": episode_id[-12:],
                    "phase": f"match {done}/{total}",
                    "gt_cand": int(info.get("gt_candidates", 0)),
                    "eval": int(info.get("cost_evals", 0)),
                },
                refresh=False,
            )
        now = time.time()
        should_log = done == total or done % max(1, gt_progress_every) == 0 or (now - last_gt_log_t) >= gt_progress_sec
        if not should_log:
            return

        if detailed_logs:
            msg = (
                f"[{model}] ep={episode_id} gt {done}/{total} | "
                f"gt_cand={int(info.get('gt_candidates', 0))} "
                f"gt_eval={int(info.get('gt_cost_evals', 0))} "
                f"gt_eval_sec={float(info.get('gt_eval_sec', 0.0)):.3f}s "
                f"gt_union_sec={float(info.get('gt_union_render_sec', 0.0)):.3f}s "
                f"gt_union_hit/miss={int(info.get('gt_union_cache_hit', 0))}/"
                f"{int(info.get('gt_union_cache_miss', 0))} "
                f"cum_eval={int(info.get('cost_evals', 0))}"
            )
        else:
            msg = (
                f"[{model}] ep={episode_id} gt {done}/{total} | "
                f"avg_cand={float(info.get('avg_candidates', 0.0)):.1f} "
                f"cost_eval={int(info.get('cost_evals', 0))}"
            )
        if show_progress and tqdm is not None:
            tqdm.write(msg)
        elif trace_episodes or detailed_logs:
            print(msg, flush=True)
        last_gt_log_t = now

    return _on_gt_progress


def _episode_output_path(per_episode_dir: Path, episode_id: str) -> Path:
    return per_episode_dir / f"{episode_id}.json"


def _cap_workers(num_workers: int, has_visualization: bool) -> int:
    cpu_n = os.cpu_count() or 1
    effective = max(1, min(int(num_workers), int(cpu_n)))
    # Pair/episode visualization adds substantial memory + IO pressure per process.
    if has_visualization:
        effective = min(effective, 16)
    return max(1, effective)


def _run_single_episode(
    task: EpisodeTask,
    model: str,
    per_episode_dir: Path,
    cfg: MatchConfig,
    trace_episodes: bool = False,
    detailed_logs: bool = False,
    gt_progress_every: int = 25,
    gt_progress_sec: float = 10.0,
    visualize_episode_out_dir: Optional[Path] = None,
    visualize_pair_out_dir: Optional[Path] = None,
    viz_max_rows: int = 80,
    viz_panel_width: int = 220,
) -> Dict[str, Any]:
    ep_t0 = time.time()

    load_t0 = time.time()
    gt_elements, pred_elements, canvas_size = load_episode_elements(task, model=model)
    load_dt = time.time() - load_t0

    progress_cb = None
    if detailed_logs or trace_episodes:
        progress_cb = _make_gt_progress_cb(
            model=model,
            episode_id=task.episode_id,
            show_progress=False,
            iterable=None,
            trace_episodes=trace_episodes,
            detailed_logs=detailed_logs,
            gt_progress_every=gt_progress_every,
            gt_progress_sec=gt_progress_sec,
        )

    match_t0 = time.time()
    matches, match_stats = greedy_match_gt_to_pred(
        gt_elements,
        pred_elements,
        canvas_size,
        cfg,
        progress_cb=progress_cb,
    )
    match_dt = time.time() - match_t0

    viz_dt = 0.0
    viz_info: Dict[str, Any] = {}
    if visualize_episode_out_dir is not None or visualize_pair_out_dir is not None:
        viz_t0 = time.time()
        episode_viz_path = None
        pair_viz_dir = None
        if visualize_episode_out_dir is not None:
            episode_viz_path = visualize_episode_out_dir / f"{task.episode_id}.png"
        if visualize_pair_out_dir is not None:
            pair_viz_dir = visualize_pair_out_dir / task.episode_id
        viz_info = save_match_visualizations(
            episode_id=task.episode_id,
            split_name=task.split_name,
            payload={
                "episode_id": task.episode_id,
                "split": task.split_name,
                "canvas_size": list(canvas_size),
                "counts": {"gt": len(gt_elements), "pred": len(pred_elements), "matches": len(matches)},
                "matches": matches,
            },
            gt_elements=gt_elements,
            pred_elements=pred_elements,
            canvas_size=canvas_size,
            episode_out_path=episode_viz_path,
            pair_out_dir=pair_viz_dir,
            parsed_layers_src_dir=(task.qwen_episode_dir if model == "qwen" else None),
            max_rows=viz_max_rows,
            panel_width=viz_panel_width,
        )
        viz_dt = time.time() - viz_t0

    episode_payload: Dict[str, Any] = {
        "episode_id": task.episode_id,
        "split": task.split_name,
        "canvas_size": list(canvas_size),
        "counts": {
            "gt": len(gt_elements),
            "pred": len(pred_elements),
            "matches": len(matches),
        },
        "timing_sec": {
            "load": float(load_dt),
            "match": float(match_dt),
            "total": float(time.time() - ep_t0),
        },
        "match_stats": match_stats,
        "matches": matches,
    }
    if viz_info:
        episode_payload["visualization"] = viz_info
        episode_payload["timing_sec"]["visualize"] = float(viz_dt)

    save_t0 = time.time()
    episode_path = per_episode_dir / f"{task.episode_id}.json"
    episode_payload["timing_sec"]["save"] = 0.0
    episode_payload["timing_sec"]["total"] = float(time.time() - ep_t0)
    save_json(episode_path, episode_payload)
    save_dt = time.time() - save_t0
    episode_payload["timing_sec"]["save"] = float(save_dt)
    episode_payload["timing_sec"]["total"] = float(time.time() - ep_t0)

    if detailed_logs:
        _print_match_breakdown(model, task.episode_id, match_stats)

    return {
        "episode_id": task.episode_id,
        "split": task.split_name,
        "counts": episode_payload["counts"],
        "timing_sec": episode_payload["timing_sec"],
        "match_stats": match_stats,
        "visualization": viz_info,
        "path": str(episode_path.resolve()),
    }


def _run_matching_single_worker(
    tasks: Sequence[EpisodeTask],
    model: str,
    per_episode_dir: Path,
    cfg: MatchConfig,
    show_progress: bool,
    progress_desc: Optional[str],
    trace_episodes: bool,
    detailed_logs: bool,
    slow_episode_sec: float,
    gt_progress_every: int,
    gt_progress_sec: float,
    visualize_episode_out_dir: Optional[Path],
    visualize_pair_out_dir: Optional[Path],
    viz_max_rows: int,
    viz_panel_width: int,
) -> List[Dict[str, Any]]:
    episodes_out: List[Dict[str, Any]] = []
    iterable = tasks
    if show_progress and tqdm is not None:
        iterable = tqdm(
            tasks,
            desc=progress_desc or f"matching:{model}",
            unit="ep",
            dynamic_ncols=True,
            leave=True,
        )

    num_tasks = len(tasks)
    for i, task in enumerate(iterable, start=1):
        ep_t0 = time.time()
        if trace_episodes:
            if show_progress and tqdm is not None:
                tqdm.write(f"[{model}] episode {i}/{num_tasks} start: {task.episode_id}")
            else:
                print(f"[{model}] episode {i}/{num_tasks} start: {task.episode_id}", flush=True)

        load_t0 = time.time()
        gt_elements, pred_elements, canvas_size = load_episode_elements(task, model=model)
        load_dt = time.time() - load_t0

        if show_progress and tqdm is not None and hasattr(iterable, "set_postfix"):
            iterable.set_postfix(
                {
                    "episode": task.episode_id[-12:],
                    "phase": "match",
                    "gt": len(gt_elements),
                    "pred": len(pred_elements),
                },
                refresh=False,
            )

        _on_gt_progress = _make_gt_progress_cb(
            model=model,
            episode_id=task.episode_id,
            show_progress=show_progress,
            iterable=iterable,
            trace_episodes=trace_episodes,
            detailed_logs=detailed_logs,
            gt_progress_every=gt_progress_every,
            gt_progress_sec=gt_progress_sec,
        )

        match_t0 = time.time()
        matches, match_stats = greedy_match_gt_to_pred(
            gt_elements,
            pred_elements,
            canvas_size,
            cfg,
            progress_cb=_on_gt_progress,
        )
        match_dt = time.time() - match_t0

        viz_dt = 0.0
        viz_info: Dict[str, Any] = {}
        if visualize_episode_out_dir is not None or visualize_pair_out_dir is not None:
            viz_t0 = time.time()
            episode_viz_path = None
            pair_viz_dir = None
            if visualize_episode_out_dir is not None:
                episode_viz_path = visualize_episode_out_dir / f"{task.episode_id}.png"
            if visualize_pair_out_dir is not None:
                pair_viz_dir = visualize_pair_out_dir / task.episode_id
            viz_info = save_match_visualizations(
                episode_id=task.episode_id,
                split_name=task.split_name,
                payload={
                    "episode_id": task.episode_id,
                    "split": task.split_name,
                    "canvas_size": list(canvas_size),
                    "counts": {"gt": len(gt_elements), "pred": len(pred_elements), "matches": len(matches)},
                    "matches": matches,
                },
                gt_elements=gt_elements,
                pred_elements=pred_elements,
                canvas_size=canvas_size,
                episode_out_path=episode_viz_path,
                pair_out_dir=pair_viz_dir,
                parsed_layers_src_dir=(task.qwen_episode_dir if model == "qwen" else None),
                max_rows=viz_max_rows,
                panel_width=viz_panel_width,
            )
            viz_dt = time.time() - viz_t0

        episode_payload: Dict[str, Any] = {
            "episode_id": task.episode_id,
            "split": task.split_name,
            "canvas_size": list(canvas_size),
            "counts": {
                "gt": len(gt_elements),
                "pred": len(pred_elements),
                "matches": len(matches),
            },
            "timing_sec": {
                "load": float(load_dt),
                "match": float(match_dt),
                "total": float(time.time() - ep_t0),
            },
            "match_stats": match_stats,
            "matches": matches,
        }
        if viz_info:
            episode_payload["visualization"] = viz_info
            episode_payload["timing_sec"]["visualize"] = float(viz_dt)

        save_t0 = time.time()
        episode_path = per_episode_dir / f"{task.episode_id}.json"
        episode_payload["timing_sec"]["save"] = 0.0
        episode_payload["timing_sec"]["total"] = float(time.time() - ep_t0)
        save_json(episode_path, episode_payload)
        save_dt = time.time() - save_t0
        episode_payload["timing_sec"]["save"] = float(save_dt)
        episode_payload["timing_sec"]["total"] = float(time.time() - ep_t0)

        episodes_out.append(
            {
                "episode_id": task.episode_id,
                "split": task.split_name,
                "counts": episode_payload["counts"],
                "timing_sec": episode_payload["timing_sec"],
                "match_stats": match_stats,
                "visualization": viz_info,
                "path": str(episode_path.resolve()),
            }
        )

        if show_progress and tqdm is not None and hasattr(iterable, "set_postfix"):
            iterable.set_postfix(
                {
                    "episode": task.episode_id[-12:],
                    "gt": len(gt_elements),
                    "pred": len(pred_elements),
                    "pairs": len(matches),
                },
                refresh=False,
            )

        ep_total = float(time.time() - ep_t0)
        if trace_episodes or ep_total >= slow_episode_sec:
            msg = (
                f"[{model}] ep {i}/{num_tasks} done {task.episode_id} | "
                f"load={load_dt:.2f}s match={match_dt:.2f}s save={save_dt:.2f}s total={ep_total:.2f}s | "
                f"cost_eval={match_stats.get('total_cost_evals', 0)} "
                f"avg_cand={match_stats.get('avg_candidates_per_gt', 0.0):.2f} "
                f"max_cand={match_stats.get('max_candidates_per_gt', 0)}"
            )
            if show_progress and tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg, flush=True)
        if detailed_logs:
            _print_match_breakdown(model, task.episode_id, match_stats)

    return episodes_out


def _run_matching_multi_worker(
    tasks: Sequence[EpisodeTask],
    model: str,
    per_episode_dir: Path,
    cfg: MatchConfig,
    num_workers: int,
    mp_start_method: str,
    max_tasks_per_child: Optional[int],
    show_progress: bool,
    progress_desc: Optional[str],
    trace_episodes: bool,
    detailed_logs: bool,
    gt_progress_every: int,
    gt_progress_sec: float,
    visualize_episode_out_dir: Optional[Path],
    visualize_pair_out_dir: Optional[Path],
    viz_max_rows: int,
    viz_panel_width: int,
) -> List[Dict[str, Any]]:
    episodes_out: List[Dict[str, Any]] = []
    num_tasks = len(tasks)
    has_visualization = (visualize_episode_out_dir is not None) or (visualize_pair_out_dir is not None)
    effective_workers = _cap_workers(num_workers, has_visualization=has_visualization)
    if effective_workers != int(num_workers):
        print(
            f"[{model}] worker cap applied: requested={num_workers} -> effective={effective_workers} "
            f"(cpu_count={os.cpu_count()}, visualization={has_visualization})",
            flush=True,
        )
    print(f"[{model}] multi-worker mode: workers={effective_workers}, backend=CPU(numpy)", flush=True)

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(
            total=num_tasks,
            desc=progress_desc or f"matching:{model}",
            unit="ep",
            dynamic_ncols=True,
            leave=True,
        )
    done_episode_ids = set()
    run_t0 = time.time()

    done_idx = 0
    total_match_sec = 0.0
    total_episode_sec = 0.0
    last_episode_tail = "-"
    last_refresh_t = 0.0

    def _refresh_multi_progress(pending_futures: Sequence[Any], force: bool = False) -> None:
        nonlocal last_refresh_t
        if progress is None or not hasattr(progress, "set_postfix"):
            return
        now = time.time()
        if not force and (now - last_refresh_t) < 0.5:
            return
        outstanding = len(pending_futures)
        running = min(outstanding, effective_workers)
        queued = max(0, outstanding - running)
        avg_match = (total_match_sec / done_idx) if done_idx > 0 else 0.0
        avg_total = (total_episode_sec / done_idx) if done_idx > 0 else 0.0
        remaining = max(0, num_tasks - done_idx)
        eta_sec = avg_total * remaining
        oldest_run = 0.0
        if outstanding > 0:
            oldest_started = min(float(futures[f][2]) for f in pending_futures)
            oldest_run = max(0.0, now - oldest_started)
        progress.set_postfix(
            {
                "done": f"{done_idx}/{num_tasks}",
                "run/q": f"{running}/{queued}",
                "avg_m": f"{avg_match:.1f}s",
                "eta": f"{eta_sec/60.0:.1f}m",
                "oldest": f"{oldest_run:.1f}s",
                "last": last_episode_tail,
            },
            refresh=False,
        )
        last_refresh_t = now

    try:
        pool_kwargs: Dict[str, Any] = {
            "max_workers": effective_workers,
            "mp_context": mp.get_context(mp_start_method),
        }
        if max_tasks_per_child is not None and int(max_tasks_per_child) > 0:
            pool_kwargs["max_tasks_per_child"] = int(max_tasks_per_child)
        with ProcessPoolExecutor(**pool_kwargs) as ex:
            futures = {}
            for idx, task in enumerate(tasks):
                fut = ex.submit(
                    _run_single_episode,
                    task,
                    model,
                    per_episode_dir,
                    cfg,
                    trace_episodes,
                    detailed_logs,
                    gt_progress_every,
                    gt_progress_sec,
                    visualize_episode_out_dir,
                    visualize_pair_out_dir,
                    viz_max_rows,
                    viz_panel_width,
                )
                futures[fut] = (idx + 1, task, time.time())

            pending = set(futures.keys())
            _refresh_multi_progress(pending, force=True)
            while pending:
                finished, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                if not finished:
                    _refresh_multi_progress(pending, force=False)
                    continue

                for fut in sorted(finished, key=lambda x: futures[x][0]):
                    submit_idx, task, _submitted_t = futures[fut]
                    try:
                        episode_summary = fut.result()
                    except BrokenProcessPool as e:
                        raise BrokenProcessPool(
                            f"[{model}] BrokenProcessPool while processing episode={task.episode_id}. "
                            f"This usually indicates worker OOM/segfault. "
                            f"Try fewer workers and/or disable pair visualization."
                        ) from e
                    except Exception as e:
                        raise RuntimeError(
                            f"[{model}] worker failed on episode={task.episode_id} (submitted #{submit_idx})"
                        ) from e

                    done_idx += 1
                    total_match_sec += float(episode_summary["timing_sec"]["match"])
                    total_episode_sec += float(episode_summary["timing_sec"]["total"])
                    last_episode_tail = task.episode_id[-12:]

                    episodes_out.append(episode_summary)
                    done_episode_ids.add(task.episode_id)

                    if progress is not None:
                        progress.update(1)
                    _refresh_multi_progress(pending, force=True)

                    if trace_episodes:
                        print(
                            f"[{model}] done {done_idx}/{num_tasks} ep={task.episode_id} "
                            f"(submitted #{submit_idx}) "
                            f"load={episode_summary['timing_sec']['load']:.2f}s "
                            f"match={episode_summary['timing_sec']['match']:.2f}s",
                            flush=True,
                        )
    except BrokenProcessPool as e:
        remaining = [t for t in tasks if t.episode_id not in done_episode_ids]
        print(str(e), flush=True)
        print(
            f"[{model}] fallback to single-worker for remaining episodes={len(remaining)}",
            flush=True,
        )
        if progress is not None:
            progress.close()
            progress = None
        if remaining:
            fallback = _run_matching_single_worker(
                tasks=remaining,
                model=model,
                per_episode_dir=per_episode_dir,
                cfg=cfg,
                show_progress=show_progress,
                progress_desc=(progress_desc or f"matching:{model}") + ":fallback",
                trace_episodes=trace_episodes,
                detailed_logs=detailed_logs,
                slow_episode_sec=0.0,
                gt_progress_every=gt_progress_every,
                gt_progress_sec=gt_progress_sec,
                visualize_episode_out_dir=visualize_episode_out_dir,
                visualize_pair_out_dir=visualize_pair_out_dir,
                viz_max_rows=viz_max_rows,
                viz_panel_width=viz_panel_width,
            )
            episodes_out.extend(fallback)

    if progress is not None and done_idx > 0:
        elapsed = time.time() - run_t0
        avg_match = total_match_sec / max(1, done_idx)
        avg_total = total_episode_sec / max(1, done_idx)
        progress.set_postfix(
            {
                "done": f"{done_idx}/{num_tasks}",
                "avg_m": f"{avg_match:.1f}s",
                "avg_ep": f"{avg_total:.1f}s",
                "wall": f"{elapsed/60.0:.1f}m",
                "last": last_episode_tail,
            },
            refresh=False,
        )

    if progress is not None:
        progress.close()

    episodes_out.sort(key=lambda x: x["episode_id"])
    return episodes_out


def run_matching(
    figma_data: Path,
    exp_pairs: Sequence[str],
    model: str,
    output_dir: Path,
    max_episodes: Optional[int],
    cfg: MatchConfig,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
    trace_episodes: bool = False,
    detailed_logs: bool = False,
    slow_episode_sec: float = 30.0,
    gt_progress_every: int = 25,
    gt_progress_sec: float = 10.0,
    num_workers: int = 1,
    mp_start_method: str = "spawn",
    max_tasks_per_child: Optional[int] = 8,
    visualize_episode_dir: Optional[Path] = None,
    visualize_pair_dir: Optional[Path] = None,
    viz_max_rows: int = 80,
    viz_panel_width: int = 220,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    t0 = time.time()
    collected_tasks = collect_episode_tasks(
        figma_data_dir=figma_data,
        exp_pairs=exp_pairs,
        model=model,
        max_episodes=max_episodes,
    )
    print(f"[{model}] collected episodes={len(collected_tasks)}", flush=True)

    if mp_start_method not in {"spawn", "fork", "forkserver"}:
        raise ValueError(
            f"Invalid mp_start_method={mp_start_method!r}. Expected one of: spawn, fork, forkserver"
        )

    summary: Dict[str, Any] = {
        "model": model,
        "num_episodes": len(collected_tasks),
        "config": {
            "lambda_l1": cfg.lambda_l1,
            "lambda_iou": cfg.lambda_iou,
            "max_merge_n": cfg.max_merge_n,
            "min_gt_overlap": cfg.min_gt_overlap,
            "min_cost_improve": cfg.min_cost_improve,
            "num_workers": int(num_workers),
            "mp_start_method": str(mp_start_method),
            "max_tasks_per_child": (None if max_tasks_per_child is None else int(max_tasks_per_child)),
            "detailed_logs": bool(detailed_logs),
            "viz_max_rows": int(viz_max_rows),
            "viz_panel_width": int(viz_panel_width),
            "skip_existing": bool(skip_existing),
        },
        "num_skipped_existing": 0,
        "num_to_run": 0,
        "episodes": [],
    }

    per_episode_dir = output_dir / model / "episodes"
    per_episode_dir.mkdir(parents=True, exist_ok=True)
    model_episode_viz_dir = None
    model_pair_viz_dir = None
    if visualize_episode_dir is not None:
        model_episode_viz_dir = visualize_episode_dir / model / "episodes"
        model_episode_viz_dir.mkdir(parents=True, exist_ok=True)
    if visualize_pair_dir is not None:
        model_pair_viz_dir = visualize_pair_dir / model / "pairs"
        model_pair_viz_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing:
        tasks = []
        skipped = 0
        for t in collected_tasks:
            if _episode_output_path(per_episode_dir, t.episode_id).exists():
                skipped += 1
                continue
            tasks.append(t)
        summary["num_skipped_existing"] = int(skipped)
        print(f"[{model}] skip_existing=True skipped={skipped} to_run={len(tasks)}", flush=True)
    else:
        tasks = list(collected_tasks)
    summary["num_to_run"] = int(len(tasks))

    if len(tasks) == 0:
        save_json(output_dir / model / "summary.json", summary)
        return summary

    if num_workers <= 1:
        episodes = _run_matching_single_worker(
            tasks=tasks,
            model=model,
            per_episode_dir=per_episode_dir,
            cfg=cfg,
            show_progress=show_progress,
            progress_desc=progress_desc,
            trace_episodes=trace_episodes,
            detailed_logs=detailed_logs,
            slow_episode_sec=slow_episode_sec,
            gt_progress_every=gt_progress_every,
            gt_progress_sec=gt_progress_sec,
            visualize_episode_out_dir=model_episode_viz_dir,
            visualize_pair_out_dir=model_pair_viz_dir,
            viz_max_rows=viz_max_rows,
            viz_panel_width=viz_panel_width,
        )
    else:
        episodes = _run_matching_multi_worker(
            tasks=tasks,
            model=model,
            per_episode_dir=per_episode_dir,
            cfg=cfg,
            num_workers=num_workers,
            mp_start_method=mp_start_method,
            max_tasks_per_child=max_tasks_per_child,
            show_progress=show_progress,
            progress_desc=progress_desc,
            trace_episodes=trace_episodes,
            detailed_logs=detailed_logs,
            gt_progress_every=gt_progress_every,
            gt_progress_sec=gt_progress_sec,
            visualize_episode_out_dir=model_episode_viz_dir,
            visualize_pair_out_dir=model_pair_viz_dir,
            viz_max_rows=viz_max_rows,
            viz_panel_width=viz_panel_width,
        )

    summary["episodes"] = episodes
    save_json(output_dir / model / "summary.json", summary)
    dt = time.time() - t0
    print(
        f"[{model}] done to_run={len(tasks)} skipped_existing={summary['num_skipped_existing']} elapsed={dt:.1f}s",
        flush=True,
    )
    return summary
