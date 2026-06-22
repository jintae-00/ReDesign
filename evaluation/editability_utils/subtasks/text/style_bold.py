#!/usr/bin/env python3
"""Text style bold subtask."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..common import aggregate_results, summarize_capacity
from ._shared import build_style_candidates, collect_text_pairs, evaluate_style_subtask


def run(
    figma_data: Path,
    exp_pairs: Sequence[str],
    model: str,
    match_root: Path,
    output_dir: Path,
    seed: int = 123,
    max_tasks: Optional[int] = None,
    max_episodes: Optional[int] = None,
    strengths: Sequence[int] = (2, 3, 4),
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    log_every: int = 0,
    save_pair_viz: bool = False,
    pair_viz_max: Optional[int] = None,
    reference_results: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    build_log_every: int = 0,
) -> Dict:
    _, pairs, cache = collect_text_pairs(
        figma_data=figma_data,
        exp_pairs=exp_pairs,
        model=model,
        match_root=match_root,
        max_episodes=max_episodes,
        subset_keys=subset_keys,
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        build_log_every=build_log_every,
        need_pred_text=False,
    )
    candidates = build_style_candidates(pairs, "text_bold", [{"strength": int(v)} for v in strengths])
    results = evaluate_style_subtask(
        candidates,
        cache,
        seed=seed,
        max_tasks=max_tasks,
        progress_prefix=f"[{model}][style_bold]",
        log_every=log_every,
        save_pair_viz_dir=(output_dir / model / "style_bold" / "element_pairs") if save_pair_viz else None,
        pair_viz_max=pair_viz_max,
        reference_results=reference_results,
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        checkpoint_path=output_dir / model / "style_bold_results.json",
        resume=True,
    )
    return {
        "capacity": summarize_capacity(candidates),
        "sampled_count": len(results),
        "results": results,
        "summary": aggregate_results(results),
    }
