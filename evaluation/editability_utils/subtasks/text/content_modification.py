#!/usr/bin/env python3
"""Text content modification subtask (rule-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ...common_utils import sample_with_seed_balanced_by_key
from ._shared import (
    aggregate_content_modification,
    collect_text_pairs,
    evaluate_content_modification_pairs,
)


def run(
    figma_data: Path,
    exp_pairs: Sequence[str],
    model: str,
    match_root: Path,
    output_dir: Path,
    seed: int = 123,
    max_tasks: Optional[int] = None,
    max_episodes: Optional[int] = None,
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    use_nanobanana_for_image: bool = True,
    require_nanobanana_for_image: bool = True,
    nanobanana_retries: int = 2,
    max_nanobanana_calls: Optional[int] = None,
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
        need_pred_text=True,
        max_pairs=max_tasks,
        sample_seed=seed,
    )

    sampled = sample_with_seed_balanced_by_key(
        pairs, key_fn=lambda x: str(x["episode_id"]), max_count=max_tasks, seed=seed
    )
    results = evaluate_content_modification_pairs(
        sampled,
        cache=cache,
        model=model,
        seed=seed,
        use_nanobanana_for_image=use_nanobanana_for_image,
        require_nanobanana_for_image=require_nanobanana_for_image,
        nanobanana_retries=nanobanana_retries,
        max_nanobanana_calls=max_nanobanana_calls,
        log_every=log_every,
        save_pair_viz_dir=(output_dir / model / "content_modification" / "element_pairs") if save_pair_viz else None,
        pair_viz_max=pair_viz_max,
        reference_results=reference_results,
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        checkpoint_path=output_dir / model / "content_modification_results.json",
        resume=True,
    )
    summary = aggregate_content_modification(results)

    return {
        "capacity": {"total": len(pairs), "by_task_type": {"text_content_modification": len(pairs)}},
        "sampled_count": len(sampled),
        "results": results,
        "summary": summary,
    }
