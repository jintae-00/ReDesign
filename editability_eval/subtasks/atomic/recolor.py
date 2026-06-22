#!/usr/bin/env python3
"""Atomic recolor subtask."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ._shared import run_atomic_subtask


def run(
    figma_data: Path,
    exp_pairs: Sequence[str],
    model: str,
    match_root: Path,
    output_dir: Path,
    seed: int = 123,
    max_tasks: Optional[int] = None,
    max_episodes: Optional[int] = None,
    hue_shifts: Sequence[float] = (-40.0, -20.0, 20.0, 40.0),
    sat_muls: Sequence[float] = (0.8, 1.2),
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    log_every: int = 25,
    save_pair_viz: bool = False,
    pair_viz_max: Optional[int] = None,
    reference_results: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    build_log_every: int = 0,
) -> Dict:
    param_grid = []
    for h in hue_shifts:
        for s in sat_muls:
            param_grid.append({"hue_shift_deg": float(h), "sat_mul": float(s), "val_mul": 1.0})

    return run_atomic_subtask(
        task_type="recolor",
        param_grid=param_grid,
        figma_data=figma_data,
        exp_pairs=exp_pairs,
        model=model,
        match_root=match_root,
        output_dir=output_dir,
        seed=seed,
        max_tasks=max_tasks,
        max_episodes=max_episodes,
        include_iou=False,
        roi_mode="source",
        roi_dilation_ratio=0.0,
        subset_keys=subset_keys,
        log_every=log_every,
        save_pair_viz=save_pair_viz,
        pair_viz_max=pair_viz_max,
        reference_results=reference_results,
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        build_log_every=build_log_every,
    )
