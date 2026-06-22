#!/usr/bin/env python3
"""Atomic transition subtask."""

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
    aggressive_fractions: Sequence[float] = (0.7, 0.85, 0.95),
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    log_every: int = 25,
    save_pair_viz: bool = False,
    pair_viz_max: Optional[int] = None,
    reference_results: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    build_log_every: int = 0,
) -> Dict:
    # Aggressive transition: push elements close to far corners while remaining in-canvas.
    param_grid = []
    for frac in aggressive_fractions:
        for sx in (-1, 1):
            for sy in (-1, 1):
                param_grid.append(
                    {
                        "aggressive": True,
                        "aggressive_fraction": float(frac),
                        "x_sign": int(sx),
                        "y_sign": int(sy),
                    }
                )

    return run_atomic_subtask(
        task_type="transition",
        param_grid=param_grid,
        figma_data=figma_data,
        exp_pairs=exp_pairs,
        model=model,
        match_root=match_root,
        output_dir=output_dir,
        seed=seed,
        max_tasks=max_tasks,
        max_episodes=max_episodes,
        include_iou=True,
        roi_mode="transition_dual",
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
