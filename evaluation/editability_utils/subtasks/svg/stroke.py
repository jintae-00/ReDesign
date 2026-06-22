#!/usr/bin/env python3
"""SVG stroke subtask."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ._shared import run_svg_subtask


def run(
    figma_data: Path,
    exp_pairs: Sequence[str],
    model: str,
    match_root: Path,
    output_dir: Path,
    seed: int = 123,
    max_tasks: Optional[int] = None,
    max_episodes: Optional[int] = None,
    stroke_widths: Sequence[int] = (2, 4, 6),
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    log_every: int = 0,
    save_pair_viz: bool = False,
    pair_viz_max: Optional[int] = None,
    reference_results: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    build_log_every: int = 0,
) -> Dict:
    param_grid = [{"stroke_width": int(w), "stroke_rgb": [0, 0, 0]} for w in stroke_widths]

    return run_svg_subtask(
        task_type="stroke",
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
        include_edge_sharpness=True,
        include_lpips=True,
        roi_mode="source",
        roi_dilation_ratio=0.0,
        require_stroke=True,
        subset_keys=subset_keys,
        log_every=log_every,
        save_pair_viz=save_pair_viz,
        pair_viz_max=pair_viz_max,
        reference_results=reference_results,
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        build_log_every=build_log_every,
    )
