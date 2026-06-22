#!/usr/bin/env python3
"""SVG recolor subtask (vector: deterministic, image: nanobanana)."""

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
    hue_shifts: Sequence[float] = (-90.0, -45.0, 45.0, 90.0),
    sat_muls: Sequence[float] = (0.5, 1.8),
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    use_nanobanana_for_image_recolor: bool = True,
    require_nanobanana_for_image_recolor: bool = True,
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
    params = []
    for h in hue_shifts:
        for s in sat_muls:
            params.append(
                {
                    "hue_shift_deg": float(h),
                    "sat_mul": float(s),
                    "val_mul": 1.0,
                }
            )

    return run_svg_subtask(
        task_type="recolor",
        param_grid=params,
        figma_data=figma_data,
        exp_pairs=exp_pairs,
        model=model,
        match_root=match_root,
        output_dir=output_dir,
        seed=seed,
        max_tasks=max_tasks,
        max_episodes=max_episodes,
        include_iou=False,
        include_edge_sharpness=False,
        include_lpips=True,
        roi_mode="source",
        roi_dilation_ratio=0.0,
        require_stroke=False,
        subset_keys=subset_keys,
        use_nanobanana_for_image_recolor=use_nanobanana_for_image_recolor,
        require_nanobanana_for_image_recolor=require_nanobanana_for_image_recolor,
        nanobanana_retries=nanobanana_retries,
        max_nanobanana_calls=max_nanobanana_calls,
        log_every=log_every,
        save_pair_viz=save_pair_viz,
        pair_viz_max=pair_viz_max,
        reference_results=reference_results,
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        build_log_every=build_log_every,
    )
