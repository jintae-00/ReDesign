#!/usr/bin/env python3
"""SVG point edit subtask (corner warp)."""

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
    offset_ratios: Sequence[float] = (0.25, 0.35, 0.45),
    subset_keys: Optional[Set[Tuple[str, int]]] = None,
    log_every: int = 0,
    save_pair_viz: bool = False,
    pair_viz_max: Optional[int] = None,
    reference_results: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 1,
    show_tqdm: bool = True,
    build_log_every: int = 0,
) -> Dict:
    params = []

    # Aggressive single-corner displacements.
    for c in [0, 1, 2, 3]:
        for r in offset_ratios:
            params.append({"corner": int(c), "dx_ratio": float(r), "dy_ratio": float(r)})
            params.append({"corner": int(c), "dx_ratio": -float(r), "dy_ratio": float(r)})
            params.append({"corner": int(c), "dx_ratio": float(r), "dy_ratio": -float(r)})
            params.append({"corner": int(c), "dx_ratio": -float(r), "dy_ratio": -float(r)})

    # Complex 4-corner perspective warps (harder than simple one-corner edits).
    for r in offset_ratios:
        rr = float(r)
        params.extend(
            [
                # Top edge squeeze / spread
                {"corner_offsets": [[rr, 0.0], [-rr, 0.0], [0.0, 0.0], [0.0, 0.0]]},
                {"corner_offsets": [[-rr, 0.0], [rr, 0.0], [0.0, 0.0], [0.0, 0.0]]},
                # Bottom edge squeeze / spread
                {"corner_offsets": [[0.0, 0.0], [0.0, 0.0], [-rr, 0.0], [rr, 0.0]]},
                {"corner_offsets": [[0.0, 0.0], [0.0, 0.0], [rr, 0.0], [-rr, 0.0]]},
                # Left/right perspective tilt
                {"corner_offsets": [[0.0, rr], [0.0, -rr], [0.0, 0.0], [0.0, 0.0]]},
                {"corner_offsets": [[0.0, -rr], [0.0, rr], [0.0, 0.0], [0.0, 0.0]]},
                # Twisted perspective
                {"corner_offsets": [[rr, rr], [-rr, rr], [rr, -rr], [-rr, -rr]]},
                {"corner_offsets": [[-rr, -rr], [rr, -rr], [-rr, rr], [rr, rr]]},
            ]
        )

    return run_svg_subtask(
        task_type="point_edit",
        param_grid=params,
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
        roi_mode="source_target",
        require_stroke=False,
        subset_keys=subset_keys,
        log_every=log_every,
        save_pair_viz=save_pair_viz,
        pair_viz_max=pair_viz_max,
        reference_results=reference_results,
        num_workers=num_workers,
        show_tqdm=show_tqdm,
        build_log_every=build_log_every,
    )
