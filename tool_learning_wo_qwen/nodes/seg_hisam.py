# src/langgraph/nodes/seg_hisam.py
from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import cv2
from ..state import GraphState
from ..reducers import (
    r_save_artifact,
    r_update_segment,
    r_pack_state,
    r_abort_to_verify_sequence
)
from ..utils import resolve_args, resolve_img, boxes_to_aabbs
from ._common import _pop_current_action, _bump_next_or, _call_tool

SEG_MASK_MIN_PX = 8  # 통일 기준


def node(state: GraphState) -> Dict[str, Any]:
    act, upd0 = _pop_current_action(state)

    raw_args = act.get("args") or {}
    args = resolve_args(raw_args, last=None, state=state, labels=None, image_kind="segment")

    img_sel = resolve_img("segment", state)
    args.setdefault("image_path", img_sel["path"])

    prev_boxes = (state["_runtime"]["detect"].get("boxes") or [])
    det_ids = (state["_runtime"]["detect"].get("det_ids") or [])

    aabbs = boxes_to_aabbs(prev_boxes)
    args["boxes"] = aabbs
    args["det_ids"] = det_ids

    out = _call_tool("HiSAMUnion", args, state)  # {"mask_union": ..., "masks_by_id": {...}}

    pieces = [upd0]
    union = (out or {}).get("mask_union")
    by_id = (out or {}).get("masks_by_id") or {}

    saved = None
    if union:
        extra, saved = r_save_artifact(union, "hisam", "segment", state)
        pieces.append(extra)

    # === Guard: mask too small ===
    area = 0
    if saved:
        try:
            m = cv2.imread(saved, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                area = int((m > 0).sum())
        except Exception:
            area = 0

    if area < SEG_MASK_MIN_PX:
        return r_abort_to_verify_sequence(
            state,
            node="hisam",
            error_msg=f"hisam segmentation failed: segmentation mask area too small (area={int(area)})",
            details={"mask_area": int(area), "threshold": int(SEG_MASK_MIN_PX)},
        )

    # 정상 경로: update_segment 한 번으로 segment + img["seg"] + base_img 동시 반영
    pieces += [
        r_update_segment(
            "hisam", saved, state,
            masks_by_id=by_id,
            base_img=args["image_path"],
            img_slot="seg",
            coord_id="seg",
        ),
        _bump_next_or(state, "tool_sequence_plan"),
    ]
    return r_pack_state(state, *pieces)
