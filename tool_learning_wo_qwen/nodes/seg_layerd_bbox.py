# src/langgraph/nodes/seg_layerd_bbox.py
from __future__ import annotations
from typing import Dict, Any
import cv2
from pathlib import Path

from ..state import GraphState
from ..reducers import (
    r_save_artifact, r_update_segment,
    r_pack_state,
    r_abort_to_verify_sequence
)
from ..utils import resolve_img
from ._common import _pop_current_action, _bump_next_or, _call_tool

SEG_MASK_MIN_PX = 8  # 통일 기준

def node(state: GraphState) -> Dict[str, Any]:
    act, upd0 = _pop_current_action(state)

    img_sel = resolve_img("segment", state)
    det = (state.get("_runtime") or {}).get("detect") or {}
    boxes   = det.get("boxes") or []
    det_ids = det.get("det_ids") or []

    out = _call_tool("LayerD_SegBBox", {
        "image_path": img_sel["path"],
        "boxes": boxes,
        "det_ids": det_ids
    }, state)

    pieces = [upd0]

    union = (out or {}).get("mask_union")
    by_id = (out or {}).get("masks_by_id") or {}

    saved = None
    if union:
        extra, saved = r_save_artifact(union, "layerd_bbox", "segment", state)
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
            node="layerd_bbox", 
            error_msg=f"layerd_bbox segmentation failed: segmentation mask area too small (area={int(area)})",
            details={"mask_area": int(area), "threshold": int(SEG_MASK_MIN_PX)},
        )

    # 정상 경로: update_segment 한 번으로 segment + img["seg"] + base_img 동시 반영
    pieces += [
        r_update_segment(
            "layerd_bbox", saved, state,
            masks_by_id=by_id,
            base_img=img_sel["path"],
            img_slot="seg",
            coord_id="seg",
        ),
        _bump_next_or(state, "tool_sequence_plan"),
    ]
    return r_pack_state(state, *pieces)

