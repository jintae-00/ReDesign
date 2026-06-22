# src/langgraph/nodes/detect_gdino.py
from __future__ import annotations
from typing import Dict, Any, List
from ..state import GraphState
from ..reducers import (
    r_update_detect, r_save_artifact,
    r_set_img_slot, r_pack_state,
    r_abort_to_verify_sequence
)
from ..utils import resolve_args, resolve_img
from ._common import _pop_current_action, _bump_next_or, _call_tool

def node(state: GraphState) -> Dict[str, Any]:

    act, upd0 = _pop_current_action(state)

    raw_args = act.get("args") or {}
    labels = state.get("front_labels")
    args = resolve_args(raw_args, last=None, state=state, labels=None, image_kind="detect")
    img_sel = resolve_img("detect", state)
    args.setdefault("image_path", img_sel["path"])
    args.setdefault("labels", labels)

    out = _call_tool("GDINO", args, state)

    pieces = [upd0]

    viz_path = out.get("viz")
    if viz_path:
        extra, saved = r_save_artifact(viz_path, "gdino", "detect", state)
        pieces.append(extra)
        viz_path = saved

    boxes  = out.get("boxes") or []
    scores = out.get("confs") or []
    labs   = out.get("labels") or []

    # === Guard: no boxes ===
    if len(boxes) == 0:
        return r_abort_to_verify_sequence(
            state,
            node="gdino",
            error_msg="gdino detection failed: no object detected",
            details={"num_object_bboxes": 0},
        )

    seq_id = state.get("seq_id")
    det_ids = [f"o-seq_{seq_id:03d}-gdino-box_{i:02d}" for i in range(len(boxes))]

    pieces += [
        r_set_img_slot("detect", args["image_path"], coord_id="detect", state=state),
        r_update_detect("gdino", boxes, scores, viz_path, state, labels=labs, det_ids=det_ids),
        _bump_next_or(state, "tool_sequence_plan"),
    ]

    return r_pack_state(state, *pieces)
