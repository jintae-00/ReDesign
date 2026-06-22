# src/langgraph/nodes/detect_ocr.py
from __future__ import annotations
from typing import Dict, Any
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
    args = resolve_args(raw_args, last=None, state=state, labels=None, image_kind="detect")
    img_sel = resolve_img("detect", state)
    args.setdefault("image_path", img_sel["path"])

    out = _call_tool("OCR", args, state)

    pieces = [upd0]

    viz_path = out.get("viz")
    if viz_path:
        extra, saved = r_save_artifact(viz_path, "ocr", "detect", state)
        pieces.append(extra)
        viz_path = saved

    boxes  = out.get("boxes")
    scores = out.get("scores")
    texts  = out.get("texts")

    # === Guard: no boxes ===
    if not boxes or len(boxes) == 0:
        return r_abort_to_verify_sequence(
            state,
            node="ocr",
            error_msg="ocr detection failed: no text regions detected",
            details={"num_text_bboxes": 0},
        )

    seq_id = state.get("seq_id")
    det_ids = [f"t-seq_{seq_id:03d}-ocr-box_{i:02d}" for i, _ in enumerate(boxes or [])]

    pieces += [
        r_set_img_slot("detect", args["image_path"], coord_id="detect", state=state),
        r_update_detect("ocr", boxes, scores, viz_path, state, texts=texts, det_ids=det_ids),
        _bump_next_or(state, "tool_sequence_plan"),
    ]
    return r_pack_state(state, *pieces)
