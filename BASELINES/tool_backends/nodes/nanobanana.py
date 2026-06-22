# BASELINES/tool_backends/nodes/nanobanana.py
from __future__ import annotations
from typing import Dict, Any
from PIL import Image as _PIL
from ..state import GraphState
from ..reducers import (
    r_save_artifact, r_update_refine,
    r_set_img_slot, r_pack_state,
    r_abort_to_verify_sequence
)
from ..utils import resolve_args, resolve_img
from ._common import _pop_current_action, _bump_next_or, _call_tool
import gc


def node(state: GraphState) -> Dict[str, Any]:

    act, upd0 = _pop_current_action(state)

    raw_args = act.get("args") or {}
    args = resolve_args(raw_args, last=None, state=state, labels=None, image_kind="refine")

    img_sel = resolve_img("refine", state) or {}
    args.setdefault("image_path", img_sel.get("path"))

    instruction = args.get("instruction")
    if not instruction:
        return r_abort_to_verify_sequence(
            state,
            node="nanobanana",
            error_msg = "nanobanana Image Refinement Failed: " \
                        "Reason : missing 'instruction'. " \
                        "Text Instruction for nanobanana had not been generated during tool sequence planning." \
                        "Text Instruction for nanobanana Must be generated during tool sequence planning !",
            details={"instruction": instruction},
        )



    try:
        out_img = _call_tool("nanobanana", args, state)
    except Exception as e:
        return r_abort_to_verify_sequence(
            state,
            node="nanobanana",
            error_msg="nanobanana Image Refinement Failed: LLM/tool invocation error",
            details={
                "instruction": instruction,
                "exception_type": type(e).__name__,
                "raw_error": str(e),
            },
        )


    if not out_img:
        return r_abort_to_verify_sequence(
            state,
            node="nanobanana",
            error_msg = "nanobanana Image Refinement Failed: " \
                        "Reason : nanobanana tool returned no image." \
                        "Although Text Instruction for nanobanana had been generated during tool sequence planning," \
                        "nanobanana returned no image !",
            details={"instruction": instruction, "out_img": out_img},
        )

    pieces: list[Dict[str, Any]] = [upd0]

    extra, saved = r_save_artifact(out_img, "nanobanana", "refine", state)
    pieces.append(extra)
    pieces.append(r_update_refine("nanobanana", saved, state))

    try:
        W, H = _PIL.open(saved).size
    except Exception:
        W = H = None
    pieces.append(r_set_img_slot("base", saved, coord_id="base", W=W, H=H, state=state))
    pieces.append(_bump_next_or(state, "tool_sequence_plan"))

    return r_pack_state(state, *pieces)
