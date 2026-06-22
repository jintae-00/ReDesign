# baselines/tool_backends/nodes/fontstyle_1016.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
from ..state import GraphState
from ..reducers import r_pack_state, r_abort_to_verify_sequence
from ..utils import to_json_safe
from ._common import _pop_current_action, _bump_next_or, _call_tool

def _resolve_img_from_runtime(state: GraphState) -> Optional[Path]:
    """Resolve the image path using only the _runtime['img'] slots."""
    img_rt = ((state.get("_runtime") or {}).get("img") or {})
    for slot in ("seg", "detect", "front", "base"):
        slot_dict = img_rt.get(slot)
        if isinstance(slot_dict, dict):
            p = slot_dict.get("path")
            if p:
                return Path(p).resolve()
    return None

def node(state: GraphState) -> Dict[str, Any]:

    act, upd0 = _pop_current_action(state)

    upd: Dict[str, Any] = {}
    parse_doc = state.get("parse") or {}
    elements: List[Dict[str, Any]] = list(parse_doc.get("elements") or [])

    # ids = _select_ids(args, state, parse_doc)
    ids = list(((state.get("_runtime") or {}).get("extract") or {}).get("ids_text") or [])

    if not ids:
        return r_abort_to_verify_sequence(
            state,
            node="fontstyle_1016",
            error_msg="fontstyle_1016 Font Style Fitting Failed: no id tagged text elements extracted in preceeding nodes",
            details={"ids_text": ids},
        )

    # Image path: prefer runtime, otherwise parse.raw_image_uri
    img_path = _resolve_img_from_runtime(state)

    changed = 0

    '''
        11.08 : Font style fitting is no longer restricted to text extracted in the current sequence.
                Previously un-fitted earlier text is also fitted now.
                To later evaluate planning ability more strictly, re-enable the
                "current-sequence extracted text only" condition.
    '''

    for el in elements:
        # if el.get("id") not in ids:
        ##    Font style fitting only for elements extracted in the current tool sequence _runtime.
        ##    To avoid duplicate fitting, assume elements extracted in earlier tool sequences were all already font-style fitted.
        #     continue
        if el.get("type") != "text":
            # Font style fitting only for text-type elements
            continue
        fr = el.get("font_render") or {}
        fam = fr.get("font_family")
        if not fam:
            # Font family is required first
            continue

        out = _call_tool("fontstyle_1016", {
            "image_path": str(img_path) if img_path else None,
            "mask_path": el.get("mask_uri"),
            "extracted_image_path": el.get("extracted_image_uri"),
            "content": el.get("content", ""),
            "font_family": fam
        }, state)
        fr2 = (out or {}).get("font_render")
        if not fr2:
            continue
        # Merge
        merged = {**fr, **fr2}
        el["font_render"] = merged
        changed += 1

    if changed == 0:
        return r_abort_to_verify_sequence(
            state,
            node="fontstyle_1016",
            error_msg="fontstyle_1016 Font Style Fitting Failed: none of text targets' font style fitted",
            details={"id tagged text targets": ids, "changed": changed},
        )

    parse_doc["elements"] = elements
    with open(state["parse_path"], "w", encoding="utf-8") as f:
        import json; f.write(json.dumps(to_json_safe(parse_doc), ensure_ascii=False, indent=2))

    upd.update(upd0)  # remove the current action from pending_actions
    upd.update(_bump_next_or(state, "tool_sequence_plan"))

    return r_pack_state(state, upd)
