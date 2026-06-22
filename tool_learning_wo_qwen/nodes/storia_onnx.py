# src/langgraph/nodes/storia_onnx.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..state import GraphState
from ..reducers import r_pack_state, r_abort_to_verify_sequence
from ..utils import to_json_safe
from ._common import _pop_current_action, _bump_next_or, _call_tool

def _resolve_img_from_runtime(state: GraphState) -> Optional[Path]:
    """_runtime['img'] 슬롯만 이용하여 이미지 경로를 해석."""
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

    ids = list(((state.get("_runtime") or {}).get("extract") or {}).get("ids_text") or [])

    if not ids:
         return r_abort_to_verify_sequence(
            state,
            node="storia_onnx",
            error_msg="storia_onnx Font Family Prediction Failed: no id tagged text elements extracted in preceeding nodes",
            details={"ids_text": ids},
        )

    # 이미지 경로: runtime 우선, 없으면 parse.raw_image_uri
    img_path = _resolve_img_from_runtime(state)

    '''
        11.08 : Font Family Prediction 할 Elements 이번 sequence 추출 Text 에만 한정짓지 않음.
                Font Family 추출되지 않았던 이전 텍스트까지 모두 변환되도록 함.
                이후에 planning 능력을 더욱 엄밀히 평가하려면, 현재 sequence 추출 Text 조건 다시 활성화
    '''
    changed = 0
    for el in elements:
        # if el.get("id") not in ids:
        #     ## 현재 tool sequence _runtime 에서 추출한 element 에 대해서만 Font Family Prediction
        #     ## 중복 변환 피하고자, 이전 tool sequence 에서 추출한 elements 는 모두 Font Family Prediction 되었다고 가정
        #     continue
        if el.get("type") != "text":
            # text type element 에 대해서만 font family prediction
            continue
        fr = el.get("font_render") or {}
        if fr.get("font_family"):
            # 중복 예측 피하고자, Font Fomaily 예측되지 않은 element 에 대해서만 예측
            continue  # 이미 있음
        out = _call_tool("storia_onnx", {
            "image_path": str(img_path) if img_path else None,
            "mask_path": el.get("mask_uri"),
            "extracted_image_path": el.get("extracted_image_uri")
        }, state)
        fam = (out or {}).get("font_family")
        if not fam:
            continue
        new_fr = {"font_family": fam}
        new_fr.update({k: v for k, v in fr.items() if k != "font_family"})
        el["font_render"] = new_fr
        changed += 1

    if changed == 0:
        return r_abort_to_verify_sequence(
            state,
            node="storia_onnx",
            error_msg="storia_onnx Font Family Prediction Failed: none of text targets' font family predicted",
            details={"id tagged text targets": ids, "changed": changed},
        )

    # 저장
    parse_doc["elements"] = elements
    with open(state["parse_path"], "w", encoding="utf-8") as f:
        import json; f.write(json.dumps(to_json_safe(parse_doc), ensure_ascii=False, indent=2))

    upd.update(upd0)  # pending_actions에서 현재 액션 제거
    upd.update(_bump_next_or(state, "tool_sequence_plan"))

    return r_pack_state(state, upd)
