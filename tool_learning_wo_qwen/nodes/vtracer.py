# src/langgraph/nodes/vtracer.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
from ..state import GraphState
from ..reducers import r_pack_state, r_abort_to_verify_sequence
from ..utils import to_json_safe
from ._common import _pop_current_action, _bump_next_or, _call_tool

def node(state: GraphState) -> Dict[str, Any]:

    act, upd0 = _pop_current_action(state)
    
    upd: Dict[str, Any] = {}
    parse_doc = state.get("parse") or {}
    elements: List[Dict[str, Any]] = list(parse_doc.get("elements") or [])
    # args = (state.get("pending_actions") or [{}])[0].get("args", {})

    # ids = _select_ids(args, state, parse_doc)
    ids = list(((state.get("_runtime") or {}).get("extract") or {}).get("ids_object") or [])
    
    print("=================================")
    print(f"[ VTRACER ] selected object ids : {ids}")
    print("=================================")

    if not ids:
        return r_abort_to_verify_sequence(
            state,
            node="vtracer",
            error_msg="vtracer img-to-svg vectorization Failed: no id tagged object elements extracted in preceeding nodes",
            details={"ids_object": ids},
        )

    out_dir = Path(state["out_dir"])
    vector_dir = out_dir / "vector"
    vector_dir.mkdir(parents=True, exist_ok=True)

    changed = 0


    '''
        11.08 : svg 변환할 객체를 이번 sequence 추출 객체 에만 한정짓지 않음.
                변환되지 않았던 이전 객체들까지 모두 변환되도록 함.
                이후에 planning 능력을 더욱 엄밀히 평가하려면, 현재 sequence 추출 객체 조건 다시 활성화
    '''
    for el in elements:
        if el.get("id") not in ids:
            ## 현재 tool sequence _runtime 에서 추출한 element 에 대해서만 svg 변환.
            ## 중복 변환 피하고자, 이전 tool sequence 에서 추출한 elements 는 모두 svg 변환 되었다고 가정
            continue
        if el.get("type") != "object":
            # object type element 에 대해서만 svg 변환
            continue
        if el.get("vector_render"):
            # 중복 변환 피하고자, svg 변환되지 않은 element 에 대해서만 svg
            continue
        src_rgba = el.get("extracted_image_uri")
        if not src_rgba:
            # src_rgba 있어야만 img-to-svg 변환 가능
            continue
        svg_out = vector_dir / f"{el['id']}.svg"
        out = _call_tool("vtracer", {
            "src_rgba_path": src_rgba,
            "out_svg_path": str(svg_out)
        }, state)
        svg_uri = (out or {}).get("svg_uri")
        if not svg_uri:
            continue
        el["vector_render"] = {
            "svg_uri": str(Path(svg_uri).resolve()),
            "vtracer_args": (out or {}).get("args") or {}
        }
        changed += 1
    
    if changed == 0:
        return r_abort_to_verify_sequence(
            state,
            node="vtracer",
            error_msg="vtracer img-to-svg vectorization Failed: none of object targets converted",
            details={"id tagged object targets": ids, "changed": changed},
        )


    parse_doc["elements"] = elements
    with open(state["parse_path"], "w", encoding="utf-8") as f:
        import json; f.write(json.dumps(to_json_safe(parse_doc), ensure_ascii=False, indent=2))

    upd.update(upd0)  # pending_actions에서 현재 액션 제거
    upd.update(_bump_next_or(state, "tool_sequence_plan"))

    return r_pack_state(state, upd)
