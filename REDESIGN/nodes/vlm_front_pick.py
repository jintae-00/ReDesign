"""
VLM Front Pick Node - Label front-most elements for GDINO detection
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import json
import gc
import time
from PIL import Image
import io

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..state import GraphState
from ..reducers import (
    r_save_tool_output,
    r_dequeue_node,
    r_llm_inc,
    llm_can_call,
    r_set_layer_error,
    r_pack_state,
)
from ..utils import image_to_b64, extract_json, get_current_image_path
from ..prompts import VLM_FRONT_ELEMS_PICK

# [수정 1] 전역 변수 _llm 제거. 함수 내부에서 생성.

def _resize_image_for_vlm(image_path: str, max_size: int = 1024) -> str:
    """[수정 3] 이미지 크기를 줄여서 Base64로 변환 (네트워크 부하 감소)"""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) > max_size:
                ratio = max_size / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # 메모리 내에서 바이트로 변환
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        # 실패하면 그냥 원본 경로 읽기
        with open(image_path, "rb") as f:
            return f.read()

def node(state: GraphState) -> Dict[str, Any]:
    layer_id = state.get("current_layer_id")
    if not layer_id:
        return {"error": "No current layer ID"}
    
    tree = state.get("history_tree", {})
    if layer_id not in tree:
        return {"error": f"Layer {layer_id} not in history_tree"}
    
    # Dequeue this node
    _, dequeue_update = r_dequeue_node(layer_id, state)
    
    if not llm_can_call(state):
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "LLM budget exhausted", {}, state)
        )
    
    image_path = get_current_image_path(layer_id, state)
    if not image_path or not Path(image_path).exists():
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "Image not found", {"path": image_path}, state)
        )

    # [수정 3 적용] 이미지 용량 최적화
    try:
        import base64
        img_bytes = _resize_image_for_vlm(image_path)
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        return r_pack_state(
            state, dequeue_update,
            r_set_layer_error(layer_id, f"Image processing failed: {e}", {}, state)
        )

    # [수정 1 & 2] LLM 객체를 함수 내에서 생성 + 타임아웃/재시도 설정
    # max_retries: LangChain 내부적으로 429나 5xx 에러시 자동 재시도
    # request_timeout: 대용량 이미지 전송 고려하여 넉넉하게 설정 (60초 이상)
    vlm_model = os.environ.get("VLM_MODEL", "gemini-3-flash-preview")
    llm = ChatOpenAI(
        model=vlm_model,
        base_url="https://gateway.letsur.ai/v1",
        temperature=0,
        model_kwargs={"top_p": 1},
        max_retries=3,       # API 에러 시 3회 자동 재시도
        request_timeout=90,  # 타임아웃 90초
    )

    try:
        resp = llm.invoke([HumanMessage(content=[
            {"type": "text", "text": VLM_FRONT_ELEMS_PICK},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
        ])])
    except Exception as e:
        # 3회 재시도 후에도 실패하면 에러 처리
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"LLM call failed after retries: {e}", {}, state),
            r_llm_inc(state),
        )
    
    js = extract_json(resp.content)

    print(f"\n{'>'*20} VLM_FRONT OUTPUT (Layer: {layer_id}) {'<'*20}")
    print(json.dumps(js, indent=2, ensure_ascii=False))
    print(f"{'='*60}\n")

    labels = (js or {}).get("labels", [])
    
    print(f"[LLM CONFIG]: Model={getattr(llm, 'model', getattr(llm, 'model_name', '?'))}")
    print(f"[VLM Front Pick] Labels for {layer_id}: {labels}")
    
    # Save output
    output = {"labels": labels}
    tool_update = r_save_tool_output(
        layer_id=layer_id,
        tool_category="vlm_front_pick",
        tool_name="vlm_front_pick",
        output=output,
        state=state,
    )
    
    # Cleanup
    del resp, js
    gc.collect()
    
    return r_pack_state(state, dequeue_update, tool_update, r_llm_inc(state))