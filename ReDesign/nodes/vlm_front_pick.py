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

# The _llm global was removed; the LLM is now created inside the function.

def _resize_image_for_vlm(image_path: str, max_size: int = 1024) -> str:
    """Downscale the image and return its bytes (reduces network load before base64 encoding)."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) > max_size:
                ratio = max_size / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            # Convert to bytes in memory
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        # On failure, just read the original file
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

    # Optimize image size
    try:
        import base64
        img_bytes = _resize_image_for_vlm(image_path)
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        return r_pack_state(
            state, dequeue_update,
            r_set_layer_error(layer_id, f"Image processing failed: {e}", {}, state)
        )

    # Create the LLM object inside the function, with timeout/retry settings.
    # max_retries: LangChain automatically retries on 429 or 5xx errors.
    # request_timeout: set generously to account for large image uploads (60+ seconds).
    vlm_model = os.environ.get("VLM_MODEL", "gemini-3-flash-preview")
    llm = ChatOpenAI(
        model=vlm_model,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=0,
        model_kwargs={"top_p": 1},
        max_retries=3,       # Retry up to 3 times on API errors
        request_timeout=90,  # Request timeout: 90 seconds
    )

    try:
        resp = llm.invoke([HumanMessage(content=[
            {"type": "text", "text": VLM_FRONT_ELEMS_PICK},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
        ])])
    except Exception as e:
        # If it still fails after 3 retries, handle as an error
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