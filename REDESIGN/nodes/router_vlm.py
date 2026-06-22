# REDESIGN/nodes/router_vlm.py
"""
Router VLM Node - Central Brain for URLD Pipeline

Analyzes each layer and decides the decomposition action.

UPDATED: Now saves router output as JSON file in each layer_dir.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import gc
import json
import io
import base64
import os
from PIL import Image 
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import GraphState
from ..reducers import (
    r_set_router_outputs,
    r_llm_inc,
    llm_can_call,
    r_pack_state,
    r_set_layer_error,
)
from ..utils import (
    image_to_b64,
    extract_json,
    get_current_image_path,
    get_root_image_path,
)
from ..prompt_builders import build_router_system_prompt, build_router_user_prompt





# Valid action types and their tool sequences
VALID_ACTIONS = {
    "Fork_Qwen",
    "Split_DetSeg",
    "Split_Text",
    "Split_CCA",
    "Finalize_Obj",
}


def _validate_router_output(output: Dict[str, Any]) -> bool:
    """Validate router output structure"""
    required = ["image_context", "action_reasoning", "action_type", "planned_tool_sequence"]
    for key in required:
        if key not in output:
            return False
    
    if output["action_type"] not in VALID_ACTIONS:
        return False
    
    return True


def _fix_tool_sequence(action_type: str, sequence: list, params: dict) -> list:
    # Split/Fork actions always need stack_manager at the end (but only once!)
    if action_type in ["Split_DetSeg", "Split_Text", "Fork_Qwen", "Split_CCA"]:
        # Remove any existing stack_manager entries first
        sequence = [s for s in sequence if s != "stack_manager"]
        # Then add exactly one at the end
        sequence.append("stack_manager")
    
    elif action_type == "Finalize_Obj":
        # Add vtracer if not photo
        if not params.get("is_photo", False):
            if "vtracer" not in sequence:
                sequence.insert(0, "vtracer")
    
    return sequence


def _save_router_output_to_layer_dir(
    layer_id: str,
    router_output: Dict[str, Any],
    state: GraphState
) -> None:
    """
    Save router VLM output as JSON file in the layer directory.
    
    Args:
        layer_id: Current layer ID
        router_output: The full router output dict
        state: Current graph state
    """
    try:
        episode_dir = state.get("episode_dir", ".")
        layer_dir = Path(episode_dir) / "layers" / layer_id
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Save router output
        router_json_path = layer_dir / "router_vlm_output.json"
        
        # Add metadata
        output_with_meta = {
            "layer_id": layer_id,
            "depth": state.get("history_tree", {}).get(layer_id, {}).get("depth", 0),
            "parent_id": state.get("history_tree", {}).get(layer_id, {}).get("parent_id"),
            **router_output,
        }
        
        with open(router_json_path, "w", encoding="utf-8") as f:
            json.dump(output_with_meta, f, ensure_ascii=False, indent=2)
        
        print(f"[Router VLM] Saved router output to {router_json_path}")
        
    except Exception as e:
        print(f"[Router VLM] Warning: Failed to save router output: {e}")


def node(state: GraphState) -> Dict[str, Any]:
    """
    Router VLM node - decides action for current layer.
    
    UPDATED: Now saves router output as JSON in each layer_dir.
    
    Reads from:
        - state.current_layer_id
        - state.history_tree[layer_id].image_path
        - state.root_image_path
        - Ancestry chain for context
    
    Updates:
        - history_tree[layer_id]: image_context, action_reasoning, action_type,
          planned_tool_sequence, node_queue, params
    """
    layer_id = state.get("current_layer_id")
    if not layer_id:
        return {"error": "No current layer ID"}
    
    tree = state.get("history_tree", {})
    if layer_id not in tree:
        return {"error": f"Layer {layer_id} not in history_tree"}
    
    # Check LLM budget
    if not llm_can_call(state):
        return r_pack_state(
            state,
            r_set_layer_error(layer_id, "LLM budget exhausted", {}, state)
        )
    
    # Get image paths
    current_image_path = get_current_image_path(layer_id, state)
    root_image_path = get_root_image_path(state)

    root_b64 = image_to_b64(root_image_path)
    current_b64 = image_to_b64(current_image_path)

    # Build prompts
    system_prompt = build_router_system_prompt()
    user_prompt = build_router_user_prompt(layer_id, state)
    

    vlm_model = os.environ.get("VLM_MODEL", "gemini-3-flash-preview")
    llm = ChatOpenAI(
        model=vlm_model,
        base_url="https://gateway.letsur.ai/v1",
        temperature=0,
        model_kwargs={"top_p": 1},
        max_retries=3,       # 재시도 3회
        request_timeout=90,  # 타임아웃 90초
    )
    print(f"[Router VLM] Using model: {vlm_model}")

    # Prepare messages with images
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": user_prompt},
            {"type": "text", "text": "\nRoot Image: "},
            {
                "type": "image_url",
                # [수정] 리사이징된 base64 문자열 사용
                "image_url": {"url": f"data:image/png;base64,{root_b64}"},
            },
            {"type": "text", "text": "\Current Image: "},
            {
                "type": "image_url", 
                # [수정] 리사이징된 base64 문자열 사용
                "image_url": {"url": f"data:image/png;base64,{current_b64}"},
            },
        ]),
    ]
    
    response = llm.invoke(messages)
    output = extract_json(response.content)

    print(f"\n{'>'*20} ROUTER VLM OUTPUT (Layer: {layer_id}) {'<'*20}")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"{'='*60}\n")

    
    # Validate output
    if not output or not _validate_router_output(output):
        # Use fallback - finalize as atomic object
        output = {
            "image_context": "Unable to analyze",
            "action_reasoning": "LLM output validation failed, using fallback finalize",
            "action_type": "Finalize_Obj",
            "planned_tool_sequence": ["finalize_obj"],
            "params": {"is_photo": True},
        }
    
    # Extract and validate
    image_context = output.get("image_context", "")
    action_reasoning = output.get("action_reasoning", "")
    action_type = output.get("action_type", "")
    planned_sequence = output.get("planned_tool_sequence", [])
    params = output.get("params", {})
    
    # Fix tool sequence if needed
    planned_sequence = _fix_tool_sequence(action_type, planned_sequence, params)
    
    # Create full router output for saving
    full_router_output = {
        "image_context": image_context,
        "action_reasoning": action_reasoning,
        "action_type": action_type,
        "planned_tool_sequence": planned_sequence,
        "params": params,
        "raw_llm_output": output,
    }
    
    # Save router output to layer_dir (4번 수정)
    _save_router_output_to_layer_dir(layer_id, full_router_output, state)
    
    # Update state
    updates = r_set_router_outputs(
        layer_id=layer_id,
        image_context=image_context,
        action_reasoning=action_reasoning,
        action_type=action_type,
        planned_tool_sequence=planned_sequence,
        params=params,
        state=state,
    )
    
    # Increment LLM counter
    llm_update = r_llm_inc(state)
    
    # Cleanup
    del response, output, llm, root_b64, current_b64 
    gc.collect()
    
    return r_pack_state(state, updates, llm_update)


def get_next_tool(state: GraphState) -> Optional[str]:
    """Helper to get next tool from current layer's node_queue"""
    layer_id = state.get("current_layer_id")
    if not layer_id:
        return None
    
    tree = state.get("history_tree", {})
    if layer_id not in tree:
        return None
    
    queue = tree[layer_id].get("node_queue") or []
    return queue[0] if queue else None