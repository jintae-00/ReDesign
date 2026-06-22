# ReDesign/nodes/nanobanana.py
"""
Nanobanana Node - Image refinement using Gemini

Used for pre/post processing cleanup, blur removal, background extraction, etc.

UPDATED: Now saves output to tools_output directory.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import gc
import shutil
import json

from ..state import GraphState
from ..reducers import (
    r_save_tool_output,
    r_dequeue_node,
    r_set_layer_error,
    r_pack_state,
)
from ..utils import get_current_image_path


def node(state: GraphState) -> Dict[str, Any]:
    """
    Nanobanana refinement node.
    
    UPDATED: Now saves output to tools_output directory.
    
    Reads from:
        - state.current_layer_id
        - history_tree[layer_id].param_nanobanana_instruction
    
    Updates:
        - history_tree[layer_id].tool_outputs.refine (appends)
        - history_tree[layer_id].node_queue (dequeue)
    """
    layer_id = state.get("current_layer_id")
    if not layer_id:
        return {"error": "No current layer ID"}
    
    tree = state.get("history_tree", {})
    if layer_id not in tree:
        return {"error": f"Layer {layer_id} not in history_tree"}
    
    # Dequeue this node
    _, dequeue_update = r_dequeue_node(layer_id, state)
    
    node_data = tree[layer_id]
    
    # Get instruction
    instruction = node_data.get("param_nanobanana_instruction")
    
    print(f"[Nanobanana] Layer: {layer_id}")
    print(f"[Nanobanana] param_nanobanana_instruction: {instruction}")

    
    if not instruction:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No nanobanana instruction provided", {}, state)
        )
    
    # Get image
    image_path = get_current_image_path(layer_id, state)
    if not image_path or not Path(image_path).exists():
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "Image not found", {"path": image_path}, state)
        )
    
    # Import tool
    try:
        from ..tools.nanobanana_tool import run_nanobanana
    except ImportError as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"Nanobanana tool import failed: {e}", {}, state)
        )
    
    # Prepare tools_output directory
    episode_dir = state.get("episode_dir", ".")
    tools_output_dir = Path(episode_dir) / "layers" / layer_id / "tools_output"
    tools_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run nanobanana
    try:
        out_path = run_nanobanana(image_path, instruction)
    except Exception as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"Nanobanana failed: {e}", {}, state)
        )
    
    if not out_path or not Path(out_path).exists():
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "Nanobanana output not found", {}, state)
        )
    
    # Copy output to tools_output directory
    final_output_path = tools_output_dir / "nanobanana_output.png"
    shutil.copy(out_path, final_output_path)
    
    # Also copy input image for reference
    input_copy_path = tools_output_dir / "nanobanana_input.png"
    shutil.copy(image_path, input_copy_path)
    
    output = {
        "tool_name": "nanobanana",
        "image_path": str(final_output_path),
        "input_image": str(input_copy_path),
        "instruction": instruction,
        "original_output_path": out_path,
    }
    
    # Save output metadata as JSON
    output_json_path = tools_output_dir / "nanobanana_output.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"[Nanobanana] Refined {layer_id} -> {final_output_path}")
    
    # Save tool output
    tool_update = r_save_tool_output(
        layer_id=layer_id,
        tool_category="refine",
        tool_name="nanobanana",
        output=output,
        state=state,
    )
    
    gc.collect()
    
    return r_pack_state(state, dequeue_update, tool_update)