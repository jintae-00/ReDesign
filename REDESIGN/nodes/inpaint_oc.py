# REDESIGN/nodes/inpaint_oc.py
"""
ObjectClear Inpainting Node - Object-specific inpainting

Updated for URLD recursive structure.
UPDATED: Now saves output to tools_output directory.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import gc
import torch
import shutil
import json

from ..state import GraphState
from ..reducers import (
    r_save_tool_output,
    r_dequeue_node,
    r_set_layer_error,
    r_pack_state,
)
from ..utils import (
    get_current_image_path,
    find_latest_tool_output,
)


def node(state: GraphState) -> Dict[str, Any]:
    """
    ObjectClear inpainting node.
    
    UPDATED: Now saves output to tools_output directory.
    
    Reads from:
        - state.current_layer_id
        - history_tree segment output (mask_union)
    
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
    
    # Get image
    image_path = get_current_image_path(layer_id, state)
    if not image_path or not Path(image_path).exists():
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "Image not found", {"path": image_path}, state)
        )
    
    # Get mask
    seg_output = find_latest_tool_output(layer_id, "segment", None, state)
    if not seg_output:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No segmentation output found", {}, state)
        )
    
    mask_path = seg_output.get("mask_union")
    if not mask_path or not Path(mask_path).exists():
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No mask found", {}, state)
        )
    
    # Import tool
    try:
        from ..tools.objectclear_tool import run_objectclear
    except ImportError as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"ObjectClear tool import failed: {e}", {}, state)
        )
    
    # Prepare tools_output directory
    episode_dir = state.get("episode_dir", ".")
    tools_output_dir = Path(episode_dir) / "layers" / layer_id / "tools_output"
    tools_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run ObjectClear
    try:
        out_path = run_objectclear(image_path, mask_path)
    except Exception as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"ObjectClear failed: {e}", {}, state)
        )
    
    if not out_path or not Path(out_path).exists():
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "ObjectClear output not found", {}, state)
        )
    
    # Copy output to tools_output directory
    final_output_path = tools_output_dir / "objectclear_inpaint.png"
    shutil.copy(out_path, final_output_path)
    
    # Also copy the mask used
    mask_copy_path = tools_output_dir / "objectclear_mask_used.png"
    shutil.copy(mask_path, mask_copy_path)
    
    output = {
        "tool_name": "inpaint_oc",
        "image_path": str(final_output_path),
        "input_image": image_path,
        "mask_path": str(mask_copy_path),
        "original_output_path": out_path,
    }
    
    # Save output metadata as JSON
    output_json_path = tools_output_dir / "objectclear_output.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"[ObjectClear] Inpainted {layer_id} -> {final_output_path}")
    
    # Save tool output
    tool_update = r_save_tool_output(
        layer_id=layer_id,
        tool_category="refine",
        tool_name="inpaint_oc",
        output=output,
        state=state,
    )
    
    # Cleanup
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()
    
    return r_pack_state(state, dequeue_update, tool_update)