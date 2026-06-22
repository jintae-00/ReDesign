# REDESIGN/nodes/qwen_layered.py
"""
Qwen Image Layered Node - Multi-layer generation using Qwen model.

[수정 22] Fixed KeyError in log message
- Fixed logging error where result['num_layers'] was accessed but not present
- Reverted to using len(result['layer_images'])
"""
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import gc

from ..state import GraphState
from ..reducers import (
    r_pack_state,
    r_set_qwen_layered_output,
    r_dequeue_node,
    r_set_layer_error,
)
from ..utils import get_current_image_path, get_tools_output_dir


def node(state: GraphState) -> Dict[str, Any]:
    """
    Qwen Layered node - generates multi-layer decomposition.
    """
    layer_id = state.get("current_layer_id")
    if not layer_id:
        return {}
    
    tree = state.get("history_tree", {})
    node_data = tree.get(layer_id)
    if not node_data:
        return {}
    
    # Dequeue this node
    _, dequeue_update = r_dequeue_node(layer_id, state)
    
    # Get parameters
    num_layers = node_data.get("param_qwen_len") or 4
    num_layers = max(2, min(6, num_layers))
    
    print(f"[Qwen Node] Submitting to QwenPool for {layer_id} (layers={num_layers})")
    
    # Get image path
    image_path = get_current_image_path(layer_id, state)
    
    if not image_path or not Path(image_path).exists():
        error_msg = f"Image not found: {image_path}"
        error_update = r_set_layer_error(layer_id, error_msg, {"path": image_path}, state)
        return r_pack_state(state, dequeue_update, error_update)
    
    # Prepare output directory
    try:
        tools_dir = get_tools_output_dir(state, layer_id)
        qwen_output_dir = tools_dir / "qwen_layers"
        qwen_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        error_update = r_set_layer_error(layer_id, f"Dir creation failed: {e}", {}, state)
        return r_pack_state(state, dequeue_update, error_update)
    
    # Run Qwen via QwenPool
    try:
        from ..tools.qwen_layered_tool import run_qwen_layered
        result = run_qwen_layered(
            image_path=image_path,
            output_dir=str(qwen_output_dir),
            num_layers=num_layers,
        )
            
    except Exception as e:
        error_update = r_set_layer_error(
            layer_id,
            f"Qwen inference failed: {str(e)}",
            {"image_path": image_path},
            state
        )
        return r_pack_state(state, dequeue_update, error_update)
    
    # Validate result
    if not result.get("layer_images"):
        error_update = r_set_layer_error(layer_id, "Qwen returned no layers", {}, state)
        return r_pack_state(state, dequeue_update, error_update)
    
    # Save output
    output_update = r_set_qwen_layered_output(state, layer_id, result)
    
    # [FIX] Use len(layer_images) instead of result['num_layers'] to avoid KeyError
    num_generated = len(result.get('layer_images', []))
    print(f"[Qwen Node] Generated {num_generated} layers for {layer_id}")
    
    gc.collect()
    
    return r_pack_state(state, dequeue_update, output_update)