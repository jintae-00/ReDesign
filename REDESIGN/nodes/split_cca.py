# REDESIGN/nodes/split_cca.py
"""
Split CCA Node - Connected Component Analysis for spatially separated objects

Uses scipy.ndimage for connected component analysis on the alpha channel.
Fast pixel-based splitting - no deep learning needed.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path

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
    Split CCA node - splits layer using connected component analysis.
    
    Uses scipy.ndimage.label for actual CCA computation on alpha channel.
    
    Reads from:
        - state.current_layer_id
        - history_tree[layer_id].image_path
    
    Updates:
        - history_tree[layer_id].tool_outputs.split_cca
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
    
    # Get image path
    image_path = get_current_image_path(layer_id, state)
    if not image_path or not Path(image_path).exists():
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "Image not found", {"path": image_path}, state)
        )
    
    # Import and run CCA tool
    try:
        from ..tools.cca_tool import run_split_cca
    except ImportError as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"CCA tool import failed: {e}", {}, state)
        )
    
    # Run CCA with proper parameters
    try:
        print(f"[Split CCA] Running CCA on {layer_id}...")
        output = run_split_cca(
            image_path=image_path,
            min_area=100,        # Minimum pixel area for component
            alpha_threshold=10,  # Alpha threshold for foreground
            connectivity=8,      # 8-connectivity for better grouping
        )
    except Exception as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"CCA failed: {e}", {}, state)
        )
    
    # Validate output
    components = output.get("components", [])
    num_components = output.get("num_components", 0)
    
    if not components:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(
                layer_id, 
                "No components found by CCA", 
                {"num_raw_components": num_components}, 
                state
            )
        )
    
    # Verify component files exist
    valid_components = []
    for comp in components:
        layer_path = comp.get("layer_path")
        if layer_path and Path(layer_path).exists():
            valid_components.append(comp)
        else:
            print(f"[Split CCA] Warning: Component layer not found: {layer_path}")
    
    if not valid_components:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No valid component layers created", {}, state)
        )
    
    # Update output with only valid components
    output["components"] = valid_components
    output["num_components"] = len(valid_components)
    
    print(f"[Split CCA] Found {len(valid_components)} components in {layer_id}")
    for i, comp in enumerate(valid_components):
        print(f"  Component {i}: area={comp.get('area', 0)}, bbox={comp.get('bbox', [])}")
    
    # Save tool output
    tool_update = r_save_tool_output(
        layer_id=layer_id,
        tool_category="split_cca",
        tool_name="split_cca",
        output=output,
        state=state,
    )
    
    return r_pack_state(state, dequeue_update, tool_update)