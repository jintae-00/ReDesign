# REDESIGN/nodes/vtracer.py
"""
VTracer Node - Image to SVG vectorization

Converts non-photographic images (icons, logos, graphics) to SVG format.

UPDATED: Now saves output directly to tools_output directory.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import gc
import json
import shutil

from ..state import GraphState
from ..reducers import (
    r_save_tool_output,
    r_dequeue_node,
    r_set_layer_error,
    r_pack_state,
)
from ..utils import (
    get_current_image_path,
    get_tight_bbox_from_alpha,
    crop_to_tight_bbox,
)


def node(state: GraphState) -> Dict[str, Any]:
    """
    VTracer node - converts image to SVG.
    
    UPDATED: Now saves all outputs to tools_output directory.
    
    Reads from:
        - state.current_layer_id
        - Current layer image
    
    Updates:
        - history_tree[layer_id].tool_outputs.vtracer
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
    
    # Check if this is a photo (skip SVG conversion for photos)
    is_photo = node_data.get("param_is_photo", False)
    
    # Get episode directory and setup tools_output
    episode_dir = state.get("episode_dir", ".")
    tools_output_dir = Path(episode_dir) / "layers" / layer_id / "tools_output"
    tools_output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_photo:
        print(f"[VTracer] Skipping {layer_id} - is photographic")
        # Just mark as processed without SVG
        output = {
            "tool_name": "vtracer",
            "svg_uri": None,
            "skipped": True,
            "reason": "photographic_content",
        }
        
        # Save skip info to tools_output
        skip_json_path = tools_output_dir / "vtracer_output.json"
        with open(skip_json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        tool_update = r_save_tool_output(
            layer_id=layer_id,
            tool_category="vtracer",
            tool_name="vtracer",
            output=output,
            state=state,
        )
        return r_pack_state(state, dequeue_update, tool_update)
    
    # Get image
    image_path = get_current_image_path(layer_id, state)
    if not image_path or not Path(image_path).exists():
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "Image not found", {"path": image_path}, state)
        )
    
    # First, crop to tight bbox for better SVG conversion
    cropped_path = str(tools_output_dir / "vtracer_cropped_for_svg.png")
    bbox = get_tight_bbox_from_alpha(image_path)
    
    if not bbox:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No content found in image", {}, state)
        )
    
    # Crop image
    crop_to_tight_bbox(image_path, cropped_path)
    
    # Output SVG path (directly in tools_output)
    svg_path = str(tools_output_dir / f"vtracer_{layer_id}.svg")
    
    # Import tool
    try:
        from ..tools.vtracer_tool import run_vtracer
    except ImportError as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"VTracer tool import failed: {e}", {}, state)
        )
    
    # Run vtracer
    try:
        result = run_vtracer(cropped_path, svg_path)
    except Exception as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"VTracer failed: {e}", {}, state)
        )
    
    svg_uri = result.get("svg_uri")
    if not svg_uri or not Path(svg_uri).exists():
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "VTracer output not found", {}, state)
        )
    
    output = {
        "tool_name": "vtracer",
        "svg_uri": svg_uri,
        "cropped_path": cropped_path,
        "bbox": bbox,
        "args": result.get("args", {}),
        "skipped": False,
    }
    
    # Save output metadata as JSON
    output_json_path = tools_output_dir / "vtracer_output.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"[VTracer] Converted {layer_id} -> {svg_uri}")
    
    # Save tool output
    tool_update = r_save_tool_output(
        layer_id=layer_id,
        tool_category="vtracer",
        tool_name="vtracer",
        output=output,
        state=state,
    )
    
    gc.collect()
    
    return r_pack_state(state, dequeue_update, tool_update)