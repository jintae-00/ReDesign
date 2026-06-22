# REDESIGN/nodes/detect_gdino.py
"""
GDINO Detection Node - Object detection using Grounding DINO

Updated for URLD recursive structure.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import gc
import torch

from ..state import GraphState
from ..reducers import (
    r_save_tool_output,
    r_dequeue_node,
    r_set_layer_error,
    r_pack_state,
)
from ..utils import get_current_image_path, get_vlm_labels


def node(state: GraphState) -> Dict[str, Any]:
    """
    GDINO node - detects objects using labels from VLM.
    
    Reads from:
        - state.current_layer_id
        - history_tree[layer_id].image_path (or refine output)
        - VLM front pick labels (from ancestor or current)
    
    Updates:
        - history_tree[layer_id].tool_outputs.detect (appends)
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
    
    # Get labels from VLM
    labels = get_vlm_labels(layer_id, state)
    if not labels:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No VLM labels found for GDINO", {}, state)
        )
    
    # Import tool
    try:
        from ..tools.dino_tool import run_dino_batch_all
    except ImportError as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"GDINO tool import failed: {e}", {}, state)
        )
    
    # Get output directory for viz
    episode_dir = state.get("episode_dir", ".")
    vis_dir = Path(episode_dir) / "layers" / layer_id / "tools_output"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Run GDINO
    try:
        out = run_dino_batch_all(
            image_path=image_path,
            labels=labels,
            vis_dir=vis_dir,
            step=0,
        )
    except Exception as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"GDINO failed: {e}", {}, state)
        )
    
    boxes = out.get("boxes", [])
    confs = out.get("confs", [])
    out_labels = out.get("labels", [])
    
    # Check if any objects detected
    if not boxes:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No objects detected by GDINO", {"labels": labels}, state)
        )
    
    # Generate detection IDs
    det_ids = [f"o_{layer_id}_{i:02d}" for i in range(len(boxes))]
    
    output = {
        "tool_name": "detect_gdino",
        "boxes": boxes,
        "confs": confs,
        "labels": out_labels,
        "det_ids": det_ids,
        "input_labels": labels,
        "viz": out.get("viz"),
    }
    
    print(f"[GDINO] Detected {len(boxes)} objects in {layer_id}: {out_labels}")
    
    # Save tool output
    tool_update = r_save_tool_output(
        layer_id=layer_id,
        tool_category="detect",
        tool_name="detect_gdino",
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