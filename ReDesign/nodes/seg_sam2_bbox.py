# ReDesign/nodes/seg_sam2_bbox.py
"""
SAM2 BBox Segmentation Node - Object segmentation using SAM2

Updated for URLD recursive structure.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import gc
import torch
import cv2

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


SEG_MASK_MIN_PX = 8  # Minimum mask area


def node(state: GraphState) -> Dict[str, Any]:
    """
    SAM2 BBox node - segments objects using bounding boxes.
    
    Reads from:
        - state.current_layer_id
        - history_tree detect output (GDINO boxes)
    
    Updates:
        - history_tree[layer_id].tool_outputs.segment (appends)
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
    
    # Get detection boxes (prefer GDINO)
    detect_output = find_latest_tool_output(layer_id, "detect", "detect_gdino", state)
    if not detect_output:
        # Fall back to any detection
        detect_output = find_latest_tool_output(layer_id, "detect", None, state)
    
    if not detect_output:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No detection output found", {}, state)
        )
    
    boxes = detect_output.get("boxes", [])
    det_ids = detect_output.get("det_ids", [])
    
    if not boxes:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No boxes from detection", {}, state)
        )
    
    # Import tool
    try:
        from ..tools.sam2_tool import run_sam2_union
    except ImportError as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"SAM2 tool import failed: {e}", {}, state)
        )
    
    # Run SAM2
    try:
        out = run_sam2_union(
            image_path=image_path,
            boxes=boxes,
            det_ids=det_ids,
        )
    except Exception as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"SAM2 failed: {e}", {}, state)
        )
    
    mask_union = out.get("mask_union")
    masks_by_id = out.get("masks_by_id", {})
    
    # Validate mask area
    if mask_union:
        try:
            m = cv2.imread(mask_union, cv2.IMREAD_GRAYSCALE)
            area = int((m > 0).sum()) if m is not None else 0
        except Exception:
            area = 0
        
        if area < SEG_MASK_MIN_PX:
            return r_pack_state(
                state,
                dequeue_update,
                r_set_layer_error(layer_id, f"Mask too small: {area} px", {}, state)
            )
    
    output = {
        "tool_name": "seg_sam2_bbox",
        "mask_union": mask_union,
        "masks_by_id": masks_by_id,
    }
    
    print(f"[SAM2 BBox] Created {len(masks_by_id)} masks for {layer_id}")
    
    # Save tool output
    tool_update = r_save_tool_output(
        layer_id=layer_id,
        tool_category="segment",
        tool_name="seg_sam2_bbox",
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