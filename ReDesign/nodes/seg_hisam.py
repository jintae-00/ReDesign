# ReDesign/nodes/seg_hisam.py
"""
HiSAM Segmentation Node - Text segmentation using hierarchical SAM

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
    boxes_to_aabbs,
)


SEG_MASK_MIN_PX = 8  # Minimum mask area


def node(state: GraphState) -> Dict[str, Any]:
    """
    HiSAM node - segments text regions.
    
    Reads from:
        - state.current_layer_id
        - history_tree detect output (OCR boxes)
    
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
    
    # Get detection boxes
    detect_output = find_latest_tool_output(layer_id, "detect", "detect_ocr", state)
    if not detect_output:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No OCR detection output found", {}, state)
        )
    
    boxes = detect_output.get("boxes", [])
    det_ids = detect_output.get("det_ids", [])
    
    if not boxes:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No boxes from OCR", {}, state)
        )
    
    # Convert to AABBs
    aabbs = boxes_to_aabbs(boxes)
    
    # Import tool
    try:
        from ..tools.hisam_tool import run_hisam_union
    except ImportError as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"HiSAM tool import failed: {e}", {}, state)
        )
    
    # Get output directory
    episode_dir = state.get("episode_dir", ".")
    vis_dir = Path(episode_dir) / "layers" / layer_id / "tools_output"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Run HiSAM
    try:
        out = run_hisam_union(
            image_path=image_path,
            boxes=aabbs,
            det_ids=det_ids,
            vis_dir=vis_dir,
            step=0,
        )
    except Exception as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"HiSAM failed: {e}", {}, state)
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
        "tool_name": "seg_hisam",
        "mask_union": mask_union,
        "masks_by_id": masks_by_id,
    }
    
    print(f"[HiSAM] Created {len(masks_by_id)} masks for {layer_id}")
    
    # Save tool output
    tool_update = r_save_tool_output(
        layer_id=layer_id,
        tool_category="segment",
        tool_name="seg_hisam",
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