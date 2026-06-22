# ReDesign/nodes/detect_ocr.py
"""
OCR Detection Node - Text detection using PaddleOCR

Updated for URLD recursive structure.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import gc

from ..state import GraphState
from ..reducers import (
    r_save_tool_output,
    r_save_artifact,
    r_dequeue_node,
    r_set_layer_error,
    r_pack_state,
)
from ..utils import get_current_image_path, boxes_to_aabbs


def node(state: GraphState) -> Dict[str, Any]:
    """
    OCR node - detects text regions.
    
    Reads from:
        - state.current_layer_id
        - history_tree[layer_id].image_path (or refine output)
    
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
    
    # Import tool
    try:
        from ..tools.ocr_tool import run_ocr
    except ImportError as e:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"OCR tool import failed: {e}", {}, state)
        )
    
    # Get output directory for viz
    episode_dir = state.get("episode_dir", ".")
    vis_dir = Path(episode_dir) / "layers" / layer_id / "tools_output"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Run OCR
    try:
        out = run_ocr(image_path, vis_dir=vis_dir, step=0)
    except Exception as e:
        err_str = str(e)
        # Fatal OCR errors (std::exception, CUDA illegal memory access) ->
        # block saving this episode's parse.json so it becomes a re-run target
        fatal_patterns = ["std::exception", "illegal memory access", "cudaErrorIllegalAddress"]
        if any(p in err_str for p in fatal_patterns):
            # StateManager accumulates the total, so pass only delta=1
            error_update = {"_ocr_fatal_error_count": 1}
            return r_pack_state(
                state,
                dequeue_update,
                r_set_layer_error(layer_id, f"OCR failed: {e}", {}, state),
                error_update,
            )
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, f"OCR failed: {e}", {}, state)
        )
    
    boxes = out.get("boxes", [])
    texts = out.get("texts", [])
    scores = out.get("scores", [])
    
    # Check if any text detected
    if not boxes:
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "No text detected by OCR", {}, state)
        )
    
    # Generate detection IDs
    det_ids = [f"t_{layer_id}_{i:02d}" for i in range(len(boxes))]
    
    # Convert rotated boxes to AABBs for downstream tools
    aabbs = boxes_to_aabbs(boxes)
    
    output = {
        "tool_name": "detect_ocr",
        "boxes": boxes,
        "boxes_aabb": aabbs,
        "texts": texts,
        "scores": scores,
        "det_ids": det_ids,
        "viz": out.get("viz"),
    }
    
    print(f"[OCR] Detected {len(boxes)} text regions in {layer_id}")
    
    # Save tool output
    tool_update = r_save_tool_output(
        layer_id=layer_id,
        tool_category="detect",
        tool_name="detect_ocr",
        output=output,
        state=state,
    )
    
    # Cleanup (no torch.cuda calls — OCR runs in CPU-only subprocess)
    gc.collect()
    
    return r_pack_state(state, dequeue_update, tool_update)