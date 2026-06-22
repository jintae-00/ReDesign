# reducers.py

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import shutil
from copy import deepcopy
from datetime import datetime
import numpy as np
import threading

from .state import GraphState, LayerNode, generate_layer_id, VerificationAttempt


# =============================================================================
# Global Tracking for Duplicate Prevention
# =============================================================================

_enqueued_layer_ids: set = set()
_enqueued_lock = threading.Lock()

def r_save_artifact(
    src_path: str,
    tool_name: str,
    artifact_type: str,
    state: GraphState,
    layer_id: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """Save artifact with duplicate handling. Returns (saved_path, update_dict)."""
    layer_id = layer_id or state.get("current_layer_id")
    if not layer_id:
        return src_path, {}
    
    episode_dir = state.get("episode_dir", ".")
    layer_dir = Path(episode_dir) / "layers" / layer_id / "tools_output"
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    src = Path(src_path)
    if not src.exists():
        return src_path, {}
    
    suffix = src.suffix
    dest_name = f"{tool_name}_{artifact_type}{suffix}"
    dest_path = layer_dir / dest_name
    
    counter = 1
    while dest_path.exists():
        dest_name = f"{tool_name}_{artifact_type}_{counter}{suffix}"
        dest_path = layer_dir / dest_name
        counter += 1
    
    shutil.copy(src_path, dest_path)
    return str(dest_path), {}

def r_set_qwen_layered_output(state: GraphState, layer_id: str, output: Dict[str, Any]) -> Dict[str, Any]:
    """Set qwen_layered output."""
    return r_save_tool_output(layer_id, "qwen_layered", "qwen_layered", output, state)

def r_release_layer_gpu(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """Release all GPUs held by a specific layer."""
    gpu_slots = [dict(slot) for slot in state.get("gpu_slots", [])]
    
    for slot in gpu_slots:
        if slot.get("layer_id") == layer_id:
            slot["available"] = True
            slot["layer_id"] = None
    
    return {"gpu_slots": gpu_slots}


def reset_enqueued_tracking():
    """Reset the enqueued tracking set for new episode."""
    global _enqueued_layer_ids
    with _enqueued_lock:
        _enqueued_layer_ids.clear()
        print("[Reducer] Reset enqueued tracking set")


def _is_already_enqueued(layer_id: str) -> bool:
    """Check if layer_id was already enqueued."""
    with _enqueued_lock:
        return layer_id in _enqueued_layer_ids


def _mark_as_enqueued(layer_id: str) -> bool:
    """Mark layer_id as enqueued. Returns True if newly added."""
    with _enqueued_lock:
        if layer_id in _enqueued_layer_ids:
            return False
        _enqueued_layer_ids.add(layer_id)
        return True


# =============================================================================
# Layer Queue Operations
# =============================================================================

def r_dequeue_layer(state: GraphState) -> Dict[str, Any]:
    """Dequeue a layer from the front of the queue."""
    return {"_dequeue_layer": True}


def r_enqueue_layers(layer_ids, state: GraphState) -> Dict[str, Any]:
    """Return children to be enqueued (filters duplicates)."""
    if isinstance(layer_ids, str):
        layer_ids = [layer_ids]
    
    unique_ids = []
    for layer_id in layer_ids:
        if _mark_as_enqueued(layer_id):
            unique_ids.append(layer_id)
        else:
            print(f"[Reducer] WARNING: Duplicate layer_id blocked: {layer_id}")
    
    if not unique_ids:
        return {}
    
    return {"_enqueue_children": unique_ids}


# Legacy aliases
def r_pop_layer_stack(state: GraphState) -> Dict[str, Any]:
    return r_dequeue_layer(state)


def r_push_layer_stack(layer_ids, state: GraphState) -> Dict[str, Any]:
    return r_enqueue_layers(layer_ids, state)


def r_set_current_layer(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """Set current processing layer directly."""
    return {"current_layer_id": layer_id}


# =============================================================================
# Layer Count Operations
# =============================================================================

def r_increment_layer_count(state: GraphState, increment: int = 1) -> Dict[str, Any]:
    """Increment layer count."""
    return {"_increment_layer_count": increment}


# =============================================================================
# History Tree Operations
# =============================================================================

def r_create_layer_node(
    layer_id: str,
    parent_id: Optional[str],
    image_path: str,
    state: GraphState
) -> Dict[str, Any]:
    """Create a new LayerNode in history_tree."""
    tree = state.get("history_tree", {})
    
    parent_depth = 0
    if parent_id and parent_id in tree:
        parent_depth = tree[parent_id].get("depth", 0)
    
    node: LayerNode = {
        "layer_id": layer_id,
        "parent_id": parent_id,
        "depth": parent_depth + 1 if parent_id else 0,
        "image_path": image_path,
        "image_context": None,
        "action_reasoning": None,
        "action_type": None,
        "planned_tool_sequence": None,
        "node_queue": None,
        "param_qwen_len": None,
        "param_is_photo": None,
        "param_inpaint_remainder": None,
        "param_nanobanana_instruction": None,
        "tool_outputs": {
            "qwen_layered": None,
            "split_cca": None,
            "vlm_front_pick": None,
            "detect": [],
            "segment": [],
            "refine": [],
            "fontstyle": None,
            "vtracer": None,
            "verifier": None,
        },
        "children_ids": None,
        "parsed_elements": None,
        "error_info": None,
        "retry_count": 0,
        # [수정 25] Verification fields
        "verification_attempts": [],
        "verification_status": "pending",
        "rejected_child_indices": None,
        "failed_attempts": [],
    }
    
    return {
        "history_tree": {layer_id: node},
        "_increment_layer_count": 1
    }


def r_update_layer_node(
    layer_id: str,
    updates: Dict[str, Any],
    state: GraphState
) -> Dict[str, Any]:
    """Update fields of a LayerNode."""
    history_tree = state.get("history_tree") or {}
    if layer_id not in history_tree:
        return {}
    
    # [수정 30] Return only the updates, not entire node
    # merge_updates/r_pack_state will handle merging
    return {"history_tree": {layer_id: updates}}


def r_set_router_outputs(
    layer_id: str,
    image_context: str,
    action_reasoning: str,
    action_type: str,
    planned_tool_sequence: List[str],
    params: Optional[Dict[str, Any]],
    state: GraphState
) -> Dict[str, Any]:
    """Set router_vlm outputs. [수정 35] Added nanobanana_instruction support."""
    history_tree = state.get("history_tree") or {}
    if layer_id not in history_tree:
        return {}
    
    updates = {
        "image_context": image_context,
        "action_reasoning": action_reasoning,
        "action_type": action_type,
        "planned_tool_sequence": planned_tool_sequence,
        "node_queue": list(planned_tool_sequence),
    }
    
    if params:
        # [수정 35] Include ALL recognized parameters
        recognized_params = [
            "qwen_len",
            "is_photo", 
            "inpaint_remainder",
            "nanobanana_instruction",  # [수정 35] NOW INCLUDED
        ]
        
        for key in recognized_params:
            if key in params:
                updates[f"param_{key}"] = params[key]
    
    return {"history_tree": {layer_id: updates}}


def r_set_children_ids(layer_id: str, children_ids: List[str], state: GraphState) -> Dict[str, Any]:
    """
    Set children_ids for a layer.
    
    [수정 30] Return ONLY children_ids field, not entire node.
    This preserves _temp_child_ids for visualization.
    """
    history_tree = state.get("history_tree") or {}
    if layer_id not in history_tree:
        return {}
    
    # [수정 30] Return only the children_ids field
    return {"history_tree": {layer_id: {"children_ids": children_ids}}}


# =============================================================================
# Node Queue Operations
# =============================================================================

def r_dequeue_node(layer_id: str, state: GraphState) -> Tuple[Optional[str], Dict[str, Any]]:
    """Dequeue next tool from node_queue. Returns (next_node, update_dict)."""
    history_tree = state.get("history_tree") or {}
    if layer_id not in history_tree:
        return None, {}
    
    node = history_tree[layer_id]
    queue = list(node.get("node_queue") or [])
    
    if not queue:
        return None, {}
    
    next_node = queue.pop(0)
    
    # Return ONLY the node_queue field
    return next_node, {"history_tree": {layer_id: {"node_queue": queue}}}


# =============================================================================
# Tool Output Operations
# =============================================================================

def r_save_tool_output(
    layer_id: str,
    tool_category: str,
    tool_name: str,
    output: Dict[str, Any],
    state: GraphState
) -> Dict[str, Any]:
    """Save tool output to layer node."""
    history_tree = state.get("history_tree") or {}
    if layer_id not in history_tree:
        return {}
    
    node = history_tree[layer_id]
    tool_outputs = dict(node.get("tool_outputs") or {})
    
    output_with_tool = dict(output)
    output_with_tool["tool_name"] = tool_name
    
    if tool_category in ["detect", "segment", "refine"]:
        outputs_list = list(tool_outputs.get(tool_category) or [])
        outputs_list.append({"tool_name": tool_name, "output": output_with_tool})
        tool_outputs[tool_category] = outputs_list
    else:
        tool_outputs[tool_category] = output_with_tool
    
    # [수정 30] Return only tool_outputs field
    return {"history_tree": {layer_id: {"tool_outputs": tool_outputs}}}


# =============================================================================
# [수정 25] [수정 30] Verification Output Operations
# =============================================================================

def r_save_verifier_output(
    layer_id: str,
    verifier_output: Dict[str, Any],
    state: GraphState
) -> Dict[str, Any]:
    """
    Save verifier output and compute verification status.
    """
    history_tree = state.get("history_tree")
    
    node = history_tree[layer_id]
    
    # Build tool_outputs update (merge with existing)
    existing_tool_outputs = dict(node.get("tool_outputs"))
    existing_tool_outputs["verifier"] = verifier_output
    
    # ==========================================
    # Compute verification_status from coverage_check and valid_indices
    # ==========================================
    coverage = verifier_output.get("coverage_check").upper()
    valid_indices = verifier_output.get("valid_children_indices")
    invalid_indices = verifier_output.get("invalid_children_indices")
    
    # Calculate total children count
    total_children = len(valid_indices) + len(invalid_indices)
    
    # Determine status
    if coverage == "INCOMPLETE":
        status = "RETRY"
    elif len(valid_indices) == 0:
        status = "RETRY"
    elif len(valid_indices) < total_children:
        status = "PROCEED_FILTERED"
    else:
        status = "PROCEED"
    
    # Create VerificationAttempt and append to existing
    existing_attempts = list(node.get("verification_attempts") or [])
    attempt_number = len(existing_attempts) + 1
    
    attempt: VerificationAttempt = {
        "attempt_number": attempt_number,
        "layer_id": layer_id,
        "action_type": node.get("action_type", "Unknown"),
        "tool_sequence": node.get("planned_tool_sequence", []),
        "child_image_paths": verifier_output.get("_child_image_paths", []),
        "children_analysis": verifier_output.get("children_analysis", []),
        "valid_children_indices": valid_indices,
        "invalid_children_indices": invalid_indices,
        "coverage_check": coverage,  # Store original field name
        "coverage_reason": verifier_output.get("coverage_reason", ""),
        "decision": status,
        "timestamp": verifier_output.get("timestamp", datetime.now().isoformat()),
    }
    
    new_attempts = existing_attempts + [attempt]
    
    # Return ONLY the changed fields, preserving _temp_child_ids etc.
    return {"history_tree": {layer_id: {
        "tool_outputs": existing_tool_outputs,
        "verification_status": status,
        "rejected_child_indices": invalid_indices,
        "verification_attempts": new_attempts,
    }}}

def r_record_failed_attempt(
    layer_id: str,
    action_type: str,
    tool_sequence: List[str],
    failure_reason: str,
    verifier_output: Dict[str, Any],
    state: GraphState
) -> Dict[str, Any]:
    """
    Record a failed attempt for Router to avoid repeating.
    
    [수정 30] Return only changed fields.
    """
    history_tree = state.get("history_tree") or {}
    if layer_id not in history_tree:
        return {}
    
    node = history_tree[layer_id]
    
    # Build new failed_attempts list
    existing_failed = list(node.get("failed_attempts") or [])
    existing_failed.append({
        "action_type": action_type,
        "tool_sequence": tool_sequence,
        "failure_reason": failure_reason,
        "verifier_decision": "RETRY",  # Always RETRY for failed attempts
        "coverage": verifier_output.get("coverage_check", "INCOMPLETE"),
        "timestamp": datetime.now().isoformat(),
    })
    
    return {"history_tree": {layer_id: {
        "failed_attempts": existing_failed,
    }}}

def r_requeue_layer_priority(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """
    Re-queue layer at FRONT of queue for priority retry.
    """
    return {"_requeue_priority": layer_id}


def r_force_finalize(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """
    Force layer to finalize as atomic element.
    
    [수정 30] Return only changed fields.
    """
    history_tree = state.get("history_tree") or {}
    if layer_id not in history_tree:
        return {}
    
    # [수정 30] Return only changed fields
    return {"history_tree": {layer_id: {
        "action_type": "Finalize_Obj",
        "action_reasoning": "Force finalized by verifier FINALIZE decision",
        "planned_tool_sequence": ["finalize_obj"],
        "node_queue": ["finalize_obj"],
        "verification_status": "FINALIZE",
    }}}



def r_create_retry_child_node(
    parent_id: str,
    state: GraphState,
    current_failed_attempt: Optional[Dict[str, Any]] = None,  # [수정 34] 추가
) -> Tuple[str, Dict[str, Any]]:
    """
    Create a child node with the same image for retry.
    
    [수정 31] Preserves parent verification visualization.
    [수정 34] Now accepts current_failed_attempt to ensure complete history transfer.
    
    Args:
        parent_id: The parent layer that received RETRY verdict
        state: Current GraphState
        current_failed_attempt: The failed attempt that triggered this retry (optional)
    
    Returns:
        Tuple of (child_layer_id, update_dict)
    """
    history_tree = state.get("history_tree") or {}
    if parent_id not in history_tree:
        return "", {}
    
    parent_node = history_tree[parent_id]
    parent_depth = parent_node.get("depth", 0)
    parent_image_path = parent_node.get("image_path", "")
    parent_retry_count = parent_node.get("retry_count", 0)
    parent_failed_attempts = list(parent_node.get("failed_attempts") or [])
    
    # [수정 34] Add current failed attempt to history
    if current_failed_attempt:
        parent_failed_attempts.append(current_failed_attempt)
    
    # Generate retry child ID
    new_retry_count = parent_retry_count + 1
    retry_child_id = generate_layer_id(state, f"retry{new_retry_count}")
    
    # Copy image to new layer directory
    episode_dir = state.get("episode_dir", ".")
    child_layer_dir = Path(episode_dir) / "layers" / retry_child_id
    child_layer_dir.mkdir(parents=True, exist_ok=True)
    
    child_image_path = child_layer_dir / "layer_image.png"
    if parent_image_path and Path(parent_image_path).exists():
        shutil.copy(parent_image_path, child_image_path)
    else:
        child_image_path = parent_image_path
    
    # Create child node with complete failed_attempts history
    child_node: LayerNode = {
        "layer_id": retry_child_id,
        "parent_id": parent_id,
        "depth": parent_depth + 1,
        "image_path": str(child_image_path),
        "image_context": f"[Retry #{new_retry_count}] Inherited from {parent_id}",
        "action_reasoning": None,
        "action_type": None,
        "planned_tool_sequence": None,
        "node_queue": None,
        "param_qwen_len": parent_node.get("param_qwen_len"),
        "param_is_photo": parent_node.get("param_is_photo"),
        "param_inpaint_remainder": parent_node.get("param_inpaint_remainder"),
        "param_nanobanana_instruction": parent_node.get("param_nanobanana_instruction"),
        "tool_outputs": {
            "qwen_layered": None,
            "split_cca": None,
            "vlm_front_pick": None,
            "detect": [],
            "segment": [],
            "refine": [],
            "fontstyle": None,
            "vtracer": None,
            "verifier": None,
        },
        "children_ids": None,
        "parsed_elements": None,
        "error_info": None,
        "retry_count": new_retry_count,
        # [수정 34] Complete failed_attempts history (parent's + current)
        "failed_attempts": parent_failed_attempts,
        "verification_attempts": [],
        "verification_status": "pending",
        "rejected_child_indices": None,
        "_temp_child_ids": None,
        "_pending_verification": False,
    }
    
    # Update parent's children_ids
    parent_children = list(parent_node.get("children_ids") or [])
    parent_children.append(retry_child_id)
    
    # [수정 34] Also record on parent node for visualization
    parent_updates = {"children_ids": parent_children}
    if current_failed_attempt:
        existing_parent_failed = list(parent_node.get("failed_attempts") or [])
        existing_parent_failed.append(current_failed_attempt)
        parent_updates["failed_attempts"] = existing_parent_failed
    
    return retry_child_id, {
        "history_tree": {
            retry_child_id: child_node,
            parent_id: parent_updates,
        },
        "_increment_layer_count": 1,
    }


# =============================================================================
# [수정 25] Temporary Children Operations (for visualization)
# =============================================================================

def r_create_temp_child_nodes(
    parent_id: str,
    child_image_paths: List[str],
    attempt_number: int,
    state: GraphState
) -> Dict[str, Any]:
    """
    Create temporary child nodes BEFORE verification.
    
    These nodes will be visible in the tree visualization immediately,
    and their status will be updated after verification completes.
    """
    history_tree = state.get("history_tree") or {}
    if parent_id not in history_tree:
        return {}
    
    parent_node = history_tree[parent_id]
    parent_depth = parent_node.get("depth", 0)
    
    updates = {}
    temp_child_ids = []
    
    for i, image_path in enumerate(child_image_paths):
        temp_id = f"_temp_{parent_id}_a{attempt_number}_c{i}"
        
        temp_node = {
            "layer_id": temp_id,
            "parent_id": parent_id,
            "depth": parent_depth + 1,
            "image_path": image_path,
            "image_context": f"[Pending] Child {i} (Attempt #{attempt_number})",
            "action_reasoning": "Awaiting verification",
            "action_type": "_TempChild",
            "planned_tool_sequence": [],
            "node_queue": [],
            "tool_outputs": {},
            "children_ids": None,
            "parsed_elements": None,
            "error_info": None,
            "retry_count": 0,
            "verification_attempts": [],
            "verification_status": "pending",
            # Special flags
            "_is_temporary": True,
            "_child_index": i,
            "_attempt_number": attempt_number,
        }
        
        updates[temp_id] = temp_node
        temp_child_ids.append(temp_id)
    
    # [수정 30] For parent, return only the fields we're changing
    updates[parent_id] = {
        "_temp_child_ids": temp_child_ids,
        "_pending_verification": True,
    }
    
    return {"history_tree": updates}


def r_update_temp_children_status(
    parent_id: str,
    verifier_output: Dict[str, Any],
    state: GraphState
) -> Dict[str, Any]:
    """
    Update temporary children status based on verification result.
    
    - VALID children: Update status, ready for permanent conversion
    - INVALID children: Mark as rejected (remain in tree for visualization)
    - RETRY: Mark all as rejected
    """
    history_tree = state.get("history_tree") or {}
    if parent_id not in history_tree:
        return {}
    
    parent_node = history_tree[parent_id]
    temp_child_ids = parent_node.get("_temp_child_ids", [])
    
    if not temp_child_ids:
        return {}
    
    decision = verifier_output.get("decision", "PROCEED").upper()
    valid_indices = set(verifier_output.get("valid_children_indices", []))
    invalid_indices = set(verifier_output.get("invalid_children_indices", []))
    children_analysis = verifier_output.get("children_analysis", [])
    
    updates = {}
    
    for temp_id in temp_child_ids:
        if temp_id not in history_tree:
            continue
        
        temp_node = dict(history_tree[temp_id])
        child_index = temp_node.get("_child_index", 0)
        
        # Find analysis for this child
        child_analysis = None
        for analysis in children_analysis:
            if analysis.get("index") == child_index:
                child_analysis = analysis
                break
        
        if decision == "RETRY":
            # All children rejected for retry
            temp_node["verification_status"] = "REJECTED"
            temp_node["action_type"] = "_RejectedChild"
            temp_node["image_context"] = f"[REJECTED - RETRY] {temp_node.get('image_context', '')}"
            if child_analysis:
                temp_node["action_reasoning"] = child_analysis.get("reason", "Retry required")
        
        elif child_index in invalid_indices:
            # This child is invalid
            temp_node["verification_status"] = "INVALID"
            temp_node["action_type"] = "_InvalidChild"
            if child_analysis:
                reason = child_analysis.get("reason", "Invalid")
                temp_node["image_context"] = f"[INVALID] {child_analysis.get('context', '')}"
                temp_node["action_reasoning"] = reason
        
        elif child_index in valid_indices:
            # This child is valid
            temp_node["verification_status"] = "VALID"
            temp_node["action_type"] = "_ValidChild"
            if child_analysis:
                temp_node["image_context"] = f"[VALID] {child_analysis.get('context', '')}"
                temp_node["action_reasoning"] = "Passed verification"
        
        else:
            # Default to valid if not explicitly invalid
            temp_node["verification_status"] = "VALID"
            temp_node["action_type"] = "_ValidChild"
        
        updates[temp_id] = temp_node
    
    # [수정 30] For parent, return only the field we're changing
    updates[parent_id] = {"_pending_verification": False}
    
    return {"history_tree": updates}


# =============================================================================
# Parsed Elements Operations
# =============================================================================

def r_append_parsed_element(element: Dict[str, Any], layer_id: str, state: GraphState) -> Dict[str, Any]:
    """Add a parsed element to both layer node and global list."""
    history_tree = state.get("history_tree") or {}
    
    element_with_ref = dict(element)
    element_with_ref["source_layer_id"] = layer_id
    
    result = {"_append_parsed_element": element_with_ref}
    
    if layer_id in history_tree:
        node = history_tree[layer_id]
        layer_elements = list(node.get("parsed_elements") or [])
        layer_elements.append(element_with_ref)
        # [수정 30] Return only parsed_elements field
        result["history_tree"] = {layer_id: {"parsed_elements": layer_elements}}
    
    return result


# =============================================================================
# Error Handling Operations
# =============================================================================

def r_set_layer_error(
    layer_id: str,
    error_msg: str,
    details: Dict[str, Any],
    state: GraphState
) -> Dict[str, Any]:
    """Set error info for a layer."""
    history_tree = state.get("history_tree") or {}
    if layer_id not in history_tree:
        return {}
    
    node = history_tree[layer_id]
    new_retry_count = (node.get("retry_count") or 0) + 1
    
    # [수정 30] Return only changed fields
    return {"history_tree": {layer_id: {
        "error_info": {"message": error_msg, "details": details or {}},
        "retry_count": new_retry_count,
    }}}


# =============================================================================
# LLM Budget Operations
# =============================================================================

def r_llm_inc(state: GraphState) -> Dict[str, Any]:
    """Increment LLM call count."""
    return {"_increment_llm_count": 1}


def llm_can_call(state: GraphState) -> bool:
    """Check if LLM calls are within budget."""
    count = state.get("llm_call_count")
    limit = state.get("llm_call_limit")
    return count < limit


# =============================================================================
# GPU Slot Operations
# =============================================================================

def r_acquire_gpu_slot(state: GraphState, layer_id: str) -> Tuple[Dict[str, Any], Optional[int]]:
    """Acquire a single available GPU slot."""
    gpu_slots = [dict(slot) for slot in state.get("gpu_slots", [])]
    
    for slot in gpu_slots:
        if slot.get("available", False):
            slot["available"] = False
            slot["layer_id"] = layer_id
            return {"gpu_slots": gpu_slots}, slot["gpu_id"]
    
    return {}, None


def r_release_gpu_slot(gpu_id: int, state: GraphState) -> Dict[str, Any]:
    """Release a specific GPU slot."""
    gpu_slots = [dict(slot) for slot in state.get("gpu_slots", [])]
    
    for slot in gpu_slots:
        if slot["gpu_id"] == gpu_id:
            slot["available"] = True
            slot["layer_id"] = None
            break
    
    return {"gpu_slots": gpu_slots}


def r_release_gpu_slots(gpu_ids: List[int], state: GraphState) -> Dict[str, Any]:
    """Release multiple GPU slots."""
    gpu_slots = [dict(slot) for slot in state.get("gpu_slots", [])]
    
    for slot in gpu_slots:
        if slot["gpu_id"] in gpu_ids:
            slot["available"] = True
            slot["layer_id"] = None
    
    return {"gpu_slots": gpu_slots}


# =============================================================================
# Processing IDs Operations
# =============================================================================

def r_add_processing_id(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """Add layer to processing list."""
    return {"_add_processing_id": layer_id}


def r_remove_processing_id(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """Remove layer from processing list."""
    return {"_remove_processing_id": layer_id}


# =============================================================================
# State Merging
# =============================================================================

def merge_updates(state: GraphState, *updates: Dict[str, Any]) -> GraphState:
    """Merge multiple update dicts into state."""
    new_state = dict(state)
    
    for update in updates:
        if not update:
            continue
        for key, value in update.items():
            if key == "_enqueue_children" and isinstance(value, list):
                queue = list(new_state.get("layer_queue", []))
                queue.extend(value)
                new_state["layer_queue"] = queue
                
            elif key == "_requeue_priority":
                queue = list(new_state.get("layer_queue", []))
                layer_id = value
                if layer_id in queue:
                    queue.remove(layer_id)
                queue.insert(0, layer_id)
                new_state["layer_queue"] = queue
                pending = set(new_state.get("pending_retries", set()))
                pending.add(layer_id)
                new_state["pending_retries"] = pending
                
            elif key == "_dequeue_layer":
                queue = list(new_state.get("layer_queue", []))
                if queue:
                    layer_id = queue.pop(0)
                    new_state["layer_queue"] = queue
                    new_state["current_layer_id"] = layer_id
                    pending = set(new_state.get("pending_retries", set()))
                    pending.discard(layer_id)
                    new_state["pending_retries"] = pending
                else:
                    new_state["current_layer_id"] = None
                    
            elif key == "_increment_layer_count":
                new_state["layer_count"] = new_state.get("layer_count", 0) + value
                
            elif key == "_increment_llm_count":
                new_state["llm_call_count"] = new_state.get("llm_call_count", 0) + value
                
            elif key == "_add_processing_id":
                processing = list(new_state.get("processing_ids", []))
                if value not in processing:
                    processing.append(value)
                new_state["processing_ids"] = processing
                
            elif key == "_remove_processing_id":
                processing = list(new_state.get("processing_ids", []))
                if value in processing:
                    processing.remove(value)
                new_state["processing_ids"] = processing
                
            elif key == "_append_parsed_element":
                elements = list(new_state.get("parsed_elements", []))
                elements.append(value)
                new_state["parsed_elements"] = elements
                
            elif key == "_append_parsed_elements" and isinstance(value, list):
                elements = list(new_state.get("parsed_elements", []))
                elements.extend(value)
                new_state["parsed_elements"] = elements
                
            elif key == "history_tree" and isinstance(value, dict):
                existing = dict(new_state.get("history_tree", {}))
                for layer_id, node_data in value.items():
                    if layer_id in existing:
                        existing_node = dict(existing[layer_id])
                        existing_node.update(node_data)
                        existing[layer_id] = existing_node
                    else:
                        existing[layer_id] = node_data
                new_state["history_tree"] = existing
                
            else:
                new_state[key] = value
    
    return new_state


def r_pack_state(state: GraphState, *updates: Dict[str, Any]) -> Dict[str, Any]:
    """Collect all updates into a single dict for StateManager."""
    final_update = {}
    
    for upd in updates:
        if not upd:
            continue
        for key, value in upd.items():
            if key == "_enqueue_children":
                existing = final_update.get("_enqueue_children", [])
                if isinstance(value, list):
                    existing.extend(value)
                else:
                    existing.append(value)
                final_update["_enqueue_children"] = existing
                
            elif key == "_increment_layer_count":
                final_update["_increment_layer_count"] = final_update.get("_increment_layer_count", 0) + value
                
            elif key == "_increment_llm_count":
                final_update["_increment_llm_count"] = final_update.get("_increment_llm_count", 0) + value
                
            elif key == "_append_parsed_element":
                existing = final_update.get("_append_parsed_elements", [])
                existing.append(value)
                final_update["_append_parsed_elements"] = existing
                
            elif key == "_append_parsed_elements" and isinstance(value, list):
                existing = final_update.get("_append_parsed_elements", [])
                existing.extend(value)
                final_update["_append_parsed_elements"] = existing
                
            elif key == "history_tree" and isinstance(value, dict):
                existing = final_update.get("history_tree", {})
                for layer_id, node_data in value.items():
                    if layer_id in existing:
                        existing_node = dict(existing[layer_id])
                        existing_node.update(node_data)
                        existing[layer_id] = existing_node
                    else:
                        existing[layer_id] = node_data
                final_update["history_tree"] = existing
                
            else:
                final_update[key] = value
    
    return final_update