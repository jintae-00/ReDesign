# ReDesign/episode_run.py
"""
URLD Episode Runner - Main agent execution loop.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import sys
import os
from pathlib import Path
import json
import numpy as np
import time
import gc
import asyncio
import threading
import logging
import shutil
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import deque

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .reconstruction import (
    reconstruct_image,
    compute_z_order_with_info,
    get_src_root,
)
from .prompt_builders import build_final_verification_system_prompt
from .utils import image_to_b64


from .state import (
    GraphState,
    LayerNode,
    initialize_graph_state,
    save_state_to_disk,
    generate_layer_id
)
from .reducers import (
    r_dequeue_layer,
    r_enqueue_layers,
    r_add_processing_id,
    r_remove_processing_id,
    r_dequeue_node,
    r_acquire_gpu_slot,
    r_release_gpu_slot,
    r_release_layer_gpu,
    r_update_layer_node,
    merge_updates,
    reset_enqueued_tracking,
    r_pop_layer_stack,
    r_push_layer_stack,
    r_pack_state,
    r_append_parsed_element,
)


current_file = Path(__file__).resolve()
src_root = current_file.parent.parent
modules_root = src_root / "modules"

if str(modules_root) not in sys.path:
    sys.path.append(str(modules_root))


# =============================================================================
# [Revision 36] Constants
# =============================================================================

MAX_RETRIES = 3  # Maximum retry attempts per layer before force finalize


# =============================================================================
# [Revision 11] Logging Setup
# =============================================================================

_episode_logger: Optional[logging.Logger] = None


def setup_logger(episode_dir: str, name: str = "urld") -> logging.Logger:
    """
    Setup logger that writes to both console and file.
    
    Args:
        episode_dir: Directory where log file will be saved
        name: Logger name
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Path(episode_dir) / "episode.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def get_logger() -> logging.Logger:
    """Get the current episode logger."""
    global _episode_logger
    if _episode_logger is None:
        return logging.getLogger("urld")
    return _episode_logger


def log(message: str, prefix: str = ""):
    """Log message to both console and file."""
    logger = get_logger()
    if prefix:
        logger.info(f"[{prefix}] {message}")
    else:
        logger.info(message)


# =============================================================================
# Thread-safe State Manager
# =============================================================================

class StateManager:
    """
    Thread-safe state management for parallel execution.
    
    [Revision 24] Added _requeue_priority and pending_retries support.
    """
    
    def __init__(self, initial_state: GraphState):
        self._state = dict(initial_state)
        self._lock = threading.RLock()
        self._processing_set: set = set()
        # [Revision 24] Initialize pending_retries
        self._state["pending_retries"] = set()
    
    @property
    def state(self) -> GraphState:
        with self._lock:
            return dict(self._state)
    
    def update(self, *updates: Dict[str, Any]) -> GraphState:
        """
        Apply updates atomically and return new state.
        
        [Revision 24] Added _requeue_priority handling.
        """
        with self._lock:
            for update in updates:
                if not update:
                    continue
                for key, value in update.items():
                    if key == "_enqueue_children" and isinstance(value, list):
                        queue = list(self._state.get("layer_queue", []))
                        queue.extend(value)
                        self._state["layer_queue"] = queue
                        log(f"Enqueued {len(value)} children: {value}", "StateManager")
                    
                    # [Revision 24] Handle _requeue_priority for retry mechanism
                    elif key == "_requeue_priority":
                        queue = list(self._state.get("layer_queue", []))
                        layer_id = value
                        
                        # Remove if already in queue, then add to FRONT
                        if layer_id in queue:
                            queue.remove(layer_id)
                        queue.insert(0, layer_id)
                        self._state["layer_queue"] = queue
                        
                        # Track pending retries to prevent premature termination
                        pending = set(self._state.get("pending_retries", set()))
                        pending.add(layer_id)
                        self._state["pending_retries"] = pending
                        
                        log(f"Priority re-queued {layer_id} for retry. Queue: {queue[:5]}...", "StateManager")
                        
                    elif key == "_dequeue_layer":
                        queue = list(self._state.get("layer_queue", []))
                        if queue:
                            layer_id = queue.pop(0)
                            self._state["layer_queue"] = queue
                            self._state["current_layer_id"] = layer_id
                            
                            # [Revision 24] Remove from pending_retries when dequeued
                            pending = set(self._state.get("pending_retries", set()))
                            pending.discard(layer_id)
                            self._state["pending_retries"] = pending
                        else:
                            self._state["current_layer_id"] = None
                            
                    elif key == "_increment_layer_count":
                        self._state["layer_count"] = self._state.get("layer_count", 0) + value
                        
                    elif key == "_increment_llm_count":
                        self._state["llm_call_count"] = self._state.get("llm_call_count", 0) + value
                        
                    elif key == "_add_processing_id":
                        processing = list(self._state.get("processing_ids", []))
                        if value not in processing:
                            processing.append(value)
                        self._state["processing_ids"] = processing
                        self._processing_set.add(value)
                        
                    elif key == "_remove_processing_id":
                        processing = list(self._state.get("processing_ids", []))
                        if value in processing:
                            processing.remove(value)
                        self._state["processing_ids"] = processing
                        self._processing_set.discard(value)
                        
                    elif key == "_append_parsed_element":
                        elements = list(self._state.get("parsed_elements", []))
                        elements.append(value)
                        self._state["parsed_elements"] = elements
                        
                    elif key == "_append_parsed_elements" and isinstance(value, list):
                        elements = list(self._state.get("parsed_elements", []))
                        elements.extend(value)
                        self._state["parsed_elements"] = elements
                        
                    elif key == "_ocr_fatal_error_count":
                        # Cumulative counter - summed up even when raised concurrently by multiple workers
                        self._state["_ocr_fatal_error_count"] = (
                            self._state.get("_ocr_fatal_error_count", 0) + value
                        )

                    elif key == "history_tree" and isinstance(value, dict):
                        existing = dict(self._state.get("history_tree", {}))
                        for layer_id, node_data in value.items():
                            if layer_id in existing:
                                existing_node = dict(existing[layer_id])
                                existing_node.update(node_data)
                                existing[layer_id] = existing_node
                            else:
                                existing[layer_id] = node_data
                        self._state["history_tree"] = existing
                        
                    else:
                        self._state[key] = value
                        
            return dict(self._state)
    
    def dequeue_layer(self) -> Optional[str]:
        """Dequeue a layer from the front of the queue atomically."""
        with self._lock:
            queue = list(self._state.get("layer_queue", []))
            if not queue:
                return None
            layer_id = queue.pop(0)
            self._state["layer_queue"] = queue
            
            # [Revision 24] Remove from pending_retries
            pending = set(self._state.get("pending_retries", set()))
            pending.discard(layer_id)
            self._state["pending_retries"] = pending
            
            return layer_id
    
    def dequeue_layers(self, count: int) -> List[str]:
        """Dequeue multiple layers from the front of the queue atomically."""
        with self._lock:
            queue = list(self._state.get("layer_queue", []))
            dequeued = []
            pending = set(self._state.get("pending_retries", set()))
            
            for _ in range(count):
                if queue:
                    layer_id = queue.pop(0)
                    dequeued.append(layer_id)
                    # [Revision 24] Remove from pending_retries
                    pending.discard(layer_id)
            
            self._state["layer_queue"] = queue
            self._state["pending_retries"] = pending
            return dequeued
    
    def acquire_gpu(self, layer_id: str) -> Optional[int]:
        """Acquire a GPU slot atomically."""
        with self._lock:
            gpu_slots = [dict(slot) for slot in self._state.get("gpu_slots", [])]
            for slot in gpu_slots:
                if slot.get("available", False):
                    slot["available"] = False
                    slot["layer_id"] = layer_id
                    self._state["gpu_slots"] = gpu_slots
                    return slot["gpu_id"]
            return None
    
    def release_gpu(self, gpu_id: int) -> None:
        """Release a GPU slot atomically."""
        with self._lock:
            gpu_slots = [dict(slot) for slot in self._state.get("gpu_slots", [])]
            for slot in gpu_slots:
                if slot["gpu_id"] == gpu_id:
                    slot["available"] = True
                    slot["layer_id"] = None
                    break
            self._state["gpu_slots"] = gpu_slots
    
    def add_processing(self, layer_id: str) -> bool:
        """Add layer to processing list. Returns False if already processing."""
        with self._lock:
            if layer_id in self._processing_set:
                log(f"WARNING: {layer_id} already in processing_set, skipping", "StateManager")
                return False
            
            processing = list(self._state.get("processing_ids", []))
            if layer_id not in processing:
                processing.append(layer_id)
            self._state["processing_ids"] = processing
            self._processing_set.add(layer_id)
            return True
    
    def remove_processing(self, layer_id: str) -> None:
        """Remove layer from processing list."""
        with self._lock:
            processing = list(self._state.get("processing_ids", []))
            if layer_id in processing:
                processing.remove(layer_id)
            self._state["processing_ids"] = processing
            self._processing_set.discard(layer_id)
    
    def is_processing(self, layer_id: str) -> bool:
        """Check if layer is currently being processed."""
        with self._lock:
            return layer_id in self._processing_set
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current stats."""
        with self._lock:
            return {
                "queue_size": len(self._state.get("layer_queue", [])),
                "processing": len(self._state.get("processing_ids", [])),
                "elements": len(self._state.get("parsed_elements", [])),
                "layers": self._state.get("layer_count", 0),
                "llm_calls": self._state.get("llm_call_count", 0),
                "pending_retries": len(self._state.get("pending_retries", set())),
            }
    
    def should_terminate(self) -> tuple[bool, str]:
        """
        Check if processing should stop.
        
        [Revision 24] Now checks pending_retries before terminating.
        [Revision 36] Note: retry_count is NOT checked here - it's per-node, not global
        """
        with self._lock:
            queue = self._state.get("layer_queue", [])
            processing = self._state.get("processing_ids", [])
            pending_retries = self._state.get("pending_retries", set())
            
            # [Revision 24] Don't terminate if there are pending retries
            if not queue and not processing and not pending_retries:
                return True, "All layers processed"
            
            layer_count = self._state.get("layer_count", 0)
            max_layers = self._state.get("max_layers", 100)
            if layer_count >= max_layers:
                return True, f"Max layers reached ({max_layers})"
            
            llm_calls = self._state.get("llm_call_count", 0)
            llm_limit = self._state.get("llm_call_limit", 100)
            if llm_calls >= llm_limit:
                return True, f"LLM budget exhausted ({llm_limit})"
            
            # [Revision 12] Check max_depth
            max_depth = self._state.get("max_depth")
            history_tree = self._state.get("history_tree", {})
            
            all_exceed_depth = True
            for layer_id in queue:
                if layer_id in history_tree:
                    depth = history_tree[layer_id].get("depth", 0)
                    if depth < max_depth:
                        all_exceed_depth = False
                        break
            
            if queue and all_exceed_depth:
                return True, f"All pending layers exceed max_depth ({max_depth})"
            
            return False, ""


# =============================================================================
# Node Registry
# =============================================================================

def _get_node_function(node_name: str):
    """Get the node function by name."""
    if node_name == "router_vlm":
        from .nodes.router_vlm import node
        return node
    elif node_name == "vlm_front_pick":
        from .nodes.vlm_front_pick import node
        return node
    elif node_name == "qwen_layered":
        from .nodes.qwen_layered import node
        return node
    elif node_name == "split_cca":
        from .nodes.split_cca import node
        return node
    elif node_name == "ocr":
        from .nodes.detect_ocr import node
        return node
    elif node_name == "gdino":
        from .nodes.detect_gdino import node
        return node
    elif node_name == "hisam":
        from .nodes.seg_hisam import node
        return node
    elif node_name == "sam2_bbox":
        from .nodes.seg_sam2_bbox import node
        return node
    elif node_name == "lama":
        from .nodes.inpaint_lama import node
        return node
    elif node_name == "objectclear":
        from .nodes.inpaint_oc import node
        return node
    elif node_name == "nanobanana":
        from .nodes.nanobanana import node
        return node
    elif node_name == "fontstyle":
        from .nodes.fontstyle import node
        return node
    elif node_name == "vtracer":
        from .nodes.vtracer import node
        return node
    elif node_name == "stack_manager":
        from .nodes.stack_manager import node
        return node
    elif node_name == "finalize_text":
        from .nodes.finalize_text import node
        return node
    elif node_name == "finalize_obj":
        from .nodes.finalize_obj import node
        return node
    else:
        raise ValueError(f"Unknown node: {node_name}")


# =============================================================================
# [Revision 36] Force Finalize Helper Functions
# =============================================================================

def _get_unfinished_layers(state: GraphState) -> List[str]:
    """
    Get list of layer IDs that need to be finalized.
    
    [Revision 36] Now traverses entire history_tree to find ALL unfinished layers:
    - Layers in queue
    - Layers in processing (shouldn't exist after gather, but just in case)
    - Leaf nodes in history_tree that are not finalized
    
    A layer is considered "unfinished" if:
    - It exists in history_tree
    - Its action_type is NOT one of: Finalize_Text, Finalize_Obj
    - It has no children (leaf node) OR is in queue/processing
    """
    history_tree = state.get("history_tree", {})
    queue = set(state.get("layer_queue", []))
    processing = set(state.get("processing_ids", []))
    
    finalized_actions = {"Finalize_Text", "Finalize_Obj"}
    unfinished = []
    seen = set()
    
    # 1. Check layers in queue
    for layer_id in queue:
        if layer_id in history_tree and layer_id not in seen:
            node = history_tree[layer_id]
            action_type = node.get("action_type")
            if not action_type or action_type not in finalized_actions:
                unfinished.append(layer_id)
                seen.add(layer_id)
    
    # 2. Check layers in processing (shouldn't happen after gather)
    for layer_id in processing:
        if layer_id in history_tree and layer_id not in seen:
            node = history_tree[layer_id]
            action_type = node.get("action_type")
            if not action_type or action_type not in finalized_actions:
                unfinished.append(layer_id)
                seen.add(layer_id)
    
    # 3. [Revision 36] Traverse entire history_tree for unfinished leaf nodes
    for layer_id, node in history_tree.items():
        if layer_id in seen:
            continue
        
        # Skip temp/virtual nodes
        if layer_id.startswith("_temp_"):
            continue
        
        action_type = node.get("action_type")
        children_ids = node.get("children_ids") or []
        
        # Filter out temp children
        real_children = [c for c in children_ids if not c.startswith("_temp_")]
        
        # Check if this is an unfinished leaf node
        is_leaf = len(real_children) == 0
        is_finalized = action_type in finalized_actions
        
        if is_leaf and not is_finalized:
            # This is an unfinished leaf - needs force finalize
            unfinished.append(layer_id)
            seen.add(layer_id)
    
    return unfinished


def _force_finalize_layer(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """
    Force finalize a layer using finalize_obj node logic.
    
    [Revision 36] CRITICAL FIX: action_type change is now INCLUDED in return value!
    
    This ensures:
    - action_type = "Finalize_Obj" is properly set in history_tree
    - Visualization shows the correct Finalize_Obj color
    - parsed_elements includes the element
    
    Args:
        layer_id: Layer to force finalize
        state: Current graph state
    
    Returns:
        Update dict containing action_type change + finalize_obj results
    """
    from .nodes.finalize_obj import node as finalize_obj_node
    
    history_tree = state.get("history_tree", {})
    if layer_id not in history_tree:
        log(f"WARNING: {layer_id} not in history_tree, skipping force finalize", "ForceFinalize")
        return {}
    
    node_data = history_tree[layer_id]
    original_action = node_data.get("action_type", "Unknown")
    
    log(f"Force finalizing {layer_id} (was: {original_action})", "ForceFinalize")
    
    # 1. Prepare action_type change update
    action_update = {
        "history_tree": {
            layer_id: {
                "action_type": "Finalize_Obj",
                "action_reasoning": f"Force finalized due to termination (original: {original_action})",
                "node_queue": [],
            }
        }
    }
    
    # 2. Create temporary state with action_type already set
    temp_state = merge_updates(state, action_update)
    temp_state["current_layer_id"] = layer_id
    
    # 3. Run finalize_obj node
    try:
        finalize_update = finalize_obj_node(temp_state)
    except Exception as e:
        log(f"ERROR in finalize_obj for {layer_id}: {e}", "ForceFinalize")
        # Even if finalize fails, still update action_type
        return action_update
    
    # 4. Combine action_update + finalize_update
    # Use r_pack_state to properly merge both updates
    combined_update = r_pack_state(state, action_update, finalize_update)
    
    return combined_update


def _force_finalize_single_layer(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """
    [Revision 36] Force finalize a single layer (for max retry exceeded case).
    
    Called from process_layer_worker when retry_count >= MAX_RETRIES.
    Same logic as _force_finalize_layer but with different logging.
    
    Args:
        layer_id: Layer to force finalize
        state: Current graph state
    
    Returns:
        Update dict containing action_type change + finalize_obj results
    """
    log(f"Max retries ({MAX_RETRIES}) exceeded, force finalizing {layer_id}", "Worker")
    return _force_finalize_layer(layer_id, state)


# =============================================================================
# Worker Task for Layer Processing
# =============================================================================

def _should_create_error_retry(layer_id: str, state: GraphState) -> bool:
    """
    Check if we should create an error retry child for this layer.
    
    Returns False if:
    - Max retries exceeded (3)
    - Layer already has children (was already processed successfully)
    """
    history_tree = state.get("history_tree") or {}
    if layer_id not in history_tree:
        return False
    
    node = history_tree[layer_id]
    
    # Check retry count
    retry_count = node.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        return False
    
    # Check if already has children (successful split)
    children = node.get("children_ids") or []
    # Filter out retry children - only count real decomposition children
    real_children = [c for c in children if not c.startswith("_temp_") and "retry" not in c and "err_retry" not in c]
    if real_children:
        return False
    
    return True


def _create_node_error_retry_child(
    parent_id: str,
    failed_node: str,
    error_message: str,
    state: GraphState
) -> Tuple[str, Dict[str, Any]]:
    """
    Create a child node with same image after a node execution error.
    
    This allows Router VLM to see the error history and try a different approach.
    """
    from .state import generate_layer_id, LayerNode
    
    history_tree = state.get("history_tree") or {}
    if parent_id not in history_tree:
        return "", {}
    
    parent_node = history_tree[parent_id]
    parent_depth = parent_node.get("depth", 0)
    parent_image_path = parent_node.get("image_path", "")
    parent_retry_count = parent_node.get("retry_count", 0)
    parent_failed_attempts = list(parent_node.get("failed_attempts") or [])
    
    # Build the current node error record
    current_error = {
        "error_type": "node_execution_error",
        "failed_node": failed_node,
        "error_message": error_message,
        "action_type": parent_node.get("action_type", "Unknown"),
        "tool_sequence": parent_node.get("planned_tool_sequence", []),
        "image_context": parent_node.get("image_context", ""),
        "action_reasoning": parent_node.get("action_reasoning", ""),
        "timestamp": datetime.now().isoformat(),
    }
    parent_failed_attempts.append(current_error)
    
    # Check max retries
    new_retry_count = parent_retry_count + 1
    if new_retry_count > MAX_RETRIES:
        return "", {}  # Exceeded max retries
    
    # Generate retry child ID
    retry_child_id = generate_layer_id(state, f"err_retry{new_retry_count}")
    
    # Copy image to new layer directory
    episode_dir = state.get("episode_dir", ".")
    child_layer_dir = Path(episode_dir) / "layers" / retry_child_id
    child_layer_dir.mkdir(parents=True, exist_ok=True)
    
    child_image_path = child_layer_dir / "layer_image.png"
    if parent_image_path and Path(parent_image_path).exists():
        shutil.copy(parent_image_path, child_image_path)
    else:
        child_image_path = parent_image_path
    
    # Create child node - Router will be called fresh on this child
    child_node: LayerNode = {
        "layer_id": retry_child_id,
        "parent_id": parent_id,
        "depth": parent_depth + 1,
        "image_path": str(child_image_path),
        "image_context": f"[Node Error Retry #{new_retry_count}] {failed_node} failed: {error_message[:100]}",
        "action_reasoning": None,  # Router will decide new action
        "action_type": None,  # Router will decide new action
        "planned_tool_sequence": None,
        "node_queue": None,
        # Inherit params for reference
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
        # Inherit ALL failed attempts history
        "failed_attempts": parent_failed_attempts,
        "verification_attempts": [],
        "verification_status": "pending",
        "rejected_child_indices": None,
        "_temp_child_ids": None,
        "_pending_verification": False,
    }
    
    # Update parent's children_ids and record error
    parent_children = list(parent_node.get("children_ids") or [])
    parent_children.append(retry_child_id)
    
    parent_updates = {
        "children_ids": parent_children,
        "failed_attempts": parent_failed_attempts,
        "error_info": {
            "message": error_message,
            "details": {"node": failed_node, "type": "node_execution_error", "retry_child": retry_child_id}
        }
    }
    
    return retry_child_id, {
        "history_tree": {
            retry_child_id: child_node,
            parent_id: parent_updates,
        },
        "_increment_layer_count": 1,
    }


def process_layer_worker(
    state_manager: StateManager,
    layer_id: str,
    verbose: bool = True
) -> None:
    """
    Worker function to process a single layer through its tool sequence.
    Runs in a separate thread.
    
    [Revision 29] Added GPU timeout error handling and proper error recovery.
    [Revision 35] Added node error retry mechanism.
    [Revision 36] Added retry_count check at start - force finalize if exceeded.
    """
    prefix = f"Worker:{layer_id}"
    
    log(f"Starting processing", prefix)
    
    if not state_manager.add_processing(layer_id):
        log("Skipping - already being processed by another worker", prefix)
        return
    
    try:
        state = state_manager.state
        state["current_layer_id"] = layer_id
        
        node_data = state["history_tree"].get(layer_id)
        
        if not node_data:
            log(f"ERROR: {layer_id} not found in history_tree", prefix)
            return
        
        # [Revision 36] ★ Check retry_count at START - force finalize if exceeded
        retry_count = node_data.get("retry_count", 0)
        if retry_count >= MAX_RETRIES:
            log(f"retry_count ({retry_count}) >= MAX_RETRIES ({MAX_RETRIES}), force finalizing", prefix)
            force_update = _force_finalize_single_layer(layer_id, state)
            state_manager.update(force_update)
            return  # Early exit - don't process further
        
        # Run router if action not yet determined
        if not node_data.get("action_type"):
            log("Running router_vlm...", prefix)
            node_fn = _get_node_function("router_vlm")
            update = node_fn(state)
            state = state_manager.update(update)
            
            node_data = state["history_tree"].get(layer_id)
            
            if node_data.get("error_info"):
                log(f"Router error: {node_data['error_info']}", prefix)
                return
        else:
            log(f"Action pre-configured: {node_data.get('action_type')}, skipping router_vlm", prefix)
        
        action_type = node_data.get("action_type", "Unknown")
        log(f"Action: {action_type}", prefix)
        
        # Tool sequence execution
        while True:
            state = state_manager.state
            state["current_layer_id"] = layer_id
            
            next_node, queue_update = r_dequeue_node(layer_id, state)
            
            if not next_node:
                break
            
            log(f"  -> {next_node}", prefix)
            
            try:
                node_fn = _get_node_function(next_node)
                update = node_fn(state)
                state_manager.update(update, queue_update)
                
                # Check for node-level errors
                state = state_manager.state
                node_data = state["history_tree"].get(layer_id)
                
                if node_data and node_data.get("error_info"):
                    error_info = node_data["error_info"]
                    error_msg = error_info.get("message", "Unknown error")
                    
                    log(f"  NODE ERROR: {error_msg}", prefix)
                    
                    # [Revision 36] Check retry_count explicitly
                    current_retry_count = node_data.get("retry_count", 0)
                    has_real_children = bool([
                        c for c in (node_data.get("children_ids") or [])
                        if not c.startswith("_temp_") and "retry" not in c
                    ])
                    
                    if current_retry_count >= MAX_RETRIES:
                        # Max retry exceeded → force finalize
                        log(f"  Max retries exceeded ({current_retry_count}), force finalizing", prefix)
                        force_update = _force_finalize_single_layer(layer_id, state)
                        state_manager.update(force_update)
                    elif has_real_children:
                        # Already has children → normal end
                        log(f"  Has real children, ending normally", prefix)
                    elif _should_create_error_retry(layer_id, state):
                        # Can create retry child
                        log(f"  Creating error retry child for {layer_id}...", prefix)
                        
                        retry_child_id, retry_update = _create_node_error_retry_child(
                            parent_id=layer_id,
                            failed_node=next_node,
                            error_message=error_msg,
                            state=state
                        )
                        
                        if retry_child_id:
                            enqueue_update = {"_enqueue_children": [retry_child_id]}
                            state_manager.update(retry_update, enqueue_update)
                            log(f"  Created retry child: {retry_child_id}", prefix)
                        else:
                            # This shouldn't happen given the checks above
                            log(f"  Failed to create retry child, force finalizing", prefix)
                            force_update = _force_finalize_single_layer(layer_id, state)
                            state_manager.update(force_update)
                    else:
                        # Shouldn't reach here, but safety net
                        log(f"  No retry possible, force finalizing", prefix)
                        force_update = _force_finalize_single_layer(layer_id, state)
                        state_manager.update(force_update)
                    
                    break  # Stop processing this layer
                
            except TimeoutError as e:
                # [Revision 29] GPU acquisition timeout
                log(f"  GPU TIMEOUT: {e}", prefix)
                
                try:
                    from .gpu_manager import print_gpu_status
                    print_gpu_status()
                except Exception:
                    pass
                
                error_msg = f"GPU timeout in {next_node}: {str(e)}"
                _handle_worker_error(state_manager, layer_id, next_node, error_msg, "gpu_timeout", queue_update, prefix)
                break
                
            except Exception as e:
                # [Revision 29] Check if it's a CUDA error
                error_str = str(e).lower()
                is_cuda_error = any(kw in error_str for kw in [
                    'cuda', 'cudnn', 'gpu', 'device', 'illegal', 'memory'
                ])
                
                if is_cuda_error:
                    log(f"  CUDA ERROR in {next_node}: {e}", prefix)
                    try:
                        from .gpu_manager import print_gpu_status
                        print_gpu_status()
                    except Exception:
                        pass
                else:
                    log(f"  EXCEPTION in {next_node}: {e}", prefix)
                
                error_msg = f"Exception in {next_node}: {str(e)}"
                error_type = "cuda_error" if is_cuda_error else "exception"
                _handle_worker_error(state_manager, layer_id, next_node, error_msg, error_type, queue_update, prefix)
                break
            
            gc.collect()
        
        log("Completed", prefix)
        
    except Exception as e:
        log(f"WORKER EXCEPTION: {e}", prefix)
        import traceback
        traceback.print_exc()
        
    finally:
        state_manager.remove_processing(layer_id)
        gc.collect()


def _handle_worker_error(
    state_manager: StateManager,
    layer_id: str,
    failed_node: str,
    error_msg: str,
    error_type: str,
    queue_update: Dict[str, Any],
    prefix: str
) -> None:
    """
    [Revision 36] Centralized error handling for worker.
    Handles retry logic or force finalize based on retry_count.
    """
    state = state_manager.state
    node_data = state["history_tree"].get(layer_id, {})
    current_retry_count = node_data.get("retry_count", 0)
    
    has_real_children = bool([
        c for c in (node_data.get("children_ids") or [])
        if not c.startswith("_temp_") and "retry" not in c
    ])
    
    if current_retry_count >= MAX_RETRIES:
        # Max retry exceeded → force finalize
        log(f"  Max retries exceeded ({current_retry_count}), force finalizing", prefix)
        force_update = _force_finalize_single_layer(layer_id, state)
        state_manager.update(force_update, queue_update)
        
    elif has_real_children:
        # Already has children → just record error
        error_update = {
            "history_tree": {
                layer_id: {
                    "error_info": {
                        "message": error_msg,
                        "details": {"node": failed_node, "type": error_type}
                    }
                }
            }
        }
        state_manager.update(error_update, queue_update)
        
    elif _should_create_error_retry(layer_id, state):
        # Can create retry child
        retry_child_id, retry_update = _create_node_error_retry_child(
            parent_id=layer_id,
            failed_node=failed_node,
            error_message=error_msg,
            state=state
        )
        if retry_child_id:
            enqueue_update = {"_enqueue_children": [retry_child_id]}
            state_manager.update(retry_update, enqueue_update, queue_update)
            log(f"  Created retry child after {error_type}: {retry_child_id}", prefix)
        else:
            # Fallback: force finalize
            log(f"  Cannot create retry child, force finalizing", prefix)
            force_update = _force_finalize_single_layer(layer_id, state)
            state_manager.update(force_update, queue_update)
    else:
        # Fallback: force finalize
        log(f"  Cannot retry, force finalizing", prefix)
        force_update = _force_finalize_single_layer(layer_id, state)
        state_manager.update(force_update, queue_update)


# =============================================================================
# Episode Runner (Sequential Mode)
# =============================================================================

class EpisodeRunner:
    """
    Manages execution of a single episode (one image decomposition).
    Sequential execution mode with FIFO queue processing.
    """
    
    def __init__(
        self,
        run_dir: str,
        episode_id: str,
        image_path: str,
        llm_call_limit: int = 100,
        max_depth: int = 5,
        max_layers: int = 100,
        available_gpus: List[int] = None,
        max_parallel_workers: int = 4,
        verbose: bool = True,
        enable_visualization: bool = True,
        visualization_interval: int = 1,
    ):
        self.verbose = verbose
        self.start_time = None
        self.enable_visualization = enable_visualization
        
        reset_enqueued_tracking()
        
        self.state = initialize_graph_state(
            run_dir=run_dir,
            episode_id=episode_id,
            original_image_path=image_path,
            llm_call_limit=llm_call_limit,
            max_depth=max_depth,
            max_layers=max_layers,
            available_gpus=available_gpus,
            max_parallel_workers=max_parallel_workers,
        )
        
        # [Revision 24] Initialize pending_retries
        self.state["pending_retries"] = set()
        
        # [Revision 11] Setup file logging
        global _episode_logger
        _episode_logger = setup_logger(self.state["episode_dir"])
        
        self.visualizer = None
        if enable_visualization:
            try:
                from .visualizer import TreeVisualizer
                self.visualizer = TreeVisualizer(
                    self.state["episode_dir"],
                    update_interval=visualization_interval
                )
            except ImportError as e:
                self._log(f"Warning: Visualizer not available: {e}")
        
        self._log(f"Initialized episode {episode_id}")
        self._log(f"  Image: {image_path}")
        self._log(f"  Episode dir: {self.state['episode_dir']}")
        self._log(f"  Available GPUs: {[s['gpu_id'] for s in self.state['gpu_slots']]}")
        self._log(f"  Visualization: {'Enabled' if self.visualizer else 'Disabled'}")
        self._log(f"  Processing Mode: FIFO Queue (BFS)")
        self._log(f"  Log file: {self.state['episode_dir']}/episode.log")
        self._log(f"  MAX_RETRIES: {MAX_RETRIES}")
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            log(message, "Episode")
    
    def _update_visualization(self, force: bool = True):
        """Update tree visualization."""
        if self.visualizer:
            try:
                self.visualizer.update(self.state, force=force)
            except Exception as e:
                self._log(f"Visualization update failed: {e}")
    
    def _should_terminate(self) -> tuple[bool, str]:
        """
        Check if episode should terminate.
        
        [Revision 24] Now checks pending_retries.
        [Revision 36] Note: retry_count is per-node, checked in _process_layer
        """
        queue = self.state.get("layer_queue", [])
        processing = self.state.get("processing_ids", [])
        pending_retries = self.state.get("pending_retries", set())
        
        queue_size = len(queue)
        processing_size = len(processing)
        pending_size = len(pending_retries)
        
        # [Revision 24] Don't terminate if there are pending retries
        if not queue and not processing and not pending_retries:
            return True, f"All layers processed (queue={queue_size}, processing={processing_size}, pending={pending_size})"
        
        layer_count = self.state.get("layer_count", 0)
        max_layers = self.state.get("max_layers", 100)
        if layer_count >= max_layers:
            return True, f"Max layers reached ({layer_count}/{max_layers}), queue={queue_size}, processing={processing_size}"
        
        llm_calls = self.state.get("llm_call_count", 0)
        llm_limit = self.state.get("llm_call_limit", 100)
        if llm_calls >= llm_limit:
            return True, f"LLM budget exhausted ({llm_calls}/{llm_limit}), queue={queue_size}, processing={processing_size}"
        
        # [Revision 12] Check max_depth
        max_depth = self.state.get("max_depth")
        history_tree = self.state.get("history_tree", {})
        
        all_exceed_depth = True
        for layer_id in queue:
            if layer_id in history_tree:
                depth = history_tree[layer_id].get("depth", 0)
                if depth < max_depth:
                    all_exceed_depth = False
                    break
        
        if queue and all_exceed_depth:
            return True, f"All pending layers exceed max_depth ({max_depth}), queue={queue_size}"
        
        return False, ""
    
    def _execute_node(self, node_name: str) -> Dict[str, Any]:
        """Execute a single node and return state update."""
        node_fn = _get_node_function(node_name)
        update = node_fn(self.state)
        return update
    
    def _process_layer(self, layer_id: str) -> None:
        """Process a single layer through its tool sequence."""
        self._log(f"\n{'='*60}")
        self._log(f"Processing layer: {layer_id}")
        
        self.state["current_layer_id"] = layer_id
        
        update = r_add_processing_id(layer_id, self.state)
        self.state = merge_updates(self.state, update)
        
        # [Revision 24] Remove from pending_retries when actually processing
        pending = set(self.state.get("pending_retries", set()))
        pending.discard(layer_id)
        self.state["pending_retries"] = pending
        
        self._update_visualization()
        
        try:
            node_data = self.state["history_tree"].get(layer_id)
            if not node_data:
                raise ValueError(f"Layer {layer_id} not found in history_tree")
            
            # [Revision 36] ★ Check retry_count at START
            retry_count = node_data.get("retry_count", 0)
            if retry_count >= MAX_RETRIES:
                self._log(f"retry_count ({retry_count}) >= MAX_RETRIES ({MAX_RETRIES}), force finalizing")
                update = _force_finalize_single_layer(layer_id, self.state)
                self.state = merge_updates(self.state, update)
                return  # Early exit
            
            if not node_data.get("action_type"):
                self._log("Running router_vlm...")
                update = self._execute_node("router_vlm")
                self.state = merge_updates(self.state, update)
                
                node_data = self.state["history_tree"].get(layer_id)
                
                if node_data.get("error_info"):
                    self._log(f"Router error: {node_data['error_info']}")
                    return
            else:
                self._log(f"Action pre-configured: {node_data.get('action_type')}, skipping router_vlm")
            
            node_queue = node_data.get("node_queue", [])
            action_type = node_data.get("action_type", "Unknown")
            
            self._log(f"Action: {action_type}")
            self._log(f"Tool sequence: {node_queue}")
            
            # Tool sequence execution
            while True:
                next_node, queue_update = r_dequeue_node(layer_id, self.state)
                
                if not next_node:
                    break
                
                self._log(f"  -> Executing: {next_node}")
                
                try:
                    update = self._execute_node(next_node)
                    self.state = merge_updates(self.state, update, queue_update)
                except TimeoutError as e:
                    self._log(f"  GPU TIMEOUT: {e}")
                    self.state = merge_updates(self.state, queue_update, {
                        "history_tree": {
                            layer_id: {
                                "error_info": {
                                    "message": f"GPU timeout in {next_node}",
                                    "details": {"node": next_node, "type": "gpu_timeout"}
                                }
                            }
                        }
                    })
                    break
                except Exception as e:
                    self._log(f"  ERROR in {next_node}: {e}")
                    self.state = merge_updates(self.state, queue_update, {
                        "history_tree": {
                            layer_id: {
                                "error_info": {
                                    "message": f"Error in {next_node}: {str(e)}",
                                    "details": {"node": next_node}
                                }
                            }
                        }
                    })
                    break
                
                node_data = self.state["history_tree"].get(layer_id)
                if node_data and node_data.get("error_info"):
                    self._log(f"  ERROR: {node_data['error_info'].get('message', 'Unknown error')}")
                    break
                
                gc.collect()
            
            self._log(f"Completed layer: {layer_id}")
            
        finally:
            update = r_remove_processing_id(layer_id, self.state)
            self.state = merge_updates(self.state, update)
            self._update_visualization()
    
    def _finalize_remaining_layers(self) -> None:
        """
        [Revision 12] [Revision 36] Force finalize any unfinished layers on termination.
        
        This ensures all layers produce elements even when terminated
        due to max_depth, max_layers, or LLM budget limits.
        
        [Revision 36] Now includes action_type change in visualization.
        """
        unfinished = _get_unfinished_layers(self.state)
        
        if not unfinished:
            self._log("No unfinished layers to finalize")
            return
        
        self._log(f"\n{'='*60}")
        self._log(f"Force finalizing {len(unfinished)} unfinished layers...")
        self._log(f"Layers: {unfinished}")
        
        for layer_id in unfinished:
            self._log(f"  -> Force finalizing: {layer_id}")
            try:
                update = _force_finalize_layer(layer_id, self.state)
                self.state = merge_updates(self.state, update)
                self._log(f"     Done: {layer_id}")
            except Exception as e:
                self._log(f"     ERROR finalizing {layer_id}: {e}")
        
        self._update_visualization(force=True)
    
    def run(self) -> Dict[str, Any]:
        """Run the episode decomposition (sequential mode)."""
        self.start_time = time.time()
        self._log("\n" + "="*60)
        self._log("Starting URLD Episode Run (Sequential Mode - FIFO Queue)")
        self._log("="*60)
        
        self._update_visualization(force=True)
        
        iteration = 0
        max_iterations = self.state.get("max_layers", 100) * 10
        
        while iteration < max_iterations:
            iteration += 1
            
            should_term, reason = self._should_terminate()
            if should_term:
                self._log(f"\nTerminating: {reason}")
                # [Revision 12] Force finalize remaining layers before final termination
                self._finalize_remaining_layers()
                break
            
            update = r_dequeue_layer(self.state)
            self.state = merge_updates(self.state, update)
            
            layer_id = self.state.get("current_layer_id")
            if not layer_id:
                # [Revision 24] Check if there are pending retries we need to wait for
                pending = self.state.get("pending_retries", set())
                if pending:
                    self._log(f"[Warning] No layer dequeued but pending retries exist: {pending}")
                    time.sleep(0.1)
                    continue
                
                self._log("[Warning] No layer to process but not terminated - checking state...")
                queue = self.state.get("layer_queue", [])
                processing = self.state.get("processing_ids", [])
                self._log(f"  Queue: {queue}")
                self._log(f"  Processing: {processing}")
                continue
            
            self._process_layer(layer_id)
            
            queue_size = len(self.state.get("layer_queue", []))
            elements_count = len(self.state.get("parsed_elements", []))
            layer_count = self.state.get("layer_count", 0)
            pending_count = len(self.state.get("pending_retries", set()))
            self._log(f"\n[Progress] Queue: {queue_size}, Layers: {layer_count}, Elements: {elements_count}, Pending: {pending_count}")

        
        elapsed = time.time() - self.start_time
        self._log("\n" + "="*60)
        self._log("Episode Complete")
        self._log(f"  Time: {elapsed:.1f}s")
        self._log(f"  Layers processed: {self.state.get('layer_count', 0)}")
        self._log(f"  Elements extracted: {len(self.state.get('parsed_elements', []))}")
        self._log(f"  LLM calls: {self.state.get('llm_call_count', 0)}")
        self._log(f"  Final queue size: {len(self.state.get('layer_queue', []))}")
        self._log(f"  Final processing: {self.state.get('processing_ids', [])}")
        self._log("="*60)
        

        reconstruction_info = {}
        verification_report = ""
        
        try:
            reconstruction_info, verification_report = run_reconstruction_and_verification(
                state=self.state,
                verbose=self.verbose
            )
            
            # Add final_llm_verification as separate field in state
            if reconstruction_info:
                verification_data = _build_final_verification_update(
                    reconstruction_info,
                    verification_report
                )
                self.state = merge_updates(self.state, verification_data)
                
        except Exception as e:
            self._log(f"Reconstruction/Verification failed: {e}")
            import traceback
            traceback.print_exc()



        save_state_to_disk(self.state)
        self._log(f"Saved results to {self.state['episode_dir']}")
        
        if self.visualizer:
            try:
                final_viz_path = self.visualizer.final_update(self.state)
                self._log(f"Final visualization: {final_viz_path}")
            except Exception as e:
                self._log(f"Final visualization failed: {e}")
        
        return {
            "state": self.state,
            "elements": self.state.get("parsed_elements", []),
            "history_tree": self.state.get("history_tree", {}),
            "elapsed_time": elapsed,
        }


# =============================================================================
# Episode Runner (Parallel Mode)
# =============================================================================

class ParallelEpisodeRunner:
    """Manages execution with parallel layer processing using GPU slots."""
    
    def __init__(
        self,
        run_dir: str,
        episode_id: str,
        image_path: str,
        llm_call_limit: int = 100,
        max_depth: int = 5,
        max_layers: int = 100,
        available_gpus: List[int] = None,
        max_parallel_workers: int = 4,
        verbose: bool = True,
        enable_visualization: bool = True,
        visualization_interval: int = 1,
    ):
        self.verbose = verbose
        self.start_time = None
        self.max_parallel = max_parallel_workers
        self.enable_visualization = enable_visualization
        
        reset_enqueued_tracking()
        
        initial_state = initialize_graph_state(
            run_dir=run_dir,
            episode_id=episode_id,
            original_image_path=image_path,
            llm_call_limit=llm_call_limit,
            max_depth=max_depth,
            max_layers=max_layers,
            available_gpus=available_gpus,
            max_parallel_workers=max_parallel_workers,
        )
        
        # [Revision 11] Setup file logging
        global _episode_logger
        _episode_logger = setup_logger(initial_state["episode_dir"])
        
        self.state_manager = StateManager(initial_state)
        
        self.visualizer = None
        if enable_visualization:
            try:
                from .visualizer import TreeVisualizer
                self.visualizer = TreeVisualizer(
                    initial_state["episode_dir"],
                    update_interval=visualization_interval
                )
            except ImportError as e:
                self._log(f"Warning: Visualizer not available: {e}")
        
        self._log(f"Initialized parallel episode {episode_id}")
        self._log(f"  Image: {image_path}")
        self._log(f"  Episode dir: {initial_state['episode_dir']}")
        self._log(f"  Available GPUs: {[s['gpu_id'] for s in initial_state['gpu_slots']]}")
        self._log(f"  Max parallel workers: {max_parallel_workers}")
        self._log(f"  Visualization: {'Enabled' if self.visualizer else 'Disabled'}")
        self._log(f"  Processing Mode: FIFO Queue (BFS)")
        self._log(f"  Log file: {initial_state['episode_dir']}/episode.log")
        self._log(f"  MAX_RETRIES: {MAX_RETRIES}")
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            log(message, "ParallelEpisode")
    
    def _update_visualization(self, force: bool = True):
        """Update tree visualization."""
        if self.visualizer:
            try:
                self.visualizer.update(self.state_manager.state, force=force)
            except Exception as e:
                pass
    
    def _finalize_remaining_layers(self) -> None:
        """
        [Revision 12] [Revision 36] Force finalize any unfinished layers on termination (parallel mode).
        
        [Revision 36] Gets latest state from state_manager after gather completes.
        This ensures newly added children from completed workers are included.
        """
        # [Revision 36] ★ Get latest state AFTER gather completes
        current_state = self.state_manager.state
        
        # Log queue state for debugging
        queue_size = len(current_state.get("layer_queue", []))
        self._log(f"Queue size at termination: {queue_size}")
        
        unfinished = _get_unfinished_layers(current_state)
        
        if not unfinished:
            self._log("No unfinished layers to finalize")
            return
        
        self._log(f"\n{'='*60}")
        self._log(f"Force finalizing {len(unfinished)} unfinished layers...")
        self._log(f"Layers: {unfinished}")
        
        for layer_id in unfinished:
            self._log(f"  -> Force finalizing: {layer_id}")
            try:
                # [Revision 36] Get latest state each iteration
                state = self.state_manager.state
                update = _force_finalize_layer(layer_id, state)
                self.state_manager.update(update)
                self._log(f"     Done: {layer_id}")
            except Exception as e:
                self._log(f"     ERROR finalizing {layer_id}: {e}")
        
        self._update_visualization(force=True)
    
    async def run_async(self) -> Dict[str, Any]:
        """Run the episode decomposition asynchronously with parallel workers."""
        self.start_time = time.time()
        self._log("\n" + "="*60)
        self._log("Starting URLD Episode Run (Parallel Mode - FIFO Queue)")
        self._log("="*60)
        
        self._update_visualization(force=True)
        
        executor = ThreadPoolExecutor(max_workers=self.max_parallel)
        loop = asyncio.get_event_loop()
        
        active_tasks = []
        
        last_log_time = 0
        log_interval = 30.0
        last_viz_time = 0
        viz_interval = 5.0
        
        try:
            while True:
                should_term, reason = self.state_manager.should_terminate()



                if should_term:
                    if active_tasks:
                        await asyncio.gather(*active_tasks, return_exceptions=True)
                        self._log("All active tasks completed")
                        
                        # Added: re-check after gather
                        should_term_recheck, _ = self.state_manager.should_terminate()
                        if not should_term_recheck:
                            self._log("New work available after gather, continuing...")
                            active_tasks = []
                            continue  # Keep looping
                    
                    self._log(f"\nTerminating: {reason}")
                    self._finalize_remaining_layers()
                    break



                
                active_tasks = [t for t in active_tasks if not t.done()]
                
                current_workers = len(active_tasks)
                available_slots = self.max_parallel - current_workers
                
                if available_slots > 0:
                    layers_to_process = self.state_manager.dequeue_layers(available_slots)
                    
                    for layer_id in layers_to_process:
                        self._log(f"Starting worker for {layer_id}")
                        
                        task = loop.run_in_executor(
                            executor,
                            process_layer_worker,
                            self.state_manager,
                            layer_id,
                            self.verbose
                        )
                        active_tasks.append(asyncio.ensure_future(task))
                
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    stats = self.state_manager.get_stats()
                    self._log(f"[Progress] Queue: {stats['queue_size']}, "
                             f"Processing: {stats['processing']}, "
                             f"Elements: {stats['elements']}, "
                             f"Pending retries: {stats['pending_retries']}")
                    last_log_time = current_time
                
                if current_time - last_viz_time >= viz_interval:
                    self._update_visualization()
                    last_viz_time = current_time
                
                await asyncio.sleep(0.1)
                
        finally:
            executor.shutdown(wait=True)
        
        final_state = self.state_manager.state
        elapsed = time.time() - self.start_time
        

        reconstruction_info = {}
        verification_report = ""
        
        try:
            reconstruction_info, verification_report = run_reconstruction_and_verification(
                state=final_state,
                verbose=self.verbose
            )
            
            # Add final_llm_verification as separate field in state
            if reconstruction_info:
                verification_data = _build_final_verification_update(
                    reconstruction_info,
                    verification_report
                )
                self.state_manager.update(verification_data)
                final_state = self.state_manager.state
                
        except Exception as e:
            self._log(f"Reconstruction/Verification failed: {e}")
            import traceback
            traceback.print_exc()


        self._log("\n" + "="*60)
        self._log("Episode Complete")
        self._log(f"  Time: {elapsed:.1f}s")
        self._log(f"  Layers processed: {final_state.get('layer_count', 0)}")
        self._log(f"  Elements extracted: {len(final_state.get('parsed_elements', []))}")
        self._log(f"  LLM calls: {final_state.get('llm_call_count', 0)}")
        self._log("="*60)
        
        save_state_to_disk(final_state)
        self._log(f"Saved results to {final_state['episode_dir']}")
        
        if self.visualizer:
            try:
                final_viz_path = self.visualizer.final_update(final_state)
                self._log(f"Final visualization: {final_viz_path}")
            except Exception as e:
                self._log(f"Final visualization failed: {e}")
        
        return {
            "state": final_state,
            "elements": final_state.get("parsed_elements", []),
            "history_tree": final_state.get("history_tree", {}),
            "elapsed_time": elapsed,
        }
    
    def run(self) -> Dict[str, Any]:
        """Synchronous wrapper for run_async."""
        return asyncio.run(self.run_async())


# =============================================================================
# Convenience Functions
# =============================================================================

def _run_final_verification(
    original_image_path: str,
    reconstructed_image_path: str,
    verbose: bool = True
) -> str:
    """
    Run LLM-based verification comparing original vs reconstructed image.
    
    Args:
        original_image_path: Path to original image
        reconstructed_image_path: Path to reconstructed image
        verbose: Print progress
    
    Returns:
        Verification report as text string
    """
    if verbose:
        log("Running final verification with Gemini...", "Verification")
    
    # Load and encode images
    original_b64 = image_to_b64(original_image_path)
    reconstructed_b64 = image_to_b64(reconstructed_image_path)
    
    # Build prompt
    system_prompt = build_final_verification_system_prompt()
    
    # Initialize LLM (same config as router_vlm.py)
    llm = ChatOpenAI(
        model=os.environ.get("VLM_MODEL", "gemini-3-flash-preview"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=0,
        model_kwargs={"top_p": 1},
        max_retries=3,
        request_timeout=120,
    )
    
    # Prepare messages with both images
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": "Compare these two images and provide your verification report."},
            {"type": "text", "text": "\n\n## Original Image:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{original_b64}"},
            },
            {"type": "text", "text": "\n\n## Reconstructed Image:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{reconstructed_b64}"},
            },
        ]),
    ]
    
    try:
        response = llm.invoke(messages)
        verification_report = response.content.strip()
        
        if verbose:
            log("Final verification completed", "Verification")
        
        return verification_report
        
    except Exception as e:
        error_msg = f"LLM verification failed: {str(e)}"
        if verbose:
            log(error_msg, "Verification")
        return f"## Verification Error\n\n{error_msg}"
    
    finally:
        del llm, original_b64, reconstructed_b64
        gc.collect()


def run_reconstruction_and_verification(
    state: GraphState,
    verbose: bool = True
) -> Tuple[Dict[str, Any], str]:
    """
    Run reconstruction and final verification for an episode.
    """
    from .utils import apply_alpha_mask_to_reconstruction
    from .reconstruction import draw_element_borders  # [NEW]
    
    episode_dir = state.get("episode_dir", ".")
    history_tree = state.get("history_tree", {})
    parsed_elements = state.get("parsed_elements", [])
    
    image_info = state.get("original_image_info", {})
    has_alpha = image_info.get("has_alpha", False)
    alpha_mask_path = image_info.get("alpha_mask_path")
    original_image_path_for_verification = image_info.get("original_image_path", "")
    
    if verbose:
        log("\n" + "="*60, "Reconstruction")
        log("Starting Reconstruction & Final Verification", "Reconstruction")
        if has_alpha:
            log(f"  Original format: RGBA (alpha mask will be restored)", "Reconstruction")
        log("="*60, "Reconstruction")
    
    root_node = history_tree.get("layer_0000", {})
    root_image_path = root_node.get("image_path", "")
    
    if not root_image_path or not Path(root_image_path).exists():
        log(f"ERROR: Root image not found: {root_image_path}", "Reconstruction")
        return {}, "## Verification Error\n\nRoot image not found"
    
    z_order_info = compute_z_order_with_info(history_tree, "layer_0000")
    
    if verbose:
        log(f"Z-order computed: {len(z_order_info)} layers", "Reconstruction")
    
    src_root = get_src_root()
    
    try:
        reconstructed_img = reconstruct_image(
            history_tree=history_tree,
            parsed_elements=parsed_elements,
            root_image_path=root_image_path,
            src_root=src_root,
            verbose=verbose
        )
        
        if has_alpha and alpha_mask_path and Path(alpha_mask_path).exists():
            reconstructed_rgb_path = Path(episode_dir) / "reconstructed_rgb.png"
            reconstructed_img.save(reconstructed_rgb_path)
            
            if verbose:
                log(f"Saved RGB reconstruction: {reconstructed_rgb_path}", "Reconstruction")
                log("Applying original alpha mask...", "Reconstruction")
            
            final_reconstructed_path = Path(episode_dir) / "reconstructed.png"
            apply_alpha_mask_to_reconstruction(
                reconstructed_rgb_path=str(reconstructed_rgb_path),
                alpha_mask_path=alpha_mask_path,
                output_path=str(final_reconstructed_path)
            )
            
            reconstruction_info = {
                "reconstructed_path": str(final_reconstructed_path),
                "reconstructed_rgb_path": str(reconstructed_rgb_path),
                "has_alpha_restored": True,
                "alpha_mask_path": alpha_mask_path,
                "num_elements": len(parsed_elements),
                "num_layers_in_zorder": len(z_order_info),
                "canvas_size": list(reconstructed_img.size),
            }
            
            if verbose:
                log(f"Saved RGBA reconstruction: {final_reconstructed_path}", "Reconstruction")
        else:
            final_reconstructed_path = Path(episode_dir) / "reconstructed.png"
            reconstructed_img.save(final_reconstructed_path)
            
            reconstruction_info = {
                "reconstructed_path": str(final_reconstructed_path),
                "has_alpha_restored": False,
                "num_elements": len(parsed_elements),
                "num_layers_in_zorder": len(z_order_info),
                "canvas_size": list(reconstructed_img.size),
            }
            
            if verbose:
                log(f"Saved reconstructed image: {final_reconstructed_path}", "Reconstruction")
        
        # =============================================
        # [NEW] Save bordered version with contours
        # =============================================
        if verbose:
            log("\nCreating bordered visualization with contours...", "Reconstruction")
        
        bordered_img = draw_element_borders(
            reconstructed_img,
            parsed_elements,
            src_root,
            verbose=verbose
        )
        bordered_path = Path(episode_dir) / "reconstructed_bordered.png"
        bordered_img.save(bordered_path)
        
        reconstruction_info["bordered_path"] = str(bordered_path)
        
        if verbose:
            log(f"Saved bordered image: {bordered_path}", "Reconstruction")
        # =============================================
        
    except Exception as e:
        log(f"ERROR during reconstruction: {e}", "Reconstruction")
        import traceback
        traceback.print_exc()
        return {}, f"## Verification Error\n\nReconstruction failed: {str(e)}"
    
    # 4. Save z-order info
    z_order_path = Path(episode_dir) / "z_order.json"
    with open(z_order_path, "w", encoding="utf-8") as f:
        json.dump(z_order_info, f, ensure_ascii=False, indent=2)
    
    # 5. Final verification (disabled — not needed for evaluation)
    # verification_report = _run_final_verification(
    #     original_image_path=original_image_path_for_verification,
    #     reconstructed_image_path=str(final_reconstructed_path),
    #     verbose=verbose
    # )
    verification_report = "## Final Verification\n\nSkipped (disabled)."

    return reconstruction_info, verification_report


def _build_final_verification_update(
    reconstruction_info: Dict[str, Any],
    verification_report: str,
) -> Dict[str, Any]:
    """
    Build state update with verification results as a separate field in history_tree.
    """
    return {
        "history_tree": {  # Must be nested inside history_tree!
            "final_llm_verification": {
                "verified_at": datetime.now().isoformat(),
                "reconstruction_info": reconstruction_info,
                "verification_report": verification_report,
            }
        }
    }


def run_episode(
    image_path: str,
    output_dir: str = "./urld_output",
    episode_id: str = None,
    available_gpus: List[int] = None,
    parallel: bool = False,
    max_parallel_workers: int = 4,
    enable_visualization: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Run URLD decomposition on a single image."""
    if episode_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(image_path).stem
        episode_id = f"{image_name}_{timestamp}"
    
    if parallel:
        runner = ParallelEpisodeRunner(
            run_dir=output_dir,
            episode_id=episode_id,
            image_path=image_path,
            available_gpus=available_gpus,
            max_parallel_workers=max_parallel_workers,
            enable_visualization=enable_visualization,
            **kwargs
        )
    else:
        runner = EpisodeRunner(
            run_dir=output_dir,
            episode_id=episode_id,
            image_path=image_path,
            available_gpus=available_gpus,
            enable_visualization=enable_visualization,
            **kwargs
        )
    
    return runner.run()


def run_batch(
    image_paths: List[str],
    output_dir: str = "./urld_output",
    **kwargs
) -> List[Dict[str, Any]]:
    """Run URLD decomposition on multiple images."""
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\n[Batch] Processing {i+1}/{len(image_paths)}: {image_path}")
        
        result = run_episode(image_path, output_dir, **kwargs)
        results.append({
            "image_path": image_path,
            "success": True,
            **result
        })
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import torch.multiprocessing as mp
    
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser(description="URLD Episode Runner")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="./urld_output", help="Output directory")
    parser.add_argument("--episode_id", type=str, default=None, help="Episode ID")
    parser.add_argument("--llm_limit", type=int, default=100, help="LLM call limit")
    parser.add_argument("--max_depth", type=int, default=5, help="Max recursion depth")
    parser.add_argument("--max_layers", type=int, default=100, help="Max total layers")
    parser.add_argument("--gpus", type=str, default=None, 
                       help="Available GPUs (comma-separated, e.g., '0,1,4,5')")
    parser.add_argument("--parallel", action="store_true", help="Use parallel execution")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel workers")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    available_gpus = None
    if args.gpus:
        available_gpus = [int(x.strip()) for x in args.gpus.split(",")]
    
    result = run_episode(
        image_path=args.image,
        output_dir=args.output,
        episode_id=args.episode_id,
        llm_call_limit=args.llm_limit,
        max_depth=args.max_depth,
        max_layers=args.max_layers,
        available_gpus=available_gpus,
        parallel=args.parallel,
        max_parallel_workers=args.workers,
        verbose=not args.quiet,
        enable_visualization=not args.no_viz,
    )
    
    print(f"\n[Result] Extracted {len(result['elements'])} elements")
    print(f"[Result] Output: {result['state']['episode_dir']}")