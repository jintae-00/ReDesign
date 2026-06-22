# REDESIGN/build_graph.py
"""
Build Graph - Wires all nodes into a LangGraph structure

The URLD pipeline uses a queue-based execution model where:
1. router_vlm decides the action for each layer
2. Tool nodes execute based on the planned sequence
3. stack_manager creates child layers and enqueues to processing queue
4. The process continues until the queue is empty

UPDATED: Changed from stack (LIFO) to queue (FIFO) terminology.
"""
from __future__ import annotations
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from .state import GraphState

# Import all nodes
from .nodes import router_vlm as n_router_vlm
from .nodes import vlm_front_pick as n_vlm_front_pick
from .nodes import qwen_layered as n_qwen_layered
from .nodes import split_cca as n_split_cca
from .nodes import stack_manager as n_stack_manager

from .nodes import detect_ocr as n_detect_ocr
from .nodes import detect_gdino as n_detect_gdino

from .nodes import seg_hisam as n_seg_hisam
from .nodes import seg_sam2_bbox as n_seg_sam2_bbox

from .nodes import inpaint_lama as n_inpaint_lama
from .nodes import inpaint_oc as n_inpaint_oc
from .nodes import nanobanana as n_nanobanana

from .nodes import fontstyle as n_fontstyle
from .nodes import vtracer as n_vtracer

from .nodes import finalize_text as n_finalize_text
from .nodes import finalize_obj as n_finalize_obj


# Node name to module mapping
NODE_MODULES = {
    "router_vlm": n_router_vlm,
    "vlm_front_pick": n_vlm_front_pick,
    "qwen_layered": n_qwen_layered,
    "split_cca": n_split_cca,
    "stack_manager": n_stack_manager,
    "ocr": n_detect_ocr,
    "gdino": n_detect_gdino,
    "hisam": n_seg_hisam,
    "sam2_bbox": n_seg_sam2_bbox,
    "lama": n_inpaint_lama,
    "objectclear": n_inpaint_oc,
    "nanobanana": n_nanobanana,
    "fontstyle": n_fontstyle,
    "vtracer": n_vtracer,
    "finalize_text": n_finalize_text,
    "finalize_obj": n_finalize_obj,
}


def _get_next_node(state: GraphState) -> str:
    """
    Determine the next node to execute based on current state.
    
    Logic:
    1. If node_queue is not empty, return the first node in queue
    2. If node_queue is empty but layer_queue is not empty, go to router_vlm
    3. If both are empty, end
    
    UPDATED: Changed layer_stack to layer_queue
    """
    layer_id = state.get("current_layer_id")
    
    if layer_id:
        tree = state.get("history_tree", {})
        if layer_id in tree:
            queue = tree[layer_id].get("node_queue") or []
            if queue:
                next_node = queue[0]
                if next_node in NODE_MODULES:
                    return next_node
                else:
                    print(f"[Warning] Unknown node in queue: {next_node}")
    
    # [UPDATED] Check layer_queue instead of layer_stack
    queue = state.get("layer_queue", [])
    if queue:
        return "router_vlm"
    
    return "end"


def route_from_router(state: GraphState) -> str:
    """Route after router_vlm decides action"""
    return _get_next_node(state)


def route_after_tool(state: GraphState) -> str:
    """Route after any tool node completes"""
    return _get_next_node(state)


def build_graph() -> StateGraph:
    """
    Build the URLD pipeline graph.
    
    Graph Structure:
    - entry: router_vlm (dequeues layer from queue and decides action)
    - All tool nodes can route to:
      - Another tool node (from node_queue)
      - router_vlm (if node_queue empty but layer_queue not empty)
      - END (if both node_queue and layer_queue empty)
    """
    g = StateGraph(GraphState)
    
    # Add all nodes
    for name, module in NODE_MODULES.items():
        g.add_node(name, module.node)
    
    # Set entry point
    g.set_entry_point("router_vlm")
    
    # Build routing mapping
    all_node_names = list(NODE_MODULES.keys())
    mapping = {name: name for name in all_node_names}
    mapping["end"] = END
    
    # Add conditional edges from each node
    for name in all_node_names:
        g.add_conditional_edges(name, route_after_tool, mapping)
    
    return g.compile(checkpointer=None)


def build_simple_graph() -> StateGraph:
    """
    Build a simplified graph for testing.
    
    This version just routes between:
    - router_vlm -> appropriate tool based on action
    - tools -> stack_manager or finalizer
    - stack_manager -> router_vlm (if more layers) or END
    """
    g = StateGraph(GraphState)
    
    # Core nodes only
    g.add_node("router_vlm", n_router_vlm.node)
    g.add_node("stack_manager", n_stack_manager.node)
    
    # Tool nodes
    g.add_node("vlm_front_pick", n_vlm_front_pick.node)
    g.add_node("qwen_layered", n_qwen_layered.node)
    g.add_node("split_cca", n_split_cca.node)
    g.add_node("ocr", n_detect_ocr.node)
    g.add_node("gdino", n_detect_gdino.node)
    g.add_node("hisam", n_seg_hisam.node)
    g.add_node("sam2_bbox", n_seg_sam2_bbox.node)
    g.add_node("lama", n_inpaint_lama.node)
    g.add_node("finalize_text", n_finalize_text.node)
    g.add_node("finalize_obj", n_finalize_obj.node)
    g.add_node("fontstyle", n_fontstyle.node)
    g.add_node("vtracer", n_vtracer.node)
    
    g.set_entry_point("router_vlm")
    
    # Routing
    all_nodes = [
        "router_vlm", "stack_manager",
        "vlm_front_pick", "qwen_layered", "split_cca",
        "ocr", "gdino", "hisam", "sam2_bbox", "lama",
        "finalize_text", "finalize_obj", "fontstyle", "vtracer",
    ]
    mapping = {name: name for name in all_nodes}
    mapping["end"] = END
    
    for name in all_nodes:
        g.add_conditional_edges(name, route_after_tool, mapping)
    
    return g.compile(checkpointer=None)