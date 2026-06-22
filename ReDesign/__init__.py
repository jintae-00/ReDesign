# ReDesign/__init__.py
"""
URLD (Unified Recursive Layer Decomposition) Pipeline

A recursive queue-based agentic pipeline for decomposing images into editable elements.

UPDATED: Changed from stack (LIFO) to queue (FIFO) for layer processing.
This ensures breadth-first traversal of the decomposition tree.

Usage:
    from ReDesign import run_episode, initialize_graph_state
    
    state = run_episode(
        image_path="path/to/image.png",
        run_dir="./runs",
        episode_id="ep_001",
    )
"""

from .state import (
    GraphState,
    LayerNode,
    initialize_graph_state,
    save_state_to_disk,
    load_state_from_disk,
)

# Backward compatibility alias
create_initial_state = initialize_graph_state


def save_history_tree(state):
    return save_state_to_disk(state)


def save_parsed_elements(state):
    return save_state_to_disk(state)


from .episode_run import (
    run_episode,
    run_batch,
    EpisodeRunner,
)

from .build_graph import (
    build_graph,
    build_simple_graph,
)

from .registry import (
    get_tool,
    list_tools,
    register_tool,
)

from .visualizer import (
    visualize_history_tree,
    visualize_from_json,
    TreeVisualizer,
)

__version__ = "0.4.0"  # Updated version for queue-based processing

__all__ = [
    # State
    "GraphState",
    "LayerNode",
    "initialize_graph_state",
    "create_initial_state",
    "save_state_to_disk",
    "load_state_from_disk",
    "save_history_tree",
    "save_parsed_elements",
    
    # Run
    "run_episode",
    "run_batch",
    "EpisodeRunner",
    
    # Graph
    "build_graph",
    "build_simple_graph",
    
    # Tools
    "get_tool",
    "list_tools",
    "register_tool",
    
    # Visualizer
    "visualize_history_tree",
    "visualize_from_json",
    "TreeVisualizer",
]