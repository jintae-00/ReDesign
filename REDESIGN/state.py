# REDESIGN/state.py
"""
GraphState & LayerNode definitions for Unified Recursive Layer Decomposition (URLD)

Verification System Types:
- ChildValidationResult: validation result for an individual child
- CrossChildDuplicate: cross-child duplicate detection result
- VerificationAttempt: record of an entire verification attempt
- LayerNode gains the verification_attempts and rejected_child_indices fields
"""
from __future__ import annotations
from typing import TypedDict, Optional, List, Dict, Any
from pathlib import Path
import uuid
import numpy as np
import json
import threading
import shutil


# =============================================================================
# Verification System TypedDicts
# =============================================================================

class ChildValidationResult(TypedDict, total=False):
    """Validation result for an individual child layer"""
    index: int                          # Child index
    child_image_path: str               # Child image path
    status: str                         # "VALID" | "INVALID"

    # 3-check evaluation
    hallucination_check: str            # "PASS" | "FAIL"
    hallucination_detail: Optional[str] # Details on failure

    redundancy_check: str               # "PASS" | "FAIL"
    redundancy_detail: Optional[str]    # Which child it duplicates

    integrity_check: str                # "PASS" | "FAIL"
    integrity_detail: Optional[str]     # Color distortion, blur, etc.

    # Context
    context: Optional[str]              # Child content as understood by the VLM
    reason: Optional[str]               # Final decision rationale


class CrossChildDuplicate(TypedDict, total=False):
    """Cross-child duplicate object detection result"""
    object_description: str             # Description of the duplicated object
    kept_child_index: int               # Index of the child to keep
    rejected_child_indices: List[int]   # Indices of the children to reject
    reason: str                         # Decision rationale


class VerificationAttempt(TypedDict, total=False):
    """Full record of a single verification attempt"""
    attempt_number: int                 # Attempt number (starting from 1)
    layer_id: str                       # Parent layer ID
    action_type: str                    # Fork_Qwen, Split_DetSeg, etc.
    tool_sequence: List[str]            # Executed tool sequence

    # All child images (including rejected ones)
    child_image_paths: List[str]

    # Per-child analysis results
    children_analysis: List[ChildValidationResult]

    # Valid/invalid child indices
    valid_children_indices: List[int]
    invalid_children_indices: List[int]

    # Coverage assessment (COMPLETE | INCOMPLETE)
    coverage_check: str
    coverage_reason: Optional[str]

    # Final decision (computed value - PROCEED | PROCEED_FILTERED | RETRY)
    decision: str

    timestamp: str


# =============================================================================
# Tool Outputs TypedDict
# =============================================================================

class ToolOutputs(TypedDict, total=False):
    """Tool execution outputs stored per LayerNode"""
    # Fork tools
    qwen_layered: Optional[Dict[str, Any]]
    split_cca: Optional[Dict[str, Any]]
    
    # VLM labeling
    vlm_front_pick: Optional[Dict[str, Any]]
    
    # Detection (list to support multiple calls)
    detect: Optional[List[Dict[str, Any]]]
    
    # Segmentation (list to support multiple calls)
    segment: Optional[List[Dict[str, Any]]]
    
    # Refinement (list - each refine step appended)
    refine: Optional[List[Dict[str, Any]]]
    
    # Finalization
    fontstyle: Optional[Dict[str, Any]]
    vtracer: Optional[Dict[str, Any]]
    
    # Verifier output (latest)
    verifier: Optional[Dict[str, Any]]


# =============================================================================
# Enhanced LayerNode with Verification Fields
# =============================================================================

class LayerNode(TypedDict, total=False):
    """Persistent history node for each layer in decomposition tree"""
    layer_id: str
    parent_id: Optional[str]
    depth: int
    image_path: str
    image_context: Optional[str]
    action_reasoning: Optional[str]
    action_type: Optional[str]
    planned_tool_sequence: Optional[List[str]]
    node_queue: Optional[List[str]]
    param_qwen_len: Optional[int]
    param_is_photo: Optional[bool]
    param_inpaint_remainder: Optional[bool]
    param_nanobanana_instruction: Optional[str]
    tool_outputs: Optional[ToolOutputs]
    children_ids: Optional[List[str]]
    parsed_elements: Optional[List[Dict[str, Any]]]
    error_info: Optional[Dict[str, Any]]
    retry_count: Optional[int]
    
    # Verification tracking
    verification_attempts: Optional[List[VerificationAttempt]]  # Record of all attempts
    verification_status: Optional[str]  # "PROCEED" | "PROCEED_FILTERED" | "RETRY" |
    rejected_child_indices: Optional[List[int]]  # Child indices rejected on PARTIAL
    failed_attempts: Optional[List[Dict[str, Any]]]  # Failure records used for router retry

    # Temporary children tracking (before verification)
    _temp_child_ids: Optional[List[str]]  # Temp children awaiting verification
    _pending_verification: Optional[bool]  # Awaiting-verification flag


class GPUSlot(TypedDict):
    """GPU slot status"""
    gpu_id: int
    available: bool
    layer_id: Optional[str]


# =============================================================================
# Enhanced GraphState
# =============================================================================

class GraphState(TypedDict, total=False):
    """Global state for the URLD agentic pipeline"""
    run_id: str
    run_dir: str
    episode_id: str
    episode_dir: str
    layer_count: int
    layer_queue: List[str]  # FIFO queue
    processing_ids: List[str]
    history_tree: Dict[str, LayerNode]
    parsed_elements: List[Dict[str, Any]]
    gpu_slots: List[GPUSlot]
    max_parallel_workers: int
    root_layer_id: str
    root_image_path: str
    current_layer_id: Optional[str]
    max_depth: int
    max_layers: int
    llm_call_count: int
    llm_call_limit: int
    
    # Retry support
    pending_retries: set

    # Current verification context
    current_verification_attempt: Optional[VerificationAttempt]

    original_image_info: Dict[str, Any]  # Result of analyze_and_convert_image


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


# =============================================================================
# Thread-Safe Layer ID Generation
# =============================================================================

_layer_id_lock = threading.Lock()

def generate_layer_id(state: GraphState, prefix: str = "layer") -> str:
    """Generate a unique layer ID with thread safety."""
    global _layer_id_lock
    
    with _layer_id_lock:
        count = state.get("layer_count", 0)
        unique_suffix = uuid.uuid4().hex[:4]
        return f"{prefix}_{count:04d}_{unique_suffix}"


def create_root_layer_node(layer_id: str, image_path: str, depth: int = 0) -> LayerNode:
    """Create the root LayerNode"""
    return LayerNode(
        layer_id=layer_id, parent_id=None, depth=depth, image_path=image_path,
        image_context=None, action_reasoning=None, action_type=None,
        planned_tool_sequence=None, node_queue=None, param_qwen_len=None,
        param_is_photo=None, param_inpaint_remainder=None,
        param_nanobanana_instruction=None, tool_outputs={}, children_ids=None,
        parsed_elements=None, error_info=None, retry_count=0,
        verification_attempts=[], verification_status="pending",
        rejected_child_indices=None, failed_attempts=[],
        _temp_child_ids=None, _pending_verification=False,
    )


def create_child_layer_node(layer_id: str, parent_id: str, image_path: str, depth: int) -> LayerNode:
    """Create a child LayerNode"""
    return LayerNode(
        layer_id=layer_id, parent_id=parent_id, depth=depth, image_path=image_path,
        image_context=None, action_reasoning=None, action_type=None,
        planned_tool_sequence=None, node_queue=None, param_qwen_len=None,
        param_is_photo=None, param_inpaint_remainder=None,
        param_nanobanana_instruction=None, tool_outputs={}, children_ids=None,
        parsed_elements=None, error_info=None, retry_count=0,
        verification_attempts=[], verification_status="pending",
        rejected_child_indices=None, failed_attempts=[],
        _temp_child_ids=None, _pending_verification=False,
    )


# =============================================================================
# Temporary Child Node (for verification visualization)
# =============================================================================

def create_temp_child_node(
    temp_id: str,
    parent_id: str,
    image_path: str,
    depth: int,
    child_index: int,
    attempt_number: int
) -> LayerNode:
    """
    Create a temporary child node for verification.
    
    These nodes are created BEFORE verification to ensure they appear
    in the tree visualization regardless of verification result.
    """
    return LayerNode(
        layer_id=temp_id,
        parent_id=parent_id,
        depth=depth,
        image_path=image_path,
        image_context=f"[Pending Verification] Child {child_index}",
        action_reasoning=f"Awaiting verification (attempt #{attempt_number})",
        action_type="_TempChild",  # Special type for visualization
        planned_tool_sequence=[],
        node_queue=[],
        tool_outputs={},
        children_ids=None,
        parsed_elements=None,
        error_info=None,
        retry_count=0,
        verification_attempts=[],
        verification_status="pending",
        _temp_child_ids=None,
        _pending_verification=True,
        # Extra fields for temp tracking
        _is_temporary=True,
        _child_index=child_index,
        _attempt_number=attempt_number,
    )


# =============================================================================
# State Initialization
# =============================================================================

def initialize_graph_state(
    run_dir: str,
    episode_id: str,
    original_image_path: str,
    llm_call_limit: int = 100,
    max_depth: int = 5,
    max_layers: int = 100,
    available_gpus: List[int] = None,
    max_parallel_workers: int = 4,
) -> GraphState:
    """
    Initialize GraphState for a new episode.
    
    Now handles RGBA images:
    - Analyzes input image for alpha channel
    - Saves alpha mask if present
    - Converts to RGB with white background for processing
    """
    from .utils import analyze_and_convert_image  # Added import
    
    run_dir = Path(run_dir)
    episode_dir = run_dir / "episodes" / episode_id
    layers_dir = episode_dir / "layers"
    elements_dir = episode_dir / "elements"
    
    layers_dir.mkdir(parents=True, exist_ok=True)
    elements_dir.mkdir(parents=True, exist_ok=True)
    
    root_layer_id = "layer_0000"
    root_layer_dir = layers_dir / root_layer_id
    root_layer_dir.mkdir(parents=True, exist_ok=True)
    
    # Key change: analyze and convert the image
    image_info = analyze_and_convert_image(
        image_path=original_image_path,
        output_dir=str(root_layer_dir),
        background_color=(255, 255, 255)
    )
    
    # The root layer uses the RGB-converted image
    root_image_path = image_info["rgb_image_path"]

    # Copy the original image into episode_dir (for reference)
    original_copy_path = episode_dir / "original_input.png"
    shutil.copy(original_image_path, original_copy_path)
    
    root_node = create_root_layer_node(layer_id=root_layer_id, image_path=str(root_image_path), depth=0)
    history_tree = {root_layer_id: root_node}
    
    if available_gpus is None:
        available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    
    gpu_slots = [GPUSlot(gpu_id=gpu_id, available=True, layer_id=None) for gpu_id in available_gpus]
    
    return GraphState(
        run_id=str(uuid.uuid4())[:8], run_dir=str(run_dir), episode_id=episode_id,
        episode_dir=str(episode_dir), layer_count=1,
        layer_queue=[root_layer_id],
        processing_ids=[], history_tree=history_tree, parsed_elements=[],
        gpu_slots=gpu_slots, max_parallel_workers=max_parallel_workers,
        root_layer_id=root_layer_id, root_image_path=str(root_image_path),
        current_layer_id=None, max_depth=max_depth, max_layers=max_layers,
        llm_call_count=0, llm_call_limit=llm_call_limit,
        pending_retries=set(),
        current_verification_attempt=None,
        # Store alpha information
        original_image_info=image_info,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def get_available_gpu(state: GraphState) -> Optional[int]:
    """Get an available GPU ID"""
    for slot in state.get("gpu_slots", []):
        if slot.get("available", False):
            return slot["gpu_id"]
    return None


def count_available_gpus(state: GraphState) -> int:
    """Count number of available GPU slots"""
    return sum(1 for slot in state.get("gpu_slots", []) if slot.get("available", False))


def get_available_gpu_ids(state: GraphState) -> List[int]:
    """Get list of all currently available GPU IDs"""
    return [slot["gpu_id"] for slot in state.get("gpu_slots", []) if slot.get("available", False)]


def save_state_to_disk(state: GraphState) -> None:
    """Save history_tree and parsed_elements to disk.

    For an episode where a fatal OCR error occurred (_ocr_fatal_error_count > 0),
    parse.json is not saved, so the skip_completed logic will pick it up for re-runs.
    history_tree.json is always saved for debugging.
    """
    episode_dir = Path(state["episode_dir"])

    history_path = episode_dir / "history_tree.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(state["history_tree"], f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    # If there was a fatal OCR error, do not save parse.json
    ocr_fatal_count = state.get("_ocr_fatal_error_count", 0)
    if ocr_fatal_count > 0:
        print(
            f"[save_state] SKIPPING parse.json — "
            f"{ocr_fatal_count} OCR fatal error(s) detected in episode "
            f"{state.get('episode_id', '?')}"
        )
        return

    parse_path = episode_dir / "parse.json"
    parse_doc = {
        "episode_id": state["episode_id"],
        "root_image": state["root_image_path"],
        "elements": state["parsed_elements"],
    }
    with open(parse_path, "w", encoding="utf-8") as f:
        json.dump(parse_doc, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def load_state_from_disk(episode_dir: str) -> Dict[str, Any]:
    """Load history_tree and parsed_elements from disk"""
    episode_dir = Path(episode_dir)
    result = {}
    
    history_path = episode_dir / "history_tree.json"
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            result["history_tree"] = json.load(f)
    
    parse_path = episode_dir / "parse.json"
    if parse_path.exists():
        with open(parse_path, "r", encoding="utf-8") as f:
            result["parse_doc"] = json.load(f)
    
    return result