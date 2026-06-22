# ReDesign/visualizer.py

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
import textwrap

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


# =============================================================================
# Color Schemes
# =============================================================================

# Node background colors by action type
ACTION_COLORS = {
    "Fork_Qwen": "#4B0082",      # Indigo
    "Router_VLM": "#8A2BE2",     # BlueViolet
    "Split_DetSeg": "#008000",   # Green
    "Split_Text": "#00CED1",     # DarkTurquoise
    "Split_CCA": "#2E8B57",      # SeaGreen
    "Finalize_Text": "#FFA500",  # Orange
    "Finalize_Obj": "#FFD700",   # Gold
    "Processing": "#FF0000",     # Red (Bright)
    "Pending": "#D3D3D3",        # Light Gray
    "Error": "#8B0000",          # Dark Red
    # Virtual/Temp node types
    "_Element_Obj": "#32CD32",   # LimeGreen
    "_TempChild": "#A9A9A9",     # DarkGray
    "_ValidChild": "#228B22",    # ForestGreen
    "_InvalidChild": "#DC143C",  # Crimson
    "_RejectedChild": "#9932CC", # DarkOrchid (Purple)
}

# Verification status colors for badges
VERIFICATION_BADGE_COLORS = {
    "PROCEED": "#228B22",        # ForestGreen - success
    "PROCEED_FILTERED": "#FF8C00", # DarkOrange - partial success
    "RETRY": "#DC143C",          # Crimson - failure/retry
    "VALID": "#228B22",          # ForestGreen
    "INVALID": "#DC143C",        # Crimson
    "REJECTED": "#9932CC",       # DarkOrchid
    "pending": "#A9A9A9",        # Gray
    "NODE_ERROR": "#FF4500",     # OrangeRed
    "CUDA_ERROR": "#8B0000",     # DarkRed
    "GPU_TIMEOUT": "#FF8C00",    # DarkOrange
}

# Border styles by verification status
VERIFICATION_BORDER_STYLES = {
    "PROCEED": {"color": "#228B22", "linewidth": 3.0, "linestyle": "-"},
    "PROCEED_FILTERED": {"color": "#FF8C00", "linewidth": 3.0, "linestyle": "-"},
    "RETRY": {"color": "#DC143C", "linewidth": 3.0, "linestyle": "--"},
    "VALID": {"color": "#228B22", "linewidth": 3.0, "linestyle": "-"},
    "INVALID": {"color": "#DC143C", "linewidth": 3.0, "linestyle": "-"},
    "REJECTED": {"color": "#9932CC", "linewidth": 3.0, "linestyle": "--"},
    "pending": {"color": "#A9A9A9", "linewidth": 1.5, "linestyle": ":"},
    "NODE_ERROR": {"color": "#FF4500", "linewidth": 3.0, "linestyle": "-."},
    "CUDA_ERROR": {"color": "#8B0000", "linewidth": 4.0, "linestyle": "-"},
    "GPU_TIMEOUT": {"color": "#FF8C00", "linewidth": 3.0, "linestyle": ":"},
}

VERIFICATION_SYMBOLS = {
    "PROCEED": "✓",
    "PROCEED_FILTERED": "⚠",
    "RETRY": "✗",
    "VALID": "✓",
    "INVALID": "✗",
    "REJECTED": "↺",
    "pending": "?",
    "NODE_ERROR": "⚠",
    "CUDA_ERROR": "⛔",
    "GPU_TIMEOUT": "⏱",
}


# =============================================================================
# Layout Constants
# =============================================================================

NODE_WIDTH_INCHES = 2.5
NODE_HEIGHT_INCHES = 3.5
TEMP_NODE_WIDTH_INCHES = 2.0
TEMP_NODE_HEIGHT_INCHES = 2.5
HORIZONTAL_GAP_INCHES = 0.8
VERTICAL_GAP_INCHES = 1.2
THUMBNAIL_MAX_SIZE = (200, 200)
IMAGE_RATIO = 0.40
TEXT_RATIO = 0.60
TEXT_FONTSIZE = 7
TEXT_WRAP_WIDTH = 35
OUTPUT_DPI = 100


# =============================================================================
# Helper Functions
# =============================================================================

def _get_node_error_status(node_data: Dict[str, Any]) -> Optional[str]:
    """
    Check if node has an execution error and return error type.
    
    Returns:
        - "NODE_ERROR": General node execution error
        - "GPU_TIMEOUT": GPU acquisition timeout
        - "CUDA_ERROR": CUDA/GPU runtime error
        - None: No error
    """
    if not node_data:
        return None
    
    error_info = node_data.get("error_info")
    if not error_info:
        return None
    
    details = error_info.get("details") or {}
    error_type = details.get("type", "")
    
    if error_type == "gpu_timeout":
        return "GPU_TIMEOUT"
    elif error_type == "cuda_error":
        return "CUDA_ERROR"
    elif error_type in ["node_execution_error", "general_error"]:
        return "NODE_ERROR"
    
    # Fallback: if error_info exists but no specific type
    if error_info.get("message"):
        return "NODE_ERROR"
    
    return None

def _has_failed_attempts(node_data: Dict[str, Any]) -> bool:
    """Check if node has any failed attempts in history."""
    if not node_data:
        return False
    failed_attempts = node_data.get("failed_attempts") or []
    return len(failed_attempts) > 0


def _is_error_retry_child(node_data: Dict[str, Any]) -> bool:
    """Check if this node is a retry child created after node error."""
    if not node_data:
        return False
    
    layer_id = node_data.get("layer_id", "")
    image_context = node_data.get("image_context", "")
    
    # Check ID pattern or context
    if "err_retry" in str(layer_id):
        return True
    if "[Node Error Retry" in str(image_context):
        return True
    
    return False

def _format_failed_attempts_summary(failed_attempts: List[Dict[str, Any]]) -> str:
    """Format failed attempts as a compact summary for display."""
    if not failed_attempts:
        return ""
    
    lines = []
    for i, attempt in enumerate(failed_attempts[-3:]):  # Show last 3 attempts
        error_type = attempt.get("error_type", "?")
        failed_node = attempt.get("failed_node", "?")
        error_msg = attempt.get("error_message", "?")
        action_type = attempt.get("action_type", "?")
        
        # Truncate error message
        if len(error_msg) > 40:
            error_msg = error_msg[:37] + "..."
        
        symbol = "⚠" if error_type == "node_execution_error" else "⛔"
        lines.append(f"{symbol} #{i+1}: {failed_node} ({action_type})")
        lines.append(f"   → {error_msg}")
    
    if len(failed_attempts) > 3:
        lines.insert(0, f"[{len(failed_attempts)} total errors, showing last 3]")
    
    return "\n".join(lines)


def _format_node_error_info(node_data: Dict[str, Any]) -> str:
    """Format current node error info for display."""
    error_info = node_data.get("error_info")
    if not error_info:
        return ""
    
    message = error_info.get("message", "Unknown error")
    details = error_info.get("details") or {}
    failed_node = details.get("node", "?")
    error_type = details.get("type", "general")
    retry_child = details.get("retry_child", "")
    
    # Truncate message
    if len(message) > 60:
        message = message[:57] + "..."
    
    lines = [
        f"⚠ ERROR in: {failed_node}",
        f"Type: {error_type}",
        f"Msg: {message}",
    ]
    
    if retry_child:
        lines.append(f"→ Retry: {retry_child}")
    
    return "\n".join(lines)


def _draw_node_error_badge(ax, x: float, y: float, node_data: Dict[str, Any], node_width: float):
    """Draw node error badge in top-left corner."""
    node_error_status = _get_node_error_status(node_data)
    
    if not node_error_status:
        return
    
    badge_color = VERIFICATION_BADGE_COLORS.get(node_error_status, "#FF4500")
    symbol = VERIFICATION_SYMBOLS.get(node_error_status, "⚠")
    
    # Get failed node name
    error_info = node_data.get("error_info") or {}
    details = error_info.get("details") or {}
    failed_node = details.get("node", "?")
    
    badge_text = f"{symbol} {failed_node}"
    
    # Position: top-left corner
    badge_x = x - node_width / 2 + 0.1
    badge_y = y + NODE_HEIGHT_INCHES / 2 - 0.1
    
    ax.text(
        badge_x, badge_y,
        badge_text,
        fontsize=6,
        fontweight='bold',
        ha='left',
        va='top',
        color='white',
        bbox=dict(
            boxstyle='round,pad=0.2',
            facecolor=badge_color,
            edgecolor='none',
            alpha=0.9
        ),
        zorder=10
    )

def _draw_failed_attempts_indicator(ax, x: float, y: float, failed_attempts: List[Dict], node_width: float):
    """Draw small indicator showing number of failed attempts."""
    if not failed_attempts:
        return
    
    count = len(failed_attempts)
    
    # Position: bottom-left corner
    indicator_x = x - node_width / 2 + 0.1
    indicator_y = y - NODE_HEIGHT_INCHES / 2 + 0.15
    
    ax.text(
        indicator_x, indicator_y,
        f"⚠×{count}",
        fontsize=5,
        fontweight='bold',
        ha='left',
        va='bottom',
        color='white',
        bbox=dict(
            boxstyle='round,pad=0.15',
            facecolor='#FF8C00',  # DarkOrange
            edgecolor='none',
            alpha=0.8
        ),
        zorder=10
    )




def _safe_str(value: Any, default: str = "") -> str:
    """Safely convert value to string, handling None."""
    if value is None:
        return default
    return str(value)


def _safe_startswith(value: Any, prefix: str) -> bool:
    """Safely check if string starts with prefix, handling None."""
    if value is None:
        return False
    if not isinstance(value, str):
        return False
    return value.startswith(prefix)


def _create_thumbnail(image_path: str, size: Tuple[int, int] = THUMBNAIL_MAX_SIZE) -> Optional[np.ndarray]:
    """Create thumbnail with checkerboard background."""
    try:
        # Null check for image_path
        if not image_path or not isinstance(image_path, str):
            return None
        if not Path(image_path).exists():
            return None
            
        img = Image.open(image_path).convert("RGBA")
        
        def create_checkerboard(width: int, height: int, square_size: int = 8) -> Image.Image:
            checker = Image.new("RGB", (width, height))
            pixels = checker.load()
            light_gray = (220, 220, 220)
            dark_gray = (180, 180, 180)
            for y in range(height):
                for x in range(width):
                    if (x // square_size + y // square_size) % 2 == 0:
                        pixels[x, y] = light_gray
                    else:
                        pixels[x, y] = dark_gray
            return checker
        
        bg = create_checkerboard(img.size[0], img.size[1], square_size=10)
        bg.paste(img, mask=img.split()[3])
        bg.thumbnail(size, Image.LANCZOS)
        return np.array(bg)
    except Exception as e:
        # Silently fail for thumbnails
        return None


def _get_verification_status(node_data: Dict[str, Any]) -> str:
    """Get verification status from node data."""
    if not node_data:
        return "pending"
    
    # Check action type for temp/virtual nodes
    action_type = node_data.get("action_type")
    
    # Safe check for temp/virtual node types
    if action_type and isinstance(action_type, str):
        if action_type in ["_TempChild", "_ValidChild", "_InvalidChild", "_RejectedChild"]:
            status_map = {
                "_TempChild": "pending",
                "_ValidChild": "VALID",
                "_InvalidChild": "INVALID",
                "_RejectedChild": "REJECTED",
            }
            return status_map.get(action_type, "pending")
    
    # Check explicit verification_status field
    status = node_data.get("verification_status")
    if status and isinstance(status, str):
        status_upper = status.upper()
        if status_upper in ["PROCEED", "PROCEED_FILTERED", "RETRY", "VALID", "INVALID", "REJECTED"]:
            return status_upper
    
    # Fallback: check tool_outputs.verifier
    tool_outputs = node_data.get("tool_outputs") or {}
    verifier = tool_outputs.get("verifier") or {}
    decision = verifier.get("decision")
    if decision and isinstance(decision, str):
        decision_upper = decision.upper()
        if decision_upper in ["PROCEED", "PROCEED_FILTERED", "RETRY"]:
            return decision_upper
    
    return "pending"


def _is_temp_or_virtual_node(node_data: Dict[str, Any]) -> bool:
    """Check if node is temporary or virtual."""
    if not node_data:
        return False
    
    action_type = node_data.get("action_type")
    
    # Safe string check
    if action_type and isinstance(action_type, str) and action_type.startswith("_"):
        return True
    
    return node_data.get("_is_temporary", False)


def _format_children_analysis_summary(children_analysis: List[Dict[str, Any]]) -> str:
    """Format children analysis as a compact summary."""
    if not children_analysis:
        return ""
    
    summary_parts = []
    for child in children_analysis:
        idx = child.get("index", "?")
        status = child.get("status", "?")
        symbol = "✓" if status == "VALID" else "✗"
        
        # Get check results
        hall = child.get("hallucination_check", "?")[:1]  # P or F
        redun = child.get("redundancy_check", "?")[:1]
        integ = child.get("integrity_check", "?")[:1]
        
        summary_parts.append(f"C{idx}:{symbol}({hall}{redun}{integ})")
    
    return " ".join(summary_parts)


def _format_metadata(node_data: Dict[str, Any]) -> str:
    """
    Format metadata for regular nodes.
    
    Enhanced to show more verification details.
    Added node error display.
    Handles the mapping between coverage_assessment and coverage_check.
    """
    if not node_data:
        return "No data"
    
    verification_status = _get_verification_status(node_data)
    node_error_status = _get_node_error_status(node_data)
    symbol = VERIFICATION_SYMBOLS.get(verification_status, "?")
    
    retry_count = node_data.get("retry_count") or 0
    verification_attempts = node_data.get("verification_attempts") or []
    failed_attempts = node_data.get("failed_attempts") or []
    
    tool_outputs = node_data.get("tool_outputs") or {}
    verifier = tool_outputs.get("verifier") or {}
    
    # coverage_check (new) corresponds to coverage_assessment (the stored value);
    # VerificationAttempt stores it under coverage_assessment.
    if verification_attempts:
        latest_attempt = verification_attempts[-1]
        coverage = latest_attempt.get("coverage_check")
        reasoning = latest_attempt.get("coverage_reason")
    else:
        coverage = verifier.get("coverage_check")
        reasoning = verifier.get("coverage_reason")
    
    children_analysis = verifier.get("children_analysis", [])
    
    fields = [
        ("ID", _safe_str(node_data.get("layer_id"), "?")),
        ("Context", _safe_str(node_data.get("image_context"), "-")),
        ("Action", _safe_str(node_data.get("action_type"), "-")),
        ("Tools", str(node_data.get("planned_tool_sequence") or [])),
    ]
    
    # Add node error info if present
    if node_error_status:
        error_symbol = VERIFICATION_SYMBOLS.get(node_error_status, "⚠")
        error_info = node_data.get("error_info") or {}
        error_msg = error_info.get("message", "Unknown")
        details = error_info.get("details") or {}
        failed_node = details.get("node", "?")
        
        if len(error_msg) > 50:
            error_msg = error_msg[:47] + "..."
        
        fields.append(("⚠ ERROR", f"{error_symbol} {node_error_status}"))
        fields.append(("Failed@", failed_node))
        fields.append(("ErrMsg", error_msg))
        
        retry_child = details.get("retry_child")
        if retry_child:
            fields.append(("RetryChild", retry_child))
    
    # Show failed attempts history if any
    if failed_attempts:
        fail_count = len(failed_attempts)
        last_fail = failed_attempts[-1]
        last_node = last_fail.get("failed_node", "?")
        last_action = last_fail.get("action_type", "?")
        fields.append(("FailHist", f"{fail_count}x (last: {last_node}@{last_action})"))
    
    # Add verification info
    if verification_status != "pending" and not node_error_status:
        fields.append(("V-Status", f"{symbol} {verification_status} (Cov: {coverage})"))
        
        if children_analysis:
            analysis_summary = _format_children_analysis_summary(children_analysis)
            if analysis_summary:
                fields.append(("V-Children", analysis_summary))

        
        if reasoning and reasoning != "-":
            short_reason = reasoning[:50] + "..." if len(str(reasoning)) > 50 else str(reasoning)
            fields.append(("V-Reason", short_reason))
    
    # Add attempt/retry count
    if verification_attempts or retry_count > 0:
        fields.append(("Attempts", f"{len(verification_attempts)} (Retries: {retry_count})"))
    
    fields.append(("Children", str(node_data.get("children_ids") or [])))
    
    lines = []
    wrapper = textwrap.TextWrapper(width=TEXT_WRAP_WIDTH, initial_indent="", subsequent_indent="  ")
    
    for label, value in fields:
        if value is None or value == "":
            value = "-"
        if not isinstance(value, str):
            value = str(value)
        if len(value) > 100:
            value = value[:97] + "..."
        wrapped = wrapper.fill(f"{label}: {value}")
        lines.append(wrapped)
    
    return "\n".join(lines)

def _format_temp_metadata(node_data: Dict[str, Any]) -> str:
    """
    Format metadata for temp/virtual nodes.
    
    Enhanced to show more detail about verification result.
    """
    if not node_data:
        return "No data"
    
    verification_status = _get_verification_status(node_data)
    symbol = VERIFICATION_SYMBOLS.get(verification_status, "?")
    
    child_index = node_data.get("_child_index", "?")
    attempt_number = node_data.get("_attempt_number", "?")
    
    context = _safe_str(node_data.get("image_context"), "-")
    # Remove prefix tags for cleaner display
    for prefix in ["[Pending]", "[REJECTED - RETRY]", "[INVALID]", "[VALID]"]:
        context = context.replace(prefix, "").strip()
    if len(context) > 60:
        context = context[:60] + "..."
    
    fields = [
        ("Status", f"{symbol} {verification_status}"),
        ("Child#", str(child_index)),
        ("Attempt", f"#{attempt_number}"),
    ]
    
    # Add context only if meaningful
    if context and context != "-":
        fields.append(("Ctx", context))
    
    # Add reason if invalid/rejected
    reason = node_data.get("action_reasoning")
    if reason and verification_status in ["INVALID", "REJECTED", "DISCARDED"]:
        short_reason = str(reason)[:100] + "..." if len(str(reason)) > 100 else str(reason)
        fields.append(("Why", short_reason))
    
    lines = []
    wrapper = textwrap.TextWrapper(width=25, initial_indent="", subsequent_indent="  ")
    
    for label, value in fields:
        if not isinstance(value, str):
            value = str(value)
        wrapped = wrapper.fill(f"{label}: {value}")
        lines.append(wrapped)
    
    return "\n".join(lines)


# =============================================================================
# Layout Calculation
# =============================================================================

def _calculate_tree_layout(
    history_tree: Dict[str, Dict[str, Any]],
    root_id: str
) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float]]:
    """Calculate positions for all nodes including temp children."""
    positions = {}
    
    if root_id not in history_tree:
        return {}, (10, 10)
    
    def get_all_children(layer_id: str) -> List[str]:
        """Get all children including temp children."""
        node = history_tree.get(layer_id, {})
        children = list(node.get("children_ids") or [])
        temp_children = list(node.get("_temp_child_ids") or [])
        return children + temp_children
    
    def is_small_node(layer_id: str) -> bool:
        node = history_tree.get(layer_id, {})
        return _is_temp_or_virtual_node(node)
    
    subtree_widths = {}
    
    def calc_subtree_width(layer_id: str) -> float:
        children = get_all_children(layer_id)
        if not children:
            width = 0.7 if is_small_node(layer_id) else 1.0
        else:
            width = sum(calc_subtree_width(c) for c in children if c in history_tree)
            width = max(width, 0.7 if is_small_node(layer_id) else 1.0)
        subtree_widths[layer_id] = width
        return width
    
    calc_subtree_width(root_id)
    
    def get_max_depth(layer_id: str, current_depth: int = 0) -> int:
        children = get_all_children(layer_id)
        valid_children = [c for c in children if c in history_tree]
        if not valid_children:
            return current_depth
        return max(get_max_depth(c, current_depth + 1) for c in valid_children)
    
    max_depth = get_max_depth(root_id)
    total_width_slots = subtree_widths.get(root_id, 1)
    
    slot_width = NODE_WIDTH_INCHES + HORIZONTAL_GAP_INCHES
    level_height = NODE_HEIGHT_INCHES + VERTICAL_GAP_INCHES
    
    margin = 1.0
    canvas_width = max(total_width_slots * slot_width + margin * 2, 10)
    canvas_height = max((max_depth + 1) * level_height + margin * 2, 8)
    
    def assign_positions(layer_id: str, x_center: float, y: float):
        if layer_id not in history_tree:
            return
        positions[layer_id] = (x_center, y)
        
        children = get_all_children(layer_id)
        valid_children = [c for c in children if c in history_tree]
        if not valid_children:
            return
        
        total_children_width = sum(subtree_widths.get(c, 1) for c in valid_children)
        current_x = x_center - (total_children_width * slot_width / 2)
        
        for child in valid_children:
            child_width = subtree_widths.get(child, 1)
            child_center_x = current_x + (child_width * slot_width / 2)
            
            # Smaller vertical gap for temp nodes
            child_y = y - level_height * (0.7 if is_small_node(child) else 1.0)
            
            assign_positions(child, child_center_x, child_y)
            current_x += child_width * slot_width
    
    start_x = canvas_width / 2
    start_y = canvas_height - margin - NODE_HEIGHT_INCHES / 2
    
    assign_positions(root_id, start_x, start_y)
    
    return positions, (canvas_width, canvas_height)


# =============================================================================
# Drawing Functions
# =============================================================================

def _draw_verification_badge(ax, x: float, y: float, node_data: Dict[str, Any], node_width: float):
    """Draw verification status badge."""
    verification_status = _get_verification_status(node_data)
    
    if verification_status == "pending":
        return
    
    badge_color = VERIFICATION_BADGE_COLORS.get(verification_status, "#A9A9A9")
    symbol = VERIFICATION_SYMBOLS.get(verification_status, "?")
    
    verification_attempts = node_data.get("verification_attempts") or []
    retry_count = node_data.get("retry_count") or 0
    
    badge_text = f"{symbol} {verification_status}"
    if len(verification_attempts) > 1 or retry_count > 0:
        badge_text += f" #{max(len(verification_attempts), retry_count + 1)}"
    
    badge_x = x + node_width / 2 - 0.1
    badge_y = y + NODE_HEIGHT_INCHES / 2 - 0.1
    
    ax.text(
        badge_x, badge_y,
        badge_text,
        fontsize=6,
        fontweight='bold',
        ha='right',
        va='top',
        color='white',
        bbox=dict(
            boxstyle='round,pad=0.2',
            facecolor=badge_color,
            edgecolor='none',
            alpha=0.9
        ),
        zorder=10
    )


def _draw_retry_badge(ax, x: float, y: float, retry_count: int, node_width: float):
    """Draw retry count badge for retry child nodes."""
    if retry_count <= 0:
        return
    
    badge_x = x - node_width / 2 + 0.1
    badge_y = y + NODE_HEIGHT_INCHES / 2 - 0.1
    
    ax.text(
        badge_x, badge_y,
        f"↺ Retry #{retry_count}",
        fontsize=5,
        fontweight='bold',
        ha='left',
        va='top',
        color='white',
        bbox=dict(
            boxstyle='round,pad=0.2',
            facecolor='#9932CC',  # Purple
            edgecolor='none',
            alpha=0.8
        ),
        zorder=10
    )


def _draw_status_marker(ax, x: float, y: float, status: str, node_height: float):
    """Draw status marker (checkmark, X, etc.) for temp nodes."""
    symbol = VERIFICATION_SYMBOLS.get(status, "?")
    color = VERIFICATION_BADGE_COLORS.get(status, "#A9A9A9")
    
    # Large centered symbol
    ax.text(
        x, y + node_height * 0.15,
        symbol,
        fontsize=24,
        fontweight='bold',
        ha='center',
        va='center',
        color=color,
        alpha=0.7,
        zorder=5
    )


# =============================================================================
# Main Visualization Function
# =============================================================================

def visualize_history_tree(
    state: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = None,
    thumbnail_size: Tuple[int, int] = THUMBNAIL_MAX_SIZE,
    show_images: bool = True,
    title: Optional[str] = None,
    show_temp_children: bool = True,
    show_verification_badges: bool = True,
) -> str:
    """
    Visualize the history tree including all verification attempts.
    
    Added comprehensive null checks throughout.
    Enhanced verification display.
    """
    history_tree = state.get("history_tree") or {}
    root_id = state.get("root_layer_id", "layer_0000")
    episode_dir = state.get("episode_dir", ".")
    processing_ids = state.get("processing_ids") or []
    
    if not history_tree:
        print("[Visualizer] Warning: Empty history tree")
        return ""
    
    if output_path is None:
        output_path = str(Path(episode_dir) / "tree_visualization.png")
    
    # Calculate layout
    positions, (canvas_width, canvas_height) = _calculate_tree_layout(history_tree, root_id)
    
    if not positions:
        return ""
    
    if figsize is None:
        figsize = (canvas_width, canvas_height)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=OUTPUT_DPI)
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_aspect('equal')
    
    # --- 1. Draw Edges ---
    for layer_id, node_data in history_tree.items():
        if layer_id not in positions:
            continue
        
        if not node_data:
            continue
        
        # Get all children including temp
        children = list(node_data.get("children_ids") or [])
        temp_children = list(node_data.get("_temp_child_ids") or [])
        all_children = children + temp_children
        
        p_x, p_y = positions[layer_id]
        is_parent_temp = _is_temp_or_virtual_node(node_data)
        p_half_h = TEMP_NODE_HEIGHT_INCHES / 2 if is_parent_temp else NODE_HEIGHT_INCHES / 2
        p_bottom = p_y - p_half_h
        
        for child_id in all_children:
            if child_id not in positions or child_id not in history_tree:
                continue
            
            c_x, c_y = positions[child_id]
            child_node = history_tree.get(child_id) or {}
            child_is_temp = _is_temp_or_virtual_node(child_node)
            c_half_h = TEMP_NODE_HEIGHT_INCHES / 2 if child_is_temp else NODE_HEIGHT_INCHES / 2
            c_top = c_y + c_half_h
            
            # Edge style based on child verification status
            child_status = _get_verification_status(child_node)
            
            if child_status == "INVALID":
                edge_color = '#DC143C'
                edge_style = '-'
                edge_width = 1.5
                edge_alpha = 0.6
            elif child_status == "REJECTED":
                edge_color = '#9932CC'
                edge_style = '--'
                edge_width = 2.0
                edge_alpha = 0.7
            elif child_status == "VALID":
                edge_color = '#228B22'
                edge_style = '-'
                edge_width = 2.0
                edge_alpha = 0.8
            elif child_status == "pending":
                edge_color = '#A9A9A9'
                edge_style = ':'
                edge_width = 1.5
                edge_alpha = 0.5
            else:
                edge_color = '#555555'
                edge_style = '-'
                edge_width = 1.5
                edge_alpha = 0.6
            
            ax.plot(
                [p_x, c_x], [p_bottom, c_top],
                color=edge_color, linewidth=edge_width, 
                linestyle=edge_style, alpha=edge_alpha, zorder=1
            )
    
    # --- 2. Draw Nodes ---
    for layer_id, node_data in history_tree.items():
        if layer_id not in positions:
            continue
        
        if not node_data:
            continue
        
        x, y = positions[layer_id]
        is_temp = _is_temp_or_virtual_node(node_data)
        verification_status = _get_verification_status(node_data)
        node_error_status = _get_node_error_status(node_data)
        
        # Safe action_type retrieval
        action_type = node_data.get("action_type")
        if not action_type or not isinstance(action_type, str):
            action_type = "Pending"
        
        # Determine node size
        if is_temp:
            node_w = TEMP_NODE_WIDTH_INCHES
            node_h = TEMP_NODE_HEIGHT_INCHES
        else:
            node_w = NODE_WIDTH_INCHES
            node_h = NODE_HEIGHT_INCHES
        
        half_w = node_w / 2
        half_h = node_h / 2
        
        # Get colors
        bg_color = ACTION_COLORS.get(action_type, ACTION_COLORS["Pending"])
        
        # Border style based on verification status
        border_style = VERIFICATION_BORDER_STYLES.get(
            verification_status, 
            VERIFICATION_BORDER_STYLES["pending"]
        )
        
        # --- A. Main Container (Base Background, NO Border here) ---
        main_rect = mpatches.FancyBboxPatch(
            (x - half_w, y - half_h),
            node_w, node_h,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor="white",
            edgecolor="none", 
            zorder=2
        )
        ax.add_patch(main_rect)
        
        # --- B. Image Section Background (Top) ---
        img_section_height = node_h * IMAGE_RATIO
        txt_section_height = node_h * TEXT_RATIO
        img_section_bottom = y + half_h - img_section_height
        
        img_rect = mpatches.FancyBboxPatch(
            (x - half_w, img_section_bottom),
            node_w, img_section_height,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor="#FAFAFA",
            edgecolor="none",
            zorder=2.1
        )
        ax.add_patch(img_rect)
        
        # --- C. Text Section Background (Bottom) ---
        txt_section_bottom = y - half_h
        txt_rect = mpatches.FancyBboxPatch(
            (x - half_w, txt_section_bottom),
            node_w, txt_section_height,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor=bg_color,
            alpha=0.2,
            edgecolor="none",
            zorder=2.1
        )
        ax.add_patch(txt_rect)
        
        # --- D. Special overlays for verification status ---
        if node_error_status:
            overlay_color = VERIFICATION_BADGE_COLORS.get(node_error_status, "#FF4500")
            overlay = mpatches.FancyBboxPatch(
                (x - half_w, y - half_h),
                node_w, node_h,
                boxstyle="round,pad=0,rounding_size=0.1",
                facecolor=overlay_color,
                alpha=0.2,  # Slightly more visible for errors
                edgecolor="none",
                zorder=2.15
            )
            ax.add_patch(overlay)
            
            # Draw error symbol for emphasis
            error_symbol = VERIFICATION_SYMBOLS.get(node_error_status, "⚠")
            ax.text(
                x, y + node_h * 0.1,
                error_symbol,
                fontsize=20,
                fontweight='bold',
                ha='center',
                va='center',
                color=overlay_color,
                alpha=0.4,
                zorder=5
            )
        
        elif verification_status in ["INVALID", "REJECTED", "DISCARDED"]:
            overlay_color = VERIFICATION_BADGE_COLORS.get(verification_status, "#A9A9A9")
            overlay = mpatches.FancyBboxPatch(
                (x - half_w, y - half_h),
                node_w, node_h,
                boxstyle="round,pad=0,rounding_size=0.1",
                facecolor=overlay_color,
                alpha=0.15,
                edgecolor="none",
                zorder=2.15
            )
            ax.add_patch(overlay)
            
            # Draw status marker for temp nodes
            if is_temp:
                _draw_status_marker(ax, x, y, verification_status, node_h)
        
        elif verification_status == "VALID":
            overlay = mpatches.FancyBboxPatch(
                (x - half_w, y - half_h),
                node_w, node_h,
                boxstyle="round,pad=0,rounding_size=0.1",
                facecolor="#228B22",
                alpha=0.1,
                edgecolor="none",
                zorder=2.15
            )
            ax.add_patch(overlay)
        
        # --- E. Processing Glow ---
        if layer_id in processing_ids:
            glow = mpatches.FancyBboxPatch(
                (x - half_w - 0.1, y - half_h - 0.1),
                node_w + 0.2, node_h + 0.2,
                boxstyle="round,pad=0,rounding_size=0.1",
                facecolor="none",
                edgecolor="#FF0000",
                linewidth=3,
                linestyle="--",
                zorder=2.2
            )
            ax.add_patch(glow)

        # --- F. Border (Explicit Top Layer) ---
        border_rect = mpatches.FancyBboxPatch(
            (x - half_w, y - half_h),
            node_w, node_h,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor="none",
            edgecolor=border_style["color"],
            linewidth=border_style["linewidth"],
            linestyle=border_style["linestyle"],
            zorder=2.5
        )
        ax.add_patch(border_rect)
        
        # --- G. Verification Badge (for non-temp nodes) ---
        if show_verification_badges and not is_temp:
            _draw_verification_badge(ax, x, y, node_data, node_w)
        
        node_error_status = _get_node_error_status(node_data)
        if node_error_status and not is_temp:
            _draw_node_error_badge(ax, x, y, node_data, node_w)
        
        if node_error_status:
            # Node has an execution error - use error border style
            border_style = VERIFICATION_BORDER_STYLES.get(
                node_error_status,
                VERIFICATION_BORDER_STYLES["NODE_ERROR"]
            )
        else:
            # Use verification status border style
            border_style = VERIFICATION_BORDER_STYLES.get(
                verification_status, 
                VERIFICATION_BORDER_STYLES["pending"]
            )

        
        # --- G3. Failed Attempts Indicator ---
        failed_attempts = node_data.get("failed_attempts") or []
        if failed_attempts and not is_temp:
            _draw_failed_attempts_indicator(ax, x, y, failed_attempts, node_w)



        # --- G2. Retry Badge (for retry child nodes) ---
        retry_count = node_data.get("retry_count", 0)
        if retry_count > 0 and not is_temp:
            # Check if this is a retry child (has "retry" in context or ID)
            image_context = node_data.get("image_context", "")
            if "Retry" in str(image_context) or "retry" in str(layer_id):
                _draw_retry_badge(ax, x, y, retry_count, node_w)
        
        # --- H. Image ---
        if show_images:
            image_path = node_data.get("image_path")
            img_arr = _create_thumbnail(image_path, thumbnail_size)
            
            if img_arr is not None:
                img_center_y = img_section_bottom + (img_section_height / 2)
                
                img_pixel_h, img_pixel_w = img_arr.shape[:2]
                avail_w_inches = node_w * 0.85
                avail_h_inches = img_section_height * 0.85
                avail_w_px = avail_w_inches * OUTPUT_DPI
                avail_h_px = avail_h_inches * OUTPUT_DPI
                
                zoom_w = avail_w_px / img_pixel_w
                zoom_h = avail_h_px / img_pixel_h
                zoom = min(zoom_w, zoom_h, 1.0)
                
                imagebox = OffsetImage(img_arr, zoom=zoom)
                ab = AnnotationBbox(imagebox, (x, img_center_y), frameon=False, zorder=3)
                ax.add_artist(ab)
        
        # --- I. Metadata Text ---
        if is_temp:
            meta_text = _format_temp_metadata(node_data)
            txt_fontsize = 6
        else:
            meta_text = _format_metadata(node_data)
            txt_fontsize = TEXT_FONTSIZE
        
        txt_center_y = txt_section_bottom + (txt_section_height / 2)
        
        ax.text(
            x, txt_center_y,
            meta_text,
            ha='center', va='center',
            fontsize=txt_fontsize,
            fontfamily='monospace',
            color="black",
            zorder=4,
            linespacing=1.2,
            clip_on=True
        )
    
    # --- 3. Legend ---
    legend_handles = []
    
    # Action types
    priority_actions = ["Fork_Qwen", "Split_DetSeg", "Split_Text", "Split_CCA", 
                       "Finalize_Text", "Finalize_Obj", "Discard"]
    for action in priority_actions:
        if action in ACTION_COLORS:
            legend_handles.append(mpatches.Patch(color=ACTION_COLORS[action], label=action))
    
    # Verification statuses
    legend_handles.append(mpatches.Patch(
        facecolor='white', edgecolor=VERIFICATION_BADGE_COLORS["PROCEED"], 
        linewidth=2, label="✓ PROCEED"
    ))
    legend_handles.append(mpatches.Patch(
        facecolor='white', edgecolor=VERIFICATION_BADGE_COLORS["PROCEED_FILTERED"], 
        linewidth=2, label="⚠ FILTERED"
    ))
    legend_handles.append(mpatches.Patch(
        facecolor='white', edgecolor=VERIFICATION_BADGE_COLORS["INVALID"], 
        linewidth=2, label="✗ INVALID"
    ))
    legend_handles.append(mpatches.Patch(
        facecolor='white', edgecolor=VERIFICATION_BADGE_COLORS["REJECTED"], 
        linewidth=2, linestyle='--', label="↺ RETRY"
    ))
    
    # Node error statuses
    legend_handles.append(mpatches.Patch(
        facecolor='white', edgecolor=VERIFICATION_BADGE_COLORS["NODE_ERROR"], 
        linewidth=2, label="⚠ NODE_ERR"
    ))
    legend_handles.append(mpatches.Patch(
        facecolor='white', edgecolor=VERIFICATION_BADGE_COLORS["CUDA_ERROR"], 
        linewidth=2, label="⛔ CUDA_ERR"
    ))
    
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.01, 0.99), 
              fontsize=6, framealpha=0.9, ncol=2)
    
    # --- 4. Title ---
    if title is None:
        episode_id = state.get('episode_id', '?')
        layer_count = state.get('layer_count', len(history_tree))
        elements_count = len(state.get('parsed_elements') or [])
        
        # Count verification stats
        v_stats = {"PROCEED": 0, "PROCEED_FILTERED": 0, "RETRY": 0,
                  "VALID": 0, "INVALID": 0, "REJECTED": 0}
        # Count error stats
        e_stats = {"NODE_ERROR": 0, "GPU_TIMEOUT": 0, "CUDA_ERROR": 0}
        
        for node_data in history_tree.values():
            if node_data:
                v_status = _get_verification_status(node_data)
                if v_status in v_stats:
                    v_stats[v_status] += 1
                
                # Count errors
                e_status = _get_node_error_status(node_data)
                if e_status in e_stats:
                    e_stats[e_status] += 1
        
        title = f"URLD Decomposition Tree - Episode: {episode_id}\n"
        title += f"Layers: {layer_count} | Elements: {elements_count} | "
        title += f"V: ✓{v_stats['PROCEED']+v_stats['VALID']} ⚠{v_stats['PROCEED_FILTERED']} "
        title += f"✗{v_stats['RETRY']+v_stats['INVALID']} ↺{v_stats['REJECTED']}"
        
        # Add error counts if any
        total_errors = sum(e_stats.values())
        if total_errors > 0:
            title += f" | Err: ⚠{e_stats['NODE_ERROR']} ⏱{e_stats['GPU_TIMEOUT']} ⛔{e_stats['CUDA_ERROR']}"
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')
    
    # Save
    plt.tight_layout(pad=0.5)
    try:
        plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight', 
                   facecolor='#F8F8F8', edgecolor='none')
    except Exception as e:
        print(f"[Visualizer] Save failed: {e}")
    plt.close(fig)
    
    print(f"[Visualizer] Saved tree visualization to {output_path}")
    return output_path


def visualize_from_json(
    history_tree_path: str,
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """Load history_tree from JSON and visualize."""
    path = Path(history_tree_path)
    if not path.exists():
        print(f"[Visualizer] Error: {path} not found")
        return ""
    
    with open(path, "r", encoding="utf-8") as f:
        history_tree = json.load(f)
    
    root_id = None
    for layer_id, node_data in history_tree.items():
        if node_data and node_data.get("parent_id") is None:
            root_id = layer_id
            break
    
    if root_id is None:
        root_id = "layer_0000"
    
    episode_dir = path.parent
    state = {
        "history_tree": history_tree,
        "root_layer_id": root_id,
        "episode_dir": str(episode_dir),
        "episode_id": episode_dir.name,
        "processing_ids": [],
        "parsed_elements": [],
        "layer_count": len(history_tree),
    }
    
    if output_path is None:
        output_path = str(episode_dir / "tree_visualization.png")
    
    return visualize_history_tree(state, output_path, **kwargs)


# =============================================================================
# Real-time Update Helper
# =============================================================================

class TreeVisualizer:
    """Helper class for real-time tree visualization updates."""
    
    def __init__(self, episode_dir: str, update_interval: int = 1):
        self.episode_dir = Path(episode_dir)
        self.update_interval = update_interval
        self.last_layer_count = 0
        self.output_path = self.episode_dir / "tree_visualization.png"
    
    def update(self, state: Dict[str, Any], force: bool = False) -> Optional[str]:
        """Update visualization if enough changes have occurred."""
        layer_count = state.get("layer_count", 0)
        
        if not force and (layer_count - self.last_layer_count) < self.update_interval:
            return None
        
        self.last_layer_count = layer_count
        
        try:
            return visualize_history_tree(
                state,
                output_path=str(self.output_path),
                show_images=True,
                show_temp_children=True,
                show_verification_badges=True,
            )
        except Exception as e:
            print(f"[Visualizer] Update failed: {e}")
            return None
    
    def final_update(self, state: Dict[str, Any]) -> str:
        """Generate final visualization."""
        return visualize_history_tree(
            state,
            output_path=str(self.episode_dir / "tree_visualization_final.png"),
            show_images=True,
            show_temp_children=True,
            show_verification_badges=True,
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize URLD History Tree")
    parser.add_argument("--json", type=str, required=True, help="Path to history_tree.json")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--no-images", action="store_true", help="Don't show thumbnails")
    parser.add_argument("--no-temp", action="store_true", help="Don't show temp children")
    parser.add_argument("--no-badges", action="store_true", help="Don't show verification badges")
    
    args = parser.parse_args()
    
    output = visualize_from_json(
        args.json,
        args.output,
        show_images=not args.no_images,
        show_temp_children=not args.no_temp,
        show_verification_badges=not args.no_badges,
    )
    
    if output:
        print(f"Visualization saved to: {output}")