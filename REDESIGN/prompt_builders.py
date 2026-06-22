# REDESIGN/prompt_builders.py
"""
Prompt builders for URLD (Unified Recursive Layer Decomposition) Pipeline

Builds prompts with ancestry context for router_vlm node.

[Revision 27] Verifier response parsing simplified:
- Removed _normalize_verifier_result()
- Removed _default_verifier_result()
- parse_verifier_response() now returns raw JSON with minimal processing
"""
from __future__ import annotations
from typing import Dict, Any, List
import json

from datetime import datetime

from .prompts import (
    ROUTER_VLM_SYSTEM_PROMPT,
    ROUTER_VLM_USER_PROMPT_TEMPLATE,
    VERIFIER_VLM_SYSTEM_PROMPT,
    VERIFIER_VLM_USER_PROMPT_TEMPLATE,
    FINAL_VERIFICATION_SYSTEM_PROMPT
)
from .utils import get_ancestor_chain


def build_final_verification_system_prompt() -> str:
    """Build system prompt for final verification."""
    return FINAL_VERIFICATION_SYSTEM_PROMPT

def build_verifier_system_prompt() -> str:
    """Build system prompt for verifier VLM."""
    return VERIFIER_VLM_SYSTEM_PROMPT


def build_verifier_user_prompt(
    layer_id: str,
    state: Dict[str, Any],
    num_children: int,
) -> str:
    tree = state.get("history_tree", {})
    node = tree.get(layer_id, {})
    
    action_type = node.get("action_type", "Unknown")
    tool_sequence = node.get("planned_tool_sequence", [])
    depth = node.get("depth", 0)
    
    return VERIFIER_VLM_USER_PROMPT_TEMPLATE.format(
        layer_id=layer_id,
        depth=depth,
        action_type=action_type,
        tool_sequence=str(tool_sequence),
        num_children=num_children,
        last_image=num_children + 1,
        last_child=num_children - 1,
    )



def build_router_system_prompt() -> str:
    """
    Build system prompt for router_vlm.
    Injects detailed tool definitions into the prompt template.
    """
    # 1. Get tool info map
    tool_info_map = build_tool_static_info()
    
    # 2. Format as a readable string
    tool_definitions = []
    for tool_name, description in tool_info_map.items():
        # Remove extra whitespace/indentation for cleaner prompt
        clean_desc = "\n".join([line.strip() for line in description.strip().split('\n')])
        tool_definitions.append(f"### {tool_name}\n{clean_desc}")
    
    tool_definitions_str = "\n\n".join(tool_definitions)
    
    # 3. Format the system prompt with tool definitions
    return ROUTER_VLM_SYSTEM_PROMPT.format(
        tool_definitions=tool_definitions_str
    )


def build_router_user_prompt(
    layer_id: str,
    state: Dict[str, Any]
) -> str:
    """Build user prompt with ancestry, retry context, AND node error context."""
    tree = state.get("history_tree", {})
    node = tree.get(layer_id, {})
    
    # 1. Ancestry chain
    ancestry = get_ancestor_chain(layer_id, state)
    ancestry_json = json.dumps(ancestry, ensure_ascii=False, indent=2) if ancestry else "[]"
    
    # 2. Failed Attempts
    failed_attempts_context = _format_failed_attempts(layer_id, state)
    
    # 3. [Revision 35] Node Error Context
    node_error_context = _format_node_errors(layer_id, state)
    
    # 4. Parent action
    parent_id = node.get("parent_id")
    parent_action = "ROOT"
    if parent_id and parent_id in tree:
        parent_action = tree[parent_id].get("action_type", "Unknown")
    
    # 5. Build prompt
    prompt = ROUTER_VLM_USER_PROMPT_TEMPLATE.format(
        ancestry_json=ancestry_json,
        failed_attempts_context=failed_attempts_context,
        node_error_context=node_error_context,  # [Revision 35] NEW
        layer_id=layer_id,
        depth=node.get("depth", 0),
        parent_action=parent_action,
    )
    
    return prompt


def build_ancestry_summary(layer_id: str, state: Dict[str, Any]) -> str:
    """
    Build a concise summary of ancestry for context.
    
    Returns a readable string summarizing the decomposition path.
    """
    ancestry = get_ancestor_chain(layer_id, state)
    
    if not ancestry:
        return "This is the root layer."
    
    lines = ["Decomposition path from root:"]
    for i, node in enumerate(ancestry):
        action = node.get("action_type", "?")
        context = node.get("image_context", "")[:50] + "..." if node.get("image_context") else ""
        lines.append(f"  {i+1}. [{action}] {context}")
    
    return "\n".join(lines)


def _format_node_errors(layer_id: str, state: Dict[str, Any]) -> str:
    """
    Format node execution errors from failed_attempts.
    [Revision 35] Extracts node_execution_error entries for Router VLM.
    """
    tree = state.get("history_tree", {})
    node = tree.get(layer_id, {})
    failed_attempts = node.get("failed_attempts", [])
    
    # Filter for node execution errors
    node_errors = [
        fa for fa in failed_attempts 
        if fa.get("error_type") == "node_execution_error"
    ]
    
    if not node_errors:
        return "No previous node execution errors."
    
    lines = [f"⚠️ {len(node_errors)} NODE EXECUTION ERROR(S) occurred:"]
    lines.append("")
    
    for i, error in enumerate(node_errors):
        failed_node = error.get("failed_node", "Unknown")
        error_msg = error.get("error_message", "No message")
        action_type = error.get("action_type", "Unknown")
        tool_sequence = error.get("tool_sequence", [])
        
        lines.append(f"Error #{i+1}:")
        lines.append(f"  - Action Attempted: {action_type}")
        lines.append(f"  - Tool Sequence: {tool_sequence}")
        lines.append(f"  - Failed at Node: {failed_node}")
        lines.append(f"  - Error Message: {error_msg}")
        lines.append("")
        
    return "\\n".join(lines)



def _format_failed_attempts(layer_id: str, state: Dict[str, Any]) -> str:
    """
    Format failed attempts for the current layer into a readable string.
    
    [Revision 34] Enhanced to show all historical failed attempts for retry decision.
    [Revision 38] Added params output for hyperparameter (qwen_len, etc.) tracking.
    [Revision 39] Added ACTION DIVERSITY analysis - detect consecutive same-action failures
             and recommend considering alternative approaches.
    
    Crucial for the Router to learn from immediate past mistakes and avoid repeating.
    """
    tree = state.get("history_tree", {})
    node = tree.get(layer_id, {})
    failed_attempts = node.get("failed_attempts", [])

    if not failed_attempts:
        return "No previous failed attempts. This is the first attempt for this layer."

    lines = [f"Total {len(failed_attempts)} failed attempt(s):"]
    lines.append("")
    
    # Track action types and their sequence
    action_counts = {}  # action_type -> count
    action_sequence = []  # ordered list of action_types tried
    
    # Track tried hyperparameters
    tried_qwen_lens = set()
    tried_inpaint_variants = set()
    
    for i, attempt in enumerate(failed_attempts):
        action = attempt.get("action_type", "Unknown")
        tools = attempt.get("tool_sequence", [])
        params = attempt.get("params", {})
        decision = attempt.get("verifier_decision", "REJECTED")
        reason = attempt.get("failure_reason", "No reason provided")
        coverage = attempt.get("coverage", "Unknown")
        
        # Track action sequence
        action_sequence.append(action)
        action_counts[action] = action_counts.get(action, 0) + 1
        
        # Track qwen_len for Fork_Qwen
        if action == "Fork_Qwen" and "qwen_len" in params:
            tried_qwen_lens.add(params["qwen_len"])
        
        # Check for inpainting variants
        tool_str = str(tools)
        inpaint_variant = ""
        if "objectclear" in tool_str and "lama" in tool_str:
            inpaint_variant = " (lama + objectclear)"
            tried_inpaint_variants.add("lama+objectclear")
        elif "objectclear" in tool_str:
            inpaint_variant = " (objectclear only)"
            tried_inpaint_variants.add("objectclear")
        elif "lama" in tool_str:
            inpaint_variant = " (lama only)"
            tried_inpaint_variants.add("lama")
        
        lines.append(f"Attempt #{i+1}: {action}{inpaint_variant}")
        lines.append(f"  - Tool Sequence: {tools}")
        
        if params:
            lines.append(f"  - Params: {params}")
        
        lines.append(f"  - Verifier Decision: {decision}")
        lines.append(f"  - Coverage Issue: {coverage}")
        lines.append(f"  - Failure Reason: {reason[:200]}...")
        lines.append("")

    # =========================================================================
    # PATTERN ANALYSIS
    # =========================================================================
    lines.append("═" * 60)
    lines.append("PATTERN ANALYSIS:")
    lines.append(f"  Action sequence: {' → '.join(action_sequence)}")
    lines.append(f"  Action counts: {action_counts}")
    
    # Analyze consecutive same-action failures
    last_action = action_sequence[-1] if action_sequence else None
    consecutive_same_action = 0
    for action in reversed(action_sequence):
        if action == last_action:
            consecutive_same_action += 1
        else:
            break
    
    lines.append(f"  Recent pattern: '{last_action}' has failed {consecutive_same_action} time(s) consecutively")
    
    # Show untried options
    if tried_qwen_lens:
        untried_lens = {2, 3, 4, 5, 6} - tried_qwen_lens
        lines.append(f"  Fork_Qwen: tried qwen_len={sorted(tried_qwen_lens)}, untried={sorted(untried_lens)}")
    
    if tried_inpaint_variants:
        all_variants = {"lama", "objectclear", "lama+objectclear"}
        untried_variants = all_variants - tried_inpaint_variants
        lines.append(f"  Split_DetSeg: tried={sorted(tried_inpaint_variants)}, untried={sorted(untried_variants)}")
    
    lines.append("")
    
    # =========================================================================
    # RECOMMENDATION (principle-based, not rule-based)
    # =========================================================================
    lines.append("═" * 60)
    lines.append("★ RECOMMENDATION:")
    
    qwen_count = action_counts.get("Fork_Qwen", 0)
    detseg_count = action_counts.get("Split_DetSeg", 0)
    
    # Check if same action has been failing consecutively
    if consecutive_same_action > 1:
        lines.append(f"  → '{last_action}' has failed multiple times consecutively.")
        lines.append(f"    Consider switching to a different action type for diversity.")
        
        if last_action == "Fork_Qwen" and detseg_count == 0:
            lines.append(f"    Split_DetSeg has not been tried yet - it may succeed where Qwen struggles.")
        elif last_action == "Split_DetSeg" and qwen_count == 0:
            lines.append(f"    Fork_Qwen has not been tried yet - semantic decomposition may work better.")
        elif last_action == "Fork_Qwen":
            lines.append(f"    Split_DetSeg (detection-based) offers a fundamentally different approach.")
        elif last_action == "Split_DetSeg":
            lines.append(f"    Fork_Qwen (semantic) offers a fundamentally different approach.")
    
    elif consecutive_same_action == 1:
        lines.append(f"  → '{last_action}' failed once. You may try a different hyperparameter,")
        lines.append(f"    but also consider whether a different action type might be more effective.")
        
        if last_action == "Fork_Qwen":
            untried_lens = {2, 3, 4, 5, 6} - tried_qwen_lens
            if untried_lens:
                lines.append(f"    Option A: Try Fork_Qwen with qwen_len in {sorted(untried_lens)}")
            lines.append(f"    Option B: Try Split_DetSeg for a detection-based approach")
        
        elif last_action == "Split_DetSeg":
            untried_variants = {"lama", "objectclear", "lama+objectclear"} - tried_inpaint_variants
            if untried_variants:
                lines.append(f"    Option A: Try Split_DetSeg with inpainting variant in {sorted(untried_variants)}")
            lines.append(f"    Option B: Try Fork_Qwen for semantic decomposition")
    
    # Check for exhaustion pattern
    if qwen_count >= 1 and detseg_count >= 1:
        lines.append("")
        lines.append(f"  ⚠️ Both Fork_Qwen ({qwen_count}x) and Split_DetSeg ({detseg_count}x) have been tried.")
        
        # Check if there are still untried hyperparameters
        untried_lens = {2, 3, 4, 5, 6} - tried_qwen_lens
        untried_variants = {"lama", "objectclear", "lama+objectclear"} - tried_inpaint_variants
        
        if untried_lens or untried_variants:
            lines.append(f"    Some hyperparameter variations remain untried.")
            if untried_lens:
                lines.append(f"      - Fork_Qwen qwen_len: {sorted(untried_lens)}")
            if untried_variants:
                lines.append(f"      - Split_DetSeg inpainting: {sorted(untried_variants)}")
        else:
            lines.append(f"    Most variations exhausted. Consider Finalize_Obj if decomposition seems infeasible.")
    
    lines.append("═" * 60)

    return "\n".join(lines)


def build_tool_static_info() -> Dict[str, str]:
    """
    Build static tool information for reference.
    
    Returns dict mapping tool_name to description.
    """
    return {
        "qwen_layered": """
            — I/O: qwen_layered(image_path, layers=N) -> List[layer_images]
            — Purpose: Generate semantic multi-layer decomposition using MM-DiT model
            — Use when: Objects are complexly intertwined, standard detection fails
            — Hyperparameter: layers (2-6, higher for more complex images)
            — Output: Layers with transparent backgrounds, objects may be spatially separated
        """,
        
        "split_cca": """
            — I/O: split_cca(image_path) -> List[component_masks]
            — Purpose: Fast pixel-based Connected Component Analysis
            — ⚠️ PREREQUISITE: Only valid after Fork_Qwen has been used in ancestry
            — Use when: Qwen output has spatially separated objects on transparent background
            — Advantage: No deep learning, very fast
            — Invalid if: No Fork_Qwen in ancestry history
        """,
        
        "vlm_front_pick": """
            — I/O: vlm_front_pick(image_path) -> {"labels": [...]}
            — Purpose: VLM-based front element labeling for GDINO input
            — Use for: Object detection with GDINO
        """,
        
        "ocr": """
            — I/O: ocr(image_path) -> {"boxes": [...], "texts": [...], "scores": [...]}
            — Purpose: Text detection using PaddleOCR
            — Use for: Detecting text regions before segmentation
        """,
        
        "gdino": """
            — I/O: gdino(image_path, labels) -> {"boxes": [...], "confs": [...], "labels": [...]}
            — Purpose: Open-vocabulary object detection
            — Requires: Labels from vlm_front_pick
        """,
        
        "hisam": """
            — I/O: hisam(image_path, boxes) -> {"mask_union": path, "masks_by_id": {...}}
            — Purpose: Text segmentation using hierarchical SAM
            — Use after: OCR detection
        """,
        
        "sam2_bbox": """
            — I/O: sam2_bbox(image_path, boxes) -> {"mask_union": path, "masks_by_id": {...}}
            — Purpose: Object segmentation using SAM2 with bbox prompts
            — Use after: GDINO detection
        """,
        
        "lama": """
            — I/O: lama(image_path, mask_path) -> inpainted_image_path
            — Purpose: Inpainting to remove objects/text from background
            — Strong for: Text removal, general cleanup
        """,
        
        "objectclear": """
            — I/O: objectclear(image_path, mask_path) -> inpainted_image_path
            — Purpose: Object-specific inpainting
            — objectclear focuses on background-consistentcy so it can often clean residual blur/halos after lama, but may frequently insert unwanted objects/artifacts.
            — You must consider optional objectclear inpainting tool context-wisely.
        """,
        
        "fontstyle": """
            — I/O: fontstyle(element_info) -> {"font_family", "size_px", "color", ...}
            — Purpose: Font property analysis for text elements
            — ⚠️ SYSTEM USE ONLY: This tool is automatically called by Split_Text.
            — Do NOT directly plan this tool in your action sequence.
        """,
        
        "vtracer": """
            — I/O: vtracer(rgba_path) -> {"svg_uri": path}
            — Purpose: Image to SVG vectorization
            — Use for: Non-photographic graphics in Finalize_Obj
        """,
    }


'''
        "nanobanana": """
            — I/O: nanobanana(image_path, nanobanana_instruction) -> refined_image_path
            — Purpose: General image refinement/cleanup using Gemini
            — Requires: Text instruction describing the refinement
            — You MUST provide `nanobanana_instruction` in params: Without `nanobanana_instruction`, the nanobanana node will FAIL!
        """,

'''