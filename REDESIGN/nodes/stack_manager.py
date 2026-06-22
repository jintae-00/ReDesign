# REDESIGN/nodes/stack_manager.py

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import shutil
import numpy as np
import json
import re
import gc
import os
from PIL import Image
import io             
import base64
from datetime import datetime
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import GraphState, generate_layer_id
from ..reducers import (
    r_create_layer_node,
    r_set_children_ids,
    r_enqueue_layers,
    r_dequeue_node,
    r_pack_state,
    r_save_verifier_output,
    r_record_failed_attempt,
    r_create_temp_child_nodes,
    r_update_temp_children_status,
    r_create_retry_child_node,
    r_llm_inc,
    llm_can_call,
    merge_updates,
)
from ..utils import (
    get_current_image_path,
    create_remainder_layer,
    image_to_b64,
    extract_json,
    apply_transparency_to_inpainted_image,
)
from ..prompt_builders import (
    build_verifier_system_prompt,
    build_verifier_user_prompt,
)


def _log(message: str):
    """Simple logging function."""
    print(f"[Stack Manager] {message}")


# =============================================================================
# Configuration
# =============================================================================

MAX_RETRIES = 3  # Maximum verification retries per layer
MIN_VALID_CHILDREN = 2  # [수정 37] Minimum valid children for successful split/fork


def has_exceeded_max_retries(layer_id: str, state: GraphState) -> bool:
    """Check if layer has exceeded max retry attempts."""
    tree = state.get("history_tree", {})
    node = tree.get(layer_id, {})
    retry_count = node.get("retry_count", 0)
    return retry_count >= MAX_RETRIES


def _should_retry_from_verification(verifier_output: Dict[str, Any]) -> Tuple[bool, str]:
    """
    [수정 37] Check if retry is needed based on verification result.
    
    Retry conditions:
    1. Coverage is INCOMPLETE
    2. Valid children count < 2 (all split/fork actions require >= 2 children)
    
    Returns:
        Tuple of (should_retry: bool, reason: str)
    """
    coverage = verifier_output.get("coverage_check", "").upper()
    valid_indices = verifier_output.get("valid_children_indices", [])
    valid_count = len(valid_indices)
    total_children = valid_count + len(verifier_output.get("invalid_children_indices", []))
    
    # Condition 1: Coverage is INCOMPLETE
    if coverage == "INCOMPLETE":
        return True, f"Coverage is INCOMPLETE"
    
    # Condition 2: Less than 2 valid children (split/fork requires >= 2)
    if valid_count < MIN_VALID_CHILDREN:
        return True, f"Only {valid_count} valid children (produced {total_children}, requires >= {MIN_VALID_CHILDREN} for split/fork)"
    
    return False, ""


# =============================================================================
# Verifier VLM Setup
# =============================================================================

def parse_verifier_response(response_text: str) -> Dict[str, Any]:
    """
    Parse verifier VLM response - extract JSON only, no normalization.
    
    [수정 27] Simplified: Returns raw JSON as-is, only adds timestamp if missing.
    No more normalization or default fallback.
    
    Args:
        response_text: Raw response from VLM
    
    Returns:
        Parsed JSON dict (raw, unmodified except for timestamp)
    """
    # Extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        json_str = json_match.group(0)
    
    result = json.loads(json_str)
    return result


def _save_verifier_output_to_file(
    layer_id: str,
    verifier_output: Dict[str, Any],
    attempt_number: int,
    state: GraphState
) -> None:
    """
    [수정 32] Save verifier VLM output as JSON file in the layer directory.
    
    Args:
        layer_id: Current layer ID
        verifier_output: The full verifier output dict
        attempt_number: Verification attempt number
        state: Current graph state
    """
    try:
        episode_dir = state.get("episode_dir", ".")
        layer_dir = Path(episode_dir) / "layers" / layer_id
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Save verifier output with attempt number
        verifier_json_path = layer_dir / f"verifier_vlm_output_{attempt_number}.json"
        
        # Add metadata
        output_with_meta = {
            "layer_id": layer_id,
            "attempt_number": attempt_number,
            "depth": state.get("history_tree", {}).get(layer_id, {}).get("depth", 0),
            "parent_id": state.get("history_tree", {}).get(layer_id, {}).get("parent_id"),
            "action_type": state.get("history_tree", {}).get(layer_id, {}).get("action_type"),
            "timestamp": datetime.now().isoformat(),
            **verifier_output,
        }
        
        with open(verifier_json_path, "w", encoding="utf-8") as f:
            json.dump(output_with_meta, f, ensure_ascii=False, indent=2)
        
        _log(f"Saved verifier output to {verifier_json_path}")
        
    except Exception as e:
        _log(f"Warning: Failed to save verifier output: {e}")


def _call_verifier_vlm(
    parent_image_path: str,
    child_image_paths: List[str],
    layer_id: str,
    state: GraphState,
) -> Dict[str, Any]:
    """
    Call Verifier VLM to validate decomposition results.
    """
    # [수정] LLM 객체 로컬 생성 (타임아웃, 재시도 설정)
    verifier_llm = ChatOpenAI(
        model=os.environ.get("VLM_MODEL", "gemini-3-flash-preview"),
        base_url="https://gateway.letsur.ai/v1",
        temperature=0,
        model_kwargs={"top_p": 1},
        max_retries=3,       # 재시도 3회
        request_timeout=90,  # 타임아웃 90초
    )

    system_prompt = build_verifier_system_prompt()
    user_prompt = build_verifier_user_prompt(
        layer_id, state, len(child_image_paths)
    )
    
    content = [{"type": "text", "text": user_prompt}]
    
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_to_b64(parent_image_path)}"}
    })
    
    for child_path in child_image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_to_b64(child_path)}"}
        })
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ]
    
    # [수정] 로컬 객체 사용
    response = verifier_llm.invoke(messages)
    output = parse_verifier_response(response.content)
    
    # Add child paths for later reference
    output["_child_image_paths"] = child_image_paths
    
    # [추가] 메모리 정리
    del verifier_llm
    
    return output


# =============================================================================
# Image Processing Utilities
# =============================================================================

def _clean_transparent_pixels(arr: np.ndarray, alpha_threshold: int = 10) -> np.ndarray:
    """Zero out RGB values for pixels with low alpha."""
    result = arr.copy()
    transparent_mask = result[:, :, 3] < alpha_threshold
    result[transparent_mask, 0] = 0
    result[transparent_mask, 1] = 0
    result[transparent_mask, 2] = 0
    return result


def _create_layer_dir(episode_dir: str, layer_id: str) -> Path:
    """Create directory for a new layer."""
    layer_dir = Path(episode_dir) / "layers" / layer_id
    layer_dir.mkdir(parents=True, exist_ok=True)
    return layer_dir


def _extract_masked_image(src_path: str, mask_path: str, dest_path: str, alpha_threshold: int = 10) -> None:
    """Extracts the region defined by mask from src_path."""
    with Image.open(src_path) as src:
        src = src.convert("RGBA")
        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            
            if mask.size != src.size:
                mask = mask.resize(src.size, Image.NEAREST)
            
            src_arr = np.array(src)
            mask_arr = np.array(mask)
            
            src_a = src_arr[:, :, 3]
            final_a = np.minimum(src_a, mask_arr)
            
            result_arr = src_arr.copy()
            result_arr[:, :, 3] = final_a
            result_arr = _clean_transparent_pixels(result_arr, alpha_threshold=alpha_threshold)
            
            result = Image.fromarray(result_arr)
            result.save(dest_path)


def _copy_and_clean_layer_image(src_path: str, dest_path: str, alpha_threshold: int = 10) -> None:
    """Copy layer image while cleaning transparent pixels."""
    with Image.open(src_path) as img:
        img = img.convert("RGBA")
        arr = np.array(img)
        arr = _clean_transparent_pixels(arr, alpha_threshold=alpha_threshold)
        result = Image.fromarray(arr)
        result.save(dest_path)


# =============================================================================
# [수정 25] [수정 26] [수정 32] Core Verification Flow
# =============================================================================

def _execute_verification_flow(
    layer_id: str,
    parent_image_path: str,
    child_image_paths: List[str],
    state: GraphState
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute complete verification flow:
    
    1. Create temp children in history_tree
    2. Call Verifier VLM
    3. Save verifier output to JSON file
    4. Update temp children status
    5. Return verifier output and state updates
    
    [수정 26] CRITICAL FIX:
    - temp_update must be merged into state BEFORE r_update_temp_children_status()
    - Otherwise, temp children won't exist in the state when trying to update them
    
    [수정 32] Added JSON file saving for verifier output
    """
    tree = state.get("history_tree", {})
    node = tree.get(layer_id, {})
    
    # Calculate attempt number
    verification_attempts = node.get("verification_attempts", [])
    attempt_number = len(verification_attempts) + 1
    
    _log(f"=== Verification Flow (Attempt #{attempt_number}) for {layer_id} ===")
    _log(f"Children count: {len(child_image_paths)}")
    
    # Step 1: Create temp children in history_tree
    temp_update = r_create_temp_child_nodes(
        parent_id=layer_id,
        child_image_paths=child_image_paths,
        attempt_number=attempt_number,
        state=state
    )
    
    # [수정 26] ★ CRITICAL: temp_update를 state에 먼저 머지!
    # 이렇게 해야 r_update_temp_children_status()가 temp children을 찾을 수 있음
    temp_merged_state = merge_updates(state, temp_update)
    
    _log(f"Created temp children, tree size: {len(temp_merged_state.get('history_tree', {}))}")
    
    # Step 2: Call Verifier VLM
    if llm_can_call(state):
        verifier_output = _call_verifier_vlm(
            parent_image_path,
            child_image_paths,
            layer_id,
            state,
        )
        print(f"\n{'>'*20} VERIFIER VLM OUTPUT (Layer: {layer_id}) {'<'*20}")
        print(json.dumps(verifier_output, indent=2, ensure_ascii=False))
        print(f"{'='*60}\n")
        llm_update = r_llm_inc(state)
        
        # [수정 32] Save verifier output to JSON file
        _save_verifier_output_to_file(layer_id, verifier_output, attempt_number, state)
        
    else:
        _log("LLM budget exhausted, defaulting to PROCEED")
        verifier_output = {
            "decision": "PROCEED",
            "reasoning": "LLM budget exhausted",
            "children_analysis": [],
            "cross_child_duplicates": [],
            "valid_children_indices": list(range(len(child_image_paths))),
            "invalid_children_indices": [],
            "coverage_check": "COMPLETE",
            "_child_image_paths": child_image_paths,
        }
        llm_update = {}
    
    # Step 3: Update temp children status
    # [수정 26] ★ temp_merged_state를 사용해야 temp children을 찾을 수 있음!
    status_update = r_update_temp_children_status(
        parent_id=layer_id,
        verifier_output=verifier_output,
        state=temp_merged_state
    )
    
    _log(f"Status update keys: {list(status_update.get('history_tree', {}).keys())}")
    
    # Combine updates
    combined_update = r_pack_state(state, temp_update, llm_update, status_update)
    
    return verifier_output, combined_update


def _create_permanent_children(
    layer_id: str,
    child_image_paths: List[str],
    valid_indices: List[int],
    state: GraphState,
    pre_verification_update: Dict[str, Any],
    verifier_save_update: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create permanent child nodes for valid children.
    """
    tree = state.get("history_tree", {})
    node = tree.get(layer_id, {})
    action_type = node.get("action_type", "Unknown")
    
    episode_dir = state.get("episode_dir", ".")
    history_updates = {}
    increment_count = 0
    child_ids = []
    queue_ids = []
    
    for i, src_path in enumerate(child_image_paths):
        if not Path(src_path).exists():
            _log(f"WARNING: Child image not found: {src_path}")
            continue
        
        # Generate suffix based on action type
        if action_type == "Fork_Qwen":
            suffix = f"qwen_{valid_indices[i] if i < len(valid_indices) else i}"
        elif action_type == "Split_CCA":
            suffix = f"cca_{valid_indices[i] if i < len(valid_indices) else i}"
        elif action_type == "Split_DetSeg":
            suffix = f"obj_{valid_indices[i] if i < len(valid_indices) else i}"
        elif action_type == "Split_Text":
            suffix = "text" if i == 0 else "bg"
        else:
            suffix = f"child_{i}"
        
        child_id = generate_layer_id(state, suffix)
        layer_dir = _create_layer_dir(episode_dir, child_id)
        dest_path = layer_dir / "layer_image.png"
        
        _copy_and_clean_layer_image(src_path, str(dest_path))
        
        node_update = r_create_layer_node(
            layer_id=child_id,
            parent_id=layer_id,
            image_path=str(dest_path),
            state=state,
        )
        
        if "history_tree" in node_update:
            history_updates.update(node_update["history_tree"])
        if "_increment_layer_count" in node_update:
            increment_count += node_update["_increment_layer_count"]
        
        child_ids.append(child_id)
        queue_ids.append(child_id)
    
    if not child_ids:
        raise ValueError(f"No valid children created for {layer_id}")
    
    _log(f"Created {len(child_ids)} permanent children: {child_ids}")
    
    updates = {}
    if history_updates:
        updates["history_tree"] = history_updates
    if increment_count > 0:
        updates["_increment_layer_count"] = increment_count
    
    children_update = r_set_children_ids(layer_id, child_ids, state)
    queue_update = r_enqueue_layers(queue_ids, state)
    
    return r_pack_state(state, pre_verification_update, verifier_save_update,
                       updates, children_update, queue_update)


def _handle_retry(
    layer_id: str,
    verifier_output: Dict[str, Any],
    retry_reason: str,
    state: GraphState,
    pre_verification_update: Dict[str, Any],
    verifier_save_update: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle RETRY decision from Verifier.
    
    [수정 31] Creates a child node with same image instead of reusing same node.
    [수정 34] Passes current_failed_attempt directly to ensure complete history.
    [수정 37] Added retry_reason parameter for detailed failure tracking.
    [수정 38] Added params extraction for hyperparameter retry tracking.
    """
    _log(f"=== RETRY: Creating child node for {layer_id} ===")
    _log(f"Retry reason: {retry_reason}")
    
    tree = state.get("history_tree")
    node = tree.get(layer_id)
    
    action_type = node.get("action_type", "Unknown")
    tool_sequence = node.get("planned_tool_sequence", [])
    
    # [수정 38] Extract params for retry tracking
    params = {}
    if node.get("param_qwen_len") is not None:
        params["qwen_len"] = node.get("param_qwen_len")
    if node.get("param_is_photo") is not None:
        params["is_photo"] = node.get("param_is_photo")
    if node.get("param_inpaint_remainder") is not None:
        params["inpaint_remainder"] = node.get("param_inpaint_remainder")
    
    # [수정 37] Use the explicit retry_reason as the primary failure reason
    coverage = verifier_output.get("coverage_check", "UNKNOWN")
    valid_indices = verifier_output.get("valid_children_indices", [])
    invalid_indices = verifier_output.get("invalid_children_indices", [])
    children_analysis = verifier_output.get("children_analysis", [])
    
    # Build detailed failure info
    detailed_failures = []
    for idx in invalid_indices:
        for child in children_analysis:
            if child.get("index") == idx:
                failed_checks = []
                if child.get("hallucination_check") == "FAIL":
                    failed_checks.append("Hallucination")
                if child.get("redundancy_check") == "FAIL":
                    failed_checks.append("Redundancy")
                
                fail_detail = ", ".join(failed_checks) if failed_checks else "Unknown"
                detailed_failures.append(f"Child_{idx}: {fail_detail}")
    
    # Construct complete failure reason
    if detailed_failures:
        full_reason = f"{retry_reason} | Invalid children: {'; '.join(detailed_failures)}"
    else:
        full_reason = retry_reason
    
    # Build current_failed_attempt dict
    current_failed_attempt = {
        "action_type": action_type,
        "tool_sequence": tool_sequence,
        "params": params,  # [수정 38] Added params for hyperparameter tracking
        "failure_reason": full_reason,
        "verifier_decision": "RETRY",
        "coverage": coverage,
        "valid_children_count": len(valid_indices),
        "total_children_count": len(valid_indices) + len(invalid_indices),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Pass current_failed_attempt to ensure complete history
    retry_child_id, retry_child_update = r_create_retry_child_node(
        parent_id=layer_id,
        state=state,
        current_failed_attempt=current_failed_attempt,
    )
    
    # Enqueue the retry child node
    enqueue_update = r_enqueue_layers([retry_child_id], state)
    
    _log(f"Created retry child {retry_child_id} for {layer_id}")
    _log(f"Transferred failed_attempts count: {len(node.get('failed_attempts', [])) + 1}")
    _log(f"Params recorded: {params}")  # [수정 38] Log params for debugging
    
    return r_pack_state(state, pre_verification_update, verifier_save_update,
                       retry_child_update, enqueue_update)


# =============================================================================
# Action-Specific Handlers
# =============================================================================

def _handle_fork_qwen(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """Handle Fork_Qwen with verification."""
    _log(f"=== _handle_fork_qwen for {layer_id} ===")
    
    tree = state.get("history_tree", {})
    node = tree.get(layer_id, {})
    tool_outputs = node.get("tool_outputs", {})
    qwen_output = tool_outputs.get("qwen_layered")
    
    layer_images = qwen_output.get("layer_images", [])
    
    # Filter existing images
    layer_images = [p for p in layer_images if Path(p).exists()]
    
    parent_image = node.get("image_path", "")
    
    # Execute verification flow
    verifier_output, pre_update = _execute_verification_flow(
        layer_id, parent_image, layer_images, state
    )
    
    # [수정 37] Check if retry needed (returns Tuple[bool, str])
    should_retry, retry_reason = _should_retry_from_verification(verifier_output)
    
    if should_retry:
        verifier_save_update = r_save_verifier_output(layer_id, verifier_output, state)
        return _handle_retry(layer_id, verifier_output, retry_reason, state, 
                            pre_update, verifier_save_update)
    
    # Proceed with valid children
    valid_indices = verifier_output.get("valid_children_indices", [])
    valid_layer_images = [layer_images[i] for i in valid_indices 
                          if i < len(layer_images)]
    
    verifier_save_update = r_save_verifier_output(layer_id, verifier_output, state)
    
    # Create permanent children
    episode_dir = state.get("episode_dir", ".")
    history_updates = {}
    increment_count = 0
    child_ids = []
    queue_ids = []
    
    for i, src_path in enumerate(valid_layer_images):
        # Get original index for suffix
        original_idx = valid_indices[i] if i < len(valid_indices) else i
        suffix = f"qwen_{original_idx}"
        
        child_id = generate_layer_id(state, suffix)
        layer_dir = _create_layer_dir(episode_dir, child_id)
        dest_path = layer_dir / "layer_image.png"
        
        _copy_and_clean_layer_image(src_path, str(dest_path))
        
        node_update = r_create_layer_node(
            layer_id=child_id,
            parent_id=layer_id,
            image_path=str(dest_path),
            state=state,
        )
        
        if "history_tree" in node_update:
            history_updates.update(node_update["history_tree"])
        if "_increment_layer_count" in node_update:
            increment_count += node_update["_increment_layer_count"]
        
        child_ids.append(child_id)
        queue_ids.append(child_id)
    
    _log(f"Created {len(child_ids)} permanent children: {child_ids}")
    
    updates = {}
    if history_updates:
        updates["history_tree"] = history_updates
    if increment_count > 0:
        updates["_increment_layer_count"] = increment_count
    
    children_update = r_set_children_ids(layer_id, child_ids, state)
    queue_update = r_enqueue_layers(queue_ids, state)
    
    return r_pack_state(state, pre_update, verifier_save_update,
                       updates, children_update, queue_update)


def _handle_split_cca(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """
    Handle Split_CCA - Verification 스킵, 단 child 개수 < MIN_VALID_CHILDREN이면 retry.
    
    [수정] Split_CCA는 alpha mask connectivity 기반의 단순 공간 분리이므로:
    - Hallucination 불가능 (새로운 객체 생성 없음)
    - Redundancy 불가능 (각 component가 공간적으로 분리됨)
    - Coverage 항상 COMPLETE (원본의 모든 픽셀이 보존됨)
    따라서 verification VLM 호출은 스킵.
    
    단, child 개수가 2개 미만이면 분리 실패로 간주하고 retry 처리.
    """
    _log(f"=== _handle_split_cca for {layer_id} (Verification SKIP) ===")
    
    tree = state.get("history_tree", {})
    node = tree.get(layer_id, {})
    tool_outputs = node.get("tool_outputs", {})
    cca_output = tool_outputs.get("split_cca")
    
    components = cca_output.get("components", [])
    
    # Filter existing child images
    child_images = [comp.get("layer_path") for comp in components if comp.get("layer_path")]
    child_images = [p for p in child_images if Path(p).exists()]
    
    _log(f"Found {len(child_images)} CCA components")
    
    # =========================================================================
    # [수정] Child 개수가 MIN_VALID_CHILDREN 미만이면 retry
    # Split 액션은 최소 2개의 child가 필요함 (분리가 되어야 의미가 있음)
    # =========================================================================
    if len(child_images) < MIN_VALID_CHILDREN:
        _log(f"Only {len(child_images)} components (requires >= {MIN_VALID_CHILDREN}), triggering retry")
        
        # Create mock verifier output for retry handling
        mock_verifier_output = {
            "children_analysis": [],
            "valid_children_indices": list(range(len(child_images))),
            "invalid_children_indices": [],
            "coverage_check": "COMPLETE",  # CCA는 항상 complete
            "_child_image_paths": child_images,
        }
        
        retry_reason = f"Only {len(child_images)} CCA components (requires >= {MIN_VALID_CHILDREN} for split)"
        
        # Verification 스킵했으므로 pre_verification_update와 verifier_save_update는 빈 dict
        return _handle_retry(
            layer_id=layer_id,
            verifier_output=mock_verifier_output,
            retry_reason=retry_reason,
            state=state,
            pre_verification_update={},
            verifier_save_update={}
        )
    
    # =========================================================================
    # 정상 처리: 모든 component를 valid로 간주하고 permanent children 생성
    # =========================================================================
    episode_dir = state.get("episode_dir", ".")
    history_updates = {}
    increment_count = 0
    child_ids = []
    queue_ids = []
    
    for i, src_path in enumerate(child_images):
        suffix = f"cca_{i}"
        
        child_id = generate_layer_id(state, suffix)
        layer_dir = _create_layer_dir(episode_dir, child_id)
        dest_path = layer_dir / "layer_image.png"
        
        _copy_and_clean_layer_image(src_path, str(dest_path))
        
        node_update = r_create_layer_node(
            layer_id=child_id,
            parent_id=layer_id,
            image_path=str(dest_path),
            state=state,
        )
        
        if "history_tree" in node_update:
            history_updates.update(node_update["history_tree"])
        if "_increment_layer_count" in node_update:
            increment_count += node_update["_increment_layer_count"]
        
        child_ids.append(child_id)
        queue_ids.append(child_id)
    
    _log(f"Split_CCA complete (Verification SKIPPED):")
    _log(f"  - Created {len(child_ids)} permanent children: {child_ids}")
    
    # Build updates
    updates = {}
    if history_updates:
        updates["history_tree"] = history_updates
    if increment_count > 0:
        updates["_increment_layer_count"] = increment_count
    
    # Update parent node with verification skip status
    parent_update = {
        "history_tree": {
            layer_id: {
                "verification_status": "SKIP_SPLIT_CCA",
            }
        }
    }
    
    children_update = r_set_children_ids(layer_id, child_ids, state)
    queue_update = r_enqueue_layers(queue_ids, state)
    
    return r_pack_state(state, updates, parent_update, children_update, queue_update)


def _handle_split_detseg(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """
    Handle Split_DetSeg with verification.
    
    [수정 33] Added inpainting post-processing.
    """
    _log(f"=== _handle_split_detseg for {layer_id} ===")
    
    tree = state.get("history_tree", {})
    node = tree.get(layer_id, {})
    tool_outputs = node.get("tool_outputs", {})
    
    segment_list = tool_outputs.get("segment", [])
    
    seg_output = segment_list[-1].get("output", {})
    mask_union = seg_output.get("mask_union", "")
    masks_by_id = seg_output.get("masks_by_id", {})
    
    parent_image = node.get("image_path", "")
    current_image_bg = get_current_image_path(layer_id, state)
    original_image_fg = node.get("image_path", "") or current_image_bg
    
    # Get inpainted path if available
    inpainted_path = None
    refine_list = tool_outputs.get("refine", [])
    if refine_list:
        last_refine = refine_list[-1].get("output", {})
        for key in ["image_path", "output_path", "inpainted_path"]:
            if key in last_refine:
                inpainted_path = last_refine[key]
                break
    
    episode_dir = state.get("episode_dir", ".")
    temp_dir = Path(episode_dir) / "layers" / layer_id / "temp_verification"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # [수정 33] Apply transparency post-processing to inpainted image
    processed_inpaint_path = None
    if inpainted_path and Path(inpainted_path).exists():
        processed_inpaint_path = str(temp_dir / "inpaint_processed.png")
        _log(f"Applying transparency post-processing to inpainted image...")
        apply_transparency_to_inpainted_image(inpainted_path, processed_inpaint_path)
        _log(f"Processed inpainted image saved to {processed_inpaint_path}")
    
    # Create temporary child images for verification
    temp_child_paths = []
    temp_child_info = []  # (det_id, mask_path)
    
    for det_id, mask_path in masks_by_id.items():
        temp_path = temp_dir / f"temp_obj_{det_id}.png"
        _extract_masked_image(original_image_fg, mask_path, str(temp_path))
        temp_child_paths.append(str(temp_path))
        temp_child_info.append((det_id, mask_path))
    
    # Add background layer
    if mask_union and Path(mask_union).exists():
        temp_bg_path = temp_dir / "temp_bg.png"
        create_remainder_layer(
            current_image_bg, mask_union, str(temp_bg_path), 
            processed_inpaint_path or inpainted_path
        )
        temp_child_paths.append(str(temp_bg_path))
        temp_child_info.append(("bg", mask_union))
    
    # Execute verification flow
    verifier_output, pre_update = _execute_verification_flow(
        layer_id, parent_image, temp_child_paths, state
    )
    
    # [수정 37] Check if retry needed (returns Tuple[bool, str])
    should_retry, retry_reason = _should_retry_from_verification(verifier_output)
    
    if should_retry:
        verifier_save_update = r_save_verifier_output(layer_id, verifier_output, state)
        return _handle_retry(layer_id, verifier_output, retry_reason, state, 
                            pre_update, verifier_save_update)
    
    # Proceed with valid children
    valid_indices = verifier_output.get("valid_children_indices", [])
    
    verifier_save_update = r_save_verifier_output(layer_id, verifier_output, state)
    
    # Create permanent children from valid indices
    history_updates = {}
    increment_count = 0
    child_ids = []
    queue_ids = []
    
    for idx in valid_indices:
        if idx >= len(temp_child_info):
            continue
        
        det_id, mask_path = temp_child_info[idx]
        
        if det_id == "bg":
            child_id = generate_layer_id(state, "bg")
            layer_dir = _create_layer_dir(episode_dir, child_id)
            dest_path = layer_dir / "layer_image.png"
            create_remainder_layer(
                current_image_bg, mask_union, str(dest_path), 
                processed_inpaint_path or inpainted_path
            )
            child_ids.insert(0, child_id)  # BG first
        else:
            child_id = generate_layer_id(state, f"obj_{det_id}")
            layer_dir = _create_layer_dir(episode_dir, child_id)
            dest_path = layer_dir / "layer_image.png"
            _extract_masked_image(original_image_fg, mask_path, str(dest_path))
            child_ids.append(child_id)
        
        node_update = r_create_layer_node(
            layer_id=child_id,
            parent_id=layer_id,
            image_path=str(dest_path),
            state=state,
        )
        
        if "history_tree" in node_update:
            history_updates.update(node_update["history_tree"])
        if "_increment_layer_count" in node_update:
            increment_count += node_update["_increment_layer_count"]
        
        queue_ids.append(child_id)
    
    _log(f"Created {len(child_ids)} children: {child_ids}")
    
    updates = {}
    if history_updates:
        updates["history_tree"] = history_updates
    if increment_count > 0:
        updates["_increment_layer_count"] = increment_count
    
    children_update = r_set_children_ids(layer_id, child_ids, state)
    queue_update = r_enqueue_layers(queue_ids, state)
    
    return r_pack_state(state, pre_update, verifier_save_update,
                       updates, children_update, queue_update)


def _handle_split_text(layer_id: str, state: GraphState) -> Dict[str, Any]:
    """
    Handle Split_Text - Verification 스킵, Text layer pre-configured.
    
    [수정] 핵심 변경:
    1. Verification 플로우 완전 스킵
    2. Text Layer: pre-configured Finalize_Text (Router 스킵)
    3. Remainder Layer: 일반 child (Router 거침)
    4. 두 레이어 모두 큐에 추가 → 병렬 처리
    """
    _log(f"=== _handle_split_text for {layer_id} (Verification SKIP) ===")
    
    tree = state.get("history_tree", {})
    node = tree.get(layer_id, {})
    tool_outputs = node.get("tool_outputs", {})
    parent_depth = node.get("depth", 0)
    
    # Get segment output (HiSAM)
    segment_list = tool_outputs.get("segment", [])
    if not segment_list:
        _log(f"ERROR: No segment output for {layer_id}")
        return {"history_tree": {layer_id: {"error_info": {"message": "No segment output"}}}}
    
    seg_output = segment_list[-1].get("output", {})
    mask_union = seg_output.get("mask_union", "")
    
    # Get OCR output (for text content)
    detect_list = tool_outputs.get("detect", [])
    ocr_output = None
    for d in detect_list:
        if d.get("tool_name") == "ocr" or "ocr" in str(d.get("output", {})):
            ocr_output = d.get("output", {})
            break
    
    ocr_texts = ocr_output.get("texts", []) if ocr_output else []
    ocr_boxes = ocr_output.get("boxes", []) if ocr_output else []
    text_content = " | ".join(ocr_texts) if ocr_texts else "Extracted text"
    
    # Get image paths
    parent_image = node.get("image_path", "")
    current_image_bg = get_current_image_path(layer_id, state)
    original_image_fg = node.get("image_path", "") or current_image_bg
    
    # Get inpainted path
    inpainted_path = None
    refine_list = tool_outputs.get("refine", [])
    if refine_list:
        last_refine = refine_list[-1].get("output", {})
        for key in ["image_path", "output_path", "inpainted_path"]:
            if key in last_refine:
                inpainted_path = last_refine[key]
                break
    
    episode_dir = state.get("episode_dir", ".")
    
    # [수정 33] Apply transparency post-processing to inpainted image
    processed_inpaint_path = None
    if inpainted_path and Path(inpainted_path).exists():
        temp_dir = Path(episode_dir) / "layers" / layer_id / "temp_verification"
        temp_dir.mkdir(parents=True, exist_ok=True)
        processed_inpaint_path = str(temp_dir / "inpaint_processed.png")
        _log(f"Applying transparency post-processing to inpainted image...")
        apply_transparency_to_inpainted_image(inpainted_path, processed_inpaint_path)
    
    # =========================================================================
    # 1. Create Text Layer - PRE-CONFIGURED Finalize_Text (Router 스킵)
    # =========================================================================
    text_layer_id = generate_layer_id(state, "text")
    text_layer_dir = Path(episode_dir) / "layers" / text_layer_id
    text_layer_dir.mkdir(parents=True, exist_ok=True)
    
    text_image_path = text_layer_dir / "layer_image.png"
    _extract_masked_image(original_image_fg, mask_union, str(text_image_path))
    
    text_layer_node = {
        "layer_id": text_layer_id,
        "parent_id": layer_id,
        "depth": parent_depth + 1,
        "image_path": str(text_image_path),
        "image_context": f"[Auto-Finalize] Extracted text: {text_content[:100]}",
        "action_type": "Finalize_Text",  # ★ 미리 설정 - Router 스킵
        "action_reasoning": "Rule-based: text extracted by Split_Text action",
        "planned_tool_sequence": ["fontstyle", "finalize_text"],
        "node_queue": ["fontstyle", "finalize_text"],  # ★ 미리 설정
        "param_qwen_len": None,
        "param_is_photo": False,
        "param_inpaint_remainder": None,
        "param_nanobanana_instruction": None,
        "tool_outputs": {
            "qwen_layered": None,
            "split_cca": None,
            "vlm_front_pick": None,
            "detect": [{"tool_name": "ocr_inherited", "output": ocr_output}] if ocr_output else [],
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
        "verification_attempts": [],
        "verification_status": "SKIP_AUTO_FINALIZE",  # Verification 스킵 표시
        "rejected_child_indices": None,
        "failed_attempts": [],
        "_temp_child_ids": None,
        "_pending_verification": False,
        # Extra metadata for finalize_text
        "_ocr_texts": ocr_texts,
        "_ocr_boxes": ocr_boxes,
    }
    
    # =========================================================================
    # 2. Create Remainder (BG) Layer - 일반 child (Router 거침)
    # =========================================================================
    bg_layer_id = generate_layer_id(state, "bg")
    bg_layer_dir = Path(episode_dir) / "layers" / bg_layer_id
    bg_layer_dir.mkdir(parents=True, exist_ok=True)
    
    bg_image_path = bg_layer_dir / "layer_image.png"
    create_remainder_layer(
        current_image_bg, mask_union, str(bg_image_path),
        processed_inpaint_path or inpainted_path
    )
    
    bg_layer_node = {
        "layer_id": bg_layer_id,
        "parent_id": layer_id,
        "depth": parent_depth + 1,
        "image_path": str(bg_image_path),
        "image_context": None,  # Router가 분석
        "action_type": None,    # Router가 결정
        "action_reasoning": None,
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
        "verification_attempts": [],
        "verification_status": "pending",
        "rejected_child_indices": None,
        "failed_attempts": [],
        "_temp_child_ids": None,
        "_pending_verification": False,
    }
    
    # =========================================================================
    # 3. Update parent and return
    # =========================================================================
    child_ids = [bg_layer_id, text_layer_id]  # BG first, then text
    
    _log(f"Split_Text complete (Verification SKIPPED):")
    _log(f"  - Text Layer: {text_layer_id} (pre-configured Finalize_Text)")
    _log(f"  - BG Layer: {bg_layer_id} (will go through Router)")
    _log(f"  - Both enqueued for parallel processing")
    
    # Build update
    history_updates = {
        text_layer_id: text_layer_node,
        bg_layer_id: bg_layer_node,
        layer_id: {
            "children_ids": child_ids,
            "verification_status": "SKIP_SPLIT_TEXT",
        }
    }
    
    queue_update = r_enqueue_layers([text_layer_id, bg_layer_id], state)
    
    return r_pack_state(
        state,
        {"history_tree": history_updates},
        {"_increment_layer_count": 2},
        queue_update
    )



# =============================================================================
# Main Node Function
# =============================================================================

def node(state: GraphState) -> Dict[str, Any]:
    """
    Stack Manager node - creates child layers with Verifier VLM validation.
    """
    layer_id = state.get("current_layer_id")
    tree = state.get("history_tree")
    
    _, dequeue_update = r_dequeue_node(layer_id, state)
    action_type = tree[layer_id].get("action_type", "")
    
    _log(f"Processing {layer_id} with action: {action_type}")

    if action_type == "Fork_Qwen":
        result = _handle_fork_qwen(layer_id, state)
    elif action_type == "Split_CCA":
        result = _handle_split_cca(layer_id, state)
    elif action_type == "Split_DetSeg":
        result = _handle_split_detseg(layer_id, state)
    elif action_type == "Split_Text":
        result = _handle_split_text(layer_id, state)
    else:
        raise ValueError(f"Unknown action type for stack_manager: {action_type}")
    
    return r_pack_state(state, dequeue_update, result)