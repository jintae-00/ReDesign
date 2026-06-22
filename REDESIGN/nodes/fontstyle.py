# REDESIGN/nodes/fontstyle.py
"""
FontStyle Node - DUMMY VERSION

Agent 파이프라인 실행 중에는 dummy 값을 반환합니다.
실제 폰트 피팅은 추후 일괄 보충 스크립트로 수행합니다.

Color 추출은 로컬에서 빠르게 수행되므로 유지합니다.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import gc
import json

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
    get_tight_bbox_from_alpha,
    boxes_to_aabbs,
)


def _log(message: str):
    print(f"[FontStyle-Dummy] {message}")


# =============================================================================
# Image Processing Utilities (색상 추출 - 빠르므로 유지)
# =============================================================================

def _extract_color_from_image(image_path: str, bbox: List[int] = None) -> Tuple[int, int, int]:
    """Extract dominant color from text image using median of visible pixels"""
    try:
        img = Image.open(image_path)
        img_np = np.array(img)
        
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            img_np = img_np[y1:y2, x1:x2]
        
        if img_np.ndim == 2:
            img_np = np.stack((img_np,) * 3, axis=-1)
        
        if img_np.shape[2] == 4:
            rgb = img_np[:, :, :3]
            alpha = img_np[:, :, 3]
            valid = rgb[alpha > 0]
        else:
            valid = img_np.reshape(-1, 3)
        
        if valid.size == 0:
            return (0, 0, 0)
        
        median = np.median(valid, axis=0).astype(int)
        return tuple(map(int, median))
    except Exception:
        return (0, 0, 0)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb[:3])


def _crop_text_region(image_path: str, bbox: List[int], output_path: str) -> bool:
    """Crop a text region from image based on bbox"""
    try:
        img = Image.open(image_path).convert("RGBA")
        x1, y1, x2, y2 = bbox
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(output_path)
        return True
    except Exception:
        return False


def _estimate_font_size_from_bbox(bbox: List[int]) -> float:
    """BBox 높이 기반 대략적 폰트 사이즈 추정"""
    if not bbox or len(bbox) != 4:
        return 24.0
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    return max(12.0, height * 0.75)


# =============================================================================
# Node Implementation - DUMMY VERSION
# =============================================================================

def node(state: GraphState) -> Dict[str, Any]:
    """
    FontStyle node - DUMMY VERSION
    
    실제 API 호출/폰트 피팅 없이 dummy 값을 반환합니다.
    - Color 추출: 실제 수행 (로컬, 빠름)
    - Font family/size/l1_loss 등: dummy 값
    
    추후 backfill 스크립트에서 cropped_image_path, text_content, color를 사용해
    실제 폰트 피팅을 수행합니다.
    """
    layer_id = state.get("current_layer_id")
    if not layer_id:
        return {"error": "No current layer ID"}
    
    tree = state.get("history_tree", {})
    if layer_id not in tree:
        return {"error": f"Layer {layer_id} not in history_tree"}
    
    _log(f"========== FontStyle Node (DUMMY): {layer_id} ==========")
    
    # Dequeue this node
    _, dequeue_update = r_dequeue_node(layer_id, state)
    
    # Get current layer image
    image_path = get_current_image_path(layer_id, state)
    if not image_path or not Path(image_path).exists():
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "Image not found", {"path": image_path}, state)
        )
    
    _log(f"Image path: {image_path}")
    
    # Get OCR results from ancestors
    ocr_output = find_latest_tool_output(layer_id, "detect", "detect_ocr", state)
    _log(f"OCR output found: {ocr_output is not None}")
    
    # Get segmentation masks by det_id
    seg_output = find_latest_tool_output(layer_id, "segment", None, state)
    masks_by_id = {}
    if seg_output:
        masks_by_id = seg_output.get("masks_by_id", {})
    
    # Setup directories
    episode_dir = state.get("episode_dir", ".")
    layer_dir = Path(episode_dir) / "layers" / layer_id / "tools_output"
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each text element with DUMMY values
    element_results = []
    
    if ocr_output:
        boxes = ocr_output.get("boxes", [])
        texts = ocr_output.get("texts", [])
        scores = ocr_output.get("scores", [])
        det_ids = ocr_output.get("det_ids", [])
        
        aabbs = boxes_to_aabbs(boxes)
        
        _log(f"Processing {len(aabbs)} text elements (DUMMY mode)...")
        
        for idx, (bbox, text, det_id) in enumerate(zip(aabbs, texts, det_ids)):
            # Crop this specific text region (실제 수행 - 추후 피팅에 필요)
            cropped_path = str(layer_dir / f"fontstyle_crop_{det_id}.png")
            
            if not _crop_text_region(image_path, bbox, cropped_path):
                _log(f"Failed to crop text region for {det_id}")
                continue
            
            # 색상 추출 (실제 수행 - 빠름)
            color_rgb = _extract_color_from_image(cropped_path)
            hex_color = _rgb_to_hex(color_rgb)
            
            # 폰트 사이즈 추정 (bbox 기반)
            estimated_size = _estimate_font_size_from_bbox(bbox)
            
            # Get mask path for this element
            mask_path = masks_by_id.get(det_id)
            
            # ★ DUMMY 값으로 element_result 생성
            element_result = {
                "det_id": det_id,
                "text_content": text,
                "bbox": bbox,
                # === DUMMY VALUES (추후 일괄 보충) ===
                "font_family": "__DUMMY__",
                "size_px": estimated_size,  # bbox 기반 추정값
                "color": {
                    "rgb": list(color_rgb),  # ★ 실제 추출값
                    "hex": hex_color,        # ★ 실제 추출값
                },
                "bold": False,
                "italic": False,
                "angle_deg": 0,
                "l1_loss": float('inf'),  # 피팅 안됨 표시
                # === PATHS (추후 피팅에 사용) ===
                "cropped_image_path": cropped_path,  # ★ 핵심: 추후 API 호출에 사용
                "rendered_image_path": None,  # 피팅 후 생성됨
                "font_file_path": None,       # 피팅 후 생성됨
                "mask_path": mask_path,
                "score": scores[idx] if idx < len(scores) else None,
                # === DUMMY FLAG ===
                "_is_dummy": True,
            }
            
            element_results.append(element_result)
            _log(f"  [{det_id}] '{text[:20]}...' color={hex_color}, est_size={estimated_size:.0f}px (DUMMY)")
    
    # If no OCR output, process as single element
    if not element_results:
        _log(f"No OCR output, processing as single element (DUMMY)")
        
        bbox = get_tight_bbox_from_alpha(image_path)
        if bbox:
            color_rgb = _extract_color_from_image(image_path, bbox)
            hex_color = _rgb_to_hex(color_rgb)
            estimated_size = _estimate_font_size_from_bbox(bbox)
            
            cropped_path = str(layer_dir / "fontstyle_crop_single.png")
            _crop_text_region(image_path, bbox, cropped_path)
            
            element_results.append({
                "det_id": "single",
                "text_content": "",
                "bbox": bbox,
                "font_family": "__DUMMY__",
                "size_px": estimated_size,
                "color": {"rgb": list(color_rgb), "hex": hex_color},
                "bold": False,
                "italic": False,
                "angle_deg": 0,
                "l1_loss": float('inf'),
                "cropped_image_path": cropped_path,
                "rendered_image_path": None,
                "font_file_path": None,
                "mask_path": None,
                "score": None,
                "_is_dummy": True,
            })
    
    # Build output
    output = {
        "tool_name": "fontstyle",
        "elements": element_results,
        "num_elements": len(element_results),
        "_dummy_mode": True,  # 전체 출력이 dummy임을 표시
    }
    
    # Save fontstyle output to JSON
    fontstyle_json_path = layer_dir / "fontstyle_output.json"
    with open(fontstyle_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    _log(f"========== FontStyle Complete (DUMMY): {len(element_results)} elements ==========")
    
    # Save tool output
    tool_update = r_save_tool_output(
        layer_id=layer_id,
        tool_category="fontstyle",
        tool_name="fontstyle",
        output=output,
        state=state,
    )
    
    gc.collect()
    
    return r_pack_state(state, dequeue_update, tool_update)