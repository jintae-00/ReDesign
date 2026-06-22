# REDESIGN/utils.py
"""
Utility functions for URLD (Unified Recursive Layer Decomposition)

Provides helpers for:
- Image path resolution (considering refine outputs)
- Ancestor traversal for tool output lookup
- Bounding box conversions
- JSON extraction from LLM responses
- Image manipulation utilities

[수정 33] Added apply_transparency_to_inpainted_image:
- Converts RGB inpainted images back to RGBA
- Uses flood-fill to remove outer black background
- Preserves internal black pixels within objects
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import base64
import json
import re
import numpy as np
import shutil       
import cv2          
from PIL import Image


# =============================================================================
# Image Path Resolution
# =============================================================================


def get_current_image_path(layer_id: str, state: Dict[str, Any]) -> str:
    """
    Get the current working image for a layer.
    Priority: last refine output > original layer image
    """
    tree = state.get("history_tree", {})
    if layer_id not in tree:
        return ""
    
    node = tree[layer_id]
    tool_outputs = node.get("tool_outputs") or {}
    refine_list = tool_outputs.get("refine") or []
    
    # If refine outputs exist, use the last one
    if refine_list:
        last_refine = refine_list[-1]
        output = last_refine.get("output", {})
        for key in ["image_path", "output_path", "inpainted_path"]:
            if key in output and output[key]:
                return output[key]
    
    # Fall back to original layer image
    return node.get("image_path", "")


def get_root_image_path(state: Dict[str, Any]) -> str:
    """Get the root image path for reference comparison"""
    return state.get("root_image_path", "")


def get_tools_output_dir(state: Dict[str, Any], layer_id: str) -> Path:
    """Get the directory for saving tool outputs for a specific layer."""
    episode_dir = Path(state.get("episode_dir", "."))
    tools_dir = episode_dir / "layers" / layer_id / "tools_output"
    tools_dir.mkdir(parents=True, exist_ok=True)
    return tools_dir


def analyze_and_convert_image(
    image_path: str,
    output_dir: str,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Dict[str, Any]:
    """
    입력 이미지를 분석하고, RGBA인 경우 alpha 마스크를 저장 후 RGB로 변환.
    
    Args:
        image_path: 원본 이미지 경로
        output_dir: 출력 디렉토리 (alpha 마스크 저장용)
        background_color: 투명 영역에 사용할 배경색
    
    Returns:
        {
            "rgb_image_path": RGB 변환된 이미지 경로,
            "original_image_path": 원본 이미지 경로,
            "has_alpha": 알파 채널 존재 여부,
            "alpha_mask_path": 알파 마스크 경로 (없으면 None),
            "original_mode": 원본 이미지 모드,
            "original_size": (width, height),
        }
    """
    img = Image.open(image_path)
    original_mode = img.mode
    original_size = img.size
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result = {
        "original_image_path": str(image_path),
        "original_mode": original_mode,
        "original_size": original_size,
        "has_alpha": False,
        "alpha_mask_path": None,
        "rgb_image_path": None,
    }
    
    # Case 1: 이미 RGB - 그대로 사용
    if img.mode == "RGB":
        rgb_image_path = output_path / "layer_image.png"
        img.save(rgb_image_path)
        result["rgb_image_path"] = str(rgb_image_path)
        return result
    
    # Case 2: RGBA 또는 알파 채널이 있는 모드
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        # RGBA로 통일
        rgba_img = img.convert("RGBA")
        alpha_channel = np.array(rgba_img)[:, :, 3]
        
        # 실제로 투명 픽셀이 있는지 확인
        has_transparency = np.any(alpha_channel < 255)
        
        if has_transparency:
            result["has_alpha"] = True
            
            # 1. Alpha 마스크 저장 (grayscale PNG)
            alpha_mask_path = output_path / "original_alpha_mask.png"
            alpha_mask_img = Image.fromarray(alpha_channel, mode="L")
            alpha_mask_img.save(alpha_mask_path)
            result["alpha_mask_path"] = str(alpha_mask_path)
            
            # 2. 흰색 배경에 합성하여 RGB 변환
            bg_rgba = (*background_color, 255)
            background = Image.new("RGBA", rgba_img.size, bg_rgba)
            composited = Image.alpha_composite(background, rgba_img)
            rgb_img = composited.convert("RGB")
            
            # 3. RGB 이미지 저장
            rgb_image_path = output_path / "layer_image.png"
            rgb_img.save(rgb_image_path)
            result["rgb_image_path"] = str(rgb_image_path)
        else:
            # 투명 픽셀 없음 - 단순 RGB 변환
            rgb_img = rgba_img.convert("RGB")
            rgb_image_path = output_path / "layer_image.png"
            rgb_img.save(rgb_image_path)
            result["rgb_image_path"] = str(rgb_image_path)
    else:
        # 기타 모드 (L, 1, etc.) - RGB로 변환
        rgb_img = img.convert("RGB")
        rgb_image_path = output_path / "layer_image.png"
        rgb_img.save(rgb_image_path)
        result["rgb_image_path"] = str(rgb_image_path)
    
    return result


def apply_mask_to_image_and_crop(
    image_path: str, 
    mask_path: str, 
    bbox: List[int], 
    output_path: str
) -> bool:
    """
    원본 이미지의 R,G,B,A 값을 보존하면서 마스크를 적용해 Crop합니다.
    """
    try:
        # 1. 원본 RGBA 로드 (무손실 시작)
        img = Image.open(image_path).convert("RGBA")
        mask = Image.open(mask_path).convert("L")
        
        # 2. 마스크 리사이즈 시 NEAREST를 사용하여 새로운 픽셀값 생성을 방지
        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)
            
        data = np.array(img)
        mask_arr = np.array(mask)
        
        # 3. 마스크 이진화 (0 또는 255로 명확히 분리)
        binary_mask = np.where(mask_arr > 127, 255, 0).astype(np.uint8)
        
        # 4. Alpha 채널만 수정: 마스크 밖(0)인 부분만 투명하게 만듦
        # 마스크 안(255)인 부분은 원본 Alpha 값이 그대로 유지됨 (min(orig_A, 255) = orig_A)
        data[:, :, 3] = np.minimum(data[:, :, 3], binary_mask)
        
        masked_img = Image.fromarray(data)
        
        # 5. Crop
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        
        if x2 > x1 and y2 > y1:
            cropped = masked_img.crop((x1, y1, x2, y2))
            cropped.save(output_path, "PNG") # 무손실 저장
            return True
        return False
    except Exception as e:
        print(f"[utils] apply_mask_to_image_and_crop failed: {e}")
        return False


def apply_alpha_mask_to_reconstruction(
    reconstructed_rgb_path: str,
    alpha_mask_path: str,
    output_path: str,
) -> str:
    """
    Reconstruction 결과에 원본 alpha 마스크를 적용하여 RGBA로 저장.
    
    Args:
        reconstructed_rgb_path: RGB reconstruction 이미지 경로
        alpha_mask_path: 원본 alpha 마스크 경로
        output_path: RGBA 출력 경로
    
    Returns:
        저장된 RGBA 이미지 경로
    """
    rgb_img = Image.open(reconstructed_rgb_path).convert("RGB")
    alpha_mask = Image.open(alpha_mask_path).convert("L")
    
    # 크기가 다르면 alpha 마스크를 리사이즈
    if rgb_img.size != alpha_mask.size:
        alpha_mask = alpha_mask.resize(rgb_img.size, Image.Resampling.LANCZOS)
    
    # RGB + Alpha 합성
    rgba_img = rgb_img.copy()
    rgba_img.putalpha(alpha_mask)
    
    # 저장
    rgba_img.save(output_path)
    
    return str(output_path)


# =============================================================================
# Ancestor Traversal for Tool Outputs
# =============================================================================

def get_ancestor_chain(layer_id: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get chain of ancestors from root to current layer (exclusive).
    Returns list of lightweight LayerNode dicts.
    """
    tree = state.get("history_tree", {})
    if layer_id not in tree:
        return []
    
    chain = []
    current = tree[layer_id]
    parent_id = current.get("parent_id")
    
    while parent_id and parent_id in tree:
        parent_node = tree[parent_id]
        lightweight = {
            "layer_id": parent_node.get("layer_id"),
            "parent_id": parent_node.get("parent_id"),
            "depth": parent_node.get("depth"),
            "image_context": parent_node.get("image_context"),
            "action_reasoning": parent_node.get("action_reasoning"),
            "action_type": parent_node.get("action_type"),
            "planned_tool_sequence": parent_node.get("planned_tool_sequence"),
            "param_qwen_len": parent_node.get("param_qwen_len"),
            "param_is_photo": parent_node.get("param_is_photo"),
            "param_inpaint_remainder": parent_node.get("param_inpaint_remainder"),
            "children_ids": parent_node.get("children_ids"),
        }
        chain.append(lightweight)
        parent_id = parent_node.get("parent_id")
    
    chain.reverse()
    return chain


def find_latest_tool_output(
    layer_id: str,
    tool_category: str,
    tool_name: Optional[str],
    state: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Find the latest tool output by traversing from current layer up to ancestors.
    """
    tree = state.get("history_tree", {})
    current_id = layer_id
    
    while current_id and current_id in tree:
        node = tree[current_id]
        tool_outputs = node.get("tool_outputs") or {}
        
        if tool_category in ["detect", "segment", "refine"]:
            outputs_list = tool_outputs.get(tool_category) or []
            if outputs_list:
                if tool_name:
                    for item in reversed(outputs_list):
                        if item.get("tool_name") == tool_name:
                            return item.get("output")
                else:
                    return outputs_list[-1].get("output")
        else:
            output = tool_outputs.get(tool_category)
            if output:
                return output
        
        current_id = node.get("parent_id")
    
    return None


def get_vlm_labels(layer_id: str, state: Dict[str, Any]) -> List[str]:
    """Get VLM front pick labels"""
    output = find_latest_tool_output(layer_id, "vlm_front_pick", None, state)
    if not output:
        return []
    return output.get("labels") or []


def get_detection_boxes(layer_id: str, state: Dict[str, Any]) -> Tuple[List, List, List]:
    """Get detection boxes. Returns (boxes, det_ids, labels/texts)."""
    output = find_latest_tool_output(layer_id, "detect", None, state)
    if not output:
        return [], [], []
    
    boxes = output.get("boxes") or []
    det_ids = output.get("det_ids") or []
    labels = output.get("labels") or output.get("texts") or []
    
    return boxes, det_ids, labels


def get_segmentation_masks(layer_id: str, state: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """Get segmentation mask paths. Returns (mask_union_path, masks_by_id)."""
    output = find_latest_tool_output(layer_id, "segment", None, state)
    if not output:
        return "", {}
    
    mask_union = output.get("mask_union") or ""
    masks_by_id = output.get("masks_by_id") or {}
    
    return mask_union, masks_by_id


# =============================================================================
# Bounding Box Conversions
# =============================================================================

def quad_to_aabb(quad: List) -> List[int]:
    """Convert rotated quad [[x,y], ...] to axis-aligned bounding box [x1, y1, x2, y2]"""
    pts = np.array(quad, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def boxes_to_aabbs(boxes: List) -> List[List[int]]:
    """Convert list of boxes (potentially rotated) to AABBs"""
    out = []
    for b in (boxes or []):
        if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
            x1, y1, x2, y2 = b
            out.append([int(x1), int(y1), int(x2), int(y2)])
        else:
            out.append(quad_to_aabb(b))
    return out


def get_tight_bbox_from_alpha(image_path: str) -> Optional[List[int]]:
    """
    Get tight bounding box from RGBA image alpha channel.
    [수정] Handles RGB-on-Black by applying smart transparency logic internally.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
        data = np.array(img)
        
        # [수정] 불투명 이미지라면 스마트 마스크 계산하여 BBox 측정
        if np.min(data[:, :, 3]) == 255:
            alpha = get_smart_transparency_mask(data)
        else:
            alpha = data[:, :, 3]
            
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        return [int(x1), int(y1), int(x2) + 1, int(y2) + 1]
    except Exception:
        return None


# =============================================================================
# Image Utilities
# =============================================================================

def image_to_b64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def crop_to_tight_bbox(image_path: str, output_path: str) -> Tuple[str, List[int]]:
    """
    Crop RGBA image to tight bbox and save. 
    [수정] Saves with smart transparency applied if input was RGB-on-Black.
    """
    # 1. BBox 계산 (스마트 로직 포함됨)
    bbox = get_tight_bbox_from_alpha(image_path)
    if not bbox:
        # BBox가 없으면 원본 그대로 저장 (혹은 투명화만 해서 저장)
        convert_black_to_transparent(image_path, output_path)
        return image_path, [0, 0, 0, 0]
    
    # 2. 이미지 로드 및 투명화 적용
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)
    
    # [수정] 불투명 이미지라면 배경 날리기
    if np.min(data[:, :, 3]) == 255:
        new_alpha = get_smart_transparency_mask(data)
        data[:, :, 3] = new_alpha
        img = Image.fromarray(data)
    
    # 3. Crop & Save
    x1, y1, x2, y2 = bbox
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(output_path)
    
    return output_path, bbox


def create_canvas_layer_from_mask(
    original_image_path: str,
    mask_path: str,
    output_path: str
) -> str:
    """Create canvas-size layer by masking original image"""
    img = Image.open(original_image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    
    if img.size != mask.size:
        mask = mask.resize(img.size, Image.NEAREST)
    
    img_arr = np.array(img)
    mask_arr = np.array(mask)
    
    img_arr[:, :, 3] = np.where(mask_arr > 127, img_arr[:, :, 3], 0)
    
    result = Image.fromarray(img_arr)
    result.save(output_path)
    return output_path


def create_remainder_layer(
    original_image_path: str,
    union_mask_path: str,
    output_path: str,
    inpainted_path: Optional[str] = None
) -> str:
    """
    Create remainder (background) layer.
    If inpainted_path is provided, use it as base.
    Otherwise, create transparent hole in original.
    
    [수정 6] CRITICAL FIX: Now clears RGB channels too, not just alpha!
    
    Previous behavior:
    - Only set alpha=0 for masked regions
    - RGB data remained → "ghost" pixels visible in some viewers/editors
    
    New behavior:
    - Set R=G=B=A=0 for masked regions
    - Completely transparent with no color data
    
    This ensures clean separation even without inpainting.
    """
    # Case 1: Use inpainted image if available
    if inpainted_path and Path(inpainted_path).exists():
        shutil.copy(inpainted_path, output_path)
        return output_path
    
    # Case 2: Create transparent hole in original
    img = Image.open(original_image_path).convert("RGBA")
    mask = Image.open(union_mask_path).convert("L")
    
    if img.size != mask.size:
        mask = mask.resize(img.size, Image.NEAREST)
    
    img_arr = np.array(img)
    mask_arr = np.array(mask)
    
    # [수정 6] Clear ALL channels (R, G, B, A) for masked regions
    # This prevents "ghost" pixels where alpha=0 but RGB has color data
    mask_bool = mask_arr > 127
    img_arr[mask_bool, 0] = 0  # R
    img_arr[mask_bool, 1] = 0  # G
    img_arr[mask_bool, 2] = 0  # B
    img_arr[mask_bool, 3] = 0  # A
    
    result = Image.fromarray(img_arr)
    result.save(output_path)
    return output_path


# =============================================================================
# JSON Extraction from LLM
# =============================================================================

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response text"""
    if not text:
        return None
    
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                clean = match.strip()
                if not clean.startswith('{'):
                    start = clean.find('{')
                    end = clean.rfind('}')
                    if start >= 0 and end > start:
                        clean = clean[start:end+1]
                
                return json.loads(clean)
            except json.JSONDecodeError:
                continue
    
    return None


# =============================================================================
# GPU Slot Requirements by Action Type
# =============================================================================

ACTION_GPU_REQUIREMENTS = {
    "Fork_Qwen": 2,
    "Split_DetSeg": 1,
    "Split_Text": 1,
    "Split_CCA": 0,
    "Finalize_Text": 0,
    "Finalize_Obj": 0,
}

def get_gpu_requirement(action_type: str) -> int:
    """Get GPU slot requirement for an action type"""
    return ACTION_GPU_REQUIREMENTS.get(action_type, 1)


# =============================================================================
# [수정 33] Smart Transparency Functions
# =============================================================================

def get_smart_transparency_mask(img_array: np.ndarray) -> np.ndarray:
    """
    Generate a mask where Background=0 (Transparent) and Object=255 (Opaque).
    Uses Flood Fill from corners to differentiate 'background black' from 'object black'.
    
    This function identifies outer black regions connected to the image edges
    and marks them as transparent, while preserving internal black pixels.
    """
    h, w = img_array.shape[:2]
    rgb = img_array[:, :, :3]
    
    # 1. '검은색'으로 간주할 픽셀 정의 (노이즈 허용 범위 < 5)
    is_black = np.all(rgb <= 5, axis=2).astype(np.uint8) * 255
    
    # 2. Flood Fill을 위한 마스크 준비
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    background_map = is_black.copy()
    
    # 3. 네 귀퉁이에서 시작하여 연결된 검은색 영역 찾기
    corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
    for r, c in corners:
        if background_map[r, c] == 255:  # 시작점이 검은색이면
            # 128로 채워서 '배경'임을 표시
            cv2.floodFill(background_map, flood_mask, (c, r), 128)
    
    # 4. 추가: 가장자리 전체에서 시작하여 연결된 검은색 영역 찾기
    # (귀퉁이만으로는 부족할 수 있음)
    # 상단 가장자리
    for c in range(w):
        if background_map[0, c] == 255:
            cv2.floodFill(background_map, flood_mask, (c, 0), 128)
    # 하단 가장자리
    for c in range(w):
        if background_map[h-1, c] == 255:
            cv2.floodFill(background_map, flood_mask, (c, h-1), 128)
    # 좌측 가장자리
    for r in range(h):
        if background_map[r, 0] == 255:
            cv2.floodFill(background_map, flood_mask, (0, r), 128)
    # 우측 가장자리
    for r in range(h):
        if background_map[r, w-1] == 255:
            cv2.floodFill(background_map, flood_mask, (w-1, r), 128)
            
    # 5. Alpha 채널 생성
    # 128 (배경 검은색) -> 0 (투명)
    # 255 (객체 내부 검은색) -> 255 (불투명)
    # 0 (색상 영역) -> 255 (불투명)
    final_alpha = np.where(background_map == 128, 0, 255).astype(np.uint8)
    
    return final_alpha


def convert_black_to_transparent(image_path: str, output_path: str) -> None:
    """
    Load image, apply smart transparency if it's opaque RGB, and save.
    Replaces shutil.copy for canvas images.
    """
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)
    
    # 이미지가 완전히 불투명한 경우(Raw RGB 등)에만 로직 적용
    if np.min(data[:, :, 3]) == 255:
        new_alpha = get_smart_transparency_mask(data)
        data[:, :, 3] = new_alpha
        img = Image.fromarray(data)
            
    img.save(output_path)


def apply_transparency_to_inpainted_image(
    inpainted_rgb_path: str,
    output_rgba_path: str,
    shadow_threshold_start: int = 10,
    core_threshold: int = 20,
    min_canvas_ratio: float = 0.001,
    blur_amount: int = 3,
    bright_neighbor_threshold: int = 200,
    bright_neighbor_ratio: float = 0.3,
    min_dark_object_area: int = 50  # [NEW] 검정 객체 최소 면적 (성능 최적화)
) -> str:
    """
    [Shadow-Preserving + Black Object Preservation Version]
    
    Flood Fill 기반 배경 제거 + 밝은 배경에 둘러싸인 검정 객체 보존 로직.
    
    1. Flood Fill: 네 귀퉁이에서 시작하여 연결된 검정색만 배경으로 인식
    2. Core Mask: 밝기 기준으로 본체 영역 보호
    3. Black Object Recovery: 밝은 배경에 둘러싸인 검정 객체 복구 (최소 면적 필터 적용)
    4. Soft Alpha: 그림자 영역에 부드러운 투명도 적용
    5. Morphological Opening: 미세 노이즈 제거
    """
    try:
        img = cv2.imread(inpainted_rgb_path)
        if img is None:
            raise ValueError(f"Image load failed: {inpainted_rgb_path}")

        h, w = img.shape[:2]
        total_canvas_area = h * w
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---------------------------------------------------------
        # Step A: Flood Fill (배경 vs 전경 분리)
        # ---------------------------------------------------------
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
        
        for x, y in corners:
            if flood_mask[y+1, x+1] == 0: 
                cv2.floodFill(img, flood_mask, (x, y), 0, 
                              (shadow_threshold_start,)*3, (shadow_threshold_start,)*3, flags)

        extent_mask = cv2.bitwise_not(flood_mask[1:h+1, 1:w+1])

        # 노이즈 제거 (Size Filter)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(extent_mask, connectivity=8)
        if num_labels > 1:
            min_area = total_canvas_area * min_canvas_ratio
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    extent_mask[labels == i] = 0

        # ---------------------------------------------------------
        # Step B: Core Mask (단단한 본체 보호)
        # ---------------------------------------------------------
        _, core_binary = cv2.threshold(gray, core_threshold, 255, cv2.THRESH_BINARY)
        core_mask = np.zeros_like(gray)
        contours, _ = cv2.findContours(core_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0]
            cv2.drawContours(core_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

        # ---------------------------------------------------------
        # Step B-2: 밝은 배경에 둘러싸인 검정 객체 복구 (최적화)
        # ---------------------------------------------------------
        dark_in_extent = ((extent_mask == 255) & (gray < core_threshold)).astype(np.uint8) * 255
        
        num_dark_labels, dark_labels, dark_stats, _ = cv2.connectedComponentsWithStats(
            dark_in_extent, connectivity=8
        )
        
        # [최적화] 최소 면적 이상인 component만 처리 (성능 대폭 향상)
        large_dark_indices = [
            i for i in range(1, num_dark_labels) 
            if dark_stats[i, cv2.CC_STAT_AREA] >= min_dark_object_area
        ]
        
        bright_mask = (gray > bright_neighbor_threshold).astype(np.uint8) * 255
        preserved_dark_mask = np.zeros_like(gray)
        kernel = np.ones((5, 5), np.uint8)
        
        for i in large_dark_indices:
            component_mask = (dark_labels == i).astype(np.uint8) * 255
            
            dilated = cv2.dilate(component_mask, kernel, iterations=1)
            boundary = cv2.subtract(dilated, component_mask)
            
            boundary_pixels = boundary == 255
            if boundary_pixels.sum() > 0:
                bright_boundary_ratio = (bright_mask[boundary_pixels] == 255).sum() / boundary_pixels.sum()
                
                if bright_boundary_ratio >= bright_neighbor_ratio:
                    preserved_dark_mask = cv2.bitwise_or(preserved_dark_mask, component_mask)
        
        core_mask = cv2.bitwise_or(core_mask, preserved_dark_mask)

        # ---------------------------------------------------------
        # Step C: Soft Alpha Calculation
        # ---------------------------------------------------------
        alpha_gradient = gray.astype(float)
        alpha_gradient = (alpha_gradient / core_threshold) * 255.0
        alpha_gradient = np.clip(alpha_gradient, 0, 255).astype(np.uint8)

        final_alpha = np.zeros_like(gray)
        final_alpha[core_mask == 255] = 255
        
        shadow_region = cv2.bitwise_and(extent_mask, cv2.bitwise_not(core_mask))
        final_alpha[shadow_region == 255] = alpha_gradient[shadow_region == 255]

        # ---------------------------------------------------------
        # Step D: 마무리
        # ---------------------------------------------------------
        if blur_amount > 0:
            k = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
            final_alpha = cv2.GaussianBlur(final_alpha, (k, k), 0)
        
        # 미세 노이즈 제거 (morphological opening)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_alpha = cv2.morphologyEx(final_alpha, cv2.MORPH_OPEN, morph_kernel)

        b, g, r = cv2.split(img)
        b[final_alpha == 0] = 0
        g[final_alpha == 0] = 0
        r[final_alpha == 0] = 0
        
        rgba = cv2.merge([b, g, r, final_alpha])
        cv2.imwrite(output_rgba_path, rgba)
        return output_rgba_path

    except Exception as e:
        print(f"[Error] Processing failed: {e}")
        try:
            shutil.copy(inpainted_rgb_path, output_rgba_path)
        except:
            pass
        return output_rgba_path

# =============================================================================
# Failed Attempts Helper Functions
# =============================================================================

def get_failed_attempts(layer_id: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    [수정 22] Get list of failed attempts for a layer.
    
    Args:
        layer_id: The layer to check
        state: GraphState
    
    Returns:
        List of FailedAttempt dicts
    """
    tree = state.get("history_tree", {})
    if layer_id not in tree:
        return []
    
    node = tree[layer_id]
    return node.get("failed_attempts") or []


def has_exceeded_max_retries(layer_id: str, state: Dict[str, Any]) -> bool:
    """
    [수정 22] Check if a layer has exceeded maximum retry attempts.
    
    Args:
        layer_id: The layer to check
        state: GraphState
    
    Returns:
        True if max retries exceeded
    """
    max_retries = state.get("max_retry_per_layer", 3)
    failed_attempts = get_failed_attempts(layer_id, state)
    return len(failed_attempts) >= max_retries