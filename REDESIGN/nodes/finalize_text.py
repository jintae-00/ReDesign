# REDESIGN/nodes/finalize_text.py
"""
Finalize Text Node - Extract and save text element metadata

[수정] Added mask_canvas_uri for evaluator compatibility.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import uuid
import json
import shutil
import numpy as np
from PIL import Image

from ..state import GraphState
from ..reducers import (
    r_append_parsed_element,
    r_dequeue_node,
    r_set_layer_error,
    r_pack_state,
    r_update_layer_node,
    r_save_tool_output,
)
from ..utils import (
    find_latest_tool_output,
    get_current_image_path,
    get_tight_bbox_from_alpha,
    crop_to_tight_bbox,
    boxes_to_aabbs,
)


def _create_element_from_fontstyle(
    fontstyle_elem: Dict[str, Any],
    layer_id: str,
    episode_dir: str,
    image_path: str,
    canvas_size: Tuple[int, int] = None,  # [추가] (W, H)
) -> Optional[Dict[str, Any]]:
    """
    Create a parsed element from a single fontstyle element result.
    
    [수정] Added canvas_size parameter for mask_canvas_uri generation.
    """
    det_id = fontstyle_elem.get("det_id", "unknown")
    element_id = f"text_{uuid.uuid4().hex[:8]}"
    elements_dir = Path(episode_dir) / "elements" / element_id
    elements_dir.mkdir(parents=True, exist_ok=True)
    



    # Get bbox
    bbox = fontstyle_elem.get("bbox", [0, 0, 0, 0])
    mask_src = fontstyle_elem.get("mask_path") # Tool에서 저장한 raw 마스크 경로
    extracted_path = str(elements_dir / "extracted.png")
    
    # [수정] 마스크를 사용하여 원본 R,G,B,A 값을 손실 없이 추출
    from ..utils import apply_mask_to_image_and_crop
    if mask_src and Path(mask_src).exists():
        apply_mask_to_image_and_crop(image_path, mask_src, bbox, extracted_path)
    else:
        # Fallback (마스크 없을 때만)
        img = Image.open(image_path).convert("RGBA")
        img.crop(bbox).save(extracted_path)
    
    # Copy mask if available
    mask_path = None
    canvas_mask_path = None  # [추가]
    
    mask_src = fontstyle_elem.get("mask_path")
    if mask_src and Path(mask_src).exists():
        mask_path = str(elements_dir / "mask.png")
        shutil.copy(mask_src, mask_path)
        
        # ========== [추가] Canvas-size mask 생성 ==========
        if canvas_size:
            W, H = canvas_size
            canvas_mask_path = str(elements_dir / "mask_canvas.png")
            try:
                mask_img = Image.open(mask_src)
                # mask_src가 이미 canvas 크기인지 확인
                if mask_img.size == (W, H):
                    shutil.copy(mask_src, canvas_mask_path)
                else:
                    # bbox 위치에 paste
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    canvas_mask = Image.new("L", (W, H), 0)
                    # mask를 L 모드로 변환
                    if mask_img.mode == "RGBA":
                        mask_l = mask_img.split()[3]  # Alpha 채널
                    else:
                        mask_l = mask_img.convert("L")
                    # crop 크기로 resize
                    crop_w, crop_h = x2 - x1, y2 - y1
                    if crop_w > 0 and crop_h > 0:
                        mask_resized = mask_l.resize((crop_w, crop_h), Image.NEAREST)
                        canvas_mask.paste(mask_resized, (x1, y1))
                    canvas_mask.save(canvas_mask_path)
            except Exception as e:
                print(f"[Finalize Text] Canvas mask failed: {e}")
                canvas_mask_path = None
        # ==================================================
    
    # Copy rendered font image if available
    rendered_src = fontstyle_elem.get("rendered_image_path")
    rendered_path = None
    if rendered_src and Path(rendered_src).exists():
        rendered_path = str(elements_dir / "rendered.png")
        shutil.copy(rendered_src, rendered_path)
    
    # Copy font file if available
    font_file_src = fontstyle_elem.get("font_file_path")
    font_file_path = None
    if font_file_src and Path(font_file_src).exists():
        ext = Path(font_file_src).suffix
        font_file_path = str(elements_dir / f"font{ext}")
        shutil.copy(font_file_src, font_file_path)
    
    # Build element
    element = {
        "id": element_id,
        "type": "text",
        "det_id": det_id,
        "content": fontstyle_elem.get("text_content", ""),
        "bbox": bbox,
        "extracted_image_uri": extracted_path,
        "segmentation_mask_path": mask_path,
        "mask_canvas_uri": canvas_mask_path,  # [추가]
        "font_family": fontstyle_elem.get("font_family", "Unknown"),
        "font_size_px": fontstyle_elem.get("size_px", 16),
        "font_color": fontstyle_elem.get("color", {"rgb": [0, 0, 0], "hex": "#000000"}),
        "font_bold": fontstyle_elem.get("bold", False),
        "font_italic": fontstyle_elem.get("italic", False),
        "angle_deg": fontstyle_elem.get("angle_deg", 0),
        "l1_loss": fontstyle_elem.get("l1_loss"),
        "rendered_image_path": rendered_path,
        "font_file_path": font_file_path,
        "ocr_score": fontstyle_elem.get("score"),
    }
    
    # Save metadata JSON
    metadata_path = elements_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(element, f, ensure_ascii=False, indent=2)
    
    return element


def node(state: GraphState) -> Dict[str, Any]:
    """
    Finalize Text node - saves EACH text element to parsed_elements individually.
    
    [수정 7] CRITICAL FIX:
    - Now uses r_pack_state for proper handling of _append_parsed_element
    - Collects all element updates and packs them together
    - This fixes the bug where text elements were not saved to parse.json
    
    [수정] Added canvas_size for mask_canvas_uri generation.
    
    Reads from:
        - Fontstyle output (elements list with per-box results)
        - Ancestor OCR output (fallback for content, bbox)
        - Ancestor HiSAM output (mask)
    
    Updates:
        - history_tree[layer_id].parsed_elements (multiple elements)
        - state.parsed_elements
    """
    layer_id = state.get("current_layer_id")
    if not layer_id:
        return {"error": "No current layer ID"}
    
    tree = state.get("history_tree", {})
    if layer_id not in tree:
        return {"error": f"Layer {layer_id} not in history_tree"}
    
    # Dequeue this node
    _, dequeue_update = r_dequeue_node(layer_id, state)
    
    node_data = tree[layer_id]
    tool_outputs = node_data.get("tool_outputs", {})
    
    # Get current image
    image_path = get_current_image_path(layer_id, state)
    episode_dir = state.get("episode_dir", ".")
    
    # ========== [추가] Canvas size 가져오기 ==========
    canvas_size = None
    root_image = state.get("root_image_path")
    if root_image and Path(root_image).exists():
        try:
            with Image.open(root_image) as img:
                canvas_size = img.size  # (W, H)
        except Exception as e:
            print(f"[Finalize Text] Failed to get canvas size: {e}")
    # ================================================
    
    # Get tools_output directory for saving finalize results
    layer_tools_dir = Path(episode_dir) / "layers" / layer_id / "tools_output"
    layer_tools_dir.mkdir(parents=True, exist_ok=True)
    
    # Get fontstyle output
    fontstyle_output = tool_outputs.get("fontstyle")
    
    created_elements = []
    
    if fontstyle_output and fontstyle_output.get("elements"):
        # Process each fontstyle element individually
        fontstyle_elements = fontstyle_output.get("elements", [])
        
        for fs_elem in fontstyle_elements:
            element = _create_element_from_fontstyle(
                fontstyle_elem=fs_elem,
                layer_id=layer_id,
                episode_dir=episode_dir,
                image_path=image_path,
                canvas_size=canvas_size,  # [추가]
            )
            
            if element:
                created_elements.append(element)
                print(f"[Finalize Text] Created element {element['id']}: '{element['content'][:30]}...'")
    
    else:
        # Fallback: Use OCR output directly if fontstyle wasn't run or has no elements
        ocr_output = find_latest_tool_output(layer_id, "detect", "detect_ocr", state)
        seg_output = find_latest_tool_output(layer_id, "segment", None, state)
        
        masks_by_id = {}
        if seg_output:
            masks_by_id = seg_output.get("masks_by_id", {})
        
        if ocr_output:
            boxes = ocr_output.get("boxes", [])
            texts = ocr_output.get("texts", [])
            scores = ocr_output.get("scores", [])
            det_ids = ocr_output.get("det_ids", [])
            
            aabbs = boxes_to_aabbs(boxes)
            
            for idx, (bbox, text, det_id) in enumerate(zip(aabbs, texts, det_ids)):
                element_id = f"text_{uuid.uuid4().hex[:8]}"
                elements_dir = Path(episode_dir) / "elements" / element_id
                elements_dir.mkdir(parents=True, exist_ok=True)
                
                # Crop text region
                extracted_path = str(elements_dir / "extracted.png")
                try:
                    if image_path and Path(image_path).exists():
                        img = Image.open(image_path).convert("RGBA")
                        x1, y1, x2, y2 = bbox
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img.width, x2), min(img.height, y2)
                        if x2 > x1 and y2 > y1:
                            cropped = img.crop((x1, y1, x2, y2))
                            cropped.save(extracted_path)
                except Exception as e:
                    print(f"[Finalize Text] Crop failed for {det_id}: {e}")
                    continue
                
                # Copy mask if available + canvas mask
                mask_path = None
                canvas_mask_path = None  # [추가]
                
                if det_id in masks_by_id:
                    mask_src = masks_by_id[det_id]
                    if mask_src and Path(mask_src).exists():
                        mask_path = str(elements_dir / "mask.png")
                        shutil.copy(mask_src, mask_path)
                        
                        # ========== [추가] Canvas-size mask 생성 ==========
                        if canvas_size:
                            W, H = canvas_size
                            canvas_mask_path = str(elements_dir / "mask_canvas.png")
                            try:
                                mask_img = Image.open(mask_src)
                                if mask_img.size == (W, H):
                                    shutil.copy(mask_src, canvas_mask_path)
                                else:
                                    x1, y1, x2, y2 = [int(v) for v in bbox]
                                    canvas_mask = Image.new("L", (W, H), 0)
                                    if mask_img.mode == "RGBA":
                                        mask_l = mask_img.split()[3]
                                    else:
                                        mask_l = mask_img.convert("L")
                                    crop_w, crop_h = x2 - x1, y2 - y1
                                    if crop_w > 0 and crop_h > 0:
                                        mask_resized = mask_l.resize((crop_w, crop_h), Image.NEAREST)
                                        canvas_mask.paste(mask_resized, (x1, y1))
                                    canvas_mask.save(canvas_mask_path)
                            except Exception as e:
                                print(f"[Finalize Text] Canvas mask failed: {e}")
                                canvas_mask_path = None
                        # ==================================================
                
                element = {
                    "id": element_id,
                    "type": "text",
                    "det_id": det_id,
                    "content": text,
                    "bbox": bbox,
                    "extracted_image_uri": extracted_path,
                    "segmentation_mask_path": mask_path,
                    "mask_canvas_uri": canvas_mask_path,  # [추가]
                    "font_family": "Unknown",
                    "font_size_px": bbox[3] - bbox[1] if len(bbox) == 4 else 16,
                    "font_color": {"rgb": [0, 0, 0], "hex": "#000000"},
                    "font_bold": False,
                    "font_italic": False,
                    "ocr_score": scores[idx] if idx < len(scores) else None,
                }
                
                # Save metadata
                metadata_path = elements_dir / "metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(element, f, ensure_ascii=False, indent=2)
                
                created_elements.append(element)
                print(f"[Finalize Text] Created element {element_id}: '{text[:30]}...'")
        
        else:
            # Ultimate fallback: single element from image
            element_id = f"text_{uuid.uuid4().hex[:8]}"
            elements_dir = Path(episode_dir) / "elements" / element_id
            elements_dir.mkdir(parents=True, exist_ok=True)
            
            bbox = get_tight_bbox_from_alpha(image_path) if image_path else [0, 0, 0, 0]
            
            extracted_path = str(elements_dir / "extracted.png")
            if image_path and Path(image_path).exists():
                crop_to_tight_bbox(image_path, extracted_path)
            
            element = {
                "id": element_id,
                "type": "text",
                "det_id": "unknown",
                "content": "",
                "bbox": bbox or [0, 0, 0, 0],
                "extracted_image_uri": extracted_path,
                "segmentation_mask_path": None,
                "mask_canvas_uri": None,  # [추가]
                "font_family": "Unknown",
                "font_size_px": 16,
                "font_color": {"rgb": [0, 0, 0], "hex": "#000000"},
                "font_bold": False,
                "font_italic": False,
            }
            
            metadata_path = elements_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(element, f, ensure_ascii=False, indent=2)
            
            created_elements.append(element)
    
    # Save finalize_text summary to tools_output
    finalize_summary = {
        "tool_name": "finalize_text",
        "num_elements": len(created_elements),
        "element_ids": [e["id"] for e in created_elements],
        "elements_summary": [
            {
                "id": e["id"],
                "det_id": e.get("det_id"),
                "content": e.get("content", "")[:50],
                "bbox": e.get("bbox"),
                "font_family": e.get("font_family"),
            }
            for e in created_elements
        ],
    }
    
    finalize_json_path = layer_tools_dir / "finalize_text_output.json"
    with open(finalize_json_path, "w", encoding="utf-8") as f:
        json.dump(finalize_summary, f, ensure_ascii=False, indent=2)
    
    print(f"[Finalize Text] Created {len(created_elements)} text elements for {layer_id}")
    
    # [수정 7] CRITICAL FIX: Use r_pack_state to properly handle all updates
    # Collect all element updates
    element_updates = []
    for element in created_elements:
        element_update = r_append_parsed_element(element, layer_id, state)
        element_updates.append(element_update)
    
    # Save tool output
    tool_update = r_save_tool_output(
        layer_id=layer_id,
        tool_category="finalize_text",
        tool_name="finalize_text",
        output=finalize_summary,
        state=state,
    )
    
    # Pack all updates together - this properly handles _append_parsed_element
    return r_pack_state(state, dequeue_update, tool_update, *element_updates)