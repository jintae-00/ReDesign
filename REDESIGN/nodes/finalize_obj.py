"""
Finalize Object Node - Extract and save object/background element

Used as the final step for atomic objects or backgrounds.
Optionally includes SVG vectorization for non-photographic elements.

UPDATED: Now saves all outputs to tools_output directory.
Added mask_canvas_uri for evaluator compatibility.
Fixed: child layer_image is already correctly extracted by stack_manager.
       No need to re-apply parent's mask. Use alpha channel directly.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import uuid
import shutil
import json
from PIL import Image

from ..state import GraphState
from ..reducers import (
    r_append_parsed_element,
    r_dequeue_node,
    r_set_layer_error,
    r_pack_state,
)
from ..utils import (
    get_current_image_path,
    get_tight_bbox_from_alpha,
    crop_to_tight_bbox,
    convert_black_to_transparent,
)


def node(state: GraphState) -> Dict[str, Any]:
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
    if not image_path or not Path(image_path).exists():
        return r_pack_state(
            state,
            dequeue_update,
            r_set_layer_error(layer_id, "Image not found", {"path": image_path}, state)
        )

    # Prepare directories
    episode_dir = state.get("episode_dir", ".")
    tools_output_dir = Path(episode_dir) / "layers" / layer_id / "tools_output"
    tools_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build element
    element_id = f"obj_{uuid.uuid4().hex[:8]}"
    elements_dir = Path(episode_dir) / "elements" / element_id
    elements_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # CRITICAL FIX:
    # The child node's layer_image.png has already been correctly extracted by
    # stack_manager. Re-applying the parent's masks_by_id could apply the wrong mask.
    # -> Use the layer_image's alpha channel directly!
    # ==========================================================================

    # 1. Compute the bbox (based on the layer_image's alpha channel)
    bbox = get_tight_bbox_from_alpha(image_path)
    if not bbox:
        # Set the bbox to the entire image
        try:
            with Image.open(image_path) as img:
                bbox = [0, 0, img.width, img.height]
        except:
            bbox = [0, 0, 100, 100]

    # 2. Canvas image: the layer_image as is (only transparent-background processing)
    canvas_path = str(elements_dir / "canvas_image.png")
    convert_black_to_transparent(image_path, canvas_path)

    # 3. Crop image: crop to the tight bbox (without re-applying the mask)
    extracted_path = str(elements_dir / "crop_image.png")
    crop_to_tight_bbox(image_path, extracted_path)

    # 4. Canvas-size mask: extract the layer_image's alpha channel
    canvas_mask_path = str(elements_dir / "mask_canvas.png")
    try:
        img = Image.open(image_path).convert("RGBA")
        alpha = img.split()[3]
        alpha.save(canvas_mask_path)
    except Exception as e:
        print(f"[Finalize Obj] Failed to save canvas mask: {e}")
        canvas_mask_path = None

    # Copy artifacts (tools_output)
    tools_extracted_path = tools_output_dir / "finalize_obj_crop.png"
    tools_canvas_path = tools_output_dir / "finalize_obj_canvas.png"
    shutil.copy(extracted_path, tools_extracted_path)
    shutil.copy(canvas_path, tools_canvas_path)
    
    # Label and SVG handling
    image_context = node_data.get("image_context", "")
    label = image_context[:100] if image_context else "object"
    is_photo = node_data.get("param_is_photo", False)
    
    svg_path = None
    vtracer_output = tool_outputs.get("vtracer")
    if vtracer_output:
        svg_uri = vtracer_output.get("svg_uri")
        if svg_uri and Path(svg_uri).exists():
            svg_dest = elements_dir / "vector.svg"
            shutil.copy(svg_uri, svg_dest)
            svg_path = str(svg_dest)
    
    elem_type = "background" if "bg" in label.lower() or "background" in label.lower() else "object"
    
    # Build the element data
    element = {
        "id": element_id,
        "type": elem_type,
        "label": label,
        "bbox": bbox,
        "extracted_image_uri": extracted_path,
        "canvas_image_uri": canvas_path,
        "mask_canvas_uri": canvas_mask_path,
        "is_photographic": is_photo,
    }
    if svg_path:
        element["svg_uri"] = svg_path
    
    # Save metadata
    with open(elements_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(element, f, ensure_ascii=False, indent=2)

    # Save summary information
    finalize_output = {
        "tool_name": "finalize_obj",
        "element_id": element_id,
        "element_type": elem_type,
        "label": label,
        "bbox": bbox,
        "is_photographic": is_photo,
        "cropped_image_path": str(tools_extracted_path),
        "canvas_image_path": str(tools_canvas_path),
        "mask_canvas_path": canvas_mask_path,
    }
    with open(tools_output_dir / "finalize_obj_output.json", "w", encoding="utf-8") as f:
        json.dump(finalize_output, f, ensure_ascii=False, indent=2)
    
    print(f"[Finalize Obj] Created element {element_id}: '{label[:30]}...'")
    
    # Add to parsed elements
    element_update = r_append_parsed_element(element, layer_id, state)
    return r_pack_state(state, dequeue_update, element_update)