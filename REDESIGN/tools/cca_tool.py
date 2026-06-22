# REDESIGN/tools/cca_tool.py
"""
Connected Component Analysis (CCA) Tool

Fast pixel-based splitting of spatially separated objects using alpha channel.
No deep learning required - uses scipy/opencv for connected components.

[수정 8] CRITICAL FIX:
- Added _clean_transparent_pixels to zero out RGB where alpha is low
- This prevents "ghost pixels" from corrupting downstream Qwen processing
- Transparent areas now have RGB=(0,0,0) instead of random color values
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage


def _clean_transparent_pixels(arr: np.ndarray, alpha_threshold: int = 10) -> np.ndarray:
    """
    [수정 8] Zero out RGB values for pixels with low alpha.
    
    This is CRITICAL for downstream processing:
    - CCA outputs may have random RGB values in transparent areas
    - When these images are fed to Qwen, the random colors cause noise/artifacts
    - By zeroing RGB in transparent areas, we ensure clean input to Qwen
    
    Args:
        arr: RGBA numpy array (H, W, 4)
        alpha_threshold: Pixels with alpha below this have RGB zeroed (0-255)
    
    Returns:
        RGBA numpy array with cleaned RGB values
    """
    result = arr.copy()
    
    # Create mask for transparent pixels (alpha < threshold)
    transparent_mask = result[:, :, 3] < alpha_threshold
    
    # Zero out RGB values for transparent pixels
    result[transparent_mask, 0] = 0  # R
    result[transparent_mask, 1] = 0  # G
    result[transparent_mask, 2] = 0  # B
    # Alpha stays as-is (could also set to 0 for fully transparent)
    
    return result


def run_split_cca(
    image_path: str,
    min_area: int = 100,
    alpha_threshold: int = 10,
    connectivity: int = 8,
) -> Dict[str, Any]:
    """
    Split image into connected components based on alpha channel.
    
    [수정 8] Now applies _clean_transparent_pixels to all output layers.
    
    Args:
        image_path: Path to input RGBA image (canvas size)
        min_area: Minimum pixel area for a component to be kept
        alpha_threshold: Alpha value threshold for foreground (0-255)
        connectivity: 4 or 8 connectivity for CCA
    
    Returns:
        {
            "components": [
                {
                    "component_id": "comp_00",
                    "mask_path": str,
                    "layer_path": str,
                    "bbox": [x1, y1, x2, y2],
                    "area": int,
                },
                ...
            ],
            "num_components": int,
            "total_foreground_area": int,
        }
    """
    # Load image
    img = Image.open(image_path).convert("RGBA")
    img_arr = np.array(img)
    alpha = img_arr[:, :, 3]
    H, W = alpha.shape
    
    # Create binary mask from alpha
    binary = (alpha > alpha_threshold).astype(np.uint8)
    
    # Find connected components
    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:  # 8-connectivity
        structure = np.ones((3, 3), dtype=int)
    
    labeled, num_features = ndimage.label(binary, structure=structure)
    
    # Output directory
    output_dir = Path(image_path).parent / "cca_components"
    output_dir.mkdir(exist_ok=True)
    
    components = []
    total_fg_area = 0
    
    for comp_idx in range(1, num_features + 1):
        # Create mask for this component
        comp_mask = (labeled == comp_idx).astype(np.uint8)
        area = int(comp_mask.sum())
        
        # Skip small components
        if area < min_area:
            continue
        
        total_fg_area += area
        
        # Get bounding box
        rows = np.any(comp_mask, axis=1)
        cols = np.any(comp_mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        bbox = [int(x1), int(y1), int(x2) + 1, int(y2) + 1]
        
        comp_id = f"comp_{len(components):02d}"
        
        # Save mask
        mask_path = output_dir / f"{comp_id}_mask.png"
        mask_img = Image.fromarray(comp_mask * 255)
        mask_img.save(mask_path)
        
        # Create layer image (canvas size, with only this component visible)
        layer_arr = img_arr.copy()
        
        # [수정 8] Set alpha to 0 for non-component pixels
        layer_arr[:, :, 3] = np.where(comp_mask > 0, layer_arr[:, :, 3], 0)
        
        # [수정 8] CRITICAL: Clean transparent pixels to zero out RGB
        # This prevents ghost pixels from corrupting Qwen processing
        layer_arr = _clean_transparent_pixels(layer_arr, alpha_threshold=alpha_threshold)
        
        layer_img = Image.fromarray(layer_arr)
        
        layer_path = output_dir / f"{comp_id}_layer.png"
        layer_img.save(layer_path)
        
        components.append({
            "component_id": comp_id,
            "mask_path": str(mask_path),
            "layer_path": str(layer_path),
            "bbox": bbox,
            "area": area,
        })
        
        print(f"[CCA] Component {comp_id}: area={area}, bbox={bbox}, cleaned transparent pixels")
    
    # Sort by area (largest first, which typically means background/base elements)
    components.sort(key=lambda x: x["area"], reverse=True)
    
    return {
        "components": components,
        "num_components": len(components),
        "total_foreground_area": total_fg_area,
    }


def analyze_separability(image_path: str, alpha_threshold: int = 10) -> Dict[str, Any]:
    """
    Analyze whether an image is suitable for CCA splitting.
    
    Returns:
        {
            "is_separable": bool,
            "num_components": int,
            "component_sizes": List[int],
            "recommendation": str,
        }
    """
    img = Image.open(image_path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    
    binary = (alpha > alpha_threshold).astype(np.uint8)
    labeled, num_features = ndimage.label(binary)
    
    # Get component sizes
    sizes = []
    for i in range(1, num_features + 1):
        sizes.append(int((labeled == i).sum()))
    
    sizes.sort(reverse=True)
    
    # Determine if separable
    is_separable = num_features > 1 and len([s for s in sizes if s > 100]) > 1
    
    # Recommendation
    if num_features == 0:
        recommendation = "Empty layer - consider Discard"
    elif num_features == 1:
        recommendation = "Single component - consider Finalize or further detection"
    elif num_features <= 3:
        recommendation = "Few components - Split_CCA recommended"
    else:
        recommendation = "Multiple components - Split_CCA or Fork_Qwen"
    
    return {
        "is_separable": is_separable,
        "num_components": num_features,
        "component_sizes": sizes[:10],  # Top 10
        "recommendation": recommendation,
    }


def merge_small_components(
    image_path: str,
    components: List[Dict[str, Any]],
    merge_threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Merge small components (likely noise) into the largest component.
    
    Args:
        image_path: Original image path
        components: List of component dicts from run_split_cca
        merge_threshold: Components smaller than this fraction of total are merged
    
    Returns:
        Filtered list of components
    """
    if not components:
        return components
    
    total_area = sum(c["area"] for c in components)
    threshold_area = total_area * merge_threshold
    
    # Separate large and small components
    large = [c for c in components if c["area"] >= threshold_area]
    small = [c for c in components if c["area"] < threshold_area]
    
    if not large or not small:
        return components
    
    # Merge small into largest
    # In practice, we just filter them out (they're noise)
    return large