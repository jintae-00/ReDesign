# REDESIGN/reconstruction.py
"""
Reconstruction Module - Rebuild image from parsed elements using z-order
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import argparse
from datetime import datetime
from PIL import Image, ImageFilter  # Ensure ImageFilter is imported
import numpy as np
import cv2  # Ensure cv2 is imported

# =============================================================================
# Configuration
# =============================================================================

# [NEW] Alpha threshold (0-255) for removing background noise.
# Pixels with an alpha below this value are treated as 0 (fully transparent) during compositing.
ALPHA_THRESHOLD = 16 


# =============================================================================
# Path Resolution
# =============================================================================

def get_src_root() -> Path:
    """Get the src directory root (parent of REDESIGN)."""
    return Path(__file__).resolve().parent.parent


def resolve_element_path(relative_path: str, src_root: Path) -> Path:
    """Resolve element path from parse.json format to absolute path."""
    return src_root / relative_path


# =============================================================================
# Z-Order Computation
# =============================================================================

def compute_z_order(history_tree: Dict[str, Any], root_id: str = "layer_0000") -> List[str]:
    """Compute global z-order by DFS traversal of history_tree."""
    z_order = []
    visited = set()
    
    def dfs(layer_id: str):
        if layer_id in visited:
            return
        visited.add(layer_id)

        node = history_tree.get(layer_id)
        if not node:
            print(f"[Z-Order] Warning: {layer_id} not found in history_tree")
            return

        # Skip nodes that failed sparse verification
        if node.get("verification_status") == "sparse_failed":
            return

        action_type = node.get("action_type")
        children = node.get("children_ids") or []
        
        real_children = [
            c for c in children 
            if not c.startswith("_temp_")
        ]
        
        if action_type in ["Finalize_Text", "Finalize_Obj"]:
            z_order.append(layer_id)
            return
        
        if action_type == "Discard":
            return
        
        if not real_children:
            if action_type in ["Finalize_Text", "Finalize_Obj"]:
                z_order.append(layer_id)
            return
        
        for child_id in real_children:
            dfs(child_id)
    
    dfs(root_id)
    return z_order

def compute_z_order_with_info(
    history_tree: Dict[str, Any], 
    root_id: str = "layer_0000"
) -> List[Dict[str, Any]]:
    """Compute z-order with additional metadata for debugging."""
    z_order_ids = compute_z_order(history_tree, root_id)
    
    result = []
    for z_idx, layer_id in enumerate(z_order_ids):
        node = history_tree.get(layer_id, {})
        parent_id = node.get("parent_id")
        parent_node = history_tree.get(parent_id, {}) if parent_id else {}
        
        image_context = node.get("image_context") or ""
        
        result.append({
            "layer_id": layer_id,
            "z_index": z_idx,
            "action_type": node.get("action_type"),
            "depth": node.get("depth", 0),
            "parent_id": parent_id,
            "parent_action": parent_node.get("action_type"),
            "image_context": image_context[:50] if image_context else "",
        })
    
    return result

# =============================================================================
# Image Reconstruction
# =============================================================================

def draw_element_borders(
    image: Image.Image,
    parsed_elements: List[Dict[str, Any]],
    src_root: Path,
    border_color: Tuple[int, int, int, int] = (255, 150, 200, 200),
    glow_color: Tuple[int, int, int, int] = (255, 180, 220, 100),
    border_width: int = 3,
    glow_width: int = 5,
    verbose: bool = True,
) -> Image.Image:
    """Draw fancy pink glow borders around segmented elements' actual contours."""
    # (Same as before, except imports are now handled at the top of the module rather than inside the function)
    result = image.copy().convert("RGBA")
    canvas_w, canvas_h = result.size
    
    glow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    border_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    processed_count = 0
    
    for elem in parsed_elements:
        elem_type = elem.get("type")
        elem_id = elem.get("id", "unknown")
        bbox = elem.get("bbox", [0, 0, 0, 0])
        
        if elem_type in ["object", "background"]:
            img_path_rel = elem.get("canvas_image_uri") or elem.get("extracted_image_uri")
            use_canvas = elem.get("canvas_image_uri") is not None
        else:
            img_path_rel = elem.get("extracted_image_uri")
            use_canvas = False
        
        if not img_path_rel:
            continue
        
        img_path = resolve_element_path(img_path_rel, src_root)
        if not img_path.exists():
            if verbose:
                print(f"  [Borders] Image not found: {img_path}")
            continue
        
        try:
            elem_img = Image.open(img_path).convert("RGBA")
            
            if use_canvas and elem_img.size != (canvas_w, canvas_h):
                elem_img = elem_img.resize((canvas_w, canvas_h), Image.LANCZOS)
            
            elem_arr = np.array(elem_img)
            alpha = elem_arr[:, :, 3]
            binary_mask = (alpha > 128).astype(np.uint8) * 255
            
            contours, _ = cv2.findContours(
                binary_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
            
            if use_canvas:
                offset_x, offset_y = 0, 0
            else:
                offset_x, offset_y = int(bbox[0]), int(bbox[1])
            
            glow_arr = np.array(glow_layer)
            for contour in contours:
                contour_offset = contour.copy()
                contour_offset[:, :, 0] += offset_x
                contour_offset[:, :, 1] += offset_y
                
                for i in range(glow_width, 0, -1):
                    alpha_val = int(glow_color[3] * (1 - i / (glow_width + 2)))
                    cv2.drawContours(
                        glow_arr, 
                        [contour_offset], 
                        -1, 
                        (*glow_color[:3], alpha_val),
                        thickness=i * 2
                    )
            
            glow_layer = Image.fromarray(glow_arr)
            
            border_arr = np.array(border_layer)
            for contour in contours:
                contour_offset = contour.copy()
                contour_offset[:, :, 0] += offset_x
                contour_offset[:, :, 1] += offset_y
                
                cv2.drawContours(
                    border_arr, 
                    [contour_offset], 
                    -1, 
                    border_color,
                    thickness=border_width
                )
            
            border_layer = Image.fromarray(border_arr)
            processed_count += 1
            
        except Exception as e:
            if verbose:
                print(f"  [Borders] Error processing {elem_id}: {e}")
    
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=4))
    
    result = Image.alpha_composite(result, glow_layer)
    result = Image.alpha_composite(result, border_layer)
    
    if verbose:
        print(f"[Borders] Processed {processed_count} elements")
    
    return result

def reconstruct_image_with_borders(
    history_tree: Dict[str, Any],
    parsed_elements: List[Dict[str, Any]],
    root_image_path: str,
    src_root: Path,
    verbose: bool = True
) -> Tuple[Image.Image, Image.Image]:
    """Reconstruct image and create a version with pink element borders."""
    reconstructed = reconstruct_image(
        history_tree=history_tree,
        parsed_elements=parsed_elements,
        root_image_path=root_image_path,
        src_root=src_root,
        verbose=verbose
    )
    
    if verbose:
        print(f"\n[Reconstruction] Adding element contour borders...")
    
    bordered = draw_element_borders(
        reconstructed, 
        parsed_elements, 
        src_root,  # NEW: pass src_root
        verbose=verbose
    )
    
    return reconstructed, bordered

def add_borders_to_existing_episode(
    episode_dir: str,
    output_name: str = "reconstructed_bordered.png",
    verbose: bool = True
) -> str:
    """Add pink element contour borders to an existing episode's reconstructed image."""
    episode_path = Path(episode_dir)
    if not episode_path.is_absolute():
        episode_path = Path.cwd() / episode_dir
    
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_path}")
    
    src_root = get_src_root()
    if "agent_output" in str(episode_path):
        parts = episode_path.parts
        for i, part in enumerate(parts):
            if part == "agent_output":
                src_root = Path(*parts[:i])
                break
    
    if verbose:
        print(f"[Borders] Episode: {episode_path}")
        print(f"[Borders] Src root: {src_root}")
    
    parse_path = episode_path / "parse.json"
    if not parse_path.exists():
        raise FileNotFoundError(f"parse.json not found: {parse_path}")
    
    with open(parse_path, "r", encoding="utf-8") as f:
        parse_data = json.load(f)
    
    parsed_elements = parse_data.get("elements", [])
    
    reconstructed_path = episode_path / "reconstructed.png"
    if not reconstructed_path.exists():
        if verbose:
            print(f"[Borders] reconstructed.png not found, creating...")
        reconstruct_episode(str(episode_path), verbose=verbose)
    
    reconstructed = Image.open(reconstructed_path).convert("RGBA")
    
    if verbose:
        print(f"[Borders] Adding contour borders...")
    
    bordered = draw_element_borders(
        reconstructed, 
        parsed_elements,
        src_root,
        verbose=verbose
    )
    
    output_path = episode_path / output_name
    bordered.save(output_path)
    
    if verbose:
        print(f"[Borders] Saved to: {output_path}")
    
    return str(output_path)

def reconstruct_image(
    history_tree: Dict[str, Any],
    parsed_elements: List[Dict[str, Any]],
    root_image_path: str,
    src_root: Path,
    verbose: bool = True
) -> Image.Image:
    """
    Reconstruct image by compositing elements in z-order.
    Uses ALPHA_THRESHOLD to clean background noise before compositing.
    """
    # 1. Compute z-order
    z_order = compute_z_order(history_tree)
    
    if verbose:
        print(f"\n[Reconstruction] Z-order computed: {len(z_order)} layers")
    
    # 2. Build layer_id → elements mapping
    layer_to_elements: Dict[str, List[Dict]] = {}
    for elem in parsed_elements:
        source_layer = elem.get("source_layer_id")
        if source_layer:
            if source_layer not in layer_to_elements:
                layer_to_elements[source_layer] = []
            layer_to_elements[source_layer].append(elem)
    
    # 3. Create empty canvas
    root_path = resolve_element_path(root_image_path, src_root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root image not found: {root_path}")
    
    root_img = Image.open(root_path)
    canvas_w, canvas_h = root_img.size
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    if verbose:
        print(f"\n[Reconstruction] Canvas size: {canvas_w} x {canvas_h}")
    
    # 4. Composite in z-order (back → front)
    composited_count = 0
    
    for z_idx, layer_id in enumerate(z_order):
        elements = layer_to_elements.get(layer_id, [])
        if not elements: continue
        
        for elem in elements:
            elem_type = elem.get("type")
            elem_id = elem.get("id", "unknown")
            bbox = elem.get("bbox", [0, 0, 0, 0])
            
            try:
                if elem_type in ["object", "background"]:
                    img_path_rel = elem.get("canvas_image_uri")
                    use_bbox = False
                    if not img_path_rel:
                        img_path_rel = elem.get("extracted_image_uri")
                        use_bbox = True
                    
                    if img_path_rel:
                        img_path = resolve_element_path(img_path_rel, src_root)
                        if img_path.exists():
                            layer_img = Image.open(img_path).convert("RGBA")
                            
                            # [NEW] Alpha Thresholding Logic
                            # Zero out low-alpha (noise) pixels to remove compositing noise
                            arr = np.array(layer_img)
                            mask = arr[:, :, 3] < ALPHA_THRESHOLD
                            arr[mask] = [0, 0, 0, 0]
                            layer_img = Image.fromarray(arr)
                            
                            if use_bbox:
                                x1, y1 = int(bbox[0]), int(bbox[1])
                                temp = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                                temp.paste(layer_img, (x1, y1))
                                canvas = Image.alpha_composite(canvas, temp)
                            else:
                                if layer_img.size != (canvas_w, canvas_h):
                                    layer_img = layer_img.resize((canvas_w, canvas_h), Image.LANCZOS)
                                canvas = Image.alpha_composite(canvas, layer_img)
                            
                            composited_count += 1
                            if verbose:
                                print(f"  z={z_idx} {layer_id}: Composited {elem_type} (Cleaned)")
                
                elif elem_type == "text":
                    img_path_rel = elem.get("extracted_image_uri")
                    if img_path_rel:
                        img_path = resolve_element_path(img_path_rel, src_root)
                        if img_path.exists():
                            layer_img = Image.open(img_path).convert("RGBA")
                            
                            # [NEW] Apply the same cleaning to text as well
                            arr = np.array(layer_img)
                            mask = arr[:, :, 3] < ALPHA_THRESHOLD
                            arr[mask] = [0, 0, 0, 0]
                            layer_img = Image.fromarray(arr)
                            
                            x1, y1 = int(bbox[0]), int(bbox[1])
                            temp = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                            temp.paste(layer_img, (x1, y1))
                            canvas = Image.alpha_composite(canvas, temp)
                            
                            composited_count += 1
                            if verbose:
                                print(f"  z={z_idx} {layer_id}: Composited text (Cleaned)")

            except Exception as e:
                print(f"  z={z_idx} {layer_id}: Error {e}")
    
    if verbose:
        print(f"\n[Reconstruction] Composited {composited_count} elements")
    
    return canvas

def reconstruct_episode(
    episode_dir: str,
    output_name: str = "reconstructed.png",
    save_bordered: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Reconstruct image from an episode directory.
    """
    episode_path = Path(episode_dir)
    if not episode_path.is_absolute():
        episode_path = Path.cwd() / episode_dir
    
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_path}")
    
    # Determine src_root
    src_root = get_src_root()
    if "agent_output" in str(episode_path):
        parts = episode_path.parts
        for i, part in enumerate(parts):
            if part == "agent_output":
                src_root = Path(*parts[:i])
                break
    
    print(f"[Reconstruction] Episode: {episode_path}")
    print(f"[Reconstruction] Src root: {src_root}")
    
    # Load history_tree.json
    history_tree_path = episode_path / "history_tree.json"
    if not history_tree_path.exists():
        raise FileNotFoundError(f"history_tree.json not found: {history_tree_path}")
    
    with open(history_tree_path, "r", encoding="utf-8") as f:
        history_tree = json.load(f)
    
    print(f"[Reconstruction] Loaded history_tree: {len(history_tree)} nodes")
    
    # Load parse.json
    parse_path = episode_path / "parse.json"
    if not parse_path.exists():
        raise FileNotFoundError(f"parse.json not found: {parse_path}")
    
    with open(parse_path, "r", encoding="utf-8") as f:
        parse_data = json.load(f)
    
    parsed_elements = parse_data.get("elements", [])
    root_image_path = parse_data.get("root_image", "")
    
    print(f"[Reconstruction] Loaded parse.json: {len(parsed_elements)} elements")
    print(f"[Reconstruction] Root image: {root_image_path}")
    
    # Reconstruct
    result_img = reconstruct_image(
        history_tree=history_tree,
        parsed_elements=parsed_elements,
        root_image_path=root_image_path,
        src_root=src_root,
        verbose=verbose
    )
    
    # Save result
    output_path = episode_path / output_name
    result_img.save(output_path)
    print(f"\n[Reconstruction] Saved to: {output_path}")
    
    # Save bordered version with actual contours
    bordered_path = None
    if save_bordered:
        print(f"\n[Reconstruction] Creating bordered version with contours...")
        bordered_img = draw_element_borders(
            result_img, 
            parsed_elements,
            src_root,  # NEW
            verbose=verbose
        )
        bordered_path = episode_path / "reconstructed_bordered.png"
        bordered_img.save(bordered_path)
        print(f"[Reconstruction] Bordered image saved to: {bordered_path}")
    
    # Also save z-order info for debugging
    z_order_info = compute_z_order_with_info(history_tree)
    z_order_path = episode_path / "z_order.json"
    with open(z_order_path, "w", encoding="utf-8") as f:
        json.dump(z_order_info, f, ensure_ascii=False, indent=2)
    print(f"[Reconstruction] Z-order info saved to: {z_order_path}")
    
    return {
        "output_path": str(output_path),
        "bordered_path": str(bordered_path) if bordered_path else None,
        "z_order_path": str(z_order_path),
        "num_elements": len(parsed_elements),
        "num_layers_in_zorder": len(z_order_info),
        "canvas_size": result_img.size,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reconstruct image from URLD episode results")
    parser.add_argument("--episode", "-e", type=str, required=True, help="Path to episode directory")
    parser.add_argument("--output", "-o", type=str, default="reconstructed.png", help="Output filename")
    parser.add_argument("--compare", "-c", action="store_true", help="Also compare with original image")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--borders-only", "-b", action="store_true", help="Only add borders to existing reconstructed image")
    parser.add_argument("--no-borders", action="store_true", help="Skip creating bordered version")
    
    args = parser.parse_args()
    
    if args.borders_only:
        bordered_path = add_borders_to_existing_episode(args.episode, verbose=not args.quiet)
        print(f"\n[Result] Bordered image saved to: {bordered_path}")
        return
    
    result = reconstruct_episode(
        episode_dir=args.episode,
        output_name=args.output,
        save_bordered=not args.no_borders,
        verbose=not args.quiet
    )
    
    print(f"\n[Result] Reconstruction complete:")
    print(f"  Output: {result['output_path']}")
    if result.get('bordered_path'):
        print(f"  Bordered: {result['bordered_path']}")
    
    if args.compare:
        compare_result = compare_with_original(args.episode, verbose=not args.quiet)
        print(f"\n[Compare] MAE: {compare_result['mae']:.2f}")


if __name__ == "__main__":
    main()