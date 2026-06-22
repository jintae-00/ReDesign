#!/usr/bin/env python3
"""Shared scene/edit helpers for task evaluators."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .common_utils import (
    apply_opacity_rgba,
    bbox_from_mask,
    cv2,
    dilate_mask,
    element_to_rgba,
    recolor_rgba_hsv,
    relative_dilation_radius,
    rgba_to_element_like,
    rotate_rgba,
    scale_rgba,
    transform_rgba,
    translate_rgba,
    _require_cv2,
)


def render_scene_rgba(elements: List[Dict[str, Any]], canvas_size: Tuple[int, int]) -> np.ndarray:
    """
    Render scene RGBA with the same overwrite policy used in evaluation_figma.

    evaluation_figma.compute_composite_metrics enables overwrite mode for agent
    reconstructions (text-only overwrite in composite_elements_transparent).
    Keeping that parity avoids reconstruction mismatches between pipelines.
    """
    from evaluation.figma_metrics import composite_elements_transparent

    sources = {str(e.get("source", "")).lower() for e in elements if isinstance(e, dict)}
    is_agent_scene = ("agent" in sources) and not ("qwen" in sources or "gt" in sources)
    img, _ = composite_elements_transparent(elements, canvas_size, use_overwrite=is_agent_scene)
    return np.array(img.convert("RGBA"), dtype=np.uint8)


def deep_copy_elements(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Keep helper for compatibility, but avoid deep-copying PIL images/masks by default.
    # A shallow element copy is enough for most call sites.
    return [dict(e) for e in elements]


def _apply_on_element(elem: Dict[str, Any], canvas_size: Tuple[int, int], edit: Dict[str, Any]) -> Dict[str, Any]:
    task_type = edit["task_type"]
    rgba = element_to_rgba(elem, canvas_size)

    def _scale_limits_from_alpha(alpha_mask: np.ndarray) -> Tuple[Tuple[float, float], float, float]:
        h, w = alpha_mask.shape
        if not bool(alpha_mask.any()):
            return (w * 0.5, h * 0.5), 1.0, 1.0
        x1, y1, x2, y2 = bbox_from_mask(alpha_mask)
        cx = 0.5 * (float(x1) + float(x2))
        cy = 0.5 * (float(y1) + float(y2))
        left = max(1e-6, cx - float(x1))
        right = max(1e-6, float(x2) - cx)
        top = max(1e-6, cy - float(y1))
        bottom = max(1e-6, float(y2) - cy)

        max_sx = min(cx / left, (float(w) - cx) / right)
        max_sy = min(cy / top, (float(h) - cy) / bottom)
        max_sx = max(1.0, float(max_sx))
        max_sy = max(1.0, float(max_sy))
        return (cx, cy), max_sx, max_sy

    if task_type == "delete":
        rgba[:] = 0
    elif task_type == "transition":
        h, w = rgba.shape[:2]
        alpha = rgba[..., 3] > 0
        if bool(alpha.any()):
            x1, y1, x2, y2 = bbox_from_mask(alpha)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            dx = float(edit.get("dx", 0))
            dy = float(edit.get("dy", 0))
            if "dx_ratio" in edit:
                dx += float(edit.get("dx_ratio", 0.0)) * bw
            if "dy_ratio" in edit:
                dy += float(edit.get("dy_ratio", 0.0)) * bh
            if "dx_canvas_ratio" in edit:
                dx += float(edit.get("dx_canvas_ratio", 0.0)) * float(w)
            if "dy_canvas_ratio" in edit:
                dy += float(edit.get("dy_canvas_ratio", 0.0)) * float(h)
            max_left = -float(x1)
            max_right = float(w - x2)
            max_up = -float(y1)
            max_down = float(h - y2)

            # Aggressive relocation: push to far corner while staying in-canvas.
            if bool(edit.get("aggressive", False)):
                frac = float(edit.get("aggressive_fraction", 0.95))
                frac = max(0.0, min(1.0, frac))
                sx = int(edit.get("x_sign", 0))
                sy = int(edit.get("y_sign", 0))
                if sx == 0:
                    sx = 1 if abs(max_right) >= abs(max_left) else -1
                if sy == 0:
                    sy = 1 if abs(max_down) >= abs(max_up) else -1
                dx = (max_right if sx > 0 else max_left) * frac
                dy = (max_down if sy > 0 else max_up) * frac

            dx = min(max(dx, max_left), max_right)
            dy = min(max(dy, max_up), max_down)
            rgba = translate_rgba(rgba, int(round(dx)), int(round(dy)))
        else:
            rgba = translate_rgba(rgba, int(edit.get("dx", 0)), int(edit.get("dy", 0)))
    elif task_type == "rotation":
        rgba = rotate_rgba(rgba, edit.get("angle_deg", 0.0))
    elif task_type == "opacity":
        if "min_alpha_delta" in edit:
            out = rgba.copy()
            alpha = out[..., 3].astype(np.int16)
            fg = alpha > 0
            delta = max(0, int(round(float(edit.get("min_alpha_delta", 100)))))
            if bool(fg.any()) and delta > 0:
                alpha_inc = np.minimum(255, alpha + delta)
                alpha_dec = np.maximum(0, alpha - delta)
                diff_inc = float(np.mean(np.abs(alpha_inc[fg] - alpha[fg])))
                diff_dec = float(np.mean(np.abs(alpha_dec[fg] - alpha[fg])))
                alpha_new = alpha_inc if diff_inc >= diff_dec else alpha_dec
                out[..., 3] = alpha_new.astype(np.uint8)
            rgba = out
        else:
            rgba = apply_opacity_rgba(rgba, edit.get("opacity_factor", 1.0))
    elif task_type == "recolor":
        rgba = recolor_rgba_hsv(
            rgba,
            hue_shift_deg=edit.get("hue_shift_deg", 0.0),
            sat_mul=edit.get("sat_mul", 1.0),
            val_mul=edit.get("val_mul", 1.0),
        )
    elif task_type == "text_bold":
        _require_cv2()
        alpha = rgba[..., 3]
        k = np.ones((3, 3), dtype=np.uint8)
        dil = cv2.dilate(alpha, k, iterations=max(1, int(edit.get("strength", 1))))
        rgba[..., 3] = np.maximum(alpha, dil)
    elif task_type == "text_italic":
        shear = float(edit.get("shear", 0.15))
        h, w = rgba.shape[:2]
        mat = np.array([[1.0, shear, -shear * h * 0.5], [0.0, 1.0, 0.0]], dtype=np.float32)
        rgba = transform_rgba(rgba, mat)
    elif task_type == "super_scaling":
        # Super-scaling is enlargement-only; if out-of-bounds, move back inside.
        h, w = rgba.shape[:2]
        alpha = rgba[..., 3] > 0
        if bool(alpha.any()):
            x1, y1, x2, y2 = bbox_from_mask(alpha)
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            cx = 0.5 * (float(x1) + float(x2))
            cy = 0.5 * (float(y1) + float(y2))

            s_req = max(1.01, float(edit.get("scale", 1.0)))
            max_fit_x = max(1.0, (float(w) - 2.0) / bw)
            max_fit_y = max(1.0, (float(h) - 2.0) / bh)
            s = min(s_req, max_fit_x, max_fit_y)
            s = max(1.0, s)

            nx1 = cx - (cx - float(x1)) * s
            nx2 = cx + (float(x2) - cx) * s
            ny1 = cy - (cy - float(y1)) * s
            ny2 = cy + (float(y2) - cy) * s
            tx = 0.0
            ty = 0.0
            if nx1 < 0.0:
                tx += -nx1
            if nx2 + tx > float(w):
                tx += float(w) - (nx2 + tx)
            if ny1 < 0.0:
                ty += -ny1
            if ny2 + ty > float(h):
                ty += float(h) - (ny2 + ty)

            mat = np.array(
                [[s, 0.0, cx - s * cx + tx], [0.0, s, cy - s * cy + ty]],
                dtype=np.float32,
            )
            rgba = transform_rgba(rgba, mat)
    elif task_type == "aspect_ratio":
        alpha = rgba[..., 3] > 0
        center, max_sx, max_sy = _scale_limits_from_alpha(alpha)
        sx_req = float(edit.get("scale_x", 1.0))
        sy_req = float(edit.get("scale_y", 1.0))
        sx = min(sx_req, max_sx) if sx_req > 1.0 else sx_req
        sy = min(sy_req, max_sy) if sy_req > 1.0 else sy_req
        sx = max(0.01, float(sx))
        sy = max(0.01, float(sy))
        rgba = scale_rgba(rgba, sx, sy, center=center)
    elif task_type == "corner_radius":
        # Raster approximation: mild blur on alpha channel to round corners.
        alpha = rgba[..., 3]
        k = max(1, int(edit.get("radius_px", 4)))
        blur = cv2.GaussianBlur(alpha, (k * 2 + 1, k * 2 + 1), 0)
        rgba[..., 3] = np.clip(blur, 0, 255).astype(np.uint8)
    elif task_type == "stroke":
        from .common_utils import edge_mask_from_alpha

        edge = edge_mask_from_alpha(rgba[..., 3], radius=max(1, int(edit.get("stroke_width", 1))))
        color = edit.get("stroke_rgb", [0, 0, 0])
        rgba[edge, :3] = np.array(color, dtype=np.uint8)
        rgba[edge, 3] = 255
    elif task_type == "point_edit":
        _require_cv2()
        h, w = rgba.shape[:2]
        src = np.array(
            [[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]],
            dtype=np.float32,
        )
        dst = src.copy()

        # Complex mode: explicit offsets for all 4 corners.
        corner_offsets = edit.get("corner_offsets")
        if isinstance(corner_offsets, (list, tuple)) and len(corner_offsets) == 4:
            for ci in range(4):
                off = corner_offsets[ci]
                if not isinstance(off, (list, tuple)) or len(off) < 2:
                    continue
                dx = float(off[0]) * float(w)
                dy = float(off[1]) * float(h)
                dst[ci, 0] = np.clip(dst[ci, 0] + dx, 0.0, float(w - 1))
                dst[ci, 1] = np.clip(dst[ci, 1] + dy, 0.0, float(h - 1))
        else:
            # Legacy mode: single-corner edit.
            corner = int(edit.get("corner", 0)) % 4
            dx = float(edit.get("dx_ratio", 0.0)) * float(w)
            dy = float(edit.get("dy_ratio", 0.0)) * float(h)
            dst[corner, 0] = np.clip(dst[corner, 0] + dx, 0.0, float(w - 1))
            dst[corner, 1] = np.clip(dst[corner, 1] + dy, 0.0, float(h - 1))

        mat = cv2.getPerspectiveTransform(src, dst)
        rgba = cv2.warpPerspective(
            rgba,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    return rgba_to_element_like(elem, rgba)


def apply_edit_to_scene(
    elements: List[Dict[str, Any]],
    target_indices: Sequence[int],
    canvas_size: Tuple[int, int],
    edit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    # Important for memory: do not deepcopy the entire scene (each element may hold
    # full-canvas RGBA and mask). We only materialize edited elements.
    out = list(elements)
    for idx in target_indices:
        if 0 <= idx < len(out):
            out[idx] = _apply_on_element(elements[idx], canvas_size, edit)

    # z-order is scene-level operation.
    if edit.get("task_type") == "z_order" and target_indices:
        direction = edit.get("direction", "front")
        for idx in target_indices:
            if 0 <= idx < len(out):
                pair_idx = find_z_order_swap_partner(out, idx, direction=direction)
                if pair_idx is None:
                    continue
                if out[idx] is elements[idx]:
                    out[idx] = dict(out[idx])
                if out[pair_idx] is elements[pair_idx]:
                    out[pair_idx] = dict(out[pair_idx])
                z_a = int(out[idx].get("z_index", idx))
                z_b = int(out[pair_idx].get("z_index", pair_idx))
                out[idx]["z_index"] = z_b
                out[pair_idx]["z_index"] = z_a

    return out


def find_z_order_swap_partner(
    elements: List[Dict[str, Any]],
    target_idx: int,
    direction: str = "front",
) -> int | None:
    """Find immediate overlapping z-neighbor to swap with."""
    if not (0 <= target_idx < len(elements)):
        return None
    if direction not in {"front", "back"}:
        return None

    target = elements[target_idx]
    target_mask = target.get("mask")
    if target_mask is None:
        return None
    target_bin = target_mask > 0
    if not bool(target_bin.any()):
        return None

    z_t = int(target.get("z_index", target_idx))
    best_idx = None
    best_score = None

    for j, elem in enumerate(elements):
        if j == target_idx:
            continue
        m = elem.get("mask")
        if m is None:
            continue
        m_bin = m > 0
        ov = int((target_bin & m_bin).sum())
        if ov <= 0:
            continue

        z_j = int(elem.get("z_index", j))
        if direction == "front":
            dz = z_j - z_t
            if dz <= 0:
                continue
            score = (dz, -ov)
        else:
            dz = z_t - z_j
            if dz <= 0:
                continue
            score = (dz, -ov)

        if best_score is None or score < best_score:
            best_score = score
            best_idx = j

    return best_idx


def has_z_order_overlap_neighbor(
    elements: List[Dict[str, Any]],
    target_idx: int,
    direction: str,
) -> bool:
    return find_z_order_swap_partner(elements, target_idx, direction=direction) is not None


def build_local_roi(mask: np.ndarray, bbox: Sequence[int], dilation_ratio: float = 0.08) -> np.ndarray:
    r = relative_dilation_radius(bbox, ratio=dilation_ratio)
    return dilate_mask(mask > 0, r)


def union_mask_from_indices(elements: List[Dict[str, Any]], indices: Sequence[int]) -> np.ndarray:
    if not indices:
        any_elem = elements[0]
        return np.zeros_like(any_elem["mask"], dtype=bool)
    m = np.zeros_like(elements[indices[0]]["mask"], dtype=bool)
    for idx in indices:
        if 0 <= idx < len(elements):
            m |= elements[idx]["mask"] > 0
    return m


def bbox_from_indices(elements: List[Dict[str, Any]], indices: Sequence[int]) -> List[int]:
    if not indices:
        return [0, 0, 1, 1]
    union = union_mask_from_indices(elements, indices)
    return bbox_from_mask(union)
