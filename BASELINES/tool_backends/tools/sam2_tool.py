# BASELINES/tool_backends/tools/sam2_tool.py
import numpy as np
import torch, gc
import cv2
from pathlib import Path

from modules.sam2.build_sam import build_sam2
from modules.sam2.sam2_image_predictor import SAM2ImagePredictor
from config import WEIGHTS, DEVICE

CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
CKPT = WEIGHTS / "sam2.1_hiera_large.pt"

_sam2_model = None
_sam2_pred  = None

def _get_sam2_predictor() -> SAM2ImagePredictor:
    global _sam2_model, _sam2_pred
    if _sam2_pred is None:
        _sam2_model = build_sam2(str(CFG), str(CKPT), device=DEVICE)
        _sam2_pred  = SAM2ImagePredictor(_sam2_model)
    return _sam2_pred

def reset_sam2_features():
    """Reset SAM2 predictor features without unloading the model."""
    if _sam2_pred is not None:
        _sam2_pred.reset_predictor()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def unload_sam2():
    global _sam2_model, _sam2_pred
    if _sam2_pred is not None:
        _sam2_pred.reset_predictor()
        del _sam2_pred
        _sam2_pred = None
    if _sam2_model is not None:
        try:
            _sam2_model.cpu()
        except Exception:
            pass
        del _sam2_model
        _sam2_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def _safe_uint8_mask(mask: np.ndarray) -> np.ndarray:
    m = mask
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8)
    return m

def _save_mask_like(image_path: str, suffix: str, mask: np.ndarray) -> str:
    p = Path(image_path)
    out = str(p.with_name(p.stem + suffix))
    cv2.imwrite(out, _safe_uint8_mask(mask) * 255)
    return out

def _clip(v, lo, hi):
    return max(lo, min(hi, int(v)))

def run_sam2_union(image_path: str, boxes: list, det_ids: list[str], pad_ratio: float = 0.01):
    """BBOX-based segmentation (per-det_id mask + union)"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = img_bgr[:, :, ::-1].copy()

    H, W = img_rgb.shape[:2]
    diag = (H ** 2 + W ** 2) ** 0.5
    pad_px = int(pad_ratio * diag)
    union = np.zeros((H, W), np.uint8)
    by_id = {}

    for det_id, (x1, y1, x2, y2) in zip(det_ids or [], boxes or []):
        cx1 = _clip(x1 - pad_px, 0, W); cy1 = _clip(y1 - pad_px, 0, H)
        cx2 = _clip(x2 + pad_px, 0, W); cy2 = _clip(y2 + pad_px, 0, H)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = img_rgb[cy1:cy2, cx1:cx2]
        rel_box = np.array([[x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]], dtype=np.float32)

        _pred = _get_sam2_predictor()
        _pred.set_image(crop)
        with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            masks, *_ = _pred.predict(
                point_coords=None,
                point_labels=None,
                box=rel_box,
                multimask_output=False,
            )
        mask = masks[0].astype(np.uint8)
        union[cy1:cy2, cx1:cx2] = np.maximum(union[cy1:cy2, cx1:cx2], mask)

        m_full = np.zeros((H, W), np.uint8)
        m_full[cy1:cy2, cx1:cx2] = mask
        by_id[det_id] = _save_mask_like(image_path, f"_sam2_bbox_{det_id}.png", m_full)

        # --- release references to large intermediate arrays ---
        del crop, rel_box, masks, mask, m_full

    union_path = _save_mask_like(image_path, "_obj_union_mask.png", union)

    result = {"mask_union": union_path, "masks_by_id": by_id}

    # release the predictor's internal image features (GPU tensors)
    reset_sam2_features()

    del img_bgr, img_rgb, union
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result


def run_sam2_points(image_path: str, boxes: list, det_ids: list[str], points: list, pad_ratio: float = 0.01):
    pred = _get_sam2_predictor()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = img_bgr[:, :, ::-1].copy()
    H, W = img_rgb.shape[:2]
    pts = np.array(points or [], dtype=np.float32)

    diag = (H ** 2 + W ** 2) ** 0.5
    pad_px = int(pad_ratio * diag)
    union = np.zeros((H, W), np.uint8)
    by_id = {}

    for det_id, (x1, y1, x2, y2) in zip(det_ids or [], boxes or []):
        cx1 = _clip(x1 - pad_px, 0, W); cy1 = _clip(y1 - pad_px, 0, H)
        cx2 = _clip(x2 + pad_px, 0, W); cy2 = _clip(y2 + pad_px, 0, H)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crop = img_rgb[cy1:cy2, cx1:cx2]

        in_box = (pts[:, 0] >= cx1) & (pts[:, 0] <= cx2) & (pts[:, 1] >= cy1) & (pts[:, 1] <= cy2)
        sub = pts[in_box]
        rel_pts, rel_lbl = None, None
        if sub.size > 0:
            rel_pts = sub.copy()
            rel_pts[:, 0] -= cx1
            rel_pts[:, 1] -= cy1
            rel_lbl = np.ones((1, rel_pts.shape[0]), dtype=np.float32)

        _pred = _get_sam2_predictor()
        _pred.set_image(crop)
        with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            masks, *_ = _pred.predict(
                point_coords=(rel_pts[None, ...] if rel_pts is not None else None),
                point_labels=rel_lbl,
                box=None,
                multimask_output=False,
            )
        mask = masks[0].astype(np.uint8)
        union[cy1:cy2, cx1:cx2] = np.maximum(union[cy1:cy2, cx1:cx2], mask)

        m_full = np.zeros((H, W), np.uint8)
        m_full[cy1:cy2, cx1:cx2] = mask
        by_id[det_id] = _save_mask_like(image_path, f"_sam2_points_{det_id}.png", m_full)

    union_path = _save_mask_like(image_path, "_points_mask.png", union)

    reset_sam2_features()
    del img_bgr, img_rgb, union
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {"mask_union": union_path, "masks_by_id": by_id}

def run_sam2_bbox_points(image_path: str, boxes: list, det_ids: list[str], points: list, pad_ratio: float = 0.01):
    pred = _get_sam2_predictor()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = img_bgr[:, :, ::-1].copy()
    H, W = img_rgb.shape[:2]
    pts = np.array(points or [], dtype=np.float32)

    diag = (H ** 2 + W ** 2) ** 0.5
    pad_px = int(pad_ratio * diag)
    union = np.zeros((H, W), np.uint8)
    by_id = {}

    for det_id, (x1, y1, x2, y2) in zip(det_ids or [], boxes or []):
        cx1 = _clip(x1 - pad_px, 0, W); cy1 = _clip(y1 - pad_px, 0, H)
        cx2 = _clip(x2 + pad_px, 0, W); cy2 = _clip(y2 + pad_px, 0, H)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crop = img_rgb[cy1:cy2, cx1:cx2]

        rel_pts, rel_lbl = None, None
        if pts.size > 0:
            in_box = (pts[:, 0] >= cx1) & (pts[:, 0] <= cx2) & (pts[:, 1] >= cy1) & (pts[:, 1] <= cy2)
            sub = pts[in_box]
            if sub.size > 0:
                rel_pts = sub.copy()
                rel_pts[:, 0] -= cx1
                rel_pts[:, 1] -= cy1
                rel_lbl = np.ones((1, rel_pts.shape[0]), dtype=np.float32)

        rel_box = np.array([[x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]], dtype=np.float32)

        _pred = _get_sam2_predictor()
        _pred.set_image(crop)
        with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            masks, *_ = _pred.predict(
                point_coords=(rel_pts[None, ...] if rel_pts is not None else None),
                point_labels=rel_lbl,
                box=rel_box,
                multimask_output=False,
            )
        mask = masks[0].astype(np.uint8)
        union[cy1:cy2, cx1:cx2] = np.maximum(union[cy1:cy2, cx1:cx2], mask)

        m_full = np.zeros((H, W), np.uint8)
        m_full[cy1:cy2, cx1:cx2] = mask
        by_id[det_id] = _save_mask_like(image_path, f"_bbox_points_{det_id}.png", m_full)

    union_path = _save_mask_like(image_path, "_bbox_points_mask.png", union)

    reset_sam2_features()
    del img_bgr, img_rgb, union
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {"mask_union": union_path, "masks_by_id": by_id}
