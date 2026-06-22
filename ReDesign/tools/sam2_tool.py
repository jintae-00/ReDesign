"""
SAM2 - Thread-safe with ToolGPUManager

[Changes]
1. Separate the dilated union mask (for inpainting) from the per-instance raw mask (for extraction).
2. Fix the undefined 'box' variable error (use y2 - y1).
"""
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Dict
from contextlib import nullcontext

from ..tool_gpu_manager import get_tool_manager, ToolModelType
from .retry_helper import retry_on_cuda_error, aggressive_memory_cleanup


def _safe_uint8_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) if mask.dtype != np.uint8 else mask


def _save_mask_like(image_path: str, suffix: str, mask: np.ndarray) -> str:
    p = Path(image_path)
    out = str(p.with_name(p.stem + suffix))
    cv2.imwrite(out, _safe_uint8_mask(mask) * 255)
    return out


def _clip(v, lo, hi):
    return max(lo, min(hi, int(v)))


def _create_sam2_predict_fn(pred, rel_box):
    def run_predict():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks, *_ = pred.predict(
                point_coords=None, point_labels=None,
                box=rel_box, multimask_output=False,
            )
        return masks
    return run_predict


@torch.no_grad()
def run_sam2_union(
    image_path: str,
    boxes: List,
    det_ids: List[str],
    pad_ratio: float = 0.01,
    caller_id: str = None,
) -> Dict:
    """
    Thread-safe SAM2 bbox segmentation with dual-mask strategy
    """
    
    manager = get_tool_manager()
    
    with manager.acquire(ToolModelType.SAM2, caller_id=caller_id) as ctx:
        pred = ctx.model
        device = ctx.device
        gpu_id = ctx.gpu_id
        stream = ctx.stream
        
        stream_context = torch.cuda.stream(stream) if stream else nullcontext()
        
        with stream_context:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            img_rgb = img_bgr[:, :, ::-1].copy()
            
            H, W = img_rgb.shape[:2]
            diag = (H ** 2 + W ** 2) ** 0.5
            pad_px = int(pad_ratio * diag)

            # 1. Initialize canvases for the original mask and the dilated mask
            union_dil = np.zeros((H, W), np.uint8) # Union for inpainting (dilated)
            by_id = {} # Per-instance mask for extraction (original)
            
            for det_id, (x1, y1, x2, y2) in zip(det_ids or [], boxes or []):
                cx1, cy1 = _clip(x1 - pad_px, 0, W), _clip(y1 - pad_px, 0, H)
                cx2, cy2 = _clip(x2 + pad_px, 0, W), _clip(y2 + pad_px, 0, H)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                
                crop = img_rgb[cy1:cy2, cx1:cx2]
                rel_box = np.array([[x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]], dtype=np.float32)
                
                # set_image
                pred.set_image(crop)
                
                predict_fn = _create_sam2_predict_fn(pred, rel_box)
                
                masks = retry_on_cuda_error(
                    func=predict_fn,
                    gpu_id=gpu_id,
                    model_name=f"SAM2:{det_id}",
                    max_retries=3,
                    base_delay=0.5,
                )

                # (1) Extract the raw mask (original)
                m_raw = (masks[0] > 0).astype(np.uint8)

                # Store the per-ID mask as the original (raw)
                m_full_raw = np.zeros((H, W), np.uint8)
                m_full_raw[cy1:cy2, cx1:cx2] = m_raw
                by_id[det_id] = _save_mask_like(image_path, f"_raw_{det_id}.png", m_full_raw)

                # (2) Apply dilation to the union mask for better inpainting quality
                h_box = y2 - y1 # Use the unpacked coordinates instead of box[3]-box[1]
                dilate_px = max(1, int(0.01 * h_box))
                kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
                m_dil = cv2.dilate(m_raw, kernel, iterations=1)
                union_dil[cy1:cy2, cx1:cx2] = np.maximum(union_dil[cy1:cy2, cx1:cx2], m_dil)
                
            if stream:
                stream.synchronize()
        
        # Save the dilated union mask for inpainting
        union_path = _save_mask_like(image_path, "_obj_union_mask.png", union_dil)
    
    return {"mask_union": union_path, "masks_by_id": by_id}