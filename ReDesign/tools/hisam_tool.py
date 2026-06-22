# ReDesign/tools/hisam_tool.py
"""
HiSAM - Thread-safe with ToolGPUManager

[Changes]
1. Fix the closure late-binding problem (using a function factory).
2. Use the retry_helper module for unified retry logic.
3. Prevent GPU memory leaks via per-box predictor.reset_image() + empty_cache().
4. Enable eviction of other models on OOM.
"""
import cv2
import numpy as np
import torch
import gc
from pathlib import Path
from typing import List, Dict, Optional

from ..tool_gpu_manager import get_tool_manager, ToolModelType
from .retry_helper import retry_on_cuda_error, aggressive_memory_cleanup


def _clip_box_with_pad(x1, y1, x2, y2, W, H, pad_px: int):
    x1 = max(0, int(x1) - pad_px)
    y1 = max(0, int(y1) - pad_px)
    x2 = min(W, int(x2) + pad_px)
    y2 = min(H, int(y2) + pad_px)
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def _create_hisam_inference_fn(hisam, crop):
    """
    Create the HiSAM inference function (fixes the closure problem).

    Building the function outside the for loop avoids the late-binding problem.
    """
    def run_inference():
        return hisam(input_img=crop)
    return run_inference


def _reset_predictor(hisam):
    """Release the image features cached inside the predictor."""
    try:
        svc = hisam.hisam_service
        if hasattr(svc, 'predictor') and hasattr(svc.predictor, 'reset_image'):
            svc.predictor.reset_image()
    except Exception:
        pass


@torch.no_grad()
def run_hisam_union(
    image_path: str,
    boxes: List[List[int]],
    det_ids: List[str],
    vis_dir: Optional[Path] = None,
    step: int = 0,
    caller_id: str = None,
) -> Dict[str, str]:
    """
    Thread-safe HiSAM with unified CUDA error recovery
    """

    manager = get_tool_manager()

    with manager.acquire(ToolModelType.HISAM, caller_id=caller_id) as ctx:
        hisam = ctx.model
        gpu_id = ctx.gpu_id

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = img[:, :, ::-1]  # BGR -> RGB
        H, W, _ = img.shape

        diag = (H ** 2 + W ** 2) ** 0.5
        pad_px = int(0.01 * diag)



        per_id_raw = {det_id: np.zeros((H, W), np.uint8) for det_id in det_ids}
        union_dil = np.zeros((H, W), np.uint8) # For inpainting (dilated)

        for det_id, b in zip(det_ids or [], boxes or []):
            if b is None or len(b) != 4:
                continue
            x1, y1, x2, y2 = _clip_box_with_pad(*b, W, H, pad_px)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Fix the closure problem: use a function factory
            inference_fn = _create_hisam_inference_fn(hisam, crop)

            try:
                _, m_pil = retry_on_cuda_error(
                    func=inference_fn,
                    gpu_id=gpu_id,
                    model_name=f"HiSAM:{det_id}",
                    max_retries=3,
                    base_delay=0.5,
                    evict_other_models=True,
                    tool_manager=manager,
                    current_model_type=ToolModelType.HISAM,
                )
            except Exception as e:
                print(f"[HiSAM] inference failed for det_id={det_id}: {e}")
                # Release the predictor cache even on failure
                _reset_predictor(hisam)
                del crop
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                continue

            m_raw = (np.array(m_pil) > 127).astype(np.uint8)

            # 1. Store the per-ID mask as the original (raw)
            per_id_raw[det_id][y1:y2, x1:x2] = np.maximum(per_id_raw[det_id][y1:y2, x1:x2], m_raw)

            # 2. Apply dilation to the union mask for inpainting
            h_crop = y2 - y1
            dilate_px = max(1, int(0.01 * h_crop))
            kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
            m_dilated = cv2.dilate(m_raw, kernel, iterations=1)
            union_dil[y1:y2, x1:x2] = np.maximum(union_dil[y1:y2, x1:x2], m_dilated)

            # Per-box GPU memory cleanup: release the predictor cache + delete intermediates
            del crop, m_pil, m_raw, m_dilated, kernel
            _reset_predictor(hisam)
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # Release the original image after inference completes
        del img

        # Use per_id_raw (original) when saving masks_by_id
        base = Path(image_path)
        masks_by_id = {}
        for det_id in det_ids:
            m_full = per_id_raw[det_id]
            if np.any(m_full):
                m_path = str(base.with_name(f"{base.stem}_raw_{det_id}.png"))
                cv2.imwrite(m_path, m_full * 255)
                masks_by_id[det_id] = m_path

        # Use union_dil (dilated) when saving union_path
        if vis_dir:
            vis_dir.mkdir(parents=True, exist_ok=True)
            union_path = str(vis_dir / f"{step:03d}_HiSAM_union.png")
        else:
            union_path = str(base.with_name(f"{base.stem}_hisam_union.png"))

        cv2.imwrite(union_path, union_dil * 255)

        del per_id_raw, union_dil

        return {"mask_union": union_path, "masks_by_id": masks_by_id}