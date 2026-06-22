# REDESIGN/tools/lama_tool.py
"""
LaMa - Thread-safe with ToolGPUManager

[Changes]
1. Use the retry_helper module for unified retry logic.
2. Handle both OOM and Kernel errors.
"""
import torch
from PIL import Image
from pathlib import Path
import numpy as np

from ..tool_gpu_manager import get_tool_manager, ToolModelType
from .retry_helper import retry_on_cuda_error, aggressive_memory_cleanup


@torch.no_grad()
def run_lama(
    image_path: str,
    mask_path: str,
    caller_id: str = None,
) -> str:
    """
    Thread-safe LaMa inpainting with unified CUDA error recovery
    """
    
    manager = get_tool_manager()
    
    with manager.acquire(ToolModelType.LAMA, caller_id=caller_id) as ctx:
        lama = ctx.model
        gpu_id = ctx.gpu_id
        
        # Apply alpha via premultiplied alpha to remove noise
        raw_img = Image.open(image_path).convert("RGBA")
        raw_arr = np.array(raw_img)

        # Alpha normalized to the 0-1 range
        alpha = raw_arr[:, :, 3].astype(np.float32) / 255.0

        # Multiply RGB by alpha (broadcasting).
        # Where alpha is small, RGB values converge toward 0 (black).
        clean_rgb = raw_arr[:, :, :3].astype(np.float32) * alpha[..., None]

        # Convert back into an image
        img = Image.fromarray(clean_rgb.astype(np.uint8), mode="RGB")
        
        mask = Image.open(mask_path).convert("L")
        
        # LaMa inference with retry
        def run_lama_inference():
            return lama.remove_text_by_mask(img, mask)
        
        out, _, _ = retry_on_cuda_error(
            func=run_lama_inference,
            gpu_id=gpu_id,
            model_name="LaMa",
            max_retries=3,
            base_delay=1.0,
        )
        
        out_path = str(Path(image_path).with_suffix("")) + "_inpaint.png"
        out.save(out_path)
    
    return out_path