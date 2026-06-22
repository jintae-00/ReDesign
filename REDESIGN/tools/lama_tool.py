# REDESIGN/tools/lama_tool.py
"""
LaMa - Thread-safe with ToolGPUManager

[수정 사항]
1. retry_helper 모듈 사용으로 통합 재시도 로직 적용
2. OOM + Kernel 에러 모두 처리
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
        
        # [수정] Alpha 반영 방식 (Premultiplied Alpha) -> 노이즈 제거
        raw_img = Image.open(image_path).convert("RGBA")
        raw_arr = np.array(raw_img)

        # 0~1 스케일로 정규화된 Alpha
        alpha = raw_arr[:, :, 3].astype(np.float32) / 255.0

        # RGB에 Alpha를 곱함 (Broadcasting)
        # Alpha가 작으면 RGB값도 0에 수렴(검은색)하게 됨
        clean_rgb = raw_arr[:, :, :3].astype(np.float32) * alpha[..., None]

        # 다시 이미지로 변환
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