# REDESIGN/tools/qwen_layered_tool.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path

from ..qwen_pool import get_qwen_pool

def run_qwen_layered(
    image_path: str,
    output_dir: str,
    num_layers: int = 4,
    gpu_ids: List[int] = None,  # 하위 호환성 (무시)
    seed: int = 777,
    resolution: int = 640,
    num_inference_steps: int = 50,
    true_cfg_scale: float = 4.0,
    strict_alpha_threshold: int = 240,
) -> Dict[str, Any]:
    """
    Qwen 3쌍 병렬 풀에 job 제출하여 실행.
    """
    pool = get_qwen_pool()

    output_dir = str(Path(output_dir))
    data = pool.submit(
        image_path=image_path,
        output_dir=output_dir,
        num_layers=num_layers,
        seed=seed,
        resolution=resolution,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        strict_alpha_threshold=strict_alpha_threshold,
    )
    return data

def is_qwen_available() -> bool:
    # 풀/워커에서 import 실패하면 start 로그가 뜸
    return True
