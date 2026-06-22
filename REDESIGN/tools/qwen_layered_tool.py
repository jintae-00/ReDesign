# REDESIGN/tools/qwen_layered_tool.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path

from ..qwen_pool import get_qwen_pool

def run_qwen_layered(
    image_path: str,
    output_dir: str,
    num_layers: int = 4,
    gpu_ids: List[int] = None,  # Kept for backward compatibility (ignored)
    seed: int = 777,
    resolution: int = 640,
    num_inference_steps: int = 50,
    true_cfg_scale: float = 4.0,
    strict_alpha_threshold: int = 240,
) -> Dict[str, Any]:
    """
    Submit a job to the parallel Qwen pool (3 worker pairs) and run it.
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
    # If an import fails in the pool/worker, a startup log will be emitted
    return True
