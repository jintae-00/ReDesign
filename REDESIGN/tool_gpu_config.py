# REDESIGN/tool_gpu_config.py
"""
GPU Configuration for URLD Tools

Resolution priority:
1. Runtime override (via set_runtime_config)
2. Environment variables (URLD_QWEN_GPUS, URLD_TOOL_GPUS, URLD_OBJECTCLEAR_GPU)
3. Default values (defined below)

Usage examples:
    # From the CLI
    python -m REDESIGN.run_figma_split --split_idx 0 --qwen_gpus <QWEN_GPU_IDS> --tool_gpus <TOOL_GPU_IDS>

    # Via environment variables
    URLD_QWEN_GPUS="<QWEN_GPU_IDS>" URLD_TOOL_GPUS="<TOOL_GPU_IDS>" python -m REDESIGN.run_figma_split --split_idx 1

    # Replace <QWEN_GPU_IDS> and <TOOL_GPU_IDS> with your own comma-separated
    # GPU ids (e.g. "0,1").
"""
from __future__ import annotations
import os
import threading
from typing import List, Tuple, Optional

# =============================================================================
# Default Values (fallback)
# =============================================================================
# Conservative, machine-agnostic defaults: everything on GPU 0 so the pipeline
# runs out-of-the-box on any machine (including single-GPU setups). For multi-GPU
# performance, override per run via the CLI flags (--qwen_gpus / --tool_gpus /
# --objectclear_gpu) or the URLD_QWEN_GPUS / URLD_TOOL_GPUS / URLD_OBJECTCLEAR_GPU
# environment variables — e.g. --qwen_gpus 0,1,2,3 --qwen_pair_size 2 --tool_gpus 4,5.
_DEFAULT_QWEN_GPUS: List[int] = [0]
_DEFAULT_TOOL_GPUS: List[int] = [0]
_DEFAULT_OBJECTCLEAR_GPU: int = 0
_DEFAULT_MAX_MODELS_PER_GPU: int = 4
_DEFAULT_LOCK_TIMEOUT: float = 240.0

# =============================================================================
# Runtime Override Storage (thread-safe)
# =============================================================================
_runtime_config: dict = {}
_config_lock = threading.Lock()


def set_runtime_config(
    qwen_gpus: Optional[List[int]] = None,
    qwen_pair_size: Optional[int] = None,
    tool_gpus: Optional[List[int]] = None,
    objectclear_gpu: Optional[int] = None,
):
    """
    Override the GPU configuration at runtime.
    """
    global _runtime_config
    with _config_lock:
        if qwen_gpus is not None:
            _runtime_config["qwen_gpus"] = list(qwen_gpus)
        if qwen_pair_size is not None:
            _runtime_config["qwen_pair_size"] = qwen_pair_size
        if tool_gpus is not None:
            _runtime_config["tool_gpus"] = list(tool_gpus)
        if objectclear_gpu is not None:
            _runtime_config["objectclear_gpu"] = objectclear_gpu
    
    print(f"[GPU Config] Runtime override set: {_runtime_config}")


def clear_runtime_config():
    """Clear the runtime configuration."""
    global _runtime_config
    with _config_lock:
        _runtime_config.clear()


def _parse_gpu_list_from_env(env_var: str) -> Optional[List[int]]:
    """Parse a comma-separated GPU id list from an environment variable."""
    value = os.environ.get(env_var)
    if not value:
        return None
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError:
        print(f"[GPU Config] Warning: Invalid {env_var}={value}")
        return None


def _parse_gpu_int_from_env(env_var: str) -> Optional[int]:
    """Parse a single GPU id from an environment variable."""
    value = os.environ.get(env_var)
    if not value:
        return None
    try:
        return int(value.strip())
    except ValueError:
        print(f"[GPU Config] Warning: Invalid {env_var}={value}")
        return None


# =============================================================================
# Configuration Getters (with priority: runtime > env > default)
# =============================================================================

def get_qwen_gpus() -> List[int]:
    """Return the list of GPU ids for the Qwen model."""
    with _config_lock:
        if "qwen_gpus" in _runtime_config:
            return list(_runtime_config["qwen_gpus"])
    
    env_value = _parse_gpu_list_from_env("URLD_QWEN_GPUS")
    if env_value is not None:
        return env_value
    
    return list(_DEFAULT_QWEN_GPUS)


def get_qwen_pair_size() -> Optional[int]:
    """Return the Qwen GPU pair size."""
    with _config_lock:
        if "qwen_pair_size" in _runtime_config:
            return _runtime_config["qwen_pair_size"]
    
    env_value = _parse_gpu_int_from_env("URLD_QWEN_PAIR_SIZE")
    if env_value is not None:
        return env_value
    
    return None  # Default: treat all GPUs as a single pair


def get_qwen_gpu_pairs() -> List[Tuple[int, ...]]:
    """Return the GPU pairs for the Qwen model."""
    gpus = get_qwen_gpus()
    pair_size = get_qwen_pair_size()
    
    if pair_size is None or pair_size <= 0:
        return [tuple(gpus)]
    
    pairs = []
    for i in range(0, len(gpus), pair_size):
        chunk = gpus[i:i + pair_size]
        if len(chunk) == pair_size:
            pairs.append(tuple(chunk))
        else:
            print(f"[GPU Config] Warning: {len(chunk)} GPUs remaining, need {pair_size} for a pair. Skipping: {chunk}")
    
    if not pairs:
        print(f"[GPU Config] Warning: No valid pairs created. Using all GPUs as single pair.")
        return [tuple(gpus)]
    
    return pairs


def get_tool_gpus() -> List[int]:
    """Return the list of GPU ids for the tool models."""
    with _config_lock:
        if "tool_gpus" in _runtime_config:
            return list(_runtime_config["tool_gpus"])
    
    env_value = _parse_gpu_list_from_env("URLD_TOOL_GPUS")
    if env_value is not None:
        return env_value
    
    return list(_DEFAULT_TOOL_GPUS)


def get_objectclear_gpu() -> int:
    """Return the GPU id dedicated to ObjectClear."""
    with _config_lock:
        if "objectclear_gpu" in _runtime_config:
            return _runtime_config["objectclear_gpu"]
    
    env_value = _parse_gpu_int_from_env("URLD_OBJECTCLEAR_GPU")
    if env_value is not None:
        return env_value
    
    return _DEFAULT_OBJECTCLEAR_GPU


def get_max_models_per_gpu() -> int:
    return _DEFAULT_MAX_MODELS_PER_GPU


def get_lock_timeout() -> float:
    return _DEFAULT_LOCK_TIMEOUT


# =============================================================================
# Legacy Compatibility - backward compatibility with existing code (dynamic lookup)
# =============================================================================

# [FIX] property() cannot be used at module level. Use __getattr__ instead.
def __getattr__(name):
    """
    Handle dynamic attribute access at the module level.
    These attributes do not exist at import time; they call the corresponding
    getter on access to return the current value.
    """
    if name == "QWEN_GPU_PAIRS":
        return get_qwen_gpu_pairs()
    if name == "TOOL_GPUS":
        return get_tool_gpus()
    if name == "OBJECTCLEAR_GPU":
        return get_objectclear_gpu()
    if name == "MAX_MODELS_PER_GPU":
        return _DEFAULT_MAX_MODELS_PER_GPU
    if name == "DEFAULT_LOCK_TIMEOUT":
        return _DEFAULT_LOCK_TIMEOUT
        
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# =============================================================================
# Validation & Info
# =============================================================================

def validate_config() -> bool:
    """Validate the configuration."""
    qwen_gpus = set(get_qwen_gpus())
    tool_gpus = set(get_tool_gpus())
    objectclear_gpu = {get_objectclear_gpu()}
    
    overlap_qwen_tool = qwen_gpus & tool_gpus
    overlap_qwen_oc = qwen_gpus & objectclear_gpu
    
    if overlap_qwen_tool:
        print(f"[WARNING] GPU overlap between Qwen and Tools: {overlap_qwen_tool}")
    if overlap_qwen_oc:
        print(f"[WARNING] GPU overlap between Qwen and ObjectClear: {overlap_qwen_oc}")
    
    return True


def print_config():
    """Print the current configuration."""
    print(f"[GPU Config] Qwen GPUs: {get_qwen_gpus()}")
    print(f"[GPU Config] Qwen Pair Size: {get_qwen_pair_size() or 'auto (single pair)'}")
    print(f"[GPU Config] Qwen GPU Pairs: {get_qwen_gpu_pairs()}")
    print(f"[GPU Config] Tool GPUs: {get_tool_gpus()}")
    print(f"[GPU Config] ObjectClear GPU: {get_objectclear_gpu()}")


def get_gpu_memory_gb(gpu_id: int) -> float:
    """Return the total memory of a given GPU in GB."""
    try:
        import torch
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(gpu_id)
            return props.total_memory / (1024 ** 3)
    except Exception:
        pass
    return 24.0


def detect_gpu_type(gpu_id: int) -> str:
    """Detect the GPU type (e.g. A6000, RTX3090)."""
    try:
        import torch
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            name = torch.cuda.get_device_name(gpu_id)
            if "A6000" in name:
                return "A6000"
            elif "3090" in name:
                return "RTX3090"
            elif "4090" in name:
                return "RTX4090"
            elif "A100" in name:
                return "A100"
            return name
    except Exception:
        pass
    return "Unknown"