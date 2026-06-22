# REDESIGN/tool_gpu_config.py
"""
GPU Configuration for URLD Tools

설정 우선순위:
1. 런타임 오버라이드 (set_runtime_config 호출)
2. 환경 변수 (URLD_QWEN_GPUS, URLD_TOOL_GPUS, URLD_OBJECTCLEAR_GPU)
3. 기본값 (아래 정의)

사용 예시:
    # CLI에서
    python -m REDESIGN.run_figma_split --split_idx 0 --qwen_gpus 2,3 --tool_gpus 6,7
    
    # 환경 변수로
    URLD_QWEN_GPUS="3,4,5" URLD_TOOL_GPUS="6,7" python -m REDESIGN.run_figma_split --split_idx 1
"""
from __future__ import annotations
import os
import threading
from typing import List, Tuple, Optional

# =============================================================================
# Default Values (fallback)
# =============================================================================
_DEFAULT_QWEN_GPUS: List[int] = [2, 3]  # 2-GPU pair as flat list
_DEFAULT_TOOL_GPUS: List[int] = [6, 7]
_DEFAULT_OBJECTCLEAR_GPU: int = 7
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
    런타임에 GPU 설정을 오버라이드합니다.
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
    """런타임 설정을 초기화합니다."""
    global _runtime_config
    with _config_lock:
        _runtime_config.clear()


def _parse_gpu_list_from_env(env_var: str) -> Optional[List[int]]:
    """환경 변수에서 GPU 리스트 파싱"""
    value = os.environ.get(env_var)
    if not value:
        return None
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError:
        print(f"[GPU Config] Warning: Invalid {env_var}={value}")
        return None


def _parse_gpu_int_from_env(env_var: str) -> Optional[int]:
    """환경 변수에서 단일 GPU ID 파싱"""
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
    """Qwen 모델용 GPU 리스트를 반환합니다."""
    with _config_lock:
        if "qwen_gpus" in _runtime_config:
            return list(_runtime_config["qwen_gpus"])
    
    env_value = _parse_gpu_list_from_env("URLD_QWEN_GPUS")
    if env_value is not None:
        return env_value
    
    return list(_DEFAULT_QWEN_GPUS)


def get_qwen_pair_size() -> Optional[int]:
    """Qwen GPU pair 크기를 반환합니다."""
    with _config_lock:
        if "qwen_pair_size" in _runtime_config:
            return _runtime_config["qwen_pair_size"]
    
    env_value = _parse_gpu_int_from_env("URLD_QWEN_PAIR_SIZE")
    if env_value is not None:
        return env_value
    
    return None  # 기본값: 모든 GPU를 단일 pair로


def get_qwen_gpu_pairs() -> List[Tuple[int, ...]]:
    """Qwen 모델용 GPU pairs를 반환합니다."""
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
    """Tool 모델용 GPU 리스트를 반환합니다."""
    with _config_lock:
        if "tool_gpus" in _runtime_config:
            return list(_runtime_config["tool_gpus"])
    
    env_value = _parse_gpu_list_from_env("URLD_TOOL_GPUS")
    if env_value is not None:
        return env_value
    
    return list(_DEFAULT_TOOL_GPUS)


def get_objectclear_gpu() -> int:
    """ObjectClear 전용 GPU를 반환합니다."""
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
# Legacy Compatibility - 기존 코드와의 호환성 (동적 조회)
# =============================================================================

# [FIX] property() cannot be used at module level. Use __getattr__ instead.
def __getattr__(name):
    """
    모듈 레벨의 동적 속성 접근을 처리합니다.
    imports 시점에는 존재하지 않지만, 접근 시점에 getter를 호출하여 값을 반환합니다.
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
    """설정 유효성 검사"""
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
    """현재 설정 출력"""
    print(f"[GPU Config] Qwen GPUs: {get_qwen_gpus()}")
    print(f"[GPU Config] Qwen Pair Size: {get_qwen_pair_size() or 'auto (single pair)'}")
    print(f"[GPU Config] Qwen GPU Pairs: {get_qwen_gpu_pairs()}")
    print(f"[GPU Config] Tool GPUs: {get_tool_gpus()}")
    print(f"[GPU Config] ObjectClear GPU: {get_objectclear_gpu()}")


def get_gpu_memory_gb(gpu_id: int) -> float:
    """특정 GPU의 총 메모리를 GB 단위로 반환"""
    try:
        import torch
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(gpu_id)
            return props.total_memory / (1024 ** 3)
    except Exception:
        pass
    return 24.0


def detect_gpu_type(gpu_id: int) -> str:
    """GPU 타입 감지 (A6000, RTX3090 등)"""
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