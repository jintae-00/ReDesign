# ReDesign/tool_gpu_manager.py
"""
Dynamic Tool GPU Manager

Dynamic configuration support:
- Reads its configuration from tool_gpu_config at runtime
- Can be configured via the URLD_TOOL_GPUS environment variable
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Callable, List, Set
import threading
import torch
import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict


class ToolModelType(Enum):
    GDINO = "gdino"
    SAM2 = "sam2"
    HISAM = "hisam"
    OCR = "ocr"
    LAMA = "lama"
    OBJECTCLEAR = "objectclear"


# Heavy model configuration (models that use a lot of memory)
HEAVY_MODELS: Set[ToolModelType] = {ToolModelType.OBJECTCLEAR}
HEAVY_MODEL_MIN_FREE_GB: float = 8.0


@dataclass
class ToolGPUSlot:
    gpu_id: int
    lock: threading.RLock = field(default_factory=threading.RLock)
    model_cache: OrderedDict = field(default_factory=OrderedDict)
    current_user: Optional[str] = None
    current_model: Optional[ToolModelType] = None
    total_requests: int = 0
    cache_hits: int = 0
    stream: Optional[torch.cuda.Stream] = None


@dataclass
class ToolContext:
    model: Any
    gpu_id: int
    device: str
    model_type: ToolModelType
    stream: Optional[torch.cuda.Stream] = None


class ToolGPUManager:
    _instance: Optional['ToolGPUManager'] = None
    _init_lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, tool_gpus: List[int] = None, force_reinit: bool = False):
        """
        Args:
            tool_gpus: List of tool GPU ids. If None, read from the config.
            force_reinit: If True, ignore existing settings and reinitialize.
        """
        if hasattr(self, '_initialized') and self._initialized and not force_reinit:
            return

        # Determine the GPU list
        if tool_gpus is not None:
            self.tool_gpus = tool_gpus
        else:
            # Read dynamically from the config
            from .tool_gpu_config import get_tool_gpus, get_max_models_per_gpu, get_lock_timeout
            self.tool_gpus = get_tool_gpus()
            self._max_models_per_gpu = get_max_models_per_gpu()
            self._lock_timeout = get_lock_timeout()

        # Configure the global SDP backend
        self._configure_global_sdp()
        
        self.slots: Dict[int, ToolGPUSlot] = {}
        for gpu_id in self.tool_gpus:
            slot = ToolGPUSlot(gpu_id=gpu_id)
            if torch.cuda.is_available():
                try:
                    with torch.cuda.device(gpu_id):
                        slot.stream = torch.cuda.Stream(device=gpu_id)
                except Exception as e:
                    print(f"[ToolGPUManager] Warning: Could not create stream for GPU {gpu_id}: {e}")
            self.slots[gpu_id] = slot
        
        self._model_loaders: Dict[ToolModelType, Callable[[int], Any]] = {}
        self._register_loaders()
        
        self._cuda_lock = threading.Lock()
        self._selection_lock = threading.Lock()
        
        self._initialized = True
        print(f"[ToolGPUManager] Initialized with GPUs: {self.tool_gpus}")
        print(f"[ToolGPUManager] Global SDP backend set to: MATH only")
        print(f"[ToolGPUManager] Max models per GPU: {getattr(self, '_max_models_per_gpu', 4)}")
    
    def _configure_global_sdp(self):
        """Pin the global SDP configuration at process startup."""
        if not torch.cuda.is_available():
            return
        
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            print(f"[ToolGPUManager] Warning: Could not configure SDP: {e}")
    
    def _register_loaders(self):
        """Register the model loaders."""

        def _safe_sync(gpu_id: int):
            """synchronize wrapper - avoids hanging even when CUDA is in an error state."""
            try:
                torch.cuda.synchronize(gpu_id)
            except RuntimeError as e:
                print(f"[ToolGPUManager] synchronize skipped (GPU {gpu_id}): {e}")

        def load_gdino(gpu_id: int):
            print(f"[ToolGPUManager] Loading GDINO on GPU {gpu_id}...")

            torch.cuda.set_device(gpu_id)
            _safe_sync(gpu_id)

            from modules.grounding_dino.groundingdino.util.inference import load_model
            from config import MODULES, WEIGHTS

            cfg = MODULES / "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
            ckpt = WEIGHTS / "groundingdino_swinb_cogcoor.pth"

            model = load_model(str(cfg), str(ckpt), device=f"cuda:{gpu_id}")

            _safe_sync(gpu_id)
            print(f"[ToolGPUManager] GDINO loaded on GPU {gpu_id}")
            return model
        
        def load_sam2(gpu_id: int):
            print(f"[ToolGPUManager] Loading SAM2 on GPU {gpu_id}...")

            torch.cuda.set_device(gpu_id)
            _safe_sync(gpu_id)

            from torch import _dynamo
            _dynamo.config.suppress_errors = True

            from modules.sam2.build_sam import build_sam2
            from modules.sam2.sam2_image_predictor import SAM2ImagePredictor
            from config import WEIGHTS

            cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            ckpt = WEIGHTS / "sam2.1_hiera_large.pt"

            with torch.jit.optimized_execution(False):
                sam2 = build_sam2(str(cfg), str(ckpt), device=f"cuda:{gpu_id}")

            _safe_sync(gpu_id)
            print(f"[ToolGPUManager] SAM2 loaded on GPU {gpu_id}")
            return SAM2ImagePredictor(sam2)
        
        def load_hisam(gpu_id: int):
            print(f"[ToolGPUManager] Loading HiSAM on GPU {gpu_id}...")

            torch.cuda.set_device(gpu_id)
            _safe_sync(gpu_id)

            from modules.hisam.inference import HiSam_Inference
            from config import WEIGHTS

            model = HiSam_Inference(
                check_point_dir=WEIGHTS,
                model_path="sam_tss_h_textseg.pth"
            )

            _safe_sync(gpu_id)
            print(f"[ToolGPUManager] HiSAM loaded on GPU {gpu_id}")
            return model
        
        def load_ocr(gpu_id: int):
            print(f"[ToolGPUManager] Loading OCR on GPU {gpu_id}...")
            
            torch.cuda.set_device(gpu_id)
            
            from modules.ocr.main import PaddleOCRClient
            client = PaddleOCRClient()
            
            print(f"[ToolGPUManager] OCR loaded on GPU {gpu_id}")
            return client
        
        def load_lama(gpu_id: int):
            print(f"[ToolGPUManager] Loading LaMa on GPU {gpu_id}...")

            torch.cuda.set_device(gpu_id)
            _safe_sync(gpu_id)

            from modules.textremover.lama import LaMa
            from config import WEIGHTS

            model = LaMa(model_path=str(WEIGHTS / "big-lama.pt"))

            _safe_sync(gpu_id)
            print(f"[ToolGPUManager] LaMa loaded on GPU {gpu_id}")
            return model
        
        def load_objectclear(gpu_id: int):
            print(f"[ToolGPUManager] Loading ObjectClear on GPU {gpu_id}...")

            torch.cuda.set_device(gpu_id)
            _safe_sync(gpu_id)
            
            from modules.ObjectClear.objectclear.pipelines import ObjectClearPipeline
            from config import WEIGHTS
            
            pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
                "jixin0101/ObjectClear",
                torch_dtype=torch.float16,
                apply_attention_guided_fusion=True,
                cache_dir=str(WEIGHTS),
                variant="fp16",
                low_cpu_mem_usage=False,
            )
            
            try:
                pipe = pipe.to(f"cuda:{gpu_id}")
            except RuntimeError as e:
                if "meta tensor" in str(e).lower():
                    print(f"[ToolGPUManager] Meta tensor detected, using alternative loading...")
                    del pipe
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()

                    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
                        "jixin0101/ObjectClear",
                        torch_dtype=torch.float16,
                        apply_attention_guided_fusion=True,
                        cache_dir=str(WEIGHTS),
                        variant="fp16",
                        device_map={"": f"cuda:{gpu_id}"},
                    )
                else:
                    raise

            _safe_sync(gpu_id)
            print(f"[ToolGPUManager] ObjectClear loaded on GPU {gpu_id}")
            return pipe
        
        self._model_loaders = {
            ToolModelType.GDINO: load_gdino,
            ToolModelType.SAM2: load_sam2,
            ToolModelType.HISAM: load_hisam,
            ToolModelType.OCR: load_ocr,
            ToolModelType.LAMA: load_lama,
            ToolModelType.OBJECTCLEAR: load_objectclear,
        }
    
    def _is_slot_available(self, slot: ToolGPUSlot) -> bool:
        return slot.current_user is None
    
    def _select_gpu(self, model_type: ToolModelType) -> int:
        with self._selection_lock:
            # 1. An available GPU that already has the model cached
            for gpu_id, slot in self.slots.items():
                if model_type in slot.model_cache and self._is_slot_available(slot):
                    return gpu_id

            # 2. An available GPU with the most free cache capacity
            available = []
            for gpu_id, slot in self.slots.items():
                if self._is_slot_available(slot):
                    available.append((gpu_id, len(slot.model_cache)))

            if available:
                available.sort(key=lambda x: x[1])
                return available[0][0]

            # 3. The first GPU (wait for its lock)
            return self.tool_gpus[0]
    
    def _prepare_for_heavy_model(self, slot: ToolGPUSlot, model_type: ToolModelType):
        """Free up memory before running a heavy model."""
        if model_type not in HEAVY_MODELS:
            return
        
        gpu_id = slot.gpu_id
        
        if torch.cuda.is_available():
            try:
                free_mem = torch.cuda.get_device_properties(gpu_id).total_memory
                free_mem -= torch.cuda.memory_allocated(gpu_id)
                free_gb = free_mem / 1024**3
                
                if free_gb < HEAVY_MODEL_MIN_FREE_GB:
                    print(f"[ToolGPUManager] Preparing for heavy model {model_type.value}")
                    print(f"[ToolGPUManager] Free memory: {free_gb:.1f}GB, need: {HEAVY_MODEL_MIN_FREE_GB}GB")
                    
                    models_to_evict = [
                        mt for mt in list(slot.model_cache.keys())
                        if mt != model_type
                    ]
                    
                    for mt in models_to_evict:
                        print(f"[ToolGPUManager] Evicting {mt.value} for heavy model")
                        model = slot.model_cache.pop(mt, None)
                        if model is not None:
                            del model
                    
                    self._safe_cleanup(gpu_id)
            except Exception as e:
                print(f"[ToolGPUManager] Warning: Could not check memory: {e}")
    
    def _ensure_model(self, slot: ToolGPUSlot, model_type: ToolModelType) -> Any:
        gpu_id = slot.gpu_id
        
        self._prepare_for_heavy_model(slot, model_type)
        
        if model_type in slot.model_cache:
            slot.cache_hits += 1
            slot.model_cache.move_to_end(model_type)
            return slot.model_cache[model_type]
        
        max_models = 1 if model_type in HEAVY_MODELS else getattr(self, '_max_models_per_gpu', 4)
        
        while len(slot.model_cache) >= max_models:
            oldest_type, oldest_model = slot.model_cache.popitem(last=False)
            print(f"[ToolGPUManager] Evicting {oldest_type.value} from GPU {gpu_id}")
            del oldest_model
            self._safe_cleanup(gpu_id)
        
        loader = self._model_loaders.get(model_type)
        if loader is None:
            raise ValueError(f"No loader for {model_type.value}")
        
        model = loader(gpu_id)
        slot.model_cache[model_type] = model
        return model
    
    def reinitialize_model(self, model_type: ToolModelType, gpu_id: int) -> Any:
        """
        Delete the cached model and load a fresh copy.
        Call this when an unrecoverable error occurs (e.g. std::exception,
        CUDA illegal memory access).
        Must be called while holding the lock of the corresponding slot.
        """
        slot = self.slots.get(gpu_id)
        if slot is None:
            raise ValueError(f"No slot for GPU {gpu_id}")

        # Delete the existing model
        old_model = slot.model_cache.pop(model_type, None)
        if old_model is not None:
            if hasattr(old_model, 'to'):
                try:
                    old_model.to('cpu')
                except Exception:
                    pass
            del old_model

        self._safe_cleanup(gpu_id)
        print(f"[ToolGPUManager] Reinitializing {model_type.value} on GPU {gpu_id}")

        # Load a fresh copy
        loader = self._model_loaders.get(model_type)
        if loader is None:
            raise ValueError(f"No loader for {model_type.value}")

        model = loader(gpu_id)
        slot.model_cache[model_type] = model
        return model

    def _safe_cleanup(self, gpu_id: int = None):
        with self._cuda_lock:
            if gpu_id is not None:
                try:
                    torch.cuda.set_device(gpu_id)
                except Exception:
                    pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[ToolGPUManager] empty_cache failed (GPU {gpu_id}): {e}")
            gc.collect()

    def _cleanup_model_state(self, model_type: ToolModelType, model):
        """Clean up per-model internal state (e.g. cached features) on release from acquire."""
        if model is None:
            return
        if model_type == ToolModelType.HISAM:
            # Release the HiSAM predictor's cached image features
            svc = getattr(model, 'hisam_service', None)
            if svc is not None:
                predictor = getattr(svc, 'predictor', None)
                if predictor is not None and hasattr(predictor, 'reset_image'):
                    predictor.reset_image()
        elif model_type == ToolModelType.SAM2:
            # Release the SAM2 predictor's cached features
            if hasattr(model, 'reset_predictor'):
                model.reset_predictor()
    
    @contextmanager
    def acquire(
        self,
        model_type: ToolModelType,
        timeout: float = None,
        caller_id: str = None,
    ):
        """Acquire a GPU and its model (including the lock)."""
        timeout = timeout or getattr(self, '_lock_timeout', 240.0)
        gpu_id = self._select_gpu(model_type)
        slot = self.slots[gpu_id]
        
        acquired = slot.lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(
                f"Timeout acquiring GPU {gpu_id} for {model_type.value} "
                f"(current_user={slot.current_user})"
            )
        
        try:
            slot.current_user = caller_id or threading.current_thread().name
            slot.current_model = model_type
            slot.total_requests += 1
            
            torch.cuda.set_device(gpu_id)
            
            stream = slot.stream
            
            model = self._ensure_model(slot, model_type)
            
            yield ToolContext(
                model=model,
                gpu_id=gpu_id,
                device=f"cuda:{gpu_id}",
                model_type=model_type,
                stream=stream,
            )
        finally:
            # Clean up per-model internal caches (e.g. predictor features)
            try:
                self._cleanup_model_state(model_type, slot.model_cache.get(model_type))
            except Exception:
                pass

            # CUDA cleanup - always release the lock even if synchronize hangs
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(gpu_id)
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    err_msg = str(e).lower()
                    print(
                        f"[ToolGPUManager] CUDA cleanup failed on GPU {gpu_id}: {e}"
                    )
                    # Fatal CUDA error (e.g. illegal memory access) ->
                    # invalidate all cached models on this GPU (reloaded on the next request)
                    if "illegal" in err_msg or "assert" in err_msg:
                        print(
                            f"[ToolGPUManager] Fatal CUDA error detected on GPU {gpu_id}, "
                            f"invalidating all cached models"
                        )
                        slot.model_cache.clear()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                except Exception as e:
                    print(
                        f"[ToolGPUManager] Unexpected CUDA cleanup error on GPU {gpu_id}: {e}"
                    )

            slot.current_user = None
            slot.current_model = None
            slot.lock.release()
    
    def get_status(self) -> Dict[str, Any]:
        status = {"tool_gpus": self.tool_gpus, "slots": {}}
        for gpu_id, slot in self.slots.items():
            cache_rate = slot.cache_hits / max(1, slot.total_requests) * 100
            status["slots"][f"gpu_{gpu_id}"] = {
                "current_user": slot.current_user,
                "current_model": slot.current_model.value if slot.current_model else None,
                "cached_models": [m.value for m in slot.model_cache.keys()],
                "cache_hit_rate": f"{cache_rate:.1f}%",
                "locked": slot.current_user is not None,
            }
        return status


# =============================================================================
# Singleton Management
# =============================================================================

_manager: Optional[ToolGPUManager] = None
_manager_lock = threading.Lock()


def get_tool_manager(tool_gpus: List[int] = None) -> ToolGPUManager:
    """Return the Tool GPU Manager singleton."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ToolGPUManager(tool_gpus=tool_gpus)
    return _manager


def reset_tool_manager():
    """Reset the manager (for testing)."""
    global _manager
    with _manager_lock:
        _manager = None


def reconfigure_tool_manager(tool_gpus: List[int]):
    """Reconfigure the manager."""
    global _manager
    with _manager_lock:
        _manager = ToolGPUManager(tool_gpus=tool_gpus, force_reinit=True)
    return _manager


def print_tool_gpu_status():
    manager = get_tool_manager()
    status = manager.get_status()
    print("\n" + "=" * 60)
    print("Tool GPU Manager Status")
    print("=" * 60)
    for gpu_name, info in status["slots"].items():
        print(f"  {gpu_name}: locked={info['locked']}, user={info['current_user']}, cached={info['cached_models']}")
    print("=" * 60)