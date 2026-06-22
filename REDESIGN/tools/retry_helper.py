# REDESIGN/tools/retry_helper.py
"""
Unified retry helper for CUDA errors (OOM + Kernel + JIT errors)
"""
import torch
import gc
import time
from typing import Callable, Any, Optional, List


class CUDAErrorType:
    OOM = "oom"
    KERNEL = "kernel"
    JIT = "jit"
    FATAL = "fatal"    # Unrecoverable errors such as illegal memory access
    UNKNOWN = "unknown"


def classify_cuda_error(error: Exception) -> Optional[str]:
    """Classify the type of CUDA error."""
    error_str = str(error).lower()

    # Fatal errors — check first (takes priority over OOM)
    fatal_patterns = [
        "cuda error: an illegal memory access",
        "illegal memory access",
        "cudaerrorillegaladd",
        "device-side assert",
    ]
    for pattern in fatal_patterns:
        if pattern in error_str:
            return CUDAErrorType.FATAL

    oom_patterns = [
        "out of memory",
        "cuda out of memory",
        "cudnn error: cudnn_status_alloc_failed",
        "cuda error: out of memory",
        "tried to allocate",
        "memory allocation failed",
        "cudnn_status_internal_error",
    ]
    
    kernel_patterns = [
        "no available kernel",
        "no kernel",
        "cudnn_status_execution_failed",
        "cudnn_status_not_supported",
    ]
    
    jit_patterns = [
        "can't redefine method",
        "redefine method",
        "python compilation unit",
        "jit compilation",
        "__torch__",
        "torchscript",
    ]
    
    for pattern in oom_patterns:
        if pattern in error_str:
            return CUDAErrorType.OOM
    
    for pattern in kernel_patterns:
        if pattern in error_str:
            return CUDAErrorType.KERNEL
    
    for pattern in jit_patterns:
        if pattern in error_str:
            return CUDAErrorType.JIT
    
    cuda_keywords = ["cuda", "cudnn", "gpu", "device", "nccl"]
    if any(kw in error_str for kw in cuda_keywords):
        return CUDAErrorType.UNKNOWN
    
    return None


def aggressive_memory_cleanup(gpu_id: Optional[int] = None):
    """Aggressive memory cleanup — guards against synchronize hangs."""
    gc.collect()

    if torch.cuda.is_available():
        try:
            if gpu_id is not None:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(gpu_id)
            else:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"[MemoryCleanup] CUDA cleanup failed (GPU {gpu_id}): {e}")
        except Exception as e:
            print(f"[MemoryCleanup] Unexpected error during cleanup (GPU {gpu_id}): {e}")

    gc.collect()


def restore_sdp_settings():
    """Reapply SDP settings to force the Math kernel."""
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def get_gpu_memory_info(gpu_id: int) -> dict:
    """Retrieve GPU memory information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    try:
        try:
            torch.cuda.synchronize(gpu_id)
        except RuntimeError:
            pass  # CUDA error state — still report memory stats if possible
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        free = total - allocated
        
        return {
            "gpu_id": gpu_id,
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(free, 2),
            "utilization_pct": round(allocated / total * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def print_gpu_memory_status(gpu_ids: List[int] = None):
    """Print the GPU memory status."""
    if not torch.cuda.is_available():
        print("[GPU Memory] CUDA not available")
        return
    
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    print("\n" + "=" * 60)
    print("GPU Memory Status")
    print("=" * 60)
    for gpu_id in gpu_ids:
        info = get_gpu_memory_info(gpu_id)
        if "error" in info:
            print(f"  GPU {gpu_id}: Error - {info['error']}")
        else:
            print(f"  GPU {gpu_id}: {info['allocated_gb']:.1f}GB / {info['total_gb']:.1f}GB "
                  f"({info['utilization_pct']:.0f}%) [free: {info['free_gb']:.1f}GB]")
    print("=" * 60 + "\n")


def retry_on_cuda_error(
    func: Callable[[], Any],
    gpu_id: int,
    model_name: str = "Model",
    max_retries: int = 3,
    base_delay: float = 1.0,
    clear_cache_on_oom: bool = True,
    evict_other_models: bool = False,
    tool_manager=None,
    current_model_type=None,
) -> Any:
    """Retry the callable when a CUDA error (OOM + Kernel + JIT) occurs."""
    delays = [base_delay, base_delay * 2, base_delay * 4]
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return func()
            
        except RuntimeError as e:
            error_type = classify_cuda_error(e)
            
            if error_type is None:
                raise
            
            last_error = e
            
            if attempt >= max_retries - 1:
                print(f"[{model_name}] {error_type.upper()} error - all {max_retries} retries failed")
                print_gpu_memory_status([gpu_id])
                raise
            
            delay = delays[min(attempt, len(delays) - 1)]
            print(f"[{model_name}] {error_type.upper()} error (attempt {attempt + 1}/{max_retries})")
            print(f"[{model_name}] Error: {str(e)[:300]}...")
            
            # FATAL (illegal memory access): retrying is pointless; the model must be reinitialized
            if error_type == CUDAErrorType.FATAL:
                print(f"[{model_name}] FATAL CUDA error — reinitializing model on GPU {gpu_id}")
                if tool_manager and current_model_type:
                    try:
                        tool_manager.reinitialize_model(current_model_type, gpu_id)
                        print(f"[{model_name}] Model reinitialized successfully")
                    except Exception as reinit_err:
                        print(f"[{model_name}] Model reinitialization failed: {reinit_err}")
                        raise e  # Re-raise the original error
                else:
                    # Without a manager, recovery is impossible
                    raise

                aggressive_memory_cleanup(gpu_id)
                delay = base_delay * 2
                print(f"[{model_name}] Waiting {delay:.1f}s before retry with fresh model...")
                time.sleep(delay)
                continue

            if error_type == CUDAErrorType.OOM:
                print(f"[{model_name}] Attempting OOM recovery...")
                print_gpu_memory_status([gpu_id])
                
                if clear_cache_on_oom:
                    aggressive_memory_cleanup(gpu_id)
                
                # On OOM, unload other models starting from the first attempt
                if evict_other_models and tool_manager and current_model_type:
                    print(f"[{model_name}] Evicting ALL other models from GPU {gpu_id}...")
                    _evict_other_models(tool_manager, gpu_id, current_model_type)
                    aggressive_memory_cleanup(gpu_id)
                
                print_gpu_memory_status([gpu_id])
                delay *= 2
                
            elif error_type == CUDAErrorType.KERNEL:
                print(f"[{model_name}] Restoring SDP settings...")
                restore_sdp_settings()
                try:
                    torch.cuda.synchronize(gpu_id)
                except RuntimeError as sync_err:
                    print(f"[{model_name}] synchronize failed after kernel error: {sync_err}")

                if attempt >= 1:
                    aggressive_memory_cleanup(gpu_id)

            elif error_type == CUDAErrorType.JIT:
                print(f"[{model_name}] JIT compilation conflict detected...")
                print(f"[{model_name}] Waiting for other threads...")
                delay *= 3

                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize(gpu_id)
                    except RuntimeError as sync_err:
                        print(f"[{model_name}] synchronize failed after JIT error: {sync_err}")
            
            else:
                aggressive_memory_cleanup(gpu_id)
                restore_sdp_settings()
            
            print(f"[{model_name}] Waiting {delay:.1f}s before retry...")
            time.sleep(delay)
    
    raise last_error


def _evict_other_models(tool_manager, gpu_id: int, keep_model_type):
    """Unload all models on a given GPU except the current one."""
    try:
        slot = tool_manager.slots.get(gpu_id)
        if not slot:
            print(f"[MemoryManager] No slot found for GPU {gpu_id}")
            return
        
        print(f"[MemoryManager] Current cache on GPU {gpu_id}: {[mt.value for mt in slot.model_cache.keys()]}")
        
        models_to_evict = [
            model_type for model_type in list(slot.model_cache.keys())
            if model_type != keep_model_type
        ]
        
        if not models_to_evict:
            print(f"[MemoryManager] No other models to evict on GPU {gpu_id} (only {keep_model_type.value} cached)")
            return
        
        evicted_count = 0
        for model_type in models_to_evict:
            print(f"[MemoryManager] Evicting {model_type.value} from GPU {gpu_id}")
            model = slot.model_cache.pop(model_type, None)
            if model is not None:
                # Move all model parameters to CPU, then delete
                if hasattr(model, 'to'):
                    try:
                        model.to('cpu')
                    except:
                        pass
                del model
                evicted_count += 1
        
        if evicted_count > 0:
            aggressive_memory_cleanup(gpu_id)
            print(f"[MemoryManager] Evicted {evicted_count} models from GPU {gpu_id}")
        
    except Exception as e:
        print(f"[MemoryManager] Eviction failed: {e}")