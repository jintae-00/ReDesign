# REDESIGN/qwen_worker.py
from __future__ import annotations
import os
import gc
import torch
import traceback
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List
from PIL import Image

def _detect_gpu_memory() -> Dict[str, Any]:
    """Detect the currently available GPU memory and return memory-limit information."""
    gpu_count = torch.cuda.device_count()
    limit_mem_dict = {}
    total_gb = 0
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / (1024 ** 3)
        total_gb += gb
        # Reserve about 2.5GB of headroom for computation when capping the allocation
        limit_mem_dict[i] = f"{int(gb - 2.5)}GiB"

    # Configuration for CPU offloading (assumes about 120GB of free CPU RAM)
    limit_mem_dict["cpu"] = "120GiB"
    return {"gpu_count": gpu_count, "total_gb": total_gb, "limit_mem_dict": limit_mem_dict}

def _load_pipeline_no_quant(offload_dir: str):
    """
    Revised loading strategy:
    1. Removed the unnecessary disk-offloading strategy (Strategy 2).
    2. Strengthened explicit memory cleanup on failure.
    """
    from diffusers import QwenImageLayeredPipeline
    
    gpu_info = _detect_gpu_memory()
    print(f"[QwenWorker] Detected Total VRAM: {gpu_info['total_gb']:.2f} GB")

    strategies = [
        # 1. Default balanced strategy (fastest, when VRAM is sufficient)
        {
            "name": "Vanilla Balanced (16-bit)",
            "kwargs": {
                "device_map": "balanced",
                "max_memory": None,
            }
        },
        # [Changed] Removed the intermediate 'No CPU' strategy.
        # When VRAM is insufficient, skipping CPU and going straight to disk is so slow it causes timeouts.

        # 2. CPU offloading (uses CPU RAM when GPU memory is insufficient - stable)
        {
            "name": "CPU-GPU Hybrid (16-bit, Balanced)",
            "kwargs": {
                "device_map": "balanced",
                # Pass the full dictionary including CPU -> falls back to CPU RAM when GPU memory runs out
                "max_memory": gpu_info["limit_mem_dict"],
            }
        },
        # 3. Last resort: Sequential CPU Offload (very slow but uses the least memory)
        # Enable if needed; here the balanced strategy is expected to be sufficient on its own.
    ]

    last_error = None
    pipeline = None  # Declare the variable up front

    for strategy in strategies:
        # [Important] Make sure to fully clear any leftovers from the previous attempt
        if pipeline is not None:
            del pipeline
            pipeline = None
        
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        
        try:
            print(f"\n[QwenWorker] >>> Attempting Strategy: {strategy['name']}")
            
            pipeline = QwenImageLayeredPipeline.from_pretrained(
                "Qwen/Qwen-Image-Layered",
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                offload_folder=offload_dir,
                offload_state_dict=True,
                low_cpu_mem_usage=True,
                **strategy["kwargs"]
            )
            
            print(f"[QwenWorker] Success! Loaded with {strategy['name']}.")
            # Print the allocation result
            for i in range(gpu_info["gpu_count"]):
                alloc = torch.cuda.memory_allocated(i) / (1024**3)
                print(f"  - GPU {i} Usage: {alloc:.2f} GB")
            
            return pipeline

        except Exception as e:
            print(f"[QwenWorker] Strategy {strategy['name']} failed: {e}")
            last_error = e
            
            # [Important] Perform memory cleanup immediately on failure
            if 'pipeline' in locals() and pipeline is not None:
                del pipeline
                pipeline = None
            
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue

    raise RuntimeError(f"All 16-bit loading strategies failed. Last error: {last_error}")

def worker_main(worker_id: str, physical_pair: Tuple[int, ...], in_q, out_q):
    """Worker main process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, physical_pair))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # HuggingFace cache location. Respects an externally-set HF_HOME, otherwise
    # falls back to the default user cache (~/.cache/huggingface). Checkpoints are
    # downloaded automatically on first use (see scripts/download_checkpoints.py).
    os.environ.setdefault(
        "HF_HOME",
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    )
    
    offload_dir = tempfile.mkdtemp(prefix=f"qwen_offload_{worker_id}_")
    
    try:
        pipeline = _load_pipeline_no_quant(offload_dir)
        print(f"[{worker_id}] Paper Experiment Worker Ready (BF16 Mode).")
    except Exception as e:
        out_q.put({"worker_id": worker_id, "ok": False, "error": str(e), "trace": traceback.format_exc()})
        return

    while True:
        job = in_q.get()
        if job is None: break
        
        job_id = job["job_id"]
        try:
            image_path = job["image_path"]
            output_dir = job["output_dir"]
            
            image = Image.open(image_path).convert("RGBA")
            original_size = image.size
            
            inputs = {
                "image": image,
                "generator": torch.Generator(device="cpu").manual_seed(int(job.get("seed", 777))),
                "num_inference_steps": int(job.get("num_inference_steps", 50)),
                # [FIX] Reverted to "layers" which is the correct argument name for the pipeline
                "layers": int(job.get("num_layers", 4)),
                "resolution": int(job.get("resolution", 640)),
                "true_cfg_scale": float(job.get("true_cfg_scale", 4.0)),
                "cfg_normalize": True,
                "use_en_prompt": True,
            }
            
            with torch.inference_mode():
                output = pipeline(**inputs)
                output_images = output.images[0]
            
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            layer_paths = []
            for i, layer_img in enumerate(output_images):
                layer_img = layer_img.convert("RGBA")
                if layer_img.size != original_size:
                    layer_img = layer_img.resize(original_size, Image.LANCZOS)
                
                p = out_path / f"layer_{i:02d}.png"
                layer_img.save(p)
                layer_paths.append(str(p))
            
            out_q.put({"worker_id": worker_id, "job_id": job_id, "ok": True, "data": {"layer_images": layer_paths}})
            
        except Exception as e:
            out_q.put({"worker_id": worker_id, "job_id": job_id, "ok": False, "error": str(e), "trace": traceback.format_exc()})
        finally:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if os.path.exists(offload_dir):
        shutil.rmtree(offload_dir)