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
    """현재 가용 GPU 메모리를 감지하여 제한 정보를 반환합니다."""
    gpu_count = torch.cuda.device_count()
    limit_mem_dict = {}
    total_gb = 0
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / (1024 ** 3)
        total_gb += gb
        # 계산용 여유 공간을 위해 2.5GB 정도 제외하고 할당 제한
        limit_mem_dict[i] = f"{int(gb - 2.5)}GiB"
    
    # CPU 오프로딩을 위한 설정 (120GB 여유가 있다고 가정)
    limit_mem_dict["cpu"] = "120GiB" 
    return {"gpu_count": gpu_count, "total_gb": total_gb, "limit_mem_dict": limit_mem_dict}

def _load_pipeline_no_quant(offload_dir: str):
    """
    수정된 로딩 전략:
    1. 불필요한 Disk Offloading(Strategy 2) 제거
    2. 실패 시 명시적 메모리 해제 강화
    """
    from diffusers import QwenImageLayeredPipeline
    
    gpu_info = _detect_gpu_memory()
    print(f"[QwenWorker] Detected Total VRAM: {gpu_info['total_gb']:.2f} GB")

    strategies = [
        # 1. 기본 Balanced (가장 빠름, VRAM 충분할 때)
        {
            "name": "Vanilla Balanced (16-bit)",
            "kwargs": {
                "device_map": "balanced",
                "max_memory": None,
            }
        },
        # [변경] 중간에 있던 'No CPU' 전략은 제거했습니다. 
        # VRAM 부족 시 CPU를 건너뛰고 Disk로 가면 너무 느려져서 Timeout 발생함.
        
        # 2. CPU 오프로딩 (GPU 부족 시 CPU RAM 사용 - 안정적)
        {
            "name": "CPU-GPU Hybrid (16-bit, Balanced)",
            "kwargs": {
                "device_map": "balanced",
                # CPU 포함된 전체 딕셔너리 전달 -> 부족하면 CPU RAM 사용
                "max_memory": gpu_info["limit_mem_dict"],
            }
        },
        # 3. 최후의 수단: Sequential CPU Offload (매우 느리지만 메모리 가장 적게 씀)
        # 필요하다면 활성화, 여기서는 Balanced 전략만으로 충분할 것으로 예상
    ]

    last_error = None
    pipeline = None # 변수 미리 선언

    for strategy in strategies:
        # [중요] 이전 시도의 잔여물 확실히 제거
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
            # 할당 결과 출력
            for i in range(gpu_info["gpu_count"]):
                alloc = torch.cuda.memory_allocated(i) / (1024**3)
                print(f"  - GPU {i} Usage: {alloc:.2f} GB")
            
            return pipeline

        except Exception as e:
            print(f"[QwenWorker] Strategy {strategy['name']} failed: {e}")
            last_error = e
            
            # [중요] 실패 시 즉시 메모리 클린업 수행
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
    """워커 메인 프로세스"""
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