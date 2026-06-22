# ReDesign/tools/objectclear_tool.py
"""
ObjectClear - runs independently on a dedicated GPU

[Strategy]
- Use a dedicated GPU (isolated from other tools).
- Prevent concurrent execution via an inter-process file lock.
- Bypass ToolGPUManager to eliminate memory conflicts at the source.
"""
import torch
import gc
import fcntl
from PIL import Image
from pathlib import Path
import numpy as np

from ..tool_gpu_config import OBJECTCLEAR_GPU


# Lock file path
_LOCK_FILE = Path(f"/tmp/objectclear_gpu_{OBJECTCLEAR_GPU}.lock")

# Model cache (reused within the process)
_cached_pipe = None
_cached_gpu_id = None


def _get_or_load_pipe(gpu_id: int, device: str):
    """Load the model (cached within the process)."""
    global _cached_pipe, _cached_gpu_id
    
    if _cached_pipe is not None and _cached_gpu_id == gpu_id:
        print(f"[ObjectClear] Using cached model on GPU {gpu_id}")
        return _cached_pipe
    
    # Clear the existing cache
    if _cached_pipe is not None:
        del _cached_pipe
        _cached_pipe = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
    
    print(f"[ObjectClear] Loading model on GPU {gpu_id}...")
    
    from modules.ObjectClear.objectclear.pipelines import ObjectClearPipeline
    from config import WEIGHTS
    
    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        "jixin0101/ObjectClear",
        torch_dtype=torch.float16,
        apply_attention_guided_fusion=False,
        cache_dir=str(WEIGHTS),
        variant="fp16",
        low_cpu_mem_usage=False,
    )
    
    try:
        pipe = pipe.to(device)
    except RuntimeError as e:
        if "meta tensor" in str(e).lower():
            print(f"[ObjectClear] Meta tensor detected, retrying with device_map...")
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
                device_map={"": device},
            )
        else:
            raise
    
    _cached_pipe = pipe
    _cached_gpu_id = gpu_id
    
    try:
        torch.cuda.synchronize(gpu_id)
    except RuntimeError as e:
        print(f"[ObjectClear] synchronize warning on GPU {gpu_id}: {e}")
    print(f"[ObjectClear] Model loaded on GPU {gpu_id}")
    
    return pipe


@torch.no_grad()
def run_objectclear(
    image_path: str,
    mask_path: str,
    caller_id: str = None,
) -> str:
    gpu_id = OBJECTCLEAR_GPU
    device = f"cuda:{gpu_id}"
    
    _LOCK_FILE.touch(exist_ok=True)
    
    with open(_LOCK_FILE, 'w') as lock_handle:
        print(f"[ObjectClear] Acquiring lock for GPU {gpu_id}...")
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        
        try:
            print(f"[ObjectClear] Lock acquired, starting inference...")
            
            torch.cuda.set_device(gpu_id)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            
            pipe = _get_or_load_pipe(gpu_id, device)
            
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing("auto")
            
            if hasattr(pipe, 'enable_vae_tiling'):
                pipe.enable_vae_tiling()
            
            # ----------------------------------------------------------------
            # Apply alpha via premultiplied alpha to remove noise
            # ----------------------------------------------------------------
            raw_img = Image.open(image_path).convert("RGBA")
            raw_arr = np.array(raw_img)

            # Alpha normalized to the 0-1 range
            alpha = raw_arr[:, :, 3].astype(np.float32) / 255.0

            # Multiply RGB by alpha (broadcasting)
            clean_rgb = raw_arr[:, :, :3].astype(np.float32) * alpha[..., None]

            # Use the variable name `image` so it connects with the logic below
            image = Image.fromarray(clean_rgb.astype(np.uint8), mode="RGB")

            mask = Image.open(mask_path).convert("L")

            # The `image` variable now exists, so no error occurs
            orig_w, orig_h = image.size
            print(f"[ObjectClear] Original size: {orig_w}x{orig_h}")
            
            # ----------------------------------------------------------------
            # Padding logic (edge-extend approach)
            # ----------------------------------------------------------------
            target_w = (orig_w + 7) // 8 * 8
            target_h = (orig_h + 7) // 8 * 8
            
            is_padded = False
            if target_w != orig_w or target_h != orig_h:
                is_padded = True
                print(f"[ObjectClear] Padding image to {target_w}x{target_h} using Edge Extend...")
                
                # 1. Create a new canvas
                new_image = Image.new("RGB", (target_w, target_h))
                new_image.paste(image, (0, 0))

                # 2. Fill the right edge (copy the last column of pixels)
                if target_w > orig_w:
                    edge_right = image.crop((orig_w - 1, 0, orig_w, orig_h))
                    edge_right_stretched = edge_right.resize((target_w - orig_w, orig_h))
                    new_image.paste(edge_right_stretched, (orig_w, 0))
                
                # 3. Fill the bottom edge (copy the last row of pixels, including the extended right region)
                if target_h > orig_h:
                    edge_bottom = new_image.crop((0, orig_h - 1, target_w, orig_h))
                    edge_bottom_stretched = edge_bottom.resize((target_w, target_h - orig_h))
                    new_image.paste(edge_bottom_stretched, (0, orig_h))
                
                image = new_image
                
                # 4. Pad the mask (the mask must be padded with black (0) to exclude it from the generated region)
                new_mask = Image.new("L", (target_w, target_h), 0)
                new_mask.paste(mask, (0, 0))
                mask = new_mask
            # ----------------------------------------------------------------

            generator = torch.Generator(device=device).manual_seed(42)
            
            result = pipe(
                prompt="remove the instance of object",
                image=image,
                mask_image=mask,
                generator=generator,
                num_inference_steps=20,
                strength=0.99,
                guidance_scale=2.5,
                height=target_h, width=target_w,
                return_attn_map=False,
            )
            
            out_img = result.images[0]
            
            del result
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            
            # ----------------------------------------------------------------
            # Crop logic
            # ----------------------------------------------------------------
            if is_padded:
                print(f"[ObjectClear] Cropping output back to {orig_w}x{orig_h}...")
                out_img = out_img.crop((0, 0, orig_w, orig_h))
            
            p = Path(image_path)
            out_path = str(p.with_name(p.stem + "_oc.png"))
            out_img.save(out_path)
            
            del out_img, image, mask
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            
            print(f"[ObjectClear] Completed: {out_path}")
            return out_path
            
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            print(f"[ObjectClear] Lock released for GPU {gpu_id}")