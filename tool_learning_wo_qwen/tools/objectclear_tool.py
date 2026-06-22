# src/REDESIGN/tools/objectclear_tool.py
from config import WEIGHTS
from modules.ObjectClear.objectclear.pipelines import ObjectClearPipeline
from PIL import Image, ImageOps
import torch, gc
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
    "jixin0101/ObjectClear",
    torch_dtype=torch.float16,
    apply_attention_guided_fusion=True,
    cache_dir=str(WEIGHTS),
    variant="fp16"
)
pipe.to(device)

@torch.no_grad()
def run_objectclear(image_path: str, mask_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    mask  = Image.open(mask_path).convert("L")
    w, h = image.size
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    if pad_h or pad_w:
        image  = ImageOps.expand(image,  border=(0,0,pad_w,pad_h), fill=0)
        mask   = ImageOps.expand(mask,   border=(0,0,pad_w,pad_h), fill=0)
        h += pad_h; w += pad_w
    result = pipe(
        prompt="remove the instance of object",
        image=image,
        mask_image=mask,
        generator=generator,
        num_inference_steps=20,
        strength=0.99,
        guidance_scale=2.5,
        height=h, width=w,
        return_attn_map=False,
    )
    out_img = result.images[0]
    if pad_h or pad_w:
        crop_box = (0, 0, out_img.width - pad_w, out_img.height - pad_h)
        out_img = out_img.crop(crop_box)
    p = Path(image_path)
    out_path = str(p.with_name(p.stem + "_oc.png"))
    out_img.save(out_path)

    del result, out_img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return out_path
