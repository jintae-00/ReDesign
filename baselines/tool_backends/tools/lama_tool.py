# baselines/tool_backends/tools/lama_tool.py
from PIL import Image
from pathlib import Path
import torch, gc

_lama = None

def _get_lama():
    global _lama
    if _lama is None:
        from modules.textremover.lama import LaMa
        from config import WEIGHTS
        _lama = LaMa(model_path=str(WEIGHTS / "big-lama.pt"))
    return _lama

def unload_lama():
    global _lama
    if _lama is not None:
        try:
            if hasattr(_lama, 'model'):
                _lama.model.cpu()
        except Exception:
            pass
        del _lama
        _lama = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@torch.no_grad()
def run_lama(image_path: str, mask_path: str) -> str:
    img  = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    out, _, _ = _get_lama().remove_text_by_mask(img, mask)
    out_path = str(Path(image_path).with_suffix("")) + "_inpaint.png"
    out.save(out_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return out_path
