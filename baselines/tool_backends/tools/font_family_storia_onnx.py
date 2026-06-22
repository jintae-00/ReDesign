# baselines/tool_backends/tools/font_family_storia_onnx.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2, yaml
import onnxruntime as ort
import huggingface_hub

# ---- Global singleton cache ----
_STORIA = {
    "sess": None,
    "input_name": None,
    "classnames": None,
    "input_hw": None,  # (H, W) = (size, size)
}

def _load_storia(repo_id: str = "storia/font-classify-onnx"):
    if _STORIA["sess"] is not None:
        return
    cfg = huggingface_hub.hf_hub_download(repo_id=repo_id, filename="model_config.yaml")
    onx = huggingface_hub.hf_hub_download(repo_id=repo_id, filename="model.onnx")
    with open(cfg, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    sess = ort.InferenceSession(onx, providers=["CPUExecutionProvider"])
    _STORIA["sess"] = sess
    _STORIA["input_name"] = sess.get_inputs()[0].name
    _STORIA["classnames"] = list(conf["classnames"])
    size = int(conf["size"])
    _STORIA["input_hw"] = (size, size)

def _preprocess_square_center(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 255, dtype=np.uint8)
    oy, ox = (s - h)//2, (s - w)//2
    canvas[oy:oy+h, ox:ox+w] = rgb
    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2,0,1))[None, ...]
    return x

def _extract_from_alpha(extracted_rgba_path: str) -> Tuple[np.ndarray, Tuple[int,int]]:
    rgba = cv2.imread(extracted_rgba_path, cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(extracted_rgba_path)
    if rgba.ndim == 2:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGRA)
    if rgba.shape[2] == 3:
        # no alpha channel: use the whole image
        rgb = rgba[:, :, :3]
        mask = np.ones(rgb.shape[:2], np.uint8) * 255
    else:
        rgb = rgba[:, :, :3]
        mask = rgba[:, :, 3]
    sel = (mask > 0)
    if not np.any(sel):
        # empty: use the whole image
        return rgb, (rgb.shape[1], rgb.shape[0])
    ys, xs = np.where(sel)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    roi = rgb[y1:y2, x1:x2]
    return roi, (roi.shape[1], roi.shape[0])

def run_storia_onnx(
    image_path: str,
    mask_path: Optional[str] = None,
    extracted_image_path: Optional[str] = None,
    onnx_repo: str = "storia/font-classify-onnx"
) -> Dict[str, Any]:
    """
    returns: {"font_family": str or None, "debug": {"roi_w":int,"roi_h":int}}
    By default, the ROI is built primarily from the alpha channel of extracted_image_path (RGBA).
    If unavailable, the bounding AABB of mask_path is used. If neither is available, the whole image is used.
    """
    _load_storia(onnx_repo)
    size = _STORIA["input_hw"][0]

    # --- Determine ROI ---
    roi_rgb = None
    if extracted_image_path:
        try:
            roi_rgb, (Wd, Hd) = _extract_from_alpha(extracted_image_path)
        except Exception:
            roi_rgb = None

    if roi_rgb is None:
        base = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if base is None:
            return {"font_family": None}
        base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        if mask_path and Path(mask_path).exists():
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None and np.any(m > 0):
                ys, xs = np.where(m > 0)
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
                roi_rgb = base[y1:y2, x1:x2].copy()
            else:
                roi_rgb = base
        else:
            roi_rgb = base
        Hd, Wd = roi_rgb.shape[:2]

    # --- Preprocess & inference ---
    x = _preprocess_square_center(roi_rgb, size)
    sess = _STORIA["sess"]
    inp = { _STORIA["input_name"]: x }
    logits = sess.run(None, inp)[0][0]
    # softmax top-1
    k = int(np.argmax(logits))
    family = str(_STORIA["classnames"][k]) if 0 <= k < len(_STORIA["classnames"]) else None
    return {"font_family": family, "debug": {"roi_w": int(Wd), "roi_h": int(Hd)}}
