#!/usr/bin/env python3
"""Common utilities for editability evaluation.

This module is intentionally standalone and does not modify existing evaluation code.
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from skimage.metrics import structural_similarity as ssim_func
except Exception:  # pragma: no cover
    ssim_func = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import lpips as lpips_lib
except Exception:  # pragma: no cover
    lpips_lib = None

try:
    from transformers import ViTImageProcessor, ViTModel
except Exception:  # pragma: no cover
    ViTImageProcessor = None
    ViTModel = None

try:
    import torch.nn.functional as torch_f
except Exception:  # pragma: no cover
    torch_f = None

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

_LPIPS_MODEL = None
_LPIPS_DEVICE = "cpu"
_LPIPS_INIT_LOCK = Lock()
_DINO_MODEL = None
_DINO_PROCESSOR = None
_DINO_DEVICE = "cpu"
_DINO_INIT_LOCK = Lock()
T = TypeVar("T")
R = TypeVar("R")


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv-python (cv2) is required for this operation.")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def parse_exp_pairs(exp_pairs: Sequence[str]) -> List[Tuple[Path, Path, str]]:
    parsed: List[Tuple[Path, Path, str]] = []
    for pair in exp_pairs:
        parts = pair.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid exp-pair '{pair}'. Expected: agent_dir:qwen_dir:gt_subset_prefix"
            )
        parsed.append((Path(parts[0]), Path(parts[1]), parts[2]))
    return parsed


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    # Collapse internal whitespace.
    return " ".join(s.split())


def preprocess_text_content(s: Any) -> str:
    """Parsed/OCR text 공통 전처리: strip + 내부 공백 정규화."""
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def compute_cer(gt_text: str, pred_text: str) -> float:
    g = list(normalize_text(gt_text))
    p = list(normalize_text(pred_text))
    if not g:
        return 0.0 if not p else 1.0
    return float(_levenshtein(g, p) / max(1, len(g)))


def compute_wer(gt_text: str, pred_text: str) -> float:
    g = normalize_text(gt_text).split()
    p = normalize_text(pred_text).split()
    if not g:
        return 0.0 if not p else 1.0
    return float(_levenshtein(g, p) / max(1, len(g)))


def seed_rng(seed: int) -> random.Random:
    return random.Random(seed)


def sample_with_seed(items: Sequence[Any], max_count: Optional[int], seed: int) -> List[Any]:
    items = list(items)
    rng = seed_rng(seed)
    rng.shuffle(items)
    if max_count is None or max_count >= len(items):
        return items
    return items[:max_count]


def sample_with_seed_balanced_by_key(
    items: Sequence[T],
    key_fn: Callable[[T], Any],
    max_count: Optional[int],
    seed: int,
) -> List[T]:
    """Sample items with per-key round-robin balance (e.g., episode-balanced)."""
    xs = list(items)
    rng = seed_rng(seed)
    rng.shuffle(xs)
    if max_count is None or max_count >= len(xs):
        return xs

    buckets: Dict[Any, List[T]] = {}
    for x in xs:
        k = key_fn(x)
        buckets.setdefault(k, []).append(x)

    keys = list(buckets.keys())
    rng.shuffle(keys)

    out: List[T] = []
    while len(out) < max_count and keys:
        next_keys: List[Any] = []
        for k in keys:
            b = buckets.get(k, [])
            if not b:
                continue
            out.append(b.pop())
            if len(out) >= max_count:
                break
            if b:
                next_keys.append(k)
        keys = next_keys
    return out


def thread_map(
    items: Sequence[T],
    fn: Callable[[T], R],
    *,
    num_workers: int = 1,
    desc: Optional[str] = None,
    show_tqdm: bool = True,
) -> List[R]:
    """Threaded map with optional tqdm. Preserves input order."""
    xs = list(items)
    n = len(xs)
    if n == 0:
        return []

    use_tqdm = bool(show_tqdm and _tqdm is not None)
    if num_workers <= 1:
        out: List[R] = []
        it = enumerate(xs)
        if use_tqdm:
            it = _tqdm(it, total=n, desc=desc)
        for _, item in it:
            out.append(fn(item))
        return out

    out2: List[Optional[R]] = [None] * n
    pbar = _tqdm(total=n, desc=desc) if use_tqdm else None
    with ThreadPoolExecutor(max_workers=int(num_workers)) as ex:
        fut_to_idx = {ex.submit(fn, item): i for i, item in enumerate(xs)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            out2[idx] = fut.result()
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    missing = [i for i, x in enumerate(out2) if x is None]
    if missing:
        raise RuntimeError(f"thread_map failed to produce outputs for indices: {missing[:5]}")
    return [x for x in out2 if x is not None]


def ensure_canvas_rgba(img: Image.Image, canvas_size: Tuple[int, int]) -> np.ndarray:
    if img.size != canvas_size:
        img = img.resize(canvas_size, Image.LANCZOS)
    return np.array(img.convert("RGBA"), dtype=np.uint8)


def element_to_rgba(element: Dict[str, Any], canvas_size: Tuple[int, int]) -> np.ndarray:
    return ensure_canvas_rgba(element["image"], canvas_size)


def rgba_to_element_like(base_elem: Dict[str, Any], rgba: np.ndarray) -> Dict[str, Any]:
    new_elem = dict(base_elem)
    rgba = np.clip(rgba, 0, 255).astype(np.uint8)
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    ys, xs = np.where(alpha > 0)
    if ys.size == 0:
        bbox = [0, 0, 1, 1]
    else:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    new_elem["image"] = Image.fromarray(rgba, "RGBA")
    new_elem["mask"] = alpha
    new_elem["area"] = float((alpha > 0).sum())
    new_elem["bbox"] = bbox
    return new_elem


def binary_mask(mask: np.ndarray, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if shape is not None and mask.shape != shape:
        if cv2 is None:
            pil = Image.fromarray((mask.astype(np.float32) * 255).astype(np.uint8))
            pil = pil.resize((shape[1], shape[0]), Image.BILINEAR)
            mask = np.array(pil, dtype=np.float32) / 255.0
        else:
            mask = cv2.resize(mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return mask > 0


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    _require_cv2()
    k = radius * 2 + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def relative_dilation_radius(bbox: Sequence[int], ratio: float = 0.08, min_r: int = 2, max_r: int = 48) -> int:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    r = int(min(w, h) * ratio)
    return max(min_r, min(max_r, r))


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / (union + 1e-6)


def compute_l1(gt_rgb: np.ndarray, pred_rgb: np.ndarray, roi: np.ndarray) -> float:
    if roi.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(gt_rgb[roi] - pred_rgb[roi])))


def compute_l2(gt_rgb: np.ndarray, pred_rgb: np.ndarray, roi: np.ndarray) -> float:
    if roi.sum() == 0:
        return float("nan")
    diff = gt_rgb[roi] - pred_rgb[roi]
    return float(np.mean(diff * diff))


def compute_psnr_from_l2(mse: float) -> float:
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def compute_ssim(gt_rgb: np.ndarray, pred_rgb: np.ndarray, roi: np.ndarray) -> float:
    if ssim_func is None:
        return 0.0
    if roi.sum() == 0:
        return float("nan")

    ys, xs = np.where(roi)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    gt_crop = gt_rgb[y1:y2, x1:x2]
    pred_crop = pred_rgb[y1:y2, x1:x2]

    win_size = min(11, gt_crop.shape[0], gt_crop.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return 0.0

    return float(
        ssim_func(
            gt_crop,
            pred_crop,
            data_range=1.0,
            channel_axis=2,
            win_size=win_size,
            gaussian_weights=True,
            sigma=1.5,
        )
    )


def edge_sharpness(rgb: np.ndarray, roi: np.ndarray) -> float:
    if roi.sum() == 0:
        return float("nan")
    _require_cv2()
    gray = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    vals = lap[roi]
    if vals.size == 0:
        return float("nan")
    return float(np.var(vals))


def _get_lpips_model():
    global _LPIPS_MODEL, _LPIPS_DEVICE
    if lpips_lib is None or torch is None:
        return None
    if _LPIPS_MODEL is None:
        with _LPIPS_INIT_LOCK:
            if _LPIPS_MODEL is None:
                # AlexNet-based LPIPS is the most common baseline in image-editing papers.
                _LPIPS_MODEL = lpips_lib.LPIPS(net="alex")
                if torch.cuda.is_available():
                    _LPIPS_DEVICE = "cuda"
                    _LPIPS_MODEL = _LPIPS_MODEL.to(_LPIPS_DEVICE)
                _LPIPS_MODEL.eval()
    return _LPIPS_MODEL


def _get_dino_model():
    global _DINO_MODEL, _DINO_PROCESSOR, _DINO_DEVICE
    if ViTImageProcessor is None or ViTModel is None or torch is None or torch_f is None:
        return None, None, "cpu"
    if _DINO_MODEL is None or _DINO_PROCESSOR is None:
        with _DINO_INIT_LOCK:
            if _DINO_MODEL is None or _DINO_PROCESSOR is None:
                try:
                    _DINO_PROCESSOR = ViTImageProcessor.from_pretrained("facebook/dino-vits16")
                    # DINO checkpoints do not ship pooler weights.
                    _DINO_MODEL = ViTModel.from_pretrained(
                        "facebook/dino-vits16",
                        add_pooling_layer=False,
                    )
                    if torch.cuda.is_available():
                        _DINO_DEVICE = "cuda"
                        _DINO_MODEL = _DINO_MODEL.to(_DINO_DEVICE)
                    _DINO_MODEL.eval()
                except Exception:
                    _DINO_MODEL = None
                    _DINO_PROCESSOR = None
                    _DINO_DEVICE = "cpu"
    return _DINO_MODEL, _DINO_PROCESSOR, _DINO_DEVICE


def compute_lpips(gt_rgb: np.ndarray, pred_rgb: np.ndarray, roi: np.ndarray) -> float:
    model = _get_lpips_model()
    if model is None:
        return float("nan")
    if roi.sum() == 0:
        return float("nan")

    ys, xs = np.where(roi)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    gt_crop = gt_rgb[y1:y2, x1:x2]
    pr_crop = pred_rgb[y1:y2, x1:x2]
    if gt_crop.size == 0 or pr_crop.size == 0:
        return float("nan")

    # LPIPS backbones can become unstable on tiny crops; upsample to a safe minimum.
    h, w = gt_crop.shape[:2]
    min_hw = 32
    if h < min_hw or w < min_hw:
        tw = max(min_hw, w)
        th = max(min_hw, h)
        gt_crop = np.array(Image.fromarray((gt_crop * 255.0).astype(np.uint8), "RGB").resize((tw, th), Image.BILINEAR), dtype=np.float32) / 255.0
        pr_crop = np.array(Image.fromarray((pr_crop * 255.0).astype(np.uint8), "RGB").resize((tw, th), Image.BILINEAR), dtype=np.float32) / 255.0

    try:
        with torch.no_grad():
            gt_t = torch.from_numpy(gt_crop.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
            pr_t = torch.from_numpy(pr_crop.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
            if _LPIPS_DEVICE != "cpu":
                gt_t = gt_t.to(_LPIPS_DEVICE, non_blocking=True)
                pr_t = pr_t.to(_LPIPS_DEVICE, non_blocking=True)
            val = model(gt_t, pr_t)
            return float(val.mean().item())
    except Exception:
        return float("nan")


def compute_dino(gt_rgb: np.ndarray, pred_rgb: np.ndarray, roi: np.ndarray) -> float:
    model, processor, device = _get_dino_model()
    if model is None or processor is None:
        return float("nan")
    if roi.sum() == 0:
        return float("nan")

    ys, xs = np.where(roi)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    gt_crop = gt_rgb[y1:y2, x1:x2]
    pr_crop = pred_rgb[y1:y2, x1:x2]
    if gt_crop.size == 0 or pr_crop.size == 0:
        return float("nan")

    h, w = gt_crop.shape[:2]
    min_hw = 32
    if h < min_hw or w < min_hw:
        tw = max(min_hw, w)
        th = max(min_hw, h)
        gt_crop = np.array(
            Image.fromarray((gt_crop * 255.0).astype(np.uint8), "RGB").resize((tw, th), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        pr_crop = np.array(
            Image.fromarray((pr_crop * 255.0).astype(np.uint8), "RGB").resize((tw, th), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0

    try:
        p1 = Image.fromarray((np.clip(gt_crop, 0, 1) * 255.0).astype(np.uint8), "RGB")
        p2 = Image.fromarray((np.clip(pr_crop, 0, 1) * 255.0).astype(np.uint8), "RGB")
        in1 = processor(images=p1, return_tensors="pt")
        in2 = processor(images=p2, return_tensors="pt")
        with torch.no_grad():
            if device != "cpu":
                in1 = {k: v.to(device, non_blocking=True) for k, v in in1.items()}
                in2 = {k: v.to(device, non_blocking=True) for k, v in in2.items()}
            emb1 = model(**in1).last_hidden_state[:, 0, :]
            emb2 = model(**in2).last_hidden_state[:, 0, :]
            return float(torch_f.cosine_similarity(emb1, emb2).item())
    except Exception:
        return float("nan")


def compute_region_metrics(
    gt_rgba: np.ndarray,
    pred_rgba: np.ndarray,
    roi_mask: np.ndarray,
    include_iou: bool = True,
    include_edge_sharpness: bool = False,
    include_lpips: bool = False,
    include_dino: bool = False,
) -> Dict[str, float]:
    gt_f = gt_rgba.astype(np.float32) / 255.0
    pr_f = pred_rgba.astype(np.float32) / 255.0
    roi = roi_mask.astype(bool)

    gt_alpha = gt_f[..., 3:4]
    pr_alpha = pr_f[..., 3:4]
    gt_rgb_black = gt_f[..., :3] * gt_alpha
    pr_rgb_black = pr_f[..., :3] * pr_alpha
    gt_rgb_white = gt_rgb_black + (1.0 - gt_alpha)
    pr_rgb_white = pr_rgb_black + (1.0 - pr_alpha)

    metric_bg_mode = os.environ.get("EDITABILITY_METRIC_BG_MODE", "premultiplied").strip().lower()
    use_best_bw = metric_bg_mode in {"best_of_black_white", "best_bw", "min_bw", "black_white_best"}

    def _calc_metric_set(gt_rgb: np.ndarray, pr_rgb: np.ndarray) -> Dict[str, float]:
        l1 = compute_l1(gt_rgb, pr_rgb, roi)
        l2 = compute_l2(gt_rgb, pr_rgb, roi)
        psnr = compute_psnr_from_l2(l2) if not math.isnan(l2) else float("nan")
        ssim = compute_ssim(gt_rgb, pr_rgb, roi)
        out_local = {
            "l1": l1,
            "l2": l2,
            "psnr": psnr,
            "ssim": ssim,
        }
        if include_lpips:
            out_local["lpips"] = compute_lpips(gt_rgb, pr_rgb, roi)
        if include_dino:
            out_local["dino"] = compute_dino(gt_rgb, pr_rgb, roi)
        if include_edge_sharpness:
            out_local["edge_sharpness_gt"] = edge_sharpness(gt_rgb, roi)
            out_local["edge_sharpness_pred"] = edge_sharpness(pr_rgb, roi)
        return out_local

    def _pick_best(v_black: float, v_white: float, *, prefer: str) -> float:
        b_nan = bool(math.isnan(v_black))
        w_nan = bool(math.isnan(v_white))
        if b_nan and w_nan:
            return float("nan")
        if b_nan:
            return float(v_white)
        if w_nan:
            return float(v_black)
        if prefer == "higher":
            return float(max(v_black, v_white))
        return float(min(v_black, v_white))

    if use_best_bw:
        black_metrics = _calc_metric_set(gt_rgb_black, pr_rgb_black)
        white_metrics = _calc_metric_set(gt_rgb_white, pr_rgb_white)
        out: Dict[str, float] = {}
        for k in sorted(set(black_metrics.keys()) | set(white_metrics.keys())):
            vb = float(black_metrics.get(k, float("nan")))
            vw = float(white_metrics.get(k, float("nan")))
            if k in {"psnr", "ssim", "dino", "edge_sharpness_gt", "edge_sharpness_pred"}:
                out[k] = _pick_best(vb, vw, prefer="higher")
            else:
                out[k] = _pick_best(vb, vw, prefer="lower")
    else:
        # Legacy mode: use premultiplied RGB directly (equivalent to black background case).
        out = _calc_metric_set(gt_rgb_black, pr_rgb_black)

    if include_iou:
        gt_roi_alpha = (gt_rgba[..., 3] > 0) & roi
        pred_roi_alpha = (pred_rgba[..., 3] > 0) & roi
        out["iou"] = compute_iou(gt_roi_alpha, pred_roi_alpha)

    return out


def transform_rgba(rgba: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    _require_cv2()
    h, w = rgba.shape[:2]
    return cv2.warpAffine(
        rgba,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def translate_rgba(rgba: np.ndarray, dx: float, dy: float) -> np.ndarray:
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return transform_rgba(rgba, mat)


def rotate_rgba(rgba: np.ndarray, angle_deg: float, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    h, w = rgba.shape[:2]
    if center is None:
        center = (w * 0.5, h * 0.5)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return transform_rgba(rgba, mat)


def scale_rgba(rgba: np.ndarray, scale_x: float, scale_y: Optional[float] = None, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    h, w = rgba.shape[:2]
    if scale_y is None:
        scale_y = scale_x
    if center is None:
        center = (w * 0.5, h * 0.5)
    cx, cy = center
    mat = np.array(
        [[scale_x, 0.0, cx - scale_x * cx], [0.0, scale_y, cy - scale_y * cy]],
        dtype=np.float32,
    )
    return transform_rgba(rgba, mat)


def apply_opacity_rgba(rgba: np.ndarray, factor: float) -> np.ndarray:
    out = rgba.copy()
    alpha = out[..., 3].astype(np.float32)
    out[..., 3] = np.clip(alpha * factor, 0, 255).astype(np.uint8)
    return out


def recolor_rgba_hsv(rgba: np.ndarray, hue_shift_deg: float = 0.0, sat_mul: float = 1.0, val_mul: float = 1.0) -> np.ndarray:
    _require_cv2()
    out = rgba.copy()
    rgb = cv2.cvtColor(out[..., :3], cv2.COLOR_RGB2HSV).astype(np.float32)
    rgb[..., 0] = (rgb[..., 0] + (hue_shift_deg / 2.0)) % 180.0
    rgb[..., 1] = np.clip(rgb[..., 1] * sat_mul, 0, 255)
    rgb[..., 2] = np.clip(rgb[..., 2] * val_mul, 0, 255)
    out[..., :3] = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def edge_mask_from_alpha(alpha: np.ndarray, radius: int = 1) -> np.ndarray:
    _require_cv2()
    fg = alpha > 0
    if fg.sum() == 0:
        return fg
    k = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    dil = cv2.dilate(fg.astype(np.uint8), k, iterations=1) > 0
    ero = cv2.erode(fg.astype(np.uint8), k, iterations=1) > 0
    return dil ^ ero


def bbox_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return [0, 0, 1, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def concat_ocr_text(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""

    def key_fn(item: Dict[str, Any]) -> Tuple[float, float]:
        box = item.get("box", [])
        if isinstance(box, np.ndarray):
            box = box.tolist()
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            # Polygon format: [[x,y], ...]
            if isinstance(box[0], (list, tuple, np.ndarray)):
                xs: List[float] = []
                ys: List[float] = []
                for p in box:
                    try:
                        xs.append(float(p[0]))
                        ys.append(float(p[1]))
                    except Exception:
                        continue
                if xs and ys:
                    return (float(min(ys)), float(min(xs)))
            # Rect-like format: [x1,y1,x2,y2]
            try:
                x1, y1, _, _ = box
                return (float(y1), float(x1))
            except Exception:
                pass
        return (0.0, 0.0)

    items = sorted(items, key=key_fn)
    return " ".join([str(it.get("text", "")).strip() for it in items if str(it.get("text", "")).strip()])


_OCR_CLIENT = None
_OCR_CLIENTS: List[Any] = []
_OCR_CLIENT_QUEUE: Optional["queue.Queue[Any]"] = None
_OCR_INIT_LOCK = Lock()


def _ocr_pool_size() -> int:
    raw = os.environ.get("EDITABILITY_OCR_WORKERS", "1")
    try:
        v = int(raw)
    except Exception:
        v = 1
    # Keep this conservative; Paddle OCR model instances are heavy.
    return max(1, min(4, v))


def warmup_ocr_clients() -> None:
    """Initialize OCR client pool once per process."""
    global _OCR_CLIENT, _OCR_CLIENTS, _OCR_CLIENT_QUEUE
    with _OCR_INIT_LOCK:
        if _OCR_CLIENT_QUEUE is not None and _OCR_CLIENTS:
            return

        from modules.ocr.main import PaddleOCRClient

        pool = _ocr_pool_size()
        _OCR_CLIENTS = [PaddleOCRClient() for _ in range(pool)]
        _OCR_CLIENT = _OCR_CLIENTS[0]
        q: "queue.Queue[Any]" = queue.Queue(maxsize=pool)
        for c in _OCR_CLIENTS:
            q.put(c)
        _OCR_CLIENT_QUEUE = q


def run_ocr_on_rgba(rgba: np.ndarray) -> str:
    global _OCR_CLIENT_QUEUE
    try:
        if _OCR_CLIENT_QUEUE is None:
            warmup_ocr_clients()
        if _OCR_CLIENT_QUEUE is None:
            return ""
        pil_img = Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), "RGBA")
        client = _OCR_CLIENT_QUEUE.get()
        try:
            items = client.run_ocr(pil_img)
        finally:
            _OCR_CLIENT_QUEUE.put(client)
        return concat_ocr_text(items)
    except Exception:
        return ""


def estimate_text_contrast_background_rgb(rgba: np.ndarray) -> Tuple[int, int, int]:
    arr = np.array(rgba, copy=False)
    if arr.ndim != 3:
        return (255, 255, 255)
    if arr.shape[-1] == 3:
        arr = np.concatenate(
            [arr.astype(np.uint8), np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)],
            axis=2,
        )
    if arr.shape[-1] != 4:
        return (255, 255, 255)

    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3].astype(np.float32) / 255.0
    fg = alpha > 0.05
    if bool(fg.any()):
        w = alpha[fg]
        pix = rgb[fg]
        denom = float(w.sum()) + 1e-6
        mean_rgb = (pix * w[:, None]).sum(axis=0) / denom
    else:
        mean_rgb = rgb.reshape(-1, 3).mean(axis=0) if rgb.size > 0 else np.array([127.0, 127.0, 127.0], dtype=np.float32)

    inv_rgb = 255.0 - mean_rgb
    luma = float(0.2126 * mean_rgb[0] + 0.7152 * mean_rgb[1] + 0.0722 * mean_rgb[2]) / 255.0
    if 0.4 <= luma <= 0.6:
        bg = np.array([0.0, 0.0, 0.0], dtype=np.float32) if luma > 0.5 else np.array([255.0, 255.0, 255.0], dtype=np.float32)
    else:
        bg = inv_rgb
    bg_u8 = np.clip(np.round(bg), 0, 255).astype(np.uint8)
    return (int(bg_u8[0]), int(bg_u8[1]), int(bg_u8[2]))


def composite_rgba_on_background(
    rgba: np.ndarray,
    background_rgb: Tuple[int, int, int],
) -> np.ndarray:
    arr = np.array(rgba, copy=False)
    if arr.ndim != 3:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    if arr.shape[-1] == 3:
        arr = np.concatenate(
            [arr.astype(np.uint8), np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)],
            axis=2,
        )
    if arr.shape[-1] != 4:
        return np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)

    src = arr.astype(np.float32)
    alpha = src[..., 3:4] / 255.0
    bg = np.array(background_rgb, dtype=np.float32).reshape(1, 1, 3)
    out_rgb = src[..., :3] * alpha + bg * (1.0 - alpha)

    out = np.empty_like(arr, dtype=np.uint8)
    out[..., :3] = np.clip(np.round(out_rgb), 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def prepare_text_rgba_with_contrast_background(
    rgba: np.ndarray,
    background_rgb: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
    arr = np.array(rgba, copy=False)
    if arr.ndim != 3:
        return np.zeros((1, 1, 4), dtype=np.uint8), None
    if arr.shape[-1] == 3:
        arr = np.concatenate(
            [arr.astype(np.uint8), np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)],
            axis=2,
        )
    if arr.shape[-1] != 4:
        return np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8), None

    has_transparent = bool((arr[..., 3] < 250).any())
    bg = background_rgb
    if bg is None and has_transparent:
        bg = estimate_text_contrast_background_rgb(arr)
    if bg is None:
        return arr.astype(np.uint8), None
    return composite_rgba_on_background(arr, bg), tuple(int(v) for v in bg)


def run_ocr_on_rgba_contrast_background(
    rgba: np.ndarray,
    background_rgb: Optional[Tuple[int, int, int]] = None,
) -> str:
    prep, _ = prepare_text_rgba_with_contrast_background(rgba, background_rgb=background_rgb)
    return run_ocr_on_rgba(prep)


def run_ocr_on_rgba_black_white_best(
    rgba: np.ndarray,
    gt_text: str,
) -> Dict[str, Any]:
    """
    OCR on black/white solid backgrounds, then select the best text against GT.
    Selection priority: lower CER, then lower WER.
    """
    gt = preprocess_text_content(gt_text)
    backgrounds = [(0, 0, 0), (255, 255, 255)]
    candidates: List[Dict[str, Any]] = []
    for bg in backgrounds:
        composed = composite_rgba_on_background(rgba, bg)
        text_raw = run_ocr_on_rgba(composed)
        text = preprocess_text_content(text_raw)
        cer = compute_cer(gt, text)
        wer = compute_wer(gt, text)
        candidates.append(
            {
                "background_rgb": [int(bg[0]), int(bg[1]), int(bg[2])],
                "text": text,
                "cer": float(cer),
                "wer": float(wer),
            }
        )

    if not candidates:
        return {
            "best_text": "",
            "best_background_rgb": None,
            "best_cer": float("nan"),
            "best_wer": float("nan"),
            "min_cer": float("nan"),
            "min_wer": float("nan"),
            "candidates": [],
            "selection": "cer_then_wer",
        }

    ranked = sorted(
        candidates,
        key=lambda x: (
            float(x.get("cer", float("inf"))),
            float(x.get("wer", float("inf"))),
        ),
    )
    best = ranked[0]
    min_cer = min(float(x.get("cer", float("inf"))) for x in candidates)
    min_wer = min(float(x.get("wer", float("inf"))) for x in candidates)

    return {
        "best_text": str(best.get("text", "")),
        "best_background_rgb": best.get("background_rgb"),
        "best_cer": float(best.get("cer", float("nan"))),
        "best_wer": float(best.get("wer", float("nan"))),
        "min_cer": float(min_cer),
        "min_wer": float(min_wer),
        "candidates": candidates,
        "selection": "cer_then_wer",
    }


def render_prompt_overlay_rgba(
    image_rgba: np.ndarray,
    prompt: str,
    *,
    title: Optional[str] = None,
    wrap_width: int = 110,
) -> np.ndarray:
    """
    Add prompt text as a header block on top of an RGBA image.
    """
    arr = np.array(image_rgba, copy=False)
    if arr.ndim != 3:
        arr = np.zeros((1, 1, 4), dtype=np.uint8)
    if arr.shape[-1] == 3:
        arr = np.concatenate(
            [arr.astype(np.uint8), np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)],
            axis=2,
        )
    if arr.shape[-1] != 4:
        arr = np.zeros((max(1, arr.shape[0]), max(1, arr.shape[1]), 4), dtype=np.uint8)

    text = " ".join(str(prompt or "").split())
    lines: List[str] = []
    if title:
        lines.append(str(title))
    if text:
        lines.extend(textwrap.wrap(text, width=max(20, int(wrap_width))))
    else:
        lines.append("(no prompt)")

    base = Image.fromarray(arr.astype(np.uint8), "RGBA")
    font = ImageFont.load_default()
    pad = 8

    tmp = Image.new("RGBA", (1, 1), (255, 255, 255, 255))
    draw_tmp = ImageDraw.Draw(tmp)
    line_heights: List[int] = []
    text_width = 0
    for ln in lines:
        bb = draw_tmp.textbbox((0, 0), ln, font=font)
        w = max(1, int(bb[2] - bb[0]))
        h = max(1, int(bb[3] - bb[1]))
        text_width = max(text_width, w)
        line_heights.append(h)
    line_gap = 3
    text_h = pad * 2 + sum(line_heights) + max(0, len(lines) - 1) * line_gap
    out_w = max(base.width, text_width + pad * 2)
    out_h = base.height + text_h

    out = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(out)
    y = pad
    for idx, ln in enumerate(lines):
        fill = (0, 0, 0, 255)
        if idx == 0 and title:
            fill = (20, 20, 20, 255)
        draw.text((pad, y), ln, fill=fill, font=font)
        y += line_heights[idx] + line_gap
    out.paste(base, (0, text_h))
    return np.array(out, dtype=np.uint8)
