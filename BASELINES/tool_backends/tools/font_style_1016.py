# BASELINES/tool_backends/tools/font_style_1016.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2, math, os

# Qt for metrics
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontMetricsF
from PySide6.QtCore import Qt

# lazy-init Qt
def _ensure_qapp():
    QApplication.instance() or QApplication(["fontstyle-1016", "-platform", os.environ.get("QT_QPA_PLATFORM","offscreen")])

def _median_color(roi_rgb: np.ndarray, alpha: Optional[np.ndarray]) -> Tuple[list, str]:
    if alpha is not None:
        sel = roi_rgb[alpha > 0]
        if sel.size == 0:
            sel = roi_rgb.reshape(-1, 3)
    else:
        sel = roi_rgb.reshape(-1, 3)
    med = np.median(sel, axis=0)
    rgb = np.clip(np.round(med), 0, 255).astype(np.uint8)
    hexstr = "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return rgb.tolist(), hexstr

def _estimate_italic_angle(roi_rgb: np.ndarray, alpha: Optional[np.ndarray]) -> float:
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    if alpha is not None:
        mag = mag * (alpha > 0)
    weights = (mag / (mag.max() + 1e-6)).astype(np.float32)
    angle_x = np.degrees(np.arctan2(gy, gx))
    dev = (angle_x - 90.0 + 180.0) % 180.0 - 90.0
    ang = float((dev * weights).sum() / (weights.sum() + 1e-6))
    return float(np.round(ang, 1))

def _measure_text_size(content: str, family: str, italic: bool, px: int):
    f = QFont(family)
    f.setPixelSize(int(px))
    f.setItalic(bool(italic))
    fm = QFontMetricsF(f)
    tw = fm.horizontalAdvance(content)
    th = fm.height()
    pad = max(1.0, 0.05 * max(tw, th))
    return int(math.ceil(tw + pad)), int(math.ceil(th + pad)), fm

def _fit_px_singleline(content: str, family: str, italic: bool, target_w: int, target_h: int, hint_px: int) -> int:
    lo, hi = 4, max(8, int(hint_px * 1.6))
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        tw, th, _ = _measure_text_size(content, family, italic, mid)
        if tw <= target_w and th <= target_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return max(1, best)

def _roi_from_inputs(image_path: str, mask_path: Optional[str], extracted_image_path: Optional[str]):
    if extracted_image_path:
        img = cv2.imread(extracted_image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(extracted_image_path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        if img.shape[2] == 3:
            rgb = img
            alpha = None
        else:
            rgb = img[:, :, :3]
            alpha = img[:, :, 3]
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), alpha

    base = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(image_path)
    base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    if mask_path and Path(mask_path).exists():
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None and np.any(m > 0):
            ys, xs = np.where(m > 0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
            return base[y1:y2, x1:x2].copy(), (m[y1:y2, x1:x2] if m is not None else None)
    return base, None

def run_fontstyle_1016(
    image_path: str,
    mask_path: Optional[str] = None,
    extracted_image_path: Optional[str] = None,
    content: str = "",
    font_family: str = "",
    fit_hint_px: Optional[int] = None
) -> Dict[str, Any]:
    """
    returns: {"font_render": {...}}
    """
    _ensure_qapp()
    roi_rgb, alpha = _roi_from_inputs(image_path, mask_path, extracted_image_path)
    H, W = roi_rgb.shape[:2]

    rgb, hexstr = _median_color(roi_rgb, alpha)
    angle = _estimate_italic_angle(roi_rgb, alpha)
    italic_flag = bool(abs(angle) >= 15.0)
    hint_px = int(fit_hint_px or max(1, int(0.95 * H)))
    fit_px = _fit_px_singleline(content or "", font_family or "", italic_flag, W, H, hint_px)

    fr = {
        "size_px": float(fit_px),
        "italic": {"flag": bool(italic_flag), "angle_deg": float(angle)},
        "bold": False,  # conservatively False (can be improved with rule-based logic if desired)
        "color": {"rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])], "hex": hexstr}
    }
    return {"font_render": fr}
