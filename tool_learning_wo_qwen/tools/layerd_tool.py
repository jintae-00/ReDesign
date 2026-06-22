# layerd_tool.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import torch, gc
import os

from config import DEVICE, WEIGHTS
from layerd import LayerD

# ---- 글로벌 싱글톤 ----
_LAYERD = None

def _get_layerd(matting_hf_card: str = "cyberagent/layerd-birefnet"):
    global _LAYERD
    if _LAYERD is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _LAYERD = LayerD(matting_hf_card=matting_hf_card, device=dev).to(dev)
    return _LAYERD

def _ensure_dirs(out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

def _save_rgba(pil_rgba: Image.Image, out_path: str) -> str:
    _ensure_dirs(out_path)
    pil_rgba.save(out_path, "PNG")
    return str(Path(out_path).resolve())

def _alpha_to_mask(pil_rgba: Image.Image) -> np.ndarray:
    a = np.array(pil_rgba.split()[-1], dtype=np.uint8)
    return (a > 0).astype(np.uint8)

def _save_mask_like(image_path: str, suffix: str, mask: np.ndarray) -> str:
    p = Path(image_path)
    out = str(p.with_name(p.stem + suffix))
    cv2.imwrite(out, (mask > 0).astype(np.uint8) * 255)
    return str(Path(out).resolve())

# -----------------------------
# 공통: bbox 포맷 정규화 (xyxy)
# -----------------------------
def _to_xyxy_any(box, W: int, H: int) -> Optional[List[int]]:
    """
    box를 다음 포맷 중 무엇이든 받아서 AABB(x1,y1,x2,y2)로 반환:
      - [x1,y1,x2,y2] (xyxy)
      - [x,y,w,h]     (xywh) → x2=x+w, y2=y+h (xyxy가 아니라고 판단될 때)
      - [x1,y1,x2,y2,x3,y3,x4,y4] (회전 사각형 4점)
      - [[x1,y1],[x2,y2],...]     (다각형)
    유효하지 않으면 None.
    최종적으로 이미지 경계로 clamp하고 정수형으로 반환.
    """
    if box is None:
        return None
    arr = np.asarray(box, dtype=float).reshape(-1)

    def _clamp_xyxy(x1,y1,x2,y2):
        x1 = max(0.0, min(W - 1.0, x1))
        y1 = max(0.0, min(H - 1.0, y1))
        x2 = max(1.0, min(W * 1.0, x2))
        y2 = max(1.0, min(H * 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [int(x1), int(y1), int(x2), int(y2)]

    if arr.size == 4:
        # 우선 xyxy 가정
        x1, y1, x2, y2 = arr.tolist()
        if x2 > x1 and y2 > y1:
            return _clamp_xyxy(x1, y1, x2, y2)
        # 아니라면 xywh로 해석
        x, y, w, h = arr.tolist()
        return _clamp_xyxy(x, y, x + max(0.0, w), y + max(0.0, h))

    if arr.size == 8:
        # 회전 사각형 4점
        xs = arr[0::2]; ys = arr[1::2]
        return _clamp_xyxy(xs.min(), ys.min(), xs.max(), ys.max())

    if arr.size >= 6 and arr.size % 2 == 0:
        # 임의 다각형 (N×2)
        pts = arr.reshape(-1, 2)
        xs, ys = pts[:, 0], pts[:, 1]
        return _clamp_xyxy(xs.min(), ys.min(), xs.max(), ys.max())

    return None

# -------------------------------------------------------
# 1) Front Layer Extraction (1-step)
# -------------------------------------------------------
@torch.no_grad()
def run_layerd_front(image_path: str) -> Dict[str, Any]:
    """
    base → 1-step 분해 후, 알파 없는 3채널 RGB(배경 0)와 마스크만 저장
    return: {"front_rgba": <path>, "front_mask": <path>}
    """
    layerd = _get_layerd()
    img = Image.open(image_path).convert("RGB")
    fg, _ = layerd._decompose_step(img)  # RGBA (straight alpha)

    p = Path(image_path)
    front_rgb_path = str(p.with_name(p.stem + "_front_rgba.png"))
    front_mask_path = str(p.with_name(p.stem + "_front_mask.png"))

    arr = np.array(fg)            # H,W,4 (RGBA)
    a   = arr[:, :, 3]            # 0~255
    TH  = 16

    keep = (a > TH)

    rgb = arr[:, :, :3].copy()
    rgb[~keep] = 0
    cv2.imwrite(front_rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    m = keep.astype(np.uint8) * 255
    cv2.imwrite(front_mask_path, m)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    del img, fg, arr, a, keep, rgb, m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {"front_rgb": front_rgb_path, "front_mask": front_mask_path}

# -------------------------------------------------------
# 2) Front Split by BBoxes (on front image)
# -------------------------------------------------------
def run_layerd_front_split(front_rgb_path: str,
                           mask_path: str,
                           boxes: List[Any],
                           det_ids: List[str],
                           *,
                           min_area_px: int = 64,
                           min_area_ratio: float = 5e-4) -> Dict[str, Any]:
    """
    전제:
      - front_rgb_path: 3채널 RGB 이미지(배경 0으로 클리어된 전경)
      - mask_path: GRAY 마스크(같은 크기)
    도구별 boxes 포맷(ocr/yolo/dino)을 모두 수용:
      - xyxy / xywh / 회전사각형(4점) / 다각형
    """
    body_bgr = cv2.imread(front_rgb_path, cv2.IMREAD_COLOR)
    if body_bgr is None:
        raise FileNotFoundError(front_rgb_path)
    H, W = body_bgr.shape[:2]

    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    alpha = (m > 0).astype(np.uint8)

    masks_by_id, extracts_by_id = {}, {}
    bboxes_by_id, bboxes_by_id_det = {}, {}
    masks_by_id_full = {}
    union = np.zeros((H, W), np.uint8)

    stem = Path(front_rgb_path).stem
    base_dir = Path(front_rgb_path).parent

    def _save_rgba_np(path: str, rgba: np.ndarray) -> str:
        Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(path)
        return str(Path(path).resolve())

    # --- (A) 매칭 요소: bbox 정규화 → tight bbox → tight RGBA extract
    for det_id, box in zip(det_ids or [], boxes or []):
        bb = _to_xyxy_any(box, W, H)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb

        crop_a = (alpha[y1:y2, x1:x2] > 0).astype(np.uint8)
        tys, txs = np.where(crop_a > 0)
        if not (txs.size and tys.size):
            # 비어있으면 skip
            continue

        tx1, ty1 = int(txs.min()), int(tys.min())
        tx2, ty2 = int(txs.max()) + 1, int(tys.max()) + 1

        rx1, ry1 = x1 + tx1, y1 + ty1
        rx2, ry2 = x1 + tx2, y1 + ty2

        crop_rgb = body_bgr[ry1:ry2, rx1:rx2, ::-1]   # BGR→RGB
        crop_a_t = (alpha[ry1:ry2, rx1:rx2] > 0).astype(np.uint8) * 255
        ext = np.zeros((ry2 - ry1, rx2 - rx1, 4), np.uint8)
        ext[:, :, :3] = crop_rgb
        ext[:, :, 3] = crop_a_t

        e_path = str(base_dir / f"{stem}_front_id-{det_id}_extract.png")
        _save_rgba_np(e_path, ext)

        # 디버그용 crop mask (tight)
        m_path = str(base_dir / f"{stem}_front_id-{det_id}_mask.png")
        Image.fromarray((crop_a[ty1:ty2, tx1:tx2] > 0).astype(np.uint8) * 255).save(m_path)

        # canvas-size RGBA full mask (A 채널만 채움)
        m_full = np.zeros((H, W, 4), np.uint8)
        m_full[ry1:ry2, rx1:rx2, 3] = crop_a_t
        full_path = str(base_dir / f"{stem}_front_id-{det_id}_mask_full.png")
        _save_rgba_np(full_path, m_full)

        union = np.maximum(union, (m_full[..., 3] > 0).astype(np.uint8))

        masks_by_id[det_id]       = str(Path(m_path).resolve())
        extracts_by_id[det_id]    = str(Path(e_path).resolve())
        bboxes_by_id_det[det_id]  = [x1, y1, x2, y2]           # 정규화된 원 검출 bbox
        bboxes_by_id[det_id]      = [rx1, ry1, rx2, ry2]       # tight bbox
        masks_by_id_full[det_id]  = str(Path(full_path).resolve())

        del crop_a, tys, txs, crop_rgb, crop_a_t, ext, m_full

    # --- (B) residual: 정규화된 det bbox 영역을 비운 후 남은 컴포넌트
    residual = (alpha > 0).astype(np.uint8)
    for did, bb in bboxes_by_id_det.items():
        x1, y1, x2, y2 = bb
        residual[y1:y2, x1:x2] = 0

    num_labels, labels = cv2.connectedComponents(residual, connectivity=8)
    min_area = max(min_area_px, int(min_area_ratio * H * W))
    residuals = []
    for lab in range(1, num_labels):
        ys, xs = np.where(labels == lab)
        if len(xs) < min_area:
            continue
        x1, x2 = int(xs.min()), int(xs.max() + 1)
        y1, y2 = int(ys.min()), int(ys.max() + 1)

        crop_a = (labels[y1:y2, x1:x2] == lab).astype(np.uint8)
        rtys, rtxs = np.where(crop_a > 0)
        if not (rtxs.size and rtys.size):
            continue
        rx1, ry1 = x1 + int(rtxs.min()), y1 + int(rtys.min())
        rx2, ry2 = x1 + int(rtxs.max()) + 1, y1 + int(rtys.max()) + 1

        crop_rgb = body_bgr[ry1:ry2, rx1:rx2, ::-1]  # RGB
        crop_a_t = crop_a[int(rtys.min()):int(rtys.max())+1, int(rtxs.min()):int(rtxs.max())+1].astype(np.uint8) * 255
        ext = np.zeros((ry2 - ry1, rx2 - rx1, 4), np.uint8)
        ext[:, :, :3] = crop_rgb
        ext[:, :, 3]  = crop_a_t

        r_m = str(base_dir / f"{stem}_front_residual_{lab}_mask.png")
        r_e = str(base_dir / f"{stem}_front_residual_{lab}_extract.png")
        Image.fromarray(crop_a_t).save(r_m)
        _save_rgba_np(r_e, ext)

        mf = np.zeros((H, W, 4), np.uint8)
        mf[ry1:ry2, rx1:rx2, 3] = crop_a_t
        union = np.maximum(union, (mf[..., 3] > 0).astype(np.uint8))

        residuals.append({
            "bbox": [rx1, ry1, rx2, ry2],
            "mask_path": str(Path(r_m).resolve()),
            "extract_path": str(Path(r_e).resolve())
        })

        del crop_a, rtys, rtxs, crop_rgb, crop_a_t, ext, mf

    union_path = str((Path(front_rgb_path).with_name(stem + "_front_union.png")).resolve())
    Image.fromarray((union > 0).astype(np.uint8) * 255).save(union_path)

    del body_bgr, m, alpha, union, residual, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "masks_by_id": masks_by_id,
        "extracts_by_id": extracts_by_id,
        "bboxes_by_id": bboxes_by_id,             # tight bbox(캔버스 좌표)
        "bboxes_by_id_det": bboxes_by_id_det,     # 정규화된 원 검출 bbox(xyxy)
        "masks_by_id_full": masks_by_id_full,     # canvas-size RGBA
        "mask_union_front": union_path,
        "residuals": residuals
    }

# -------------------------------------------------------
# 3) BBox Segmentation using Bi-Ref-Net (crop-wise)
# -------------------------------------------------------
@torch.no_grad()
def run_layerd_seg_bbox(image_path: str, boxes: List[Any], det_ids: List[str]) -> Dict[str, Any]:
    """
    각 bbox 크롭에 대해 Bi-Ref-Net으로 알파 추정 → full-size 마스크 합성.
    boxes 포맷은 자유롭게 들어와도 됨(ocr/yolo/dino), 내부에서 xyxy로 정규화.
    return: {"mask_union": <path>, "masks_by_id": {...}}
    """
    base = Image.open(image_path).convert("RGB")
    W, H = base.size
    layerd = _get_layerd()

    union = np.zeros((H, W), np.uint8)
    masks_by_id = {}

    for det_id, b in zip(det_ids or [], boxes or []):
        bb = _to_xyxy_any(b, W, H)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        if x2 <= x1 or y2 <= y1:
            continue

        crop = base.crop((x1, y1, x2, y2))
        # 1-step decompose on crop
        fg, _ = layerd._decompose_step(crop)
        a = np.array(fg.split()[-1], dtype=np.uint8)
        m = (a > 0).astype(np.uint8)

        # paste to full
        m_full = np.zeros((H, W), np.uint8)
        m_full[y1:y2, x1:x2] = np.maximum(m_full[y1:y2, x1:x2], m)
        union = np.maximum(union, m_full)

        p = Path(image_path)
        out = str(p.with_name(p.stem + f"_biref_bbox_{det_id}.png"))
        cv2.imwrite(out, m_full * 255)
        masks_by_id[det_id] = str(Path(out).resolve())

        del crop, fg, a, m, m_full

    union_path = _save_mask_like(image_path, "_biref_bbox_union.png", union)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    del base, union
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {"mask_union": union_path, "masks_by_id": masks_by_id}
