# src/langgraph/tools/hisam_tool.py
import cv2
import numpy as np
import torch, gc
from pathlib import Path
from typing import List, Dict, Optional

_hisam = None

def _get_hisam():
    global _hisam
    if _hisam is None:
        from modules.hisam.inference import HiSam_Inference
        from config import WEIGHTS
        _hisam = HiSam_Inference(check_point_dir=WEIGHTS, model_path="sam_tss_h_textseg.pth")
    return _hisam

def unload_hisam():
    global _hisam
    if _hisam is not None:
        try:
            # HiSam_Inference → .hisam_service → .hisam (nn.Module) + .predictor
            svc = _hisam.hisam_service
            if hasattr(svc, 'hisam'):
                svc.hisam.cpu()
            if hasattr(svc, 'predictor'):
                del svc.predictor
        except Exception:
            pass
        del _hisam
        _hisam = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _clip_box_with_pad(x1, y1, x2, y2, W, H, pad_px: int):
    x1 = max(0, int(x1) - pad_px)
    y1 = max(0, int(y1) - pad_px)
    x2 = min(W, int(x2) + pad_px)
    y2 = min(H, int(y2) + pad_px)
    # 최소 1픽셀 이상 보장
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2

@torch.no_grad()
def run_hisam_union(
    image_path: str,
    boxes: List[List[int]],
    det_ids: List[str],
    vis_dir: Optional[Path] = None,
    step: int = 0
) -> Dict[str, str]:
    """
    Hi-SAM을 이용해 텍스트 세그멘테이션을 수행.
    각 det_id 마다 별도의 마스크 파일을 저장하고, dilate까지 적용한 union 마스크 경로를 반환합니다.

    Args:
        image_path: 원본 이미지 경로
        boxes: AABB 리스트 [[x1,y1,x2,y2], ...]
        det_ids: 각 박스에 대응하는 detection ID 리스트 (길이 boxes와 동일한 것을 권장)
        vis_dir: (선택) 저장 디렉토리 힌트. 주어지면 union 마스크 이름에 step 프리픽스를 붙입니다.
        step: (선택) 스텝 인덱스 (파일명에 사용)

    Returns:
        {
            "mask_union": <union_mask_path>,
            "masks_by_id": { det_id: <mask_path>, ... }
        }
    """
    # --- 입력 이미지 로드 (RGB) ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = img[:, :, ::-1]  # BGR -> RGB
    H, W, _ = img.shape

    # --- pad 픽셀 계산 (기존 로직과 동일) ---
    diag = (H ** 2 + W ** 2) ** 0.5
    pad_px = int(0.01 * diag)

    # --- per-id raw mask 및 union 초기화 ---
    per_id_raw = {det_id: np.zeros((H, W), np.uint8) for det_id in det_ids}
    union = np.zeros((H, W), np.uint8)

    # --- 박스별 Hi-SAM 실행 & 마스크 합성 ---
    for det_id, b in zip(det_ids or [], boxes or []):
        if b is None or len(b) != 4:
            continue
        x1, y1, x2, y2 = b
        x1, y1, x2, y2 = _clip_box_with_pad(x1, y1, x2, y2, W, H, pad_px)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Hi-SAM 추론 (기존 코드와 동일)
        try:
            _, m_pil = _get_hisam()(input_img=crop)
        except Exception as e:
            # 한 박스 실패해도 전체는 계속
            print(f"[HiSAM] inference failed for det_id={det_id}: {e}")
            del crop
            # predictor 상태 초기화 (실패한 추론의 partial features 제거)
            try:
                svc = _get_hisam().hisam_service
                if hasattr(svc, 'predictor') and hasattr(svc.predictor, 'reset_image'):
                    svc.predictor.reset_image()
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        m = (np.array(m_pil) > 127).astype(np.uint8)  # 0/1

        # per-id raw 마스크 및 union에 paste
        # (crop 범위에만 값 존재)
        per_id_raw[det_id][y1:y2, x1:x2] = np.maximum(per_id_raw[det_id][y1:y2, x1:x2], m)
        union[y1:y2, x1:x2] = np.maximum(union[y1:y2, x1:x2], m)

        del crop, m_pil, m

        # box별 추론 후 predictor 내부 image features 즉시 해제
        # → 다음 박스 처리 전 GPU에 캐싱된 feature 텐서 제거
        try:
            svc = _get_hisam().hisam_service
            if hasattr(svc, 'predictor') and hasattr(svc.predictor, 'reset_image'):
                svc.predictor.reset_image()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- dilate: 기존 로직을 union과 per-id 모두에 동일하게 적용 ---
    # union dilate
    union_dil = np.zeros_like(union)
    for b in boxes or []:
        if b is None or len(b) != 4:
            continue
        x1, y1, x2, y2 = map(int, b)
        h = y2 - y1
        if h <= 0 or x2 <= x1:
            continue
        dilate_px = int(0.01 * h)
        if dilate_px < 1:
            dilate_px = 1
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        crop = union[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        dilated_crop = cv2.dilate(crop, kernel, iterations=1)
        union_dil[y1:y2, x1:x2] = np.maximum(union_dil[y1:y2, x1:x2], dilated_crop)

        del crop, dilated_crop, kernel

    # per-id dilate + 저장
    base = Path(image_path)
    masks_by_id: Dict[str, str] = {}
    for det_id, b in zip(det_ids or [], boxes or []):
        if b is None or len(b) != 4:
            continue
        x1, y1, x2, y2 = map(int, b)
        h = y2 - y1
        if h <= 0 or x2 <= x1:
            continue
        dilate_px = int(0.01 * h)
        if dilate_px < 1:
            dilate_px = 1
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)

        m_raw = per_id_raw.get(det_id, None)
        if m_raw is None:
            continue
        crop = m_raw[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        dilated_crop = cv2.dilate(crop, kernel, iterations=1)

        m_full = np.zeros((H, W), np.uint8)
        m_full[y1:y2, x1:x2] = dilated_crop

        m_path = str(base.with_name(f"{base.stem}_hisam_{det_id}.png"))
        cv2.imwrite(m_path, m_full * 255)
        masks_by_id[det_id] = m_path

        del m_raw, crop, dilated_crop, m_full, kernel

    # --- union 마스크 저장 (파일명 정책: vis_dir가 있으면 step prefix) ---
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        union_path = str(vis_dir / f"{step:03d}_HiSAM_union.png")
    else:
        union_path = str(base.with_name(f"{base.stem}_hisam_union.png"))
    cv2.imwrite(union_path, union_dil * 255)

    del img, per_id_raw, union, union_dil

    # Fully unload HiSAM from GPU after each inference so other tools
    # (LaMa, SAM2, etc.) have room on the same GPU.
    # The model reloads on the next call via _get_hisam().
    unload_hisam()

    return {"mask_union": union_path, "masks_by_id": masks_by_id}
