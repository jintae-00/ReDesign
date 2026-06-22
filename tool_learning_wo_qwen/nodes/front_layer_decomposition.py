from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path

import numpy as np
from PIL import Image

from ..state import GraphState
from ..reducers import (
    parse_doc_append, r_update_extract,
    r_save_artifact, r_update_segment,
    r_pack_state, r_abort_to_verify_sequence,
)
from ._common import _pop_current_action, _bump_next_or, _call_tool

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _make_canvas_mask_from_crop(mask_crop_path: Path,
                                bbox: List[int],
                                canvas_wh: List[int],
                                out_path: Path) -> Path:
    """
    crop-size GRAY/RGBA mask와 bbox, canvas(W,H)를 이용해 canvas-size RGBA mask 생성.
    - 입력 crop 마스크: (y1:y2, x1:x2) 영역의 알파/그레이 값을 그대로 사용(임계화 X)
    - 출력: 캔버스 크기의 RGBA PNG (RGB=0, A는 원래 세그 형태 보존)
    """
    W, H = int(canvas_wh[0]), int(canvas_wh[1])
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(W - 1, x1)); x2 = max(1, min(W, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(1, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        x2 = min(W, x1 + 1)
        y2 = min(H, y1 + 1)

    m = Image.open(str(mask_crop_path))
    if m.mode == "RGBA":
        a = np.array(m)[..., 3]
    else:
        a = np.array(m.convert("L"))
    w, h = x2 - x1, y2 - y1
    a_res = np.array(Image.fromarray(a).resize((w, h), Image.NEAREST))

    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    canvas[y1:y2, x1:x2, 3] = a_res  # ★ 임계화(>0)*255 하지 않음 → 원래 세그 모양 보존

    _ensure_parent(out_path)
    Image.fromarray(canvas, mode="RGBA").save(out_path)
    return out_path

def _get_canvas_wh_from_state_or_img(state: GraphState, fallback_img: str | None) -> List[int]:
    parse_doc = state.get("parse") or {}
    wh = (parse_doc.get("canvas") or {}).get("size_px")
    if isinstance(wh, (list, tuple)) and len(wh) == 2:
        return [int(wh[0]), int(wh[1])]
    target = fallback_img
    if not target:
        img_rt = ((state.get("_runtime") or {}).get("img") or {})
        for slot in ("front", "base", "detect", "seg"):
            p = (img_rt.get(slot) or {}).get("path")
            if p and Path(p).exists():
                target = p
                break
    if target and Path(target).exists():
        with Image.open(target) as im:
            W, H = im.size
        return [int(W), int(H)]
    return [1, 1]


def _build_detected_union_mask(
    masks_by_id_full: Dict[str, str],
    det_ids: List[str],
    canvas_wh: List[int],
    front_rgb: str,
) -> str | None:
    """
    LayerD_FrontSplit 결과 중,
      - masks_by_id_full[det_id] (canvas-size RGBA)
    만을 사용해서
      - '검출(det_id)된 영역만' union 한 mask 를 생성.
    residual 은 포함하지 않는다.
    """
    W, H = int(canvas_wh[0]), int(canvas_wh[1])
    union = np.zeros((H, W), dtype=np.uint8)
    has_any = False

    for did in det_ids:
        m_path = masks_by_id_full.get(did)
        if not m_path or not Path(m_path).exists():
            continue
        m_im = Image.open(str(m_path))
        if m_im.mode == "RGBA":
            a = np.array(m_im)[..., 3]
        else:
            a = np.array(m_im.convert("L"))
        if a.shape != (H, W):
            a = np.array(Image.fromarray(a).resize((W, H), Image.NEAREST))
        union = np.maximum(union, (a > 0).astype(np.uint8))
        has_any = True

    if not has_any:
        return None

    # LaMa / ObjectClear가 기대하는 형태: 0/255 단일 채널 마스크(L 모드)
    union_u8 = (union * 255).astype(np.uint8)  # 0 또는 255
    out = Path(front_rgb).with_name(Path(front_rgb).stem + "_front_detected_union.png")
    _ensure_parent(out)
    Image.fromarray(union_u8, mode="L").save(str(out))   # ★ 단일 채널로 저장

    return str(out.resolve())


def node(state: GraphState) -> Dict[str, Any]:
    act, upd0 = _pop_current_action(state)

    front = ((state.get("_runtime") or {}).get("front") or {})
    front_rgb = front.get("front_rgb_path") or front.get("front_rgb")
    front_mask_path = state.get("_mask_path") or front.get("front_mask_path") or front.get("front_mask")

    if not front_rgb or not Path(front_rgb).exists():
        return r_abort_to_verify_sequence(
                state,
                node="front_split",
                error_msg="front_split failed: NO front layer rgb image extracted by layerd_front node",
                details={"front_rgb" : front_rgb}
            )
    if not front_mask_path or not Path(front_mask_path).exists():
        return r_abort_to_verify_sequence(
                state,
                node="front_split",
                error_msg="front_split failed: NO front layer segmentation mask extracted by layerd_front node",
                details={"front_mask" : front_mask_path}
            )

    det = (state.get("_runtime") or {}).get("detect") or {}
    boxes   = det.get("boxes") or []
    det_ids = det.get("det_ids") or []
    texts   = det.get("texts") or None
    labels  = det.get("labels") or None
    if not boxes or not det_ids:
        return r_abort_to_verify_sequence(
                state,
                node="front_split",
                error_msg="front_split failed: NO object detected bboxes had been generated by preceding detection node.",
                details={"boxes" : boxes, "det_ids" : det_ids}
            )

    out = _call_tool("LayerD_FrontSplit", {
        "front_rgb_path": front_rgb,
        "mask_path": front_mask_path,
        "boxes": boxes,
        "det_ids": det_ids,
    }, state)

    masks_by_id       = (out or {}).get("masks_by_id") or {}
    extracts_by_id    = (out or {}).get("extracts_by_id") or {}
    bboxes_by_id      = (out or {}).get("bboxes_by_id") or {}
    bboxes_by_id_det  = (out or {}).get("bboxes_by_id_det") or {}
    masks_by_id_full  = (out or {}).get("masks_by_id_full") or {}
    mask_union_front  = (out or {}).get("mask_union_front")
    residuals         = (out or {}).get("residuals") or []

    canvas_wh = _get_canvas_wh_from_state_or_img(state, front_rgb)
    W, H = canvas_wh

    rows = []
    obj_ids, text_ids, added_ids = [], [], []

    # 모든 patch 를 모아두는 pieces (front_layer_extraction_layerd 와 동일 패턴)
    pieces: List[Dict[str, Any]] = [upd0]

    # (A) 매칭된 det_id 요소
    for did in det_ids:
        if did not in masks_by_id:
            continue
        i = det_ids.index(did)
        etype = "text" if (texts and i < len(texts) and texts[i] is not None) else "object"

        crop_mask_path = masks_by_id.get(did)                          # crop-size (tight)
        full_mask_path = masks_by_id_full.get(did) or crop_mask_path   # prefer full

        row = {
            "id": did,
            "type": etype,
            "bbox": bboxes_by_id.get(did) or (boxes[i] if i < len(boxes) else None),  # tight bbox
            "bbox_det": bboxes_by_id_det.get(did) or (boxes[i] if i < len(boxes) else None),  # 원검출 bbox
            "mask_uri": crop_mask_path,
            "mask_canvas_uri": full_mask_path,     # 이미 세그 형태를 캔버스에 보존
            "mask_is_canvas": True,
            "extracted_image_uri": extracts_by_id.get(did),
            "parsing_step": int(state.get("seq_id")),
            "coord_id": "canvas",
            "source_box": det.get("source") or "detect",
            "front_cycle": int(state.get("sequence_counter", 0)),
        }
        if etype == "text" and texts:
            row["content"] = texts[i]
            text_ids.append(did)
        else:
            if labels and i < len(labels) and labels[i] is not None:
                row["label"] = labels[i]
            obj_ids.append(did)

        rows.append(row)
        added_ids.append(did)

    '''
    # (B) residual 컴포넌트
    seq_idx = state.get("sequence_counter", 0)
    front_dir = Path(front_rgb).parent
    front_stem = Path(front_rgb).stem

    for k, r in enumerate(residuals):
        rid = f"r-{seq_idx:03d}-{k:02d}"
        bbox = r.get("bbox")
        crop_mask = r.get("mask_path")
        extract   = r.get("extract_path")

        full_path = front_dir / f"{front_stem}_front_residual_{k}_mask_full.png"
        try:
            _make_canvas_mask_from_crop(Path(crop_mask), bbox, [W, H], full_path)
            full_mask_uri = str(full_path.resolve())
        except Exception:
            full_mask_uri = crop_mask

        row = {
            "id": rid,
            "type": "object",
            "bbox": bbox,
            "mask_uri": crop_mask,
            "mask_canvas_uri": full_mask_uri,  # 세그 형태 보존
            "mask_is_canvas": True,
            "extracted_image_uri": extract,
            "parsing_step": int(state.get("seq_id")),
            "coord_id": "canvas",
            "source_box": "residual",
            "front_cycle": int(seq_idx),
            "label": "residual"
        }
        rows.append(row)
        obj_ids.append(rid)
        added_ids.append(rid)
    '''

    # (1) 검출된(det_id) 요소만 parse / extract 에 반영
    pieces.append(parse_doc_append(rows, state))
    pieces.append(r_update_extract(added_ids, text_ids, obj_ids, state))


    # (2) 검출된 영역만 union 한 front mask 생성 → artifact 저장 + segment 상태 업데이트
    det_union_path = _build_detected_union_mask(masks_by_id_full, added_ids, canvas_wh, front_rgb)
    art_patch, art_saved = r_save_artifact(det_union_path, "front_layer_decomposition", "segment", state)
    pieces.append(art_patch)        

    seg_patch = r_update_segment(
        tool="layerd_front_split",
        mask_path=art_saved,
        state=state,
        masks_by_id=masks_by_id_full,
        base_img=front_rgb,
        img_slot=None,              # _runtime.img 슬롯은 건드리지 않음
        coord_id="canvas",
        W=W,
        H=H,
        to_base={"scale_x": 1.0, "scale_y": 1.0, "dx": 0, "dy": 0},
    )
    pieces.append(seg_patch)


    pieces.append(_bump_next_or(state, "tool_sequence_plan"))

    return r_pack_state(state, *pieces)