from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
from PIL import Image

from ..state import GraphState
from ..reducers import (
    r_update_extract,
    r_pack_state, r_abort_to_verify_sequence
)
from ..utils import to_json_safe, quad_to_aabb
from ._common import _pop_current_action, _bump_next_or

def _write_parse(state: GraphState, doc: Dict[str, Any]):
    state["parse"] = doc
    with open(state["parse_path"], "w", encoding="utf-8") as f:
        import json
        json.dump(to_json_safe(doc), f, ensure_ascii=False, indent=2)

def _resolve_base_img(state: GraphState) -> Optional[Path]:
    """반드시 base 원본 이미지만 사용"""
    img_rt = ((state.get("_runtime") or {}).get("img") or {})
    base = (img_rt.get("base") or {}).get("path")
    if base and Path(base).exists():
        return Path(base).resolve()
    return None

def _quad_to_aabb_safe(box) -> List[int]:
    try:
        if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            x1, y1, x2, y2 = box
            return [int(x1), int(y1), int(x2), int(y2)]
        return quad_to_aabb(box)
    except Exception:
        return [0, 0, 1, 1]

def node(state: GraphState) -> Dict[str, Any]:
    act, upd0 = _pop_current_action(state)

    pieces: List[Dict[str, Any]] = []
    def push(x: Optional[Dict[str, Any]]):
        if x:
            pieces.append(x)

    parse_doc = state.get("parse") or {}
    elements: List[Dict[str, Any]] = list(parse_doc.get("elements") or [])
    out_dir = Path(state["out_dir"])

    # === 캔버스 크기 ===
    canvas_wh = (parse_doc.get("canvas") or {}).get("size_px") or None
    base_img_abs = _resolve_base_img(state)
    if (not canvas_wh or len(canvas_wh) != 2) and base_img_abs is None:
        return r_abort_to_verify_sequence(
            state, node="extract_elements",
            error_msg="extract_elements failed: no canvas size and no base image",
            details={"canvas_wh": canvas_wh}
        )
    if not canvas_wh or len(canvas_wh) != 2:
        with Image.open(base_img_abs) as im:
            canvas_wh = list(im.size)  # [W,H]
    W, H = int(canvas_wh[0]), int(canvas_wh[1])

    if base_img_abs is None:
        return r_abort_to_verify_sequence(
            state, node="extract_elements",
            error_msg="extract_elements failed: base image not found in _runtime.img.base",
            details={}
        )

    # === _runtime 입력 ===
    det = (state.get("_runtime") or {}).get("detect") or {}
    seg = (state.get("_runtime") or {}).get("segment") or {}


    det_tool = det.get("tool")
    boxes   = det.get("boxes") or []
    det_ids = det.get("det_ids") or []
    texts   = det.get("texts")
    labels  = det.get("labels")

    masks_by_id: Dict[str, str] = (seg.get("masks_by_id") or {})

    assets_dir   = out_dir / "assets"
    extracts_dir = assets_dir / "extracts"
    extracts_dir.mkdir(parents=True, exist_ok=True)

    if not masks_by_id:
        return r_abort_to_verify_sequence(
            state, node="extract_elements",
            error_msg="extract_elements failed: masks_by_id is empty.",
            details={}
        )

    # det_id → 메타
    lookup: Dict[str, Dict[str, Any]] = {}
    for i, did in enumerate(det_ids):
        lookup[did] = {
            "idx": i,
            "box": boxes[i] if i < len(boxes) else None,
            "text": texts[i] if (texts and i < len(texts)) else None,
            "label": labels[i] if (labels and i < len(labels)) else None,
        }

    # 원본 로드 1회
    base_rgba = np.array(Image.open(str(base_img_abs)).convert("RGBA"))  # H,W,4

    added_ids: List[str] = []
    text_ids: List[str] = []
    obj_ids:  List[str] = []

    def _upsert(el: Dict[str, Any]):
        nonlocal elements
        eid = el.get("id")
        for j, e in enumerate(elements):
            if e.get("id") == eid:
                elements[j] = {**e, **el}
                return
        elements.append(el)

    # === per det_id ===
    for det_id, mask_path in masks_by_id.items():
        meta = lookup.get(det_id, {})
        box  = meta.get("box")
        txt  = meta.get("text")
        lab  = meta.get("label")
        etype = "text" if (txt is not None) else "object"

        # 1) 검출 bbox(AABB, 캔버스 좌표)
        aabb_det = _quad_to_aabb_safe(box)
        x1, y1, x2, y2 = map(int, aabb_det)
        x1 = max(0, min(W-1, x1)); x2 = max(1, min(W, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(1, min(H, y2))

        # 2) 마스크 로드
        m_im = Image.open(str(mask_path))
        if m_im.mode == "RGBA":
            a_in = np.array(m_im)[..., 3].astype(np.uint8)
        else:
            a_in = np.array(m_im.convert("L")).astype(np.uint8)
        mw, mh = m_im.size  # (W,H) 주의

        # 3) 캔버스 사이즈 RGBA 마스크 구성(알파만 채움, 임계/보정 없음)
        #    - 캔버스 사이즈면 그대로 사용
        #    - 크롭 사이즈면 bbox_det 크기와 정확히 같아야 하며, 아니면 즉시 예외
        a_full = np.zeros((H, W), dtype=np.uint8)
        a_full = a_in
        # if (mw, mh) == (W, H):
        #     a_full = a_in
        # else:
        #     # bbox_det 크기와 정확히 동일해야 함
        #     if (mw, mh) != (x2 - x1, y2 - y1):
        #         raise RuntimeError(
        #             f"[extract] det_id={det_id} mask size {mw}x{mh} != bbox_det {x2-x1}x{y2-y1}"
                # )
            # a_full[y1:y2, x1:x2] = a_in

        # 4) 타이트 박스 계산 (A>0)
        ys, xs = np.where(a_full > 0)
        if xs.size == 0 or ys.size == 0:
            # 비어있으면 요소 스킵
            continue
        tx1, ty1 = int(xs.min()), int(ys.min())
        tx2, ty2 = int(xs.max()) + 1, int(ys.max()) + 1
        bbox_refined = [tx1, ty1, tx2, ty2]

        # 5) 타이트 RGBA 추출 (항상 base에서)
        crop_rgb = base_rgba[ty1:ty2, tx1:tx2, :3]
        crop_a   = (a_full[ty1:ty2, tx1:tx2] > 0).astype(np.uint8) * 255
        ext = np.zeros((ty2 - ty1, tx2 - tx1, 4), np.uint8)
        ext[:, :, :3] = crop_rgb
        ext[:, :, 3]  = crop_a

        # 6) 저장
        ext_path           = extracts_dir / f"{det_id}.png"
        mask_canvas_path   = extracts_dir / f"{det_id}_mask_canvas.png"

        # 캔버스 사이즈 RGBA(알파만 채움)
        mask_canvas_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        mask_canvas_rgba[..., 3] = a_full
        Image.fromarray(mask_canvas_rgba, mode="RGBA").save(str(mask_canvas_path))
        Image.fromarray(ext,               mode="RGBA").save(str(ext_path))

        # 7) element upsert
        el: Dict[str, Any] = {
            "id": det_id,
            "type": etype,
            "bbox": bbox_refined,                         # tight bbox(캔버스 좌표)
            "bbox_det": [x1, y1, x2, y2],                 # 원 검출 bbox
            "mask_uri": str(Path(mask_path).resolve()),   # 원본 제공 마스크(크기는 다양)
            "mask_canvas_uri": str(mask_canvas_path.resolve()),  # 캔버스 사이즈 RGBA
            "mask_is_canvas": True,
            "extracted_image_uri": str(ext_path.resolve()),      # tight RGBA
            "parsing_step": int(state.get("seq_id")),
            "coord_id": "canvas",
            "source_box": det.get("source") or "detect",
        }
        if etype == "text":
            # 회전 박스 보조정보는 필요시 유지
            if isinstance(box, (list, tuple)) and len(box) == 4 and isinstance(box[0], (list, tuple)):
                el["rotated_bbox"] = {"pts": box}
            el["content"] = txt
            text_ids.append(det_id)
        else:
            if lab is not None:
                el["label"] = lab
            obj_ids.append(det_id)

        _upsert(el)
        added_ids.append(det_id)

    # parse.json 저장 + runtime.extract 업데이트
    parse_doc["elements"] = elements
    _write_parse(state, parse_doc)
    push(r_update_extract(added_ids, text_ids, obj_ids, state))

    # 마무리
    push(upd0)
    push(_bump_next_or(state, "tool_sequence_plan"))
    return r_pack_state(state, *pieces)
