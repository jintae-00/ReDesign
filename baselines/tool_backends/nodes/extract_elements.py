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
    """Always use only the original base image."""
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

    # === Canvas size ===
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

    # === _runtime inputs ===
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

    # det_id -> metadata
    lookup: Dict[str, Dict[str, Any]] = {}
    for i, did in enumerate(det_ids):
        lookup[did] = {
            "idx": i,
            "box": boxes[i] if i < len(boxes) else None,
            "text": texts[i] if (texts and i < len(texts)) else None,
            "label": labels[i] if (labels and i < len(labels)) else None,
        }

    # Load the original once
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

        # 1) Detection bbox (AABB, canvas coordinates)
        aabb_det = _quad_to_aabb_safe(box)
        x1, y1, x2, y2 = map(int, aabb_det)
        x1 = max(0, min(W-1, x1)); x2 = max(1, min(W, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(1, min(H, y2))

        # 2) Load the mask
        m_im = Image.open(str(mask_path))
        if m_im.mode == "RGBA":
            a_in = np.array(m_im)[..., 3].astype(np.uint8)
        else:
            a_in = np.array(m_im.convert("L")).astype(np.uint8)
        mw, mh = m_im.size  # note: (W, H)

        # 3) Build a canvas-size RGBA mask (fill alpha only, no thresholding/correction)
        #    - if already canvas-size, use as is
        #    - if crop-size, it must exactly match the bbox_det size, otherwise raise immediately
        a_full = np.zeros((H, W), dtype=np.uint8)
        a_full = a_in
        # if (mw, mh) == (W, H):
        #     a_full = a_in
        # else:
        #     # must exactly match the bbox_det size
        #     if (mw, mh) != (x2 - x1, y2 - y1):
        #         raise RuntimeError(
        #             f"[extract] det_id={det_id} mask size {mw}x{mh} != bbox_det {x2-x1}x{y2-y1}"
                # )
            # a_full[y1:y2, x1:x2] = a_in

        # 4) Compute the tight box (A>0)
        ys, xs = np.where(a_full > 0)
        if xs.size == 0 or ys.size == 0:
            # skip the element if empty
            continue
        tx1, ty1 = int(xs.min()), int(ys.min())
        tx2, ty2 = int(xs.max()) + 1, int(ys.max()) + 1
        bbox_refined = [tx1, ty1, tx2, ty2]

        # 5) Extract tight RGBA (always from base)
        crop_rgb = base_rgba[ty1:ty2, tx1:tx2, :3]
        crop_a   = (a_full[ty1:ty2, tx1:tx2] > 0).astype(np.uint8) * 255
        ext = np.zeros((ty2 - ty1, tx2 - tx1, 4), np.uint8)
        ext[:, :, :3] = crop_rgb
        ext[:, :, 3]  = crop_a

        # 6) Save
        ext_path           = extracts_dir / f"{det_id}.png"
        mask_canvas_path   = extracts_dir / f"{det_id}_mask_canvas.png"

        # Canvas-size RGBA (fill alpha only)
        mask_canvas_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        mask_canvas_rgba[..., 3] = a_full
        Image.fromarray(mask_canvas_rgba, mode="RGBA").save(str(mask_canvas_path))
        Image.fromarray(ext,               mode="RGBA").save(str(ext_path))

        # 7) Upsert element
        el: Dict[str, Any] = {
            "id": det_id,
            "type": etype,
            "bbox": bbox_refined,                         # tight bbox (canvas coordinates)
            "bbox_det": [x1, y1, x2, y2],                 # original detection bbox
            "mask_uri": str(Path(mask_path).resolve()),   # source-provided mask (size varies)
            "mask_canvas_uri": str(mask_canvas_path.resolve()),  # canvas-size RGBA
            "mask_is_canvas": True,
            "extracted_image_uri": str(ext_path.resolve()),      # tight RGBA
            "parsing_step": int(state.get("seq_id")),
            "coord_id": "canvas",
            "source_box": det.get("source") or "detect",
        }
        if etype == "text":
            # keep rotated-box auxiliary info if needed
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

    # Save parse.json + update runtime.extract
    parse_doc["elements"] = elements
    _write_parse(state, parse_doc)
    push(r_update_extract(added_ids, text_ids, obj_ids, state))

    # Finalize
    push(upd0)
    push(_bump_next_or(state, "tool_sequence_plan"))
    return r_pack_state(state, *pieces)
