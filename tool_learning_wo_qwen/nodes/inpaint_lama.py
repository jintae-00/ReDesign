# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from PIL import Image as _PIL
from ..state import GraphState
from ..reducers import (
    r_save_artifact, r_update_inpaint,
    r_set_img_slot, r_pack_state
)
from ..utils import resolve_args, resolve_img
from ._common import _pop_current_action, _bump_next_or, _call_tool


def node(state: GraphState) -> Dict[str, Any]:

    act, upd0 = _pop_current_action(state)

    raw_args = act.get("args") or {}
    args = resolve_args(raw_args, last=None, state=state, labels=None, image_kind="inpaint")
    args.setdefault("image_path", resolve_img("inpaint", state)["path"])

    mask_path = state.get("_mask_path") or (
        (state.get("_runtime") or {}).get("segment") or {}
    ).get("mask_path")
    args.setdefault("mask_path", mask_path)


    # print(f"[LaMa] args : {args}")
    out_img = _call_tool("LaMa", args, state)

    pieces: list[Dict[str, Any]] = [upd0]

    # 1) 아티팩트 저장 (경로 확정)
    u_art, saved = r_save_artifact(out_img, "lama", "inpaint", state)
    pieces.append(u_art)

    # 2) inpaint 업데이트 (seq.artifacts 필드가 실수로 비워지지 않게 sanitize)
    u_inp = r_update_inpaint("lama", saved, state)
    pieces.append(u_inp)

    # 3) base 슬롯 갱신
    try:
        W, H = _PIL.open(saved).size
    except Exception:
        W = H = None
    pieces.append(r_set_img_slot("base", saved, coord_id="base", W=W, H=H, state=state))

    # 5) 다음으로
    pieces.append(_bump_next_or(state, "tool_sequence_plan"))
    return r_pack_state(state, *pieces)
