# src/langgraph/nodes/vlm_front_pick.py
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import gc

from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..state import GraphState
from ..reducers import (
    r_set_instance, r_llm_inc, llm_can_call, 
    r_pack_state,
    r_abort_to_verify_sequence
)
from ..utils import image_to_b64, extract_json, resolve_img
from ..prompts import VLM_FRONT_ELEMS_PICK
from ._common import _pop_current_action, _bump_next_or

_llm = ChatOpenAI(
    model_name="gpt-5-mini",
    base_url="https://gateway.letsur.ai/v1",
    temperature=0,           # greedy decoding → 거의 결정론적
    top_p= 1,
)


def node(state: GraphState) -> Dict[str, Any]:
    act, upd0 = _pop_current_action(state)

    if not llm_can_call(state):
        return r_abort_to_verify_sequence(
            state,
            node="vlm_front_elems_pick",
            error_msg="vlm_front_elems_pick failed: LLM budget exhausted",
            details={"llm_can_call": llm_can_call(state)},
        )

    img_sel = resolve_img("vlm_pick", state)  # ← front > base
    img = Path(img_sel["path"])

    try:
        resp = _llm.invoke([HumanMessage(content=[
            {"type": "text", "text": VLM_FRONT_ELEMS_PICK},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_b64(img)}"}},
        ])])
    except Exception as e:
        return r_abort_to_verify_sequence(
            state,
            node="vlm_front_elems_pick",
            error_msg="vlm_front_elems_pick LLM API error",
            details={
                "exception_type": type(e).__name__,
                "raw_error": str(e),
            },
        )


    js = extract_json(resp.content)
    labels = (js or {}).get("labels")


    print(f"\n[VLM FRONT PICK] labels : {labels}\n")

    upd: Dict[str, Any] = {}
    upd.update(upd0)

    if not labels:
        return r_abort_to_verify_sequence(
            state,
            node="vlm_front_elems_pick",
            error_msg="vlm_front_elems_pick failed: no labels for front placed elements proposed",
            details={"labels (llm output)": labels},
        )

    # Set instance (object/text)
    upd.update(r_set_instance(labels, state))
    upd.update(r_llm_inc(state))

    upd.update(_bump_next_or(state, "tool_sequence_plan"))

    del resp, js, labels, img_sel, img, act
    gc.collect()

    return r_pack_state(state, upd)
