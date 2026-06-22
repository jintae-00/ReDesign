# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import json, base64, os
import gc
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from ..prompt_builders import build_plan_prompt, build_plan_system_prompt
from ..memory import _MEM_ROOT, read_seq_fewshot_heuristics_for_planner, bump_seq_fewshot_ref_counts
from ..reducers import (
    r_bump, r_set_pending, r_reset_runtime_for_next_seq, r_pack_state, 
    r_set_current_tool_trajectory,  r_llm_inc,
    r_abort_to_verify_sequence
)


def _load_static_texts(tool_ids: List[str]) -> Dict[str, Any]:
    ROOT = _MEM_ROOT()
    tools = {}
    for tid in tool_ids:
        p = ROOT / "ToolMem" / tid / f"{tid}_static.json"
        try:
            tools[tid] = json.loads(p.read_text(encoding="utf-8") or "{}") if p.exists() else {}
        except Exception:
            tools[tid] = {}
    seq = {}
    try:
        seq = json.loads((ROOT / "SequenceMem" / "seq_static.json").read_text(encoding="utf-8") or "{}")
    except Exception:
        seq = {}
    # epi = {}
    # try:
    #     epi = json.loads((ROOT / "EpisodeMem" / "epi_static.json").read_text(encoding="utf-8") or "{}")
    # except Exception:
    #     epi = {}
    # return {"tools": tools, "sequence": seq, "episode": epi}
    return {"tools": tools, "sequence": seq}

def _extract_last_json_object(text: str) -> Dict[str, Any]:
    depth = 0
    start: Optional[int] = None
    last_obj: Optional[str] = None

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    # the point where one JSON object ends
                    candidate = text[start : i + 1]
                    last_obj = candidate

    if last_obj is None:
        raise ValueError("No JSON object found in text")

    return json.loads(last_obj)

def node(state: Dict[str, Any]) -> Dict[str, Any]:

    # import json
    # from ..utils import to_json_safe
    # print(json.dumps(to_json_safe(state), ensure_ascii=False, indent=2))

    # Always: start a new sequence (increment seq_id / create directory) + force-reset (replace) _runtime
    init = r_reset_runtime_for_next_seq(state)

    # base_pre to use in the LLM prompt
    base_pre_for_llm = init.get('_runtime').get('img').get('base').get('path')
    img_b64 = base64.b64encode(Path(base_pre_for_llm).read_bytes()).decode()

    tool_ids = state.get("tool_ids") or [
        "nanobanana","vlm_front_elems_pick",
        "layerd_front","front_split",
        "ocr","gdino","yolo",
        "sam2_bbox","layerd_bbox","hisam",
        "lama","objectclear",
        "extract_elements",
        "storia_onnx","fontstyle_1016",
        "vtracer",
    ]

    k = int(state.get("history_window_size") or 3)
    full = state.get("history_window_full_episode") or []
    recent_k = full[-k:] if len(full) > k else full

    # Load static descriptions
    st = _load_static_texts(tool_ids)

    # Read a snapshot of the Tool Sequence Few-shot Heuristics learned up to now
    seq_fewshot_heur_all = read_seq_fewshot_heuristics_for_planner()


    # 1) System Prompt: STATIC_PLAN_EXPLANATION + Tools Static + Sequence Static
    sys_prompt = build_plan_system_prompt(
        tools_static=st["tools"],
        seq_static=st["sequence"],
    )
    # 2) Human Prompt: Few-shot Heuristics + Recent histories + Image label
    human_prompt = build_plan_prompt(
        image_label="Current base image (path hidden to LLM)",
        recent_k=recent_k,
        seq_fewshot_heur_all=seq_fewshot_heur_all,
    )

    # ===== DEBUG: print Planner Prompt =====
    print("\n" + "=" * 80)
    print("[TOOL SEQUENCE PLAN — SYSTEM PROMPT]")
    print("=" * 80 + "\n")
    print(sys_prompt)
    print("\n" + "=" * 80)
    print("[TOOL SEQUENCE PLAN — HUMAN PROMPT]")
    print("=" * 80 + "\n")
    print(human_prompt)
    print("\n" + "=" * 80)
    print("[/TOOL SEQUENCE PLAN — PROMPT END]")
    print("=" * 80 + "\n")


    _llm = ChatOpenAI(
        model_name="gpt-5-mini",
        base_url=os.environ.get("OPENAI_BASE_URL", "https://gateway.letsur.ai/v1"),
        temperature=0,
        top_p=1,
    )

    human_content = [
        {"type": "text", "text": human_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        },
    ]


    try:
        resp = _llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_content),
        ])
    except Exception as e:
        print("\n[TOOL SEQUENCE PLAN] LLM API error:", repr(e))
        return r_abort_to_verify_sequence(
            state,
            node="tool_sequence_plan",
            error_msg="tool_sequence_plan LLM API error",
            details={
                "exception_type": type(e).__name__,
                "raw_error": str(e),
            },
        )

    raw = resp.content

    # ===== DEBUG: print LLM Raw Response =====
    print("\n" + "-" * 80)
    print("[TOOL SEQUENCE PLAN — LLM RAW RESPONSE]")
    print("-" * 80 + "\n")
    print(raw)
    print("\n" + "-" * 80)
    print("[/TOOL SEQUENCE PLAN — LLM RAW RESPONSE END]")
    print("-" * 80 + "\n")

    plan = {}
    try:
        plan = _extract_last_json_object(raw)
    except Exception as e:
        print("\nplan json parse error:", e)
        plan = {}

    # --- Planner output: expecting { "used_heuristics": [...], "actions": [...] } ---
    used_heuristics: List[Dict[str, Any]] = []
    if isinstance(plan, dict):
        uh = plan.get("used_heuristics")
        if isinstance(uh, list):
            used_heuristics = uh

    # Increment ref_count based on used_heuristics
    '''
    if used_heuristics:
        bump_seq_fewshot_ref_counts(used_heuristics)
    '''
        
    actions = plan.get("actions") or []

    # (A) First extract only the trajectory from the raw action list given by the LLM
    planned_nodes: List[str] = []
    for a in actions:
        if isinstance(a, dict):
            node_name = a.get("node")
            if node_name:
                planned_nodes.append(str(node_name))

    # Append only metrics_gate as the tail (subsequent routing is done explicitly in each node)
    # tail_chain = [{"node": "metrics_gate"}]
    tail_chain = [{"node": "verify_sequence"}]
    full_actions = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        node_name = a.get("node")
        args = a.get("args") or {}
        full_actions.append({"node": node_name, "args": args})
    full_actions.extend(tail_chain)


    del resp, raw, plan, actions, used_heuristics
    del img_b64, st, seq_fewshot_heur_all, recent_k, full, tool_ids
    gc.collect()

    return r_pack_state(
        state,
        init,
        r_set_current_tool_trajectory(planned_nodes, state),
        r_set_pending(full_actions, state),
        r_llm_inc(state),
        r_bump(full_actions[0]["node"] if full_actions else "verify_sequence", state),
    )
