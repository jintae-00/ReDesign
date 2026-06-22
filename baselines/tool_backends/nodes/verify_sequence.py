# baselines/tool_backends/nodes/verify_sequence.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
import json, base64, os
from shutil import copy2
from PIL import Image
from ..metrics import reconstruct_with_elements
import gc


from ..prompt_builders import build_verify_sequence_prompt, build_verify_sequence_system_prompt
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from .tool_sequence_plan import _extract_last_json_object

from ..memory import (
    save_sequence_raw_global, save_sequence_raw_sidecar,
    seq_path,
    epi_path
)
from ..reducers import r_full_history_append, r_llm_inc


def image_to_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

def _save_sequence_elements_only_canvas(
    state: Dict[str, Any],
    seq_dir: Path,
    base_pre_path: str,
) -> None:
    """
    - From parse.elements, pick only elements whose parsing_step == current seq_id,
    - create a fully transparent canvas the same size as base_pre,
    - and composite only those elements onto it using reconstruct_with_elements.
    - If no element was extracted in this sequence, save only the fully transparent canvas.
    Example output path:
      <seq_dir>/sequence_elements_seq{seq_id:03d}.png
    """
    parse_doc = state.get("parse")
    elements_all = list(parse_doc.get("elements"))
    seq_id = int(state.get("seq_id"))

    seq_dir.mkdir(parents=True, exist_ok=True)

    # 1) Filter to only elements extracted in the current sequence
    elems_this_seq: List[Dict[str, Any]] = []
    for e in elements_all:
        ps = e.get("parsing_step")
        if int(ps) == seq_id:
            elems_this_seq.append(e)

    # 2) Create a transparent canvas based on the base_pre size
    base_pre = Path(base_pre_path)
    with Image.open(str(base_pre)) as im:
        W, H = im.size

    seq_id_int = seq_id
    blank_base_path = seq_dir / f"sequence_elements_seq{seq_id_int:03d}_blank.png"
    out_path = seq_dir / f"sequence_elements_seq{seq_id_int:03d}.png"

    # Create and save a fully transparent canvas (RGBA, A=0)
    blank = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    blank.save(blank_base_path)

    # 3) If no element was extracted in this sequence,
    #    save only a fully transparent image to out_path and finish.
    if not elems_this_seq:
        blank.save(out_path)
        return str(out_path)

    # 4) If there are elements, use the existing reconstruct_with_elements logic as is
    reconstruct_with_elements(
        base_img_path=str(blank_base_path),
        elements=elems_this_seq,
        out_path=str(out_path),
        dump_steps=False,
        steps_dir=None,
        save_masks=False,
    )

    return str(out_path)


def node(state: Dict[str, Any]) -> Dict[str, Any]:

    # import json
    # from ..utils import to_json_safe
    # print(json.dumps(to_json_safe(state), ensure_ascii=False, indent=2))

    episode_id = state.get("episode_id")
    seq_id = state.get("seq_id")
    seq_dir = seq_path(str(episode_id), int(seq_id))
    seq_dir.mkdir(parents=True, exist_ok=True)

    cur_traj = state.get("current_tool_trajectory") or []
    err_info = state.get("error_info")

    # sequence directory: save sequence_pre, sequence_post, and artifact images
    base_pre = (state.get("_runtime") or {}).get("seq", {}).get("base_pre_path")
    # state > _runtime > seq > base_post_path is updated only via r_update_inpaint / r_update_refine. Reset to None during Tool Sequence Plan.
    base_post = (state.get("_runtime") or {}).get("seq", {}).get("base_post_path") or base_pre
    copy2(base_pre, seq_dir / "sequence_pre.png")
    copy2(base_post, seq_dir / "sequence_post.png")

    # episode directory: refresh final_base.png
    epi_dir = epi_path(str(episode_id))
    fb = epi_dir / "final_base.png"
    bp = Path(base_post)
    if bp.resolve() != fb.resolve():    # when neither inpainting nor nanobanana was called due to a tool-sequence error or planning error
        copy2(bp, fb)                   # update when the tool sequence ran normally and changed _runtime > seq > base_post_path


    arts = ((state.get("_runtime") or {}).get("seq") or {}).get("artifacts") or []
    for a in arts:
        a_path = Path(a.get("path"))
        dst_path = seq_dir / a_path.name
        copy2(a_path, dst_path)




    k = int(state.get("history_window_size") or 3)
    full = state.get("history_window_full_episode") or []
    recent_k = full[-k:] if len(full) > k else full

    cur_metrics = state.get("current_seq_metrics")

    cur_seq_extrct_elems_img_path = _save_sequence_elements_only_canvas(
        state=state, 
        seq_dir=seq_dir, 
        base_pre_path=base_pre
    )



    # --- Build the Verify Sequence prompt ---
    sys_prompt = build_verify_sequence_system_prompt()
    human_prompt = build_verify_sequence_prompt(
        recent_k=recent_k,
        current_seq_metrics=cur_metrics,
        planned_trajectory=cur_traj,
        current_error_info=err_info,
    )


    _llm = ChatOpenAI(
        model_name="gpt-5-mini",
        base_url=os.environ.get("OPENAI_BASE_URL","https://gateway.letsur.ai/v1"),
        temperature=0,
        top_p=1
    )
    
    human_content = [{"type": "text", "text": human_prompt}]

    human_content.append({"type":"text", "text": "Base image BEFORE current sequence: "})
    human_content.append({"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_to_b64(base_pre)}"}})

    # for a in arts:
    #     human_content.append({"type":"text", "text": f"{a['tool_id']} artifact image : "})
    #     human_content.append({"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_to_b64(a['path'])}"}})
    
    human_content.append({"type":"text", "text": "Visualization of elements detected and extracted in the current sequence : "})
    human_content.append({"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_to_b64(cur_seq_extrct_elems_img_path)}"}})
    
    human_content.append({"type":"text", "text": "Final image AFTER current sequence:"})
    human_content.append({"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_to_b64(base_post)}"}})


    # Wrap the LLM call + JSON parsing in try/except, and use a fallback out on failure
    out: Dict[str, Any] = {}
    try:
        resp = _llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_content),
        ])
        raw = resp.content or ""
        out = _extract_last_json_object(str(raw))
    except Exception as e:
        llm_ok = False
        print("\n\n[VERIFY SEQUENCE] LLM error:", repr(e))
        # Fallback filled with minimal info when the LLM fails
        out = {
            "agent_context_history": "",
            "start_image_context": "",
            "extraction_explanation": "",
            "end_image_context": "",
            "error_explanation": f"VERIFY_SEQUENCE LLM API error: {type(e).__name__}: {e}",
        }


    '''
    # File/global saving (kept as before)
    seq_rec = {
        "run_id": state.get("run_id"),
        "episode_id": episode_id,
        "seq_id": seq_id,
        "agent_context_history": out.get("agent_context_history"),
        "image_context": out.get("image_context"),
        "sequence": out.get("sequence"),
        "trajectory": cur_traj, 
        "metrics_snapshot": cur_metrics,
        "error_info" : err_info,
    }
    save_sequence_raw_global(seq_rec)
    save_sequence_raw_sidecar(episode_id, seq_id, seq_rec)

    for t in (out.get("tools") or []):
        tool_id = t.get("tool_id")
        tool_rec = {
            "run_id": state.get("run_id"),
            "episode_id": episode_id,
            "seq_id": seq_id,
            "agent_context_history": out.get("agent_context_history"),
            "image_context": out.get("image_context"),
            "tool_id": tool_id,
            "did_meet_criteria": t.get("did_meet_criteria"),
            "notes": t.get("notes"),
        }
        save_tool_raw_global(tool_id, tool_rec)
        save_tool_raw_sidecar(episode_id, seq_id, tool_rec)

    '''
        
    # Save Sequence Memory
    seq_rec = {
        "run_id": state.get("run_id"),
        "episode_id": episode_id,
        "seq_id": seq_id,
        "planned_trajectory": cur_traj,
        "error_info": err_info,
        "metrics_snapshot": cur_metrics,

        "agent_context_history": out.get("agent_context_history"),
        "start_image_context": out.get("start_image_context"),
        "extraction_explanation": out.get("extraction_explanation"),
        "end_image_context": out.get("end_image_context"),
        "error_explanation": out.get("error_explanation"),
    }
    save_sequence_raw_global(seq_rec)
    save_sequence_raw_sidecar(episode_id, seq_id, seq_rec)


    # History push_item
    push_item = {
        "seq_id": seq_id,
        "planned_trajectory": cur_traj,  # passed through as is
        "error_info": err_info,
        "metrics_snapshot": cur_metrics,

        "agent_context_history": out.get("agent_context_history"),
        "start_image_context": out.get("start_image_context"),
        "extraction_explanation": out.get("extraction_explanation"),
        "end_image_context": out.get("end_image_context"),
        "error_explanation": out.get("error_explanation"),  
    }


    # Accumulate full history + route to the next node
    upd = r_full_history_append(push_item, state)
    upd.update(r_llm_inc(state))
    upd["route"] = "verify"


    del human_prompt, human_content, arts, sys_prompt
    del recent_k, cur_metrics, cur_traj, err_info
    del cur_seq_extrct_elems_img_path, push_item
    del seq_dir, episode_id, seq_id, base_pre, base_post
    gc.collect()

    return upd
