# src/langgraph/nodes/verify_sequence.py
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
    - parse.elements 중에서 parsing_step == 현재 seq_id 인 element 들만 골라서
    - base_pre 와 동일한 크기의 완전 투명 캔버스를 만든 뒤
    - reconstruct_with_elements 를 이용해 그 위에만 합성.
    - 해당 시퀀스에서 추출된 element 가 하나도 없으면, 완전 투명 캔버스만 저장.
    저장 경로 예시:
      <seq_dir>/sequence_elements_seq{seq_id:03d}.png
    """
    parse_doc = state.get("parse")
    elements_all = list(parse_doc.get("elements"))
    seq_id = int(state.get("seq_id"))

    seq_dir.mkdir(parents=True, exist_ok=True)

    # 1) 현재 시퀀스에서 추출된 element 만 필터링
    elems_this_seq: List[Dict[str, Any]] = []
    for e in elements_all:
        ps = e.get("parsing_step")
        if int(ps) == seq_id:
            elems_this_seq.append(e)

    # 2) base_pre 크기 기준으로 투명 캔버스 생성
    base_pre = Path(base_pre_path)
    with Image.open(str(base_pre)) as im:
        W, H = im.size

    seq_id_int = seq_id
    blank_base_path = seq_dir / f"sequence_elements_seq{seq_id_int:03d}_blank.png"
    out_path = seq_dir / f"sequence_elements_seq{seq_id_int:03d}.png"

    # 완전 투명 캔버스 생성 및 저장 (RGBA, A=0)
    blank = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    blank.save(blank_base_path)

    # 3) 이 시퀀스에서 추출된 element 가 하나도 없으면,
    #    out_path 에도 완전 투명 이미지만 저장하고 끝.
    if not elems_this_seq:
        blank.save(out_path)
        return str(out_path)

    # 4) element 가 있으면 기존 reconstruct_with_elements 로직을 그대로 사용
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

    # sequence directory : sequence_pre, sequence_post, artifacts 이미지 저장
    base_pre = (state.get("_runtime") or {}).get("seq", {}).get("base_pre_path")
    # r_update_inpaint, r_update_refine 을 통해서만 state > _runtime > seq > base_post_path 업데이트. Tool Sequence Plan 시 None 초기화
    base_post = (state.get("_runtime") or {}).get("seq", {}).get("base_post_path") or base_pre
    copy2(base_pre, seq_dir / "sequence_pre.png")
    copy2(base_post, seq_dir / "sequence_post.png")

    # episode directory : final_base.png 갱신
    epi_dir = epi_path(str(episode_id))
    fb = epi_dir / "final_base.png"
    bp = Path(base_post)
    if bp.resolve() != fb.resolve():    # tool sequence 에러 발생 혹은 Planning 에러로 인해, inpainting, nanobanana 둘 다 호출 없었을 경우 
        copy2(bp, fb)                   # tool sequence 정상 시행 통해, _runtime > seq > base_post_path 바뀌었을 경우 없데이트


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



    # --- Verify Sequence 프롬프트 구성 ---
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


    # ★ LLM 호출 + JSON 파싱을 try/except 로 감싸고, 실패 시 fallback out 사용
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
        # LLM 실패 시 최소 정보만 채운 fallback
        out = {
            "agent_context_history": "",
            "start_image_context": "",
            "extraction_explanation": "",
            "end_image_context": "",
            "error_explanation": f"VERIFY_SEQUENCE LLM API error: {type(e).__name__}: {e}",
        }


    '''
    # 파일/글로벌 저장(기존 유지)
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
        
    # Sequence Memory 저장
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


    # 히스토리 push_item
    push_item = {
        "seq_id": seq_id,
        "planned_trajectory": cur_traj,  # ← 그대로 전달
        "error_info": err_info,
        "metrics_snapshot": cur_metrics,

        "agent_context_history": out.get("agent_context_history"),
        "start_image_context": out.get("start_image_context"),
        "extraction_explanation": out.get("extraction_explanation"),
        "end_image_context": out.get("end_image_context"),
        "error_explanation": out.get("error_explanation"),  
    }


    # 풀 히스토리 누적 + 다음 노드 라우팅
    upd = r_full_history_append(push_item, state)
    upd.update(r_llm_inc(state))
    upd["route"] = "verify"


    del human_prompt, human_content, arts, sys_prompt
    del recent_k, cur_metrics, cur_traj, err_info
    del cur_seq_extrct_elems_img_path, push_item
    del seq_dir, episode_id, seq_id, base_pre, base_post
    gc.collect()

    return upd
