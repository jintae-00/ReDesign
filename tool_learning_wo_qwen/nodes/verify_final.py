# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
import os, base64, json
from datetime import datetime
import gc


from ..reducers import (
    r_bump, r_llm_inc, llm_can_call,
    r_pack_state,
)
from ..prompts import VERIFY_FINAL
from ..memory import epi_path
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


def _image_to_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

def _should_end_episode_nonllm(state: Dict[str, Any]) -> bool:
    """
    Non-LLM 종료 조건 (하나라도 True면 에피소드 종료):
      1) LLM used >= max                    (기본 50)
      2) seq_id >= max_sequences            (기본 15)
      3) now - start_ts >= max_duration_sec (기본 3600s == 1시간)
    각 판정 시 현재/최대 값을 콘솔에 출력합니다.
    """

    # (1) LLM 사용량
    llm = state.get("llm")
    used = int(llm.get("used"))
    mx   = llm.get("max")
    print(f"[verify_final.nonllm] llm call : {used} / {mx}")
    if mx is not None and used >= mx:
        print("[verify_final.nonllm] stop reason: LLM budget reached")
        return True

    # (2) 시퀀스 인덱스
    seq_id = int(state.get("seq_id"))
    max_sequences = int(state.get("max_sequences"))
    print(f"[verify_final.nonllm] sequence : {seq_id} / {max_sequences}")
    if max_sequences and seq_id >= max_sequences:
        print("[verify_final.nonllm] stop reason: max_sequences reached")
        return True

    # (3) 경과 시간(초)
    # 시작 시각: state["ts"] (ISO8601 문자열)
    # 한도: state["max_duration_sec"] (기본 3600)
    max_dur = int(state.get("max_duration_sec"))
    now = datetime.now()
    start_iso = state.get("ts")
    start = datetime.fromisoformat(start_iso)    
    dur = (now - start).total_seconds()
    mm, ss = divmod(int(dur), 60)
    max_mm, max_ss = divmod(int(max_dur), 60)
    print(f"[verify_final.nonllm] duration : {mm:02d}:{ss:02d} / {max_mm:02d}:{max_ss:02d} (mm:ss)")
    if max_dur > 0 and dur >= max_dur:
        print("[verify_final.nonllm] stop reason: duration limit reached")
        return True

    return False


def _parse_final_verdict(text: str) -> str:
    import re as _re
    lines = [ln.strip().lower() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    last = lines[-1]
    if last == "done":
        return "done"
    if last == "not yet":
        return "not yet"
    m = _re.search(r"(done|not yet)\s*$", (text or "").strip().lower())
    return m.group(1) if m else ""

# =========================
# Real-time Visualization
# =========================

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _episode_dir(state: Dict[str, Any]) -> Path:
    ep = str(state.get("episode_id"))
    d = epi_path(ep)
    _ensure_dir(d)
    return d

def _plot_seq_quadrants(state: Dict[str, Any]) -> Path:
    """
    previous_seq_metrics 전부 사용하여 4분면 그래프 생성:
    - Row1: MSE (abs)
    - Row2: PSNR (abs)
    - Row3: DINO similarity (abs)
    - Row4: Editable Count (cum)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epi_dir = _episode_dir(state)
    out_path = epi_dir / "seq_metrics_quadrants.png"

    rows: List[Dict[str, Any]] = list(state.get("previous_seq_metrics") or [])
    # rows가 비어있으면 matplotlib 자체가 빈 축만 그리도록 놔둔다 (데이터 없음도 에러 아님)
    xs   = [int(r.get("seq_id")) for r in rows]
    mse  = [float(r.get("mse_abs")) for r in rows]
    psnr = [float(r.get("psnr_abs")) for r in rows]
    dino = [float(r.get("dino_abs")) for r in rows]
    cnt  = [int(((r.get("edit")) .get("count_cum"))) for r in rows]
    # clip = [float(r.get("clip_abs")) for r in rows]
    # area = [float(((r.get("edit")) .get("area_cum_pct"))) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    (ax1, ax2), (ax3, ax4) = (axes[0], axes[1])

    ax1.plot(xs, mse, marker="o")
    ax1.set_title("MSE (abs)")
    ax1.set_xlabel("seq_id")
    ax1.set_ylabel("MSE")

    ax2.plot(xs, psnr, marker="o")
    ax2.set_title("PSNR (abs)")
    ax2.set_xlabel("seq_id")
    ax2.set_ylabel("PSNR (dB)")

    ax3.plot(xs, dino, marker="o")
    ax3.set_title("DINO similarity (abs)")
    ax3.set_xlabel("seq_id")
    ax3.set_ylabel("similarity")

    ax4.plot(xs, cnt, marker="o")
    ax4.set_title("Editable Count (cum)")
    ax4.set_xlabel("seq_id")
    ax4.set_ylabel("#elements")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _realtime_viz(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    p = _plot_seq_quadrants(state)
    print(f"[verify_final] metrics plot updated → {p}")

# =========================
# Main node
# =========================

def node(state: Dict[str, Any]) -> Dict[str, Any]:

    # import json
    # from ..utils import to_json_safe
    # print(json.dumps(to_json_safe(state), ensure_ascii=False, indent=2))

    base_post = (state.get("_runtime") or {}).get("seq", {}).get("base_post_path") or \
                ((state.get("_runtime") or {}).get("img") or {}).get("base", {}).get("path")

    pieces: list[Dict[str, Any]] = []

    if not base_post or not Path(base_post).exists():
        pieces += [
            r_bump("tool_sequence_plan", state),
        ]
        return r_pack_state(state, *pieces)

    # 실시간 시각화(기존 유지)
    _realtime_viz(state)

    llm_done = False
    nonllm_done = _should_end_episode_nonllm(state)


    ''' Agent History 추가 전달 코드
    
    # -----------------------------
    # 1) recent history 정리 (history_window_full_episode 기반)
    #    → verify_sequence 에서 push_item 으로 넣어준 필드를 그대로 사용
    # -----------------------------
    k = int(state.get("history_window_size") or 3)
    full_hist = state.get("history_window_full_episode") or []
    if len(full_hist) > k:
        recent_k = full_hist[-k:]
    else:
        recent_k = full_hist

    recent_payload: List[Dict[str, Any]] = []
    for h in recent_k:
        row = {
            "seq_id": h.get("seq_id"),
            "planned_trajectory": h.get("planned_trajectory"),
            "error_info": h.get("error_info"),
            "metrics_snapshot": h.get("metrics_snapshot"),

            "agent_context_history": h.get("agent_context_history"),
            "start_image_context": h.get("start_image_context"),
            "end_image_context": h.get("end_image_context"),
            "tools": h.get("tools"),
            "error_explanation": h.get("error_explanation"),
            "extraction_explanation": h.get("extraction_explanation"),
        }
        recent_payload.append(row)

    recent_json = json.dumps(recent_payload, ensure_ascii=False, indent=2)
    history_block = "\n\n[recent_sequences]\n" + recent_json
    '''

    # -----------------------------
    # 2) LLM 호출
    # -----------------------------
    if llm_can_call(state):
        _llm = ChatOpenAI(
            model_name="gpt-5-mini",
            base_url=os.environ.get("OPENAI_BASE_URL", "https://gateway.letsur.ai/v1"),
            temperature=0,
            top_p=1
        )

        ''' Agent History 추가 전달 코드
        prompt_text = VERIFY_FINAL.strip() + history_block
        '''
        
        try:
            ''' Agent History 추가 전달 코드
            resp = _llm.invoke([HumanMessage(content=[
                {
                    "type": "text",
                    "text": prompt_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_image_to_b64(base_post)}"
                    },
                },
            ])])
            '''
            resp = _llm.invoke([HumanMessage(content=[
                {"type": "text", "text": VERIFY_FINAL.strip()},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_image_to_b64(base_post)}"}}
            ])])
            print(f"[VERIFY FINAL] LLM Response : {resp.content}")
            verdict = _parse_final_verdict(resp.content or "")
            pieces.append(r_llm_inc(state))
            llm_done = (verdict == "done")
        except Exception as e:
            print(f"verify_final: LLM error → fallback to non-LLM rules ({type(e).__name__}: {e})")

    # -----------------------------
    # 3) 종료 여부 결정 (기존 로직 유지)
    # -----------------------------
    if llm_done or nonllm_done:
        reason = "LLM" if llm_done else "non-LLM"
        pieces += [
            {"_finish_episode": True},
            r_bump("metrics_gate", state),
        ]
        gc.collect()
        return r_pack_state(state, *pieces)

    pieces += [
        r_bump("tool_sequence_plan", state),
    ]
    gc.collect()
    return r_pack_state(state, *pieces)
