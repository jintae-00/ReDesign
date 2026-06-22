# src/langgraph/nodes/metrics_gate.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict
from ..reducers import r_bump
from ..metrics import compute_epi_metrics, persist_epi_metrics

'''
def node(state: Dict[str, Any]) -> Dict[str, Any]:
    at_epi_end = bool(state.get("_finish_episode"))
    if not at_epi_end:
        seq_metrics = compute_seq_metrics(state)
        persist_seq_metrics(state, seq_metrics)
        prev_list = list(state.get("previous_seq_metrics") or [])
        prev_list.append(seq_metrics)  # ✅ 누적
        return {
            "current_seq_metrics": seq_metrics,        # ✅ 현재 시퀀스에 사용
            "previous_seq_metrics": prev_list,         # ✅ Δ 계산용 누적 리스트
            **r_bump("verify_sequence", state)
        }
    else:
        epi_metrics = compute_epi_metrics(state)
        persist_epi_metrics(state, epi_metrics)
        return {
            "_last_epi_metrics": epi_metrics,
            **r_bump("verify_episode", state)
        }
'''


def node(state: Dict[str, Any]) -> Dict[str, Any]:
    at_epi_end = state.get("_finish_episode")
    print(f"[METRCIS_GATE] at_epi_end : {at_epi_end}")
    if not at_epi_end:
        # seq_metrics = compute_seq_metrics(state)
        # persist_seq_metrics(state, seq_metrics)
        # prev_list = list(state.get("previous_seq_metrics") or [])
        # prev_list.append(seq_metrics)  # ✅ 누적
        return {
            # "current_seq_metrics": seq_metrics,        # ✅ 현재 시퀀스에 사용
            # "previous_seq_metrics": prev_list,         # ✅ Δ 계산용 누적 리스트
            **r_bump("verify_sequence", state)
        }
    else:
        epi_metrics = compute_epi_metrics(state)
        persist_epi_metrics(state, epi_metrics)
        return {
            "_last_epi_metrics": epi_metrics,
            **r_bump("end", state)
        }