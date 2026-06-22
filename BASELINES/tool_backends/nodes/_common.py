 # BASELINES/tool_backends/nodes/_common.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
from pathlib import Path
import inspect

from ..state import GraphState
from ..reducers import r_bump
from ..registry import get_tool

def _pop_current_action(state: GraphState) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    q: List[Dict[str, Any]] = state.get("pending_actions") or []
    if not q:
        return None, {}
    act = q[0]
    upd = {"pending_actions": q[1:]}
    return act, upd

def _bump_next_or(state: GraphState, fallback: str) -> Dict[str, Any]:
    q: List[Dict[str, Any]] = state.get("pending_actions") or []
    if q[1:] and q[1].get("node"):
        return r_bump(q[1]["node"], state)
    return r_bump(fallback, state)

def _call_tool(tool_name: str, args: Dict[str, Any], state: GraphState):
    spec = get_tool(tool_name)
    runner = spec.runner
    import inspect
    sig = inspect.signature(runner)
    kwargs = dict(args or {})
    if "vis_dir" in sig.parameters and "vis_dir" not in kwargs:
        kwargs["vis_dir"] = Path(state["out_dir"])
    if "step" in sig.parameters and "step" not in kwargs:
        kwargs["step"] = state["step"]
    safe = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return runner(**safe)
