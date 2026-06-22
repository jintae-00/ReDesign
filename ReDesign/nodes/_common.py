from __future__ import annotations
from typing import Dict, Any
from pathlib import Path

from ..state import GraphState
from ..registry import get_tool

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
