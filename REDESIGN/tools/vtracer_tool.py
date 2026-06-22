# src/langgraph/tools/vtracer_tool.py
from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import vtracer

_FINE = dict(
    colormode="color",
    hierarchical="stacked",
    mode="pixel",
    filter_speckle=0,
    color_precision=8,
    layer_difference=34
)

def run_vtracer(
    src_rgba_path: str,
    out_svg_path: str,
) -> Dict[str, Any]:
    """
    returns: {"svg_uri": out_svg_path, "args": used_args}
    """
    Path(out_svg_path).parent.mkdir(parents=True, exist_ok=True)
    # args = dict(_FINE if mode == "fine" else _AGGR)
    args = dict(_FINE)
    vtracer.convert_image_to_svg_py(src_rgba_path, out_svg_path, **args)
    return {"svg_uri": out_svg_path, "args": args}
