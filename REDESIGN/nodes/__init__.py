# REDESIGN/nodes/__init__.py
"""
URLD Pipeline Nodes

All nodes follow the same pattern:
- Input: GraphState
- Output: Dict[str, Any] (state updates)
- Each node reads from current_layer_id and history_tree
- Each node updates tool_outputs and dequeues from node_queue
"""

from . import router_vlm
from . import vlm_front_pick
from . import qwen_layered
from . import split_cca
from . import stack_manager

from . import detect_ocr
from . import detect_gdino

from . import seg_hisam
from . import seg_sam2_bbox

from . import inpaint_lama
from . import inpaint_oc
from . import nanobanana

from . import fontstyle
from . import vtracer

from . import finalize_text
from . import finalize_obj

__all__ = [
    "router_vlm",
    "vlm_front_pick",
    "qwen_layered",
    "split_cca",
    "stack_manager",
    "detect_ocr",
    "detect_gdino",
    "seg_hisam",
    "seg_sam2_bbox",
    "inpaint_lama",
    "inpaint_oc",
    "nanobanana",
    "fontstyle",
    "vtracer",
    "finalize_text",
    "finalize_obj",
]