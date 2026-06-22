# REDESIGN/registry.py
"""
Tool Registry for URLD Pipeline

Registers all available tools for the pipeline.
Each tool has a 'runner' function that executes the tool.
"""
from __future__ import annotations


class ToolSpec:
    """Simple tool specification holder"""
    def __init__(self, runner):
        self.runner = runner


# Lazy loading of tools to avoid import errors if dependencies missing
_REGISTRY = {}


def _lazy_load_tools():
    """Load tools lazily to handle missing dependencies gracefully"""
    global _REGISTRY
    
    if _REGISTRY:
        return
    
    # Nanobanana (Gemini-based refinement)
    try:
        from .tools.nanobanana_tool import run_nanobanana
        _REGISTRY["nanobanana"] = ToolSpec(run_nanobanana)
    except ImportError:
        pass
    
    # OCR
    try:
        from .tools.ocr_tool import run_ocr
        _REGISTRY["OCR"] = ToolSpec(run_ocr)
    except ImportError:
        pass
    
    # GDINO (Grounding DINO)
    try:
        from .tools.dino_tool import run_dino_batch_all
        _REGISTRY["GDINO"] = ToolSpec(run_dino_batch_all)
    except ImportError:
        pass
    
    # HiSAM (Text segmentation)
    try:
        from .tools.hisam_tool import run_hisam_union
        _REGISTRY["HiSAMUnion"] = ToolSpec(run_hisam_union)
    except ImportError:
        pass
    
    # SAM2 (Object segmentation)
    try:
        from .tools.sam2_tool import run_sam2_union
        _REGISTRY["SAM2_BBOX"] = ToolSpec(run_sam2_union)
    except ImportError:
        pass
    
    # LaMa (Inpainting)
    try:
        from .tools.lama_tool import run_lama
        _REGISTRY["LaMa"] = ToolSpec(run_lama)
    except ImportError:
        pass
    
    # ObjectClear (Inpainting)
    try:
        from .tools.objectclear_tool import run_objectclear
        _REGISTRY["ObjectClear"] = ToolSpec(run_objectclear)
    except ImportError:
        pass
    
    # FontStyle
    try:
        from .tools.font_style_1016 import run_fontstyle_1016
        _REGISTRY["fontstyle_1016"] = ToolSpec(run_fontstyle_1016)
    except ImportError:
        pass
    
    # VTracer (SVG conversion)
    try:
        from .tools.vtracer_tool import run_vtracer
        _REGISTRY["vtracer"] = ToolSpec(run_vtracer)
    except ImportError:
        pass
    
    # Qwen Layered (Multi-layer generation)
    try:
        from .tools.qwen_layered_tool import run_qwen_layered
        _REGISTRY["qwen_layered"] = ToolSpec(run_qwen_layered)
    except ImportError:
        pass
    
    # CCA (Connected Component Analysis)
    try:
        from .tools.cca_tool import run_split_cca
        _REGISTRY["CCA"] = ToolSpec(run_split_cca)
    except ImportError:
        pass


def get_tool(name: str) -> ToolSpec:
    """Get a tool by name"""
    _lazy_load_tools()
    
    if name not in _REGISTRY:
        raise KeyError(f"Tool '{name}' not found. Available tools: {list(_REGISTRY.keys())}")
    
    return _REGISTRY[name]


def list_tools() -> list:
    """List all available tools"""
    _lazy_load_tools()
    return list(_REGISTRY.keys())


def register_tool(name: str, runner):
    """Register a custom tool"""
    _REGISTRY[name] = ToolSpec(runner)