# Lazy imports: models are only loaded when the specific tool function is first accessed.
# This prevents loading ALL GPU models (OCR, DINO, SAM2, HiSAM, etc.) when only
# a subset of tools is needed (e.g., only layerd_tool + lama_tool).

import importlib as _importlib

__all__ = [
    "run_ocr", "run_dino_batch_all", "run_yolo",
    "run_hisam_union", "run_sam2_union",
    "run_lama", "run_objectclear",
    "run_storia_onnx", "run_fontstyle_1016", "run_vtracer"
]

_LAZY_IMPORTS = {
    "run_ocr": ".ocr_tool",
    "run_hisam_union": ".hisam_tool",
    "run_lama": ".lama_tool",
    "run_dino_batch_all": ".dino_tool",
    "run_yolo": ".yolo_tool",
    "run_sam2_union": ".sam2_tool",
    "run_objectclear": ".objectclear_tool",
    "run_storia_onnx": ".font_family_storia_onnx",
    "run_fontstyle_1016": ".font_style_1016",
    "run_vtracer": ".vtracer_tool",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = _importlib.import_module(_LAZY_IMPORTS[name], __package__)
        attr = getattr(module, name)
        globals()[name] = attr  # cache for subsequent access
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
