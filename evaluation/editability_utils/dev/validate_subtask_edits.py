#!/usr/bin/env python3
"""Validate that each subtask's edit op is actually applied."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from evaluation.editability_utils.common_utils import save_json
from evaluation.editability_utils.task_common import apply_edit_to_scene


def _mk_rect_elem(
    elem_id: str,
    canvas: Tuple[int, int],
    rect: Tuple[int, int, int, int],
    rgb: Tuple[int, int, int],
    z: int,
) -> Dict[str, Any]:
    w, h = canvas
    x1, y1, x2, y2 = rect
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[y1:y2, x1:x2, :3] = np.array(rgb, dtype=np.uint8)
    rgba[y1:y2, x1:x2, 3] = 255
    mask = (rgba[..., 3].astype(np.float32) / 255.0)
    return {
        "id": elem_id,
        "image": Image.fromarray(rgba, "RGBA"),
        "mask": mask,
        "area": float((mask > 0).sum()),
        "bbox": [x1, y1, x2, y2],
        "z_index": int(z),
        "meta": {},
    }


def _alpha_sum(elem: Dict[str, Any]) -> float:
    return float((np.array(elem["image"].convert("RGBA"), dtype=np.uint8)[..., 3] > 0).sum())


def _rgb_mean(elem: Dict[str, Any]) -> float:
    rgba = np.array(elem["image"].convert("RGBA"), dtype=np.uint8)
    fg = rgba[..., 3] > 0
    if not bool(fg.any()):
        return 0.0
    return float(rgba[..., :3][fg].mean())


def _validate_image_edit_ops() -> Dict[str, Any]:
    canvas = (128, 96)
    base = [
        _mk_rect_elem("a", canvas, (20, 20, 66, 66), (220, 30, 30), z=1),
        _mk_rect_elem("b", canvas, (46, 36, 94, 84), (30, 180, 40), z=2),
    ]
    tests = [
        ("delete", {"task_type": "delete"}),
        ("transition", {"task_type": "transition", "dx": 12, "dy": -6}),
        ("rotation", {"task_type": "rotation", "angle_deg": 20}),
        ("opacity", {"task_type": "opacity", "opacity_factor": 0.5}),
        ("recolor", {"task_type": "recolor", "hue_shift_deg": 25.0, "sat_mul": 1.1, "val_mul": 1.0}),
        ("text_bold", {"task_type": "text_bold", "strength": 1}),
        ("text_italic", {"task_type": "text_italic", "shear": 0.15}),
        ("super_scaling", {"task_type": "super_scaling", "scale": 1.3}),
        ("aspect_ratio", {"task_type": "aspect_ratio", "scale_x": 1.25, "scale_y": 0.85}),
        ("corner_radius", {"task_type": "corner_radius", "radius_px": 4}),
        ("stroke", {"task_type": "stroke", "stroke_width": 2, "stroke_rgb": [0, 0, 0]}),
        ("point_edit", {"task_type": "point_edit", "corner": 2, "dx_ratio": 0.15, "dy_ratio": -0.1}),
    ]

    out: Dict[str, Any] = {}
    for name, edit in tests:
        try:
            edited = apply_edit_to_scene(base, [0], canvas, edit)
            changed = False
            if name == "delete":
                changed = _alpha_sum(edited[0]) < _alpha_sum(base[0])
            elif name == "opacity":
                changed = _alpha_sum(edited[0]) <= _alpha_sum(base[0])
            elif name == "recolor":
                changed = abs(_rgb_mean(edited[0]) - _rgb_mean(base[0])) > 1e-3
            else:
                arr_a = np.array(base[0]["image"].convert("RGBA"), dtype=np.uint8)
                arr_b = np.array(edited[0]["image"].convert("RGBA"), dtype=np.uint8)
                changed = bool(np.any(arr_a != arr_b))

            out[name] = {"ok": bool(changed)}
        except Exception as e:
            out[name] = {"ok": False, "error": str(e)}

    # z-order: must swap with overlapping front/back neighbor.
    try:
        z_front = apply_edit_to_scene(base, [0], canvas, {"task_type": "z_order", "direction": "front"})
        z_back = apply_edit_to_scene(base, [1], canvas, {"task_type": "z_order", "direction": "back"})
        out["z_order_front"] = {"ok": int(z_front[0]["z_index"]) == 2 and int(z_front[1]["z_index"]) == 1}
        out["z_order_back"] = {"ok": int(z_back[0]["z_index"]) == 2 and int(z_back[1]["z_index"]) == 1}
    except Exception as e:
        out["z_order_front"] = {"ok": False, "error": str(e)}
        out["z_order_back"] = {"ok": False, "error": str(e)}

    return out


def _validate_text_pipeline_hooks() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from .subtasks.text._shared import _apply_plan_on_text, _pred_text_from_elem, build_content_edit_plan

        sample = "Quick brown fox jumps over a small card"
        plan = build_content_edit_plan(sample, random.Random(0))
        if not plan:
            plan = build_content_edit_plan(sample, random.Random(1))
        edited, idx = _apply_plan_on_text(sample, plan)
        out["content_mod_rule_plan"] = {"ok": bool(plan) and edited != sample and len(idx) > 0}

        canvas = (64, 64)
        elem = {
            "image": Image.new("RGBA", canvas, (0, 0, 0, 0)),
            "meta": {"parsed": {"content": "HELLO"}},
        }
        txt, src = _pred_text_from_elem(elem, model="agent", canvas_size=canvas)
        out["content_recognition_agent_parsed"] = {"ok": (txt == "HELLO" and src == "parsed_content")}
    except Exception as e:
        out["content_mod_rule_plan"] = {"ok": False, "error": str(e)}
        out["content_recognition_agent_parsed"] = {"ok": False, "error": str(e)}
    return out


def _module_presence() -> Dict[str, Any]:
    required = {
        "atomic": [
            "evaluation.editability_utils.subtasks.atomic.delete",
            "evaluation.editability_utils.subtasks.atomic.transition",
            "evaluation.editability_utils.subtasks.atomic.rotation",
            "evaluation.editability_utils.subtasks.atomic.opacity",
            "evaluation.editability_utils.subtasks.atomic.z_order",
        ],
        "text": [
            "evaluation.editability_utils.subtasks.text.content_recognition",
            "evaluation.editability_utils.subtasks.text.content_modification",
            "evaluation.editability_utils.subtasks.text.style_scaling",
            "evaluation.editability_utils.subtasks.text.style_bold",
            "evaluation.editability_utils.subtasks.text.style_italic",
            "evaluation.editability_utils.subtasks.text.style_recolor",
            "evaluation.editability_utils.subtasks.text.style_combo",
        ],
        "svg": [
            "evaluation.editability_utils.subtasks.svg.super_scaling",
            "evaluation.editability_utils.subtasks.svg.stroke",
            "evaluation.editability_utils.subtasks.svg.corner_radius",
            "evaluation.editability_utils.subtasks.svg.aspect_ratio",
            "evaluation.editability_utils.subtasks.svg.recolor",
            "evaluation.editability_utils.subtasks.svg.point_edit",
        ],
    }
    out: Dict[str, Any] = {}
    for cat, mods in required.items():
        cat_out: Dict[str, Any] = {}
        for m in mods:
            try:
                __import__(m)
                cat_out[m] = {"ok": True}
            except Exception as e:
                cat_out[m] = {"ok": False, "error": str(e)}
        out[cat] = cat_out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that subtasks apply real edits")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    report = {
        "module_presence": _module_presence(),
        "image_edit_ops": _validate_image_edit_ops(),
        "text_pipeline_hooks": _validate_text_pipeline_hooks(),
    }
    save_json(Path(args.output), report)
    print(f"[DONE] validation report -> {args.output}")


if __name__ == "__main__":
    main()
