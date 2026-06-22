#!/usr/bin/env python3
"""Thin adapter for Nanobanana image editing in editability eval."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def run_nanobanana_on_rgba(
    rgba: np.ndarray,
    instruction: str,
    retries: int = 2,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Run Nanobanana on a single RGBA image and return edited RGBA buffer."""
    meta: Dict[str, Any] = {
        "ok": False,
        "instruction": instruction,
        "retries": int(retries),
        "attempts": 0,
        "error": None,
    }
    try:
        from tool_learning.tools.nanobanana_tool import run_nanobanana
    except Exception as e:
        meta["error"] = f"import_failed: {e}"
        return None, meta

    with tempfile.TemporaryDirectory(prefix="editability_nanobanana_") as td:
        in_path = Path(td) / "input.png"
        Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), "RGBA").save(in_path)

        for attempt in range(1, max(1, int(retries)) + 1):
            meta["attempts"] = attempt
            try:
                out_path = Path(run_nanobanana(str(in_path), instruction))
                if not out_path.exists():
                    raise FileNotFoundError(f"nanobanana output missing: {out_path}")
                out_rgba = np.array(Image.open(out_path).convert("RGBA"), dtype=np.uint8)
                meta["ok"] = True
                meta["output_path"] = str(out_path)
                return out_rgba, meta
            except Exception as e:
                meta["error"] = str(e)
                continue

    return None, meta


def build_text_edit_instruction(
    target_text: str,
    source_text: str = "",
    changed_words: Optional[List[Tuple[str, str]]] = None,
) -> str:
    pairs = changed_words or []
    pair_txt = "; ".join([f"'{a}' -> '{b}'" for a, b in pairs[:10]])
    source_clause = f"Original text is: \"{source_text}\". " if source_text else ""
    pair_clause = f"Apply these word edits exactly: {pair_txt}. " if pair_txt else ""
    return (
        "You are editing only the text inside this image. "
        + source_clause
        + pair_clause
        + f"Final text must be exactly: \"{target_text}\". "
        "Preserve layout, alignment, font family, weight, size, color, spacing, and all non-text pixels. "
        "Do not add/remove decorations, icons, or background changes."
    )


def build_recolor_instruction(hue_shift_deg: float, sat_mul: float, val_mul: float) -> str:
    return (
        "Recolor only the main foreground object while preserving shape, position, and opacity. "
        f"Approximate transformation: hue shift {hue_shift_deg:.1f} degrees, "
        f"saturation x{sat_mul:.2f}, value x{val_mul:.2f}. "
        "Do not alter geometry, text content, or background."
    )
