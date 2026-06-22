#!/usr/bin/env python3
"""Shared visualization helpers for matching outputs."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .matching_core import _pred_union_rgba_and_mask


def _fmt_metric(v: Any) -> str:
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "nan"


def _rgba_to_checker_rgb(rgba: np.ndarray, tile: int = 10) -> np.ndarray:
    h, w = rgba.shape[:2]
    yy, xx = np.indices((h, w))
    checker = (((xx // tile) + (yy // tile)) % 2).astype(np.uint8)
    bg = np.empty((h, w, 3), dtype=np.float32)
    bg[checker == 0] = (242, 242, 242)
    bg[checker == 1] = (214, 214, 214)

    rgb = rgba[..., :3].astype(np.float32)
    alpha = (rgba[..., 3:4].astype(np.float32) / 255.0)
    out = rgb * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _intersection_rgba(gt_rgba: np.ndarray, pred_rgba: np.ndarray) -> np.ndarray:
    gt_a = gt_rgba[..., 3] > 0
    pr_a = pred_rgba[..., 3] > 0
    inter = gt_a & pr_a
    out = np.zeros_like(gt_rgba)
    if inter.any():
        gt_rgb = gt_rgba[..., :3].astype(np.float32)
        pr_rgb = pred_rgba[..., :3].astype(np.float32)
        mix = ((gt_rgb + pr_rgb) * 0.5).astype(np.uint8)
        out[..., :3][inter] = mix[inter]
        out[..., 3][inter] = 255
    return out


def _panel_image(rgba: np.ndarray, panel_size: Tuple[int, int]) -> Image.Image:
    rgb = _rgba_to_checker_rgb(rgba)
    return Image.fromarray(rgb, mode="RGB").resize(panel_size, resample=Image.BILINEAR)


def _pred_union(
    pred_elements: Sequence[Dict[str, Any]],
    selected_indices: Sequence[int],
    canvas_size: Tuple[int, int],
) -> np.ndarray:
    h, w = canvas_size[1], canvas_size[0]
    if not selected_indices:
        return np.zeros((h, w, 4), dtype=np.uint8)
    preds: List[Dict[str, Any]] = []
    for i in selected_indices:
        if 0 <= i < len(pred_elements):
            preds.append(pred_elements[i])
    if not preds:
        return np.zeros((h, w, 4), dtype=np.uint8)
    pred_rgba, _ = _pred_union_rgba_and_mask(preds, canvas_size)
    return pred_rgba


def _collect_rows(
    payload: Dict[str, Any],
    gt_elements: Sequence[Dict[str, Any]],
    pred_elements: Sequence[Dict[str, Any]],
    canvas_size: Tuple[int, int],
    max_rows: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    matches = payload.get("matches", [])
    matches = sorted(matches, key=lambda x: int(x.get("gt_index", 0)))
    if max_rows > 0:
        matches = matches[:max_rows]

    for row_idx, m in enumerate(matches, start=1):
        gt_idx = int(m.get("gt_index", -1))
        if not (0 <= gt_idx < len(gt_elements)):
            continue
        gt = gt_elements[gt_idx]
        gt_rgba = np.array(gt["image"].convert("RGBA"), dtype=np.uint8)
        selected = list(m.get("selected_pred_indices", []))
        pred_rgba = _pred_union(pred_elements, selected, canvas_size)
        inter_rgba = _intersection_rgba(gt_rgba, pred_rgba)
        rows.append(
            {
                "row_idx": row_idx,
                "match": m,
                "gt_rgba": gt_rgba,
                "pred_rgba": pred_rgba,
                "inter_rgba": inter_rgba,
            }
        )
    return rows


def _draw_row(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    row: Dict[str, Any],
    y: int,
    margin: int,
    row_text_w: int,
    panel_width: int,
    panel_gap: int,
    panel_h: int,
    font: ImageFont.ImageFont,
) -> None:
    m = row["match"]
    gt_panel = _panel_image(row["gt_rgba"], (panel_width, panel_h))
    pred_panel = _panel_image(row["pred_rgba"], (panel_width, panel_h))
    inter_panel = _panel_image(row["inter_rgba"], (panel_width, panel_h))

    img_w = canvas.size[0]
    row_h = panel_h + 18
    x0 = margin
    draw.rectangle((x0, y, img_w - margin, y + row_h), outline=(228, 228, 228), width=1)

    selected = list(m.get("selected_pred_indices", []))
    pred_ids = m.get("selected_pred_ids", [])
    pred_ids_short = ",".join(str(v) for v in pred_ids[:6])
    if len(pred_ids) > 6:
        pred_ids_short += ",..."
    metrics = m.get("merged_metrics", {})
    text_lines = [
        f"[{row['row_idx']}] gt_idx={m.get('gt_index')} gt_id={m.get('gt_id')} gt_type={m.get('gt_type')}",
        f"pred_idx={selected} pred_ids=[{pred_ids_short}]",
        f"cost={_fmt_metric(metrics.get('cost'))}  l1={_fmt_metric(metrics.get('l1'))}  cand={m.get('num_candidates', 0)}",
    ]
    ty = y + 10
    for line in text_lines:
        draw.text((x0 + 10, ty), line, fill=(30, 30, 30), font=font)
        ty += 18

    px = margin + row_text_w
    py = y + 8
    canvas.paste(gt_panel, (px, py))
    canvas.paste(pred_panel, (px + panel_width + panel_gap, py))
    canvas.paste(inter_panel, (px + (panel_width + panel_gap) * 2, py))


def save_match_visualizations(
    episode_id: str,
    split_name: str,
    payload: Dict[str, Any],
    gt_elements: Sequence[Dict[str, Any]],
    pred_elements: Sequence[Dict[str, Any]],
    canvas_size: Tuple[int, int],
    episode_out_path: Optional[Path],
    pair_out_dir: Optional[Path],
    parsed_layers_src_dir: Optional[Path] = None,
    max_rows: int = 80,
    panel_width: int = 220,
) -> Dict[str, Any]:
    font = ImageFont.load_default()
    cw, ch = canvas_size
    panel_h = max(96, int(round(panel_width * ch / max(1, cw))))
    row_gap = 10
    row_text_w = 520
    panel_gap = 12
    margin = 16
    row_h = panel_h + 18
    rows = _collect_rows(payload, gt_elements, pred_elements, canvas_size, max_rows=max_rows)

    out: Dict[str, Any] = {
        "rows_rendered": int(len(rows)),
        "episode_image_path": None,
        "pair_image_dir": None,
        "pair_image_count": 0,
        "copied_layer_png_count": 0,
    }

    if episode_out_path is not None:
        header_h = 74
        img_w = margin * 2 + row_text_w + (panel_width * 3) + (panel_gap * 2)
        img_h = margin * 2 + header_h + (row_h + row_gap) * len(rows)
        canvas = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        counts = payload.get("counts", {})
        header_lines = [
            f"episode={episode_id} split={split_name} canvas={cw}x{ch}",
            f"gt={counts.get('gt', len(gt_elements))} pred={counts.get('pred', len(pred_elements))} matches={counts.get('matches', len(rows))}",
            "panel: GT | merged Pred | intersection",
        ]
        y = margin
        for line in header_lines:
            draw.text((margin, y), line, fill=(0, 0, 0), font=font)
            y += 18
        y = margin + header_h
        for row in rows:
            _draw_row(
                draw=draw,
                canvas=canvas,
                row=row,
                y=y,
                margin=margin,
                row_text_w=row_text_w,
                panel_width=panel_width,
                panel_gap=panel_gap,
                panel_h=panel_h,
                font=font,
            )
            y += row_h + row_gap
        episode_out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(episode_out_path)
        out["episode_image_path"] = str(episode_out_path.resolve())

    if pair_out_dir is not None:
        pair_out_dir.mkdir(parents=True, exist_ok=True)
        pair_count = 0
        for row in rows:
            img_w = margin * 2 + row_text_w + (panel_width * 3) + (panel_gap * 2)
            img_h = margin * 2 + row_h
            canvas = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            _draw_row(
                draw=draw,
                canvas=canvas,
                row=row,
                y=margin,
                margin=margin,
                row_text_w=row_text_w,
                panel_width=panel_width,
                panel_gap=panel_gap,
                panel_h=panel_h,
                font=font,
            )
            gt_idx = int(row["match"].get("gt_index", row["row_idx"]))
            out_path = pair_out_dir / f"gt_{gt_idx:04d}.png"
            canvas.save(out_path)
            pair_count += 1

        copied_layers = 0
        if parsed_layers_src_dir is not None and parsed_layers_src_dir.exists():
            for src in sorted(parsed_layers_src_dir.glob("layer_*.png")):
                if not src.is_file():
                    continue
                shutil.copy2(src, pair_out_dir / src.name)
                copied_layers += 1
        out["pair_image_dir"] = str(pair_out_dir.resolve())
        out["pair_image_count"] = int(pair_count)
        out["copied_layer_png_count"] = int(copied_layers)

    return out
