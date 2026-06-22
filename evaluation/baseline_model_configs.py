#!/usr/bin/env python3
"""Shared model configurations and scanning utilities for baseline evaluation.

Provides directory scanning functions and extraction format mappings for
3 new baselines (layered, multi_tools, sparse_verif) alongside existing
agent and qwen models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Model configuration registry
# ---------------------------------------------------------------------------
# format: "qwen" → extract_qwen_elements_cca,  "agent" → extract_agent_elements

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "layered": {
        "format": "qwen",
        "description": "LayerD baseline (layer_XX.png format)",
    },
    "multi_tools": {
        "format": "agent",
        "description": "Rule-based multi-tools pipeline (parse.json + elements/)",
    },
    "sparse_verif": {
        "format": "agent",
        "description": "Sparse verification agent (parse.json + elements/)",
    },
    "simple_verif": {
        "format": "agent",
        "description": "Simple reconstruction verifier (parse.json + elements/)",
    },
    "agent": {
        "format": "agent",
        "description": "Main agent (parse.json + elements/)",
    },
    "qwen": {
        "format": "qwen",
        "description": "Qwen baseline (layer_XX.png / CCA format)",
    },
    "omnisvg": {
        "format": "omnisvg",
        "description": "OmniSVG image-to-SVG baseline (output.svg per episode)",
    },
    "vtracer": {
        "format": "omnisvg",
        "description": "VTracer image-to-SVG baseline (output.svg per episode)",
    },
}


# ---------------------------------------------------------------------------
# Directory scanning functions
# ---------------------------------------------------------------------------

def scan_layered(base_dir: Path) -> Dict[str, Path]:
    """Scan the layered baseline output for episodes with layer_00.png.

    Supports the flat output of run_layerd_*.py ({base_dir}/{episode_id}/) and the
    legacy {base_dir}/all_splits/{episode_id}/ layout.
    """
    found: Dict[str, Path] = {}
    roots = [base_dir / "all_splits", base_dir]
    for root in roots:
        if not root.exists():
            continue
        for ep_dir in sorted(root.iterdir()):
            if ep_dir.is_dir() and ep_dir.name != "all_splits" \
                    and (ep_dir / "layer_00.png").exists():
                found.setdefault(ep_dir.name, ep_dir)
    return found


def scan_multi_tools(base_dir: Path) -> Dict[str, Path]:
    """Scan the multi-tools baseline output for episodes with parse.json.

    Supports the flat output of run_multi_tools_*.py ({base_dir}/episodes/{id}/)
    and the legacy {base_dir}/all_splits/episodes/{id}/ layout.
    """
    found: Dict[str, Path] = {}
    roots = [base_dir / "all_splits" / "episodes", base_dir / "episodes"]
    for episodes_dir in roots:
        if not episodes_dir.exists():
            continue
        for ep_dir in sorted(episodes_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "parse.json").exists():
                found.setdefault(ep_dir.name, ep_dir)
    return found


def scan_sparse_verif(base_dir: Path) -> Dict[str, Path]:
    """Scan baseline_sparse_verification_agent_experiment/episodes/{episode_id}/ for parse.json."""
    found: Dict[str, Path] = {}
    episodes_dir = base_dir / "episodes"
    if not episodes_dir.exists():
        return found
    for ep_dir in sorted(episodes_dir.iterdir()):
        if ep_dir.is_dir() and (ep_dir / "parse.json").exists():
            found[ep_dir.name] = ep_dir
    return found


def scan_simple_verif(base_dir: Path) -> Dict[str, Path]:
    """Scan baseline_simple_recon_verifier_experiment/episodes/{episode_id}/ for parse.json.

    The root parse.json contains ``_simple_recon_verif_meta.final_try`` which
    indicates the best try directory (e.g. ``try_2``).  We return that try
    directory so that ``extract_agent_elements`` can find ``history_tree.json``
    and ``elements/`` inside it.
    """
    import json as _json

    found: Dict[str, Path] = {}
    episodes_dir = base_dir / "episodes"
    if not episodes_dir.exists():
        return found
    for ep_dir in sorted(episodes_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        root_parse = ep_dir / "parse.json"
        if not root_parse.exists():
            continue
        # Determine the best try directory
        try:
            data = _json.loads(root_parse.read_text(encoding="utf-8"))
            final_try = data.get("_simple_recon_verif_meta", {}).get("final_try")
        except Exception:
            final_try = None
        if final_try:
            try_dir = ep_dir / final_try
            if try_dir.is_dir() and (try_dir / "parse.json").exists():
                found[ep_dir.name] = try_dir
                continue
        # Fallback: pick the highest-numbered try_N that has parse.json
        try_dirs = sorted(ep_dir.glob("try_*"), reverse=True)
        for td in try_dirs:
            if td.is_dir() and (td / "parse.json").exists():
                found[ep_dir.name] = td
                break
    return found


def scan_agent(base_dir: Path) -> Dict[str, Path]:
    """Scan agent experiment directories for episodes with parse.json.

    Supports two layouts:
      1. {base_dir}/episodes/{episode_id}/parse.json
      2. {base_dir}/split_*/episodes/{episode_id}/parse.json
    """
    found: Dict[str, Path] = {}
    # Layout 1: direct episodes/ under base_dir
    episodes_dir = base_dir / "episodes"
    if episodes_dir.exists():
        for ep_dir in sorted(episodes_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "parse.json").exists():
                found[ep_dir.name] = ep_dir
    # Layout 2: split_*/episodes/ under base_dir
    for split_dir in sorted(base_dir.glob("split_*")):
        episodes_dir = split_dir / "episodes"
        if not episodes_dir.exists():
            continue
        for ep_dir in sorted(episodes_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "parse.json").exists():
                if ep_dir.name not in found:
                    found[ep_dir.name] = ep_dir
    return found


def scan_qwen(base_dir: Path) -> Dict[str, Path]:
    """Scan qwen experiment directory for episodes with layer_00.png.

    Supports two layouts:
      1. {base_dir}/{episode_id}/layer_00.png
      2. {base_dir}/split_*/{episode_id}/layer_00.png
    """
    found: Dict[str, Path] = {}
    if not base_dir.exists():
        return found
    # Layout 1: direct episode dirs under base_dir
    for ep_dir in sorted(base_dir.iterdir()):
        if ep_dir.is_dir() and (ep_dir / "layer_00.png").exists():
            found[ep_dir.name] = ep_dir
    # Layout 2: split_*/{episode_id}/ under base_dir
    for split_dir in sorted(base_dir.glob("split_*")):
        if not split_dir.is_dir():
            continue
        for ep_dir in sorted(split_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "layer_00.png").exists():
                if ep_dir.name not in found:
                    found[ep_dir.name] = ep_dir
    # Layout 3: all_splits/{episode_id}/ under base_dir
    all_splits = base_dir / "all_splits"
    if all_splits.exists():
        for ep_dir in sorted(all_splits.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "layer_00.png").exists():
                if ep_dir.name not in found:
                    found[ep_dir.name] = ep_dir
    return found


def scan_omnisvg(base_dir: Path) -> Dict[str, Path]:
    """Scan OmniSVG experiment directories for episodes with output.svg.

    Supports two layouts:
      1. {base_dir}/all_splits/{episode_id}/output.svg  (Figma)
      2. {base_dir}/{episode_id}/output.svg              (Crello / direct)
    """
    found: Dict[str, Path] = {}
    # Layout 1: all_splits/{eid}/
    all_splits = base_dir / "all_splits"
    if all_splits.exists():
        for ep_dir in sorted(all_splits.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "output.svg").exists():
                found[ep_dir.name] = ep_dir
    # Layout 2: direct {eid}/
    if base_dir.exists():
        for ep_dir in sorted(base_dir.iterdir()):
            if ep_dir.is_dir() and ep_dir.name != "all_splits" and (ep_dir / "output.svg").exists():
                if ep_dir.name not in found:
                    found[ep_dir.name] = ep_dir
    return found


# Registry of scan functions per model name
SCAN_FUNCTIONS = {
    "layered": scan_layered,
    "multi_tools": scan_multi_tools,
    "sparse_verif": scan_sparse_verif,
    "simple_verif": scan_simple_verif,
    "agent": scan_agent,
    "qwen": scan_qwen,
    "omnisvg": scan_omnisvg,
    "vtracer": None,  # defined below
}


def scan_vtracer(base_dir: Path) -> Dict[str, Path]:
    """Scan vtracer experiment directories for episodes with output.svg.

    Supports layouts:
      1. {base_dir}/figma/{episode_id}/output.svg
      2. {base_dir}/crello/{episode_id}/output.svg
      3. {base_dir}/all_splits/{episode_id}/output.svg  (legacy)
      4. {base_dir}/{episode_id}/output.svg              (direct)
    """
    found: Dict[str, Path] = {}
    for subdir_name in ("figma", "crello", "all_splits"):
        subdir = base_dir / subdir_name
        if subdir.exists():
            for ep_dir in sorted(subdir.iterdir()):
                if ep_dir.is_dir() and (ep_dir / "output.svg").exists():
                    if ep_dir.name not in found:
                        found[ep_dir.name] = ep_dir
    # Direct layout fallback
    if base_dir.exists():
        for ep_dir in sorted(base_dir.iterdir()):
            if ep_dir.is_dir() and ep_dir.name not in ("figma", "crello", "all_splits") \
                    and (ep_dir / "output.svg").exists():
                if ep_dir.name not in found:
                    found[ep_dir.name] = ep_dir
    return found


SCAN_FUNCTIONS["vtracer"] = scan_vtracer


def scan_model_episodes(model_name: str, base_dir: Path) -> Dict[str, Path]:
    """Scan episodes for a baseline model by name."""
    fn = SCAN_FUNCTIONS.get(model_name)
    if fn is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(SCAN_FUNCTIONS)}")
    return fn(base_dir)


# ---------------------------------------------------------------------------
# GT episode collection (reuses evaluation_figma.py logic)
# ---------------------------------------------------------------------------

def collect_gt_episodes(
    figma_data_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Scan GT episodes from a flat Figma dataset directory.

    Expects the released flat layout::

        {figma_data_dir}/valid_frames/*.json

    All episodes live in one directory; per-episode JSON ``unit_images_dir``
    paths are relative to ``figma_data_dir``. Every episode is collected.

    Returns dict: episode_id -> {gt_json_path, split_name, split_dir}
    (``split_name`` is hardcoded to "merged" for downstream compatibility.)
    """
    gt_map: Dict[str, Dict[str, Any]] = {}

    flat_valid_frames = figma_data_dir / "valid_frames"
    if not flat_valid_frames.is_dir():
        print(f"[WARNING] No valid_frames directory found at {flat_valid_frames}")
        return gt_map

    for gt_json in sorted(flat_valid_frames.glob("*.json")):
        if gt_json.stem not in gt_map:
            gt_map[gt_json.stem] = {
                "gt_json_path": gt_json,
                "split_name": "merged",
                "split_dir": figma_data_dir,
            }
    return gt_map


def get_common_episodes(
    gt_map: Dict[str, Any],
    model_map: Dict[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """Intersect GT and model episodes, returning task info per episode."""
    common = set(gt_map.keys()) & set(model_map.keys())
    task_info: Dict[str, Dict[str, Any]] = {}
    for eid in sorted(common):
        gt = gt_map[eid]
        task_info[eid] = {
            "episode_id": eid,
            "gt_json_path": gt["gt_json_path"],
            "split_name": gt["split_name"],
            "split_dir": gt["split_dir"],
            "model_dir": model_map[eid],
        }
    return task_info


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def add_baseline_dir_args(parser):
    """Add directory arguments for all models."""
    parser.add_argument("--layered-dir", type=str, default="./baseline_layerd_experiment",
                        help="Root directory for layered baseline experiment")
    parser.add_argument("--multi-tools-dir", type=str, default="./baseline_muilti_tools_experiment",
                        help="Root directory for multi-tools baseline experiment")
    parser.add_argument("--sparse-verif-dir", type=str, default="./baseline_sparse_verification_agent_experiment",
                        help="Root directory for sparse verification agent baseline experiment")
    parser.add_argument("--simple-verif-dir", type=str, default="./baseline_simple_recon_verifier_experiment",
                        help="Root directory for simple reconstruction verifier baseline experiment")
    parser.add_argument("--omnisvg-dir", type=str, default="./baseline_omnisvg_experiment",
                        help="Root directory for OmniSVG baseline experiment")
    parser.add_argument("--vtracer-dir", type=str, default="./baseline_vtracer_experiment",
                        help="Root directory for VTracer baseline experiment")
    parser.add_argument("--agent-dir", type=str, default=None,
                        help="Agent inference output directory (episodes/<id>/parse.json)")
    parser.add_argument("--qwen-dir", type=str, default=None,
                        help="Qwen inference output directory (<id>/layer_00.png)")


def _resolve_multi_dirs(args, model_name: str) -> Optional[List[Path]]:
    """Resolve the inference output directory for agent / qwen models.

    Reads the direct ``--agent-dir`` / ``--qwen-dir`` arguments. Returns a
    single-element list (kept as a list for ``scan_model_episodes_multi``).
    """
    if model_name == "agent":
        d = getattr(args, "agent_dir", None)
        if d:
            return [Path(d)]
    elif model_name == "qwen":
        d = getattr(args, "qwen_dir", None)
        if d:
            return [Path(d)]
    return None


def scan_model_episodes_multi(model_name: str, dirs: List[Path]) -> Dict[str, Path]:
    """Scan episodes from multiple directories (for agent/qwen)."""
    found: Dict[str, Path] = {}
    fn = SCAN_FUNCTIONS.get(model_name)
    if fn is None:
        raise ValueError(f"Unknown model: {model_name}")
    for d in dirs:
        found.update(fn(d))
    return found


def get_model_dir(args, model_name: str) -> Path:
    """Get the base directory for a model from parsed args."""
    dir_map = {
        "layered": getattr(args, "layered_dir", None),
        "multi_tools": getattr(args, "multi_tools_dir", None),
        "sparse_verif": getattr(args, "sparse_verif_dir", None),
        "simple_verif": getattr(args, "simple_verif_dir", None),
        "omnisvg": getattr(args, "omnisvg_dir", None),
        "vtracer": getattr(args, "vtracer_dir", None),
    }
    d = dir_map.get(model_name)
    if d is None:
        raise ValueError(f"No single directory for model '{model_name}'. "
                         f"Use _resolve_multi_dirs() for agent/qwen.")
    return Path(d)
