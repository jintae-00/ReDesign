#!/usr/bin/env python3
"""Data loading wrappers for editability evaluation.

Reuses existing loader functions from evaluation_figma.py without modifying it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common_utils import load_json, parse_exp_pairs


@dataclass
class EpisodeTask:
    episode_id: str
    split_name: str
    split_dir: Path
    gt_json_path: Path
    agent_episode_dir: Optional[Path]
    qwen_episode_dir: Optional[Path]


_EVAL_EXTRACTORS: Optional[Tuple[Any, Any, Any]] = None


def _resolve_eval_extractors() -> Tuple[Any, Any, Any]:
    global _EVAL_EXTRACTORS
    if _EVAL_EXTRACTORS is not None:
        return _EVAL_EXTRACTORS

    try:
        mod = importlib.import_module("evaluation_figma")
    except Exception as e:
        raise ImportError(
            "Failed to import evaluation_figma. "
            "Check runtime env/dependencies (e.g., cv2) and PYTHONPATH."
        ) from e

    def _pick(required: str, aliases: Sequence[str]) -> Any:
        for name in (required, *aliases):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
        available = sorted(
            n for n in dir(mod) if n.startswith("extract_") and ("element" in n or "gt" in n)
        )
        raise ImportError(
            f"evaluation_figma is missing extractor '{required}'. "
            f"looked_for={[required, *aliases]} available={available[:20]}"
        )

    extract_gt = _pick("extract_gt_elements", ("extract_gt",))
    extract_agent = _pick("extract_agent_elements", ("extract_agent",))
    extract_qwen = _pick("extract_qwen_elements_cca", ("extract_qwen_elements", "extract_qwen"))

    _EVAL_EXTRACTORS = (extract_gt, extract_agent, extract_qwen)
    return _EVAL_EXTRACTORS


def _detect_gt_splits(figma_data_dir: Path, gt_prefix: str) -> List[Tuple[str, Path]]:
    subset_base = figma_data_dir / "process" / "subset"
    gt_split_dirs = sorted(subset_base.glob(f"{gt_prefix}_split_*"))
    out: List[Tuple[str, Path]] = []
    for d in gt_split_dirs:
        suffix = d.name[len(gt_prefix) + 1 :]
        out.append((suffix, d))
    return out


def _scan_agent(agent_exp_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    if not agent_exp_dir.exists():
        return found
    for subdir in sorted(agent_exp_dir.iterdir()):
        if not subdir.is_dir():
            continue
        episodes_dir = subdir / "episodes"
        if not episodes_dir.exists():
            continue
        for ep_dir in sorted(episodes_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "parse.json").exists():
                found[ep_dir.name] = ep_dir
    return found


def _scan_qwen(qwen_exp_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    if not qwen_exp_dir.exists():
        return found
    for subdir in sorted(qwen_exp_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for ep_dir in sorted(subdir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "layer_00.png").exists():
                found[ep_dir.name] = ep_dir
    return found


def collect_episode_tasks(
    figma_data_dir: Path,
    exp_pairs: Sequence[str],
    model: str,
    max_episodes: Optional[int] = None,
) -> List[EpisodeTask]:
    """Collect GT + model episode tuples for matching.

    model: "agent" or "qwen"
    """
    if model not in {"agent", "qwen"}:
        raise ValueError(f"Unsupported model: {model}")

    pairs = parse_exp_pairs(exp_pairs)
    task_map: Dict[str, EpisodeTask] = {}

    for agent_exp_dir, qwen_exp_dir, gt_prefix in pairs:
        gt_splits = _detect_gt_splits(figma_data_dir, gt_prefix)
        if not gt_splits:
            continue

        gt_map: Dict[str, Dict[str, Any]] = {}
        for split_name, split_dir in gt_splits:
            valid_frames_dir = split_dir / "valid_frames"
            if not valid_frames_dir.exists():
                continue
            for gt_json in valid_frames_dir.glob("*.json"):
                gt_map[gt_json.stem] = {
                    "split_name": split_name,
                    "split_dir": split_dir,
                    "gt_json_path": gt_json,
                }

        if model == "agent":
            model_map = _scan_agent(agent_exp_dir)
        else:
            model_map = _scan_qwen(qwen_exp_dir)

        common = set(gt_map.keys()) & set(model_map.keys())
        for eid in common:
            if eid in task_map:
                continue
            gt_info = gt_map[eid]
            task_map[eid] = EpisodeTask(
                episode_id=eid,
                split_name=gt_info["split_name"],
                split_dir=gt_info["split_dir"],
                gt_json_path=gt_info["gt_json_path"],
                agent_episode_dir=model_map[eid] if model == "agent" else None,
                qwen_episode_dir=model_map[eid] if model == "qwen" else None,
            )

    tasks = [task_map[k] for k in sorted(task_map.keys())]
    if max_episodes is not None:
        tasks = tasks[: max_episodes]
    return tasks


def _attach_gt_metadata(gt_elements: List[Dict[str, Any]], gt_json_path: Path) -> None:
    frame = load_json(gt_json_path)
    unit_map: Dict[str, Dict[str, Any]] = {}
    for unit in frame.get("unit_images", []):
        unit_id = str(unit.get("unit_id", ""))
        if unit_id:
            unit_map[unit_id] = unit

    for elem in gt_elements:
        elem.setdefault("meta", {})
        elem_id = str(elem.get("id", ""))
        if elem_id.startswith("gt_"):
            unit_id = elem_id[3:]
            if unit_id in unit_map:
                elem["meta"]["gt_unit"] = unit_map[unit_id]


def _attach_agent_metadata(agent_elements: List[Dict[str, Any]], agent_episode_dir: Path) -> None:
    parse_data = load_json(agent_episode_dir / "parse.json")
    raw_map: Dict[str, Dict[str, Any]] = {str(e.get("id", "")): e for e in parse_data.get("elements", [])}

    for elem in agent_elements:
        elem.setdefault("meta", {})
        eid = str(elem.get("id", ""))
        if eid.startswith("agent_"):
            core_id = eid[len("agent_") :]
            raw = raw_map.get(core_id)
            if raw:
                elem["meta"]["parsed"] = raw


def load_episode_elements(task: EpisodeTask, model: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Tuple[int, int]]:
    extract_gt_elements, extract_agent_elements, extract_qwen_elements_cca = _resolve_eval_extractors()

    gt_elements, canvas_size, _ = extract_gt_elements(task.gt_json_path, task.split_dir, logger=None)
    _attach_gt_metadata(gt_elements, task.gt_json_path)

    if model == "agent":
        if task.agent_episode_dir is None:
            raise ValueError(f"Episode {task.episode_id} has no agent dir")
        pred_elements = extract_agent_elements(
            task.agent_episode_dir,
            canvas_size,
            apply_alpha_correction=True,
            text_refinement=True,
            logger=None,
        )
        _attach_agent_metadata(pred_elements, task.agent_episode_dir)
    elif model == "qwen":
        if task.qwen_episode_dir is None:
            raise ValueError(f"Episode {task.episode_id} has no qwen dir")
        pred_elements = extract_qwen_elements_cca(task.qwen_episode_dir, canvas_size, logger=None)
        # qwen CCA has no text metadata by design.
        for elem in pred_elements:
            elem.setdefault("meta", {})
    else:
        raise ValueError(f"Unsupported model: {model}")

    return gt_elements, pred_elements, canvas_size
