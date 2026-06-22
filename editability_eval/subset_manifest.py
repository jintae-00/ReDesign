#!/usr/bin/env python3
"""Subset manifest helpers shared by editability runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from .common_utils import load_json


SubsetKey = Tuple[str, int]


def _parse_key_str(s: str) -> Optional[SubsetKey]:
    if "::" not in s:
        return None
    eid, idx = s.rsplit("::", 1)
    try:
        return eid, int(idx)
    except Exception:
        return None


def _keys_from_items(items: Iterable[Dict[str, Any]]) -> Set[SubsetKey]:
    out: Set[SubsetKey] = set()
    for item in items:
        eid = item.get("episode_id")
        gt_idx = item.get("gt_index")
        if eid is None or gt_idx is None:
            continue
        try:
            out.add((str(eid), int(gt_idx)))
        except Exception:
            continue
    return out


def load_subset_keys(manifest_path: Optional[Path], category: str) -> Optional[Set[SubsetKey]]:
    """Load allowed (episode_id, gt_index) keys from a category subset manifest."""
    if manifest_path is None:
        return None

    payload = load_json(manifest_path)
    categories = payload.get("categories", {})
    cat_payload = categories.get(category, {})
    keys: Set[SubsetKey] = set()

    # Primary: explicit item list.
    items = cat_payload.get("items", [])
    if isinstance(items, list):
        keys |= _keys_from_items(items)

    # Optional compact form: ["episode::gt_idx", ...]
    key_strs = cat_payload.get("keys", [])
    if isinstance(key_strs, list):
        for s in key_strs:
            if not isinstance(s, str):
                continue
            parsed = _parse_key_str(s)
            if parsed is not None:
                keys.add(parsed)

    # Backward compatibility: category payload itself may be a list of items.
    if not keys and isinstance(cat_payload, list):
        keys |= _keys_from_items(cat_payload)

    return keys

