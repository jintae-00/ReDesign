#!/usr/bin/env python3
"""Task candidate counting and seed-based sampling."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from .common_utils import sample_with_seed


def summarize_task_capacity(candidates: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    c = Counter()
    total = 0
    for item in candidates:
        total += 1
        c[item.get("task_type", "unknown")] += 1
    return {
        "total": total,
        "by_task_type": dict(c),
    }


def select_tasks(candidates: List[Dict[str, Any]], max_tasks: Optional[int], seed: int) -> List[Dict[str, Any]]:
    return sample_with_seed(candidates, max_tasks, seed)
