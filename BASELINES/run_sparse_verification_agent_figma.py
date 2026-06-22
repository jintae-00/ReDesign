#!/usr/bin/env python3
"""
run_baseline3_figma.py - Baseline 3: Agent with Sparse Verification

Same agent pipeline as REDESIGN (full URLD), but with sparse verification:
- Verify every N generations (default: 3) instead of every action
- CCA and Split_Text are verification barriers
- Leaf images batch-verified against sparse ancestor
- On fail: fallback to ancestor node with full action chain context

Output Format: parse.json + elements/ (Agent 호환)
Evaluation: editability_eval → extract_agent_elements()

Usage:
    python run_baseline3_figma.py --qwen_gpus 0,1 --tool_gpus 2,3 --objectclear_gpu 3
    python run_baseline3_figma.py --qwen_gpus 0,1 --tool_gpus 2,3 --limit 5 --dry_run

How it works:
    1. Monkey-patches REDESIGN.episode_run._get_node_function to use
       REDESIGN_ablation.nodes.stack_manager instead of the original
    2. Iterates over ALL dataset splits (dino80 splits 0-3 + dino90 splits 0-4 = 909 frames)
    3. Resume mode: skips already-completed frames (parse.json exists)
    4. Output directory: baseline3_experiment/episodes/{frame_id}/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging


# =============================================================================
# Configuration
# =============================================================================

FIGMA_DATA_BASE = "figma_data/process/subset"
BASELINE3_EXPERIMENT_BASE = "baseline_sparse_verification_agent_experiment"

# All dataset splits: (prefix, num_splits) — dino80 (435) + dino90 (474) = 909 frames
ALL_DATASET_SPLITS = [
    ("dino80_obj_5_60_char_25_split_", 4),   # splits 0-3
    ("dino90_obj_5_25_char_50_split_", 5),   # splits 0-4
]

DEFAULT_WORKERS = 6
DEFAULT_LLM_LIMIT = 100
DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_LAYERS = 100


# =============================================================================
# Monkey-patch for sparse verification
# =============================================================================

def apply_sparse_verification_patch(sparse_period: int = 3):
    """
    Monkey-patch REDESIGN modules with ablation overrides:

    1. stack_manager → REDESIGN_ablation.nodes.stack_manager
       - [1] Leaf nodes not marked verified
       - [2] Barrier node handling (CCA, Split_Text)
       - [5] Empty image filtering
       - [6][7] Verification failure cleanup

    2. finalize_obj → REDESIGN_ablation.nodes.finalize_obj
       - [3] Residual verification at finalize time
       - [4] Single-node auto-verify

    2b. finalize_text → REDESIGN_ablation.nodes.finalize_text
       - [3][4] Same residual verification for text nodes

    3. merge_updates / r_pack_state → extended for new special keys
       - _set_parsed_elements: replace parsed_elements entirely
       - _remove_from_queue_ids: filter specific ids from layer_queue

    4. visualizer → extended for sparse_failed visualization
       - [7] Red X overlay, dashed borders for failed nodes

    Args:
        sparse_period: Verify every N generations (2 or 3).
    """
    import REDESIGN.episode_run as episode_module
    import REDESIGN.reducers as reducers_module
    from REDESIGN_ablation.nodes.stack_manager import set_sparse_period

    # Set the verification interval
    set_sparse_period(sparse_period)

    # =========================================================================
    # 1. Patch _get_node_function: stack_manager + finalize_obj
    # =========================================================================
    _original_get_node_function = episode_module._get_node_function

    def _patched_get_node_function(node_name: str):
        if node_name == "stack_manager":
            from REDESIGN_ablation.nodes.stack_manager import node
            return node
        if node_name == "finalize_obj":
            from REDESIGN_ablation.nodes.finalize_obj import node
            return node
        if node_name == "finalize_text":
            from REDESIGN_ablation.nodes.finalize_text import node
            return node
        return _original_get_node_function(node_name)

    episode_module._get_node_function = _patched_get_node_function

    # =========================================================================
    # 2. Patch _force_finalize_layer to use ablation finalize_obj
    # =========================================================================
    _original_force_finalize = episode_module._force_finalize_layer

    def _patched_force_finalize_layer(layer_id, state):
        from REDESIGN_ablation.nodes.finalize_obj import node as finalize_obj_node
        from REDESIGN.reducers import merge_updates, r_pack_state

        history_tree = state.get("history_tree", {})
        if layer_id not in history_tree:
            return {}

        node_data = history_tree[layer_id]
        original_action = node_data.get("action_type", "Unknown")

        action_update = {
            "history_tree": {
                layer_id: {
                    "action_type": "Finalize_Obj",
                    "action_reasoning": f"Force finalized due to termination (original: {original_action})",
                    "node_queue": [],
                }
            }
        }

        temp_state = merge_updates(state, action_update)
        temp_state["current_layer_id"] = layer_id

        try:
            finalize_update = finalize_obj_node(temp_state)
        except Exception as e:
            print(f"[ForceFinalize] ERROR in finalize_obj for {layer_id}: {e}")
            return action_update

        return r_pack_state(state, action_update, finalize_update)

    episode_module._force_finalize_layer = _patched_force_finalize_layer

    # =========================================================================
    # 3. Patch merge_updates to handle new special keys
    # =========================================================================
    _original_merge_updates = reducers_module.merge_updates

    def _patched_merge_updates(state, *updates):
        # Pre-process: extract special ablation keys and apply them to
        # a mutable copy of state BEFORE the original merge_updates runs.
        pre_state = dict(state)
        cleaned_updates = []

        for upd in updates:
            if not upd:
                cleaned_updates.append(upd)
                continue
            new_upd = dict(upd)

            if "_set_parsed_elements" in new_upd:
                # Legacy compat: full replacement from snapshot
                pre_state["parsed_elements"] = new_upd.pop("_set_parsed_elements")

            if "_remove_parsed_element_layer_ids" in new_upd:
                # Concurrency-safe: filter by source_layer_id on pre_state
                lids = new_upd.pop("_remove_parsed_element_layer_ids")
                pre_state["parsed_elements"] = [
                    e for e in pre_state.get("parsed_elements", [])
                    if not (isinstance(e, dict) and e.get("source_layer_id", "") in lids)
                ]

            if "_remove_from_queue_ids" in new_upd:
                ids_to_remove = new_upd.pop("_remove_from_queue_ids")
                queue = list(pre_state.get("layer_queue", []))
                pre_state["layer_queue"] = [lid for lid in queue if lid not in ids_to_remove]

            cleaned_updates.append(new_upd)

        return _original_merge_updates(pre_state, *cleaned_updates)

    reducers_module.merge_updates = _patched_merge_updates

    # Also patch the module-level reference in episode_run if it imported merge_updates
    if hasattr(episode_module, 'merge_updates'):
        episode_module.merge_updates = _patched_merge_updates

    # =========================================================================
    # 4. Patch StateManager.update() to handle _remove_from_queue_ids / _set_parsed_elements
    # =========================================================================
    from REDESIGN.episode_run import StateManager

    _original_sm_update = StateManager.update

    def _patched_sm_update(self, *updates):
        """
        Extended StateManager.update that handles ablation-specific keys:

        - _remove_from_queue_ids: set of layer_ids to filter out of queue
          (concurrency-safe: filters the live queue at apply time)
        - _remove_parsed_element_layer_ids: set of layer_ids whose elements to
          remove from parsed_elements (concurrency-safe: filters live state by
          source_layer_id rather than replacing the entire list)
        - _set_parsed_elements: legacy full-replacement (kept for compat, but
          no longer generated by r_remove_parsed_elements_by_layers)

        Strategy: let the original update() run first (which processes
        _enqueue_children, _append_parsed_elements, etc.), then apply
        filter/removal operations afterwards so they take precedence.

        [BugFix v2] Replaced _set_parsed_elements (full replacement from
        snapshot) with _remove_parsed_element_layer_ids (filter by source
        layer_id on the LIVE state). This prevents the concurrent race where
        a worker computing _set_parsed_elements from a stale snapshot would
        overwrite elements added by other concurrent workers.
        """
        remove_ids = None
        remove_layer_ids = None
        set_elements = None
        appended_in_batch = []
        cleaned = []

        for upd in updates:
            if not upd:
                cleaned.append(upd)
                continue
            new_upd = dict(upd)
            if "_remove_from_queue_ids" in new_upd:
                ids = new_upd.pop("_remove_from_queue_ids")
                if remove_ids is None:
                    remove_ids = set(ids)
                else:
                    remove_ids.update(ids)
            if "_remove_parsed_element_layer_ids" in new_upd:
                # Concurrency-safe: accumulate layer_ids to filter out
                lids = new_upd.pop("_remove_parsed_element_layer_ids")
                if remove_layer_ids is None:
                    remove_layer_ids = set(lids)
                else:
                    remove_layer_ids.update(lids)
            if "_set_parsed_elements" in new_upd:
                # Legacy compat: full replacement (snapshot-based, less safe)
                set_elements = new_upd.pop("_set_parsed_elements")
            # Track elements being appended in this same batch (for legacy compat)
            if "_append_parsed_element" in new_upd:
                elem = new_upd["_append_parsed_element"]
                if isinstance(elem, dict):
                    appended_in_batch.append(elem)
            if "_append_parsed_elements" in new_upd and isinstance(new_upd["_append_parsed_elements"], list):
                appended_in_batch.extend(
                    e for e in new_upd["_append_parsed_elements"] if isinstance(e, dict)
                )
            cleaned.append(new_upd)

        # Run the original update with cleaned dicts
        result = _original_sm_update(self, *cleaned)

        # Now apply removals/replacements under the lock
        if remove_ids is not None or remove_layer_ids is not None or set_elements is not None:
            with self._lock:
                if remove_ids is not None:
                    queue = list(self._state.get("layer_queue", []))
                    filtered = [lid for lid in queue if lid not in remove_ids]
                    removed_count = len(queue) - len(filtered)
                    self._state["layer_queue"] = filtered
                    print(f"[StateManager] Removed {removed_count} ids from queue: {remove_ids}")
                if remove_layer_ids is not None:
                    # [BugFix v2] Filter by source_layer_id on the LIVE state —
                    # safe for concurrent additions from unrelated groups.
                    elements = list(self._state.get("parsed_elements", []))
                    original_count = len(elements)
                    elements = [
                        e for e in elements
                        if not (isinstance(e, dict) and e.get("source_layer_id", "") in remove_layer_ids)
                    ]
                    removed_count = original_count - len(elements)
                    if removed_count > 0:
                        self._state["parsed_elements"] = elements
                        print(f"[StateManager] _remove_parsed_element_layer_ids: removed {removed_count} elements for layers {remove_layer_ids}")
                if set_elements is not None:
                    # Legacy compat: full replacement — merge batch-appended elements
                    final = list(set_elements)
                    if appended_in_batch:
                        existing_ids = {e.get("id") for e in final if isinstance(e, dict)}
                        for elem in appended_in_batch:
                            if elem.get("id") not in existing_ids:
                                final.append(elem)
                                existing_ids.add(elem.get("id"))
                        print(f"[StateManager] _set_parsed_elements (legacy) merged {len(final) - len(set_elements)} batch-appended elements")
                    self._state["parsed_elements"] = final
                    print(f"[StateManager] _set_parsed_elements (legacy) applied: {len(final)} items")
                result = dict(self._state)

        return result

    StateManager.update = _patched_sm_update

    # =========================================================================
    # 5. Patch visualizer for sparse_failed support
    # =========================================================================
    try:
        from REDESIGN_ablation.visualizer import apply_visualizer_patch
        apply_visualizer_patch()
    except ImportError:
        print("[Baseline 3] Warning: Could not import ablation visualizer")

    # =========================================================================
    # 6. Patch _get_unfinished_layers to skip sparse_failed nodes
    # =========================================================================
    import REDESIGN.episode_run as _ep_mod
    _original_get_unfinished = _ep_mod._get_unfinished_layers

    def _patched_get_unfinished_layers(state):
        """
        Skip nodes whose verification_status is 'sparse_failed'.

        These nodes were explicitly rejected by sparse verification
        (PROCEED_FILTERED or RETRY) and must NOT be force-finalized.
        """
        unfinished = _original_get_unfinished(state)
        history_tree = state.get("history_tree", {})
        filtered = []
        for layer_id in unfinished:
            node = history_tree.get(layer_id, {})
            vs = node.get("verification_status", "")
            if vs == "sparse_failed":
                print(f"[Baseline 3] Skipping sparse_failed layer from force-finalize: {layer_id}")
                continue
            filtered.append(layer_id)
        return filtered

    _ep_mod._get_unfinished_layers = _patched_get_unfinished_layers

    # =========================================================================
    # 7. Patch process_layer_worker: skip sparse_failed nodes (race condition)
    # =========================================================================
    _original_process_layer_worker = _ep_mod.process_layer_worker

    def _patched_process_layer_worker(state_manager, layer_id, verbose=True):
        """
        Guard at worker start: skip nodes already marked sparse_failed.

        This catches the case where a node was enqueued and dequeued before
        sparse verification marked it as sparse_failed. Checks LIVE state
        (not snapshot) for the most up-to-date verification_status.

        For race conditions where a worker is already mid-execution when
        verification marks its node as sparse_failed, Patch 8 provides
        comprehensive StateManager-level protection:
        (a) Blocks verification_status overwrite
        (b) Blocks parsed_elements from failed nodes
        (c) Blocks enqueue of children with failed parents
        (d) Post-hoc cleanup of already-committed artifacts
        """
        from REDESIGN.episode_run import log

        # Check LIVE state before starting any work
        live_state = state_manager.state
        node_data = live_state.get("history_tree", {}).get(layer_id, {})
        if node_data.get("verification_status") == "sparse_failed":
            log(f"Skipping sparse_failed node (race condition guard)", f"Worker:{layer_id}")
            return

        return _original_process_layer_worker(state_manager, layer_id, verbose)

    _ep_mod.process_layer_worker = _patched_process_layer_worker

    # =========================================================================
    # 8. Comprehensive sparse_failed protection in StateManager
    # =========================================================================
    # Three-layer defense against race conditions where a worker is already
    # mid-execution when verification marks its node as sparse_failed:
    #
    # (a) Protect verification_status: don't allow overwriting sparse_failed
    # (b) Block parsed_elements: drop _append_parsed_element(s) for failed nodes
    # (c) Block enqueue: drop _enqueue_children if parent is sparse_failed
    # (d) Post-hoc cleanup: after history_tree updates that introduce NEW
    #     sparse_failed nodes, retroactively clean parsed_elements and queue
    #
    _original_sm_update_v2 = StateManager.update  # already patched in step 4

    def _patched_sm_update_v2(self, *updates):
        """
        Comprehensive sparse_failed protection wrapper for StateManager.update.

        Handles the full race condition: even if a worker is mid-execution when
        its node gets marked sparse_failed, this ensures:
        1. verification_status is never overwritten from sparse_failed
        2. parsed_elements from sparse_failed nodes are blocked/removed
        3. children of sparse_failed nodes are not enqueued
        """
        # ── Snapshot of currently-failed nodes BEFORE this update ──
        with self._lock:
            tree_before = self._state.get("history_tree", {})
            failed_before = {
                nid for nid, nd in tree_before.items()
                if nd.get("verification_status") == "sparse_failed"
            }

        # ── Pre-filter updates to protect sparse_failed nodes ──
        cleaned = []
        for upd in updates:
            if not upd:
                cleaned.append(upd)
                continue
            new_upd = dict(upd)

            # (a) Protect verification_status from overwrite
            if "history_tree" in new_upd and isinstance(new_upd["history_tree"], dict) and failed_before:
                ht = dict(new_upd["history_tree"])
                for nid in failed_before:
                    if nid in ht:
                        node_upd = ht[nid]
                        if "verification_status" in node_upd and node_upd["verification_status"] != "sparse_failed":
                            node_upd = dict(node_upd)
                            del node_upd["verification_status"]
                            if node_upd:
                                ht[nid] = node_upd
                            else:
                                del ht[nid]
                new_upd["history_tree"] = ht

            # (b) Block _append_parsed_element for sparse_failed source_layer_id
            if "_append_parsed_element" in new_upd and failed_before:
                elem = new_upd["_append_parsed_element"]
                source_lid = elem.get("source_layer_id", "") if isinstance(elem, dict) else ""
                if source_lid in failed_before:
                    print(f"[SM Guard] Blocked _append_parsed_element for sparse_failed node {source_lid}")
                    del new_upd["_append_parsed_element"]
                    # Also remove the history_tree parsed_elements update for this node
                    if "history_tree" in new_upd and isinstance(new_upd["history_tree"], dict):
                        ht = dict(new_upd["history_tree"])
                        if source_lid in ht and "parsed_elements" in ht.get(source_lid, {}):
                            node_upd = dict(ht[source_lid])
                            del node_upd["parsed_elements"]
                            if node_upd:
                                ht[source_lid] = node_upd
                            else:
                                del ht[source_lid]
                            new_upd["history_tree"] = ht

            if "_append_parsed_elements" in new_upd and isinstance(new_upd["_append_parsed_elements"], list) and failed_before:
                filtered_elems = [
                    e for e in new_upd["_append_parsed_elements"]
                    if not (isinstance(e, dict) and e.get("source_layer_id", "") in failed_before)
                ]
                blocked_count = len(new_upd["_append_parsed_elements"]) - len(filtered_elems)
                if blocked_count > 0:
                    print(f"[SM Guard] Blocked {blocked_count} parsed_elements for sparse_failed nodes")
                if filtered_elems:
                    new_upd["_append_parsed_elements"] = filtered_elems
                else:
                    del new_upd["_append_parsed_elements"]

            # (c) Block _enqueue_children whose parent is sparse_failed
            if "_enqueue_children" in new_upd and isinstance(new_upd["_enqueue_children"], list) and failed_before:
                # Check history_tree (including this update's tree changes) for parent_id
                ht_update = new_upd.get("history_tree", {})
                children_to_enqueue = new_upd["_enqueue_children"]
                allowed = []
                for child_id in children_to_enqueue:
                    # Check if this child's parent is sparse_failed
                    child_data = ht_update.get(child_id, {}) if isinstance(ht_update, dict) else {}
                    parent_id = child_data.get("parent_id", "")
                    if not parent_id:
                        # Check existing tree
                        parent_id = tree_before.get(child_id, {}).get("parent_id", "")
                    if parent_id in failed_before:
                        print(f"[SM Guard] Blocked enqueue of {child_id} (parent {parent_id} is sparse_failed)")
                    else:
                        allowed.append(child_id)
                if allowed:
                    new_upd["_enqueue_children"] = allowed
                else:
                    del new_upd["_enqueue_children"]

            cleaned.append(new_upd)

        # ── Apply the cleaned updates ──
        result = _original_sm_update_v2(self, *cleaned)

        # ── (d) Post-hoc cleanup: detect NEWLY failed nodes ──
        # If this update introduced new sparse_failed nodes (e.g., from
        # _handle_sparse_proceed_filtered), clean up any parsed_elements
        # and queue entries that racing workers may have already committed.
        with self._lock:
            tree_after = self._state.get("history_tree", {})
            failed_after = {
                nid for nid, nd in tree_after.items()
                if nd.get("verification_status") == "sparse_failed"
            }
            newly_failed = failed_after - failed_before

            if newly_failed:
                print(f"[SM Guard] Post-hoc cleanup for {len(newly_failed)} newly failed nodes: {newly_failed}")

                # Clean parsed_elements: remove any elements whose source_layer_id is newly failed
                elements = list(self._state.get("parsed_elements", []))
                original_count = len(elements)
                elements = [
                    e for e in elements
                    if not (isinstance(e, dict) and e.get("source_layer_id", "") in newly_failed)
                ]
                removed_elem_count = original_count - len(elements)
                if removed_elem_count > 0:
                    self._state["parsed_elements"] = elements
                    print(f"[SM Guard] Post-hoc: removed {removed_elem_count} parsed_elements from newly failed nodes")

                # Collect all descendants of newly failed nodes
                all_failed_and_descendants = set(newly_failed)
                def _collect_descendants(node_id):
                    node = tree_after.get(node_id, {})
                    for child_id in (node.get("children_ids") or []):
                        all_failed_and_descendants.add(child_id)
                        _collect_descendants(child_id)
                for nid in newly_failed:
                    _collect_descendants(nid)

                # Mark ALL descendants as sparse_failed in history_tree
                # This ensures Patch 7 catches already-dequeued descendants
                descendant_set = all_failed_and_descendants - newly_failed
                if descendant_set:
                    tree = dict(self._state.get("history_tree", {}))
                    for did in descendant_set:
                        if did in tree:
                            node = dict(tree[did])
                            node["verification_status"] = "sparse_failed"
                            tree[did] = node
                    self._state["history_tree"] = tree
                    print(f"[SM Guard] Post-hoc: marked {len(descendant_set)} descendants as sparse_failed")

                # Remove from queue
                queue = list(self._state.get("layer_queue", []))
                original_queue_len = len(queue)
                queue = [lid for lid in queue if lid not in all_failed_and_descendants]
                removed_queue_count = original_queue_len - len(queue)
                if removed_queue_count > 0:
                    self._state["layer_queue"] = queue
                    print(f"[SM Guard] Post-hoc: removed {removed_queue_count} entries from queue (failed nodes + descendants)")

                # Clean parsed_elements of all failed + descendant nodes
                if descendant_set:
                    elements = list(self._state.get("parsed_elements", []))
                    original_count = len(elements)
                    elements = [
                        e for e in elements
                        if not (isinstance(e, dict) and e.get("source_layer_id", "") in descendant_set)
                    ]
                    removed_desc_count = original_count - len(elements)
                    if removed_desc_count > 0:
                        self._state["parsed_elements"] = elements
                        print(f"[SM Guard] Post-hoc: removed {removed_desc_count} parsed_elements from descendant nodes")

                result = dict(self._state)

        return result

    StateManager.update = _patched_sm_update_v2

    print(f"[Baseline 3] Monkey-patched: stack_manager, finalize_obj, merge_updates, StateManager, visualizer, _get_unfinished_layers, process_layer_worker (period={sparse_period})")


# =============================================================================
# GPU Configuration
# =============================================================================

def setup_gpu_config(
    qwen_gpus: Optional[List[int]] = None,
    qwen_pair_size: Optional[int] = None,
    tool_gpus: Optional[List[int]] = None,
    objectclear_gpu: Optional[int] = None,
) -> None:
    from REDESIGN.tool_gpu_config import set_runtime_config, print_config

    if qwen_gpus or qwen_pair_size or tool_gpus or objectclear_gpu:
        set_runtime_config(
            qwen_gpus=qwen_gpus,
            qwen_pair_size=qwen_pair_size,
            tool_gpus=tool_gpus,
            objectclear_gpu=objectclear_gpu,
        )

    print_config()


def parse_gpu_list(gpu_str: Optional[str]) -> Optional[List[int]]:
    if not gpu_str:
        return None
    try:
        return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]
    except ValueError:
        return None


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("baseline3_runner")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# Path Resolution & Frame Loading
# =============================================================================

def get_src_root() -> Path:
    current = Path(__file__).resolve().parent
    if current.name == "src":
        return current
    elif (current / "src").exists():
        return current / "src"
    else:
        return current


def get_output_dir(src_root: Path) -> Path:
    return src_root / BASELINE3_EXPERIMENT_BASE


def load_all_frames(src_root: Path) -> List[Dict[str, Any]]:
    """Load all frames from all dataset splits (dino80 + dino90 = 909)."""
    frames = []
    for prefix, num_splits in ALL_DATASET_SPLITS:
        for split_idx in range(num_splits):
            split_name = f"{prefix}{split_idx}"
            split_dir = src_root / FIGMA_DATA_BASE / split_name
            valid_frames_dir = split_dir / "valid_frames"

            if not valid_frames_dir.exists():
                print(f"[Warning] Not found: {valid_frames_dir}")
                continue

            json_files = sorted(valid_frames_dir.glob("*.json"))
            for json_path in json_files:
                frame_id = json_path.stem
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    rel_path = json_data.get("reconstructed_image_path")
                    unit_images_dir = json_data.get("unit_images_dir")
                    if rel_path and unit_images_dir:
                        image_path = split_dir / unit_images_dir / rel_path
                        if image_path.exists():
                            frames.append({
                                "frame_id": frame_id,
                                "image_path": image_path,
                                "split_name": split_name,
                            })
                except Exception:
                    continue

    return frames


def is_frame_completed(output_dir: Path, frame_id: str) -> bool:
    return (output_dir / "episodes" / frame_id / "parse.json").exists()


# =============================================================================
# Episode Runner (in-process, with monkey-patch)
# =============================================================================

def run_episode_for_frame(
    image_path: Path,
    output_dir: Path,
    frame_id: str,
    workers: int = DEFAULT_WORKERS,
    llm_limit: int = DEFAULT_LLM_LIMIT,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_layers: int = DEFAULT_MAX_LAYERS,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    Run a single frame using REDESIGN.run_episode (in-process, monkey-patched).
    """
    try:
        start_time = time.time()

        from REDESIGN.episode_run import run_episode

        result = run_episode(
            image_path=str(image_path),
            output_dir=str(output_dir),
            episode_id=frame_id,
            parallel=True,
            max_parallel_workers=workers,
            llm_call_limit=llm_limit,
            max_depth=max_depth,
            max_layers=max_layers,
        )

        elapsed = time.time() - start_time

        if result and not result.get("error"):
            return True, f"Completed in {elapsed:.1f}s"
        else:
            error = result.get("error", "Unknown error") if result else "No result"
            return False, f"Failed: {error}"

    except Exception as e:
        return False, f"Exception: {str(e)}"


def run_episode_subprocess(
    image_path: Path,
    output_dir: Path,
    frame_id: str,
    workers: int = DEFAULT_WORKERS,
    llm_limit: int = DEFAULT_LLM_LIMIT,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_layers: int = DEFAULT_MAX_LAYERS,
    sparse_period: int = 3,
    qwen_gpus: Optional[str] = None,
    qwen_pair_size: Optional[int] = None,
    tool_gpus: Optional[str] = None,
    timeout: int = 3600,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    Run a single frame as subprocess.
    Uses a wrapper that applies the monkey-patch before running episode_run.
    """
    cmd = [
        sys.executable, "-c",
        (
            "import sys; sys.path.insert(0, '.');"
            "import multiprocessing as mp; mp.set_start_method('spawn', force=True);"
            "from BASELINES.run_sparse_verification_agent_figma import apply_sparse_verification_patch;"
            f"apply_sparse_verification_patch(sparse_period={sparse_period});"
            "from REDESIGN.episode_run import run_episode;"
            f"run_episode("
            f"  image_path='{image_path}',"
            f"  output_dir='{output_dir}',"
            f"  episode_id='{frame_id}',"
            f"  parallel=True,"
            f"  max_parallel_workers={workers},"
            f"  llm_call_limit={llm_limit},"
            f"  max_depth={max_depth},"
            f"  max_layers={max_layers},"
            f")"
        ),
    ]

    env = os.environ.copy()
    if qwen_gpus:
        env["URLD_QWEN_GPUS"] = qwen_gpus
    if qwen_pair_size:
        env["URLD_QWEN_PAIR_SIZE"] = str(qwen_pair_size)
    if tool_gpus:
        env["URLD_TOOL_GPUS"] = tool_gpus

    try:
        start_time = time.time()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(get_src_root()),
            env=env,
        )

        if logger:
            for line in process.stdout:
                logger.info(f"  [Child] {line.strip()}")

        stdout, _ = process.communicate(timeout=timeout)
        return_code = process.returncode
        elapsed = time.time() - start_time

        if return_code == 0:
            return True, f"Completed in {elapsed:.1f}s"
        else:
            return False, f"Failed (code {return_code})"

    except subprocess.TimeoutExpired:
        process.kill()
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, f"Exception: {str(e)}"


# =============================================================================
# Main Runner
# =============================================================================

def run_all(
    workers: int = DEFAULT_WORKERS,
    qwen_gpus: Optional[str] = None,
    qwen_pair_size: Optional[int] = None,
    tool_gpus: Optional[str] = None,
    sparse_period: int = 3,
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip_completed: bool = True,
    src_root: Optional[Path] = None,
    use_subprocess: bool = True,
    llm_limit: int = DEFAULT_LLM_LIMIT,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_layers: int = DEFAULT_MAX_LAYERS,
    split_index: Optional[int] = None,
    num_splits: Optional[int] = None,
) -> Dict[str, Any]:
    from tqdm import tqdm

    if src_root is None:
        src_root = get_src_root()

    output_dir = get_output_dir(src_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "baseline3_run.log"
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("Baseline 3: Sparse Verification Agent")
    logger.info("=" * 70)
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"sparse_period: {sparse_period}")
    logger.info(f"workers: {workers}")
    logger.info(f"qwen_gpus: {qwen_gpus or 'default'}")
    logger.info(f"tool_gpus: {tool_gpus or 'default'}")
    logger.info(f"dry_run: {dry_run}, limit: {limit}")
    logger.info(f"use_subprocess: {use_subprocess}")

    # Load ALL frames (dino80 + dino90 = 909)
    all_frames = load_all_frames(src_root)
    logger.info(f"Found {len(all_frames)} total frames across all splits")

    # Split assignment FIRST (before filtering completed), then filter.
    #
    # IMPORTANT: split must be applied to the FULL deterministic 909-frame list,
    # NOT to the pending (not-yet-completed) list. If we split after filtering,
    # servers starting at different times would see different pending-list lengths
    # and get non-overlapping but shifted modular assignments — causing some frames
    # to be processed by two servers and others by none.
    #
    # Correct approach:
    #   1. Split the full list by stable modular index → each frame always belongs
    #      to exactly one split regardless of how many frames are already done.
    #   2. Filter completed within each server's assigned subset (resume safety).
    #
    # This guarantees:
    # - No overlap: frame i goes to split (i % num_splits), always.
    # - Real-time resume: each restart re-checks is_frame_completed() live.
    # - Independence: servers don't need to coordinate; they read the same
    #   deterministic frame list and pick their own slice.
    if split_index is not None and num_splits is not None:
        if not (0 <= split_index < num_splits):
            raise ValueError(f"split_index {split_index} out of range [0, {num_splits})")
        assigned = [f for i, f in enumerate(all_frames) if i % num_splits == split_index]
        logger.info(f"Split mode: {split_index+1}/{num_splits} → {len(assigned)} frames assigned from {len(all_frames)} total")
    else:
        assigned = all_frames

    # Filter completed (resume mode) — applied per-split so each server independently
    # skips its own already-done frames without affecting other splits.
    frames_to_process = []
    skipped = 0
    for frame in assigned:
        if skip_completed and is_frame_completed(output_dir, frame["frame_id"]):
            skipped += 1
            continue
        frames_to_process.append(frame)

    if limit:
        frames_to_process = frames_to_process[:limit]

    logger.info(f"Will process {len(frames_to_process)} frames (skipped {skipped} completed)")

    if dry_run:
        logger.info("\n[DRY RUN] Would process these frames:")
        for i, frame in enumerate(frames_to_process[:30]):
            logger.info(f"  [{i+1:3d}] {frame['frame_id']} ({frame['split_name']})")
        if len(frames_to_process) > 30:
            logger.info(f"  ... and {len(frames_to_process) - 30} more")
        return {"dry_run": True, "total": len(all_frames), "pending": len(frames_to_process), "skipped": skipped}

    # Apply monkey-patch if running in-process
    if not use_subprocess:
        apply_sparse_verification_patch(sparse_period=sparse_period)

    results = {
        "start_time": datetime.now().isoformat(),
        "sparse_period": sparse_period,
        "processed": [],
        "skipped_count": skipped,
        "failed": [],
    }

    pbar = tqdm(frames_to_process, desc=f"Baseline3 (period={sparse_period})",
                unit="frame", ncols=100)

    for frame in pbar:
        frame_id = frame["frame_id"]
        image_path = frame["image_path"]

        pbar.set_postfix_str(f"{frame_id}", refresh=True)
        logger.info(f"\nProcessing: {frame_id} ({frame['split_name']})")

        try:
            if use_subprocess:
                success, message = run_episode_subprocess(
                    image_path=image_path,
                    output_dir=output_dir,
                    frame_id=frame_id,
                    workers=workers,
                    llm_limit=llm_limit,
                    max_depth=max_depth,
                    max_layers=max_layers,
                    sparse_period=sparse_period,
                    qwen_gpus=qwen_gpus,
                    qwen_pair_size=qwen_pair_size,
                    tool_gpus=tool_gpus,
                    logger=logger,
                )
            else:
                success, message = run_episode_for_frame(
                    image_path=image_path,
                    output_dir=output_dir,
                    frame_id=frame_id,
                    workers=workers,
                    llm_limit=llm_limit,
                    max_depth=max_depth,
                    max_layers=max_layers,
                    logger=logger,
                )

            if success:
                logger.info(f"OK: {message}")
                results["processed"].append({"frame_id": frame_id, "message": message})
            else:
                logger.error(f"FAIL: {message}")
                results["failed"].append({"frame_id": frame_id, "error": message})

        except Exception as e:
            logger.exception(f"Exception processing {frame_id}")
            results["failed"].append({"frame_id": frame_id, "error": str(e)})

        # Save incremental results (resume-safe)
        results["end_time"] = datetime.now().isoformat()
        results_file = output_dir / "baseline3_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        ok = len(results["processed"])
        fail = len(results["failed"])
        pbar.set_postfix_str(f"ok={ok} fail={fail}", refresh=True)

    pbar.close()

    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Processed: {len(results['processed'])}")
    logger.info(f"Skipped: {results['skipped_count']}")
    logger.info(f"Failed: {len(results['failed'])}")

    if results["failed"]:
        logger.info("\nFailed frames:")
        for f in results["failed"]:
            logger.info(f"  - {f['frame_id']}: {f['error'][:100]}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Baseline 3: Agent with Sparse Verification (909 frames)",
    )

    parser.add_argument("--sparse_period", "-p", type=int, default=3, choices=[2, 3],
                        help="Sparse verification period: verify every N generations (default: 3)")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--qwen_gpus", type=str, default=None,
                        help="GPU IDs for Qwen model (comma-separated, e.g., '0,1')")
    parser.add_argument("--qwen_pair_size", type=int, default=None,
                        help="Number of GPUs per Qwen pair (e.g., 2 for A6000)")
    parser.add_argument("--tool_gpus", type=str, default=None,
                        help="GPU IDs for Tool models (comma-separated, e.g., '2,3')")
    parser.add_argument("--objectclear_gpu", type=int, default=None,
                        help="GPU ID for ObjectClear model (e.g., 3)")
    parser.add_argument("--llm_limit", type=int, default=DEFAULT_LLM_LIMIT)
    parser.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max_layers", type=int, default=DEFAULT_MAX_LAYERS)
    parser.add_argument("--dry_run", "-d", action="store_true")
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--no_skip", action="store_true",
                        help="Don't skip completed frames (re-process all)")
    parser.add_argument("--in_process", action="store_true",
                        help="Run in-process instead of subprocess (for debugging)")
    parser.add_argument("--src_root", type=str, default=None)
    parser.add_argument("--split_index", type=int, default=None,
                        help="0-based index of this server's split (requires --num_splits)")
    parser.add_argument("--num_splits", type=int, default=None,
                        help="Total number of splits (e.g., 3 for 3 servers). "
                             "Each server gets every Nth pending frame by modular assignment. "
                             "Resume-safe: already-completed frames are always skipped first.")

    args = parser.parse_args()

    # GPU setup
    qwen_gpu_list = parse_gpu_list(args.qwen_gpus)
    tool_gpu_list = parse_gpu_list(args.tool_gpus)

    setup_gpu_config(
        qwen_gpus=qwen_gpu_list,
        qwen_pair_size=args.qwen_pair_size,
        tool_gpus=tool_gpu_list,
        objectclear_gpu=args.objectclear_gpu,
    )

    src_root = Path(args.src_root) if args.src_root else None

    try:
        results = run_all(
            workers=args.workers,
            qwen_gpus=args.qwen_gpus,
            qwen_pair_size=args.qwen_pair_size,
            tool_gpus=args.tool_gpus,
            sparse_period=args.sparse_period,
            dry_run=args.dry_run,
            limit=args.limit,
            skip_completed=not args.no_skip,
            src_root=src_root,
            use_subprocess=not args.in_process,
            llm_limit=args.llm_limit,
            max_depth=args.max_depth,
            max_layers=args.max_layers,
            split_index=args.split_index,
            num_splits=args.num_splits,
        )

        if not args.dry_run:
            print(f"\nResults saved to: {BASELINE3_EXPERIMENT_BASE}/")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
