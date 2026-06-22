#!/usr/bin/env python3
"""
monitor_figma_parsing.py - 실시간 파싱 진행상황 모니터링

Usage:
    python monitor_figma_parsing.py                    # 모든 split 모니터링
    python monitor_figma_parsing.py --split_idx 0     # 특정 split만 모니터링
    python monitor_figma_parsing.py --watch           # 5초마다 갱신
    python monitor_figma_parsing.py --detail          # 상세 정보 출력

Features:
- 각 split별 완료/진행중/대기 상태 표시
- 실시간 completion rate 계산
- 최근 완료된 frame 목록
- evaluator.py에서 사용할 수 있는 완료된 frame 목록 출력
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import os


# =============================================================================
# Configuration
# =============================================================================

FIGMA_DATA_BASE = "figma_data/process/subset"
FIGMA_EXPERIMENT_BASE = "figma_experiment"
SPLIT_PREFIX = "dino90_obj_5_25_char_50_split_"
NUM_SPLITS = 5


# =============================================================================
# Path Resolution
# =============================================================================

def get_src_root() -> Path:
    """Get the src directory root."""
    current = Path.cwd()
    if current.name == "src":
        return current
    elif (current / "src").exists():
        return current / "src"
    elif (current.parent / "src").exists():
        return current.parent / "src"
    else:
        return current


def get_paths(src_root: Path, split_idx: int) -> Dict[str, Path]:
    """Get paths for a specific split."""
    split_name = f"{SPLIT_PREFIX}{split_idx}"
    return {
        "valid_frames_dir": src_root / FIGMA_DATA_BASE / split_name / "valid_frames",
        "output_dir": src_root / FIGMA_EXPERIMENT_BASE / f"split_{split_idx}",
    }


# =============================================================================
# Status Checking
# =============================================================================

def get_split_status(src_root: Path, split_idx: int) -> Dict[str, Any]:
    """Get detailed status for a split."""
    paths = get_paths(src_root, split_idx)
    
    status = {
        "split_idx": split_idx,
        "valid_frames_dir": str(paths["valid_frames_dir"]),
        "output_dir": str(paths["output_dir"]),
        "exists": paths["valid_frames_dir"].exists(),
        "total": 0,
        "completed": 0,
        "failed": 0,
        "pending": 0,
        "completion_rate": 0.0,
        "completed_frames": [],
        "failed_frames": [],
        "pending_frames": [],
        "recent_completed": [],
    }
    
    if not status["exists"]:
        return status
    
    # Get all JSON files
    json_files = sorted(paths["valid_frames_dir"].glob("*.json"))
    status["total"] = len(json_files)
    
    if not json_files:
        return status
    
    # Check each frame
    for json_file in json_files:
        frame_id = json_file.stem
        episode_dir = paths["output_dir"] / "episodes" / frame_id
        parse_json = episode_dir / "parse.json"
        
        if parse_json.exists():
            status["completed"] += 1
            status["completed_frames"].append({
                "frame_id": frame_id,
                "parse_json": str(parse_json),
                "mtime": parse_json.stat().st_mtime,
            })
        elif episode_dir.exists():
            # Episode dir exists but no parse.json - possibly failed or in progress
            log_file = episode_dir / "episode.log"
            if log_file.exists():
                # Check if recently modified (within 5 minutes) - likely in progress
                mtime = log_file.stat().st_mtime
                if time.time() - mtime < 300:  # 5 minutes
                    status["pending"] += 1
                    status["pending_frames"].append(frame_id)
                else:
                    status["failed"] += 1
                    status["failed_frames"].append(frame_id)
            else:
                status["failed"] += 1
                status["failed_frames"].append(frame_id)
        else:
            status["pending"] += 1
            status["pending_frames"].append(frame_id)
    
    if status["total"] > 0:
        status["completion_rate"] = status["completed"] / status["total"] * 100
    
    # Get recent completed (sorted by mtime)
    status["completed_frames"].sort(key=lambda x: x["mtime"], reverse=True)
    status["recent_completed"] = status["completed_frames"][:5]
    
    return status


def get_all_splits_status(src_root: Path) -> List[Dict[str, Any]]:
    """Get status for all splits."""
    return [get_split_status(src_root, i) for i in range(NUM_SPLITS)]


def get_completed_frames_for_evaluation(src_root: Path, split_indices: List[int] = None) -> List[Dict[str, Any]]:
    """
    Get list of completed frames ready for evaluation.
    
    Returns list of dicts with:
    - frame_id
    - gt_json_path (original JSON with GT elements)
    - parse_json_path (agent parsed elements)
    - reconstructed_image_path (GT image)
    """
    if split_indices is None:
        split_indices = list(range(NUM_SPLITS))
    
    completed = []
    
    for split_idx in split_indices:
        paths = get_paths(src_root, split_idx)
        
        if not paths["valid_frames_dir"].exists():
            continue
        
        for json_file in sorted(paths["valid_frames_dir"].glob("*.json")):
            frame_id = json_file.stem
            parse_json = paths["output_dir"] / "episodes" / frame_id / "parse.json"
            
            if parse_json.exists():
                # Load GT JSON to get reconstructed_image_path
                try:
                    with open(json_file, 'r') as f:
                        gt_data = json.load(f)
                    
                    reconstructed_path = gt_data.get("reconstructed_image_path", "")
                    if reconstructed_path:
                        reconstructed_path = paths["valid_frames_dir"] / reconstructed_path
                    
                    completed.append({
                        "split_idx": split_idx,
                        "frame_id": frame_id,
                        "gt_json_path": str(json_file),
                        "parse_json_path": str(parse_json),
                        "reconstructed_image_path": str(reconstructed_path),
                        "episode_dir": str(parse_json.parent),
                    })
                except Exception as e:
                    print(f"Warning: Failed to load {json_file}: {e}")
    
    return completed


# =============================================================================
# Display Functions
# =============================================================================

def print_summary(statuses: List[Dict[str, Any]], detailed: bool = False):
    """Print summary of all splits."""
    print("\n" + "=" * 70)
    print("FIGMA PARSING STATUS")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    total_all = 0
    completed_all = 0
    
    for status in statuses:
        if not status["exists"]:
            print(f"\nSplit {status['split_idx']}: NOT FOUND")
            continue
        
        total_all += status["total"]
        completed_all += status["completed"]
        
        bar_len = 30
        filled = int(bar_len * status["completion_rate"] / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        print(f"\nSplit {status['split_idx']}: [{bar}] {status['completion_rate']:5.1f}%")
        print(f"  Completed: {status['completed']:3d}/{status['total']:3d}")
        print(f"  Failed:    {status['failed']:3d}")
        print(f"  Pending:   {status['pending']:3d}")
        
        if detailed and status["recent_completed"]:
            print(f"  Recent completed:")
            for frame in status["recent_completed"][:3]:
                mtime = datetime.fromtimestamp(frame["mtime"]).strftime("%H:%M:%S")
                print(f"    - {frame['frame_id']} ({mtime})")
    
    if total_all > 0:
        overall_rate = completed_all / total_all * 100
        print(f"\n{'='*70}")
        print(f"OVERALL: {completed_all}/{total_all} ({overall_rate:.1f}%)")
        print(f"{'='*70}")


def print_evaluation_list(completed: List[Dict[str, Any]], output_file: Optional[str] = None):
    """Print or save list of completed frames for evaluation."""
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(completed, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(completed)} completed frames to: {output_file}")
    else:
        print(f"\n{len(completed)} frames ready for evaluation:")
        for i, frame in enumerate(completed[:10]):
            print(f"  {i+1}. {frame['frame_id']} (split {frame['split_idx']})")
        if len(completed) > 10:
            print(f"  ... and {len(completed) - 10} more")


# =============================================================================
# Watch Mode
# =============================================================================

def watch_mode(src_root: Path, interval: int = 5, split_idx: Optional[int] = None):
    """Continuously monitor parsing status."""
    try:
        while True:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            if split_idx is not None:
                statuses = [get_split_status(src_root, split_idx)]
            else:
                statuses = get_all_splits_status(src_root)
            
            print_summary(statuses, detailed=True)
            print(f"\n(Refreshing every {interval}s, Ctrl+C to stop)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nStopped monitoring.")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Monitor Figma Parsing Progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show status of all splits
    python monitor_figma_parsing.py
    
    # Show detailed status
    python monitor_figma_parsing.py --detail
    
    # Watch mode (auto-refresh every 5 seconds)
    python monitor_figma_parsing.py --watch
    
    # Monitor specific split
    python monitor_figma_parsing.py --split_idx 0 --watch
    
    # Export completed frames for evaluation
    python monitor_figma_parsing.py --export_completed completed_frames.json
        """
    )
    
    parser.add_argument(
        "--split_idx", "-s",
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4],
        help="Monitor specific split only"
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Watch mode (auto-refresh)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=5,
        help="Refresh interval for watch mode (seconds)"
    )
    parser.add_argument(
        "--detail", "-d",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--export_completed", "-e",
        type=str,
        default=None,
        help="Export completed frames to JSON file"
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default=None,
        help="Source root directory (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    src_root = Path(args.src_root) if args.src_root else get_src_root()
    
    if args.export_completed:
        split_indices = [args.split_idx] if args.split_idx is not None else None
        completed = get_completed_frames_for_evaluation(src_root, split_indices)
        print_evaluation_list(completed, args.export_completed)
        return
    
    if args.watch:
        watch_mode(src_root, args.interval, args.split_idx)
    else:
        if args.split_idx is not None:
            statuses = [get_split_status(src_root, args.split_idx)]
        else:
            statuses = get_all_splits_status(src_root)
        
        print_summary(statuses, args.detail)
        
        # Also show evaluation-ready count
        completed = get_completed_frames_for_evaluation(src_root)
        print(f"\n{len(completed)} frames ready for evaluation (use --export_completed to save list)")


if __name__ == "__main__":
    main()