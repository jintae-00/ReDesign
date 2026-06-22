#!/usr/bin/env python3
"""
run_agent_figma.py - Run the ReDesign agent on a Figma dataset (split-agnostic)

Processes EVERY episode found under <data_dir>/valid_frames/. Run from the
repository root so the `REDESIGN` package is importable.

Usage:
    # Full 909-episode benchmark
    python -m REDESIGN.run_agent_figma \
        --data_dir figma_data --output_dir outputs/figma_agent \
        --qwen_gpus 2,3,4,5 --qwen_pair_size 2 --tool_gpus 6,7

    # GPU config via environment variables
    URLD_QWEN_GPUS="3,4,5" URLD_TOOL_GPUS="6,7" \
        python -m REDESIGN.run_agent_figma --data_dir figma_data --output_dir outputs/figma_agent

Features:
- GPU configuration via CLI flags or environment variables
- Processes all episodes in the dataset (no split-specific logic)
- Completed episodes are skipped automatically (resume support)
- Progress logging and per-episode error isolation

Directory Structure:
    Input (the merged 909-episode dataset published on HuggingFace):
        <data_dir>/valid_frames/<episode_id>.json
        <data_dir>/unit_images/<figma_dir>/...        (GT layers + reconstruction)
        <data_dir>/reconstructed_images/<episode_id>.png  (optional convenience copy)

    Output:
        <output_dir>/episodes/<episode_id>/
            - parse.json
            - history_tree.json
            - reconstructed.png
            - episode.log
            - ...
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

# Repository root = parent of the REDESIGN package directory. Required so the
# `python -m REDESIGN.episode_run` worker subprocess can resolve the package.
REPO_ROOT = Path(__file__).resolve().parent.parent

# episode_run.py 실행 설정
DEFAULT_WORKERS = 6
DEFAULT_LLM_LIMIT = 100
DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_LAYERS = 100


# =============================================================================
# GPU Configuration Setup
# =============================================================================

def setup_gpu_config(
    qwen_gpus: Optional[List[int]] = None,
    qwen_pair_size: Optional[int] = None,
    tool_gpus: Optional[List[int]] = None,
    objectclear_gpu: Optional[int] = None,
) -> None:
    """
    런타임 GPU 설정을 적용합니다.
    
    이 함수는 프로세스 시작 시 호출되어 tool_gpu_config를 설정합니다.
    환경 변수보다 명시적 인자가 우선합니다.
    """
    from REDESIGN.tool_gpu_config import set_runtime_config, print_config
    
    # 명시적 인자가 있으면 적용
    if qwen_gpus or qwen_pair_size or tool_gpus or objectclear_gpu:
        set_runtime_config(
            qwen_gpus=qwen_gpus,
            qwen_pair_size=qwen_pair_size,
            tool_gpus=tool_gpus,
            objectclear_gpu=objectclear_gpu,
        )
    
    print_config()


def parse_gpu_list(gpu_str: Optional[str]) -> Optional[List[int]]:
    """GPU 리스트 문자열 파싱 (예: '2,3' -> [2, 3])"""
    if not gpu_str:
        return None
    try:
        return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]
    except ValueError:
        return None


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logger for this run."""
    logger = logging.getLogger("figma_split_runner")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# Path Resolution
# =============================================================================

def get_data_paths(data_dir: Path, output_dir: Path) -> Dict[str, Path]:
    """
    Resolve dataset paths from a clean, split-agnostic Figma dataset directory.

    Expected layout of ``data_dir`` (the merged 909-episode dataset published on
    HuggingFace, or any subset following the same structure)::

        data_dir/
            valid_frames/<episode_id>.json
            unit_images/<figma_dir>/...        (per-episode GT layers + recon)
            reconstructed_images/<episode_id>.png   (optional convenience copy)

    Each ``valid_frames`` JSON stores ``unit_images_dir`` / ``reconstructed_image_path``
    as paths *relative to data_dir*, so the GT reconstruction (the agent's input
    image) resolves to ``data_dir / unit_images_dir / reconstructed_image_path``.
    """
    return {
        "data_dir": data_dir,
        "valid_frames_dir": data_dir / "valid_frames",
        "unit_images_dir": data_dir / "unit_images",
        "output_dir": output_dir,
    }


# =============================================================================
# Frame Processing
# =============================================================================

def get_frame_id_from_json(json_path: Path) -> str:
    """Extract frame ID from JSON filename."""
    return json_path.stem


def get_reconstructed_image_path(json_path: Path, json_data: Dict[str, Any], data_dir: Path) -> Optional[Path]:
    """Get absolute path to the GT reconstruction (agent input) from JSON data."""
    rel_path = json_data.get("reconstructed_image_path")
    unit_images_dir = json_data.get("unit_images_dir")

    if not rel_path or not unit_images_dir:
        return None

    abs_path = data_dir / unit_images_dir / rel_path
    return abs_path


def is_frame_completed(output_dir: Path, frame_id: str) -> Tuple[bool, Optional[Path]]:
    """Check if a frame has already been processed."""
    parse_json_path = output_dir / "episodes" / frame_id / "parse.json"
    
    if parse_json_path.exists():
        return True, parse_json_path
    
    return False, None


def get_completion_stats(output_dir: Path, frame_ids: List[str]) -> Dict[str, Any]:
    """Get completion statistics for a split."""
    completed = []
    pending = []
    
    for frame_id in frame_ids:
        is_done, _ = is_frame_completed(output_dir, frame_id)
        if is_done:
            completed.append(frame_id)
        else:
            pending.append(frame_id)
    
    return {
        "total": len(frame_ids),
        "completed": len(completed),
        "pending": len(pending),
        "completed_ids": completed,
        "pending_ids": pending,
        "completion_rate": len(completed) / len(frame_ids) * 100 if frame_ids else 0,
    }


# =============================================================================
# Episode Runner
# =============================================================================

def run_episode_for_frame(
    image_path: Path,
    output_dir: Path,
    frame_id: str,
    workers: int = DEFAULT_WORKERS,
    llm_limit: int = DEFAULT_LLM_LIMIT,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_layers: int = DEFAULT_MAX_LAYERS,
    gpus: Optional[str] = None,
    qwen_gpus: Optional[str] = None,
    qwen_pair_size: Optional[int] = None,
    tool_gpus: Optional[str] = None,
    timeout: int = 3600,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    단일 프레임에 대해 episode_run.py 실행
    
    GPU 설정은 환경변수로 자식 프로세스에 전달됩니다.
    """
    cmd = [
        sys.executable, "-m", "REDESIGN.episode_run",
        "--image", str(image_path),
        "--output", str(output_dir),
        "--episode_id", frame_id,
        "--parallel",
        "--workers", str(workers),
        "--llm_limit", str(llm_limit),
        "--max_depth", str(max_depth),
        "--max_layers", str(max_layers),
    ]
    if gpus:
        cmd.extend(["--gpus", gpus])
    
    # 환경변수 설정 (GPU 설정 전달)
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
            cwd=str(REPO_ROOT),
            env=env,  # GPU 설정이 포함된 환경변수
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

def run_agent(
    data_dir: Path,
    output_dir: Path,
    workers: int = DEFAULT_WORKERS,
    gpus: Optional[str] = None,
    qwen_gpus: Optional[str] = None,
    qwen_pair_size: Optional[int] = None,
    tool_gpus: Optional[str] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip_completed: bool = True,
) -> Dict[str, Any]:
    """
    Run agent parsing for ALL episodes in a Figma dataset directory.

    This is split-agnostic: it processes every ``valid_frames/*.json`` found in
    ``data_dir`` (e.g. the full 909-episode benchmark, or any subset).
    """
    paths = get_data_paths(data_dir, output_dir)

    if not paths["valid_frames_dir"].exists():
        raise FileNotFoundError(f"valid_frames directory not found: {paths['valid_frames_dir']}")

    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    log_file = paths["output_dir"] / "run.log"
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("Starting ReDesign Agent Runner (Figma)")
    logger.info("=" * 70)
    logger.info(f"valid_frames_dir: {paths['valid_frames_dir']}")
    logger.info(f"output_dir: {paths['output_dir']}")
    logger.info(f"workers: {workers}")
    logger.info(f"qwen_gpus: {qwen_gpus or 'default'}")
    logger.info(f"qwen_pair_size: {qwen_pair_size or 'auto'}")
    logger.info(f"tool_gpus: {tool_gpus or 'default'}")
    logger.info(f"dry_run: {dry_run}, limit: {limit}")
    
    json_files = sorted(paths["valid_frames_dir"].glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files")
    
    if not json_files:
        logger.warning("No JSON files found!")
        return {"error": "No JSON files found"}
    
    frame_ids = [get_frame_id_from_json(f) for f in json_files]
    
    stats = get_completion_stats(paths["output_dir"], frame_ids)
    logger.info(f"Completion: {stats['completed']}/{stats['total']} ({stats['completion_rate']:.1f}%)")
    
    if dry_run:
        logger.info("\n[DRY RUN] Would process these frames:")
        for i, (json_file, frame_id) in enumerate(zip(json_files, frame_ids)):
            is_done, _ = is_frame_completed(paths["output_dir"], frame_id)
            status = "SKIP (completed)" if is_done and skip_completed else "PROCESS"
            logger.info(f"  [{i+1:3d}] {frame_id}: {status}")
            if limit and i + 1 >= limit:
                logger.info(f"  ... (limited to {limit})")
                break
        return stats
    
    results = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "start_time": datetime.now().isoformat(),
        "gpu_config": {
            "qwen_gpus": qwen_gpus,
            "qwen_pair_size": qwen_pair_size,
            "tool_gpus": tool_gpus,
        },
        "processed": [],
        "skipped": [],
        "failed": [],
    }
    
    frames_to_process = []
    for json_file, frame_id in zip(json_files, frame_ids):
        is_done, _ = is_frame_completed(paths["output_dir"], frame_id)
        if is_done and skip_completed:
            results["skipped"].append(frame_id)
            continue
        frames_to_process.append((json_file, frame_id))
    
    if limit:
        frames_to_process = frames_to_process[:limit]
    
    logger.info(f"\nWill process {len(frames_to_process)} frames")
    logger.info(f"Skipping {len(results['skipped'])} already completed frames")
    
    for idx, (json_file, frame_id) in enumerate(frames_to_process):
        logger.info(f"\n{'='*50}")
        logger.info(f"[{idx+1}/{len(frames_to_process)}] Processing: {frame_id}")
        logger.info(f"{'='*50}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            image_path = get_reconstructed_image_path(json_file, json_data, paths["data_dir"])
            if not image_path or not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                results["failed"].append({
                    "frame_id": frame_id,
                    "error": f"Image not found: {image_path}",
                })
                continue
            
            logger.info(f"Input image: {image_path}")
            
            success, message = run_episode_for_frame(
                image_path=image_path,
                output_dir=paths["output_dir"],
                frame_id=frame_id,
                workers=workers,
                gpus=gpus,
                qwen_gpus=qwen_gpus,
                qwen_pair_size=qwen_pair_size,
                tool_gpus=tool_gpus,
                logger=logger,
            )
            
            if success:
                logger.info(f"✓ {message}")
                results["processed"].append({
                    "frame_id": frame_id,
                    "message": message,
                })
            else:
                logger.error(f"✗ {message}")
                results["failed"].append({
                    "frame_id": frame_id,
                    "error": message,
                })
                
        except Exception as e:
            logger.exception(f"Exception processing {frame_id}")
            results["failed"].append({
                "frame_id": frame_id,
                "error": str(e),
            })
        
        results["end_time"] = datetime.now().isoformat()
        results_file = paths["output_dir"] / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Processed: {len(results['processed'])}")
    logger.info(f"Skipped (already done): {len(results['skipped'])}")
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
        description="Run the ReDesign agent on a Figma dataset (split-agnostic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        The runner processes EVERY episode found under <data_dir>/valid_frames/.
        Run it from the repository root so the `REDESIGN` package is importable.

        Examples:
            # Full 909-episode benchmark (downloaded to ./figma_data)
            python -m REDESIGN.run_agent_figma \\
                --data_dir figma_data --output_dir outputs/figma_agent \\
                --qwen_gpus 2,3,4,5 --qwen_pair_size 2 --tool_gpus 6,7

            # Single-pair GPU setup
            python -m REDESIGN.run_agent_figma \\
                --data_dir figma_data --output_dir outputs/figma_agent \\
                --qwen_gpus 3,4,5 --tool_gpus 6,7

            # GPU config via environment variables
            URLD_QWEN_GPUS="3,4,5" URLD_QWEN_PAIR_SIZE="3" URLD_TOOL_GPUS="6,7" \\
                python -m REDESIGN.run_agent_figma --data_dir figma_data --output_dir outputs/figma_agent

            # Dry run / quick test
            python -m REDESIGN.run_agent_figma --data_dir figma_data --output_dir outputs/figma_agent --dry_run
            python -m REDESIGN.run_agent_figma --data_dir figma_data --output_dir outputs/figma_agent --limit 5
                """
            )

    parser.add_argument(
        "--data_dir", "-i",
        type=str,
        required=True,
        help="Path to the Figma dataset directory (containing valid_frames/, unit_images/)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Path to the output directory (an episodes/ subfolder is created here)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})"
    )
    
    # GPU 설정 인자
    parser.add_argument(
        "--gpus", "-g",
        type=str,
        default=None,
        help="GPU IDs for episode_run.py (comma-separated, e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--qwen_gpus",
        type=str,
        default=None,
        help="GPU IDs for Qwen model (comma-separated, e.g., '2,3,4,5')"
    )
    parser.add_argument(
        "--qwen_pair_size",
        type=int,
        default=None,
        help="Number of GPUs per Qwen pair (e.g., 2 for A6000, 3 for RTX3090)"
    )
    parser.add_argument(
        "--tool_gpus",
        type=str,
        default=None,
        help="GPU IDs for Tool models (comma-separated, e.g., '6,7')"
    )
    parser.add_argument(
        "--objectclear_gpu",
        type=int,
        default=None,
        help="GPU ID for ObjectClear model (e.g., 7)"
    )
    
    parser.add_argument(
        "--dry_run", "-d",
        action="store_true",
        help="Show what would be done without actually running"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of frames to process (for testing)"
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Don't skip already completed frames (re-process all)"
    )

    args = parser.parse_args()
    
    # GPU 설정 적용
    qwen_gpu_list = parse_gpu_list(args.qwen_gpus)
    tool_gpu_list = parse_gpu_list(args.tool_gpus)
    
    setup_gpu_config(
        qwen_gpus=qwen_gpu_list,
        qwen_pair_size=args.qwen_pair_size,
        tool_gpus=tool_gpu_list,
        objectclear_gpu=args.objectclear_gpu,
    )
    
    try:
        results = run_agent(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            workers=args.workers,
            gpus=args.gpus,
            qwen_gpus=args.qwen_gpus,
            qwen_pair_size=args.qwen_pair_size,
            tool_gpus=args.tool_gpus,
            dry_run=args.dry_run,
            limit=args.limit,
            skip_completed=not args.no_skip,
        )

        if not args.dry_run:
            print(f"\nResults saved to: {Path(args.output_dir) / 'results.json'}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()