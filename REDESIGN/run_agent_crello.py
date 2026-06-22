#!/usr/bin/env python3
"""
run_agent_crello.py - Run the ReDesign agent on a Crello dataset (split-agnostic)

Processes EVERY crello_test_* record directory under <data_dir>. Run from the
repository root so the `REDESIGN` package is importable.

Usage:
    python -m REDESIGN.run_agent_crello \
        --data_dir crello_data/records --output_dir outputs/crello_agent \
        --qwen_gpus 2,3,4,5 --qwen_pair_size 2 --tool_gpus 6,7

    # GPU config via environment variables
    URLD_QWEN_GPUS="3,4,5" URLD_TOOL_GPUS="6,7" \
        python -m REDESIGN.run_agent_crello --data_dir crello_data/records --output_dir outputs/crello_agent

Features:
- GPU configuration via CLI flags or environment variables
- Processes all records in the dataset (no split-specific logic)
- Completed records are skipped automatically (resume support)
- Progress logging and per-record error isolation

Directory Structure:
    Input:
        <data_dir>/crello_test_XXXX/
            - composite.png          (agent parsing input)
            - gt_metadata.json       (GT metadata, used by evaluation)
            - elements/element_*.png (GT element images)

    Output:
        <output_dir>/episodes/<record_id>/
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
    """
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
    logger = logging.getLogger("crello_split_runner")
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
    Resolve dataset paths from a Crello dataset directory.

    Expected layout of ``data_dir``::

        data_dir/
            crello_test_0001/composite.png
            crello_test_0002/composite.png
            ...

    where each ``crello_test_*`` record directory holds the rendered design
    canvas as ``composite.png`` (the agent's input image). See the Crello
    download guide in the repository for how to obtain/build this structure.
    """
    return {
        "data_dir": data_dir,
        "output_dir": output_dir,
    }


# =============================================================================
# Record Processing
# =============================================================================

def get_record_dirs(split_dir: Path) -> List[Path]:
    """split 디렉토리에서 crello_test_XXXX 디렉토리들을 정렬 순서로 반환."""
    dirs = sorted([
        d for d in split_dir.iterdir()
        if (d.is_dir() or d.is_symlink()) and d.name.startswith("crello_test_")
    ])
    return dirs


def get_record_id(record_dir: Path) -> str:
    """디렉토리명이 곧 record_id."""
    return record_dir.name


def get_composite_image_path(record_dir: Path) -> Optional[Path]:
    """record 디렉토리에서 composite.png 경로 반환."""
    composite_path = record_dir / "composite.png"
    if composite_path.exists():
        return composite_path
    return None


def is_record_completed(output_dir: Path, record_id: str) -> Tuple[bool, Optional[Path]]:
    """Check if a record has already been processed."""
    parse_json_path = output_dir / "episodes" / record_id / "parse.json"
    if parse_json_path.exists():
        return True, parse_json_path
    return False, None


def get_completion_stats(output_dir: Path, record_ids: List[str]) -> Dict[str, Any]:
    """Get completion statistics for a split."""
    completed = []
    pending = []
    
    for record_id in record_ids:
        is_done, _ = is_record_completed(output_dir, record_id)
        if is_done:
            completed.append(record_id)
        else:
            pending.append(record_id)
    
    return {
        "total": len(record_ids),
        "completed": len(completed),
        "pending": len(pending),
        "completed_ids": completed,
        "pending_ids": pending,
        "completion_rate": len(completed) / len(record_ids) * 100 if record_ids else 0,
    }


# =============================================================================
# Episode Runner
# =============================================================================

def run_episode_for_record(
    image_path: Path,
    output_dir: Path,
    record_id: str,
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
    단일 record에 대해 episode_run.py 실행
    """
    cmd = [
        sys.executable, "-m", "REDESIGN.episode_run",
        "--image", str(image_path),
        "--output", str(output_dir),
        "--episode_id", record_id,
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
    Run agent parsing for ALL records in a Crello dataset directory.

    This is split-agnostic: it processes every ``crello_test_*`` record directory
    found in ``data_dir``.
    """
    paths = get_data_paths(data_dir, output_dir)

    if not paths["data_dir"].exists():
        raise FileNotFoundError(f"Dataset directory not found: {paths['data_dir']}")

    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    log_file = paths["output_dir"] / "run.log"
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("Starting ReDesign Agent Runner (Crello)")
    logger.info("=" * 70)
    logger.info(f"data_dir: {paths['data_dir']}")
    logger.info(f"output_dir: {paths['output_dir']}")
    logger.info(f"workers: {workers}")
    logger.info(f"qwen_gpus: {qwen_gpus or 'default'}")
    logger.info(f"qwen_pair_size: {qwen_pair_size or 'auto'}")
    logger.info(f"tool_gpus: {tool_gpus or 'default'}")
    logger.info(f"dry_run: {dry_run}, limit: {limit}")
    
    # ---- Crello: enumerate crello_test_* record directories ----
    record_dirs = get_record_dirs(paths["data_dir"])
    record_ids = [get_record_id(d) for d in record_dirs]
    logger.info(f"Found {len(record_dirs)} record directories")
    
    if not record_dirs:
        logger.warning("No crello_test_* directories found!")
        return {"error": "No record directories found"}
    
    stats = get_completion_stats(paths["output_dir"], record_ids)
    logger.info(f"Completion: {stats['completed']}/{stats['total']} ({stats['completion_rate']:.1f}%)")
    
    if dry_run:
        logger.info("\n[DRY RUN] Would process these records:")
        for i, (record_dir, record_id) in enumerate(zip(record_dirs, record_ids)):
            is_done, _ = is_record_completed(paths["output_dir"], record_id)
            status = "SKIP (completed)" if is_done and skip_completed else "PROCESS"
            logger.info(f"  [{i+1:3d}] {record_id}: {status}")
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
    
    records_to_process = []
    for record_dir, record_id in zip(record_dirs, record_ids):
        is_done, _ = is_record_completed(paths["output_dir"], record_id)
        if is_done and skip_completed:
            results["skipped"].append(record_id)
            continue
        records_to_process.append((record_dir, record_id))
    
    if limit:
        records_to_process = records_to_process[:limit]
    
    logger.info(f"\nWill process {len(records_to_process)} records")
    logger.info(f"Skipping {len(results['skipped'])} already completed records")
    
    for idx, (record_dir, record_id) in enumerate(records_to_process):
        logger.info(f"\n{'='*50}")
        logger.info(f"[{idx+1}/{len(records_to_process)}] Processing: {record_id}")
        logger.info(f"{'='*50}")
        
        try:
            # ---- Figma와의 핵심 차이: 이미지 경로 결정 ----
            image_path = get_composite_image_path(record_dir)
            if not image_path:
                logger.error(f"composite.png not found in {record_dir}")
                results["failed"].append({
                    "record_id": record_id,
                    "error": f"composite.png not found: {record_dir}",
                })
                continue
            
            logger.info(f"Input image: {image_path}")
            
            success, message = run_episode_for_record(
                image_path=image_path,
                output_dir=paths["output_dir"],
                record_id=record_id,
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
                    "record_id": record_id,
                    "message": message,
                })
            else:
                logger.error(f"✗ {message}")
                results["failed"].append({
                    "record_id": record_id,
                    "error": message,
                })
                
        except Exception as e:
            logger.exception(f"Exception processing {record_id}")
            results["failed"].append({
                "record_id": record_id,
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
        logger.info("\nFailed records:")
        for f in results["failed"]:
            logger.info(f"  - {f['record_id']}: {f['error'][:100]}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the ReDesign agent on a Crello dataset (split-agnostic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The runner processes EVERY crello_test_* record directory under <data_dir>.
Run it from the repository root so the `REDESIGN` package is importable.

Examples:
    # All records in the dataset directory
    python -m REDESIGN.run_agent_crello \\
        --data_dir crello_data/records --output_dir outputs/crello_agent \\
        --qwen_gpus 2,3,4,5 --qwen_pair_size 2 --tool_gpus 6,7

    # Dry run / quick test
    python -m REDESIGN.run_agent_crello --data_dir crello_data/records --output_dir outputs/crello_agent --dry_run
    python -m REDESIGN.run_agent_crello --data_dir crello_data/records --output_dir outputs/crello_agent --limit 5
        """
    )

    parser.add_argument(
        "--data_dir", "-i",
        type=str,
        required=True,
        help="Path to the Crello dataset directory (containing crello_test_*/ record dirs)"
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
    parser.add_argument("--gpus", "-g", type=str, default=None,
                        help="GPU IDs for episode_run.py (comma-separated)")
    parser.add_argument("--qwen_gpus", type=str, default=None,
                        help="GPU IDs for Qwen model (comma-separated)")
    parser.add_argument("--qwen_pair_size", type=int, default=None,
                        help="Number of GPUs per Qwen pair")
    parser.add_argument("--tool_gpus", type=str, default=None,
                        help="GPU IDs for Tool models (comma-separated)")
    parser.add_argument("--objectclear_gpu", type=int, default=None,
                        help="GPU ID for ObjectClear model")
    
    parser.add_argument("--dry_run", "-d", action="store_true",
                        help="Show what would be done without actually running")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit number of records to process (for testing)")
    parser.add_argument("--no_skip", action="store_true",
                        help="Don't skip already completed records (re-process all)")

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