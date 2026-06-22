#!/usr/bin/env python3
"""
run_single_image.py - Run the ReDesign agent on a single image.

Run from the repository root so the `REDESIGN` package is importable.

Usage:
    python -m REDESIGN.run_single_image \
        --image path/to/design.png --output_dir outputs/single \
        --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --tool_gpus <TOOL_GPU_IDS>

    # Replace <QWEN_GPU_IDS> / <TOOL_GPU_IDS> with your own comma-separated GPU
    # ids. The Qwen layered model needs ~2 GPUs (see README "Compute & API
    # configuration"); the tools need one more.

Output: <output_dir>/episodes/<image_stem>/ (parse.json, history_tree.json, ...).
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional

# Repository root = parent of the REDESIGN package directory. Required so the
# `python -m REDESIGN.episode_run` worker subprocess can resolve the package.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Run settings
DEFAULT_WORKERS = 6
DEFAULT_LLM_LIMIT = 100
DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_LAYERS = 100


# =============================================================================
# GPU Configuration Helper
# =============================================================================

def setup_gpu_config(
    qwen_gpus: Optional[List[int]] = None,
    qwen_pair_size: Optional[int] = None,
    tool_gpus: Optional[List[int]] = None,
    objectclear_gpu: Optional[int] = None,
) -> None:
    """Apply the runtime GPU configuration."""
    try:
        from REDESIGN.tool_gpu_config import set_runtime_config, print_config

        if qwen_gpus or qwen_pair_size or tool_gpus or objectclear_gpu:
            set_runtime_config(
                qwen_gpus=qwen_gpus,
                qwen_pair_size=qwen_pair_size,
                tool_gpus=tool_gpus,
                objectclear_gpu=objectclear_gpu,
            )
        # Commented out to avoid duplicate logging; enable if needed.
        # print_config()
    except ImportError:
        pass  # Ignore when not running inside the REDESIGN package

def parse_gpu_list(gpu_str: Optional[str]) -> Optional[List[int]]:
    if not gpu_str: return None
    try:
        return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]
    except ValueError: return None


# =============================================================================
# Runner Logic
# =============================================================================

def run_single_episode(
    image_path: Path,
    output_dir: Path,
    episode_id: str,
    workers: int,
    qwen_gpus: Optional[str],
    qwen_pair_size: Optional[int],
    tool_gpus: Optional[str],
) -> None:
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "REDESIGN.episode_run",
        "--image", str(image_path),
        "--output", str(output_dir),
        "--episode_id", episode_id,
        "--parallel",
        "--workers", str(workers),
        "--llm_limit", str(DEFAULT_LLM_LIMIT),
        "--max_depth", str(DEFAULT_MAX_DEPTH),
        "--max_layers", str(DEFAULT_MAX_LAYERS),
    ]

    # Set up environment variables
    env = os.environ.copy()
    if qwen_gpus:
        env["URLD_QWEN_GPUS"] = qwen_gpus
    if qwen_pair_size:
        env["URLD_QWEN_PAIR_SIZE"] = str(qwen_pair_size)
    if tool_gpus:
        env["URLD_TOOL_GPUS"] = tool_gpus

    # Prepend the repo root to PYTHONPATH so the subprocess can import REDESIGN.
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n{'='*60}")
    print(f"Running Episode: {episode_id}")
    print(f"CWD (Subprocess): {REPO_ROOT}")
    print(f"Image Path: {image_path}")
    print(f"Output Dir: {output_dir / episode_id}")
    print(f"{'='*60}\n")

    try:
        # The cwd must be the repo root so the REDESIGN package can be imported.
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
            env=env,
        )

        for line in process.stdout:
            print(line, end='')

        process.communicate()
        
        if process.returncode == 0:
            print(f"\nSUCCESS: Processing completed for {episode_id}")
            print(f"Result: {output_dir / episode_id / 'parse.json'}")
        else:
            print(f"\nFAILURE: Process exited with code {process.returncode}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        process.kill()
    except Exception as e:
        print(f"\nError: {e}")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run the ReDesign agent on a single image")
    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="Path to the input design image (PNG/JPG).",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True,
        help="Output directory; results go to <output_dir>/episodes/<image_stem>/.",
    )
    parser.add_argument(
        "--qwen_gpus", type=str, default=None,
        help="Comma-separated, user-specific GPU ids to run the Qwen model on "
             "(e.g. \"0,1\"). Defaults to the configured value when omitted.",
    )
    parser.add_argument(
        "--qwen_pair_size", type=int, default=None,
        help="Number of GPUs to group together per Qwen worker.",
    )
    parser.add_argument(
        "--tool_gpus", type=str, default=None,
        help="Comma-separated, user-specific GPU ids for the auxiliary tools "
             "(e.g. \"0,1\"). Defaults to the configured value when omitted.",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Number of parallel worker processes.",
    )

    args = parser.parse_args()

    # 1. Resolve paths
    image_path = Path(args.image).resolve()
    output_dir = Path(args.output_dir).resolve()
    episode_id = image_path.stem

    if not image_path.exists():
        print(f"Error: input image not found: {image_path}")
        sys.exit(1)

    # 2. Apply GPU configuration
    setup_gpu_config(
        qwen_gpus=parse_gpu_list(args.qwen_gpus),
        qwen_pair_size=args.qwen_pair_size,
        tool_gpus=parse_gpu_list(args.tool_gpus)
    )

    # 3. Run
    run_single_episode(
        image_path=image_path,
        output_dir=output_dir,
        episode_id=episode_id,
        workers=args.workers,
        qwen_gpus=args.qwen_gpus,
        qwen_pair_size=args.qwen_pair_size,
        tool_gpus=args.tool_gpus
    )

if __name__ == "__main__":
    main()