#!/usr/bin/env python3
"""
run_single_image.py - Run agent parsing (episode_run.py) on a single image.

Usage:
    # Run from inside the "src" folder:
    python -m REDESIGN.run_single_image

    # Example with GPU configuration:
    python -m REDESIGN.run_single_image --qwen_gpus <QWEN_GPU_IDS> --tool_gpus <TOOL_GPU_IDS>

    # Replace <QWEN_GPU_IDS> and <TOOL_GPU_IDS> with your own comma-separated
    # GPU ids (e.g. "0,1").
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional

# =============================================================================
# User Configuration (Hardcoded)
# =============================================================================

# Image filename to process (path relative to the "src" folder)
TARGET_IMAGE_FILENAME = "Figma_01.png"

# Base directory where results are saved (relative to "src")
OUTPUT_BASE_DIR = "figma_experiment/single_test"

# Run settings
DEFAULT_WORKERS = 6
DEFAULT_LLM_LIMIT = 100
DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_LAYERS = 100


# =============================================================================
# Path Resolution
# =============================================================================

def get_src_root() -> Path:
    """
    Locate the "src" directory root.
    If the current working directory (CWD) is "src", return it as is;
    otherwise look for a "src" subdirectory.
    """
    current = Path.cwd()

    # 1. The current path is "src" (the most common case)
    if current.name == "src":
        return current

    # 2. The current path contains a "src" subdirectory
    if (current / "src").exists():
        return current / "src"

    # 3. Fall back to the current path (e.g. when run from inside REDESIGN).
    # Note: with "python -m", the CWD is usually the launch location.
    return current

def resolve_paths():
    src_root = get_src_root()

    # Resolve the image path (src/Figma_01.png)
    image_path = src_root / TARGET_IMAGE_FILENAME

    # Episode ID (derived from the filename)
    episode_id = image_path.stem

    # Output path (src/figma_experiment/single_test)
    output_dir = src_root / OUTPUT_BASE_DIR
    
    return src_root, image_path, output_dir, episode_id


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
    src_root: Path,
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

    # Important: prepend src_root to PYTHONPATH so the subprocess can import the modules
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n{'='*60}")
    print(f"Running Episode: {episode_id}")
    print(f"CWD (Subprocess): {src_root}")
    print(f"Image Path: {image_path}")
    print(f"Output Dir: {output_dir / episode_id}")
    print(f"{'='*60}\n")

    try:
        # The cwd must be src_root (the "src" folder) so the REDESIGN module can be found.
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(src_root), 
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
    parser = argparse.ArgumentParser(description="Run agent parsing on a single image")
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
    src_root, image_path, output_dir, episode_id = resolve_paths()

    if not image_path.exists():
        print(f"Error: Image file not found at: {image_path}")
        print(f"Please place 'Figma_01.png' in the 'src' folder.")
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
        src_root=src_root,
        workers=args.workers,
        qwen_gpus=args.qwen_gpus,
        qwen_pair_size=args.qwen_pair_size,
        tool_gpus=args.tool_gpus
    )

if __name__ == "__main__":
    main()