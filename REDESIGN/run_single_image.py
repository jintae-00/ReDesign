#!/usr/bin/env python3
"""
run_single_image.py - 단일 이미지에 대해 Agent Parsing (episode_run.py) 실행

Usage:
    # src 폴더에서 실행 시:
    python -m REDESIGN.run_single_image

    # GPU 설정 예시:
    python -m REDESIGN.run_single_image --qwen_gpus 3,4,5,7 --tool_gpus 6,7
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

# 처리할 이미지 파일명 (src 폴더 기준 상대 경로)
TARGET_IMAGE_FILENAME = "Figma_01.png"

# 결과가 저장될 기본 디렉토리 (src 기준 상대 경로)
OUTPUT_BASE_DIR = "figma_experiment/single_test"

# 실행 설정
DEFAULT_WORKERS = 6
DEFAULT_LLM_LIMIT = 100
DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_LAYERS = 100


# =============================================================================
# Path Resolution
# =============================================================================

def get_src_root() -> Path:
    """
    src 디렉토리 루트를 찾습니다.
    현재 작업 디렉토리(CWD)가 src인 경우 그대로 반환하고,
    아니면 부모 디렉토리를 탐색합니다.
    """
    current = Path.cwd()
    
    # 1. 현재 경로가 src인 경우 (가장 일반적)
    if current.name == "src":
        return current
    
    # 2. 현재 경로 하위에 src가 있는 경우
    if (current / "src").exists():
        return current / "src"
        
    # 3. 상위 경로 탐색 (REDESIGN 내부에서 실행되는 경우 등 대비)
    # 다만 python -m 으로 실행 시 CWD는 보통 실행 위치입니다.
    return current

def resolve_paths():
    src_root = get_src_root()
    
    # 이미지 경로 찾기 (src/Figma_01.png)
    image_path = src_root / TARGET_IMAGE_FILENAME
    
    # 에피소드 ID (파일명 사용)
    episode_id = image_path.stem
        
    # 출력 경로 설정 (src/figma_experiment/single_test)
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
    """런타임 GPU 설정을 적용합니다."""
    try:
        from REDESIGN.tool_gpu_config import set_runtime_config, print_config
        
        if qwen_gpus or qwen_pair_size or tool_gpus or objectclear_gpu:
            set_runtime_config(
                qwen_gpus=qwen_gpus,
                qwen_pair_size=qwen_pair_size,
                tool_gpus=tool_gpus,
                objectclear_gpu=objectclear_gpu,
            )
        # 로그가 중복될 수 있으므로 필요시 주석 처리
        # print_config() 
    except ImportError:
        pass # REDESIGN 패키지 내부가 아닌 경우 무시

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

    # 환경변수 설정
    env = os.environ.copy()
    if qwen_gpus:
        env["URLD_QWEN_GPUS"] = qwen_gpus
    if qwen_pair_size:
        env["URLD_QWEN_PAIR_SIZE"] = str(qwen_pair_size)
    if tool_gpus:
        env["URLD_TOOL_GPUS"] = tool_gpus

    # 중요: PYTHONPATH에 src_root 추가 (서브프로세스가 모듈을 찾을 수 있도록)
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n{'='*60}")
    print(f"Running Episode: {episode_id}")
    print(f"CWD (Subprocess): {src_root}")
    print(f"Image Path: {image_path}")
    print(f"Output Dir: {output_dir / episode_id}")
    print(f"{'='*60}\n")

    try:
        # cwd를 src_root(src 폴더)로 설정해야 REDESIGN 모듈을 찾을 수 있습니다.
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
    parser = argparse.ArgumentParser(description="Run Agent Parsing for Single Image")
    parser.add_argument("--qwen_gpus", type=str, default=None)
    parser.add_argument("--qwen_pair_size", type=int, default=None)
    parser.add_argument("--tool_gpus", type=str, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    
    args = parser.parse_args()

    # 1. 경로 확인
    src_root, image_path, output_dir, episode_id = resolve_paths()

    if not image_path.exists():
        print(f"Error: Image file not found at: {image_path}")
        print(f"Please place 'Figma_01.png' in the 'src' folder.")
        sys.exit(1)

    # 2. GPU Config 설정
    setup_gpu_config(
        qwen_gpus=parse_gpu_list(args.qwen_gpus),
        qwen_pair_size=args.qwen_pair_size,
        tool_gpus=parse_gpu_list(args.tool_gpus)
    )

    # 3. 실행
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