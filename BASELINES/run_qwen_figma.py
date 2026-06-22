#!/usr/bin/env python3
"""
run_qwen_baseline.py - Qwen Image Layered Baseline 실험 실행

Qwen Image Layered 모델을 baseline으로 특정 split에 대해 실행합니다.
주어진 qwen_gpus를 qwen_pair_size로 나눠서 병렬 처리합니다.

Usage:
    # 기본 사용
    python run_qwen_baseline.py --split_idx 0 --qwen_gpus 3,4,5,6 --qwen_pair_size 2
    
    # GPU 3,4와 5,6 두 pair로 나눠서 split 0을 병렬 처리
    # → pair (3,4): frame 0, 2, 4, ...
    # → pair (5,6): frame 1, 3, 5, ...
    
    # Dry run (실행 없이 확인)
    python run_qwen_baseline.py --split_idx 0 --qwen_gpus 3,4,5,6 --qwen_pair_size 2 --dry_run
    
    # 테스트용 (처음 10개만)
    python run_qwen_baseline.py --split_idx 0 --qwen_gpus 3,4 --qwen_pair_size 2 --limit 10

Directory Structure:
    Input:
        src/figma_data/process/subset/dino90_obj_5_25_char_50_split_{idx}/valid_frames/*.json
        src/figma_data/process/subset/dino90_obj_5_25_char_50_split_{idx}/unit_images/...
    
    Output:
        src/qwen_experiment/split_{idx}/{frame_id}/
            - input.png          # 원본 입력 이미지
            - layer_00.png
            - layer_01.png
            - ...
            - metadata.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import os
import logging
import multiprocessing as mp
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import traceback
import cv2
import numpy as np
from PIL import Image, ImageFilter


# =============================================================================
# Configuration
# =============================================================================

FIGMA_DATA_BASE = "figma_data/process/subset"
QWEN_EXPERIMENT_BASE = "qwen_experiment_0208"
SPLIT_PREFIX = "dino80_obj_5_60_char_25_split_"
NUM_TOTAL_SPLITS = 4  # 합칠 split 수

# Qwen 기본 파라미터
DEFAULT_NUM_LAYERS = 4
DEFAULT_SEED = 777
DEFAULT_RESOLUTION = 640
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_ALPHA_THRESHOLD = 0

# Input image 파일명
INPUT_IMAGE_NAME = "input.png"


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Path, name: str = "qwen_baseline") -> logging.Logger:
    """Setup logger for this run."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console)
    
    # File handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# Path Resolution
# =============================================================================

def get_src_root() -> Path:
    """Get the src directory root."""
    current = Path(__file__).resolve().parent
    
    if current.name == "src":
        return current
    elif (current / "src").exists():
        return current / "src"
    else:
        # 현재 위치가 src 내부라고 가정
        return current


def get_all_split_paths(src_root: Path) -> Dict[str, Any]:
    """모든 split의 경로를 합쳐서 반환"""
    all_valid_frames_dirs = []
    all_split_dirs = []
    
    for i in range(NUM_TOTAL_SPLITS):
        split_name = f"{SPLIT_PREFIX}{i}"
        split_dir = src_root / FIGMA_DATA_BASE / split_name
        all_split_dirs.append(split_dir)
        all_valid_frames_dirs.append(split_dir / "valid_frames")
    
    return {
        "split_dirs": all_split_dirs,
        "valid_frames_dirs": all_valid_frames_dirs,
        "output_dir": src_root / QWEN_EXPERIMENT_BASE / "all_splits",
    }


# =============================================================================
# GPU Configuration
# =============================================================================

def parse_gpu_list(gpu_str: Optional[str]) -> List[int]:
    """GPU 리스트 문자열 파싱 (예: '2,3,4,5' -> [2, 3, 4, 5])"""
    if not gpu_str:
        return []
    try:
        return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]
    except ValueError:
        return []


def create_gpu_pairs(gpu_ids: List[int], pair_size: int) -> List[Tuple[int, ...]]:
    """GPU ID 리스트를 pair_size 크기의 튜플들로 분할"""
    if not gpu_ids or pair_size <= 0:
        return []
    
    pairs = []
    for i in range(0, len(gpu_ids), pair_size):
        pair = tuple(gpu_ids[i:i + pair_size])
        if len(pair) == pair_size:  # 완전한 pair만 사용
            pairs.append(pair)
    
    return pairs


# =============================================================================
# Frame Data Loading
# =============================================================================

@dataclass
class FrameInfo:
    """프레임 정보를 담는 데이터 클래스"""
    frame_id: str
    json_path: Path
    image_path: Path


def load_frame_list(paths: Dict[str, Any]) -> List[FrameInfo]:
    """모든 split의 valid_frames에서 프레임 목록 로드"""
    frames = []
    
    for split_dir, vf_dir in zip(paths["split_dirs"], paths["valid_frames_dirs"]):
        if not vf_dir.exists():
            print(f"[Warning] Not found: {vf_dir}")
            continue
        
        json_files = sorted(vf_dir.glob("*.json"))
        
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
                        frames.append(FrameInfo(
                            frame_id=frame_id,
                            json_path=json_path,
                            image_path=image_path,
                        ))
            except Exception as e:
                print(f"[Warning] Failed to load {json_path}: {e}")
                continue
    
    return frames


def is_frame_completed(output_dir: Path, frame_id: str) -> bool:
    """프레임이 이미 처리되었는지 확인"""
    metadata_path = output_dir / frame_id / "metadata.json"
    return metadata_path.exists()


def is_input_image_missing(output_dir: Path, frame_id: str) -> bool:
    """완료된 프레임에 input image가 없는지 확인"""
    input_image_path = output_dir / frame_id / INPUT_IMAGE_NAME
    return not input_image_path.exists()


def create_reconstructions(
    frame_output_dir: Path, 
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    저장된 layer 이미지들을 사용하여 reconstructed.png와 reconstructed_bordered.png를 생성합니다.
    REDESIGN/reconstruction.py의 로직을 Qwen Layer 구조에 맞게 적용했습니다.
    """
    try:
        # 1. Layer 이미지 찾기 및 정렬
        layer_paths = sorted(frame_output_dir.glob("layer_*.png"))
        if not layer_paths:
            return False

        # 2. 캔버스 준비 (첫 번째 레이어 기준 크기)
        base_layer = Image.open(layer_paths[0]).convert("RGBA")
        canvas_w, canvas_h = base_layer.size
        
        # ---------------------------------------------------------
        # A. Vanilla Reconstruction (reconstructed.png)
        # ---------------------------------------------------------
        reconstructed = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        
        layers_data = [] # (image_path, pil_image) 튜플 저장
        
        for p in layer_paths:
            img = Image.open(p).convert("RGBA")
            if img.size != (canvas_w, canvas_h):
                img = img.resize((canvas_w, canvas_h), Image.LANCZOS)
            reconstructed = Image.alpha_composite(reconstructed, img)
            layers_data.append((p, img))
            
        reconstructed_path = frame_output_dir / "reconstructed.png"
        reconstructed.save(reconstructed_path)
        
        # ---------------------------------------------------------
        # B. Bordered Reconstruction (reconstructed_bordered.png)
        # ---------------------------------------------------------
        # 설정 (reconstruction.py와 동일)
        border_color = (255, 150, 200, 200)  # Light Pink
        glow_color = (255, 180, 220, 100)    # Soft Pink
        border_width = 3
        glow_width = 5
        
        # 복사본 생성
        result = reconstructed.copy()
        
        # Glow Layer (Blur 효과용)
        glow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        # Border Layer (선명한 외곽선용)
        border_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        
        glow_arr_base = np.array(glow_layer)
        border_arr_base = np.array(border_layer)
        
        for _, layer_img in layers_data:
            # Alpha 채널 추출
            elem_arr = np.array(layer_img)
            alpha = elem_arr[:, :, 3]
            
            # Binary Mask 생성 (Threshold 128)
            binary_mask = (alpha > 128).astype(np.uint8) * 255
            
            # OpenCV로 윤곽선 찾기
            contours, _ = cv2.findContours(
                binary_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
                
            # Glow 그리기 (여러 번 겹쳐서 그라데이션 효과)
            for i in range(glow_width, 0, -1):
                alpha_val = int(glow_color[3] * (1 - i / (glow_width + 2)))
                cv2.drawContours(
                    glow_arr_base, 
                    contours, 
                    -1, 
                    (*glow_color[:3], alpha_val),
                    thickness=i * 2
                )
            
            # Border 그리기
            cv2.drawContours(
                border_arr_base, 
                contours, 
                -1, 
                border_color,
                thickness=border_width
            )
            
        # 배열을 이미지로 변환
        glow_layer = Image.fromarray(glow_arr_base)
        border_layer = Image.fromarray(border_arr_base)
        
        # Glow에 Blur 적용
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=4))
        
        # 합성: Base -> Glow -> Border
        result = Image.alpha_composite(result, glow_layer)
        result = Image.alpha_composite(result, border_layer)
        
        bordered_path = frame_output_dir / "reconstructed_bordered.png"
        result.save(bordered_path)
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to create reconstructions for {frame_output_dir.name}: {e}")
        else:
            print(f"Failed to create reconstructions: {e}")
        return False


# =============================================================================
# Input Image Backfill
# =============================================================================


def backfill_missing_files(
    frames: List[FrameInfo],
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, int]:
    """
    완료된 프레임들 중 누락된 파일(input image, reconstruction)을 보충
    """
    stats = {"input_image": 0, "reconstruction": 0}
    
    for frame in frames:
        frame_output_dir = output_dir / frame.frame_id
        
        # 완료된 프레임인지 확인 (metadata 존재 여부)
        if not is_frame_completed(output_dir, frame.frame_id):
            continue
        
        # 1. Input Image 보충
        input_image_path = frame_output_dir / INPUT_IMAGE_NAME
        if not input_image_path.exists():
            try:
                shutil.copy2(frame.image_path, input_image_path)
                stats["input_image"] += 1
                logger.debug(f"Backfilled input image for {frame.frame_id}")
            except Exception as e:
                logger.warning(f"Failed to backfill input image for {frame.frame_id}: {e}")

        # 2. Reconstruction Image 보충
        recon_path = frame_output_dir / "reconstructed.png"
        border_path = frame_output_dir / "reconstructed_bordered.png"
        
        if not recon_path.exists() or not border_path.exists():
            # Layer 파일들이 존재하는지 확인
            if list(frame_output_dir.glob("layer_*.png")):
                success = create_reconstructions(frame_output_dir, logger)
                if success:
                    stats["reconstruction"] += 1
                    logger.debug(f"Backfilled reconstructions for {frame.frame_id}")
    
    return stats


# =============================================================================
# Qwen Worker Process
# =============================================================================

def worker_process(
    worker_id: int,
    gpu_pair: Tuple[int, ...],
    frame_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: Path,
    qwen_params: Dict[str, Any],
):
    """
    개별 GPU pair에서 Qwen 모델을 실행하는 워커 프로세스
    """
    import os
    import gc
    import torch
    import tempfile
    import shutil
    from PIL import Image
    import numpy as np
    
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_pair))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    offload_dir = tempfile.mkdtemp(prefix=f"qwen_baseline_{worker_id}_")
    pipeline = None
    
    try:
        # Pipeline 로드
        print(f"[Worker {worker_id}] Loading Qwen pipeline on GPUs {gpu_pair}...")
        from diffusers import QwenImageLayeredPipeline
        
        pipeline = QwenImageLayeredPipeline.from_pretrained(
            "Qwen/Qwen-Image-Layered",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            offload_folder=offload_dir,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )
        print(f"[Worker {worker_id}] Pipeline loaded successfully")
        
        # 프레임 처리 루프
        while True:
            try:
                item = frame_queue.get(timeout=1.0)
            except:
                # 큐가 비었는지 확인
                if frame_queue.empty():
                    break
                continue
            
            if item is None:  # 종료 신호
                break
            
            frame_id, image_path = item
            start_time = time.time()
            
            try:
                # 출력 디렉토리 생성
                frame_output_dir = output_dir / frame_id
                frame_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 이미지 로드
                image = Image.open(image_path).convert("RGBA")
                original_size = image.size
                
                # Input image 저장 (원본 복사)
                input_image_dest = frame_output_dir / INPUT_IMAGE_NAME
                shutil.copy2(image_path, input_image_dest)
                
                # Qwen 추론
                inputs = {
                    "image": image,
                    "generator": torch.Generator(device="cpu").manual_seed(qwen_params["seed"]),
                    "num_inference_steps": qwen_params["num_inference_steps"],
                    "layers": qwen_params["num_layers"],
                    "resolution": qwen_params["resolution"],
                    "true_cfg_scale": qwen_params["true_cfg_scale"],
                    "cfg_normalize": True,
                    "use_en_prompt": True,
                }
                
                with torch.inference_mode():
                    output = pipeline(**inputs)
                    output_images = output.images[0]
                
                # 레이어 저장
                layer_paths = []
                for i, layer_img in enumerate(output_images):
                    layer_img = layer_img.convert("RGBA")
                    if layer_img.size != original_size:
                        layer_img = layer_img.resize(original_size, Image.LANCZOS)
                    
                    # 반투명 픽셀 필터링
                    arr = np.array(layer_img)
                    alpha_threshold = qwen_params["alpha_threshold"]
                    mask = arr[:, :, 3] < alpha_threshold
                    arr[mask] = [0, 0, 0, 0]
                    
                    layer_path = frame_output_dir / f"layer_{i:02d}.png"
                    Image.fromarray(arr).save(layer_path)
                    layer_paths.append(str(layer_path))
                
                # 메타데이터 저장
                elapsed = time.time() - start_time
                metadata = {
                    "frame_id": frame_id,
                    "source_image": str(image_path),
                    "input_image": str(input_image_dest),
                    "original_size": list(original_size),
                    "num_layers": len(layer_paths),
                    "layer_paths": layer_paths,
                    "gpu_pair": list(gpu_pair),
                    "worker_id": worker_id,
                    "processing_time_sec": elapsed,
                    "timestamp": datetime.now().isoformat(),
                    "qwen_params": qwen_params,
                }
                
                metadata_path = frame_output_dir / "metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                create_reconstructions(frame_output_dir)
                
                result_queue.put({
                    "worker_id": worker_id,
                    "frame_id": frame_id,
                    "success": True,
                    "num_layers": len(layer_paths),
                    "elapsed": elapsed,
                })
                
            except Exception as e:
                result_queue.put({
                    "worker_id": worker_id,
                    "frame_id": frame_id,
                    "success": False,
                    "error": str(e),
                    "trace": traceback.format_exc(),
                })
            
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        traceback.print_exc()
    
    finally:
        if pipeline is not None:
            del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        if os.path.exists(offload_dir):
            shutil.rmtree(offload_dir, ignore_errors=True)
        
        print(f"[Worker {worker_id}] Shutdown complete")


# =============================================================================
# Main Runner
# =============================================================================

def run_qwen_baseline(
    qwen_gpus: List[int],
    qwen_pair_size: int,
    num_layers: int = DEFAULT_NUM_LAYERS,
    seed: int = DEFAULT_SEED,
    resolution: int = DEFAULT_RESOLUTION,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    true_cfg_scale: float = DEFAULT_TRUE_CFG_SCALE,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip_completed: bool = True,
    src_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Qwen Image Layered baseline 실험 실행
    """
    if src_root is None:
        src_root = get_src_root()
    
    paths = get_all_split_paths(src_root)
    
    for vf_dir in paths["valid_frames_dirs"]:
        if not vf_dir.exists():
            print(f"Warning: {vf_dir} not found")
    
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    
    # 로깅 설정
    log_file = paths["output_dir"] / "qwen_baseline_all_splits.log"
    logger = setup_logging(log_file)
    
    # GPU pair 생성
    gpu_pairs = create_gpu_pairs(qwen_gpus, qwen_pair_size)
    if not gpu_pairs:
        raise ValueError(f"Cannot create GPU pairs from {qwen_gpus} with pair_size {qwen_pair_size}")
    
    logger.info("=" * 70)
    logger.info(f"Qwen Image Layered Baseline - Split 0,1,2,3")
    logger.info("=" * 70)
    logger.info(f"valid_frames_dir: {paths['valid_frames_dirs']}")
    logger.info(f"output_dir: {paths['output_dir']}")
    logger.info(f"GPU pairs: {gpu_pairs}")
    logger.info(f"num_layers: {num_layers}, resolution: {resolution}")
    logger.info(f"dry_run: {dry_run}, limit: {limit}")
    
    # 프레임 목록 로드
    frames = load_frame_list(paths)
    logger.info(f"Found {len(frames)} valid frames")
    
    if not frames:
        logger.warning("No valid frames found!")
        return {"error": "No valid frames found"}
    
    # 완료된 프레임 중 input image 없는 경우 보충
    logger.info("Checking for missing input images in completed frames...")


    backfill_stats = backfill_missing_files(frames, paths["output_dir"], logger)
    if backfill_stats["input_image"] > 0 or backfill_stats["reconstruction"] > 0:
        logger.info(f"Backfilled: {backfill_stats['input_image']} inputs, {backfill_stats['reconstruction']} reconstructions")
    else:
        logger.info("No missing files found in completed frames")




    
    # 완료된 프레임 필터링
    if skip_completed:
        pending_frames = [f for f in frames if not is_frame_completed(paths["output_dir"], f.frame_id)]
        completed_count = len(frames) - len(pending_frames)
        logger.info(f"Already completed: {completed_count}, Pending: {len(pending_frames)}")
    else:
        pending_frames = frames
    
    # Limit 적용
    if limit:
        pending_frames = pending_frames[:limit]
        logger.info(f"Limited to {len(pending_frames)} frames")
    
    if dry_run:
        logger.info("\n[DRY RUN] Would process these frames:")
        for i, frame in enumerate(pending_frames[:20]):
            logger.info(f"  [{i+1:3d}] {frame.frame_id}")
        if len(pending_frames) > 20:
            logger.info(f"  ... and {len(pending_frames) - 20} more")
        
        logger.info(f"\nGPU pair assignments:")
        for i, pair in enumerate(gpu_pairs):
            assigned = len([f for j, f in enumerate(pending_frames) if j % len(gpu_pairs) == i])
            logger.info(f"  Worker {i} (GPUs {pair}): {assigned} frames")
        
        return {
            "dry_run": True,
            "total_frames": len(pending_frames),
            "gpu_pairs": [list(p) for p in gpu_pairs],
            "backfilled_input_images": backfilled_count,
        }
    
    # 처리할 프레임이 없으면 종료
    if not pending_frames:
        logger.info("No pending frames to process. All done!")
        return {
            "processed": [],
            "failed": [],
            "backfilled_stats": backfill_stats,
            "message": "All frames already completed",
        }
    
    # Qwen 파라미터
    qwen_params = {
        "num_layers": num_layers,
        "seed": seed,
        "resolution": resolution,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "alpha_threshold": alpha_threshold,
    }
    
    # 멀티프로세싱 설정
    mp.set_start_method("spawn", force=True)
    
    frame_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 프레임을 큐에 추가
    for frame in pending_frames:
        frame_queue.put((frame.frame_id, str(frame.image_path)))
    
    # 종료 신호 추가
    for _ in gpu_pairs:
        frame_queue.put(None)
    
    # 워커 프로세스 시작
    workers = []
    for i, pair in enumerate(gpu_pairs):
        p = mp.Process(
            target=worker_process,
            args=(i, pair, frame_queue, result_queue, paths["output_dir"], qwen_params),
            daemon=True,
        )
        p.start()
        workers.append(p)
        logger.info(f"Started worker {i} on GPUs {pair}")
    
    # 결과 수집
    results = {
        "start_time": datetime.now().isoformat(),
        "gpu_pairs": [list(p) for p in gpu_pairs],
        "qwen_params": qwen_params,
        "backfilled_stats": backfill_stats,
        "processed": [],
        "failed": [],
    }
    
    total = len(pending_frames)
    processed = 0
    
    try:
        while processed < total:
            try:
                result = result_queue.get(timeout=600)  # 10분 타임아웃
            except:
                # 워커 상태 확인
                alive = sum(1 for w in workers if w.is_alive())
                if alive == 0:
                    logger.warning("All workers have stopped")
                    break
                continue
            
            processed += 1
            
            if result["success"]:
                logger.info(f"[{processed}/{total}] ✓ {result['frame_id']} "
                          f"({result['num_layers']} layers, {result['elapsed']:.1f}s)")
                results["processed"].append({
                    "frame_id": result["frame_id"],
                    "num_layers": result["num_layers"],
                    "elapsed": result["elapsed"],
                })
            else:
                logger.error(f"[{processed}/{total}] ✗ {result['frame_id']}: {result['error']}")
                results["failed"].append({
                    "frame_id": result["frame_id"],
                    "error": result["error"],
                })
            
            # 중간 결과 저장
            if processed % 10 == 0:
                results["end_time"] = datetime.now().isoformat()
                results_file = paths["output_dir"] / f"qwen_baseline_results.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
    
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
    
    finally:
        # 워커 종료 대기
        for w in workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()
    
    # 최종 결과 저장
    results["end_time"] = datetime.now().isoformat()
    results_file = paths["output_dir"] / f"qwen_baseline_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 요약 출력
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total processed: {len(results['processed'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    logger.info(f"Backfilled: {backfill_stats['input_image']} inputs, {backfill_stats['reconstruction']} reconstructions")
    
    if results["failed"]:
        logger.info("\nFailed frames:")
        for f in results["failed"][:10]:
            logger.info(f"  - {f['frame_id']}: {f['error'][:80]}")
        if len(results["failed"]) > 10:
            logger.info(f"  ... and {len(results['failed']) - 10} more")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen Image Layered Baseline Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 기본 실행: GPU 3,4,5,6을 2개씩 pair로 사용
    python run_qwen_baseline.py --split_idx 0 --qwen_gpus 3,4,5,6 --qwen_pair_size 2
    
    # RTX 3090: 3개씩 pair 사용
    python run_qwen_baseline.py --split_idx 1 --qwen_gpus 0,1,2,3,4,5 --qwen_pair_size 3
    
    # Dry run (실행 없이 확인)
    python run_qwen_baseline.py --split_idx 0 --qwen_gpus 3,4 --qwen_pair_size 2 --dry_run
    
    # 테스트용 (처음 10개만)
    python run_qwen_baseline.py --split_idx 0 --qwen_gpus 3,4 --qwen_pair_size 2 --limit 10
    
    # 레이어 수 조정
    python run_qwen_baseline.py --split_idx 0 --qwen_gpus 3,4 --qwen_pair_size 2 --num_layers 6
        """
    )
    
    parser.add_argument(
        "--qwen_gpus",
        type=str,
        required=True,
        help="GPU IDs for Qwen model (comma-separated, e.g., '3,4,5,6')"
    )
    parser.add_argument(
        "--qwen_pair_size",
        type=int,
        required=True,
        help="Number of GPUs per Qwen pair (e.g., 2 for A6000, 3 for RTX3090)"
    )
    
    # Qwen 파라미터
    parser.add_argument(
        "--num_layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help=f"Number of layers to generate (default: {DEFAULT_NUM_LAYERS})"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Resolution for Qwen (default: {DEFAULT_RESOLUTION})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    
    # 실행 옵션
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
    parser.add_argument(
        "--src_root",
        type=str,
        default=None,
        help="Source root directory (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # GPU 파싱
    qwen_gpus = parse_gpu_list(args.qwen_gpus)
    if not qwen_gpus:
        print(f"Error: Invalid qwen_gpus: {args.qwen_gpus}")
        sys.exit(1)
    
    src_root = Path(args.src_root) if args.src_root else None
    
    try:
        results = run_qwen_baseline(
            qwen_gpus=qwen_gpus,
            qwen_pair_size=args.qwen_pair_size,
            num_layers=args.num_layers,
            resolution=args.resolution,
            seed=args.seed,
            dry_run=args.dry_run,
            limit=args.limit,
            skip_completed=not args.no_skip,
            src_root=src_root,
        )
        
        if not args.dry_run:
            print(f"\nResults saved to: qwen_experiment/all_splits/qwen_baseline_results.json")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()