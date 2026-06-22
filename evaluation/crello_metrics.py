#!/usr/bin/env python3
"""
evaluation_crello.py

Element-Level Evaluation for Crello Dataset
Adapted from evaluation_figma.py - same metrics, matching, visualization logic.
Only GT extraction, directory paths, and task collection are modified for Crello structure.

Usage:
    python evaluation_crello.py \
        --crello-subset ./crello_subset \
        --qwen-exp ./crello_experiment_qwen_0206 \
        --agent-exp ./crello_experiment_agent_0206 \
        --output ./evaluation_crello_results \
        --max-episodes 10 \
        --matching optimal \
        --num-workers 8
"""

import argparse
import json
import warnings
import sys
import os
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from queue import Empty

from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

# Metrics
from skimage.metrics import (
    structural_similarity as ssim_func,
    mean_squared_error, 
    peak_signal_noise_ratio,
    normalized_root_mse as nrmse_func
)
from skimage.morphology import binary_erosion, square

# ILP Solver
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_MILP_AVAILABLE = True
except ImportError:
    SCIPY_MILP_AVAILABLE = False

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

warnings.filterwarnings("ignore")

# Optional: Deep learning metrics (LPIPS, DINO)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from transformers import ViTImageProcessor, ViTModel
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

ALPHA_THRESHOLD = 16
MIN_ELEMENT_AREA = 100
BACKGROUND_L1_THRESHOLD = 0.5

# Optimal Matching Hyperparameters
MERGE_IOU_THRESHOLD = 0.05
CONTAINMENT_MERGE_THRESHOLD = 0.5  # 0.5 이상 포함되면 무조건 병합 
OPTIONAL_MERGE_THRESHOLD = 0.1     # 0.1~0.75 사이면 조합 생성 후보 
LAMBDA_L1 = 0.7
LAMBDA_IOU = 0.3
PENALTY_GT_MERGE = 0.05
PENALTY_PE_MERGE = 0.0
DUMMY_COST = 0.4
MAX_MERGE_SIZE = None  # 제한 없음


def _json_safe_default(obj):
    """JSON serializer that handles NaN, inf, and numpy types."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return float(obj)


# =============================================================================
# Enhanced Logging System with Worker ID
# =============================================================================

class WorkerLogger:
    """Logger for individual worker with worker ID prefix."""
    
    def __init__(self, worker_id: int, log_queue: Queue, log_file_path: Optional[Path] = None):
        self.worker_id = worker_id
        self.log_queue = log_queue
        self.log_file = None
        if log_file_path:
            self.log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp and worker ID."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}][Worker-{self.worker_id}][{level}] {message}"
        
        # Send to queue for main process to print
        self.log_queue.put(formatted)
        
        # Also write to worker-specific log file
        if self.log_file:
            self.log_file.write(formatted + "\n")
            self.log_file.flush()
    
    def info(self, message: str):
        self.log(message, "INFO")
    
    def debug(self, message: str):
        self.log(message, "DEBUG")
    
    def warn(self, message: str):
        self.log(message, "WARN")
    
    def error(self, message: str):
        self.log(message, "ERROR")
    
    def progress(self, current: int, total: int, episode_id: str, extra: str = ""):
        """Log progress with percentage."""
        pct = (current / total) * 100 if total > 0 else 0
        msg = f"Progress: {current}/{total} ({pct:.1f}%) - Episode: {episode_id}"
        if extra:
            msg += f" | {extra}"
        self.log(msg, "PROG")
    
    def close(self):
        if self.log_file:
            self.log_file.close()


class DualLogger:
    """Logger that writes to both console and file simultaneously (for main process)."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')
        self.console = sys.stdout
        
    def write(self, message: str):
        self.console.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def print(self, *args, **kwargs):
        message = ' '.join(str(arg) for arg in args)
        end = kwargs.get('end', '\n')
        self.write(message + end)
    
    def close(self):
        self.log_file.close()


# Global logger placeholder (will be set per-worker)
worker_logger: Optional[WorkerLogger] = None


def log_print(*args, **kwargs):
    """Print function that uses worker logger if available."""
    message = ' '.join(str(arg) for arg in args)
    if worker_logger is not None:
        worker_logger.info(message)
    else:
        print(message)


# =============================================================================
# Metric Models (LPIPS, DINO)
# =============================================================================

class MetricModels:
    """Wrapper for deep learning based metrics."""
    
    def __init__(self, device: str = "cuda:0", logger: Optional[WorkerLogger] = None):
        self.device = device if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.lpips_model = None
        self.dino_model = None
        self.dino_processor = None
        self.logger = logger
        
        if LPIPS_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()
                if self.logger:
                    self.logger.info(f"LPIPS loaded on {self.device}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load LPIPS: {e}")
        
        if DINO_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.dino_processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
                self.dino_model = ViTModel.from_pretrained('facebook/dino-vits16').to(self.device).eval()
                if self.logger:
                    self.logger.info(f"DINO loaded on {self.device}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load DINO: {e}")
    
    @torch.no_grad()
    def compute_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        if self.lpips_model is None:
            return 0.0
        try:
            t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            t1 = t1 * 2.0 - 1.0
            t2 = t2 * 2.0 - 1.0
            return float(self.lpips_model(t1, t2).item())
        except Exception:
            return 0.0
    
    @torch.no_grad()
    def compute_dino(self, img1: np.ndarray, img2: np.ndarray) -> float:
        if self.dino_model is None or self.dino_processor is None:
            return 0.0
        try:
            p1 = Image.fromarray((img1 * 255).astype(np.uint8))
            p2 = Image.fromarray((img2 * 255).astype(np.uint8))
            inputs1 = self.dino_processor(images=p1, return_tensors="pt").to(self.device)
            inputs2 = self.dino_processor(images=p2, return_tensors="pt").to(self.device)
            emb1 = self.dino_model(**inputs1).last_hidden_state[:, 0, :]
            emb2 = self.dino_model(**inputs2).last_hidden_state[:, 0, :]
            return float(F.cosine_similarity(emb1, emb2).item())
        except Exception:
            return 0.0


# =============================================================================
# Element Data Structure
# =============================================================================

def create_element(
    elem_id: str,
    elem_type: str,
    mask: np.ndarray,
    image: Image.Image,
    bbox: List[int],
    z_index: int,
    source: str = "unknown"
) -> Dict[str, Any]:
    return {
        "id": elem_id,
        "type": elem_type,
        "mask": mask.astype(np.float32),
        "image": image,
        "bbox": bbox,
        "z_index": z_index,
        "area": float(np.sum(mask > 0)),
        "source": source,
    }


# =============================================================================
# Alpha Noise Cleaning
# =============================================================================

def clean_alpha_noise(img: Image.Image, threshold: int = ALPHA_THRESHOLD) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    mask = arr[:, :, 3] < threshold
    arr[mask] = [0, 0, 0, 0]
    return Image.fromarray(arr, "RGBA")


def clean_element_alpha(elem: Dict, canvas_size: Tuple[int, int]) -> Dict:
    W, H = canvas_size
    
    cleaned_img = clean_alpha_noise(elem["image"])
    
    if cleaned_img.size != (W, H):
        cleaned_img = cleaned_img.resize((W, H), Image.LANCZOS)
    
    alpha = np.array(cleaned_img.getchannel("A"))
    cleaned_mask = (alpha > 0).astype(np.float32)
    
    cleaned_elem = elem.copy()
    cleaned_elem["image"] = cleaned_img
    cleaned_elem["mask"] = cleaned_mask
    cleaned_elem["area"] = float(cleaned_mask.sum())
    
    return cleaned_elem


# =============================================================================
# Element Extraction
# =============================================================================

def extract_gt_elements(
    record_dir: Path,
    logger: Optional[WorkerLogger] = None
) -> Tuple[List[Dict], Tuple[int, int], Optional[Image.Image]]:
    """Extract GT elements from Crello data.
    
    Crello elements are already placed on full canvas (RGBA),
    so no coordinate transforms are needed - just load and extract mask.
    """
    gt_meta_path = record_dir / "gt_metadata.json"
    with open(gt_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    canvas_w = int(meta["canvas_width"])
    canvas_h = int(meta["canvas_height"])
    canvas_size = (canvas_w, canvas_h)
    
    elements_dir = record_dir / meta.get("unit_images_dir", "elements")
    
    # Load composite (GT reconstruction) image
    gt_recon_img = None
    recon_rel = meta.get("reconstructed_image_path", "composite.png")
    recon_path = record_dir / recon_rel
    if recon_path.exists():
        try:
            gt_recon_img = Image.open(recon_path).convert("RGBA")
            if gt_recon_img.size != canvas_size:
                gt_recon_img = gt_recon_img.resize(canvas_size, Image.LANCZOS)
        except Exception as e:
            if logger:
                logger.warn(f"Failed to load composite image: {e}")
    
    elements = []
    units = meta.get("unit_images", [])
    units_sorted = sorted(units, key=lambda u: u.get("z_index", 0))
    
    for idx, unit in enumerate(units_sorted):
        # Skip invalid elements
        if not unit.get("valid", True):
            continue
        
        img_rel = unit.get("image_path")
        if not img_rel:
            continue
        
        img_path = elements_dir / img_rel
        if not img_path.exists():
            continue
        
        try:
            # Element images are already on full canvas (RGBA)
            elem_img = Image.open(img_path).convert("RGBA")
            elem_img = clean_alpha_noise(elem_img)
            
            if elem_img.size != canvas_size:
                elem_img = elem_img.resize(canvas_size, Image.LANCZOS)
            
            mask = np.array(elem_img.getchannel("A")).astype(np.float32) / 255.0
            
            if mask.sum() > MIN_ELEMENT_AREA:
                bbox = unit.get("bbox", [0, 0, canvas_w, canvas_h])
                
                elements.append(create_element(
                    elem_id=f"gt_{unit.get('unit_id', f'elem_{idx}')}",
                    elem_type=unit.get("unit_type", "object"),
                    mask=mask,
                    image=elem_img,
                    bbox=[int(b) for b in bbox],
                    z_index=unit.get("z_index", idx),
                    source="crello_gt"
                ))
        except Exception as e:
            if logger:
                logger.warn(f"Failed to load {img_path}: {e}")
    
    return elements, canvas_size, gt_recon_img


def apply_soft_kmeans_refinement(elem: Dict) -> Dict:
    """
    K-Means로 색상을 분리하되, 'Soft-Thresholding'을 적용하여 
    텍스트의 획(Stroke)이 얇아지는 것을 방지하고 IoU를 보존합니다.
    """
    # 원본 데이터 추출
    img_pil = elem["image"].convert("RGBA")
    img_arr = np.array(img_pil).astype(np.float32)
    
    alpha = img_arr[..., 3]
    rgb = img_arr[..., :3]
    
    # 유효 픽셀(알파가 10 이상)만 대상으로 군집화 수행
    valid_mask = alpha > 16 
    if np.sum(valid_mask) < 32: 
        return elem

    # 1. K-Means로 전경/배경 대표 색상 추출
    pixels = rgb[valid_mask]
    try:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=3).fit(pixels)
        centers = kmeans.cluster_centers_ # [2, 3] 형태
    except Exception:
        # sklearn이 없거나 에러 발생 시 원본 반환
        return elem

    # 2. 전경(글자) 중심점 찾기 (알파값이 높고 진한 곳의 라벨을 전경으로 가정)
    full_labels = np.full(alpha.shape, -1, dtype=int)
    full_labels[valid_mask] = kmeans.labels_
    
    # 더 확실한 전경 영역(알파 > 220)에서 가장 많이 등장한 라벨을 전경으로 선택
    core_mask = alpha > 200
    if np.sum(core_mask) < 10: 
        core_mask = valid_mask
    
    # core_mask 영역의 라벨들 중 빈도수가 높은 것을 foreground로 설정
    valid_labels = full_labels[core_mask]
    valid_labels = valid_labels[valid_labels != -1]
    
    if len(valid_labels) == 0:
        return elem
        
    fg_label = np.argmax(np.bincount(valid_labels))
    bg_label = 1 - fg_label
    
    C_fg = centers[fg_label]
    C_bg = centers[bg_label]

    # 3. 모든 픽셀에 대해 전경색/배경색과의 거리 계산
    dist_fg = np.linalg.norm(rgb - C_fg, axis=2)
    dist_bg = np.linalg.norm(rgb - C_bg, axis=2)

    # 4. Soft-Weighting 로직
    # ratio가 0에 가까울수록 전경, 1에 가까울수록 배경
    ratio = dist_fg / (dist_fg + dist_bg + 1e-6)
    
    # protection_threshold: 이 값보다 전경에 가까우면(ratio가 작으면) 알파값을 건드리지 않음
    # 값을 높일수록(예: 0.6~0.7) 텍스트가 더 두껍게 유지됨
    protection_threshold = 0.6
    
    new_alpha = alpha.copy()
    
    # 배경에 훨씬 가까운 애들만 골라냅니다 (ratio > threshold)
    kill_indices = (ratio > protection_threshold) & (valid_mask)
    
    # 5. 급격하게 지우지 않고, ratio에 따라 부드럽게 감쇄 (Linear Decay)
    if np.any(kill_indices):
        decay = (1.0 - ratio[kill_indices]) / (1.0 - protection_threshold)
        # 제곱을 해서 경계면을 더 부드럽게 처리
        new_alpha[kill_indices] = new_alpha[kill_indices] * (decay ** 2)

    # 6. 결과 반영
    new_img_arr = np.array(img_pil)
    new_img_arr[..., 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)
    
    new_elem = elem.copy()
    new_elem["image"] = Image.fromarray(new_img_arr)
    # 0~1 range float32 mask 업데이트
    new_elem["mask"] = (new_alpha / 255.0).astype(np.float32)
    # area 재계산
    new_elem["area"] = float(np.sum(new_elem["mask"] > 0))
    
    return new_elem


def extract_agent_elements(
    episode_dir: Path,
    canvas_size: Tuple[int, int],
    apply_alpha_correction: bool = True,
    text_refinement: bool = True,
    logger: Optional[WorkerLogger] = None
) -> List[Dict]:
    """Extract Agent parsed elements with bbox clipping for noise reduction."""
    parse_path = episode_dir / "parse.json"
    history_tree_path = episode_dir / "history_tree.json"
    
    if not parse_path.exists() or not history_tree_path.exists():
        return []
    
    with open(parse_path, "r", encoding="utf-8") as f:
        parse_data = json.load(f)
    
    with open(history_tree_path, "r", encoding="utf-8") as f:
        history_tree = json.load(f)
    
    parsed_elements = parse_data.get("elements", [])
    src_root = episode_dir.parent.parent.parent
    
    z_order = compute_z_order(history_tree)
    layer_to_z = {layer_id: idx for idx, layer_id in enumerate(z_order)}
    
    W, H = canvas_size
    elements = []
    
    original_alpha = None
    if apply_alpha_correction:
        original_alpha = load_original_alpha_mask(episode_dir)
        if original_alpha is not None:
            if original_alpha.shape != (H, W):
                original_alpha = cv2.resize(
                    original_alpha.astype(np.float32), 
                    (W, H), 
                    interpolation=cv2.INTER_LINEAR
                ).astype(np.uint8)
    
    for idx, elem in enumerate(parsed_elements):
        source_layer = elem.get("source_layer_id", "")
        z_idx = layer_to_z.get(source_layer, idx)
        
        elem_type = elem.get("type", "object")
        bbox = elem.get("bbox", [0, 0, 100, 100])
        x1_b, y1_b, x2_b, y2_b = [int(round(b)) for b in bbox] # 정수화 및 반올림
        
        img_path_rel = elem.get("canvas_image_uri") or elem.get("extracted_image_uri")
        if not img_path_rel:
            continue
        
        # [경로 처리] Crello/Figma 환경에 따라 절대경로/상대경로 분기 처리 로직 유지
        img_path = Path(img_path_rel)
        if not img_path.is_absolute():
            img_path = src_root / img_path_rel
            
        if not img_path.exists():
            continue
        
        try:
            elem_img = Image.open(img_path).convert("RGBA")
            elem_img = clean_alpha_noise(elem_img)
            
            # 빈 캔버스 생성 (완전 투명)
            canvas_arr = np.zeros((H, W, 4), dtype=np.uint8)
            
            if elem.get("canvas_image_uri") and elem_img.size == (W, H):
                # =============================================================
                # [핵심 추가] BBox Clipping 로직
                # 전체 크기의 이미지이지만, bbox 영역만 캔버스에 복사하고 나머지는 버림
                # =============================================================
                elem_arr = np.array(elem_img)
                
                # 인덱스 슬라이싱 안전 범위 계산
                y1_clip, y2_clip = max(0, y1_b), min(H, y2_b)
                x1_clip, x2_clip = max(0, x1_b), min(W, x2_b)
                
                # 타겟 영역만 복사 (나머지는 초기값 0,0,0,0 유지)
                if y2_clip > y1_clip and x2_clip > x1_clip:
                    canvas_arr[y1_clip:y2_clip, x1_clip:x2_clip] = \
                        elem_arr[y1_clip:y2_clip, x1_clip:x2_clip]
                
                canvas = Image.fromarray(canvas_arr, "RGBA")
            else:
                # extracted_image(이미 Crop된 조각)일 경우 기존의 numpy 직접 배치 로직 유지
                x1, y1 = x1_b, y1_b
                if x1 < W and y1 < H:
                    elem_arr = np.array(elem_img)
                    eh, ew = elem_arr.shape[:2]
                    copy_h = min(eh, H - y1)
                    copy_w = min(ew, W - x1)
                    if copy_h > 0 and copy_w > 0:
                        canvas_arr[y1:y1+copy_h, x1:x1+copy_w] = elem_arr[:copy_h, :copy_w]
                
                canvas = Image.fromarray(canvas_arr, "RGBA")
            
            # Alpha correction (원본 알파와 비교하여 더 정교하게 마스킹)
            if original_alpha is not None:
                # 이미 캔버스에 배치되었으므로 intersection 모드 사용
                canvas = apply_original_alpha_to_element(
                    canvas, [0, 0, W, H], original_alpha, canvas_size,
                    mode="zero_mask"
                )
            
            mask = np.array(canvas.getchannel("A")).astype(np.float32) / 255.0
            
            if mask.sum() > MIN_ELEMENT_AREA:
                element_dict = create_element(
                    elem_id=f"agent_{elem.get('id', idx)}",
                    elem_type=elem_type,
                    mask=mask,
                    image=canvas,
                    bbox=[int(b) for b in bbox],
                    z_index=z_idx,
                    source="agent"
                )
                
                if text_refinement and elem_type == "text":
                    element_dict = apply_soft_kmeans_refinement(element_dict)

                elements.append(element_dict)

        except Exception as e:
            if logger:
                logger.warn(f"Failed to process {img_path}: {e}")
    
    return elements

def compute_z_order(history_tree: Dict, root_id: str = "layer_0000") -> List[str]:
    z_order = []
    visited = set()
    
    def dfs(layer_id):
        if layer_id in visited:
            return
        visited.add(layer_id)
        
        node = history_tree.get(layer_id)
        if not node:
            return
        
        action_type = node.get("action_type")
        children = node.get("children_ids") or []
        real_children = [c for c in children if not c.startswith("_temp_")]
        
        if action_type in ["Finalize_Text", "Finalize_Obj"]:
            z_order.append(layer_id)
            return
        
        if action_type == "Discard":
            return
        
        for child_id in real_children:
            dfs(child_id)
    
    dfs(root_id)
    return z_order


def extract_qwen_elements_cca(
    episode_dir: Path,
    canvas_size: Tuple[int, int],
    logger: Optional[WorkerLogger] = None
) -> List[Dict]:
    """Extract elements from Qwen layers using Connected Component Analysis."""
    elements = []
    
    for layer_idx in range(10):
        layer_path = episode_dir / f"layer_{layer_idx:02d}.png"
        if not layer_path.exists():
            if layer_idx >= 4:
                break
            continue
        
        try:
            layer_img = Image.open(layer_path).convert("RGBA")
            if layer_img.size != canvas_size:
                layer_img = layer_img.resize(canvas_size, Image.LANCZOS)
            layer_img = clean_alpha_noise(layer_img)
            
            alpha = np.array(layer_img.getchannel("A"))
            binary = (alpha > ALPHA_THRESHOLD).astype(np.uint8)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            
            for comp_idx in range(1, num_labels):
                x, y, w, h, area = stats[comp_idx]
                
                if area < MIN_ELEMENT_AREA:
                    continue
                
                comp_mask = (labels == comp_idx).astype(np.float32)
                
                element_arr = np.array(layer_img)
                element_arr[labels != comp_idx] = [0, 0, 0, 0]
                element_img = Image.fromarray(element_arr, "RGBA")
                
                elements.append(create_element(
                    elem_id=f"qwen_L{layer_idx}_C{comp_idx}",
                    elem_type="object",
                    mask=comp_mask,
                    image=element_img,
                    bbox=[int(x), int(y), int(x+w), int(y+h)],
                    z_index=layer_idx * 10000 + comp_idx,
                    source="qwen"
                ))
        except Exception as e:
            if logger:
                logger.warn(f"Failed to process {layer_path}: {e}")
    
    return elements


def extract_omnisvg_elements(
    episode_dir: Path,
    canvas_size: Tuple[int, int],
    logger: Optional[WorkerLogger] = None,
    render_scale: float = 1.0,
) -> List[Dict]:
    """Extract elements from OmniSVG SVG output by isolating each <path> element.

    Parses the SVG, renders each path individually to RGBA at the target canvas size,
    and returns standard element dicts compatible with the matching pipeline.
    """
    import io
    import xml.etree.ElementTree as ET

    try:
        import cairosvg
    except ImportError:
        if logger:
            logger.warn("cairosvg not installed — cannot extract OmniSVG elements")
        return []

    svg_path = episode_dir / "output.svg"
    if not svg_path.exists():
        if logger:
            logger.warn(f"output.svg not found in {episode_dir}")
        return []

    W, H = canvas_size
    rW = max(1, int(W * render_scale))
    rH = max(1, int(H * render_scale))

    try:
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
    except Exception as e:
        if logger:
            logger.warn(f"Failed to read {svg_path}: {e}")
        return []

    try:
        root = ET.fromstring(svg_content)
    except ET.ParseError as e:
        if logger:
            logger.warn(f"Failed to parse SVG {svg_path}: {e}")
        return []

    # Extract viewBox.
    # OmniSVG uses "0 0 200 200"; VTracer has no viewBox (coordinates are
    # already in pixel space matching width/height).  Fall back to the SVG's
    # own dimensions so that pixel-space paths render correctly.
    svg_w = root.get('width', str(W))
    svg_h = root.get('height', str(H))
    viewbox = root.get('viewBox', f'0 0 {svg_w} {svg_h}')

    # Extract all path elements (handle SVG namespace)
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    paths = root.findall('.//svg:path', ns)
    if not paths:
        paths = root.findall('.//{http://www.w3.org/2000/svg}path')
    if not paths:
        paths = root.findall('.//path')
    if not paths:
        if logger:
            logger.warn(f"No <path> elements found in {svg_path}")
        return []

    # Build render tasks
    render_tasks = []
    for idx, path_elem in enumerate(paths):
        path_str = ET.tostring(path_elem, encoding='unicode')
        single_svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{viewbox}" width="{W}" height="{H}">'
            f'{path_str}</svg>'
        )
        render_tasks.append((idx, single_svg))

    # Parallel cairosvg rendering (cairosvg releases GIL)
    from concurrent.futures import ThreadPoolExecutor

    def _render_one(args_tuple):
        idx, svg_str = args_tuple
        try:
            png_data = cairosvg.svg2png(
                bytestring=svg_str.encode('utf-8'),
                output_width=rW,
                output_height=rH,
            )
            return (idx, png_data)
        except Exception:
            return (idx, None)

    n_threads = min(8, len(render_tasks))
    rendered = []
    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            rendered = list(pool.map(_render_one, render_tasks))
    else:
        rendered = [_render_one(t) for t in render_tasks]

    elements = []
    for idx, png_data in rendered:
        if png_data is None:
            continue

        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        if render_scale != 1.0:
            img = img.resize((W, H), Image.NEAREST)
        alpha = np.array(img.getchannel("A"))
        mask = alpha.astype(np.float32) / 255.0

        area = float(mask.sum())
        if area < MIN_ELEMENT_AREA:
            continue
        if area > 0.9 * W * H:
            continue

        ys, xs = np.where(alpha > 0)
        if len(ys) == 0:
            continue
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

        elements.append(create_element(
            elem_id=f"omnisvg_P{idx:04d}",
            elem_type="object",
            mask=mask,
            image=img,
            bbox=bbox,
            z_index=idx,
            source="omnisvg"
        ))

    if logger:
        logger.info(f"Extracted {len(elements)} OmniSVG elements from {len(paths)} paths")

    return elements


def load_original_alpha_mask(episode_dir: Path) -> Optional[np.ndarray]:
    original_path = episode_dir / "original_input.png"

    if not original_path.exists():
        layer_0000_path = episode_dir / "layers" / "layer_0000" / "layer_image.png"
        if layer_0000_path.exists():
            original_path = layer_0000_path
        else:
            return None

    try:
        img = Image.open(original_path).convert("RGBA")
        alpha = np.array(img.getchannel("A"))
        return alpha
    except Exception:
        return None


def apply_original_alpha_to_element(
    elem_img: Image.Image,
    bbox: List[int],
    original_alpha: np.ndarray,
    canvas_size: Tuple[int, int],
    mode: str = "zero_mask"
) -> Image.Image:
    W, H = canvas_size
    x1, y1, x2, y2 = [int(b) for b in bbox]
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    
    if x2 <= x1 or y2 <= y1:
        return elem_img
    
    elem_arr = np.array(elem_img.convert("RGBA"))
    elem_h, elem_w = elem_arr.shape[:2]
    
    try:
        orig_region = original_alpha[y1:y2, x1:x2]
        
        if orig_region.shape != (elem_h, elem_w):
            orig_region = cv2.resize(
                orig_region.astype(np.float32), 
                (elem_w, elem_h), 
                interpolation=cv2.INTER_LINEAR
            ).astype(np.uint8)
        
        elem_alpha = elem_arr[:, :, 3].astype(np.float32)
        orig_alpha_float = orig_region.astype(np.float32)
        
        if mode == "zero_mask":
            # 원본에서 alpha가 거의 없었던 곳(<=16)을 강제로 투명하게, 나머지는 그대로
            orig_transparent = orig_alpha_float <= ALPHA_THRESHOLD
            new_alpha = elem_alpha.copy()
            new_alpha[orig_transparent] = 0
            # RGBA 전체를 0으로 (RGB 오염 방지)
            elem_arr[orig_transparent] = [0, 0, 0, 0]
        elif mode == "replace":
            mask = elem_alpha > ALPHA_THRESHOLD
            new_alpha = np.where(mask, orig_alpha_float, 0)
        elif mode == "intersection":
            new_alpha = np.minimum(elem_alpha, orig_alpha_float)
        elif mode == "multiply":
            new_alpha = (elem_alpha / 255.0) * (orig_alpha_float / 255.0) * 255.0
        else:
            return elem_img
        
        elem_arr[:, :, 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)
        
        return Image.fromarray(elem_arr, "RGBA")
        
    except Exception:
        return elem_img


# =============================================================================
# Visible Mask Computation
# =============================================================================

def compute_visible_masks(
    elements: List[Dict],
    canvas_size: Tuple[int, int],
    apply_alpha_cleaning: bool = False
) -> List[Dict]:
    W, H = canvas_size
    
    if apply_alpha_cleaning:
        elements = [clean_element_alpha(elem, canvas_size) for elem in elements]
    
    sorted_elems = sorted(elements, key=lambda x: x.get("z_index", 0), reverse=True)
    accumulated_occlusion = np.zeros((H, W), dtype=bool)
    visible_elements = []
    
    for elem in sorted_elems:
        mask = elem["mask"]
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        
        mask_bin = mask > 0
        visible_mask = mask_bin & ~accumulated_occlusion
        visible_area = float(visible_mask.sum())
        
        elem_with_visible = elem.copy()
        elem_with_visible["visible_mask"] = visible_mask
        elem_with_visible["visible_area"] = visible_area
        elem_with_visible["mask_bin"] = mask_bin
        
        visible_elements.append(elem_with_visible)
        accumulated_occlusion = accumulated_occlusion | mask_bin
    
    visible_elements = sorted(visible_elements, key=lambda x: x.get("z_index", 0))
    return visible_elements


# =============================================================================
# Visible Area Compositing
# =============================================================================

def composite_visible_elements(
    elements: List[Dict],
    canvas_size: Tuple[int, int]
) -> Tuple[Image.Image, np.ndarray]:
    W, H = canvas_size
    
    canvas_arr = np.zeros((H, W, 4), dtype=np.uint8)
    union_mask = np.zeros((H, W), dtype=np.float32)
    
    sorted_elems = sorted(elements, key=lambda x: x.get("z_index", 0))
    
    for elem in sorted_elems:
        elem_img = elem["image"].convert("RGBA")
        if elem_img.size != (W, H):
            elem_img = elem_img.resize((W, H), Image.LANCZOS)
        
        elem_arr = np.array(elem_img)
        
        visible_mask = elem.get("visible_mask")
        if visible_mask is None:
            visible_mask = elem["mask"] > 0
        
        if visible_mask.shape != (H, W):
            visible_mask = cv2.resize(visible_mask.astype(np.float32), (W, H), 
                                      interpolation=cv2.INTER_LINEAR) > 0
        
        canvas_arr[visible_mask] = elem_arr[visible_mask]
        union_mask = np.maximum(union_mask, visible_mask.astype(np.float32))
    
    composite_rgba = Image.fromarray(canvas_arr, "RGBA")
    return composite_rgba, union_mask


# =============================================================================
# Combinatorial Group Generation
# =============================================================================

def compute_pairwise_visible_iou(elem1: Dict, elem2: Dict) -> float:
    mask1 = elem1.get("visible_mask", elem1["mask"] > 0)
    mask2 = elem2.get("visible_mask", elem2["mask"] > 0)
    
    if mask1.shape != mask2.shape:
        H, W = max(mask1.shape[0], mask2.shape[0]), max(mask1.shape[1], mask2.shape[1])
        if mask1.shape != (H, W):
            mask1 = cv2.resize(mask1.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0
        if mask2.shape != (H, W):
            mask2 = cv2.resize(mask2.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0
    
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    
    return float(intersection / (union + 1e-6))


def generate_combinatorial_groups(
    gt_elements: List[Dict],
    pred_elements: List[Dict],
    logger: Optional[WorkerLogger] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    n_gt = len(gt_elements)
    n_pred = len(pred_elements)

    # 1. Coverage Matrix 계산
    # gt_side_coverage[i, j]: GT[i]가 Pred[j]를 얼마나 포함하는가 (GT에 의한 Pred 커버리지)
    gt_side_coverage = np.zeros((n_gt, n_pred))
    # pred_side_coverage[j, i]: Pred[j]가 GT[i]를 얼마나 포함하는가 (Pred에 의한 GT 커버리지)
    pred_side_coverage = np.zeros((n_pred, n_gt))

    # Auto downscale masks for coverage computation when element count is high
    # Coverage ratios (intersection/area) are scale-invariant, so this doesn't affect grouping decisions
    total_elements = n_gt + n_pred
    if total_elements > 100:
        cov_scale = 0.25
    elif total_elements > 50:
        cov_scale = 0.5
    else:
        cov_scale = 1.0

    # Pre-compute bboxes for fast overlap check
    def _mask_bbox(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return (0, 0, 0, 0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        return (r0, c0, r1 + 1, c1 + 1)

    def _get_cov_mask(elem):
        m = elem.get("visible_mask", elem["mask"] > 0)
        if cov_scale < 1.0:
            h, w = m.shape[:2]
            new_w = max(1, int(w * cov_scale))
            new_h = max(1, int(h * cov_scale))
            m = cv2.resize(m.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
        return m

    gt_masks = [_get_cov_mask(gt_elements[i]) for i in range(n_gt)]
    pred_masks = [_get_cov_mask(pred_elements[j]) for j in range(n_pred)]

    if cov_scale < 1.0 and logger:
        h0, w0 = gt_elements[0]["mask"].shape[:2] if n_gt > 0 else (0, 0)
        logger.info(f"[coverage] Auto downscale: {w0}x{h0} -> {gt_masks[0].shape[1]}x{gt_masks[0].shape[0]} "
                     f"(scale={cov_scale}, elements={total_elements})")

    gt_bboxes = [_mask_bbox(m) for m in gt_masks]
    pred_bboxes = [_mask_bbox(m) for m in pred_masks]
    gt_areas = [float(m.sum()) + 1e-6 for m in gt_masks]
    pred_areas = [float(m.sum()) + 1e-6 for m in pred_masks]

    for i in range(n_gt):
        br0, bc0, br1, bc1 = gt_bboxes[i]
        for j in range(n_pred):
            pr0, pc0, pr1, pc1 = pred_bboxes[j]
            # Skip if bboxes don't overlap
            if br1 <= pr0 or pr1 <= br0 or bc1 <= pc0 or pc1 <= bc0:
                continue
            # Only compute intersection within overlapping ROI
            r0 = max(br0, pr0)
            c0 = max(bc0, pc0)
            r1 = min(br1, pr1)
            c1 = min(bc1, pc1)
            inter = (gt_masks[i][r0:r1, c0:c1] & pred_masks[j][r0:r1, c0:c1]).sum()
            if inter == 0:
                continue
            gt_side_coverage[i, j] = inter / pred_areas[j]
            pred_side_coverage[j, i] = inter / gt_areas[i]

    # 그룹 초기화 (기본적으로 자기 자신은 단독 그룹으로 포함)
    gt_groups_set = set()
    for i in range(n_gt):
        gt_groups_set.add(tuple([i]))
        
    pred_groups_set = set()
    for j in range(n_pred):
        pred_groups_set.add(tuple([j]))

    # ---------------------------------------------------------
    # 로직 수정: 교차 커버리지를 이용하여 "상대방" 그룹을 생성해야 함
    # ---------------------------------------------------------

    # 1. Pred 병합 그룹 생성 (Over-segmentation 해결)
    # 하나의 GT(i)가 여러 Pred(j1, j2...)를 포함하고 있다면, 이 Pred들은 하나의 그룹이 되어야 함
    for i in range(n_gt):
        # GT i에 의해 확실히 포함되는 Pred들의 인덱스
        mandatory = np.where(gt_side_coverage[i] >= CONTAINMENT_MERGE_THRESHOLD)[0].tolist()
        
        # 부분적으로 포함되는 Pred들 (조합 후보)
        optional = np.where((gt_side_coverage[i] >= OPTIONAL_MERGE_THRESHOLD) & 
                            (gt_side_coverage[i] < CONTAINMENT_MERGE_THRESHOLD))[0].tolist()
        
        # Mandatory가 있거나 Optional이 있으면 조합 생성
        base_group = mandatory
        
        # 조합 생성 (속도를 위해 Optional 개수가 너무 많으면 제한하거나, 전체 사용)
        # 여기서는 Pred Group을 만듭니다. (GT 인덱스 i는 포함하지 않음!)
        if not optional:
            if len(base_group) > 1:
                pred_groups_set.add(tuple(sorted(base_group)))
        else:
            # Optional 요소들에 대한 PowerSet 조합
            limit_optional = optional[:5] 
            for r in range(len(limit_optional) + 1):
                for combo in combinations(limit_optional, r):
                    new_group = sorted(list(set(base_group + list(combo))))
                    if len(new_group) > 0:
                        pred_groups_set.add(tuple(new_group))

    # 2. GT 병합 그룹 생성 (Under-segmentation 해결)
    # 하나의 Pred(j)가 여러 GT(i1, i2...)를 포함하고 있다면, 이 GT들은 하나의 그룹이 되어야 함
    for j in range(n_pred):
        # Pred j에 의해 확실히 포함되는 GT들의 인덱스
        mandatory = np.where(pred_side_coverage[j] >= CONTAINMENT_MERGE_THRESHOLD)[0].tolist()
        
        optional = np.where((pred_side_coverage[j] >= OPTIONAL_MERGE_THRESHOLD) & 
                            (pred_side_coverage[j] < CONTAINMENT_MERGE_THRESHOLD))[0].tolist()
        
        base_group = mandatory
        
        # 여기서는 GT Group을 만듭니다. (Pred 인덱스 j는 포함하지 않음!)
        if not optional:
            if len(base_group) > 1:
                gt_groups_set.add(tuple(sorted(base_group)))
        else:
            limit_optional = optional[:5]
            for r in range(len(limit_optional) + 1):
                for combo in combinations(limit_optional, r):
                    new_group = sorted(list(set(base_group + list(combo))))
                    if len(new_group) > 0:
                        gt_groups_set.add(tuple(new_group))

    # set -> list 변환
    gt_groups = [list(g) for g in gt_groups_set]
    pred_groups = [list(p) for p in pred_groups_set]

    if logger:
        logger.info(f"[Groups] Containment-based logic: GT groups={len(gt_groups)}, Pred groups={len(pred_groups)}")
    
    return gt_groups, pred_groups

# =============================================================================
# Cost Computation
# =============================================================================

def to_background(img: Image.Image, bg_color: Tuple[int, int, int]) -> np.ndarray:
    bg = Image.new("RGBA", img.size, bg_color + (255,))
    bg.paste(img, (0, 0), img)
    return np.array(bg).astype(np.float32)[..., :3] / 255.0


def merge_elements_visible(
    elements: List[Dict],
    indices: List[int],
    canvas_size: Tuple[int, int]
) -> Tuple[np.ndarray, Image.Image]:
    selected = [elements[i] for i in indices]
    composite_rgba, union_mask = composite_visible_elements(selected, canvas_size)
    return union_mask, composite_rgba


def compute_matching_cost(
    gt_elements: List[Dict],
    pred_elements: List[Dict],
    gt_indices: List[int],
    pred_indices: List[int],
    canvas_size: Tuple[int, int],
) -> Tuple[float, float, float]:
    W, H = canvas_size
    
    gt_union_mask, gt_composite = merge_elements_visible(gt_elements, gt_indices, canvas_size)
    pred_union_mask, pred_composite = merge_elements_visible(pred_elements, pred_indices, canvas_size)
    
    gt_bin = gt_union_mask > 0
    pred_bin = pred_union_mask > 0
    roi = gt_bin | pred_bin
    
    intersection = (gt_bin & pred_bin).sum()
    union = roi.sum()
    iou = float(intersection / (union + 1e-6))
    
    # Raw RGB on alpha region (LayerD 방식과 동일)
    gt_arr = np.array(gt_composite.convert("RGBA")).astype(np.float32) / 255.0
    pred_arr = np.array(pred_composite.convert("RGBA")).astype(np.float32) / 255.0
    
    gt_rgb_black = to_background(gt_composite, (0, 0, 0))
    pred_rgb_black = to_background(pred_composite, (0, 0, 0))
    gt_rgb_white = to_background(gt_composite, (255, 255, 255))
    pred_rgb_white = to_background(pred_composite, (255, 255, 255))

    l1_black = float(np.mean(np.abs(gt_rgb_black[roi] - pred_rgb_black[roi])))
    l1_white = float(np.mean(np.abs(gt_rgb_white[roi] - pred_rgb_white[roi])))
    l1 = min(l1_black, l1_white)
    
    penalty = ((len(gt_indices) - 1) * PENALTY_GT_MERGE + 
               (len(pred_indices) - 1) * PENALTY_PE_MERGE)
    
    cost = LAMBDA_L1 * l1 + LAMBDA_IOU * (1 - iou) + penalty
    
    return cost, l1, iou, int(gt_bin.sum()), int(pred_bin.sum()), int(intersection)


def build_cost_matrix_gpu(
    gt_elements: List[Dict],
    pred_elements: List[Dict],
    gt_groups: List[List[int]],
    pred_groups: List[List[int]],
    canvas_size: Tuple[int, int],
    device: str = "cuda:0",
    logger: Optional[WorkerLogger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    W, H = canvas_size
    n_gt_grp = len(gt_groups)
    n_pred_grp = len(pred_groups)

    # Auto downscale for large element/group counts (speed optimization)
    total_elements = len(gt_elements) + len(pred_elements)
    total_groups = n_gt_grp + n_pred_grp
    if total_elements > 100 or total_groups > 200:
        cost_scale = 0.5
    else:
        cost_scale = 1.0

    cW = max(1, int(W * cost_scale))
    cH = max(1, int(H * cost_scale))
    if cost_scale < 1.0 and logger:
        logger.info(f"[cost_matrix] Auto downscale: {W}x{H} -> {cW}x{cH} "
                     f"(elements={total_elements}, groups={total_groups})")

    # -------------------------------------------------------------------------
    # 1. Mask Tensor 준비 (IoU 계산용)
    # -------------------------------------------------------------------------
    def get_mask_tensor(elems):
        tensors = []
        for e in elems:
            mask = e["mask"]
            if mask.shape != (cH, cW):
                mask = cv2.resize(mask.astype(np.float32), (cW, cH), interpolation=cv2.INTER_LINEAR)
            tensors.append(torch.from_numpy(mask).to(device))
        return torch.stack(tensors)

    gt_masks_gpu = get_mask_tensor(gt_elements)
    pred_masks_gpu = get_mask_tensor(pred_elements)

    # -------------------------------------------------------------------------
    # 2. RGB Tensor 준비 (Compositing을 위해 일단 Premultiplied로 준비)
    # -------------------------------------------------------------------------
    # Compositing 과정 자체는 Alpha 연산이 필요하므로 입력은 Premultiplied로 준비합니다.
    # 나중에 비교 직전에 Un-multiply 할 것입니다.
    def get_prem_rgb_tensor_cpu(elems):
        tensors = []
        for e in elems:
            img_rgba = np.array(e["image"].convert("RGBA")).astype(np.float32) / 255.0
            if img_rgba.shape[:2] != (cH, cW):
                img_rgba = cv2.resize(img_rgba, (cW, cH), interpolation=cv2.INTER_AREA)
            
            alpha = img_rgba[..., 3:4]
            rgb = img_rgba[..., :3]
            rgb_prem = rgb * alpha  # Compositing용
            
            tensors.append(torch.from_numpy(rgb_prem).permute(2, 0, 1))
        return torch.stack(tensors)

    gt_rgbs_cpu = get_prem_rgb_tensor_cpu(gt_elements)
    pred_rgbs_cpu = get_prem_rgb_tensor_cpu(pred_elements)

    # -------------------------------------------------------------------------
    # 3. 그룹 렌더링 & Un-multiply (순수 RGB 복원)
    # -------------------------------------------------------------------------
    def render_groups_unmultiplied(groups, masks_gpu, rgbs_cpu):
        grp_masks = []
        grp_rgbs = []   # 순수 RGB (Non-premultiplied)
        grp_bboxes = []
        
        for idx_list in groups:
            # 3-1. Mask Combine (Union)
            union_mask = torch.max(masks_gpu[idx_list].float(), dim=0)[0]
            grp_masks.append(union_mask)
            
            sorted_indices = sorted(idx_list)
            
            # 캔버스 초기화 (Premultiplied 상태로 누적)
            # canvas_rgb: 누적된 RGB*Alpha
            # canvas_alpha: 누적된 Alpha
            canvas_rgb = torch.zeros((3, cH, cW), device=device)
            canvas_alpha = torch.zeros((1, cH, cW), device=device)

            if len(sorted_indices) == 0:
                grp_rgbs.append(torch.zeros((3, cH, cW), device=device))
                grp_bboxes.append((0, 0, 1, 1))
                continue
            
            # Painter's Algorithm (Compositing)
            for idx in sorted_indices:
                src_rgb = rgbs_cpu[idx].to(device)       # [3, H, W] (Premultiplied)
                src_a = masks_gpu[idx].to(device).unsqueeze(0) # [1, H, W]
                
                # RGB Compositing: Out = Src + Dst * (1 - Src_Alpha)
                canvas_rgb = src_rgb + canvas_rgb * (1.0 - src_a)
                
                # Alpha Compositing: Out_A = Src_A + Dst_A * (1 - Src_A)
                # (정확한 Alpha 합성을 위해 필요)
                canvas_alpha = src_a + canvas_alpha * (1.0 - src_a)
            
            # [핵심 수정] Un-multiply: RGB_pure = RGB_prem / Alpha
            # Alpha가 0에 가까운 곳은 나눗셈 오류 방지를 위해 0 처리
            safe_alpha = torch.clamp(canvas_alpha, min=1e-6)
            pure_rgb = canvas_rgb / safe_alpha
            
            # Alpha가 거의 없는 곳(완전 투명)은 RGB도 0으로 마스킹 (노이즈 방지)
            is_transparent = canvas_alpha < 1e-3
            pure_rgb = torch.where(is_transparent.expand(3, -1, -1), torch.zeros_like(pure_rgb), pure_rgb)
            
            grp_rgbs.append(pure_rgb)  # GPU에 유지

            # 3-3. BBox 계산
            coords = torch.nonzero(union_mask > 0)
            if coords.shape[0] > 0:
                y1, x1 = coords.min(dim=0)[0]
                y2, x2 = coords.max(dim=0)[0]
                grp_bboxes.append((int(x1), int(y1), int(x2+1), int(y2+1)))
            else:
                grp_bboxes.append((0, 0, 1, 1))

        return grp_masks, grp_rgbs, grp_bboxes

    # 렌더링 수행 (결과는 순수 RGB, GPU에 유지)
    gt_grp_masks, gt_grp_rgbs, gt_grp_bboxes = render_groups_unmultiplied(gt_groups, gt_masks_gpu, gt_rgbs_cpu)
    pred_grp_masks, pred_grp_rgbs, pred_grp_bboxes = render_groups_unmultiplied(pred_groups, pred_masks_gpu, pred_rgbs_cpu)

    # -------------------------------------------------------------------------
    # 4. Cost Matrix 계산
    # -------------------------------------------------------------------------
    cost_matrix = np.full((n_gt_grp, n_pred_grp), 10000.0, dtype=np.float64)
    l1_matrix = np.zeros((n_gt_grp, n_pred_grp))
    iou_matrix = np.zeros((n_gt_grp, n_pred_grp))

    # Pre-compute bbox overlap matrix to skip non-overlapping pairs
    gt_bb = np.array(gt_grp_bboxes)   # [n_gt_grp, 4] (x1, y1, x2, y2)
    pred_bb = np.array(pred_grp_bboxes)  # [n_pred_grp, 4]
    bb_overlap = (
        (gt_bb[:, 0:1] < pred_bb[None, :, 2]) &
        (gt_bb[:, 2:3] > pred_bb[None, :, 0]) &
        (gt_bb[:, 1:2] < pred_bb[None, :, 3]) &
        (gt_bb[:, 3:4] > pred_bb[None, :, 1])
    )

    for i in range(n_gt_grp):
        gm, gb = gt_grp_masks[i], gt_grp_bboxes[i]
        gr = gt_grp_rgbs[i]

        # Only iterate over pred groups with bbox overlap
        overlap_js = np.where(bb_overlap[i])[0]
        if len(overlap_js) == 0:
            continue

        for j in overlap_js:
            pm, pb = pred_grp_masks[j], pred_grp_bboxes[j]
            pr = pred_grp_rgbs[j]

            rx1, ry1 = min(gb[0], pb[0]), min(gb[1], pb[1])
            rx2, ry2 = max(gb[2], pb[2]), max(gb[3], pb[3])

            if rx2 <= rx1 or ry2 <= ry1:
                continue

            g_m_roi = gm[ry1:ry2, rx1:rx2]
            p_m_roi = pm[ry1:ry2, rx1:rx2]

            # IoU (Full Mask Union 기반)
            g_bin = g_m_roi > 0
            p_bin = p_m_roi > 0
            inter = torch.logical_and(g_bin, p_bin).sum()
            union_mask = torch.logical_or(g_bin, p_bin)
            union_count = union_mask.sum()

            iou = float(inter / (union_count + 1e-6))

            if iou <= 1e-6:
                continue

            # L1 Calculation on Union ROI (Pure RGB)
            if union_count > 0:
                g_r_roi = gr[:, ry1:ry2, rx1:rx2]
                p_r_roi = pr[:, ry1:ry2, rx1:rx2]

                diff = torch.abs(g_r_roi - p_r_roi)
                mask_broadcast = union_mask.unsqueeze(0).expand(3, -1, -1)
                valid_diff = diff[mask_broadcast]

                if valid_diff.numel() > 0:
                    l1 = float(valid_diff.mean())
                else:
                    l1 = 1.0
            else:
                l1 = 1.0

            penalty = (len(gt_groups[i])-1)*PENALTY_GT_MERGE + (len(pred_groups[j])-1)*PENALTY_PE_MERGE
            cost = LAMBDA_L1 * l1 + LAMBDA_IOU * (1 - iou) + penalty
            
            if cost > DUMMY_COST:
                cost_matrix[i, j] = 10000.0
            else:
                cost_matrix[i, j] = cost
                
            l1_matrix[i, j] = l1
            iou_matrix[i, j] = iou

    return cost_matrix, l1_matrix, iou_matrix

# =============================================================================
# ILP-based Optimal Matching
# =============================================================================

def solve_optimal_matching_ilp(
    cost_matrix: np.ndarray,
    gt_groups: List[List[int]],
    pred_groups: List[List[int]],
    n_gt_elements: int,
    n_pred_elements: int,
    logger: Optional[WorkerLogger] = None,
) -> List[Tuple[int, int]]:
    """
    Solve optimal matching using Integer Linear Programming.
    
    Key insight: We need to penalize NOT matching, otherwise ILP will match nothing
    (since matching nothing = cost 0, which is "optimal").
    
    Solution: Add dummy variables for unmatched elements with penalty cost.
    """
    n_gt_groups = len(gt_groups)
    n_pred_groups = len(pred_groups)
    total_vars = n_gt_groups * n_pred_groups
    
    if not PULP_AVAILABLE:
        if logger:
            logger.warn("[ILP] PuLP not available, using greedy matching")
        return solve_optimal_matching_greedy(cost_matrix, gt_groups, pred_groups, 
                                             n_gt_elements, n_pred_elements)
    
    if logger:
        logger.info(f"[ILP] Setting up problem: {n_gt_groups}x{n_pred_groups} = {total_vars} variables")
    
    start_time = time.time()
    
    prob = pulp.LpProblem("ElementMatching", pulp.LpMinimize)
    
    # Create decision variables: x[i,j] = 1 if gt_group[i] matches pred_group[j]
    if logger:
        logger.info(f"[ILP] Creating {total_vars} binary variables...")
    
    x = {}
    var_start = time.time()
    for i in range(n_gt_groups):
        for j in range(n_pred_groups):
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
    
    # Create dummy variables for unmatched GT elements
    # dummy_gt[i] = 1 if GT element i is not matched
    dummy_gt = {}
    for i in range(n_gt_elements):
        dummy_gt[i] = pulp.LpVariable(f"dummy_gt_{i}", cat=pulp.LpBinary)
    
    # Create dummy variables for unmatched Pred elements
    dummy_pred = {}
    for j in range(n_pred_elements):
        dummy_pred[j] = pulp.LpVariable(f"dummy_pred_{j}", cat=pulp.LpBinary)
    
    if logger:
        logger.info(f"[ILP] Variables created in {time.time() - var_start:.2f}s")
        logger.info(f"[ILP]   Match variables: {total_vars}, GT dummies: {n_gt_elements}, Pred dummies: {n_pred_elements}")
    
    # Objective: Minimize matching cost + penalty for unmatched elements
    # UNMATCHED_PENALTY should be set such that good matches are preferred over not matching
    UNMATCHED_PENALTY = DUMMY_COST  # Same as threshold - matching with cost < this is better than not matching
    
    if logger:
        logger.info(f"[ILP] Setting objective function (UNMATCHED_PENALTY={UNMATCHED_PENALTY})...")
    obj_start = time.time()
    
    # Cost for matched pairs
    matching_cost = pulp.lpSum(cost_matrix[i, j] * x[i, j] 
                               for i in range(n_gt_groups) 
                               for j in range(n_pred_groups))
    
    # Penalty for unmatched GT elements
    gt_penalty = pulp.lpSum(UNMATCHED_PENALTY * dummy_gt[i] for i in range(n_gt_elements))
    
    # Penalty for unmatched Pred elements
    pred_penalty = pulp.lpSum(UNMATCHED_PENALTY * dummy_pred[j] for j in range(n_pred_elements))
    
    prob += matching_cost + gt_penalty + pred_penalty
    
    if logger:
        logger.info(f"[ILP] Objective set in {time.time() - obj_start:.2f}s")
    
    # Constraint: Each GT element must be either matched (via some group) or marked as unmatched
    if logger:
        logger.info(f"[ILP] Adding GT element constraints ({n_gt_elements} elements)...")
    gt_const_start = time.time()
    
    for gt_elem_idx in range(n_gt_elements):
        # Find all gt_groups containing this element
        groups_with_elem = [i for i, grp in enumerate(gt_groups) if gt_elem_idx in grp]
        
        # Sum of all x[i,j] where group i contains this element + dummy = 1
        # This means: either matched through some group, or unmatched (dummy=1)
        prob += (pulp.lpSum(x[i, j] 
                          for i in groups_with_elem 
                          for j in range(n_pred_groups)) 
                + dummy_gt[gt_elem_idx]) >= 1
        
        # Also ensure at most one match (can't be in multiple matched groups)
        prob += pulp.lpSum(x[i, j] 
                          for i in groups_with_elem 
                          for j in range(n_pred_groups)) <= 1
    
    if logger:
        logger.info(f"[ILP] GT constraints added in {time.time() - gt_const_start:.2f}s")
    
    # Constraint: Each Pred element must be either matched or marked as unmatched
    if logger:
        logger.info(f"[ILP] Adding Pred element constraints ({n_pred_elements} elements)...")
    pred_const_start = time.time()
    
    for pred_elem_idx in range(n_pred_elements):
        groups_with_elem = [j for j, grp in enumerate(pred_groups) if pred_elem_idx in grp]
        
        prob += (pulp.lpSum(x[i, j] 
                          for i in range(n_gt_groups) 
                          for j in groups_with_elem)
                + dummy_pred[pred_elem_idx]) >= 1
        
        prob += pulp.lpSum(x[i, j] 
                          for i in range(n_gt_groups) 
                          for j in groups_with_elem) <= 1
    
    if logger:
        logger.info(f"[ILP] Pred constraints added in {time.time() - pred_const_start:.2f}s")
    
    # Solve
    if logger:
        logger.info(f"[ILP] Solving optimization problem...")
    solve_start = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    solve_time = time.time() - solve_start
    
    if logger:
        logger.info(f"[ILP] Solved in {solve_time:.2f}s | Status: {pulp.LpStatus[prob.status]}")
    
    # Extract solution
    matched_pairs = []
    if prob.status == pulp.LpStatusOptimal:
        # Count matched and unmatched
        n_matched_gt = sum(1 for i in range(n_gt_elements) 
                          if dummy_gt[i].value() is not None and dummy_gt[i].value() < 0.4)
        n_matched_pred = sum(1 for j in range(n_pred_elements) 
                            if dummy_pred[j].value() is not None and dummy_pred[j].value() < 0.4)
        
        for i in range(n_gt_groups):
            for j in range(n_pred_groups):
                if x[i, j].value() is not None and x[i, j].value() > 0.4:
                    matched_pairs.append((i, j))
        
        if logger:
            total_time = time.time() - start_time
            logger.info(f"[ILP] Found {len(matched_pairs)} optimal matches | Total: {total_time:.2f}s")
            logger.info(f"[ILP] Matched GT elements: {n_matched_gt}/{n_gt_elements}, "
                       f"Matched Pred elements: {n_matched_pred}/{n_pred_elements}")
            
            # Log matched pairs details
            if matched_pairs:
                logger.info(f"[ILP] Matched pairs details:")
                for i, j in matched_pairs[:10]:  # Show first 10
                    logger.info(f"[ILP]   GT_group[{i}]={gt_groups[i]} <-> Pred_group[{j}]={pred_groups[j]} | "
                               f"cost={cost_matrix[i,j]:.4f}")
                if len(matched_pairs) > 10:
                    logger.info(f"[ILP]   ... and {len(matched_pairs) - 10} more")
    else:
        if logger:
            logger.warn(f"[ILP] Solver failed with status: {pulp.LpStatus[prob.status]}")
    
    return matched_pairs


def solve_optimal_matching_hungarian(
    cost_matrix, gt_groups, pred_groups, 
    n_gt_elements, n_pred_elements, logger=None
):
    """Hungarian Algorithm - 최적 + 빠름"""
    from scipy.optimize import linear_sum_assignment
    
    # Cost matrix를 square로
    max_dim = max(len(gt_groups), len(pred_groups))
    padded = np.full((max_dim, max_dim), 999.0)
    padded[:len(gt_groups), :len(pred_groups)] = cost_matrix
    
    # Hungarian (O(n³), 하지만 실제로는 매우 빠름)
    row_ind, col_ind = linear_sum_assignment(padded)
    
    # Element 중복 체크하며 결과 필터링
    matched_pairs = []
    used_gt = set()
    used_pred = set()
    
    for i, j in zip(row_ind, col_ind):
        if i >= len(gt_groups) or j >= len(pred_groups):
            continue
        if padded[i, j] >= DUMMY_COST:
            continue
            
        gt_elems = gt_groups[i]
        pred_elems = pred_groups[j]
        
        if any(e in used_gt for e in gt_elems):
            continue
        if any(e in used_pred for e in pred_elems):
            continue
        
        matched_pairs.append((i, j))
        used_gt.update(gt_elems)
        used_pred.update(pred_elems)
    
    return matched_pairs

def solve_optimal_matching_greedy(
    cost_matrix: np.ndarray,
    gt_groups: List[List[int]],
    pred_groups: List[List[int]],
    n_gt_elements: int,
    n_pred_elements: int,
) -> List[Tuple[int, int]]:
    n_gt_groups = len(gt_groups)
    n_pred_groups = len(pred_groups)
    
    all_pairs = []
    for i in range(n_gt_groups):
        for j in range(n_pred_groups):
            if cost_matrix[i, j] < DUMMY_COST:
                all_pairs.append((cost_matrix[i, j], i, j))
    
    all_pairs.sort(key=lambda x: x[0])
    
    matched_pairs = []
    used_gt_elems: Set[int] = set()
    used_pred_elems: Set[int] = set()
    
    for cost, i, j in all_pairs:
        gt_idx_list = gt_groups[i]
        pred_idx_list = pred_groups[j]
        
        if any(idx in used_gt_elems for idx in gt_idx_list):
            continue
        if any(idx in used_pred_elems for idx in pred_idx_list):
            continue
        
        matched_pairs.append((i, j))
        used_gt_elems.update(gt_idx_list)
        used_pred_elems.update(pred_idx_list)
    
    return matched_pairs


# =============================================================================
# Main Optimal Matching Function
# =============================================================================

def match_elements_optimal(
    gt_elements: List[Dict],
    pred_elements: List[Dict],
    canvas_size: Tuple[int, int],
    apply_pred_alpha_cleaning: bool = True,
    verbose: bool = True,
    logger: Optional[WorkerLogger] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if not gt_elements or not pred_elements:
        return [], gt_elements.copy(), pred_elements.copy()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu" [cite: 1]
    
    # 1. 가시 영역 마스크 계산
    gt_with_visible = compute_visible_masks(gt_elements, canvas_size)
    pred_with_visible = compute_visible_masks(pred_elements, canvas_size, apply_alpha_cleaning=apply_pred_alpha_cleaning)
    
    # 2. Containment 기반 그룹 생성 (개선된 로직) 
    gt_groups, pred_groups = generate_combinatorial_groups(gt_with_visible, pred_with_visible, logger=logger)
    
    # 3. GPU 가속 비용 행렬 생성 (개선된 로직) 
    cost_matrix, l1_matrix, iou_matrix = build_cost_matrix_gpu(
        gt_with_visible, pred_with_visible, gt_groups, pred_groups, canvas_size, device=device, logger=logger
    )
    
    # 4. ILP 최적 매칭 (기존 로직 유지) 
    matched_group_pairs = solve_optimal_matching_hungarian(
        cost_matrix, gt_groups, pred_groups, len(gt_with_visible), len(pred_with_visible), logger=logger
    )
    
    if verbose and logger:
        logger.info(f"[OptimalMatch] Completed: {len(matched_group_pairs)} matched pairs")
    
    matched_pairs = []
    used_gt_indices: Set[int] = set()
    used_pred_indices: Set[int] = set()
    
    for gt_grp_idx, pred_grp_idx in matched_group_pairs:
        gt_idx_list = gt_groups[gt_grp_idx]
        pred_idx_list = pred_groups[pred_grp_idx]
        
        used_gt_indices.update(gt_idx_list)
        used_pred_indices.update(pred_idx_list)
        
        if len(gt_idx_list) == 1 and len(pred_idx_list) == 1:
            match_type = "one_to_one"
        elif len(gt_idx_list) == 1:
            match_type = "one_to_many"
        elif len(pred_idx_list) == 1:
            match_type = "many_to_one"
        else:
            match_type = "many_to_many"
        
        gt_union_mask, gt_composite = merge_elements_visible(
            gt_with_visible, gt_idx_list, canvas_size
        )
        

        gt_selected = [gt_with_visible[i] for i in gt_idx_list]
        gt_full_img, gt_full_mask = composite_elements_transparent(gt_selected, canvas_size)


        rows = np.any(gt_union_mask > 0, axis=1)
        cols = np.any(gt_union_mask > 0, axis=0)

        if rows.any() and cols.any():
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            merged_bbox = [int(x1), int(y1), int(x2+1), int(y2+1)]
        else:
            merged_bbox = [0, 0, 1, 1]


        if len(gt_idx_list) == 1:
            elem_type = gt_with_visible[gt_idx_list[0]].get("type", "object")
        else:
            elem_type = "merged"

        merged_gt = {
            "id": f"merged_gt_{'_'.join(str(i) for i in gt_idx_list)}",
            "type": elem_type,
            "mask": gt_full_mask,      # 가려짐 없는 전체 마스크
            "image": gt_full_img,      # 가려짐 없는 전체 이미지
            "bbox": merged_bbox,
            "area": float(gt_full_mask.sum()),
            "visible_mask": gt_union_mask, # 매칭용 가시 마스크는 참조용으로 유지
            "visible_area": float(gt_union_mask.sum()),
            "z_index": min(gt_with_visible[i].get("z_index", 0) for i in gt_idx_list),
        }
        
        
        
        matched_pairs.append({
            "gt": merged_gt,
            "gt_indices": gt_idx_list,
            "gt_elements": [gt_with_visible[i] for i in gt_idx_list],
            "preds": [pred_with_visible[j] for j in pred_idx_list],
            "pred_indices": pred_idx_list,
            "match_type": match_type,
            "cost": float(cost_matrix[gt_grp_idx, pred_grp_idx]),
            "metrics": {
                "l1": float(l1_matrix[gt_grp_idx, pred_grp_idx]),
                "iou": float(iou_matrix[gt_grp_idx, pred_grp_idx]),
            },
            "gt_coverage": float(iou_matrix[gt_grp_idx, pred_grp_idx]),
        })
    
    unmatched_gt = [
        gt_with_visible[i] for i in range(len(gt_with_visible))
        if i not in used_gt_indices and gt_with_visible[i].get("visible_area", gt_with_visible[i]["area"]) >= MIN_ELEMENT_AREA
    ]
    unmatched_pred = [
        pred_with_visible[j] for j in range(len(pred_with_visible))
        if j not in used_pred_indices and pred_with_visible[j].get("visible_area", pred_with_visible[j]["area"]) >= MIN_ELEMENT_AREA
    ]
    
    return matched_pairs, unmatched_gt, unmatched_pred


# =============================================================================
# Legacy Matching
# =============================================================================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    bin1 = mask1 > 0
    bin2 = mask2 > 0
    intersection = (bin1 & bin2).sum()
    union = (bin1 | bin2).sum()
    return float(intersection / (union + 1e-6))


def resize_mask_if_needed(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    return cv2.resize(mask.astype(np.float32), (target_shape[1], target_shape[0]), 
                      interpolation=cv2.INTER_LINEAR)


# =============================================================================
# Metric Computation
# =============================================================================

def calc_soft_iou(mask1: np.ndarray, mask2: np.ndarray, eps: float = 1e-8) -> float:
    """
    Soft IoU (LayerD 방식)
    Alpha 값(0~1 연속값)을 그대로 사용
    """
    m1 = mask1.astype(np.float32)
    m2 = mask2.astype(np.float32)
    
    # Normalize to [0, 1] if needed
    if m1.max() > 1.0:
        m1 = m1 / 255.0
    if m2.max() > 1.0:
        m2 = m2 / 255.0
    
    intersection = np.minimum(m1, m2).sum()
    union = np.maximum(m1, m2).sum()
    
    return float(intersection / (union + eps))


def calc_binary_iou(mask1: np.ndarray, mask2: np.ndarray, eps: float = 1e-8) -> float:
    """
    Binary IoU (기존 방식)
    Alpha를 0/1로 이진화 후 계산
    """
    bin1 = mask1 > 0
    bin2 = mask2 > 0
    
    intersection = (bin1 & bin2).sum()
    union = (bin1 | bin2).sum()
    
    return float(intersection / (union + eps))


# =============================================================================
# Visual Quality 계산 함수 (LayerD 방식)
# =============================================================================

def compute_visual_quality_no_composite(
    gt_rgba: np.ndarray,
    pred_rgba: np.ndarray,
    region: str = "intersection", # 'intersection' or 'union'
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    Visual Quality Calculation with Pre-multiplied Alpha (Min-Error Strategy)
    
    - RGB 값에 Alpha를 곱한 Pre-multiplied RGB를 사용합니다.
    - Black 배경과 White 배경에 대해 정식 Alpha Compositing을 수행한 뒤,
      두 경우 중 더 오차가 적은(Min) 값을 선택합니다.
    - 이는 객체가 어두운 배경이나 밝은 배경 중 적어도 한 곳에서는 
      아티팩트 없이 자연스럽게 보이는지를 평가합니다.
    
    Args:
        gt_rgba: GT RGBA [H, W, 4], uint8
        pred_rgba: Pred RGBA [H, W, 4], uint8
        region: "intersection" or "union" (default: intersection)
    
    Returns:
        {"l1": float, "l2": float, "psnr": float} (Min error / Max PSNR of Black/White cases)
    """
    # 1. Normalize to [0, 1]
    gt_float = gt_rgba.astype(np.float32) / 255.0
    pred_float = pred_rgba.astype(np.float32) / 255.0
    
    # 2. Extract & Pre-multiply Alpha
    # (H, W, 1) 형태로 유지하여 브로드캐스팅 가능하게 함
    gt_alpha = gt_float[..., 3:4]
    pred_alpha = pred_float[..., 3:4]
    
    # Pre-multiply: RGB * Alpha
    gt_prem = gt_float[..., :3] * gt_alpha
    pred_prem = pred_float[..., :3] * pred_alpha
    
    # 3. Define ROI Mask
    gt_visible = gt_float[..., 3] > 0
    pred_visible = pred_float[..., 3] > 0
    
    if region == "intersection":
        roi_mask = gt_visible & pred_visible
    elif region == "union":
        roi_mask = gt_visible | pred_visible
    else:
        roi_mask = gt_visible | pred_visible
    
    if roi_mask.sum() == 0:
        return {"l1": float('nan'), "l2": float('nan'), "psnr": float('nan')}

    # 4. Prepare Background Versions (Alpha Compositing)
    
    # Case 1: Black Background (0, 0, 0)
    # Formula: Color * Alpha + Black * (1 - Alpha) 
    # Black=0 이므로 Color * Alpha (즉, Premultiplied 그 자체)
    gt_img_b = gt_prem
    pred_img_b = pred_prem
    
    # Case 2: White Background (1, 1, 1)
    # Formula: Color * Alpha + White * (1 - Alpha)
    gt_img_w = gt_prem + (1.0 - gt_alpha)
    pred_img_w = pred_prem + (1.0 - pred_alpha)
    
    # 5. Helper Function for Metrics
    def calc_metrics(p_gt_full, p_pred_full):
        # ROI 영역 내의 픽셀만 추출하여 비교
        p_gt = p_gt_full[roi_mask]
        p_pred = p_pred_full[roi_mask]
        
        val_l1 = float(np.mean(np.abs(p_gt - p_pred)))
        val_l2 = float(mean_squared_error(p_gt, p_pred))
        val_psnr = float(peak_signal_noise_ratio(p_gt, p_pred, data_range=1.0))
        
        # inf는 그대로 보존 → 집계 시 유한 최댓값으로 대체
        return val_l1, val_l2, val_psnr

    # 6. Calculate & Select Best (Min Error / Max PSNR)
    l1_b, l2_b, psnr_b = calc_metrics(gt_img_b, pred_img_b)
    l1_w, l2_w, psnr_w = calc_metrics(gt_img_w, pred_img_w)
    
    return {
        "l1": min(l1_b, l1_w),
        "l2": min(l2_b, l2_w),
        "psnr": max(psnr_b, psnr_w)
    }

# =============================================================================
# Element-wise Dual 메트릭 계산
# =============================================================================

def compute_element_metrics_dual(
    gt_elem: Dict,
    pred_elems: List[Dict],
    canvas_size: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Element-wise 메트릭을 2가지 방식으로 계산
    
    Returns:
        {
            "visual_quality": {
                "gt_region": {"l1", "l2", "psnr"},   # GT 영역만
                "union_region": {"l1", "l2", "psnr"} # 합집합 영역
            },
            "iou": {
                "soft": float,   # Soft IoU (LayerD)
                "binary": float  # Binary IoU
            },
            "areas": {"gt": float, "pred": float, "union": float}
        }
    """
    W, H = canvas_size
    
    # =========================================================================
    # GT 이미지 준비
    # =========================================================================
    gt_img = gt_elem["image"].convert("RGBA")
    if gt_img.size != (W, H):
        gt_img = gt_img.resize((W, H), Image.LANCZOS)
    gt_rgba = np.array(gt_img)
    
    # GT 마스크
    gt_mask = gt_elem["mask"]
    if gt_mask.shape != (H, W):
        gt_mask = cv2.resize(gt_mask.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    
    # =========================================================================
    # Pred 합성 이미지 준비
    # =========================================================================
    pred_canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pred_mask = np.zeros((H, W), dtype=np.float32)
    
    sorted_preds = sorted(pred_elems, key=lambda x: x.get("z_index", 0))
    for elem in sorted_preds:
        elem_img = elem["image"].convert("RGBA")
        if elem_img.size != (W, H):
            elem_img = elem_img.resize((W, H), Image.LANCZOS)
        pred_canvas.paste(elem_img, (0, 0), elem_img)
        
        elem_mask = elem["mask"]
        if elem_mask.shape != (H, W):
            elem_mask = cv2.resize(elem_mask.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

        pred_mask = pred_mask + elem_mask * (1.0 - pred_mask)
    
    pred_rgba = np.array(pred_canvas)
    
    # =========================================================================
    # 영역 계산
    # =========================================================================
    gt_area = float((gt_mask > 0).sum())
    pred_area = float((pred_mask > 0).sum())
    union_area = float(((gt_mask > 0) | (pred_mask > 0)).sum())
    
    # =========================================================================
    # Visual Quality 계산 (Alpha Composite 없이!)
    # =========================================================================
    vq_intersection  = compute_visual_quality_no_composite(gt_rgba, pred_rgba, region="intersection")
    vq_union_region = compute_visual_quality_no_composite(gt_rgba, pred_rgba, region="union")
    
    # =========================================================================
    # IoU 계산
    # =========================================================================
    soft_iou = calc_soft_iou(gt_mask, pred_mask)
    binary_iou = calc_binary_iou(gt_mask, pred_mask)
    
    return {
        "visual_quality": {
            "intersection_region": vq_intersection,
            "union_region": vq_union_region,
        },
        "iou": {
            "soft": soft_iou,
            "binary": binary_iou,
        },
        "areas": {
            "gt": gt_area,
            "pred": pred_area,
            "union": union_area,
        }
    }


# =============================================================================
# Panoptic Quality (PQ, SQ, RQ) 계산
# =============================================================================

def compute_panoptic_quality_dual(
    matched_pairs: List[Dict],
    unmatched_gt: List[Dict],
    unmatched_pred: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    PQ를 두 가지 IoU 방식으로 계산 (Kirillov et al., CVPR 2019 표준)
    
    IoU > iou_threshold인 매칭만 TP로 분류.
    IoU ≤ iou_threshold인 매칭은 FP+FN으로 분류.
    
    Returns:
        {
            "soft": {"pq", "sq", "rq", "tp", "fn", "fp"},
            "binary": {"pq", "sq", "rq", "tp", "fn", "fp"}
        }
    """
    results = {}
    
    for iou_type in ["soft", "binary"]:
        # IoU threshold 기반 TP/FP/FN 분류
        tp_pairs = []
        poor_matches = 0
        
        for mp in matched_pairs:
            iou_val = mp.get("metrics_dual", {}).get("iou", {}).get(iou_type, 0)
            if iou_val > iou_threshold:
                tp_pairs.append(iou_val)
            else:
                poor_matches += 1
        
        tp_count = len(tp_pairs)
        # Poor match는 FP와 FN 양쪽에 가산 (Kirillov et al. 표준)
        fn = len(unmatched_gt) + poor_matches
        fp = len(unmatched_pred) + poor_matches
        
        if tp_count == 0:
            results[iou_type] = {
                "pq": 0.0, "sq": 0.0, "rq": 0.0,
                "tp": 0, "fn": fn, "fp": fp,
            }
            continue
        
        rq = tp_count / (tp_count + 0.5 * fp + 0.5 * fn + 1e-6)
        sq = sum(tp_pairs) / tp_count
        pq = sq * rq
        
        results[iou_type] = {
            "pq": float(pq),
            "sq": float(sq),
            "rq": float(rq),
            "tp": tp_count,
            "fn": fn,
            "fp": fp,
        }
    
    return results


# =============================================================================
# Element 메트릭 집계
# =============================================================================

def aggregate_element_metrics_dual(matched_pairs: List[Dict]) -> Dict[str, Any]:
    """
    전체 매칭 페어에 대해 Dual 메트릭 집계
    
    - NaN 값은 집계에서 제외
    - PSNR의 inf 값은 유한 최댓값으로 대체 후 집계
    
    Returns:
        {
            "visual_quality": {
                "gt_region": {"l1", "l2", "psnr"},
                "union_region": {"l1", "l2", "psnr"}
            },
            "iou": {"soft", "binary"},
            (각각 simple_avg, weighted_avg 포함)
        }
    """
    if not matched_pairs:
        return {
            "visual_quality": {
                "intersection_region": {"simple_avg": {}, "weighted_avg": {}},
                "union_region": {"simple_avg": {}, "weighted_avg": {}},
            },
            "iou": {"simple_avg": {}, "weighted_avg": {}}
        }
    
    def _aggregate_values(values, weights, is_psnr=False):
        """NaN 제거, PSNR inf→max_finite 대체 후 평균 계산"""
        # NaN 제거
        clean = [(v, w) for v, w in zip(values, weights) if not np.isnan(v)]
        if not clean:
            return float('nan'), float('nan')
        
        vals = [v for v, w in clean]
        wts = [w for v, w in clean]
        
        # PSNR: inf → 유한 최댓값으로 대체
        if is_psnr:
            finite_vals = [v for v in vals if np.isfinite(v)]
            if finite_vals:
                max_finite = max(finite_vals)
                vals = [v if np.isfinite(v) else max_finite for v in vals]
            else:
                return float('nan'), float('nan')
        
        simple = float(np.mean(vals))
        weighted = float(np.average(vals, weights=wts)) if sum(wts) > 0 else simple
        return simple, weighted
    
    # Visual Quality 집계
    vq_result = {
        "intersection_region": {"simple_avg": {}, "weighted_avg": {}},
        "union_region": {"simple_avg": {}, "weighted_avg": {}},
    }
    
    for region in ["intersection_region", "union_region"]:
        for metric in ["l1", "l2", "psnr"]:
            values = []
            weights = []
            for mp in matched_pairs:
                md = mp.get("metrics_dual", {})
                vq = md.get("visual_quality", {}).get(region, {})
                val = vq.get(metric, float('nan'))
                values.append(val)
                weights.append(mp["gt"]["area"])
            
            simple, weighted = _aggregate_values(
                values, weights, is_psnr=(metric == "psnr")
            )
            vq_result[region]["simple_avg"][metric] = simple
            vq_result[region]["weighted_avg"][metric] = weighted
    
    # IoU 집계
    iou_result = {"simple_avg": {}, "weighted_avg": {}}
    
    for iou_type in ["soft", "binary"]:
        values = []
        weights = []
        for mp in matched_pairs:
            md = mp.get("metrics_dual", {})
            iou_dict = md.get("iou", {})
            val = iou_dict.get(iou_type, 0)
            values.append(val)
            weights.append(mp["gt"]["area"])
        
        iou_result["simple_avg"][iou_type] = float(np.mean(values)) if values else 0.0
        iou_result["weighted_avg"][iou_type] = (
            float(np.average(values, weights=weights)) if sum(weights) > 0 else 0.0
        )
    
    return {
        "visual_quality": vq_result,
        "iou": iou_result,
    }


# =============================================================================
# 로깅 함수
# =============================================================================

def log_dual_metrics(matched_pairs: List[Dict], pq_dual: Dict, logger=None):
    """두 방식의 메트릭을 로깅"""
    
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    def _clean(vals, is_psnr=False):
        """NaN 제거, PSNR이면 inf→max_finite 대체"""
        clean = [v for v in vals if v is not None and not np.isnan(v)]
        if not clean:
            return [0.0]
        if is_psnr:
            finite = [v for v in clean if np.isfinite(v)]
            if finite:
                mx = max(finite)
                clean = [v if np.isfinite(v) else mx for v in clean]
        return clean
    
    if not matched_pairs:
        log("  No matched pairs to log.")
        return
    
    # 데이터 수집
    inter_l1s, inter_l2s, inter_psnrs = [], [], []
    union_l1s, union_l2s, union_psnrs = [], [], []
    soft_ious, binary_ious = [], []
    
    for mp in matched_pairs:
        md = mp.get("metrics_dual", {})
        vq = md.get("visual_quality", {})
        iou = md.get("iou", {})
        
        inter_r = vq.get("intersection_region", {})
        inter_l1s.append(inter_r.get("l1"))
        inter_l2s.append(inter_r.get("l2"))
        inter_psnrs.append(inter_r.get("psnr"))
        
        union_r = vq.get("union_region", {})
        union_l1s.append(union_r.get("l1", 0))
        union_l2s.append(union_r.get("l2", 0))
        union_psnrs.append(union_r.get("psnr", 0))

        soft_ious.append(iou.get("soft", 0))
        binary_ious.append(iou.get("binary", 0))
    
    # NaN/inf 정리
    c_inter_l1 = _clean(inter_l1s); c_inter_l2 = _clean(inter_l2s); c_inter_psnr = _clean(inter_psnrs, is_psnr=True)
    c_union_l1 = _clean(union_l1s); c_union_l2 = _clean(union_l2s); c_union_psnr = _clean(union_psnrs, is_psnr=True)
    
    log("=" * 100)
    log("[Visual Quality - No Alpha Composite]")
    log("-" * 100)
    log(f"  {'Metric':<8} {'Intersection Region':<28} {'Union Region':<28} {'GT Region (LayerD)':<28}")
    log(f"  {'-'*92}")
    log(f"  {'L1':<8} min={min(c_inter_l1):.4f} max={max(c_inter_l1):.4f} mean={np.mean(c_inter_l1):.4f}   "
        f"min={min(c_union_l1):.4f} max={max(c_union_l1):.4f} mean={np.mean(c_union_l1):.4f}   "
        )
    log(f"  {'L2':<8} min={min(c_inter_l2):.4f} max={max(c_inter_l2):.4f} mean={np.mean(c_inter_l2):.4f}   "
        f"min={min(c_union_l2):.4f} max={max(c_union_l2):.4f} mean={np.mean(c_union_l2):.4f}   "
        )
    log(f"  {'PSNR':<8} min={min(c_inter_psnr):.2f} max={max(c_inter_psnr):.2f} mean={np.mean(c_inter_psnr):.2f}   "
        f"min={min(c_union_psnr):.2f} max={max(c_union_psnr):.2f} mean={np.mean(c_union_psnr):.2f}   "
        )
    
    log("-" * 100)
    log("[Layout - IoU]")
    log(f"  {'Metric':<8} {'Soft IoU (LayerD)':<28} {'Binary IoU':<28}")
    log(f"  {'-'*64}")
    log(f"  {'IoU':<8} min={min(soft_ious):.4f} max={max(soft_ious):.4f} mean={np.mean(soft_ious):.4f}   "
        f"min={min(binary_ious):.4f} max={max(binary_ious):.4f} mean={np.mean(binary_ious):.4f}")
    
    soft = pq_dual.get("soft", {})
    binary = pq_dual.get("binary", {})
    log(f"  {'PQ':<8} {soft.get('pq', 0):.4f}                            {binary.get('pq', 0):.4f}")
    log(f"  {'SQ':<8} {soft.get('sq', 0):.4f}                            {binary.get('sq', 0):.4f}")
    log(f"  {'RQ':<8} {soft.get('rq', 0):.4f}                            {binary.get('rq', 0):.4f}")
    log("=" * 100)


def print_dual_comparison(summary: Dict, method: str):
    """Dual 메트릭 비교 테이블 출력"""
    print(f"\n{'='*95}")
    print(f"  {method.upper()} - DUAL METRIC COMPARISON")
    print(f"{'='*95}")
    
    vq = summary.get("element_metrics_dual", {}).get("visual_quality", {})
    inter_r = vq.get("intersection_region", {}).get("simple_avg", {})
    union_r = vq.get("union_region", {}).get("simple_avg", {})
    iou_agg = summary.get("element_metrics_dual", {}).get("iou", {}).get("simple_avg", {})
    pq_dual = summary.get("panoptic_quality_dual", {})
    
    print("\n  [Visual Quality - No Alpha Composite]")
    print(f"  {'Metric':<12} {'Intersection':<22} {'Union Region':<22} {'GT Region (LayerD)':<22}")
    print(f"  {'-'*78}")
    print(f"  {'L1':<12} {inter_r.get('l1', 0):<22.4f} {union_r.get('l1', 0):<22.4f}")
    print(f"  {'L2':<12} {inter_r.get('l2', 0):<22.4f} {union_r.get('l2', 0):<22.4f}")
    print(f"  {'PSNR':<12} {inter_r.get('psnr', 0):<22.2f} {union_r.get('psnr', 0):<22.2f}")
    
    print("\n  [Layout]")
    print(f"  {'Metric':<12} {'Soft IoU (LayerD)':<22} {'Binary IoU':<22}")
    print(f"  {'-'*56}")
    print(f"  {'IoU':<12} {iou_agg.get('soft', 0):<22.4f} {iou_agg.get('binary', 0):<22.4f}")
    print(f"  {'PQ':<12} {pq_dual.get('soft', {}).get('pq', 0):<22.4f} {pq_dual.get('binary', {}).get('pq', 0):<22.4f}")
    print(f"  {'SQ':<12} {pq_dual.get('soft', {}).get('sq', 0):<22.4f} {pq_dual.get('binary', {}).get('sq', 0):<22.4f}")
    print(f"  {'RQ':<12} {pq_dual.get('soft', {}).get('rq', 0):<22.4f} {pq_dual.get('binary', {}).get('rq', 0):<22.4f}")
    print()


# =============================================================================
# 여러 에피소드 결과 집계
# =============================================================================

def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """여러 에피소드 결과를 집계 (Dual 메트릭 포함)
    
    - NaN 값은 집계에서 제외
    - PSNR의 inf 값은 유한 최댓값으로 대체 후 집계
    """
    if not results:
        return {}
    
    def _safe_mean(vals, is_psnr=False):
        """NaN 제거 후 평균. PSNR이면 inf→max_finite 대체."""
        clean = [v for v in vals if not np.isnan(v)]
        if not clean:
            return float('nan')
        if is_psnr:
            finite = [v for v in clean if np.isfinite(v)]
            if finite:
                mx = max(finite)
                clean = [v if np.isfinite(v) else mx for v in clean]
            else:
                return float('nan')
        return float(np.mean(clean))
    
    # 기존 집계 (하위 호환성)
    all_simple, all_weighted = {}, {}
    for key in ["l1", "l2", "psnr", "iou"]:
        simple_vals = [r["element_metrics"]["simple_avg"].get(key, 0) for r in results]
        weighted_vals = [r["element_metrics"]["weighted_avg"].get(key, 0) for r in results]
        all_simple[key] = _safe_mean(simple_vals, is_psnr=(key == "psnr"))
        all_weighted[key] = _safe_mean(weighted_vals, is_psnr=(key == "psnr"))
    
    pq_vals = [r["panoptic_quality"]["pq_count"] for r in results]
    sq_vals = [r["panoptic_quality"]["sq"] for r in results]
    rq_vals = [r["panoptic_quality"]["rq"] for r in results]
    
    comp_agg = {}
    results_with_comp = [
        r for r in results
        if r.get("composite_metrics") is not None
        and not r.get("counts", {}).get("composite_skipped", False)
        and r.get("counts", {}).get("pred_non_text", r.get("counts", {}).get("pred_elements", 99)) > 5
    ]
    comp_agg["num_episodes_skipped"] = len(results) - len(results_with_comp)
    for key in ["l1", "psnr", "ssim", "lpips", "dino", "iou"]:
        vals = [r["composite_metrics"].get(key, 0) for r in results_with_comp]
        comp_agg[key] = _safe_mean(vals, is_psnr=(key == "psnr"))
    comp_agg["num_episodes"] = len(results_with_comp)
    
    # =========================================================================
    # [NEW] Dual 메트릭 집계
    # =========================================================================
    vq_agg = {
        # [수정] 키 이름 변경
        "intersection_region": {"simple_avg": {}, "weighted_avg": {}},
        "union_region": {"simple_avg": {}, "weighted_avg": {}},
    }
    for region in ["intersection_region", "union_region"]:
        for metric in ["l1", "l2", "psnr"]:
            simple_vals = [r["element_metrics_dual"]["visual_quality"][region]["simple_avg"].get(metric, 0) 
                          for r in results if "element_metrics_dual" in r]
            weighted_vals = [r["element_metrics_dual"]["visual_quality"][region]["weighted_avg"].get(metric, 0) 
                           for r in results if "element_metrics_dual" in r]
            vq_agg[region]["simple_avg"][metric] = _safe_mean(simple_vals, is_psnr=(metric == "psnr"))
            vq_agg[region]["weighted_avg"][metric] = _safe_mean(weighted_vals, is_psnr=(metric == "psnr"))
    
    iou_agg = {"simple_avg": {}, "weighted_avg": {}}
    for iou_type in ["soft", "binary"]:
        simple_vals = [r["element_metrics_dual"]["iou"]["simple_avg"].get(iou_type, 0) 
                      for r in results if "element_metrics_dual" in r]
        weighted_vals = [r["element_metrics_dual"]["iou"]["weighted_avg"].get(iou_type, 0) 
                        for r in results if "element_metrics_dual" in r]
        iou_agg["simple_avg"][iou_type] = float(np.mean(simple_vals)) if simple_vals else 0.0
        iou_agg["weighted_avg"][iou_type] = float(np.mean(weighted_vals)) if weighted_vals else 0.0
    
    pq_dual_agg = {"soft": {}, "binary": {}}
    for style in ["soft", "binary"]:
        for metric in ["pq", "sq", "rq"]:
            vals = [r["panoptic_quality_dual"][style][metric] for r in results if "panoptic_quality_dual" in r]
            pq_dual_agg[style][metric] = float(np.mean(vals)) if vals else 0.0
    
    return {
        "num_episodes": len(results),
        # 기존 필드
        "element_metrics": {"simple_avg": all_simple, "weighted_avg": all_weighted},
        "panoptic_quality": {"pq": float(np.mean(pq_vals)), "sq": float(np.mean(sq_vals)), "rq": float(np.mean(rq_vals))},
        "composite_metrics": comp_agg,
        # [NEW] Dual 필드
        "element_metrics_dual": {"visual_quality": vq_agg, "iou": iou_agg},
        "panoptic_quality_dual": pq_dual_agg,
    }

# =============================================================================
# 테스트
# =============================================================================

# if __name__ == "__main__":
#     print("Dual Metrics Module - Test")
#     print("=" * 50)
    
#     # 테스트용 마스크 생성
#     mask1 = np.zeros((100, 100), dtype=np.float32)
#     mask1[20:80, 20:80] = 0.8  # Semi-transparent
    
#     mask2 = np.zeros((100, 100), dtype=np.float32)
#     mask2[30:90, 30:90] = 1.0  # Opaque
    
#     soft_iou = calc_soft_iou(mask1, mask2)
#     binary_iou = calc_binary_iou(mask1, mask2)
    
#     print(f"\nIoU Comparison:")
#     print(f"  Soft IoU (LayerD): {soft_iou:.4f}")
#     print(f"  Binary IoU:        {binary_iou:.4f}")
#     print(f"  Difference:        {soft_iou - binary_iou:.4f}")
#     print(f"  (Soft IoU is typically higher because it considers partial alpha values)")
    
#     # 테스트용 RGBA 이미지 생성
#     gt_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
#     gt_rgba[20:80, 20:80, :3] = [200, 100, 50]  # RGB
#     gt_rgba[20:80, 20:80, 3] = 200  # Alpha
    
#     pred_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
#     pred_rgba[30:90, 30:90, :3] = [180, 90, 60]  # Slightly different RGB
#     pred_rgba[30:90, 30:90, 3] = 255  # Alpha
    
#     vq_intersection = compute_visual_quality_no_composite(gt_rgba, pred_rgba, region="intersection")
#     vq_union = compute_visual_quality_no_composite(gt_rgba, pred_rgba, region="union")
    
#     print(f"\nVisual Quality Comparison (No Alpha Composite):")
#     print(f"  Intersection Region Only:    L1={vq_intersection['l1']:.4f}, L2={vq_intersection['l2']:.4f}, PSNR={vq_intersection['psnr']:.2f}")
#     print(f"  Union Region:      L1={vq_union['l1']:.4f}, L2={vq_union['l2']:.4f}, PSNR={vq_union['psnr']:.2f}")
#     print(f"  (Union region includes areas where Pred exists but Intersection doesn't)")




def composite_elements(
    elements: List[Dict], 
    canvas_size: Tuple[int, int],
    use_overwrite: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    # 1. 투명 캔버스에 요소들 배치 (PIL Image RGBA 반환)
    canvas_rgba_pil, total_mask = composite_elements_transparent(elements, canvas_size, use_overwrite=use_overwrite)
    
    # 2. 배경(검정)과 섞지 않고, 순수 RGBA 값을 0~1 float로 변환하여 반환
    # (H, W, 4) Shape 유지
    composite_arr = np.array(canvas_rgba_pil).astype(np.float32) / 255.0
    
    return composite_arr, total_mask


def composite_elements_transparent(
    elements: List[Dict], 
    canvas_size: Tuple[int, int],
    use_overwrite: bool = False
) -> Tuple[Image.Image, np.ndarray]:
    W, H = canvas_size
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    total_mask = np.zeros((H, W), dtype=np.float32)
    
    sorted_elems = sorted(elements, key=lambda x: x.get("z_index", 0))
    
    for elem in sorted_elems:
        elem_img = elem["image"].convert("RGBA")
        
        # 1. 크기 보정
        if elem_img.size != (W, H):
            elem_img = elem_img.resize((W, H), Image.LANCZOS)
        
        # 2. 합성 로직 분기
        # use_overwrite가 True여도, 'text' 타입에만 적용하여 Halo 현상 방지
        # (이미지가 Flatten된 원본에서 왔다면, 텍스트는 배경색과 이미 섞여있으므로 덮어써야 함)
        is_text = elem.get("type") == "text"
        
        if use_overwrite and is_text:
            # [Overwrite Logic] Text Only
            # 마스크 영역(Alpha > 0)을 255로 이진화하여 '구멍'을 뚫고, 그 자리에 픽셀을 1:1로 채움
            alpha = elem_img.getchannel("A")
            # 1이라도 있으면 완전 불투명(255)으로 취급하여 덮어쓰기 마스크 생성
            binary_mask = alpha.point(lambda p: 255 if p > 0 else 0)
            
            # canvas의 해당 영역을 elem_img로 완전히 대체 (Blend 없음)
            canvas = Image.composite(elem_img, canvas, binary_mask)
            
        else:
            # [Standard Logic] Non-text elements or when overwrite is False
            # 투명도를 고려하여 자연스럽게 블렌딩 (Alpha Compositing)
            canvas.paste(elem_img, (0, 0), elem_img)
        
        # 3. 전체 마스크 업데이트 (기존 로직 유지)
        elem_mask = elem["mask"]
        if elem_mask.shape != (H, W):
            elem_mask = cv2.resize(elem_mask, (W, H), interpolation=cv2.INTER_LINEAR)
        
        total_mask = total_mask + elem_mask * (1.0 - total_mask)

    return canvas, total_mask


def compute_composite_metrics(
    gt_elements: List[Dict],
    pred_elements: List[Dict],
    canvas_size: Tuple[int, int],
    metric_models: Optional[MetricModels] = None,
    gt_recon_img: Optional[Image.Image] = None,
    method_name: str = ""
) -> Dict[str, float]:
    W, H = canvas_size
    
    # =========================================================
    # 1. GT RGBA 준비 (H, W, 4)
    # =========================================================
    if gt_recon_img is not None:
        # Figma 원본이 있는 경우
        gt_rgba = np.array(gt_recon_img.convert("RGBA")).astype(np.float32) / 255.0
        # GT Mask는 Alpha 채널 활용
        gt_mask = gt_rgba[..., 3]
    else:
        # 재합성이 필요한 경우 (GT는 보통 overwrite=False)
        gt_rgba, gt_mask = composite_elements(gt_elements, canvas_size, use_overwrite=False)
    
    # =========================================================
    # 2. Pred RGBA 준비 (H, W, 4)
    # =========================================================
    # method_name이 'agent'일 때만 overwrite 모드 적용
    is_agent = (method_name == "agent")
    pred_rgba, pred_mask = composite_elements(pred_elements, canvas_size, use_overwrite=is_agent)
    
    # =========================================================
    # 3. Alpha Blending Simulation (Black vs White)
    # =========================================================
    # Alpha 채널 추출
    gt_alpha = gt_rgba[..., 3:4]      # (H, W, 1)
    pred_alpha = pred_rgba[..., 3:4]  # (H, W, 1)
    
    # Pre-multiply RGB (RGB * Alpha)
    gt_prem = gt_rgba[..., :3] * gt_alpha
    pred_prem = pred_rgba[..., :3] * pred_alpha
    
    # --- Case A: Black Background (0) ---
    # Formula: Color * Alpha + 0 * (1 - Alpha) = Pre-multiplied 그 자체
    gt_rgb_black = gt_prem
    pred_rgb_black = pred_prem
    
    # --- Case B: White Background (1) ---
    # Formula: Color * Alpha + 1 * (1 - Alpha)
    gt_rgb_white = gt_prem + (1.0 - gt_alpha)
    pred_rgb_white = pred_prem + (1.0 - pred_alpha)
    
    # =========================================================
    # 4. Compute Metrics (Min-Error Strategy)
    # =========================================================
    
    # Helper: L1, L2, PSNR 계산 함수
    def calc_pixel_metrics(gt, pred):
        l1 = float(np.mean(np.abs(gt - pred)))
        l2 = float(mean_squared_error(gt, pred))
        psnr = float(peak_signal_noise_ratio(gt, pred, data_range=1.0))
        return l1, l2, psnr

    # 검정 배경 점수
    l1_b, l2_b, psnr_b = calc_pixel_metrics(gt_rgb_black, pred_rgb_black)
    # 흰색 배경 점수
    l1_w, l2_w, psnr_w = calc_pixel_metrics(gt_rgb_white, pred_rgb_white)
    
    # [핵심] 두 배경 중 더 오차가 적은(점수가 높은) 쪽을 선택
    final_l1 = min(l1_b, l1_w)
    final_l2 = min(l2_b, l2_w)
    # PSNR이 inf인 경우 처리 (보통 numpy에선 inf로 나옴)
    final_psnr = max(psnr_b, psnr_w)
    if np.isinf(final_psnr):
        final_psnr = float('inf')

    # =========================================================
    # 5. Advanced Metrics (SSIM, LPIPS, DINO)
    # =========================================================
    # SSIM 등은 계산 비용 문제로 보통 'Standard Composite(Black)'을 기준으로 하거나,
    # 위에서 선택된 'Best Background' 이미지를 사용합니다.
    # 여기서는 L1이 더 낮았던 쪽의 이미지를 사용하여 공정성을 맞춥니다.
    
    if l1_b < l1_w:
        target_gt_rgb = gt_rgb_black
        target_pred_rgb = pred_rgb_black
    else:
        target_gt_rgb = gt_rgb_white
        target_pred_rgb = pred_rgb_white
        
    # SSIM
    try:
        win_size = min(H, W, 11)
        if win_size % 2 == 0: win_size -= 1
        ssim_val = float(ssim_func(
            target_gt_rgb, target_pred_rgb, 
            data_range=1.0, channel_axis=2, 
            win_size=win_size, gaussian_weights=True, sigma=1.5
        ))
    except Exception:
        ssim_val = 0.0
    
    # IoU
    iou = compute_iou(gt_mask, pred_mask)
    
    # Deep Learning Metrics
    lpips_val = 0.0
    dino_val = 0.0
    if metric_models:
        # 마찬가지로 선택된 'Best Background' 이미지로 계산
        lpips_val = metric_models.compute_lpips(target_gt_rgb, target_pred_rgb)
        dino_val = metric_models.compute_dino(target_gt_rgb, target_pred_rgb)
    
    return {
        "l1": final_l1, 
        "l2": final_l2, 
        "psnr": final_psnr, 
        "ssim": ssim_val,
        "iou": iou, 
        "lpips": lpips_val, 
        "dino": dino_val,
    }

def extract_background_l1(matched_pairs: List[Dict]) -> float:
    bg_l1_values = []
    
    for mp in matched_pairs:
        gt_type = mp["gt"].get("type", "")
        if gt_type == "background":
            l1 = mp.get("metrics", {}).get("l1", 0.0)
            bg_l1_values.append(l1)
    
    return float(np.mean(bg_l1_values)) if bg_l1_values else 0.0


# =============================================================================
# Visualization
# =============================================================================

def create_checkerboard(size: Tuple[int, int], square_size: int = 10) -> Image.Image:
    W, H = size
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(0, H, square_size):
        for x in range(0, W, square_size):
            color = (200, 200, 200) if (x // square_size + y // square_size) % 2 == 0 else (255, 255, 255)
            draw.rectangle([x, y, x + square_size, y + square_size], fill=color)
    return img


def visualize_element_on_checker(elem: Dict, thumb_size: Tuple[int, int] = (200, 150), padding: int = 10) -> Image.Image:
    img = elem["image"].convert("RGBA")
    bbox = elem.get("bbox", [0, 0, img.width, img.height])
    
    x1, y1, x2, y2 = bbox
    W, H = img.size
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(W, x2 + padding)
    y2 = min(H, y2 + padding)
    
    if x2 <= x1 or y2 <= y1:
        cropped = img
    else:
        cropped = img.crop((x1, y1, x2, y2))
    
    cropped.thumbnail(thumb_size, Image.LANCZOS)
    
    checker = create_checkerboard(thumb_size)
    paste_x = (thumb_size[0] - cropped.width) // 2
    paste_y = (thumb_size[1] - cropped.height) // 2
    
    checker.paste(cropped, (paste_x, paste_y), cropped)
    return checker


def create_mask_comparison_visualization(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    bbox: List[int],
    size: Tuple[int, int],
    padding: int = 10
) -> Image.Image:
    if gt_mask.shape != pred_mask.shape:
        pred_mask = cv2.resize(pred_mask.astype(np.float32), 
                               (gt_mask.shape[1], gt_mask.shape[0]), 
                               interpolation=cv2.INTER_LINEAR)

    H_full, W_full = gt_mask.shape
    vis = np.zeros((H_full, W_full, 4), dtype=np.uint8)
    
    gt_bin = gt_mask > 0
    pred_bin = pred_mask > 0
    
    # [수정] 마스크 로직 명확화
    real_intersection = gt_bin & pred_bin       # 교집합 (Both)
    gt_only = gt_bin & ~pred_bin                # GT Only (차집합)
    pred_only = pred_bin & ~gt_bin              # Pred Only (차집합)
    
    # [수정] 색상 할당
    # 1. 교집합 -> 초록색 (Green)
    vis[real_intersection] = [0, 255, 0, 160]
    
    # 2. GT Only -> 빨간색 (Red) - 기존 코드에서 'intersection' 변수명이 가리키던 영역
    vis[gt_only] = [255, 0, 0, 160]
    
    # 3. Pred Only -> 파란색 (Blue)
    vis[pred_only] = [0, 0, 255, 160]
    
    vis_img = Image.fromarray(vis, "RGBA")
    
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(W_full, x2 + padding)
    y2 = min(H_full, y2 + padding)
    
    if x2 > x1 and y2 > y1:
        vis_img = vis_img.crop((x1, y1, x2, y2))
    
    vis_img.thumbnail(size, Image.LANCZOS)
    
    checker = create_checkerboard(size)
    paste_x = (size[0] - vis_img.width) // 2
    paste_y = (size[1] - vis_img.height) // 2
    
    checker.paste(vis_img, (paste_x, paste_y), vis_img)
    
    return checker


def create_matched_pair_visualization(
    match: Dict,
    canvas_size: Tuple[int, int],
    thumb_size: Tuple[int, int] = (250, 180),
) -> Image.Image:
    gt = match["gt"]
    preds = match["preds"]
    metrics = match.get("metrics", {})
    match_type = match.get("match_type", "unknown")
    
    pred_img, pred_mask = composite_elements_transparent(preds, canvas_size)
    
    rows = np.any(pred_mask > 0, axis=1)
    cols = np.any(pred_mask > 0, axis=0)
    if rows.any() and cols.any():
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        pred_bbox = [int(x1), int(y1), int(x2+1), int(y2+1)]
    else:
        pred_bbox = gt.get("bbox", [0, 0, 100, 100])
        
    pred_elem = {"image": pred_img, "mask": pred_mask, "bbox": pred_bbox}

    gt_mask = gt["mask"]
    
    gt_thumb = visualize_element_on_checker(gt, thumb_size)
    pred_thumb = visualize_element_on_checker(pred_elem, thumb_size)
    
    intersect_vis = create_mask_comparison_visualization(gt_mask, pred_mask, gt.get('bbox', [0,0,100,100]), thumb_size)
    
    padding = 10
    text_height = 80
    card_width = thumb_size[0] * 3 + padding * 4
    card_height = thumb_size[1] + text_height + padding * 2
    
    card = Image.new("RGB", (card_width, card_height), (255, 255, 255))
    draw = ImageDraw.Draw(card)
    
    y_offset = text_height
    card.paste(gt_thumb, (padding, y_offset))
    card.paste(pred_thumb, (padding * 2 + thumb_size[0], y_offset))
    card.paste(intersect_vis, (padding * 3 + thumb_size[0] * 2, y_offset))
    
    gt_count = len(match.get("gt_indices", [1]))
    pred_count = len(match.get("pred_indices", [1]))
    
    draw.text((padding, 5), f"GT: {gt_count} elem(s)", fill=(0, 0, 0))
    draw.text((padding * 2 + thumb_size[0], 5), f"Pred: {pred_count} elem(s)", fill=(0, 0, 0))
    draw.text((padding * 3 + thumb_size[0] * 2, 5), "Intersection", fill=(0, 0, 0))
    
    metrics_text = (
        f"Type: {match_type} | "
        f"Cost: {match.get('cost', 0):.4f} | "
        f"L1: {metrics.get('l1', 0):.4f} | "
        f"IoU: {metrics.get('iou', 0):.4f}"
    )
    draw.text((padding, 25), metrics_text, fill=(0, 0, 128))
    
    legend_y = 45
    draw.rectangle([padding, legend_y, padding + 12, legend_y + 12], fill=(0, 255, 0))
    draw.text((padding + 15, legend_y), "Both", fill=(0, 100, 0))
    draw.rectangle([padding + 60, legend_y, padding + 72, legend_y + 12], fill=(255, 0, 0))
    draw.text((padding + 75, legend_y), "GT only", fill=(100, 0, 0))
    draw.rectangle([padding + 140, legend_y, padding + 152, legend_y + 12], fill=(0, 0, 255))
    draw.text((padding + 155, legend_y), "Pred only", fill=(0, 0, 100))
    
    return card


def create_unmatched_visualization(
    elem: Dict,
    elem_type: str,
    thumb_size: Tuple[int, int] = (250, 180),
) -> Image.Image:
    thumb = visualize_element_on_checker(elem, thumb_size)
    
    padding = 10
    text_height = 40
    card_width = thumb_size[0] + padding * 2
    card_height = thumb_size[1] + text_height + padding * 2
    
    card = Image.new("RGB", (card_width, card_height), (255, 255, 255))
    draw = ImageDraw.Draw(card)
    
    card.paste(thumb, (padding, text_height))
    
    color = (200, 0, 0) if elem_type == "FN" else (0, 0, 200)
    label = f"[{elem_type}] {elem['id'][:25]}"
    draw.text((padding, 5), label, fill=color)
    draw.text((padding, 22), f"Area: {elem.get('visible_area', elem['area']):.0f}", fill=(100, 100, 100))
    
    return card


def save_episode_visualization(
    episode_id: str,
    matched_pairs: List[Dict],
    unmatched_gt: List[Dict],
    unmatched_pred: List[Dict],
    gt_elements: List[Dict],
    pred_elements: List[Dict],
    canvas_size: Tuple[int, int],
    composite_metrics: Dict,
    pq_metrics: Dict,
    output_dir: Path,
    method_name: str,
    # [수정] Dual Metrics를 인자로 받도록 추가
    element_metrics_dual: Optional[Dict] = None,
    panoptic_quality_dual: Optional[Dict] = None,
    gt_recon_img: Optional[Image.Image] = None,
):
    """
    Save visualization images and metrics.json including Dual Metrics.
    """
    method_dir = output_dir / episode_id / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Visualization Images 저장 (기존 로직 유지)
    matched_cards = []
    for i, match in enumerate(matched_pairs):
        card = create_matched_pair_visualization(match, canvas_size)
        gt_id = match['gt']['id'][:20] if 'id' in match['gt'] else f"merged_{i}"
        # 파일명에 특수문자 제거 등 안전장치 추가 가능
        safe_gt_id = "".join(x for x in gt_id if x.isalnum() or x in "_-")
        card.save(method_dir / f"matched_{i:02d}_{safe_gt_id}.png")
        matched_cards.append(card)
    
    for i, elem in enumerate(unmatched_gt):
        card = create_unmatched_visualization(elem, "FN")
        safe_id = "".join(x for x in elem['id'][:20] if x.isalnum() or x in "_-")
        card.save(method_dir / f"fn_{i:02d}_{safe_id}.png")
    
    for i, elem in enumerate(unmatched_pred):
        card = create_unmatched_visualization(elem, "FP")
        safe_id = "".join(x for x in elem['id'][:20] if x.isalnum() or x in "_-")
        card.save(method_dir / f"fp_{i:02d}_{safe_id}.png")
    
    if gt_recon_img:
        bg = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
        bg.paste(gt_recon_img, (0, 0), gt_recon_img)
        gt_composite_img = bg
        gt_composite_img.save(method_dir / "composite_gt.png")
    
    pred_rgb, pred_mask = composite_elements(pred_elements, canvas_size)
    pred_composite_img = Image.fromarray((pred_rgb * 255).astype(np.uint8))
    pred_composite_img.save(method_dir / "composite_pred.png")
    
    # 2. JSON 데이터 구축 (Dual Metrics 포함)
    
    # 각 매칭 쌍에 대한 상세 메트릭 정리
    matched_pairs_details = []
    for mp in matched_pairs:
        detail = {
            "gt_id": mp["gt"]["id"],
            "gt_indices": mp.get("gt_indices", []),
            "pred_ids": [p["id"] for p in mp["preds"]],
            "pred_indices": mp.get("pred_indices", []),
            "match_type": mp["match_type"],
            "cost": mp.get("cost", 0),
            # 기존 레거시 메트릭
            "metrics": mp.get("metrics", {}),
            # [수정] Dual Metrics가 계산되어 있다면 포함
            "metrics_dual": mp.get("metrics_dual", {})
        }
        matched_pairs_details.append(detail)
    
    # 전체 메트릭 데이터 구조
    metrics_data = {
        "episode_id": episode_id,
        "method": method_name,
        "counts": {
            "gt_elements": len(gt_elements),
            "pred_elements": len(pred_elements),
            "matched_pairs": len(matched_pairs),
            "fn": len(unmatched_gt),
            "fp": len(unmatched_pred),
        },
        
        # Composite 레벨 (LPIPS, DINO 등)
        "composite_metrics": composite_metrics,
        
        # Legacy 호환용 (Binary IoU 기반)
        "panoptic_quality": pq_metrics, 
        
        # [수정] Dual Metrics (Soft/Binary IoU, No-Alpha-Composite VQ)
        # 인자로 전달받은 값을 저장 (없으면 빈 딕셔너리)
        "element_metrics_dual": element_metrics_dual if element_metrics_dual else {},
        "panoptic_quality_dual": panoptic_quality_dual if panoptic_quality_dual else {},
        
        # 상세 매칭 리스트
        "matched_pairs": matched_pairs_details
    }
    
    # 3. JSON 파일 저장
    with open(method_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, default=_json_safe_default)


# =============================================================================
# Episode Evaluation Function (for worker)
# =============================================================================

def evaluate_episode(
    episode_id: str,
    gt_elements: List[Dict],
    pred_elements: List[Dict],
    canvas_size: Tuple[int, int],
    method_name: str,
    output_dir: Path,
    metric_models = None,
    save_visualization: bool = True,
    gt_recon_img = None,
    use_optimal_matching: bool = True,
    logger = None,
) -> Dict[str, Any]:
    """Evaluate a single episode with DUAL METRICS."""
    
    # 1. 매칭 수행 (기존 로직)
    matched_pairs, unmatched_gt, unmatched_pred = match_elements_optimal(
        gt_elements, pred_elements, canvas_size,
        apply_pred_alpha_cleaning=True,
        verbose=True,
        logger=logger,
    )

    # =========================================================================
    # [NEW] 각 매칭 페어에 대해 Dual 메트릭 계산
    # =========================================================================
    for match in matched_pairs:
        match["metrics_dual"] = compute_element_metrics_dual(
            match["gt"], match["preds"], canvas_size
        )
        
        # 기존 metrics 필드 유지 (하위 호환성 - Union Region + Binary IoU)
        md = match["metrics_dual"]
        match["metrics"] = {
            "l1": md["visual_quality"]["union_region"]["l1"],
            "l2": md["visual_quality"]["union_region"]["l2"],
            "psnr": md["visual_quality"]["union_region"]["psnr"],
            "iou": md["iou"]["binary"],
        }

    # =========================================================================
    # [NEW] Dual PQ 계산
    # =========================================================================
    pq_dual = compute_panoptic_quality_dual(matched_pairs, unmatched_gt, unmatched_pred)
    
    # 기존 PQ 필드 (하위 호환성 - Binary IoU 기반)
    pq_metrics = {
        "pq_count": pq_dual["binary"]["pq"],
        "sq": pq_dual["binary"]["sq"],
        "rq": pq_dual["binary"]["rq"],
        "tp": pq_dual["binary"]["tp"],
        "fn": pq_dual["binary"]["fn"],
        "fp": pq_dual["binary"]["fp"],
    }
    
    # =========================================================================
    # [NEW] Dual Element 집계
    # =========================================================================
    element_agg_dual = aggregate_element_metrics_dual(matched_pairs)
    
    # 기존 element_metrics (하위 호환성)
    element_agg = {
        "simple_avg": {
            "l1": element_agg_dual["visual_quality"]["union_region"]["simple_avg"]["l1"],
            "l2": element_agg_dual["visual_quality"]["union_region"]["simple_avg"]["l2"],
            "psnr": element_agg_dual["visual_quality"]["union_region"]["simple_avg"]["psnr"],
            "iou": element_agg_dual["iou"]["simple_avg"]["binary"],
        },
        "weighted_avg": {
            "l1": element_agg_dual["visual_quality"]["union_region"]["weighted_avg"]["l1"],
            "l2": element_agg_dual["visual_quality"]["union_region"]["weighted_avg"]["l2"],
            "psnr": element_agg_dual["visual_quality"]["union_region"]["weighted_avg"]["psnr"],
            "iou": element_agg_dual["iou"]["weighted_avg"]["binary"],
        },
    }
    
    # Composite 메트릭: non-text object가 5개 이하이면 skip (파싱이 충분하지 않아 composite가 비정상적으로 높게 나옴)
    n_non_text = sum(1 for e in pred_elements if e.get("type") != "text")
    if n_non_text <= 5:
        composite_metrics = None
    else:
        composite_metrics = compute_composite_metrics(
            gt_elements, pred_elements, canvas_size, metric_models,
            gt_recon_img=gt_recon_img,
            method_name=method_name
        )
    
    background_l1 = extract_background_l1(matched_pairs)
    
    # =========================================================================
    # [NEW] Enhanced Logging
    # =========================================================================
    if logger:
        logger.info(f"[{method_name.upper()}] Episode: {episode_id}")
        logger.info(f"  Matched: {len(matched_pairs)}, FN: {len(unmatched_gt)}, FP: {len(unmatched_pred)}")
        logger.info("")
        
        vq = element_agg_dual["visual_quality"]
        inter_r = vq["intersection_region"]["simple_avg"]
        union_r = vq["union_region"]["simple_avg"]
        iou_agg = element_agg_dual["iou"]["simple_avg"]
        
        logger.info("  [Visual Quality - No Alpha Composite]")
        logger.info(f"  {'Metric':<8} {'Intersection Region':<20} {'Union Region':<20} {'GT Region (LayerD)':<20}")
        logger.info(f"  {'-'*68}")
        logger.info(f"  {'L1':<8} {inter_r['l1']:<20.4f} {union_r['l1']:<20.4f}")
        logger.info(f"  {'L2':<8} {inter_r['l2']:<20.4f} {union_r['l2']:<20.4f}")
        logger.info(f"  {'PSNR':<8} {inter_r['psnr']:<20.2f} {union_r['psnr']:<20.2f}")
        logger.info("")
        logger.info("  [Layout]")
        logger.info(f"  {'Metric':<8} {'Soft IoU':<20} {'Binary IoU':<20}")
        logger.info(f"  {'-'*48}")
        logger.info(f"  {'IoU':<8} {iou_agg['soft']:<20.4f} {iou_agg['binary']:<20.4f}")
        logger.info(f"  {'PQ':<8} {pq_dual['soft']['pq']:<20.4f} {pq_dual['binary']['pq']:<20.4f}")
        logger.info(f"  {'SQ':<8} {pq_dual['soft']['sq']:<20.4f} {pq_dual['binary']['sq']:<20.4f}")
        logger.info(f"  {'RQ':<8} {pq_dual['soft']['rq']:<20.4f} {pq_dual['binary']['rq']:<20.4f}")
        logger.info("")
        if composite_metrics is not None:
            logger.info(f"  Composite L1: {composite_metrics['l1']:.4f}, PSNR: {composite_metrics['psnr']:.2f}, SSIM: {composite_metrics['ssim']:.4f}, LPIPS: {composite_metrics['lpips']:.4f}, DINO: {composite_metrics['dino']:.4f}")
        else:
            logger.info(f"  Composite: SKIPPED (non_text_objects={n_non_text}, <=5)")

    # Visualization 저장 (기존 로직)
    if save_visualization:
         save_episode_visualization(
            episode_id, matched_pairs, unmatched_gt, unmatched_pred,
            gt_elements, pred_elements, canvas_size,
            composite_metrics, pq_metrics,
            output_dir, method_name,
            # [수정] 아래 인자들을 추가하여 계산된 Dual Metrics를 전달합니다.
            element_metrics_dual=element_agg_dual,
            panoptic_quality_dual=pq_dual,
            gt_recon_img=gt_recon_img
        )
    
    # =========================================================================
    # 결과 반환 - Dual 메트릭 포함
    # =========================================================================
    return {
        "episode_id": episode_id,
        "method": method_name,
        "matching_type": "optimal" if use_optimal_matching else "legacy",
        "counts": {
            "gt_elements": len(gt_elements),
            "pred_elements": len(pred_elements),
            "pred_non_text": n_non_text,
            "matched_pairs": len(matched_pairs),
            "fn": len(unmatched_gt),
            "fp": len(unmatched_pred),
            "composite_skipped": composite_metrics is None,
        },
        # 기존 필드 (하위 호환성)
        "element_metrics": element_agg,
        "panoptic_quality": pq_metrics,
        "composite_metrics": composite_metrics,
        "background_l1": background_l1,
        # [NEW] Dual 메트릭
        "element_metrics_dual": element_agg_dual,
        "panoptic_quality_dual": pq_dual,
    }





def filter_episodes_by_background_l1(
    qwen_results: List[Dict],
    agent_results: List[Dict],
    threshold: float = BACKGROUND_L1_THRESHOLD
) -> Tuple[Set[str], Set[str]]:
    qwen_bg_l1 = {r["episode_id"]: r.get("background_l1", 0.0) for r in qwen_results}
    agent_bg_l1 = {r["episode_id"]: r.get("background_l1", 0.0) for r in agent_results}
    
    all_episode_ids = set(qwen_bg_l1.keys()) | set(agent_bg_l1.keys())
    
    excluded_ids = set()
    for ep_id in all_episode_ids:
        qwen_l1 = qwen_bg_l1.get(ep_id, 0.0)
        agent_l1 = agent_bg_l1.get(ep_id, 0.0)
        if qwen_l1 >= threshold or agent_l1 >= threshold:
            excluded_ids.add(ep_id)
    
    included_ids = all_episode_ids - excluded_ids
    return excluded_ids, included_ids


# =============================================================================
# Worker Process Function
# =============================================================================

def worker_process(
    worker_id: int,
    gpu_id: int,
    tasks: List[Dict],
    args_dict: Dict,
    result_queue: Queue,
    log_queue: Queue,
    progress_queue: Queue,
):
    """
    Worker function that runs on a specific GPU.
    """
    global worker_logger
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Initialize worker logger
    output_dir = Path(args_dict["output"])
    log_file_path = output_dir / f"worker_{worker_id}_gpu{gpu_id}.log"
    worker_logger = WorkerLogger(worker_id, log_queue, log_file_path)
    
    worker_logger.info(f"Started on GPU {gpu_id} with {len(tasks)} tasks")
    worker_logger.info(f"Matching algorithm: {args_dict['matching']}")
    
    # Check library availability
    if not PULP_AVAILABLE and args_dict['matching'] == 'optimal':
        worker_logger.warn("PuLP not available, falling back to greedy matching")
    
    # Initialize metric models
    device = f"cuda:0"  # Local GPU 0 (due to CUDA_VISIBLE_DEVICES)
    worker_logger.info(f"Loading metric models...")
    start_time = time.time()
    metric_models = MetricModels(device, logger=worker_logger)
    worker_logger.info(f"Metric models loaded in {time.time() - start_time:.2f}s")
    
    use_optimal = (args_dict['matching'] == 'optimal')
    crello_subset_dir = Path(args_dict['crello_subset'])
    qwen_exp_dir = Path(args_dict['qwen_exp'])
    agent_exp_dir = Path(args_dict['agent_exp'])
    
    results = {"qwen": [], "agent": []}
    
    for idx, task in enumerate(tasks):
        episode_id = task["episode_id"]
        split_name = task["split"]
        record_dir = task["record_dir"]
        
        # Crello: Qwen is flat (no split subdir), Agent has split subdir
        qwen_dir = qwen_exp_dir / episode_id
        agent_dir = agent_exp_dir / split_name / "episodes" / episode_id
        
        # Send progress update
        progress_queue.put({
            "worker_id": worker_id,
            "current": idx + 1,
            "total": len(tasks),
            "episode_id": episode_id,
            "status": "processing"
        })
        
        worker_logger.progress(idx + 1, len(tasks), episode_id, f"split={split_name}")
        
        try:
            # Extract GT elements (Crello: from record_dir, no split_dir needed)
            worker_logger.info(f"[Episode {episode_id}] === START ===")
            worker_logger.info(f"[Episode {episode_id}] Extracting GT elements...")
            gt_start = time.time()
            gt_elements, canvas_size, gt_recon_img = extract_gt_elements(
                record_dir, logger=worker_logger
            )
            worker_logger.info(f"[Episode {episode_id}] GT extraction: {len(gt_elements)} elements in {time.time() - gt_start:.2f}s, canvas={canvas_size}")
            
            if not gt_elements:
                worker_logger.warn(f"[Episode {episode_id}] No GT elements found, skipping")
                continue
            
            # Qwen Evaluation
            worker_logger.info(f"[Episode {episode_id}] Extracting Qwen elements...")
            qwen_start = time.time()
            qwen_elements = extract_qwen_elements_cca(qwen_dir, canvas_size, logger=worker_logger)
            worker_logger.info(f"[Episode {episode_id}] Qwen extraction: {len(qwen_elements)} elements in {time.time() - qwen_start:.2f}s")
            
            if qwen_elements:
                worker_logger.info(f"[Episode {episode_id}] Starting QWEN evaluation...")
                eval_start = time.time()
                qwen_res = evaluate_episode(
                    episode_id, gt_elements, qwen_elements, canvas_size,
                    "qwen", output_dir, metric_models,
                    save_visualization=not args_dict['no_viz'],
                    gt_recon_img=gt_recon_img,
                    use_optimal_matching=use_optimal,
                    logger=worker_logger,
                )
                results["qwen"].append(qwen_res)
                worker_logger.info(f"[Episode {episode_id}] QWEN evaluation completed in {time.time() - eval_start:.2f}s | "
                                  f"Matched: {qwen_res['counts']['matched_pairs']}, FN: {qwen_res['counts']['fn']}, FP: {qwen_res['counts']['fp']}")
            else:
                worker_logger.warn(f"[Episode {episode_id}] No Qwen elements found")
            
            # Agent Evaluation
            worker_logger.info(f"[Episode {episode_id}] Extracting Agent elements...")
            agent_start = time.time()
            agent_elements = extract_agent_elements(agent_dir, canvas_size, apply_alpha_correction=True, text_refinement=True, logger=worker_logger)
            worker_logger.info(f"[Episode {episode_id}] Agent extraction: {len(agent_elements)} elements in {time.time() - agent_start:.2f}s")
            
            if agent_elements:
                worker_logger.info(f"[Episode {episode_id}] Starting AGENT evaluation...")
                eval_start = time.time()
                agent_res = evaluate_episode(
                    episode_id, gt_elements, agent_elements, canvas_size,
                    "agent", output_dir, metric_models,
                    save_visualization=not args_dict['no_viz'],
                    gt_recon_img=gt_recon_img,
                    use_optimal_matching=use_optimal,
                    logger=worker_logger,
                )
                results["agent"].append(agent_res)
                worker_logger.info(f"[Episode {episode_id}] AGENT evaluation completed in {time.time() - eval_start:.2f}s | "
                                  f"Matched: {agent_res['counts']['matched_pairs']}, FN: {agent_res['counts']['fn']}, FP: {agent_res['counts']['fp']}")
            else:
                worker_logger.warn(f"[Episode {episode_id}] No Agent elements found")
            
            progress_queue.put({
                "worker_id": worker_id,
                "current": idx + 1,
                "total": len(tasks),
                "episode_id": episode_id,
                "status": "done_frame",  # 상태를 '처리중'에서 '완료'로 변경
                "qwen_res": qwen_res,    # Qwen 결과 데이터 포함
                "agent_res": agent_res   # Agent 결과 데이터 포함
            })


            worker_logger.info(f"[Episode {episode_id}] === END ===")
                
        except Exception as e:
            worker_logger.error(f"[Episode {episode_id}] Error: {e}")
            import traceback
            worker_logger.error(traceback.format_exc())
            continue
    
    # Send completion signal
    progress_queue.put({
        "worker_id": worker_id,
        "current": len(tasks),
        "total": len(tasks),
        "episode_id": "DONE",
        "status": "completed"
    })
    
    worker_logger.info(f"Completed! Processed {len(results['qwen'])} qwen, {len(results['agent'])} agent episodes")
    
    # Put results in queue
    result_queue.put({
        "worker_id": worker_id,
        "results": results,
    })
    
    worker_logger.close()


def log_listener_process(log_queue: Queue, stop_event):
    """상세 로그는 파일에만 기록되므로, CMD 출력은 무시합니다."""
    while not stop_event.is_set() or not log_queue.empty():
        try:
            message = log_queue.get(timeout=0.1)
            # tqdm.write(message)  <-- 이 부분을 주석 처리하여 CMD 출력을 끕니다.
        except Empty:
            continue
        except Exception:
            continue


import textwrap # 상단에 import 추가 (긴 ID 리스트 줄바꿈용)

def progress_monitor_process(progress_queue, stop_event, total_tasks, num_workers):
    import math
    pbar = tqdm(total=total_tasks, unit="frame", dynamic_ncols=True, position=0, leave=True)
    
    def init_stats():
        return {
            # Visual Quality: list 기반 수집 (NaN/inf 안전 집계용)
            "vq_inter": {"l1": [], "l2": [], "psnr": []},
            "vq_union": {"l1": [], "l2": [], "psnr": []},
            "vq_gt":    {"l1": [], "l2": [], "psnr": []},
            "lay_soft": {"iou": 0, "pq": 0, "sq": 0, "rq": 0},
            "lay_bin":  {"iou": 0, "pq": 0, "sq": 0, "rq": 0},
            "comp":     {"l1": [], "psnr": [], "ssim": 0, "lpips": 0, "dino": 0},
            "comp_count": 0,
            "comp_skipped": 0,
        }
    
    def _safe_mean(vals, is_psnr=False):
        """NaN 제거 후 평균. PSNR이면 inf→max_finite 대체."""
        clean = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        if not clean:
            return 0.0
        if is_psnr:
            finite = [v for v in clean if not (isinstance(v, float) and math.isinf(v))]
            if finite:
                mx = max(finite)
                clean = [v if not (isinstance(v, float) and math.isinf(v)) else mx for v in clean]
            else:
                return 0.0
        return sum(clean) / len(clean)
    
    stats = {"qwen": init_stats(), "agent": init_stats()}
    counters = {"processed": 0, "valid": 0, "skipped": 0}
    skipped_ids_list = []

    while not stop_event.is_set() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=0.1)
            if update.get("status") == "done_frame":
                counters["processed"] += 1
                episode_id = update.get("episode_id", "Unknown")
                
                # Agent Composite L1 > 0.2 필터링
                agent_res = update.get("agent_res")
                if agent_res and agent_res.get("composite_metrics", {}).get("l1", 0) > 0.2:
                    counters["skipped"] += 1
                    skipped_ids_list.append(episode_id)
                    pbar.update(1)
                    continue 
                
                counters["valid"] += 1
                for m_key in ["qwen", "agent"]:
                    res = update.get(f"{m_key}_res")
                    if not res: continue
                    s = stats[m_key]
                    
                    # 1. Visual Quality (Inter/Union/GT) - L1, L2, PSNR → list append
                    vq_dual = res["element_metrics_dual"]["visual_quality"]
                    for reg, target in [("intersection_region", "vq_inter"), ("union_region", "vq_union")]:
                        for met in ["l1", "l2", "psnr"]:
                            s[target][met].append(vq_dual[reg]["simple_avg"][met])
                    
                    # 2. Layout (Soft/Binary) - IoU, PQ, SQ, RQ
                    iou_dual = res["element_metrics_dual"]["iou"]["simple_avg"]
                    pq_dual = res["panoptic_quality_dual"]
                    for style, target in [("soft", "lay_soft"), ("binary", "lay_bin")]:
                        s[target]["iou"] += iou_dual[style]
                        for met in ["pq", "sq", "rq"]:
                            s[target][met] += pq_dual[style][met]
                    
                    # 3. Composite Metrics (skip if composite_skipped or non-text <= 5)
                    if res.get("counts", {}).get("composite_skipped", False):
                        s["comp_skipped"] += 1
                    else:
                        comp = res.get("composite_metrics") or {}
                        if comp:
                            s["comp_count"] += 1
                            for met in ["l1", "psnr"]:
                                val = comp.get(met)
                                if val is not None:
                                    s["comp"][met].append(val)
                            for met in ["ssim", "lpips", "dino"]:
                                val = comp.get(met)
                                if val is not None:
                                    s["comp"][met] += val

                pbar.update(1)

                # [표 출력 로직]
                v = counters["valid"]
                if v > 0:
                    q = stats["qwen"]; a = stats["agent"]
                    s_list = textwrap.fill(", ".join(skipped_ids_list), width=100, initial_indent="    ", subsequent_indent="    ")

                    # Visual quality: list → _safe_mean
                    def _vq(s, key, met):
                        return _safe_mean(s[key][met], is_psnr=(met == "psnr"))
                    # Composite: list for l1/psnr, sum/comp_count for others
                    def _comp(s, met):
                        cv = s["comp_count"]
                        if cv == 0:
                            return 0.0
                        if met in ["l1", "psnr"]:
                            return _safe_mean(s["comp"][met], is_psnr=(met == "psnr"))
                        return s["comp"][met] / cv

                    table = [
                        "\n" + "="*110,
                        f" [CUMULATIVE SUMMARY]  Total: {counters['processed']} | Valid: {v} | Skipped: {counters['skipped']}",
                        "-"*110,
                        f" {'Category / Metric':<25} | {'QWEN (Average)':<40} | {'AGENT (Average)':<40}",
                        "-"*110,
                        f" [Visual - Intersection]",
                        f"  L1 / L2 / PSNR          | L1:{_vq(q,'vq_inter','l1'):.4f} L2:{_vq(q,'vq_inter','l2'):.4f} PSNR:{_vq(q,'vq_inter','psnr'):.2f} | L1:{_vq(a,'vq_inter','l1'):.4f} L2:{_vq(a,'vq_inter','l2'):.4f} PSNR:{_vq(a,'vq_inter','psnr'):.2f}",
                        f" [Visual - Union]",
                        f"  L1 / L2 / PSNR          | L1:{_vq(q,'vq_union','l1'):.4f} L2:{_vq(q,'vq_union','l2'):.4f} PSNR:{_vq(q,'vq_union','psnr'):.2f} | L1:{_vq(a,'vq_union','l1'):.4f} L2:{_vq(a,'vq_union','l2'):.4f} PSNR:{_vq(a,'vq_union','psnr'):.2f}",
                        f" [Visual - GT Region]",
                        f"  L1 / L2 / PSNR          | L1:{_vq(q,'vq_gt','l1'):.4f} L2:{_vq(q,'vq_gt','l2'):.4f} PSNR:{_vq(q,'vq_gt','psnr'):.2f} | L1:{_vq(a,'vq_gt','l1'):.4f} L2:{_vq(a,'vq_gt','l2'):.4f} PSNR:{_vq(a,'vq_gt','psnr'):.2f}",
                        f" [Layout - Soft]",
                        f"  IoU / PQ / SQ / RQ      | I:{q['lay_soft']['iou']/v:.4f} P:{q['lay_soft']['pq']/v:.4f} S:{q['lay_soft']['sq']/v:.4f} R:{q['lay_soft']['rq']/v:.4f} | I:{a['lay_soft']['iou']/v:.4f} P:{a['lay_soft']['pq']/v:.4f} S:{a['lay_soft']['sq']/v:.4f} R:{a['lay_soft']['rq']/v:.4f}",
                        f" [Layout - Binary]",
                        f"  IoU / PQ / SQ / RQ      | I:{q['lay_bin']['iou']/v:.4f} P:{q['lay_bin']['pq']/v:.4f} S:{q['lay_bin']['sq']/v:.4f} R:{q['lay_bin']['rq']/v:.4f} | I:{a['lay_bin']['iou']/v:.4f} P:{a['lay_bin']['pq']/v:.4f} S:{a['lay_bin']['sq']/v:.4f} R:{a['lay_bin']['rq']/v:.4f}",
                        f" [Composite]",
                        f"  L1 / PSNR / SSIM        | L1:{_comp(q,'l1'):.4f} PSNR:{_comp(q,'psnr'):.2f} SSIM:{_comp(q,'ssim'):.4f} | L1:{_comp(a,'l1'):.4f} PSNR:{_comp(a,'psnr'):.2f} SSIM:{_comp(a,'ssim'):.4f}",
                        f"  LPIPS / DINO            | LP:{_comp(q,'lpips'):.4f} DN:{_comp(q,'dino'):.4f}              | LP:{_comp(a,'lpips'):.4f} DN:{_comp(a,'dino'):.4f}",
                        "-"*110,
                        f" [SKIPPED EPISODES (L1 > 0.2)]",
                        s_list,
                        "="*110
                    ]
                    tqdm.write("\n".join(table))
                
        except Empty:
            continue
    pbar.close()

# =============================================================================
# Main Function with Parallel Processing
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel Element-Level Evaluation for Crello (Multi-GPU)")
    parser.add_argument("--crello-subset", type=str, required=True,
                       help="Path to crello_subset directory (with crello_test_*/gt_metadata.json)")
    parser.add_argument("--qwen-exp", type=str, required=True,
                       help="Path to qwen crello experiment (flat: {record_id}/layer_*.png)")
    parser.add_argument("--agent-exp", type=str, required=True,
                       help="Path to agent crello experiment (split_{0-4}/episodes/{record_id}/)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--agent-splits", type=str, nargs="+", default=["split_0", "split_1", "split_2", "split_3", "split_4"],
                       help="Agent split directories to scan (default: split_0~4)")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    parser.add_argument("--bg-l1-threshold", type=float, default=BACKGROUND_L1_THRESHOLD)
    parser.add_argument("--matching", type=str, default="optimal", choices=["optimal", "legacy"])
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers (GPUs)")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5,6,7", 
                       help="Comma-separated GPU IDs to use")
    parser.add_argument("--text-refinement-mode", type=str, default="hybrid",
                       choices=["kill", "correct", "hybrid"],
                       help="Text refinement mode: kill (기존, Union에 유리), "
                            "correct (RGB만 교정, GT Region에 유리), "
                            "hybrid (절충)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    num_workers = min(args.num_workers, len(gpu_ids))
    
    print("=" * 80)
    print(f"PARALLEL ELEMENT EVALUATION (CRELLO)")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Crello subset: {args.crello_subset}")
    print(f"Agent splits: {args.agent_splits}")
    print(f"Matching algorithm: {args.matching}")
    print(f"Number of workers: {num_workers}")
    print(f"GPU IDs: {gpu_ids[:num_workers]}")
    print("=" * 80)
    
    crello_subset_dir = Path(args.crello_subset)
    qwen_exp_dir = Path(args.qwen_exp)
    agent_exp_dir = Path(args.agent_exp)
    
    # ---- Step 1: Collect GT records from crello_subset ----
    gt_records = {}  # record_id → record_dir (Path)
    for record_dir in sorted(crello_subset_dir.iterdir()):
        if not record_dir.name.startswith("crello_test_"):
            continue
        if not (record_dir.is_dir() or record_dir.is_symlink()):
            continue
        gt_meta = record_dir / "gt_metadata.json"
        if gt_meta.exists():
            gt_records[record_dir.name] = record_dir
    
    print(f"GT records found: {len(gt_records)}")
    
    # ---- Step 2: Collect Qwen completed records (flat directory) ----
    qwen_episodes = set()
    if qwen_exp_dir.exists():
        for ep_dir in qwen_exp_dir.iterdir():
            if ep_dir.is_dir() and ep_dir.name.startswith("crello_test_"):
                if (ep_dir / "layer_00.png").exists():
                    qwen_episodes.add(ep_dir.name)
    print(f"Qwen completed: {len(qwen_episodes)}")
    
    # ---- Step 3: Collect Agent completed records (scan all splits) ----
    agent_record_to_split = {}  # record_id → split_name
    for split_name in args.agent_splits:
        agent_split_dir = agent_exp_dir / split_name / "episodes"
        if not agent_split_dir.exists():
            print(f"  [Warning] {agent_split_dir} not found, skipping")
            continue
        count = 0
        for ep_dir in agent_split_dir.iterdir():
            if ep_dir.is_dir() and ep_dir.name.startswith("crello_test_"):
                if (ep_dir / "parse.json").exists():
                    agent_record_to_split[ep_dir.name] = split_name
                    count += 1
        print(f"  Agent {split_name}: {count} completed")
    
    agent_episodes = set(agent_record_to_split.keys())
    print(f"Agent completed (total): {len(agent_episodes)}")
    
    # ---- Step 4: Find common IDs (GT ∩ Qwen ∩ Agent) ----
    common_ids = set(gt_records.keys()) & qwen_episodes & agent_episodes
    print(f"Common episodes (GT ∩ Qwen ∩ Agent): {len(common_ids)}")
    
    # ---- Step 5: Build tasks with resume support ----
    all_valid_episode_tasks = []
    skipped_count = 0
    
    for eid in sorted(common_ids):
        episode_out_dir = output_dir / eid
        agent_done = (episode_out_dir / "agent" / "metrics.json").exists()
        qwen_done = (episode_out_dir / "qwen" / "metrics.json").exists()
        
        if agent_done and qwen_done:
            skipped_count += 1
            continue
        
        all_valid_episode_tasks.append({
            "episode_id": eid,
            "split": agent_record_to_split[eid],  # which agent split this record is in
            "record_dir": gt_records[eid],          # crello_subset/crello_test_XXXX/
        })
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} already processed episodes.")

    if args.max_episodes:
        all_valid_episode_tasks = all_valid_episode_tasks[:args.max_episodes]
        
    print(f"\nTotal episodes to evaluate: {len(all_valid_episode_tasks)}")
    
    if not all_valid_episode_tasks:
        print("No episodes to evaluate!")
        return
    
    # Split tasks among workers
    tasks_per_worker = len(all_valid_episode_tasks) // num_workers
    worker_tasks = []
    
    for i in range(num_workers):
        start_idx = i * tasks_per_worker
        if i == num_workers - 1:
            # Last worker gets remaining tasks
            end_idx = len(all_valid_episode_tasks)
        else:
            end_idx = start_idx + tasks_per_worker
        worker_tasks.append(all_valid_episode_tasks[start_idx:end_idx])
    
    print(f"\nTask distribution:")
    for i, tasks in enumerate(worker_tasks):
        print(f"  Worker {i} (GPU {gpu_ids[i]}): {len(tasks)} tasks")
    
    # Create shared queues
    manager = Manager()
    result_queue = manager.Queue()
    log_queue = manager.Queue()
    progress_queue = manager.Queue()
    stop_event = manager.Event()
    
    # Prepare args dict for workers
    args_dict = {
        "crello_subset": args.crello_subset,
        "qwen_exp": args.qwen_exp,
        "agent_exp": args.agent_exp,
        "output": args.output,
        "no_viz": args.no_viz,
        "matching": args.matching,
    }
    
    # Start log listener process
    log_listener = Process(target=log_listener_process, args=(log_queue, stop_event))
    log_listener.start()
    
    # Start progress monitor process
    progress_monitor = Process(
        target=progress_monitor_process, 
        args=(progress_queue, stop_event, len(all_valid_episode_tasks), num_workers)
    )
    progress_monitor.start()
    
    print(f"\nStarting {num_workers} worker processes...")
    print("=" * 80)
    
    # Start worker processes
    processes = []
    start_time = time.time()
    
    for i in range(num_workers):
        p = Process(
            target=worker_process,
            args=(i, gpu_ids[i], worker_tasks[i], args_dict, result_queue, log_queue, progress_queue)
        )
        p.start()
        processes.append(p)
        print(f"  Worker {i} started (PID: {p.pid})")
    
    # Wait for all workers to complete
    for p in processes:
        p.join()
    
    # Stop log listener and progress monitor
    stop_event.set()
    log_listener.join(timeout=5)
    progress_monitor.join(timeout=5)
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"All workers completed in {elapsed_time:.2f} seconds")
    print("=" * 80)
    
    # Collect results from all workers
    all_results = {"qwen": [], "agent": []}
    
    while not result_queue.empty():
        worker_result = result_queue.get()
        worker_id = worker_result["worker_id"]
        results = worker_result["results"]
        
        print(f"Collected results from Worker {worker_id}: "
              f"{len(results['qwen'])} qwen, {len(results['agent'])} agent")
        
        all_results["qwen"].extend(results["qwen"])
        all_results["agent"].extend(results["agent"])
    
    print(f"\nTotal collected: {len(all_results['qwen'])} qwen, {len(all_results['agent'])} agent")
    
    # Aggregate and print results
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS (ALL EPISODES)")
    print("=" * 80)
    
    all_episodes_summary = {}
    
    for method in ["qwen", "agent"]:
        results = all_results[method]
        if not results:
            continue
        
        summary = aggregate_results(results)
        all_episodes_summary[method] = summary

        print_dual_comparison(summary, method)
        
        print(f"\n{method.upper()} ({len(results)} episodes)")
        print("-" * 40)
        
        s = summary["element_metrics"]["simple_avg"]
        print(f"Element Metrics (Simple Avg): L1={s.get('l1', 0):.4f}, IoU={s.get('iou', 0):.4f}")
        
        pq = summary["panoptic_quality"]
        print(f"Panoptic Quality: PQ={pq['pq']:.4f}, SQ={pq['sq']:.4f}, RQ={pq['rq']:.4f}")
        
        comp = summary["composite_metrics"]
        print(f"Composite: L1={comp['l1']:.4f}, PSNR={comp['psnr']:.2f}, SSIM={comp['ssim']:.4f}")
    
    # Filter by background L1
    print("\n" + "=" * 80)
    print(f"AGGREGATED RESULTS (FILTERED: Background L1 < {args.bg_l1_threshold})")
    print("=" * 80)
    
    excluded_ids, included_ids = filter_episodes_by_background_l1(
        all_results["qwen"], all_results["agent"], threshold=args.bg_l1_threshold
    )
    
    print(f"\nFiltering Summary:")
    print(f"  Total: {len(excluded_ids) + len(included_ids)}, Excluded: {len(excluded_ids)}, Included: {len(included_ids)}")
    
    filtered_summary = {}
    
    for method in ["qwen", "agent"]:
        results = all_results[method]
        filtered_results = [r for r in results if r["episode_id"] in included_ids]
        
        if not filtered_results:
            continue
        
        summary = aggregate_results(filtered_results)
        filtered_summary[method] = summary

        print_dual_comparison(summary, method)
        
        print(f"\n{method.upper()} ({len(filtered_results)} episodes)")
        s = summary["element_metrics"]["simple_avg"]
        print(f"Element Metrics: L1={s.get('l1', 0):.4f}, IoU={s.get('iou', 0):.4f}")
        pq = summary["panoptic_quality"]
        print(f"Panoptic Quality: PQ={pq['pq']:.4f}, SQ={pq['sq']:.4f}, RQ={pq['rq']:.4f}")
    
    # Save results
    unified_summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "agent_splits": args.agent_splits,
            "matching_algorithm": args.matching,
            "num_workers": num_workers,
            "gpu_ids": gpu_ids[:num_workers],
            "elapsed_time_seconds": elapsed_time,
            "config": {
                "merge_iou_threshold": MERGE_IOU_THRESHOLD,
                "lambda_l1": LAMBDA_L1,
                "lambda_iou": LAMBDA_IOU,
                "penalty_gt_merge": PENALTY_GT_MERGE,
                "penalty_pe_merge": PENALTY_PE_MERGE,
                "dummy_cost": DUMMY_COST,
                "max_merge_size": MAX_MERGE_SIZE,
            },
        },
        "filtering": {
            "threshold": args.bg_l1_threshold,
            "total_episodes": len(excluded_ids) + len(included_ids),
            "excluded_count": len(excluded_ids),
            "included_count": len(included_ids),
        },
        "results": {
            "all_episodes": all_episodes_summary,
            "filtered_episodes": filtered_summary,
        },
        "per_episode_details": {
            "qwen": [{
                "episode_id": r["episode_id"],
                "background_l1": r.get("background_l1", 0.0),
                "element_metrics": r["element_metrics"],
                "panoptic_quality": r["panoptic_quality"],
                "composite_metrics": r["composite_metrics"],
                "counts": r["counts"],
            } for r in all_results["qwen"]],
            "agent": [{
                "episode_id": r["episode_id"],
                "background_l1": r.get("background_l1", 0.0),
                "element_metrics": r["element_metrics"],
                "panoptic_quality": r["panoptic_quality"],
                "composite_metrics": r["composite_metrics"],
                "counts": r["counts"],
            } for r in all_results["agent"]],
        },
    }
    
    unified_path = output_dir / "evaluation_unified_summary.json"
    with open(unified_path, "w") as f:
        json.dump(unified_summary, f, indent=2, default=_json_safe_default)
    
    print(f"\nSaved unified summary to {unified_path}")
    print(f"\nEvaluation completed at {datetime.now().isoformat()}")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()