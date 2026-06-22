#!/usr/bin/env python3
"""
evaluation_figma.py

Element-Level Evaluation for Figma Dataset
Based on evaluation_crello.py evaluation logic (gt_region, text_refinement, etc.)
Adapted for Figma GT structure, directory paths, and task collection.
Canvas placement / coordinate transforms follow original Figma evaluation logic.

Supports union of multiple paired experiment directories.
Each pair specifies its own agent, qwen, and GT subset prefix.
Splits are auto-detected per pair.

Usage:
    # Replace <GPU_IDS> with a comma-separated list of your own GPU ids (e.g. 0,1,2,3).
    python evaluation_figma.py \
        --figma-data <FIGMA_DATA_DIR> \
        --exp-pairs \
            <AGENT_OUTPUT_DIR>:<QWEN_OUTPUT_DIR>:merged \
        --output <OUTPUT_DIR> \
        --matching optimal \
        --num-workers 8 --gpu-ids <GPU_IDS>

    The agent/qwen output dirs are produced by running the inference runners first
    (e.g. ``python -m REDESIGN.run_agent_figma --data_dir figma_data \
    --output_dir <AGENT_OUTPUT_DIR>``), and ``--figma-data`` should point at the
    downloaded ``figma_data`` dataset.
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
CONTAINMENT_MERGE_THRESHOLD = 0.5  # Always merge if containment is 0.75 or higher
OPTIONAL_MERGE_THRESHOLD = 0.1     # Candidate for combination generation if between 0.1 and 0.75
LAMBDA_L1 = 0.7
LAMBDA_IOU = 0.3
PENALTY_GT_MERGE = 0.05
PENALTY_PE_MERGE = 0.0
DUMMY_COST = 0.4
MAX_MERGE_SIZE = None  # No limit


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
    frame_json_path: Path,
    split_dir: Path,
    logger: Optional[WorkerLogger] = None
) -> Tuple[List[Dict], Tuple[int, int], Optional[Image.Image]]:
    """Extract GT elements from Figma data.
    
    Optimized: Uses NumPy slicing instead of PIL paste for faster canvas placement.
    """
    with open(frame_json_path, "r", encoding="utf-8") as f:
        frame_data = json.load(f)
    
    canvas_w = int(frame_data.get("expanded_canvas_width", frame_data.get("width", 1440)))
    canvas_h = int(frame_data.get("expanded_canvas_height", frame_data.get("height", 900)))
    canvas_size = (canvas_w, canvas_h)
    
    origin_x = frame_data.get("canvas_origin_x", 0)
    origin_y = frame_data.get("canvas_origin_y", 0)
    
    unit_images_subdir = frame_data.get("unit_images_dir", "")
    unit_images_dir = split_dir / unit_images_subdir
    
    # Load composite (GT reconstruction) image
    gt_recon_img = None
    recon_path_rel = frame_data.get("reconstructed_image_path")
    if recon_path_rel:
        recon_full_path = unit_images_dir / recon_path_rel
        if recon_full_path.exists():
            try:
                gt_recon_img = Image.open(recon_full_path).convert("RGBA")
                if gt_recon_img.size != canvas_size:
                    gt_recon_img = gt_recon_img.resize(canvas_size, Image.LANCZOS)
            except Exception as e:
                if logger:
                    logger.warn(f"Failed to load reconstructed image: {e}")
    
    elements = []
    
    # -------------------------------------------------------------------------
    # Helper: Place Image on Canvas using NumPy (Speed Optimization)
    # -------------------------------------------------------------------------
    def place_image_numpy(src_img: Image.Image, x: int, y: int, W: int, H: int) -> Optional[Image.Image]:
        """
        Places a small image onto a large canvas using NumPy array slicing.
        Much faster and more memory-efficient than PIL.Image.new() + paste().
        """
        # Convert source image to array
        src_arr = np.array(src_img)
        h_src, w_src = src_arr.shape[:2]
        
        # 1. Compute valid coordinates within canvas bounds (Clipping)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w_src)
        y2 = min(H, y + h_src)
        
        # Return None if there is no valid region (off-screen)
        if x2 <= x1 or y2 <= y1:
            return None
            
        # 2. Compute the region to take from the source image
        # (Crop the source by as much as it extends beyond the canvas)
        src_x1 = x1 - x
        src_y1 = y1 - y
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        
        # 3. Create an empty canvas (NumPy Zeros) -> very fast
        canvas_arr = np.zeros((H, W, 4), dtype=np.uint8)
        
        # 4. Direct array assignment (Copy)
        canvas_arr[y1:y2, x1:x2] = src_arr[src_y1:src_y2, src_x1:src_x2]
        
        return Image.fromarray(canvas_arr, "RGBA")

    # -------------------------------------------------------------------------
    # Background Processing
    # -------------------------------------------------------------------------
    bg_path_rel = frame_data.get("expanded_background_path")
    if bg_path_rel:
        bg_path = unit_images_dir / bg_path_rel
        if bg_path.exists():
            try:
                bg_img = Image.open(bg_path).convert("RGBA")
                bg_img = clean_alpha_noise(bg_img)
                
                # Place the background using the NumPy approach too (consistency and speed)
                canvas = place_image_numpy(bg_img, 0, 0, canvas_w, canvas_h)
                
                if canvas is not None:
                    mask = np.array(canvas.getchannel("A")).astype(np.float32) / 255.0
                    if mask.sum() > MIN_ELEMENT_AREA:
                        elements.append(create_element(
                            elem_id="gt_background",
                            elem_type="background",
                            mask=mask,
                            image=canvas,
                            bbox=[0, 0, canvas_w, canvas_h],
                            z_index=-1,
                            source="figma_gt"
                        ))
            except Exception as e:
                if logger:
                    logger.warn(f"Failed to load background: {e}")
    
    # -------------------------------------------------------------------------
    # Unit Images Processing
    # -------------------------------------------------------------------------
    units = frame_data.get("unit_images", [])
    units_sorted = sorted(units, key=lambda u: u.get("z_index", 0))
    
    for idx, unit in enumerate(units_sorted):
        img_path = unit_images_dir / unit["image_path"]
        if not img_path.exists():
            continue
        
        try:
            img = Image.open(img_path).convert("RGBA")
            img = clean_alpha_noise(img)
            
            # Coordinate calculation
            if unit.get("is_background"):
                paste_x = int(round(unit["render_x"] - origin_x))
                paste_y = int(round(unit["render_y"] - origin_y))
            else:
                paste_x = int(round(unit.get("original_render_x", unit["render_x"]) - origin_x))
                paste_y = int(round(unit.get("original_render_y", unit["render_y"]) - origin_y))
            
            # [Optimization] Use the NumPy helper function
            canvas = place_image_numpy(img, paste_x, paste_y, canvas_w, canvas_h)
            
            if canvas is None:
                continue
            
            mask = np.array(canvas.getchannel("A")).astype(np.float32) / 255.0
            
            if mask.sum() > MIN_ELEMENT_AREA:
                # Store BBox clipped to the canvas boundaries
                x1 = max(0, paste_x)
                y1 = max(0, paste_y)
                x2 = min(canvas_w, paste_x + img.width)
                y2 = min(canvas_h, paste_y + img.height)
                
                elements.append(create_element(
                    elem_id=f"gt_{unit.get('unit_id', idx)}",
                    elem_type=unit.get("unit_type", "object"),
                    mask=mask,
                    image=canvas,
                    bbox=[x1, y1, x2, y2],
                    z_index=unit.get("z_index", idx),
                    source="figma_gt"
                ))
        except Exception as e:
            if logger:
                logger.warn(f"Failed to load {img_path}: {e}")
    
    return elements, canvas_size, gt_recon_img



def apply_soft_kmeans_refinement(elem: Dict) -> Dict:
    """
    Separates colors with K-Means while applying 'Soft-Thresholding' to
    prevent text strokes from becoming thin and to preserve IoU.
    """
    # Extract source data
    img_pil = elem["image"].convert("RGBA")
    img_arr = np.array(img_pil).astype(np.float32)
    
    alpha = img_arr[..., 3]
    rgb = img_arr[..., :3]
    
    # Perform clustering only on valid pixels (alpha 10 or higher)
    valid_mask = alpha > 16 
    if np.sum(valid_mask) < 32: 
        return elem

    # 1. Extract representative foreground/background colors with K-Means
    pixels = rgb[valid_mask]
    try:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=3).fit(pixels)
        centers = kmeans.cluster_centers_ # [2, 3] shape
    except Exception:
        # Return original if sklearn is missing or an error occurs
        return elem

    # 2. Find the foreground (text) center (assume the label in high-alpha, dense areas is the foreground)
    full_labels = np.full(alpha.shape, -1, dtype=int)
    full_labels[valid_mask] = kmeans.labels_
    
    # Select the most frequent label in the more confident foreground region (alpha > 220) as the foreground
    core_mask = alpha > 200
    if np.sum(core_mask) < 10: 
        core_mask = valid_mask
    
    # Set the most frequent label among the core_mask region as the foreground
    valid_labels = full_labels[core_mask]
    valid_labels = valid_labels[valid_labels != -1]
    
    if len(valid_labels) == 0:
        return elem
        
    fg_label = np.argmax(np.bincount(valid_labels))
    bg_label = 1 - fg_label
    
    C_fg = centers[fg_label]
    C_bg = centers[bg_label]

    # 3. Compute distance to foreground/background colors for every pixel
    dist_fg = np.linalg.norm(rgb - C_fg, axis=2)
    dist_bg = np.linalg.norm(rgb - C_bg, axis=2)

    # 4. Soft-Weighting logic
    # The closer ratio is to 0, the more foreground; the closer to 1, the more background
    ratio = dist_fg / (dist_fg + dist_bg + 1e-6)
    
    # protection_threshold: if closer to foreground than this value (smaller ratio), the alpha is left untouched
    # Higher values (e.g., 0.6~0.7) keep text thicker
    protection_threshold = 0.6
    
    new_alpha = alpha.copy()
    
    # Select only those much closer to the background (ratio > threshold)
    kill_indices = (ratio > protection_threshold) & (valid_mask)
    
    # 5. Instead of erasing abruptly, decay smoothly according to ratio (Linear Decay)
    if np.any(kill_indices):
        decay = (1.0 - ratio[kill_indices]) / (1.0 - protection_threshold)
        # Square it to make the boundary smoother
        new_alpha[kill_indices] = new_alpha[kill_indices] * (decay ** 2)

    # 6. Apply the result
    new_img_arr = np.array(img_pil)
    new_img_arr[..., 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)
    
    new_elem = elem.copy()
    new_elem["image"] = Image.fromarray(new_img_arr)
    # Update the 0~1 range float32 mask
    new_elem["mask"] = (new_alpha / 255.0).astype(np.float32)
    # Recompute area
    new_elem["area"] = float(np.sum(new_elem["mask"] > 0))
    
    return new_elem

def extract_agent_elements(
    episode_dir: Path,
    canvas_size: Tuple[int, int],
    apply_alpha_correction: bool = True,
    text_refinement: bool = True,
    logger: Optional[WorkerLogger] = None
) -> List[Dict]:
    """Extract Agent parsed elements with optional alpha correction.
    
    """
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
        
        elem_id = elem.get("id")
        
        # Priority 1: canvas_image.png (for text recovery or full placement)
        # Priority 2: extracted.png or crop_image.png (for general objects)
        potential_files = [
            episode_dir / "elements" / elem_id / "canvas_image.png",
            episode_dir / "elements" / elem_id / "extracted.png",
            episode_dir / "elements" / elem_id / "crop_image.png"
        ]
        
        img_path = None
        for p in potential_files:
            if p.exists():
                img_path = p
                break
        
        # If not found at the paths above, fall back to the filename recorded in the JSON
        if img_path is None:
            json_uri = elem.get("canvas_image_uri") or elem.get("extracted_image_uri")
            if json_uri:
                fallback_name = Path(json_uri).name
                fallback_path = episode_dir / "elements" / elem_id / fallback_name
                if fallback_path.exists():
                    img_path = fallback_path

        if img_path is None:
            if logger:
                logger.warn(f"Element {elem_id} image not found. Skipping.")
            continue
        
        try:
            elem_img = Image.open(img_path).convert("RGBA")
            elem_img = clean_alpha_noise(elem_img)
            
            # Canvas size (W, H)
            canvas_arr = np.zeros((H, W, 4), dtype=np.uint8)
            x1_b, y1_b, x2_b, y2_b = [int(b) for b in bbox]

            if elem.get("canvas_image_uri") and elem_img.size == (W, H):
                # [Key fix] Even when loading the full canvas image, keep only the bbox region and remove the rest
                full_arr = np.array(elem_img)
                # Copy only the bbox region (the rest stays at the initial 0,0,0,0)
                y1_clip, y2_clip = max(0, y1_b), min(H, y2_b)
                x1_clip, x2_clip = max(0, x1_b), min(W, x2_b)
                
                canvas_arr[y1_clip:y2_clip, x1_clip:x2_clip] = full_arr[y1_clip:y2_clip, x1_clip:x2_clip]
                canvas = Image.fromarray(canvas_arr, "RGBA")
            else:
                # For extracted_image (a cropped image), keep the existing logic
                # (Already cropped, so there is less noise)
                x1, y1 = int(bbox[0]), int(bbox[1])
                if x1 < W and y1 < H:
                    elem_arr = np.array(elem_img)
                    eh, ew = elem_img.height, elem_img.width
                    copy_h = min(eh, H - y1)
                    copy_w = min(ew, W - x1)
                    canvas_arr[y1:y1+copy_h, x1:x1+copy_w] = elem_arr[:copy_h, :copy_w]
                canvas = Image.fromarray(canvas_arr, "RGBA")
            
            # Alpha correction
            if original_alpha is not None:
                if elem.get("canvas_image_uri") and elem_img.size == (W, H):
                    canvas = apply_original_alpha_to_element(
                        canvas, [0, 0, W, H], original_alpha, canvas_size,
                        mode="zero_mask"
                    )
                else:
                    x1, y1 = int(bbox[0]), int(bbox[1])
                    x2 = min(W, x1 + elem_img.width)
                    y2 = min(H, y1 + elem_img.height)
                    
                    if x1 < W and y1 < H:
                        # [FIX] crop -> correct -> direct numpy copy
                        region = canvas.crop((x1, y1, x2, y2))
                        corrected_region = apply_original_alpha_to_element(
                            region, 
                            [x1, y1, x2, y2], 
                            original_alpha, 
                            canvas_size,
                            mode="zero_mask"
                        )
                        # Direct numpy copy instead of paste(img, pos, mask)
                        canvas_arr = np.array(canvas)
                        corrected_arr = np.array(corrected_region)
                        rh, rw = corrected_arr.shape[:2]
                        canvas_arr[y1:y1+rh, x1:x1+rw] = corrected_arr
                        canvas = Image.fromarray(canvas_arr, "RGBA")
            
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
                logger.warn(f"Failed to load {img_path}: {e}")
    
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


def _fast_path_bbox_from_d(d_attr: str, viewbox: str, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    """Quickly estimate pixel-space bbox from SVG path 'd' attribute coordinates.

    Returns (x1, y1, x2, y2) in canvas pixels, or None if parsing fails.
    """
    import re
    # Extract all floating-point numbers from the path data
    nums = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d_attr)
    if len(nums) < 2:
        return None
    vals = [float(n) for n in nums]
    xs = vals[0::2]  # even indices = x coords
    ys = vals[1::2]  # odd indices = y coords
    if not xs or not ys:
        return None
    # Coords are in viewBox space; map to canvas pixels
    vb_parts = viewbox.split()
    if len(vb_parts) == 4:
        vb_x, vb_y, vb_w, vb_h = float(vb_parts[0]), float(vb_parts[1]), float(vb_parts[2]), float(vb_parts[3])
    else:
        vb_x, vb_y, vb_w, vb_h = 0, 0, float(W), float(H)
    sx = W / max(vb_w, 1)
    sy = H / max(vb_h, 1)
    px_xs = [(x - vb_x) * sx for x in xs]
    px_ys = [(y - vb_y) * sy for y in ys]
    x1 = max(0, int(min(px_xs)))
    y1 = max(0, int(min(px_ys)))
    x2 = min(W, int(max(px_xs)) + 1)
    y2 = min(H, int(max(px_ys)) + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def extract_omnisvg_elements(
    episode_dir: Path,
    canvas_size: Tuple[int, int],
    logger: Optional[WorkerLogger] = None,
    filter_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    render_scale: float = 1.0,
    skip_upscale: bool = False,
    high_res_ids: Optional[set] = None,
) -> List[Dict]:
    """Extract elements from OmniSVG SVG output by isolating each <path> element.

    Parses the SVG, renders each path individually to RGBA at the target canvas size,
    and returns standard element dicts compatible with the matching pipeline.

    If filter_bboxes is provided, only render paths whose estimated bbox overlaps
    at least one of the given bboxes (fast pre-filtering to avoid unnecessary renders).

    If high_res_ids is provided (set of element ID strings like ``omnisvg_P0005``),
    those paths are rendered at full resolution (render_scale=1.0) regardless of
    the ``render_scale`` parameter.  All other paths are rendered at ``render_scale``
    and upscaled to canvas_size.  This gives accurate matched-element images while
    keeping scene-composition cost low.
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
    # Render at lower resolution for speed, then upscale masks/images
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
        # Try without namespace
        paths = root.findall('.//path')
    if not paths:
        if logger:
            logger.warn(f"No <path> elements found in {svg_path}")
        return []

    # Pre-filter: skip paths whose estimated bbox doesn't overlap any filter_bbox
    _skip_set: set = set()
    if filter_bboxes:
        for idx, path_elem in enumerate(paths):
            d_attr = path_elem.get('d', '')
            if not d_attr:
                _skip_set.add(idx)
                continue
            est = _fast_path_bbox_from_d(d_attr, viewbox, W, H)
            if est is None:
                continue  # can't estimate — render anyway
            ex1, ey1, ex2, ey2 = est
            overlaps = False
            for bx1, by1, bx2, by2 in filter_bboxes:
                if ex1 < bx2 and ex2 > bx1 and ey1 < by2 and ey2 > by1:
                    overlaps = True
                    break
            if not overlaps:
                _skip_set.add(idx)
        if logger:
            logger.info(f"Pre-filter: {len(_skip_set)}/{len(paths)} paths skipped (no GT overlap)")

    # Build render tasks (skip filtered paths)
    # Each task: (idx, svg_str, use_full_res)
    render_tasks = []
    for idx, path_elem in enumerate(paths):
        if idx in _skip_set:
            continue
        path_str = ET.tostring(path_elem, encoding='unicode')
        single_svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{viewbox}" width="{W}" height="{H}">'
            f'{path_str}</svg>'
        )
        # Determine if this path needs full-res rendering
        elem_id = f"omnisvg_P{idx:04d}"
        is_high = high_res_ids is not None and elem_id in high_res_ids
        render_tasks.append((idx, single_svg, is_high))

    # Parallel cairosvg rendering (cairosvg releases GIL)
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutTimeoutError

    _BATCH_TIMEOUT = 120  # max seconds for entire episode rendering

    def _render_one(args_tuple):
        idx, svg_str, is_high = args_tuple
        out_w = W if is_high else rW
        out_h = H if is_high else rH
        try:
            png_data = cairosvg.svg2png(
                bytestring=svg_str.encode('utf-8'),
                output_width=out_w,
                output_height=out_h,
            )
            return (idx, png_data, is_high)
        except Exception:
            return (idx, None, is_high)

    n_threads = min(8, len(render_tasks))
    rendered = []
    if n_threads > 1:
        pool = ThreadPoolExecutor(max_workers=n_threads)
        fut_to_idx = {}
        for t in render_tasks:
            fut_to_idx[pool.submit(_render_one, t)] = t[0]
        done_indices: set = set()
        try:
            for fut in as_completed(fut_to_idx, timeout=_BATCH_TIMEOUT):
                try:
                    result = fut.result(timeout=0)
                    rendered.append(result)
                    done_indices.add(result[0])
                except Exception:
                    idx = fut_to_idx[fut]
                    rendered.append((idx, None, False))
                    done_indices.add(idx)
        except FutTimeoutError:
            pass  # batch timeout — skip remaining paths
        # Fill None for paths that didn't finish
        for fut, idx in fut_to_idx.items():
            if idx not in done_indices:
                rendered.append((idx, None, False))
                fut.cancel()
        pool.shutdown(wait=False)  # Do NOT wait for hung cairosvg threads
    else:
        rendered = [_render_one(t) for t in render_tasks]

    elements = []
    for idx, png_data, is_high in rendered:
        if png_data is None:
            continue

        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        # Upscale low-res images to canvas size for scene composition
        if not is_high and render_scale != 1.0 and not skip_upscale:
            img = img.resize((W, H), Image.NEAREST)
        alpha = np.array(img.getchannel("A"))
        mask = alpha.astype(np.float32) / 255.0

        # Filter: minimum area and skip full-canvas backgrounds
        area = float(mask.sum())
        if area < MIN_ELEMENT_AREA:
            continue
        if area > 0.9 * W * H:
            continue

        # Compute bounding box from alpha
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
            # Force areas that had almost no alpha in the original (<=16) to transparent, keep the rest as-is
            orig_transparent = orig_alpha_float <= 16
            new_alpha = elem_alpha.copy()
            new_alpha[orig_transparent] = 0
            # Zero out the entire RGBA (prevents RGB contamination)
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

    # 1. Compute the Coverage Matrix
    # gt_side_coverage[i, j]: how much GT[i] contains Pred[j] (Pred coverage by GT)
    gt_side_coverage = np.zeros((n_gt, n_pred))
    # pred_side_coverage[j, i]: how much Pred[j] contains GT[i] (GT coverage by Pred)
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

    # Initialize groups (by default, each element is included as its own singleton group)
    gt_groups_set = set()
    for i in range(n_gt):
        gt_groups_set.add(tuple([i]))
        
    pred_groups_set = set()
    for j in range(n_pred):
        pred_groups_set.add(tuple([j]))

    # ---------------------------------------------------------
    # Logic fix: use cross coverage to build the "opposite side" groups
    # ---------------------------------------------------------

    # 1. Build Pred merge groups (resolves over-segmentation)
    # If a single GT(i) contains multiple Preds (j1, j2...), those Preds should form one group
    for i in range(n_gt):
        # Indices of Preds definitely contained by GT i
        mandatory = np.where(gt_side_coverage[i] >= CONTAINMENT_MERGE_THRESHOLD)[0].tolist()

        # Partially contained Preds (combination candidates)
        optional = np.where((gt_side_coverage[i] >= OPTIONAL_MERGE_THRESHOLD) &
                            (gt_side_coverage[i] < CONTAINMENT_MERGE_THRESHOLD))[0].tolist()

        # Generate combinations if there are Mandatory or Optional items
        base_group = mandatory

        # Generate combinations (for speed, limit when there are too many Optional items, or use all)
        # Here we build the Pred Group. (GT index i is NOT included!)
        if not optional:
            if len(base_group) > 1:
                pred_groups_set.add(tuple(sorted(base_group)))
        else:
            # PowerSet combinations over the Optional elements
            limit_optional = optional[:5] 
            for r in range(len(limit_optional) + 1):
                for combo in combinations(limit_optional, r):
                    new_group = sorted(list(set(base_group + list(combo))))
                    if len(new_group) > 0:
                        pred_groups_set.add(tuple(new_group))

    # 2. Build GT merge groups (resolves under-segmentation)
    # If a single Pred(j) contains multiple GTs (i1, i2...), those GTs should form one group
    for j in range(n_pred):
        # Indices of GTs definitely contained by Pred j
        mandatory = np.where(pred_side_coverage[j] >= CONTAINMENT_MERGE_THRESHOLD)[0].tolist()
        
        optional = np.where((pred_side_coverage[j] >= OPTIONAL_MERGE_THRESHOLD) & 
                            (pred_side_coverage[j] < CONTAINMENT_MERGE_THRESHOLD))[0].tolist()
        
        base_group = mandatory
        
        # Here we build the GT Group. (Pred index j is NOT included!)
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

    # set -> list conversion
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
    
    # Raw RGB on alpha region (same as the LayerD approach)
    gt_arr = np.array(gt_composite.convert("RGBA")).astype(np.float32) / 255.0
    pred_arr = np.array(pred_composite.convert("RGBA")).astype(np.float32) / 255.0
    
    gt_rgb = gt_arr[..., :3]
    pred_rgb = pred_arr[..., :3]
    
    if roi.sum() > 0:
        l1 = float(np.mean(np.abs(gt_rgb[roi] - pred_rgb[roi])))
    else:
        l1 = 1.0
    
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
    # 1. Prepare Mask Tensors (for IoU computation)
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
    # 2. Prepare RGB Tensors (first prepared as Premultiplied for Compositing)
    # -------------------------------------------------------------------------
    # The compositing process itself requires Alpha operations, so the input is prepared as Premultiplied.
    # We will Un-multiply later, right before comparison.
    def get_prem_rgb_tensor_cpu(elems):
        tensors = []
        for e in elems:
            img_rgba = np.array(e["image"].convert("RGBA")).astype(np.float32) / 255.0
            if img_rgba.shape[:2] != (cH, cW):
                img_rgba = cv2.resize(img_rgba, (cW, cH), interpolation=cv2.INTER_AREA)

            alpha = img_rgba[..., 3:4]
            rgb = img_rgba[..., :3]
            rgb_prem = rgb * alpha  # For compositing

            tensors.append(torch.from_numpy(rgb_prem).permute(2, 0, 1))
        return torch.stack(tensors)

    gt_rgbs_cpu = get_prem_rgb_tensor_cpu(gt_elements)
    pred_rgbs_cpu = get_prem_rgb_tensor_cpu(pred_elements)

    # -------------------------------------------------------------------------
    # 3. Group rendering & Un-multiply (restore pure RGB)
    # -------------------------------------------------------------------------
    def render_groups_unmultiplied(groups, masks_gpu, rgbs_cpu):
        grp_masks = []
        grp_rgbs = []   # Pure RGB (Non-premultiplied)
        grp_bboxes = []

        for idx_list in groups:
            # 3-1. Mask Combine (Union)
            union_mask = torch.max(masks_gpu[idx_list].float(), dim=0)[0]
            grp_masks.append(union_mask)

            sorted_indices = sorted(idx_list)

            # Initialize canvas (accumulated in Premultiplied state)
            # canvas_rgb: accumulated RGB*Alpha
            # canvas_alpha: accumulated Alpha
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
                # (Required for accurate Alpha compositing)
                canvas_alpha = src_a + canvas_alpha * (1.0 - src_a)
            
            # [Key fix] Un-multiply: RGB_pure = RGB_prem / Alpha
            # Where Alpha is near 0, set to 0 to avoid division errors
            safe_alpha = torch.clamp(canvas_alpha, min=1e-6)
            pure_rgb = canvas_rgb / safe_alpha
            
            # Where Alpha is almost absent (fully transparent), mask RGB to 0 as well (prevents noise)
            is_transparent = canvas_alpha < 1e-3
            pure_rgb = torch.where(is_transparent.expand(3, -1, -1), torch.zeros_like(pure_rgb), pure_rgb)
            
            grp_rgbs.append(pure_rgb)  # Keep on GPU

            # 3-3. BBox computation
            coords = torch.nonzero(union_mask > 0)
            if coords.shape[0] > 0:
                y1, x1 = coords.min(dim=0)[0]
                y2, x2 = coords.max(dim=0)[0]
                grp_bboxes.append((int(x1), int(y1), int(x2+1), int(y2+1)))
            else:
                grp_bboxes.append((0, 0, 1, 1))
                
        return grp_masks, grp_rgbs, grp_bboxes

    # Perform rendering (results are pure RGB, kept on GPU)
    gt_grp_masks, gt_grp_rgbs, gt_grp_bboxes = render_groups_unmultiplied(gt_groups, gt_masks_gpu, gt_rgbs_cpu)
    pred_grp_masks, pred_grp_rgbs, pred_grp_bboxes = render_groups_unmultiplied(pred_groups, pred_masks_gpu, pred_rgbs_cpu)

    # -------------------------------------------------------------------------
    # 4. Compute the Cost Matrix
    # -------------------------------------------------------------------------
    cost_matrix = np.full((n_gt_grp, n_pred_grp), 10000.0, dtype=np.float64)
    l1_matrix = np.zeros((n_gt_grp, n_pred_grp))
    iou_matrix = np.zeros((n_gt_grp, n_pred_grp))

    # Pre-compute bbox overlap matrix to skip non-overlapping pairs
    gt_bb = np.array(gt_grp_bboxes)   # [n_gt_grp, 4] (x1, y1, x2, y2)
    pred_bb = np.array(pred_grp_bboxes)  # [n_pred_grp, 4]
    # bbox overlap: gt_x1 < pred_x2 and gt_x2 > pred_x1 and gt_y1 < pred_y2 and gt_y2 > pred_y1
    bb_overlap = (
        (gt_bb[:, 0:1] < pred_bb[None, :, 2]) &  # gt_x1 < pred_x2
        (gt_bb[:, 2:3] > pred_bb[None, :, 0]) &  # gt_x2 > pred_x1
        (gt_bb[:, 1:2] < pred_bb[None, :, 3]) &  # gt_y1 < pred_y2
        (gt_bb[:, 3:4] > pred_bb[None, :, 1])    # gt_y2 > pred_y1
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

            # IoU (based on Full Mask Union)
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
    """Hungarian Algorithm - optimal + fast"""
    from scipy.optimize import linear_sum_assignment

    # Make the cost matrix square
    max_dim = max(len(gt_groups), len(pred_groups))
    padded = np.full((max_dim, max_dim), 999.0)
    padded[:len(gt_groups), :len(pred_groups)] = cost_matrix
    
    # Hungarian (O(n³), but very fast in practice)
    row_ind, col_ind = linear_sum_assignment(padded)
    
    # Filter results while checking for duplicate Elements
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
    
    # 1. Compute visible region masks
    gt_with_visible = compute_visible_masks(gt_elements, canvas_size)
    pred_with_visible = compute_visible_masks(pred_elements, canvas_size, apply_alpha_cleaning=apply_pred_alpha_cleaning)
    
    # 2. Containment-based group generation (improved logic)
    gt_groups, pred_groups = generate_combinatorial_groups(gt_with_visible, pred_with_visible, logger=logger)
    
    # 3. GPU-accelerated cost matrix generation (improved logic)
    cost_matrix, l1_matrix, iou_matrix = build_cost_matrix_gpu(
        gt_with_visible, pred_with_visible, gt_groups, pred_groups, canvas_size, device=device, logger=logger
    )
    

    # 4. Optimal matching (keep existing logic)
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
            "mask": gt_full_mask,      # Full mask without occlusion
            "image": gt_full_img,      # Full image without occlusion
            "bbox": merged_bbox,
            "area": float(gt_full_mask.sum()),
            "visible_mask": gt_union_mask, # Keep the matching visible mask for reference
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
    Soft IoU (LayerD approach)
    Uses Alpha values (continuous 0~1) directly
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
    Binary IoU (existing approach)
    Binarizes Alpha to 0/1 before computing
    """
    bin1 = mask1 > 0
    bin2 = mask2 > 0
    
    intersection = (bin1 & bin2).sum()
    union = (bin1 | bin2).sum()
    
    return float(intersection / (union + eps))


# =============================================================================
# Visual Quality computation function (LayerD approach)
# =============================================================================

def compute_visual_quality_no_composite(
    gt_rgba: np.ndarray,
    pred_rgba: np.ndarray,
    region: str = "intersection", # 'intersection' or 'union'
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    Visual Quality Calculation with Pre-multiplied Alpha (Min-Error Strategy)
    
    - Uses Pre-multiplied RGB, i.e., RGB values multiplied by Alpha.
    - Performs proper Alpha Compositing over both a Black and a White background,
      then selects the value with the smaller error (Min) of the two cases.
    - This evaluates whether the object appears natural without artifacts on at
      least one of a dark or a bright background.
    
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
    # Keep in (H, W, 1) shape so it can broadcast
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
    elif region == "gt":
        roi_mask = gt_visible
    elif region == "pred":
        roi_mask = pred_visible
    else:
        roi_mask = gt_visible | pred_visible
    
    if roi_mask.sum() == 0:
        return {"l1": float('nan'), "l2": float('nan'), "psnr": float('nan')}

    # 4. Prepare Background Versions (Alpha Compositing)
    
    # Case 1: Black Background (0, 0, 0)
    # Formula: Color * Alpha + Black * (1 - Alpha) 
    # Since Black=0, this is Color * Alpha (i.e., Premultiplied itself)
    gt_img_b = gt_prem
    pred_img_b = pred_prem
    
    # Case 2: White Background (1, 1, 1)
    # Formula: Color * Alpha + White * (1 - Alpha)
    gt_img_w = gt_prem + (1.0 - gt_alpha)
    pred_img_w = pred_prem + (1.0 - pred_alpha)
    
    # 5. Helper Function for Metrics
    def calc_metrics(p_gt_full, p_pred_full):
        # Extract only the pixels within the ROI region for comparison
        p_gt = p_gt_full[roi_mask]
        p_pred = p_pred_full[roi_mask]
        
        val_l1 = float(np.mean(np.abs(p_gt - p_pred)))
        val_l2 = float(mean_squared_error(p_gt, p_pred))
        val_psnr = float(peak_signal_noise_ratio(p_gt, p_pred, data_range=1.0))
        
        # Preserve inf as-is -> replaced with the finite maximum during aggregation
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
# Element-wise Dual metric computation
# =============================================================================

def compute_element_metrics_dual(
    gt_elem: Dict,
    pred_elems: List[Dict],
    canvas_size: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Computes Element-wise metrics in two ways

    Returns:
        {
            "visual_quality": {
                "gt_region": {"l1", "l2", "psnr"},   # GT region only
                "union_region": {"l1", "l2", "psnr"} # Union region
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
    # Prepare GT image
    # =========================================================================
    gt_img = gt_elem["image"].convert("RGBA")
    if gt_img.size != (W, H):
        gt_img = gt_img.resize((W, H), Image.LANCZOS)
    gt_rgba = np.array(gt_img)
    
    # GT mask
    gt_mask = gt_elem["mask"]
    if gt_mask.shape != (H, W):
        gt_mask = cv2.resize(gt_mask.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    
    # =========================================================================
    # Prepare composited Pred image
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
    # Area computation
    # =========================================================================
    gt_area = float((gt_mask > 0).sum())
    pred_area = float((pred_mask > 0).sum())
    union_area = float(((gt_mask > 0) | (pred_mask > 0)).sum())
    
    # =========================================================================
    # Visual Quality computation (without Alpha Composite!)
    # =========================================================================
    vq_intersection  = compute_visual_quality_no_composite(gt_rgba, pred_rgba, region="intersection")
    vq_union_region = compute_visual_quality_no_composite(gt_rgba, pred_rgba, region="union")
    vq_gt_region = compute_visual_quality_no_composite(gt_rgba, pred_rgba, region="gt")

    # =========================================================================
    # IoU computation
    # =========================================================================
    soft_iou = calc_soft_iou(gt_mask, pred_mask)
    binary_iou = calc_binary_iou(gt_mask, pred_mask)

    return {
        "visual_quality": {
            "intersection_region": vq_intersection,
            "union_region": vq_union_region,
            "gt_region": vq_gt_region,
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
# Panoptic Quality (PQ, SQ, RQ) computation
# =============================================================================

def compute_panoptic_quality_dual(
    matched_pairs: List[Dict],
    unmatched_gt: List[Dict],
    unmatched_pred: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Computes PQ with two IoU methods (Kirillov et al., CVPR 2019 standard)

    Only matches with IoU > iou_threshold are classified as TP.
    Matches with IoU <= iou_threshold are classified as FP+FN.
    
    Returns:
        {
            "soft": {"pq", "sq", "rq", "tp", "fn", "fp"},
            "binary": {"pq", "sq", "rq", "tp", "fn", "fp"}
        }
    """
    results = {}
    
    for iou_type in ["soft", "binary"]:
        # TP/FP/FN classification based on the IoU threshold
        tp_pairs = []
        poor_matches = 0
        
        for mp in matched_pairs:
            iou_val = mp.get("metrics_dual", {}).get("iou", {}).get(iou_type, 0)
            if iou_val > iou_threshold:
                tp_pairs.append(iou_val)
            else:
                poor_matches += 1
        
        tp_count = len(tp_pairs)
        # Poor matches are added to both FP and FN (Kirillov et al. standard)
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
# Element metric aggregation
# =============================================================================

def aggregate_element_metrics_dual(matched_pairs: List[Dict]) -> Dict[str, Any]:
    """
    Aggregates Dual metrics over all matched pairs

    - NaN values are excluded from aggregation
    - PSNR inf values are replaced with the finite maximum before aggregation

    Returns:
        {
            "visual_quality": {
                "gt_region": {"l1", "l2", "psnr"},
                "union_region": {"l1", "l2", "psnr"}
            },
            "iou": {"soft", "binary"},
            (each including simple_avg, weighted_avg)
        }
    """
    if not matched_pairs:
        return {
            "visual_quality": {
                "intersection_region": {"simple_avg": {}, "weighted_avg": {}},
                "union_region": {"simple_avg": {}, "weighted_avg": {}},
                "gt_region": {"simple_avg": {}, "weighted_avg": {}},
            },
            "iou": {"simple_avg": {}, "weighted_avg": {}}
        }
    
    def _aggregate_values(values, weights, is_psnr=False):
        """Remove NaN, replace PSNR inf with max_finite, then compute the average"""
        # Remove NaN
        clean = [(v, w) for v, w in zip(values, weights) if not np.isnan(v)]
        if not clean:
            return float('nan'), float('nan')
        
        vals = [v for v, w in clean]
        wts = [w for v, w in clean]
        
        # PSNR: replace inf with the finite maximum
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
    
    # Visual Quality aggregation
    vq_result = {
        "intersection_region": {"simple_avg": {}, "weighted_avg": {}},
        "union_region": {"simple_avg": {}, "weighted_avg": {}},
        "gt_region": {"simple_avg": {}, "weighted_avg": {}},
    }

    for region in ["intersection_region", "union_region", "gt_region"]:
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
    
    # IoU aggregation
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
# Logging functions
# =============================================================================

def log_dual_metrics(matched_pairs: List[Dict], pq_dual: Dict, logger=None):
    """Log the metrics for both methods"""
    
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    def _clean(vals, is_psnr=False):
        """Remove NaN; for PSNR, replace inf with max_finite"""
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
    
    # Collect data
    inter_l1s, inter_l2s, inter_psnrs = [], [], []
    union_l1s, union_l2s, union_psnrs = [], [], []
    gt_l1s, gt_l2s, gt_psnrs = [], [], []
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
        
        gt_l1s.append(gt_r.get("l1", 0))
        gt_l2s.append(gt_r.get("l2", 0))
        gt_psnrs.append(gt_r.get("psnr", 0))
        
        soft_ious.append(iou.get("soft", 0))
        binary_ious.append(iou.get("binary", 0))
    
    # Clean up NaN/inf
    c_inter_l1 = _clean(inter_l1s); c_inter_l2 = _clean(inter_l2s); c_inter_psnr = _clean(inter_psnrs, is_psnr=True)
    c_union_l1 = _clean(union_l1s); c_union_l2 = _clean(union_l2s); c_union_psnr = _clean(union_psnrs, is_psnr=True)
    c_gt_l1 = _clean(gt_l1s); c_gt_l2 = _clean(gt_l2s); c_gt_psnr = _clean(gt_psnrs, is_psnr=True)
    
    log("=" * 100)
    log("[Visual Quality - No Alpha Composite]")
    log("-" * 100)
    log(f"  {'Metric':<8} {'Intersection Region':<28} {'Union Region':<28} {'GT Region (LayerD)':<28}")
    log(f"  {'-'*92}")
    log(f"  {'L1':<8} min={min(c_inter_l1):.4f} max={max(c_inter_l1):.4f} mean={np.mean(c_inter_l1):.4f}   "
        f"min={min(c_union_l1):.4f} max={max(c_union_l1):.4f} mean={np.mean(c_union_l1):.4f}   "
        f"min={min(c_gt_l1):.4f} max={max(c_gt_l1):.4f} mean={np.mean(c_gt_l1):.4f}")
    log(f"  {'L2':<8} min={min(c_inter_l2):.4f} max={max(c_inter_l2):.4f} mean={np.mean(c_inter_l2):.4f}   "
        f"min={min(c_union_l2):.4f} max={max(c_union_l2):.4f} mean={np.mean(c_union_l2):.4f}   "
        f"min={min(c_gt_l2):.4f} max={max(c_gt_l2):.4f} mean={np.mean(c_gt_l2):.4f}")
    log(f"  {'PSNR':<8} min={min(c_inter_psnr):.2f} max={max(c_inter_psnr):.2f} mean={np.mean(c_inter_psnr):.2f}   "
        f"min={min(c_union_psnr):.2f} max={max(c_union_psnr):.2f} mean={np.mean(c_union_psnr):.2f}   "
        f"min={min(c_gt_psnr):.2f} max={max(c_gt_psnr):.2f} mean={np.mean(c_gt_psnr):.2f}")
    
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
    """Print the Dual metric comparison table"""
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
    print(f"  {'L1':<12} {inter_r.get('l1', 0):<22.4f} {union_r.get('l1', 0):<22.4f} {gt_r.get('l1', 0):<22.4f}")
    print(f"  {'L2':<12} {inter_r.get('l2', 0):<22.4f} {union_r.get('l2', 0):<22.4f} {gt_r.get('l2', 0):<22.4f}")
    print(f"  {'PSNR':<12} {inter_r.get('psnr', 0):<22.2f} {union_r.get('psnr', 0):<22.2f} {gt_r.get('psnr', 0):<22.2f}")
    
    print("\n  [Layout]")
    print(f"  {'Metric':<12} {'Soft IoU (LayerD)':<22} {'Binary IoU':<22}")
    print(f"  {'-'*56}")
    print(f"  {'IoU':<12} {iou_agg.get('soft', 0):<22.4f} {iou_agg.get('binary', 0):<22.4f}")
    print(f"  {'PQ':<12} {pq_dual.get('soft', {}).get('pq', 0):<22.4f} {pq_dual.get('binary', {}).get('pq', 0):<22.4f}")
    print(f"  {'SQ':<12} {pq_dual.get('soft', {}).get('sq', 0):<22.4f} {pq_dual.get('binary', {}).get('sq', 0):<22.4f}")
    print(f"  {'RQ':<12} {pq_dual.get('soft', {}).get('rq', 0):<22.4f} {pq_dual.get('binary', {}).get('rq', 0):<22.4f}")
    print()


# =============================================================================
# Aggregating results across multiple episodes
# =============================================================================

def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across multiple episodes (including Dual metrics)

    - NaN values are excluded from aggregation
    - PSNR inf values are replaced with the finite maximum before aggregation
    """
    if not results:
        return {}
    
    def _safe_mean(vals, is_psnr=False):
        """Average after removing NaN. For PSNR, replace inf with max_finite."""
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
    
    # Existing aggregation (backward compatibility)
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
    # [NEW] Dual metric aggregation
    # =========================================================================
    vq_agg = {
        # [Fix] Renamed keys
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
        # Existing fields
        "element_metrics": {"simple_avg": all_simple, "weighted_avg": all_weighted},
        "panoptic_quality": {"pq": float(np.mean(pq_vals)), "sq": float(np.mean(sq_vals)), "rq": float(np.mean(rq_vals))},
        "composite_metrics": comp_agg,
        # [NEW] Dual fields
        "element_metrics_dual": {"visual_quality": vq_agg, "iou": iou_agg},
        "panoptic_quality_dual": pq_dual_agg,
    }

# =============================================================================
# Test
# =============================================================================

# if __name__ == "__main__":
#     print("Dual Metrics Module - Test")
#     print("=" * 50)
    
#     # Create test masks
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
    
#     # Create test RGBA images
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
    # 1. Place elements on a transparent canvas (returns a PIL Image RGBA)
    canvas_rgba_pil, total_mask = composite_elements_transparent(elements, canvas_size, use_overwrite=use_overwrite)

    # 2. Without blending with a (black) background, convert the pure RGBA values to 0~1 float and return
    # Keep the (H, W, 4) shape
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
        
        # 1. Size correction
        if elem_img.size != (W, H):
            elem_img = elem_img.resize((W, H), Image.LANCZOS)
        
        # 2. Compositing logic branch
        # Even when use_overwrite is True, apply it only to the 'text' type to prevent the Halo effect
        # (If the image came from a flattened original, text is already blended with the background color, so it should be overwritten)
        is_text = elem.get("type") == "text"
        
        if use_overwrite and is_text:
            # [Overwrite Logic] Text Only
            # Binarize the mask region (Alpha > 0) to 255 to punch a 'hole', then fill it 1:1 with pixels
            alpha = elem_img.getchannel("A")
            # If there is any value at all, treat it as fully opaque (255) to build the overwrite mask
            binary_mask = alpha.point(lambda p: 255 if p > 0 else 0)

            # Completely replace that region of the canvas with elem_img (no Blend)
            canvas = Image.composite(elem_img, canvas, binary_mask)
            
        else:
            # [Standard Logic] Non-text elements or when overwrite is False
            # Blend naturally considering transparency (Alpha Compositing)
            canvas.paste(elem_img, (0, 0), elem_img)
        
        # 3. Update the total mask (keep existing logic)
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
    # 1. Prepare GT RGBA (H, W, 4)
    # =========================================================
    if gt_recon_img is not None:
        # When the Figma original is available
        gt_rgba = np.array(gt_recon_img.convert("RGBA")).astype(np.float32) / 255.0
        # GT Mask uses the Alpha channel
        gt_mask = gt_rgba[..., 3]
    else:
        # When re-compositing is needed (GT is usually overwrite=False)
        gt_rgba, gt_mask = composite_elements(gt_elements, canvas_size, use_overwrite=False)
    
    # =========================================================
    # 2. Prepare Pred RGBA (H, W, 4)
    # =========================================================
    # Apply overwrite mode only when method_name is 'agent'
    is_agent = (method_name == "agent")
    pred_rgba, pred_mask = composite_elements(pred_elements, canvas_size, use_overwrite=is_agent)
    
    # =========================================================
    # 3. Alpha Blending Simulation (Black vs White)
    # =========================================================
    # Extract the Alpha channel
    gt_alpha = gt_rgba[..., 3:4]      # (H, W, 1)
    pred_alpha = pred_rgba[..., 3:4]  # (H, W, 1)
    
    # Pre-multiply RGB (RGB * Alpha)
    gt_prem = gt_rgba[..., :3] * gt_alpha
    pred_prem = pred_rgba[..., :3] * pred_alpha
    
    # --- Case A: Black Background (0) ---
    # Formula: Color * Alpha + 0 * (1 - Alpha) = Pre-multiplied itself
    gt_rgb_black = gt_prem
    pred_rgb_black = pred_prem
    
    # --- Case B: White Background (1) ---
    # Formula: Color * Alpha + 1 * (1 - Alpha)
    gt_rgb_white = gt_prem + (1.0 - gt_alpha)
    pred_rgb_white = pred_prem + (1.0 - pred_alpha)
    
    # =========================================================
    # 4. Compute Metrics (Min-Error Strategy)
    # =========================================================
    
    # Helper: L1, L2, PSNR computation function
    def calc_pixel_metrics(gt, pred):
        l1 = float(np.mean(np.abs(gt - pred)))
        l2 = float(mean_squared_error(gt, pred))
        psnr = float(peak_signal_noise_ratio(gt, pred, data_range=1.0))
        return l1, l2, psnr

    # Black background score
    l1_b, l2_b, psnr_b = calc_pixel_metrics(gt_rgb_black, pred_rgb_black)
    # White background score
    l1_w, l2_w, psnr_w = calc_pixel_metrics(gt_rgb_white, pred_rgb_white)

    # [Key] Select the background with the smaller error (higher score) of the two
    final_l1 = min(l1_b, l1_w)
    final_l2 = min(l2_b, l2_w)
    # Handle the case where PSNR is inf (usually comes out as inf in numpy)
    final_psnr = max(psnr_b, psnr_w)
    if np.isinf(final_psnr):
        final_psnr = float('inf')

    # =========================================================
    # 5. Advanced Metrics (SSIM, LPIPS, DINO)
    # =========================================================
    # Due to computational cost, SSIM and similar metrics usually use the 'Standard Composite (Black)' as the reference,
    # or use the 'Best Background' image selected above.
    # Here we use the image from whichever side had the lower L1 to keep it fair.
    
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
        # Likewise compute using the selected 'Best Background' image
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
    
    # [Fix] Clarify the mask logic
    real_intersection = gt_bin & pred_bin       # Intersection (Both)
    gt_only = gt_bin & ~pred_bin                # GT Only (difference)
    pred_only = pred_bin & ~gt_bin              # Pred Only (difference)

    # [Fix] Color assignment
    # 1. Intersection -> Green
    vis[real_intersection] = [0, 255, 0, 160]

    # 2. GT Only -> Red - the region the 'intersection' variable name referred to in the old code
    vis[gt_only] = [255, 0, 0, 160]

    # 3. Pred Only -> Blue
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
    # [Fix] Added so Dual Metrics can be passed as arguments
    element_metrics_dual: Optional[Dict] = None,
    panoptic_quality_dual: Optional[Dict] = None,
    gt_recon_img: Optional[Image.Image] = None,
):
    """
    Save visualization images and metrics.json including Dual Metrics.
    """
    method_dir = output_dir / episode_id / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save Visualization Images (keep existing logic)
    matched_cards = []
    for i, match in enumerate(matched_pairs):
        card = create_matched_pair_visualization(match, canvas_size)
        gt_id = match['gt']['id'][:20] if 'id' in match['gt'] else f"merged_{i}"
        # Safeguards such as removing special characters from the filename can be added
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
    
    # 2. Build JSON data (including Dual Metrics)

    # Organize detailed metrics for each matched pair
    matched_pairs_details = []
    for mp in matched_pairs:
        detail = {
            "gt_id": mp["gt"]["id"],
            "gt_indices": mp.get("gt_indices", []),
            "pred_ids": [p["id"] for p in mp["preds"]],
            "pred_indices": mp.get("pred_indices", []),
            "match_type": mp["match_type"],
            "cost": mp.get("cost", 0),
            # Existing legacy metrics
            "metrics": mp.get("metrics", {}),
            # [Fix] Include Dual Metrics if they have been computed
            "metrics_dual": mp.get("metrics_dual", {})
        }
        matched_pairs_details.append(detail)
    
    # Full metrics data structure
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
        
        # Composite level (LPIPS, DINO, etc.)
        "composite_metrics": composite_metrics,

        # For legacy compatibility (Binary IoU based)
        "panoptic_quality": pq_metrics,

        # [Fix] Dual Metrics (Soft/Binary IoU, No-Alpha-Composite VQ)
        # Store the values passed as arguments (empty dict if none)
        "element_metrics_dual": element_metrics_dual if element_metrics_dual else {},
        "panoptic_quality_dual": panoptic_quality_dual if panoptic_quality_dual else {},

        # Detailed matching list
        "matched_pairs": matched_pairs_details
    }
    
    # 3. Save the JSON file
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
    
    # 1. Perform matching (existing logic)
    matched_pairs, unmatched_gt, unmatched_pred = match_elements_optimal(
        gt_elements, pred_elements, canvas_size,
        apply_pred_alpha_cleaning=True,
        verbose=True,
        logger=logger,
    )

    # =========================================================================
    # [NEW] Compute Dual metrics for each matched pair
    # =========================================================================
    for match in matched_pairs:
        match["metrics_dual"] = compute_element_metrics_dual(
            match["gt"], match["preds"], canvas_size
        )
        
        # Keep the existing metrics field (backward compatibility - Union Region + Binary IoU)
        md = match["metrics_dual"]
        match["metrics"] = {
            "l1": md["visual_quality"]["union_region"]["l1"],
            "l2": md["visual_quality"]["union_region"]["l2"],
            "psnr": md["visual_quality"]["union_region"]["psnr"],
            "iou": md["iou"]["binary"],
        }

    # =========================================================================
    # [NEW] Compute Dual PQ
    # =========================================================================
    pq_dual = compute_panoptic_quality_dual(matched_pairs, unmatched_gt, unmatched_pred)
    
    # Existing PQ fields (backward compatibility - Binary IoU based)
    pq_metrics = {
        "pq_count": pq_dual["binary"]["pq"],
        "sq": pq_dual["binary"]["sq"],
        "rq": pq_dual["binary"]["rq"],
        "tp": pq_dual["binary"]["tp"],
        "fn": pq_dual["binary"]["fn"],
        "fp": pq_dual["binary"]["fp"],
    }
    
    # =========================================================================
    # [NEW] Dual Element aggregation
    # =========================================================================
    element_agg_dual = aggregate_element_metrics_dual(matched_pairs)

    # Existing element_metrics (backward compatibility)
    # If matched_pairs is empty, the aggregate result has no l1 etc., so access safely with .get
    _vq = element_agg_dual.get("visual_quality", {})
    _union_s = _vq.get("union_region", {}).get("simple_avg", {})
    _union_w = _vq.get("union_region", {}).get("weighted_avg", {})
    _iou_s = element_agg_dual.get("iou", {}).get("simple_avg", {})
    _iou_w = element_agg_dual.get("iou", {}).get("weighted_avg", {})
    element_agg = {
        "simple_avg": {
            "l1": _union_s.get("l1", float("nan")),
            "l2": _union_s.get("l2", float("nan")),
            "psnr": _union_s.get("psnr", float("nan")),
            "iou": _iou_s.get("binary", 0.0),
        },
        "weighted_avg": {
            "l1": _union_w.get("l1", float("nan")),
            "l2": _union_w.get("l2", float("nan")),
            "psnr": _union_w.get("psnr", float("nan")),
            "iou": _iou_w.get("binary", 0.0),
        },
    }
    
    # Composite metrics: skip if there are 5 or fewer non-text objects (parsing is insufficient, so the composite comes out abnormally high)
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

    # Save visualization (existing logic)
    if save_visualization:
         save_episode_visualization(
            episode_id, matched_pairs, unmatched_gt, unmatched_pred,
            gt_elements, pred_elements, canvas_size,
            composite_metrics, pq_metrics,
            output_dir, method_name,
            # [Fix] Added the arguments below to pass the computed Dual Metrics.
            element_metrics_dual=element_agg_dual,
            panoptic_quality_dual=pq_dual,
            gt_recon_img=gt_recon_img
        )
    
    # =========================================================================
    # Return results - including Dual metrics
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
        # Existing fields (backward compatibility)
        "element_metrics": element_agg,
        "panoptic_quality": pq_metrics,
        "composite_metrics": composite_metrics,
        "background_l1": background_l1,
        # [NEW] Dual metrics
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
    figma_data_dir = Path(args_dict['figma_data'])
    
    results = {"qwen": [], "agent": []}
    
    for idx, task in enumerate(tasks):
        episode_id = task["episode_id"]
        split_name = task["split"]
        gt_json_path = task["gt_json_path"]
        current_split_dir = task["split_dir"]
        
        # Figma: direct paths from task (supports paired exp dirs)
        qwen_dir = Path(task["qwen_dir"])
        agent_dir = Path(task["agent_dir"])
        
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
            # Extract GT elements (Figma: from JSON + split_dir)
            worker_logger.info(f"[Episode {episode_id}] === START ===")
            worker_logger.info(f"[Episode {episode_id}] Extracting GT elements...")
            gt_start = time.time()
            gt_elements, canvas_size, gt_recon_img = extract_gt_elements(
                gt_json_path, current_split_dir, logger=worker_logger
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
                "status": "done_frame",  # Change status from 'processing' to 'done'
                "qwen_res": qwen_res,    # Include Qwen result data
                "agent_res": agent_res   # Include Agent result data
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
    """Detailed logs are written only to files, so CMD output is ignored."""
    while not stop_event.is_set() or not log_queue.empty():
        try:
            message = log_queue.get(timeout=0.1)
            # tqdm.write(message)  <-- Commented out to turn off CMD output.
        except Empty:
            continue
        except Exception:
            continue


import textwrap # Import added at the top (for wrapping long ID lists)

def progress_monitor_process(progress_queue, stop_event, total_tasks, num_workers):
    import math
    pbar = tqdm(total=total_tasks, unit="frame", dynamic_ncols=True, position=0, leave=True)
    
    def init_stats():
        return {
            # Visual Quality: list-based collection (for NaN/inf-safe aggregation)
            "vq_inter": {"l1": [], "l2": [], "psnr": []},
            "vq_union": {"l1": [], "l2": [], "psnr": []},
            "vq_gt":    {"l1": [], "l2": [], "psnr": []},
            "lay_soft": {"iou": 0, "pq": 0, "sq": 0, "rq": 0},
            "lay_bin":  {"iou": 0, "pq": 0, "sq": 0, "rq": 0},
            "comp":     {"l1": [], "psnr": [], "ssim": 0, "lpips": 0, "dino": 0}
        }
    
    def _safe_mean(vals, is_psnr=False):
        """Average after removing NaN. For PSNR, replace inf with max_finite."""
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
                
                # Filter on Agent Composite L1 > 0.2
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
                    
                    # 3. Composite Metrics (L1, PSNR -> list, rest -> sum)
                    comp = res["composite_metrics"]
                    for met in ["l1", "psnr"]:
                        s["comp"][met].append(comp.get(met, 0))
                    for met in ["ssim", "lpips", "dino"]:
                        s["comp"][met] += comp.get(met, 0)

                pbar.update(1)

                # [Table output logic]
                v = counters["valid"]
                if v > 0:
                    q = stats["qwen"]; a = stats["agent"]
                    s_list = textwrap.fill(", ".join(skipped_ids_list), width=100, initial_indent="    ", subsequent_indent="    ")

                    # Visual quality: list → _safe_mean
                    def _vq(s, key, met):
                        return _safe_mean(s[key][met], is_psnr=(met == "psnr"))
                    # Composite: list for l1/psnr, sum/v for others
                    def _comp(s, met):
                        if met in ["l1", "psnr"]:
                            return _safe_mean(s["comp"][met], is_psnr=(met == "psnr"))
                        return s["comp"][met] / v

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
                        f"  LPIPS / DINO            | LP:{q['comp']['lpips']/v:.4f} DN:{q['comp']['dino']/v:.4f}              | LP:{a['comp']['lpips']/v:.4f} DN:{a['comp']['dino']/v:.4f}",
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
    parser = argparse.ArgumentParser(description="Parallel Element-Level Evaluation for Figma (Multi-GPU)")
    parser.add_argument("--figma-data", type=str, required=True,
                       help="Path to figma_data directory (containing process/subset/)")
    parser.add_argument("--exp-pairs", type=str, nargs="+", required=True,
                       help="One or more paired inference-output directories with a GT subset "
                            "prefix, each formatted as agent_dir:qwen_dir:gt_subset_prefix. "
                            "Use 'merged' as the prefix for the released merged dataset "
                            "(e.g. <AGENT_OUTPUT_DIR>:<QWEN_OUTPUT_DIR>:merged).")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    parser.add_argument("--bg-l1-threshold", type=float, default=BACKGROUND_L1_THRESHOLD)
    parser.add_argument("--matching", type=str, default="optimal", choices=["optimal", "legacy"])
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers (GPUs)")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5,6,7", 
                       help="Comma-separated GPU IDs to use")
    parser.add_argument("--text-refinement-mode", type=str, default="hybrid",
                       choices=["kill", "correct", "hybrid"],
                       help="Text refinement mode: kill (existing, favors Union), "
                            "correct (corrects RGB only, favors GT Region), "
                            "hybrid (compromise)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    num_workers = min(args.num_workers, len(gpu_ids))
    
    # Parse experiment pairs (agent_dir:qwen_dir:gt_subset_prefix)
    exp_pairs = []
    for pair_str in args.exp_pairs:
        parts = pair_str.split(":")
        if len(parts) != 3:
            print(f"[ERROR] Invalid exp-pair format: '{pair_str}'.")
            print(f"  Expected 'agent_dir:qwen_dir:gt_subset_prefix'")
            print(f"  Example: <AGENT_OUTPUT_DIR>:<QWEN_OUTPUT_DIR>:merged")
            return
        agent_dir, qwen_dir, gt_prefix = Path(parts[0]), Path(parts[1]), parts[2]
        exp_pairs.append((agent_dir, qwen_dir, gt_prefix))
    
    figma_data_dir = Path(args.figma_data)
    
    print("=" * 80)
    print(f"PARALLEL ELEMENT EVALUATION (FIGMA)")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Figma data: {args.figma_data}")
    print(f"Experiment pairs:")
    for i, (a, q, g) in enumerate(exp_pairs):
        print(f"  Pair {i}: agent={a}  qwen={q}  gt_prefix={g}")
    print(f"Matching algorithm: {args.matching}")
    print(f"Number of workers: {num_workers}")
    print(f"GPU IDs: {gpu_ids[:num_workers]}")
    print("=" * 80)
    
    # =========================================================================
    # Per-pair collection: GT + Agent + Qwen → intersect within pair → union across pairs
    # =========================================================================
    # episode_task_info: episode_id → {gt_json_path, split_dir, split_name, agent_dir, qwen_dir}
    episode_task_info = {}
    
    for pair_idx, (agent_exp_dir, qwen_exp_dir, gt_prefix) in enumerate(exp_pairs):
        print(f"\n--- Pair {pair_idx}: agent={agent_exp_dir.name}, qwen={qwen_exp_dir.name}, gt={gt_prefix} ---")
        
        # --- Auto-detect splits by scanning figma_data/process/subset/{gt_prefix}_split_* ---
        subset_base = figma_data_dir / "process" / "subset"
        gt_split_dirs = sorted(subset_base.glob(f"{gt_prefix}_split_*"))
        
        if not gt_split_dirs:
            print(f"  [Warning] No GT directories found matching '{gt_prefix}_split_*' in {subset_base}")
            continue
        
        detected_splits = []
        for d in gt_split_dirs:
            # Extract split name: "dino90_obj_5_25_char_50_split_0" → "split_0"
            suffix = d.name[len(gt_prefix) + 1:]  # +1 for the underscore separator
            detected_splits.append((suffix, d))
        
        print(f"  Detected splits: {[s for s, _ in detected_splits]}")
        
        # --- Collect GT episodes for this pair ---
        pair_gt = {}  # episode_id → {gt_json_path, split_name, split_dir}
        for split_name, split_dir in detected_splits:
            valid_frames_dir = split_dir / "valid_frames"
            if not valid_frames_dir.exists():
                print(f"  [Warning] {valid_frames_dir} not found. Skipping...")
                continue
            gt_json_files = list(valid_frames_dir.glob("*.json"))
            for p in gt_json_files:
                pair_gt[p.stem] = {
                    "gt_json_path": p,
                    "split_name": split_name,
                    "split_dir": split_dir,
                }
            print(f"  GT {split_name}: {len(gt_json_files)} frames")
        print(f"  GT total for pair: {len(pair_gt)}")
        
        # --- Scan Qwen episodes ---
        # Supports both split_X/{episode}/ and all_splits/{episode}/ structures
        qwen_found = {}  # episode_id → full path to qwen episode dir
        if qwen_exp_dir.exists():
            for subdir in sorted(qwen_exp_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                for ep_dir in subdir.iterdir():
                    if ep_dir.is_dir() and (ep_dir / "layer_00.png").exists():
                        qwen_found[ep_dir.name] = ep_dir
        print(f"  Qwen ({qwen_exp_dir.name}): {len(qwen_found)} episodes")
        
        # --- Scan Agent episodes ---
        # Agent structure: {split}/episodes/{episode_id}/
        agent_found = {}  # episode_id → full path to agent episode dir
        if agent_exp_dir.exists():
            for subdir in sorted(agent_exp_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                episodes_dir = subdir / "episodes"
                if not episodes_dir.exists():
                    continue
                for ep_dir in episodes_dir.iterdir():
                    if ep_dir.is_dir() and (ep_dir / "parse.json").exists():
                        agent_found[ep_dir.name] = ep_dir
        print(f"  Agent ({agent_exp_dir.name}): {len(agent_found)} episodes")
        
        # --- Intersect within this pair: GT ∩ Agent ∩ Qwen ---
        pair_common = set(pair_gt.keys()) & set(agent_found.keys()) & set(qwen_found.keys())
        new_count = 0
        for eid in pair_common:
            if eid not in episode_task_info:  # first pair wins (no override)
                gt_info = pair_gt[eid]
                episode_task_info[eid] = {
                    "gt_json_path": gt_info["gt_json_path"],
                    "split_name": gt_info["split_name"],
                    "split_dir": gt_info["split_dir"],
                    "agent_dir": agent_found[eid],
                    "qwen_dir": qwen_found[eid],
                }
                new_count += 1
        print(f"  Pair {pair_idx} common (GT∩Agent∩Qwen): {len(pair_common)}, new: {new_count}")
    
    print(f"\n{'='*80}")
    print(f"Total episodes across all pairs (union): {len(episode_task_info)}")
    
    # ---- Build tasks with resume support ----
    all_valid_episode_tasks = []
    skipped_count = 0
    
    for eid in sorted(episode_task_info.keys()):
        episode_out_dir = output_dir / eid
        agent_done = (episode_out_dir / "agent" / "metrics.json").exists()
        qwen_done = (episode_out_dir / "qwen" / "metrics.json").exists()
        
        if agent_done and qwen_done:
            skipped_count += 1
            continue
        
        info = episode_task_info[eid]
        all_valid_episode_tasks.append({
            "episode_id": eid,
            "split": info["split_name"],
            "gt_json_path": info["gt_json_path"],
            "split_dir": info["split_dir"],
            "agent_dir": str(info["agent_dir"]),
            "qwen_dir": str(info["qwen_dir"]),
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
        "figma_data": args.figma_data,
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
            "exp_pairs": args.exp_pairs,
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