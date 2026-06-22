# REDESIGN/tools/dino_tool.py
"""
Grounding DINO - Thread-safe with ToolGPUManager

[수정 사항]
1. 클로저 late binding 문제 해결 (함수 팩토리 사용)
2. retry_helper 모듈 사용으로 통합 재시도 로직 적용
"""
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Dict
from contextlib import nullcontext
from torchvision.ops import box_convert

from ..tool_gpu_manager import get_tool_manager, ToolModelType
from .retry_helper import retry_on_cuda_error, aggressive_memory_cleanup

import modules.grounding_dino.groundingdino.datasets.transforms as T
from modules.grounding_dino.groundingdino.util.inference import predict_filtered_boxes_matching

_transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _to_xyxy_norm_to_abs(boxes_cxcywh_norm: torch.Tensor, w: int, h: int):
    scale = torch.tensor([w, h, w, h], device=boxes_cxcywh_norm.device)
    return box_convert(boxes_cxcywh_norm * scale, "cxcywh", "xyxy").cpu().numpy()


def _draw_boxes(image_path: str, boxes_xyxy_list, labels, confs, vis_path: Path):
    img = cv2.imread(image_path)
    for (x1, y1, x2, y2), lab, cf in zip(boxes_xyxy_list, labels, confs):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{lab} ({cf:.2f})", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(vis_path), img)


def _apply_containment_filter(boxes_xyxy, confs, labels, thresh):
    if not boxes_xyxy or len(boxes_xyxy) <= 1:
        return boxes_xyxy, confs, labels
    
    boxes_np = np.array(boxes_xyxy, dtype=float)
    areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
    keep_mask = np.ones(len(boxes_np), dtype=bool)
    
    for i in range(len(boxes_np)):
        if not keep_mask[i]:
            continue
        for j in range(len(boxes_np)):
            if i == j or not keep_mask[j] or areas[i] <= 0 or areas[j] <= 0:
                continue
            x1, y1 = max(boxes_np[i][0], boxes_np[j][0]), max(boxes_np[i][1], boxes_np[j][1])
            x2, y2 = min(boxes_np[i][2], boxes_np[j][2]), min(boxes_np[i][3], boxes_np[j][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if (areas[i] > areas[j]) and (inter / (areas[j] + 1e-6) >= thresh):
                keep_mask[i] = False
                break
    
    return (
        [b for k, b in zip(keep_mask, boxes_xyxy) if k],
        [c for k, c in zip(keep_mask, confs) if k] if confs else None,
        [l for k, l in zip(keep_mask, labels) if k],
    )


def _create_dino_inference_fn(dino, img_cv, lab, score_min, area_max, device):
    def run_inference():
        # 모델의 dtype에 맞춰 입력 변환
        model_dtype = next(dino.parameters()).dtype
        img_input = img_cv.to(device=device, dtype=model_dtype)
        
        with torch.autocast(device_type="cuda", enabled=False):
            return predict_filtered_boxes_matching(
                model=dino, image=img_input, caption=lab,
                score_min_ratio=score_min, area_max_ratio=area_max, device=device
            )
    return run_inference


@torch.no_grad()
def run_dino_batch_all(
    image_path: str,
    labels: List[str],
    vis_dir: Path = None,
    step: int = 0,
    score_min: float = 0.1,
    area_max: float = 0.8,
    top_k_per_label: int = 1,
    drop_containers: bool = True,
    contain_ratio_thresh: float = 0.9,
    caller_id: str = None,
) -> Dict:
    """
    Thread-safe GDINO with unified CUDA error recovery
    """
    
    manager = get_tool_manager()
    
    with manager.acquire(ToolModelType.GDINO, caller_id=caller_id) as ctx:
        dino = ctx.model
        device = ctx.device
        gpu_id = ctx.gpu_id
        stream = ctx.stream
        
        stream_context = torch.cuda.stream(stream) if stream else nullcontext()
        
        with stream_context:
            pil = Image.open(image_path).convert("RGB")
            w, h = pil.size
            img_cv, _ = _transform(pil, None)
            img_cv = img_cv.float()

            all_boxes_xyxy, all_confs, all_labels = [], [], []
            labels = [labels] if isinstance(labels, str) else (labels or [])
            
            for lab in labels:
                # ★ 클로저 문제 해결: 함수 팩토리 사용
                inference_fn = _create_dino_inference_fn(
                    dino, img_cv, lab, score_min, area_max, device
                )
                
                boxes, scores, _ = retry_on_cuda_error(
                    func=inference_fn,
                    gpu_id=gpu_id,
                    model_name=f"GDINO:{lab}",
                    max_retries=3,
                    base_delay=0.5,
                )
                
                if boxes is None or scores is None or boxes.numel() == 0:
                    continue
                
                confs_np = np.atleast_1d(scores.detach().cpu().numpy())
                if confs_np.size == 0:
                    continue
                
                order = np.argsort(confs_np)[::-1].copy()
                k = min(top_k_per_label, confs_np.size)
                keep_idx = order[:k]
                keep = torch.as_tensor(keep_idx, dtype=torch.long, device=boxes.device)
                
                boxes_sel = boxes[keep]
                if boxes_sel.numel() == 0:
                    continue
                
                all_boxes_xyxy.extend(_to_xyxy_norm_to_abs(boxes_sel, w, h).tolist())
                all_confs.extend([float(confs_np[i]) for i in keep_idx])
                all_labels.extend([lab] * len(keep_idx))
            
            if stream:
                stream.synchronize()
        
        if drop_containers and len(all_boxes_xyxy) > 1:
            all_boxes_xyxy, all_confs, all_labels = _apply_containment_filter(
                all_boxes_xyxy, all_confs, all_labels, contain_ratio_thresh
            )
        
        vis_path = vis_dir / f"{step:03d}_DINO_det.png" if vis_dir else Path("det.png")
        if all_boxes_xyxy:
            _draw_boxes(image_path, all_boxes_xyxy, all_labels, all_confs, vis_path)
    
    return {"boxes": all_boxes_xyxy, "confs": all_confs, "labels": all_labels, "viz": str(vis_path)}