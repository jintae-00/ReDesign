# BASELINES/tool_backends/tools/yolo_tool.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch, gc

from modules.yolo.yolo import Yolo_Client
from config import WEIGHTS

# singleton cache
_yolo: Optional[Yolo_Client] = None

def _get_yolo(model_path: Optional[str] = None) -> Yolo_Client:
    global _yolo
    if _yolo is None:
        ckpt = model_path or str(WEIGHTS / "yolov11.pt")
        _yolo = Yolo_Client(model_path=ckpt)
    return _yolo

def _draw_boxes(image_path: str,
                boxes_xyxy_list: List[List[int]] | List[List[float]],
                labels: List[str],
                confs: Optional[List[float]],
                vis_path: Path):
    """
    Same style as DINO:
      - green bbox
      - text: "{label} ({score:.2f})"
    If confs is None, the score is omitted.
    """
    img = cv2.imread(image_path)
    if img is None:
        return

    confs = confs or [None] * len(boxes_xyxy_list)

    for (x1, y1, x2, y2), lab, cf in zip(boxes_xyxy_list, labels, confs):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if lab is not None:
            if cf is not None:
                txt = f"{lab} ({cf:.2f})"
            else:
                txt = str(lab)
            cv2.putText(
                img,
                txt,
                (x1, max(10, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
    cv2.imwrite(str(vis_path), img)


def _apply_containment_filter(
    boxes: List[List[float]],
    confs: Optional[List[float]],
    labels: List[Any],
    contain_ratio_thresh: float,
):
    """
    If a larger box contains a smaller box by at least contain_ratio, remove the larger box.
    contain_ratio = area(intersection(A,B)) / area(B)
    """
    if not boxes or len(boxes) <= 1:
        return boxes, confs, labels

    boxes_np = np.array(boxes, dtype=float)
    areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
    keep_mask = np.ones(len(boxes_np), dtype=bool)

    def _inter_area(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
        return w * h

    N = len(boxes_np)
    for i in range(N):
        if not keep_mask[i]:
            continue
        for j in range(N):
            if i == j or not keep_mask[j]:
                continue
            if areas[i] <= 0 or areas[j] <= 0:
                continue

            inter = _inter_area(boxes_np[i], boxes_np[j])
            contain_ratio = inter / (areas[j] + 1e-6)

            # if i is larger than j and sufficiently contains j, remove i
            if (areas[i] > areas[j]) and (contain_ratio >= contain_ratio_thresh):
                keep_mask[i] = False
                break

    boxes = [b for k, b in zip(keep_mask, boxes) if k]
    if confs is not None:
        confs = [c for k, c in zip(keep_mask, confs) if k]
    labels = [l for k, l in zip(keep_mask, labels) if k]
    return boxes, confs, labels


@torch.no_grad()
def run_yolo(image_path: str,
             vis_dir: Optional[Path] = None,
             step: int = 0,
             model_path: Optional[str] = None,
             *,
             conf_thresh: float = 0.85,
             drop_containers: bool = True,
             contain_ratio_thresh: float = 0.9) -> Dict[str, Any]:
    """
    General-purpose YOLO-based object/picture detection.
    returns:
      {
        "boxes":  [[x1,y1,x2,y2], ...],   # final filtered bboxes
        "confs":  [float, ...] or None,   # scores corresponding to the final bboxes
        "labels": [str, ...],             # human-readable label
        "viz":    "<path/to/viz.png>",    # viz always redrawn to match 'boxes'
      }
    """
    yolo = _get_yolo(model_path)
    pil = Image.open(image_path).convert("RGB")
    out = yolo.execute(pil)  # (bboxes, classes, confs, yolo_vis) or None
    bboxes, classes, confs, _yolo_vis = out

    vis_path = (vis_dir / f"{step:03d}_YOLO_det.png") if vis_dir else Path("YOLO_det.png")

    # no YOLO bboxes -> save viz with empty boxes (= original)
    if len(bboxes) == 0:
        _draw_boxes(image_path, [], [], [], vis_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {"boxes": [], "confs": [], "labels": [], "viz": str(vis_path)}
    

    # ===============================
    # 1) apply confidence threshold
    # ===============================
    confs_np = np.asarray(confs, dtype=float)
    bboxes_np = np.asarray(bboxes, dtype=float)
    classes_np = np.asarray(classes)

    keep_mask = confs_np >= conf_thresh

    bboxes_np = bboxes_np[keep_mask]
    classes_np = classes_np[keep_mask]
    confs_np = confs_np[keep_mask]

    bboxes = bboxes_np.tolist()
    classes = classes_np.tolist()
    confs = confs_np.tolist()

    # if no boxes remain after thresholding: viz should contain no bbox
    if len(bboxes) == 0:
        _draw_boxes(image_path, [], [], [], vis_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {"boxes": [], "confs": [], "labels": [], "viz": str(vis_path)}

    # ===============================
    # 2) apply Containment filter
    # ===============================
    if drop_containers and len(bboxes) > 1:
        bboxes, confs, classes = _apply_containment_filter(
            bboxes,
            confs,
            classes,
            contain_ratio_thresh=contain_ratio_thresh,
        )

    # there may be no boxes left after containment filtering
    if len(bboxes) == 0:
        _draw_boxes(image_path, [], [], [], vis_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {"boxes": [], "confs": [], "labels": [], "viz": str(vis_path)}

    # ===============================
    # 3) label mapping + visualization + return
    # ===============================
    boxes = [list(map(int, b)) for b in (bboxes or [])]
    label_map = {0: "object", 1: "picture"}
    labs = [label_map.get(int(c), str(int(c))) for c in (classes or [])]
    
    _draw_boxes(image_path, boxes, labs, confs, vis_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "boxes": boxes,        # 1:1 correspondence with the bboxes drawn in viz
        "confs": confs,
        "labels": labs,
        "viz": str(vis_path),
    }
