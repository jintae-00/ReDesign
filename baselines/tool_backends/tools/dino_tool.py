import torch, numpy as np, gc
from PIL import Image
import cv2
from pathlib import Path
from torchvision.ops import box_convert

_dino = None
_transform = None

def _get_dino():
    global _dino, _transform
    if _dino is None:
        import modules.grounding_dino.groundingdino.datasets.transforms as T
        from modules.grounding_dino.groundingdino.util.inference import load_model
        from config import MODULES, WEIGHTS, DEVICE
        CFG  = MODULES/"grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
        CKPT = WEIGHTS/"groundingdino_swinb_cogcoor.pth"
        _dino = load_model(str(CFG), str(CKPT), device=DEVICE)
        _transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return _dino

def _get_transform():
    if _transform is None:
        _get_dino()  # initializes both
    return _transform

def unload_dino():
    global _dino, _transform
    if _dino is not None:
        try:
            _dino.cpu()
        except Exception:
            pass
        del _dino
        _dino = None
        _transform = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def _to_xyxy_norm_to_abs(boxes_cxcywh_norm: torch.Tensor, w: int, h: int):
    scale = torch.tensor([w, h, w, h], device=boxes_cxcywh_norm.device)
    boxes_xyxy = box_convert(boxes_cxcywh_norm * scale, "cxcywh", "xyxy").cpu().numpy()
    return boxes_xyxy

def _draw_boxes(image_path: str, boxes_xyxy_list, labels, confs, vis_path: Path):
    img = cv2.imread(image_path)
    for (x1, y1, x2, y2), lab, cf in zip(boxes_xyxy_list, labels, confs):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        txt = f"{lab} ({cf:.2f})"
        cv2.putText(img, txt, (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imwrite(str(vis_path), img)


def _apply_containment_filter(
    boxes_xyxy: list[list[float]],
    confs: list[float] | None,
    labels: list[str],
    contain_ratio_thresh: float,
):
    """
    If a larger box A contains a smaller box B by at least contain_ratio, remove A.
    contain_ratio = area(intersection(A,B)) / area(B)
    """
    if not boxes_xyxy or len(boxes_xyxy) <= 1:
        return boxes_xyxy, confs, labels

    boxes_np = np.array(boxes_xyxy, dtype=float)  # [N,4]: x1,y1,x2,y2
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

            # A=boxes[i], B=boxes[j]
            if (areas[i] > areas[j]) and (contain_ratio >= contain_ratio_thresh):
                keep_mask[i] = False
                break

    boxes_xyxy = [b for k, b in zip(keep_mask, boxes_xyxy) if k]
    if confs is not None:
        confs = [c for k, c in zip(keep_mask, confs) if k]
    labels = [l for k, l in zip(keep_mask, labels) if k]
    return boxes_xyxy, confs, labels



@torch.no_grad()
def run_dino_batch_all(image_path: str,
                       labels: list[str],
                       vis_dir: Path=None, step: int = 0,
                       score_min: float = 0.1, area_max: float = 0.8,
                       top_k_per_label: int = 1,
                       *,
                       drop_containers: bool = True,
                       contain_ratio_thresh: float = 0.9  # if A contains B by at least this ratio, remove A
                       ):
    """
    Repeatedly calls `predict_filtered_boxes_matching` for each label and
    returns all boxes that pass score_min_ratio(=score_min).

    Currently:
      - area_max_ratio based area filter: disabled inside `predict_filtered_boxes_matching`
      - top-k selection per label: disabled (multiple boxes are allowed simultaneously)
      - Containment Filter: disabled (drop_containers is a placeholder for future re-activation)
    """
    

    from modules.grounding_dino.groundingdino.util.inference import predict_filtered_boxes_matching
    from config import DEVICE

    model = _get_dino()
    transform = _get_transform()

    pil = Image.open(image_path).convert("RGB")
    w, h = pil.size
    img_cv, _ = transform(pil, None)

    all_boxes_xyxy = []
    all_confs = []
    all_labels = []

    labels = [labels] if isinstance(labels, str) else (labels or [])

    for lab in labels:
        boxes, scores, phrases = predict_filtered_boxes_matching(
            model=model, image=img_cv, caption=lab,
            score_min_ratio=score_min, area_max_ratio=area_max, device=DEVICE
        )
        # Guard: no results / empty arrays / scalars, etc.
        if boxes is None or scores is None or boxes.numel() == 0 or scores.numel() == 0:
            continue

        confs_np = scores.detach().cpu().numpy()
        confs_np = np.atleast_1d(confs_np)
        if confs_np.size == 0:
            continue

        # (3) enable top-k per label
        # top-k indices (at least 1, at most size)
        order = np.argsort(confs_np)[::-1].copy()
        k = int(min(top_k_per_label, confs_np.size))
        keep_idx = order[:k]
        keep = torch.as_tensor(keep_idx, dtype=torch.long, device=boxes.device)

        boxes_sel = boxes[keep]  # cxcywh (norm)
        # defensive size check
        if boxes_sel.numel() == 0:
            continue

        boxes_xyxy = _to_xyxy_norm_to_abs(boxes_sel, w, h).tolist()
        confs_sel = [float(confs_np[i]) for i in keep_idx]
        labels_sel = [lab] * len(keep_idx)

        all_boxes_xyxy.extend(boxes_xyxy)
        all_confs.extend(confs_sel)
        all_labels.extend(labels_sel)

        del boxes, scores, phrases, confs_np, keep, boxes_sel

    # release the img_cv tensor and the PIL image
    del pil, img_cv

    # ===== Containment Filter =====
    if drop_containers and len(all_boxes_xyxy) > 1:
        all_boxes_xyxy, all_confs, all_labels = _apply_containment_filter(
            all_boxes_xyxy,
            all_confs,
            all_labels,
            contain_ratio_thresh=contain_ratio_thresh,
        )
        

    # ===== Visualization & return =====
    vis_path = vis_dir / f"{step:03d}_DINO_det.png" if vis_dir else Path("det.png")
    _draw_boxes(image_path, all_boxes_xyxy, all_labels, all_confs, vis_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {"boxes": all_boxes_xyxy, "confs": all_confs, "labels": all_labels, "viz": str(vis_path)}
