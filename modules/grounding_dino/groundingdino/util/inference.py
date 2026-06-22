from typing import Tuple, List

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert
import bisect

import grounding_dino.groundingdino.datasets.transforms as T
from grounding_dino.groundingdino.models import build_model
from grounding_dino.groundingdino.util.misc import clean_state_dict
from grounding_dino.groundingdino.util.slconfig import SLConfig
from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap

# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    print(f"gdino / gdino / util / inference / predict / caption : {caption}")

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

    return boxes, logits.max(dim=1)[0], phrases

def predict_whole_caption_matching(
        model,
        image: torch.Tensor,
        caption: str,
        text_threshold: float,
        box_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (num_queries, vocab_size)
    prediction_boxes  = outputs["pred_boxes"].cpu()[0]              # (num_queries, 4)

    # Tokenize caption and remove special tokens
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    input_ids = tokenized['input_ids']
    caption_token_indices = [
        i for i, tid in enumerate(input_ids)
        if tid not in [101, 102, 1012]  # [CLS], [SEP], '.'
    ]

    # Score each query by averaging over the caption tokens
    query_scores = []
    for logit in prediction_logits:
        score = logit[caption_token_indices].mean()
        query_scores.append(score)

    # Select the best matching query (top-1)
    best_idx = torch.tensor(query_scores).argmax()
    best_score = torch.tensor([query_scores[best_idx]])
    best_phrase = caption.rstrip('.')
    best_box = prediction_boxes[best_idx].unsqueeze(0)
    return best_box, [best_score], [best_phrase]

def predict_filtered_boxes_matching(
        model,
        image: torch.Tensor,
        caption: str,
        score_min_ratio,
        area_max_ratio,      # ← 이 값보다 큰 박스는 버림
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    • score_min_ratio 이상인 박스만 반환 (현재 area_max_ratio 는 사용하지 않음)
    • boxes : (N,4) cxcywh, 0~1 정규화
    • scores: (N,)  평균 유사도
    • phrases: N개, 모두 caption.rstrip('.')
    """
    caption = preprocess_caption(caption=caption)

    model, image = model.to(device), image.to(device)

    # 2) 모델 추론
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    pred_logits = outputs["pred_logits"].cpu().sigmoid()[0]   # (Q,V)
    pred_boxes  = outputs["pred_boxes"].cpu()[0]              # (Q,4) cxcywh

    # 3) 캡션 토큰 인덱스
    ids = model.tokenizer(caption)['input_ids']
    cap_idx = [i for i, tid in enumerate(ids) if tid not in (101, 102, 1012)]

    
    # 4) 평균 유사도 (for-loop → 벡터화: 값은 동일)
    query_scores = pred_logits[:, cap_idx].mean(dim=1)        # (Q,)

    '''
    # 5) 텍스트 필터
    keep_txt = query_scores >= score_min_ratio

    # 6) 면적 계산 후 “너무 큰” 박스 제거
    #    - 변환은 필터링용으로만 잠깐 xyxy 로
    boxes_xyxy = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    wh = boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]
    area_ratio = wh[:, 0] * wh[:, 1]
    keep_area = area_ratio <= area_max_ratio    # area_max_ratio = 최대 허용 면적 비율

    keep = (keep_txt & keep_area).nonzero(as_tuple=True)[0]
    '''

    # 5) 텍스트 기반 필터만 적용
    #    - area_max_ratio 를 이용한 면적 필터는 현재 비활성화
    keep = (query_scores >= score_min_ratio).nonzero(as_tuple=True)[0]
    
    if keep.numel() == 0:
        return torch.empty((0,4)), torch.empty((0,)), []

    # 7) 점수 내림차순 정렬
    boxes_kept  = pred_boxes[keep]             # ✅ cxcywh 그대로 유지
    scores_kept = query_scores[keep]
    order = scores_kept.argsort(descending=True)

    boxes_sorted  = boxes_kept[order]
    scores_sorted = scores_kept[order]
    phrases       = [caption.rstrip('.')] * len(scores_sorted)

    return boxes_sorted, scores_sorted, phrases


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold, 
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float,
        text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)
