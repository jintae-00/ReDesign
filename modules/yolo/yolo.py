import numpy as np

from ultralytics import YOLO
from PIL import Image
from modules.yolo.config.config_layersplit import LAYOUT_CONF


class Yolo_Client:
    def __init__(self, model_path):
        self.yolo_model = YOLO(model_path)

    def execute(self, input_image: Image.Image):
        img_w, img_h = input_image.size
        result = self.yolo_model(
            [input_image],
            iou=LAYOUT_CONF.iou,
            conf=LAYOUT_CONF.conf,   # ← YOLO 내부 1차 conf threshold (그대로 둠)
            device=[0],
        )[0]

        # YOLO 결과에서 bbox / cls / conf 추출
        # result.boxes.xyxy: [N,4], result.boxes.cls: [N], result.boxes.conf: [N]
        bbox_list = result.boxes.xyxy.cpu().numpy().astype(np.int32).tolist()
        bbox_class_list = (
            result.boxes.cls.cpu().numpy().astype(np.int32).tolist()
            if result.boxes.cls is not None
            else [0] * len(bbox_list)
        )
        bbox_conf_list = (
            result.boxes.conf.cpu().numpy().astype(np.float32).tolist()
            if result.boxes.conf is not None
            else [1.0] * len(bbox_list)
        )

        res_img = Image.fromarray(result.plot()[..., [2, 1, 0]])

        filtered_bbox = []
        filtered_bbox_class = []
        filtered_conf = []

        # 기존 layout_filter(너무 크거나/작은 박스 제거)는 그대로 둠
        for box, cls, conf in zip(bbox_list, bbox_class_list, bbox_conf_list):
            w = box[2] - box[0]
            h = box[3] - box[1]
            area = w * h

            if area > (LAYOUT_CONF.layout_filter.max_region_ratio ** 2) * img_w * img_h:
                continue
            if area < LAYOUT_CONF.layout_filter.min_region_pix:
                continue

            filtered_bbox.append(box)
            filtered_bbox_class.append(cls)
            filtered_conf.append(conf)

        # ⚠ Yolo_Client는 "필터링된 결과 + conf"만 반환, conf threshold는 yolo_tool에서 수행
        return filtered_bbox, filtered_bbox_class, filtered_conf, res_img
