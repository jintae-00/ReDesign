import cv2
import numpy as np
import PIL.Image as Image

from paddleocr import PaddleOCR


class PaddleOCRClient:
    def __init__(self, mask_extend_size=10):
        self.ocr = PaddleOCR(
            text_detection_model_name='PP-OCRv5_server_det',           # 또는 'SAST'
            text_det_box_thresh=0.3,
            text_det_unclip_ratio=2.0,
            use_doc_orientation_classify=False,   # 방향(90°·180°) 보정 끔
            use_doc_unwarping=False,              # 문서 펴기(Unwarp) 끔
            use_angle_cls=False,
            device='cpu',                         # CPU로 실행 — PaddlePaddle GPU 메모리 풀 할당 방지
        )
        self.mask_extend_size = mask_extend_size

    def run_ocr(self, image: Image.Image):
        img_np = np.array(image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        results = self.ocr.ocr(img_bgr)  # dict 반환

        rec_boxes = results[0].get('rec_polys', [])
        rec_texts = results[0].get('rec_texts', [])
        rec_scores = results[0].get('rec_scores', [])
    
        ocr_items = []
        for idx, (box, text, score) in enumerate(zip(rec_boxes, rec_texts, rec_scores)):
            try:
                box_py = np.asarray(box).tolist()
            except Exception:
                try:
                    box_py = list(box)
                except Exception:
                    box_py = []
            ocr_items.append({
                "id": idx,
                "text": text,
                "box": box_py,
                "score": score
            })
       
        return ocr_items
