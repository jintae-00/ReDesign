import cv2
import numpy as np
import PIL.Image as Image

from paddleocr import PaddleOCR


class PaddleOCRClient:
    def __init__(self, mask_extend_size=10):
        self.ocr = PaddleOCR(
            text_detection_model_name='PP-OCRv5_server_det',           # or 'SAST'
            text_det_box_thresh=0.3,
            text_det_unclip_ratio=2.0,
            use_doc_orientation_classify=False,   # disable orientation (90°/180°) correction
            use_doc_unwarping=False,              # disable document unwarping
            use_angle_cls=False,
            device='cpu',                         # run on CPU — avoid PaddlePaddle GPU memory pool allocation
        )
        self.mask_extend_size = mask_extend_size

    def run_ocr(self, image: Image.Image):
        img_np = np.array(image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        results = self.ocr.ocr(img_bgr)  # returns dict

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
