# baselines/tool_backends/tools/ocr_tool.py
"""
OCR - Persistent subprocess-isolated PaddleOCR

Runs OCR in a separate persistent subprocess to protect the main process
from PaddlePaddle SIGSEGV / CUDA context corruption.
GPU access is fully blocked via CUDA_VISIBLE_DEVICES="".
"""
import json
import os
import subprocess
import sys
import threading
import select
import cv2
from pathlib import Path
from typing import Dict, Optional

_MAX_RETRIES = 2
_SUBPROCESS_TIMEOUT = 60

_OCR_WORKER_SCRIPT = r'''
import sys, json, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

def init_ocr():
    return PaddleOCR(
        text_detection_model_name='PP-OCRv5_server_det',
        text_det_box_thresh=0.3,
        text_det_unclip_ratio=2.0,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_angle_cls=False,
        device='cpu',
    )

def run_ocr(ocr, image_path):
    img = Image.open(image_path)
    img_np = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    results = ocr.ocr(img_bgr)
    rec_boxes = results[0].get('rec_polys', [])
    rec_texts = results[0].get('rec_texts', [])
    rec_scores = results[0].get('rec_scores', [])
    items = []
    for idx, (box, text, score) in enumerate(zip(rec_boxes, rec_texts, rec_scores)):
        try:
            box_py = np.asarray(box).tolist()
        except Exception:
            try:
                box_py = list(box)
            except Exception:
                box_py = []
        items.append({"id": idx, "text": text, "box": box_py, "score": float(score)})
    return items

def main():
    ocr = init_ocr()
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    for line in sys.stdin:
        image_path = line.strip()
        if not image_path or image_path == "EXIT":
            break
        try:
            items = run_ocr(ocr, image_path)
            response = {"status": "ok", "items": items}
        except Exception as e:
            response = {"status": "error", "message": str(e)[:500]}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
'''


class _OCRWorker:
    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _start(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _OCR_WORKER_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        ready_line = self._proc.stdout.readline().strip()
        if ready_line != "READY":
            raise RuntimeError(f"OCR worker failed to start: {ready_line}")
        print("[OCR] Persistent worker subprocess started")

    def _kill(self):
        if self._proc is not None:
            try:
                self._proc.stdin.write("EXIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=3)
                except Exception:
                    pass
            self._proc = None

    def _is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def run(self, image_path: str) -> list:
        with self._lock:
            if not self._is_alive():
                self._start()
            try:
                self._proc.stdin.write(image_path + "\n")
                self._proc.stdin.flush()
                rlist, _, _ = select.select([self._proc.stdout], [], [], _SUBPROCESS_TIMEOUT)
                if not rlist:
                    print(f"[OCR] Worker timeout ({_SUBPROCESS_TIMEOUT}s), killing...")
                    self._kill()
                    raise TimeoutError(f"OCR worker timed out after {_SUBPROCESS_TIMEOUT}s")
                response_line = self._proc.stdout.readline().strip()
                if not response_line:
                    stderr = ""
                    try:
                        stderr = self._proc.stderr.read()
                    except Exception:
                        pass
                    self._kill()
                    raise RuntimeError(f"OCR worker died unexpectedly:\n{stderr[-300:]}")
                response = json.loads(response_line)
                if response["status"] == "error":
                    raise RuntimeError(f"OCR error: {response['message']}")
                return response.get("items", [])
            except (BrokenPipeError, OSError) as e:
                stderr = ""
                try:
                    stderr = self._proc.stderr.read()
                except Exception:
                    pass
                self._kill()
                raise RuntimeError(f"OCR worker crashed: {e}\n{stderr[-300:]}")

    def shutdown(self):
        with self._lock:
            self._kill()


_worker: Optional[_OCRWorker] = None
_worker_init_lock = threading.Lock()

def _get_worker() -> _OCRWorker:
    global _worker
    if _worker is None:
        with _worker_init_lock:
            if _worker is None:
                _worker = _OCRWorker()
    return _worker


def _quad_to_aabb(b):
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
        return [int(x) for x in b]
    xs = [p[0] for p in b]
    ys = [p[1] for p in b]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def unload_ocr():
    """Terminate the worker subprocess"""
    global _worker
    if _worker is not None:
        _worker.shutdown()
        _worker = None


def run_ocr(image_path: str, vis_dir: Path = None, step: int = 0) -> Dict:
    worker = _get_worker()
    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            items = worker.run(str(image_path))
            break
        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                print(f"[OCR] Worker failed (attempt {attempt + 1}/{_MAX_RETRIES + 1}): {str(e)[:150]}")
                continue
            raise
    else:
        raise last_error

    boxes, texts, scores = [], [], []
    for it in items:
        if it.get("score", 0) > 0.5:
            boxes.append(it["box"])
            texts.append(it["text"])
            scores.append(it["score"])

    viz_path = None
    if boxes:
        img = cv2.imread(str(image_path))
        if img is not None:
            for rbox, txt, sc in zip(boxes, texts, scores):
                x1, y1, x2, y2 = _quad_to_aabb(rbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, txt, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, f"{sc:.2f}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            if vis_dir:
                viz_path = vis_dir / f"{step:03d}_OCR_det.png"
                cv2.imwrite(str(viz_path), img)

    out = {"boxes": boxes, "texts": texts, "scores": scores}
    if viz_path:
        out["viz"] = str(viz_path)
    return out
