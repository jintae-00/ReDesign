# `modules/` — Tool backends

Third-party model code that powers the ReDesign agent's tools. Only **source code**
is bundled here; **checkpoints are downloaded separately** with
`python scripts/download_checkpoints.py` (into `../weights/` and the HuggingFace
cache). Each module retains its upstream license.

| Module | Agent tool | Role | Checkpoint(s) | Upstream / license |
|---|---|---|---|---|
| `grounding_dino` | `detect_gdino` | Open-vocabulary object detection | `groundingdino_swinb_cogcoor.pth` | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) (Apache-2.0) |
| `sam2` | `seg_sam2_bbox` | Promptable segmentation (SAM 2.1) | `sam2.1_hiera_large.pt` + `pip install sam2` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) (Apache-2.0) |
| `hisam` | `seg_hisam` | Hierarchical text segmentation | `sam_tss_h_textseg.pth` | [ymy-k/Hi-SAM](https://github.com/ymy-k/Hi-SAM) (Apache-2.0) |
| `textremover` | `inpaint_lama` | LaMa text/object removal (inpaint) | `big-lama.pt` | [advimman/lama](https://github.com/advimman/lama) (Apache-2.0) |
| `ObjectClear` | `inpaint_oc` | Object-aware inpainting | `jixin0101/ObjectClear` (HF) | [ObjectClear](https://huggingface.co/jixin0101/ObjectClear) |
| `ocr` | `detect_ocr` | Text detection/recognition | PaddleOCR PP-OCRv5 (auto) | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0) |
| `layerd` | layered decomposition | Front-layer decomposition | — | LayerDecomposition |
| `yolo` | (optional) | Object detection (YOLO11) | `yolov11.pt` | [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) |

The layered-image generation tool (`qwen_layered`) uses
**`Qwen/Qwen-Image-Layered`** via `from diffusers import QwenImageLayeredPipeline`
(downloaded from the HuggingFace Hub; see `scripts/download_checkpoints.py
--with-qwen`).

> `grounding_dino` ships a CUDA extension that is compiled by `post_install.sh`
> (`pip install -e modules/grounding_dino`). Without a CUDA toolkit it falls back
> to a slower pure-Python path.

> Excluded from this release: `asuka-flux` (not part of the public agent).
