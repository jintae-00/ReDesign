# `modules/` — Tool backends

Third-party model code that powers the ReDesign agent's tool actions. Only
**source code** is bundled here; **checkpoints are downloaded separately** with
`python scripts/download_checkpoints.py` (into `../weights/` and the HuggingFace
cache). Each module retains its upstream license.

> **The "Fork layers" action — Qwen-Image-Layered — is not in this folder.** It is
> a diffusers pipeline loaded directly inside the agent
> (`ReDesign/tools/qwen_layered_tool.py`, via `from diffusers import
> QwenImageLayeredPipeline`) from the HuggingFace model
> [`Qwen/Qwen-Image-Layered`](https://huggingface.co/Qwen/Qwen-Image-Layered)
> (prefetch with `scripts/download_checkpoints.py --with-qwen`). It is the agent's
> heaviest component (~55 GB) and provides the core layer-forking step.

| Module | Agent tool | Role | Checkpoint(s) | Upstream / license |
|---|---|---|---|---|
| `grounding_dino` | `detect_gdino` | Open-vocabulary object detection | `groundingdino_swinb_cogcoor.pth` | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) (Apache-2.0) |
| `sam2` | `seg_sam2_bbox` | Promptable segmentation (SAM 2.1) | `sam2.1_hiera_large.pt` + `pip install sam2` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) (Apache-2.0) |
| `hisam` | `seg_hisam` | Text-stroke segmentation | `sam_tss_h_textseg.pth` (Hi-SAM head, **manual** from OneDrive) + `sam_vit_h_4b8939.pth` (SAM backbone, auto) | [ymy-k/Hi-SAM](https://github.com/ymy-k/Hi-SAM) (Apache-2.0) |
| `textremover` | `inpaint_lama` | LaMa text/object removal (inpaint) | `big-lama.pt` (TorchScript, auto via IOPaint mirror) | [advimman/lama](https://github.com/advimman/lama) (Apache-2.0) |
| `ObjectClear` | `inpaint_oc` | Object-aware inpainting | `jixin0101/ObjectClear` (HF) | [jixin0101/ObjectClear](https://huggingface.co/jixin0101/ObjectClear) |
| `ocr` | `detect_ocr` | Text detection/recognition | PaddleOCR PP-OCRv5 (auto) | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0) |
| `layerd` | (optional) | Alternative front-layer decomposition | — | [CyberAgentAILab/LayerD](https://github.com/CyberAgentAILab/LayerD) (see repo license) |
| `yolo` | (optional) | Object detection (YOLO11) | `yolov11.pt` | [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) |

> `grounding_dino` ships a CUDA extension compiled by `post_install.sh`
> (`pip install -e modules/grounding_dino`). Without a CUDA toolkit it falls back
> to a slower pure-Python path.
