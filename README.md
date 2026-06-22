# ReDesign

**Recursive, agentic decomposition of graphic designs into editable layers.**

ReDesign is a tool-using agent that takes a flat design image (a Figma frame or a
Crello canvas) and recursively decomposes it into editable elements by
orchestrating vision tools — open-vocabulary detection (GroundingDINO),
segmentation (SAM 2, Hi-SAM), inpainting (LaMa, ObjectClear), OCR (PaddleOCR),
and layered image generation (Qwen-Image-Layered) — under a VLM controller.

This repository contains everything needed to **set up the environment**,
**download checkpoints and datasets**, **run the agent**, and **reproduce the
evaluation**.

## Repository layout

```
ReDesign/
├── REDESIGN/            # the agent (inference entrypoints, nodes, tools, graph)
├── BASELINES/           # baseline methods compared in the paper
│   └── tool_backends/   #   tool wrappers used by the layered / multi-tools baselines
├── evaluation/          # accuracy + editability evaluation
│   └── editability_utils/  #   editability task/matching support library
├── modules/             # third-party tool backends (code only; checkpoints downloaded)
├── scripts/             # download_checkpoints.py, download_figma_dataset.py
├── figma_data/          # Figma-909 dataset (downloaded on demand) + dataset card → HuggingFace
├── crello_data/         # Crello download guide (not redistributed)
├── config.py            # resolves modules/ + weights/ paths, loads .env
├── environment.yml      # conda environment
├── post_install.sh      # pip/CUDA installs that can't go in environment.yml
├── .env.example         # API-key template (copy to .env)
├── ATTRIBUTION.md       # dataset & third-party attribution
└── LICENSE
```

### How the pieces connect (data flow)

Evaluation scores **inference outputs**, so the order is always
**download → inference → evaluation**. The directory placeholders used in the
commands below are produced as follows:

| Placeholder | Produced by | Contents |
|---|---|---|
| `figma_data/` | `scripts/download_figma_dataset.py` | the GT dataset (909 episodes) |
| `<AGENT_OUTPUT_DIR>` | `python -m REDESIGN.run_agent_figma --output_dir <AGENT_OUTPUT_DIR>` | agent predictions (`episodes/<id>/parse.json`, …) |
| `<QWEN_OUTPUT_DIR>` | `python -m BASELINES.run_qwen_figma … <QWEN_OUTPUT_DIR>` | Qwen baseline layer outputs |
| `<*_BASELINE_OUTPUT_DIR>` | the corresponding `BASELINES/run_*` script | that baseline's predictions |
| `<MATCH_ROOT>` | `evaluation/before_eval_editability_precompute_matches.py` | GT↔prediction element matches (editability pre-step) |

So a full Figma run is: download `figma_data` → run the agent to get
`<AGENT_OUTPUT_DIR>` → pass both to the evaluation scripts.

## 1. Environment

```bash
git clone https://github.com/sonjt00/ReDesign.git
cd ReDesign

conda env create -f environment.yml
conda activate agent_qwen_layerd
bash post_install.sh          # PyTorch cu128, PaddlePaddle, diffusers(git), sam2, GroundingDINO ext
```

`post_install.sh` ends with an import check (torch, paddle, sam2, diffusers
`QwenImageLayeredPipeline`, transformers, langchain-openai, paddleocr, lpips,
vtracer, opencv). Everything `[ OK ]` ⇒ the environment is ready.

## 2. API keys

```bash
cp .env.example .env
# edit .env:  OPENAI_API_KEY=...   (VLM router; required)
#             GEMINI_API_KEY=...   (nanobanana tool; optional)
```

## 3. Checkpoints

```bash
python scripts/download_checkpoints.py            # tool + eval checkpoints -> weights/
python scripts/download_checkpoints.py --with-qwen  # also prefetch Qwen-Image-Layered (large)
```

Auto-downloads (public sources) GroundingDINO, SAM 2.1, the SAM ViT-H backbone,
LaMa, ObjectClear, and DINO (eval). `Qwen/Qwen-Image-Layered` is fetched on first
run unless `--with-qwen` is used.

> **One manual checkpoint:** Hi-SAM's text-segmentation head
> (`sam_tss_h_textseg.pth`) is distributed only via the authors' OneDrive. The
> script prints the link and target path (`weights/sam_tss_h_textseg.pth`) —
> download it once manually. (We do not redistribute third-party checkpoints.)

## 4. Datasets

**Figma-909** (ours, CC BY 4.0):
```bash
python scripts/download_figma_dataset.py          # -> ./figma_data  (909 episodes)
```

**Crello** (CyberAgent; not redistributed) — see [`crello_data/README.md`](crello_data/README.md).

## 5. Run the agent

```bash
# Figma (all 909 episodes)
python -m REDESIGN.run_agent_figma \
    --data_dir figma_data --output_dir outputs/figma_agent \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --tool_gpus <TOOL_GPU_IDS>

# Crello
python -m REDESIGN.run_agent_crello \
    --data_dir crello_data/records --output_dir outputs/crello_agent \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --tool_gpus <TOOL_GPU_IDS>
```

Replace `<QWEN_GPU_IDS>` / `<TOOL_GPU_IDS>` with your own comma-separated GPU ids
(the Qwen layered model and the vision tools run on separate GPUs), e.g.
`--qwen_gpus 0,1 --tool_gpus 2`. On a single GPU, pass the same id to both.

Outputs are written under `--output_dir/episodes/<id>/` (`parse.json`,
`history_tree.json`, reconstructions, logs). **The input datasets are never
modified** — every artifact is written under the output directory. Completed
episodes are skipped on re-run.

## 6. Evaluate

```bash
python evaluation/eval_accuracy_baselines_figma.py \
    --figma-data figma_data --models agent \
    --exp-pairs outputs/figma_agent:outputs/figma_qwen:merged \
    --output outputs/eval_accuracy_figma
```

Full accuracy + editability pipeline (including the two-step editability
precompute) is documented in [`evaluation/README.md`](evaluation/README.md).

## Dataset, license & attribution

The Figma-909 frames are redistributed under **CC BY 4.0** (100% of 909
episodes), with full per-episode attribution preserved in every
`figma_data/valid_frames/*.json` and in `figma_data/ATTRIBUTIONS.csv`. See
[`ATTRIBUTION.md`](ATTRIBUTION.md). The Crello dataset is not redistributed.

Bundled `modules/` retain their upstream licenses (see
[`modules/README.md`](modules/README.md)); the original ReDesign code is released
under the terms in [`LICENSE`](LICENSE).

## Hardware notes

The agent uses multiple GPUs (separate GPUs for the Qwen layered model vs. the
vision tools). Configure with `--qwen_gpus` / `--tool_gpus` or the
`URLD_QWEN_GPUS` / `URLD_TOOL_GPUS` environment variables.
