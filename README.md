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
├── evaluation/          # accuracy + editability evaluation
├── editability_eval/    # editability task/matching framework (dependency of evaluation/)
├── modules/             # third-party tool backends (code only; checkpoints downloaded)
├── tool_learning_wo_qwen/  # shared tool wrappers used by some baselines
├── scripts/             # download_checkpoints.py, download_figma_dataset.py
├── figma_data/          # Figma-909 dataset (downloaded; see below) + dataset card
├── crello_data/         # Crello download guide (not redistributed)
├── config.py            # resolves modules/ + weights/ paths, loads .env
├── environment.yml      # conda environment
├── post_install.sh      # pip/CUDA installs that can't go in environment.yml
├── .env.example         # API-key template (copy to .env)
├── ATTRIBUTION.md       # dataset & third-party attribution
└── LICENSE
```

## 1. Environment

```bash
git clone https://github.com/sonjt00/ReDesign.git
cd ReDesign

conda env create -f environment.yml
conda activate agent_qwen_layerd
bash post_install.sh          # torch nightly (cu129), paddlepaddle, sam2, GroundingDINO ext
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

Downloads GroundingDINO, SAM 2.1, Hi-SAM, LaMa, ObjectClear (and DINO for eval).
`Qwen/Qwen-Image-Layered` is fetched on first run unless `--with-qwen` is used.

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
    --qwen_gpus 2,3,4,5 --qwen_pair_size 2 --tool_gpus 6,7

# Crello
python -m REDESIGN.run_agent_crello \
    --data_dir crello_data/records --output_dir outputs/crello_agent \
    --qwen_gpus 2,3,4,5 --qwen_pair_size 2 --tool_gpus 6,7
```

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
