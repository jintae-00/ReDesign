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

Replace `<QWEN_GPU_IDS>` / `<TOOL_GPU_IDS>` with your own comma-separated GPU ids.
**The Qwen layered model needs ~2 GPUs** (≈55 GB of weights — see §"Compute & API
configuration"); the vision tools use one more GPU. `--qwen_pair_size 2` shards
one Qwen worker across the 2 Qwen GPUs.

```bash
# Single image
python -m REDESIGN.run_single_image \
    --image path/to/design.png --output_dir outputs/single \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --tool_gpus <TOOL_GPU_IDS>

# Figma (all 909 episodes)
python -m REDESIGN.run_agent_figma \
    --data_dir figma_data --output_dir outputs/figma_agent \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --tool_gpus <TOOL_GPU_IDS>

# Crello
python -m REDESIGN.run_agent_crello \
    --data_dir crello_data/records --output_dir outputs/crello_agent \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --tool_gpus <TOOL_GPU_IDS>
```

Example with concrete ids (2 GPUs for Qwen, 1 for tools):
`--qwen_gpus 0,1 --qwen_pair_size 2 --tool_gpus 2`.

Outputs are written under `--output_dir/episodes/<id>/` (`parse.json`,
`history_tree.json`, reconstructions, logs). **The input datasets are never
modified** — every artifact is written under the output directory. Completed
episodes are skipped on re-run.

## 6. Evaluate

Evaluation scores inference outputs, so run the agent first (§5). Point the eval
at the dataset (`--figma-data`) and the agent's output dir (`--agent-dir`).

```bash
# Reconstruction accuracy (Figma)
python evaluation/eval_accuracy_baselines_figma.py \
    --figma-data figma_data --models agent \
    --agent-dir outputs/figma_agent \
    --output outputs/eval_accuracy_figma

# Editability — step 1: precompute GT↔prediction element matches
python evaluation/before_eval_editability_precompute_matches.py \
    --figma-data figma_data --model agent --model-dir outputs/figma_agent \
    --output outputs/editability_matches

# Editability — step 2: atomic-edit editability (set GPU via --gpu-ids / env)
REDESIGN_FIGMA_DATA=figma_data \
REDESIGN_AGENT_DIR=outputs/figma_agent \
REDESIGN_MATCH_ROOT=outputs/editability_matches \
    python evaluation/eval_editability_figma.py --models agent

# Text editability
python evaluation/eval_editability_text_figma.py \
    --figma-data figma_data --models agent --agent-dir outputs/figma_agent \
    --output outputs/eval_editability_text --ocr-gpu <GPU_ID>
```

Full accuracy + editability pipeline is documented in
[`evaluation/README.md`](evaluation/README.md).

## Dataset, license & attribution

The Figma-909 frames are redistributed under **CC BY 4.0** (100% of 909
episodes), with full per-episode attribution preserved in every
`figma_data/valid_frames/*.json` and in `figma_data/ATTRIBUTIONS.csv`. See
[`ATTRIBUTION.md`](ATTRIBUTION.md). The Crello dataset is not redistributed.

Bundled `modules/` retain their upstream licenses (see
[`modules/README.md`](modules/README.md)); the original ReDesign code is released
under the terms in [`LICENSE`](LICENSE).

## Compute & API configuration (set to your budget)

Nothing about the hardware is hard-coded — every GPU id, the number of GPUs, the
worker count, and the LLM API key are **placeholders** you set for your own
machine and budget. The agent has two kinds of cost:

**A. GPU compute** — two GPU roles, configured independently:

| Role | Flag / env var | What runs on it | Memory guidance |
|---|---|---|---|
| Qwen layered model | `--qwen_gpus <QWEN_GPU_IDS>` / `URLD_QWEN_GPUS` | `Qwen/Qwen-Image-Layered` (the costly generator) | **≈55 GB of weights in bf16** (transformer ~39 GB + text encoder ~16 GB) plus activations → needs **2 GPUs** (sharded with `device_map="balanced"`, e.g. 2×≥32 GB) **or one ≥80 GB GPU**. Use `--qwen_pair_size 2` to put one worker on 2 GPUs. CPU-offload fallback exists but is much slower. |
| Vision tools | `--tool_gpus <TOOL_GPU_IDS>` / `URLD_TOOL_GPUS` | GroundingDINO, SAM 2, Hi-SAM, LaMa, ObjectClear (PaddleOCR runs on CPU) | ~10–16 GB total → **one ≥16 GB GPU** is enough. |

- **Minimum** to run end-to-end: **2 GPUs** — e.g. 2 GPUs of ≥40 GB for Qwen
  (`--qwen_gpus 0,1 --qwen_pair_size 2`) and one of them (or a third) for the
  tools (`--tool_gpus 1`). On a **single ≥80 GB GPU** you can put everything on
  it: `--qwen_gpus 0 --qwen_pair_size 1 --tool_gpus 0 --objectclear_gpu 0`.
- **Faster**: more GPUs = more **parallel** Qwen workers. `--qwen_pair_size N`
  sets how many GPUs each Qwen worker uses; the listed `--qwen_gpus` are split
  into `len(qwen_gpus)/N` parallel workers. e.g. `--qwen_gpus 0,1,2,3
  --qwen_pair_size 2` → 2 workers (GPUs {0,1} and {2,3}) decoding episodes
  concurrently, with tools on a separate GPU (`--tool_gpus 4`).
- GPU ids are **per-machine** — use `nvidia-smi` to pick free ones. (The built-in
  defaults are a conservative single GPU 0, intended only as a placeholder.)

**B. LLM API (the VLM controller)** — the agent calls an OpenAI-compatible
chat-completions endpoint for routing/labeling decisions:

- Set the key in `.env`: `OPENAI_API_KEY=...` (and `GEMINI_API_KEY=...` only if you
  enable the optional nanobanana tool).
- `--workers <N>` sets how many episodes are processed in parallel; each worker
  issues its own API calls. **More workers = faster but more concurrent API
  usage** (watch your rate limits and spend). `--llm_limit` caps the number of
  LLM calls per episode.

**Examples (replace ids with your own free GPUs):**

```bash
# Single GPU (e.g. one 80 GB GPU), modest API usage
python -m REDESIGN.run_agent_figma --data_dir figma_data --output_dir outputs/figma_agent \
    --qwen_gpus 0 --tool_gpus 0 --objectclear_gpu 0 --workers 1

# Multi-GPU, faster (3 GPUs for parallel Qwen, 1 for tools), more API workers
python -m REDESIGN.run_agent_figma --data_dir figma_data --output_dir outputs/figma_agent \
    --qwen_gpus 0,1,2 --qwen_pair_size 1 --tool_gpus 3 --workers 4
```
