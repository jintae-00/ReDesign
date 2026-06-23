# ReDesign

### Recovering Editable Design Structures from Images via Agentic Decomposition

<p align="center">
  <a href="https://openreview.net/pdf?id=JiEr8B3WBr"><img alt="Paper" src="https://img.shields.io/badge/Paper-OpenReview-b31b1b?style=for-the-badge&logo=readthedocs&logoColor=white"></a>
  &nbsp;
  <a href="https://sonjt00.github.io/ReDesign/"><img alt="Project Page" src="https://img.shields.io/badge/Project_Page-ReDesign-2ea44f?style=for-the-badge&logo=githubpages&logoColor=white"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/Jintae-Park/ReDesign-Figma909"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-Figma--909-ffce1c?style=for-the-badge&logo=huggingface&logoColor=black"></a>
</p>

> **ReDesign turns a single flat raster image back into an editable design** — text
> with real typography, vector shapes (fill/stroke), images, groups, and z-order,
> exported as an editable **JSON hierarchy**. When the original file is lost, a flat
> export no longer says which pixels form which object or how layers stack — ReDesign
> recovers that structure.

**How it works.** A **VLM controller** grows a *layer hierarchy* from the image —
expanding it breadth-first (coarse → fine) and choosing one tool-backed action per
node — while a **modular verifier** accepts, prunes, or retries each step, driving the
tree toward clean, atomic, editable leaves.

The controller orchestrates five tool actions:

| Action | Tools | Produces |
|---|---|---|
| **Extract text** | PaddleOCR + font recognition + Hi-SAM + LaMa inpaint | a text layer (editable typography) + background |
| **Fork layers** | Qwen-Image-Layered | several z-ordered RGBA layers |
| **Split (CCA)** | connected-component analysis | disjoint elements of one layer |
| **Detect & segment** | GroundingDINO + SAM 2 + inpaint | a foreground object + background |
| **Vectorize** | VTracer | a shape-like leaf → vector path (photos stay raster) |

## Repository layout

```
ReDesign/
├── ReDesign/            # the agent (inference entrypoints, controller, nodes, tools)
├── baselines/           # baseline methods compared in the paper
│   └── tool_backends/   #   tool wrappers used by the layered / multi-tools baselines
├── evaluation/          # accuracy + editability evaluation
│   └── editability_utils/  #   editability task/matching support library
├── modules/             # third-party tool backends (code only; checkpoints downloaded)
├── scripts/             # download_checkpoints.py, download_figma_dataset.py, prepare_crello_records.py
├── figma_data/          # Figma-909 benchmark (downloaded on demand) + dataset card → HuggingFace
├── crello_data/         # Crello download + render guide (not redistributed)
├── config.py            # resolves modules/ + weights/ paths, loads .env
├── environment.yml      # conda environment
├── post_install.sh      # pip/CUDA installs that can't go in environment.yml
├── .env.example         # API-key template (copy to .env)
├── ATTRIBUTION.md       # dataset & third-party attribution
└── LICENSE
```

## Quick Start

### 1. Environment

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

### 2. API keys & VLM endpoint

The controller is any **OpenAI-compatible chat-completions** endpoint — the
official OpenAI API, a gateway/proxy, or a self-hosted vLLM server. Configure it
in `.env` (nothing is hard-coded):

```bash
cp .env.example .env
# edit .env:
#   OPENAI_API_KEY=...     # required — key for your VLM endpoint
#   OPENAI_BASE_URL=...    # optional — custom endpoint; empty = https://api.openai.com/v1
#   VLM_MODEL=...          # optional — controller model; empty = gemini-3-flash-preview
#   GEMINI_API_KEY=...     # optional — only for the nanobanana tool
```

### 3. Checkpoints

```bash
python scripts/download_checkpoints.py              # tool + eval checkpoints -> weights/
python scripts/download_checkpoints.py --with-qwen  # also prefetch Qwen-Image-Layered (large)
```

Auto-downloads (public sources) GroundingDINO, SAM 2.1, the SAM ViT-H backbone,
LaMa, ObjectClear, and DINO (eval). `Qwen/Qwen-Image-Layered` is fetched on first
run unless `--with-qwen` is used.

> **One manual checkpoint:** Hi-SAM's text-segmentation head
> (`sam_tss_h_textseg.pth`) is distributed only via the authors' OneDrive. The
> script prints the link and target path (`weights/sam_tss_h_textseg.pth`) —
> download it once manually. (We do not redistribute third-party checkpoints.)

### 4. Datasets

**Figma-909** (ours, CC BY 4.0) — 909 real Figma frames with ground-truth layers
and attributes; used for both accuracy and editability:
```bash
python scripts/download_figma_dataset.py            # -> ./figma_data  (909 episodes)
```

**Crello** (CyberAgent; not redistributed) — raster designs for accuracy
comparison against prior work; download + render guide in
[`crello_data/README.md`](crello_data/README.md).

### 5. Run the agent

The Qwen-Image-Layered model is the only heavy component: it needs **≈55 GB of GPU
memory** (≈39 GB transformer + ≈16 GB text encoder). Choose GPUs to fit *your*
machine:

- **`--qwen_gpus`** — comma-separated GPU ids for the Qwen model.
- **`--qwen_pair_size N`** — how many of those GPUs to shard one Qwen worker
  across. Pick `N` so that (number of Qwen GPUs / N) × per-GPU memory ≥ 55 GB.
  e.g. one 80 GB GPU → `--qwen_gpus 0 --qwen_pair_size 1`; two 40 GB GPUs →
  `--qwen_gpus 0,1 --qwen_pair_size 2`; four 24 GB GPUs → `--qwen_gpus 0,1,2,3
  --qwen_pair_size 4`.
- **`--tool_gpus`** — GPU(s) for the vision tools (~10–16 GB; can reuse a Qwen GPU).
- **`--workers W`** — number of **parallel VLM-API workers** (default `1`). Each
  worker drives one episode and issues its own controller (LLM) calls, so raising
  `W` — up to whatever your VLM endpoint's rate limit allows — shortens total
  runtime. The controller model itself is set with `VLM_MODEL` in `.env` (default
  `gemini-3-flash-preview`); see §2.

**Throughput scales ~linearly with the compute you give it.** Two independent
knobs: (a) more **Qwen GPUs** → more parallel Qwen workers
(`len(--qwen_gpus) / --qwen_pair_size`), which cuts the GPU-bound parsing time;
(b) more **`--workers`** → more concurrent VLM-API calls, which cuts the API-bound
time. Add both and end-to-end time drops roughly proportionally (until you hit
your API rate limit). All GPU ids are placeholders — pick free ones with `nvidia-smi`.

```bash
# Single image
python -m ReDesign.run_single_image \
    --image path/to/design.png --output_dir outputs/single \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size <N> --tool_gpus <TOOL_GPU_IDS> --workers <W>

# Figma (all 909 episodes)
python -m ReDesign.run_agent_figma \
    --data_dir figma_data --output_dir outputs/figma_agent \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size <N> --tool_gpus <TOOL_GPU_IDS> --workers <W>

# Crello (records built via crello_data/README.md)
python -m ReDesign.run_agent_crello \
    --data_dir crello_data/records --output_dir outputs/crello_agent \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size <N> --tool_gpus <TOOL_GPU_IDS> --workers <W>
```

Outputs go to `--output_dir/episodes/<id>/` (`parse.json`, `history_tree.json`,
reconstructions, logs). **The input datasets are never modified** — every artifact
is written under the output directory. Completed episodes are skipped on re-run.

### 6. Evaluate

Evaluation scores the inference outputs from §5, so run the agent (and any
baselines) first. Each script reports **every model passed in `--models`** in one
run (agent and baselines together) and writes a readable comparison table.

```bash
# Reconstruction accuracy — Figma (agent + baselines together)
python evaluation/eval_accuracy_baselines_figma.py \
    --figma-data figma_data --models agent qwen layered multi_tools vtracer \
    --agent-dir outputs/figma_agent --gpu-ids <GPU_ID> \
    --output outputs/eval_accuracy_figma
#   baseline dirs default to outputs/baseline_<model>; override with --<model>-dir.
#   → results in outputs/eval_accuracy_figma/<timestamp>/comparison_accuracy.{md,csv}

# Reconstruction accuracy — Crello
python evaluation/eval_accuracy_baselines_crello.py \
    --crello-subset crello_data/records --models agent qwen layered multi_tools vtracer \
    --agent-dir outputs/crello_agent --gpu-ids <GPU_ID> \
    --output outputs/eval_accuracy_crello

# Editability (Figma) — matches are auto-precomputed on first use
REDESIGN_FIGMA_DATA=figma_data REDESIGN_AGENT_DIR=outputs/figma_agent \
    python evaluation/eval_editability_figma.py --models agent qwen layered multi_tools vtracer
#   → outputs/eval_editability_figma/comparison_editability.{md,csv}
```

See [`evaluation/README.md`](evaluation/README.md) for the full evaluation guide
(metrics, per-baseline editability, text editability, and result layout).

## Compute & API configuration (set to your budget)

Nothing about the hardware is hard-coded — every GPU id, the number of GPUs, the
worker count, and the LLM API key are **placeholders** you set for your own
machine and budget. There are two kinds of cost:

**A. GPU compute** — two GPU roles, configured independently:

| Role | Flag / env var | What runs on it | Memory |
|---|---|---|---|
| Qwen layered model | `--qwen_gpus` / `URLD_QWEN_GPUS` | `Qwen/Qwen-Image-Layered` | **≈55 GB** (bf16: ~39 GB transformer + ~16 GB text encoder) + activations |
| Vision tools | `--tool_gpus` / `URLD_TOOL_GPUS` | GroundingDINO, SAM 2, Hi-SAM, LaMa, ObjectClear (PaddleOCR on CPU) | ~10–16 GB |

- **Fit Qwen to your GPUs with `--qwen_pair_size N`** — it shards one Qwen worker
  across `N` GPUs (`device_map="balanced"`), so you need `N × per-GPU memory ≳ 55 GB`.
  One 80 GB GPU → `N=1`; two 40 GB → `N=2`; four 24 GB → `N=4`. (A CPU-offload
  fallback exists but is much slower.)
- **More GPUs = faster**: the listed `--qwen_gpus` are split into
  `len(qwen_gpus) / N` parallel Qwen workers. e.g. `--qwen_gpus 0,1,2,3
  --qwen_pair_size 2` → 2 workers (GPUs {0,1} and {2,3}) decoding concurrently,
  with tools on `--tool_gpus 4`.
- The tools fit on one ≥16 GB GPU and may share a Qwen GPU when memory allows.

**B. LLM API (the VLM controller)** — the agent calls an OpenAI-compatible
chat-completions endpoint for its expansion decisions. All of it is configured in
`.env` (never hard-coded):

- `OPENAI_API_KEY` — key for your endpoint (required). `GEMINI_API_KEY` is only for
  the optional nanobanana tool.
- `OPENAI_BASE_URL` — the endpoint. Leave empty for the official OpenAI API
  (`https://api.openai.com/v1`), or point it at a gateway/proxy or a self-hosted
  vLLM server.
- `VLM_MODEL` — the controller model. **Default: `gemini-3-flash-preview`.** Change
  it to any chat model your endpoint serves (e.g. `gpt-5-mini`, `gpt-4o`, …).
- `--workers <W>` processes `W` episodes in parallel; each issues its own API
  calls → **more workers = faster, up to your endpoint's rate limit** (mind spend).
  `--llm_limit` caps LLM calls per episode.

## Dataset, license & attribution

The Figma-909 frames are redistributed under **CC BY 4.0** (100% of 909 episodes),
with full per-episode attribution in every `figma_data/valid_frames/*.json` and in
`figma_data/ATTRIBUTIONS.csv`. See [`ATTRIBUTION.md`](ATTRIBUTION.md). The Crello
dataset is not redistributed. Bundled `modules/` retain their upstream licenses
(see [`modules/README.md`](modules/README.md)); the original ReDesign code is
released under [`LICENSE`](LICENSE).
