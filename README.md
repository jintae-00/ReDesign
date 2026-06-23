# ReDesign

### Recovering Editable Design Structures from Images via Agentic Decomposition

When the original design file is lost, all that remains is a flat image — and a
raster export no longer says which pixels belong to which object, what attributes
produced them, or how elements were layered. **ReDesign reconstructs an editable
design from a single raster image**: text layers with real typography, vector
shapes with fill/stroke, images, groups, and z-order — exported as an editable
JSON hierarchy.

**How it works.** ReDesign casts raster-to-editable reconstruction as *growing a
layer hierarchy*. Starting from the whole image as the root, a **VLM controller**
expands the tree breadth-first (coarse → fine), at each node choosing one
tool-backed action and producing child layers. A **modular verifier** checks every
expansion — accepting it, pruning invalid branches, or retrying with a different
tool — which prevents sibling duplication and incomplete coverage and keeps the
tree growing toward atomic, editable leaves.

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

### 2. API keys

```bash
cp .env.example .env
# edit .env:  OPENAI_API_KEY=...   (the VLM controller; required)
#             GEMINI_API_KEY=...   (optional nanobanana tool)
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

(See **Compute & API configuration** below for the full reasoning and the
`--workers` / API knobs. All ids are placeholders — pick free ones with `nvidia-smi`.)

```bash
# Single image
python -m ReDesign.run_single_image \
    --image path/to/design.png --output_dir outputs/single \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size <N> --tool_gpus <TOOL_GPU_IDS>

# Figma (all 909 episodes)
python -m ReDesign.run_agent_figma \
    --data_dir figma_data --output_dir outputs/figma_agent \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size <N> --tool_gpus <TOOL_GPU_IDS>

# Crello (records built via crello_data/README.md)
python -m ReDesign.run_agent_crello \
    --data_dir crello_data/records --output_dir outputs/crello_agent \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size <N> --tool_gpus <TOOL_GPU_IDS>
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

## How the pieces connect (data flow)

The pipeline is always **download → inference → evaluation**. The directory
placeholders above are produced as follows:

| Placeholder | Produced by | Contents |
|---|---|---|
| `figma_data/` | `scripts/download_figma_dataset.py` | the ground-truth dataset (909 episodes) |
| `outputs/figma_agent` (`--agent-dir`) | `python -m ReDesign.run_agent_figma` | the agent's predictions (`episodes/<id>/parse.json`, …) |
| `outputs/baseline_<model>` (`--<model>-dir`) | the corresponding `baselines/run_*` script | each baseline's predictions (qwen is just one of them) |
| `outputs/editability_matches` | auto-built by the editability eval (or `before_eval_editability_precompute_matches.py`) | GT↔prediction element matches |

So a full Figma run is: download `figma_data` → run the agent (and baselines) →
pass the dataset + output dirs to the evaluation scripts.

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
chat-completions endpoint for its expansion decisions:

- Set `OPENAI_API_KEY` in `.env` (and `GEMINI_API_KEY` only for the optional
  nanobanana tool).
- `--workers <N>` processes `N` episodes in parallel; each issues its own API
  calls → **more workers = faster but more concurrent API usage** (mind rate
  limits and spend). `--llm_limit` caps LLM calls per episode.

## Dataset, license & attribution

The Figma-909 frames are redistributed under **CC BY 4.0** (100% of 909 episodes),
with full per-episode attribution in every `figma_data/valid_frames/*.json` and in
`figma_data/ATTRIBUTIONS.csv`. See [`ATTRIBUTION.md`](ATTRIBUTION.md). The Crello
dataset is not redistributed. Bundled `modules/` retain their upstream licenses
(see [`modules/README.md`](modules/README.md)); the original ReDesign code is
released under [`LICENSE`](LICENSE).
