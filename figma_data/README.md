---
viewer: false
license: cc-by-4.0
pretty_name: ReDesign Figma-909
task_categories:
  - image-to-image
size_categories:
  - n<1K
tags:
  - graphic-design
  - layer-decomposition
  - figma
  - editable-design
  - vector-graphics
---

# ReDesign Figma-909 Benchmark

<p align="center">
  <a href="https://arxiv.org/abs/2607.25565"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
  &emsp;&emsp;&emsp;&emsp;
  <a href="https://jintae-00.github.io/ReDesign/"><img alt="Project Page" src="https://img.shields.io/badge/Project_Page-ReDesign-2ea44f?style=for-the-badge&logo=githubpages&logoColor=white"></a>
  &emsp;&emsp;&emsp;&emsp;
  <a href="https://github.com/jintae-00/ReDesign"><img alt="Code" src="https://img.shields.io/badge/Code-GitHub-8957e5?style=for-the-badge&logo=github&logoColor=white"></a>
</p>

<video src="https://huggingface.co/datasets/Jintae-Park/ReDesign-Figma909/resolve/main/Redesign-intro.mp4" controls muted loop playsinline width="100%"></video>

![ReDesign Figma-909 dataset overview](./exp_dataset.png)

> **ReDesign turns a single flat raster image back into an editable design:** text with real typography, vector shapes (fill/stroke), images, groups, and z-order, exported as an editable **JSON hierarchy**.
> When the original file is lost, a flat export no longer says which pixels form which object or how layers stack.
> ReDesign recovers that structure.

**How it works.** ReDesign treats a design as a **tree of layers** and rebuilds it piece by piece, starting from the whole image as the root:

1. **Look & decide:** a **VLM controller** examines a region and picks *one* tool-backed action to break it down (extract text, fork into layers, split, detect & segment, or vectorize).
2. **Split coarse → fine:** it expands the tree **breadth-first**, big regions first, then their finer details.
3. **Check every step:** a **modular verifier** accepts, prunes, or retries each split, driving the tree toward clean, atomic, editable leaves.

**Figma-909** is the evaluation benchmark for ReDesign: 909 real-world Figma Community designs, each a self-contained **episode** with ground-truth layer decomposition metadata and per-element images, supporting both **reconstruction-accuracy** and **editability** evaluation.

> 📁 The dataset files (metadata, images, attribution) live in the **[Files and versions](https://huggingface.co/datasets/Jintae-Park/ReDesign-Figma909/tree/main)** tab above. See the [ReDesign GitHub repository](https://github.com/jintae-00/ReDesign) for the download script and the full pipeline.

<br>
<br>

## License & Attribution

**All 909 episodes are licensed under [Creative Commons Attribution 4.0
International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).** Each
design was published under CC BY 4.0 by its original author on the Figma
Community. We redistribute the derived decomposition data under the same license,
with full attribution preserved.

- License coverage: **909 / 909 (100%) CC BY 4.0**
- Unique original authors: **288**
- Unique Figma Community files: **389**

Per-episode attribution (author name, author URL, source URL, license type,
license URL) is preserved in **every `valid_frames/*.json`** and aggregated in
[`ATTRIBUTIONS.csv`](https://huggingface.co/datasets/Jintae-Park/ReDesign-Figma909/blob/main/ATTRIBUTIONS.csv). When you use this dataset, please credit
the original authors and retain the CC BY 4.0 license and source links.

> If you are an author and would like a frame removed, please open an issue on the
> GitHub repository.

<br>
<br>

## Dataset structure

Each design is one self-contained **episode**, identified by an `episode_id` (the `valid_frames` JSON filename stem, e.g. `1002728450918630649_2_1898`). Every episode ships its ground-truth layer decomposition plus all per-element images.

| Path | What it holds |
|---|---|
| `valid_frames/<episode_id>.json` | Ground-truth metadata: layer tree, geometry, z-order, license, attribution |
| `unit_images/<figma_dir>/` | Per-element layer PNGs, the original render, and the GT reconstruction (`_reconstructed_*.png`, the agent's input) |
| `reconstructed_images/<episode_id>.png` | GT reconstruction keyed by episode id (the `_bbox` variant overlays element boxes) |
| `ATTRIBUTIONS.csv` | Per-episode author, source URL, and license |

Inside each JSON, `unit_images_dir` and the per-element `image_path` fields are **relative to the dataset root**, so the reconstruction resolves to `<root>/<unit_images_dir>/<reconstructed_image_path>`.

<br>
<br>

## Usage with ReDesign

```bash
# Download
python scripts/download_figma_dataset.py        # -> ./figma_data

# Run the agent on all 909 episodes
python -m ReDesign.run_agent_figma \
    --data_dir figma_data --output_dir outputs/figma_agent

# Reconstruction accuracy
python evaluation/eval_accuracy_baselines_figma.py \
    --figma-data figma_data --models agent \
    --agent-dir outputs/figma_agent \
    --output outputs/eval_accuracy_figma

# Atomic-edit editability (uses the same figma_data; matches auto-precomputed)
REDESIGN_FIGMA_DATA=figma_data REDESIGN_AGENT_DIR=outputs/figma_agent \
    python evaluation/eval_editability_figma.py --models agent
```

See the [ReDesign GitHub repository](https://github.com/jintae-00/ReDesign) for the
full pipeline (environment, checkpoints, inference, evaluation).

Complete per-episode attribution for all 288 original authors is provided in
[`ATTRIBUTIONS.csv`](https://huggingface.co/datasets/Jintae-Park/ReDesign-Figma909/blob/main/ATTRIBUTIONS.csv).

<br>
<br>

## Discussion

Questions, feedback, or requests? Open a thread in the
**[Community tab](https://huggingface.co/datasets/Jintae-Park/ReDesign-Figma909/discussions)**
of this dataset. If you are the author of a frame and would like it removed,
please start a discussion here or open an issue on the
[GitHub repository](https://github.com/jintae-00/ReDesign).
