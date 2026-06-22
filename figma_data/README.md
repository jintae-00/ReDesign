---
license: cc-by-4.0
task_categories:
  - image-segmentation
  - image-to-image
pretty_name: ReDesign Figma-909 Benchmark
tags:
  - graphic-design
  - layer-decomposition
  - figma
  - ui-design
size_categories:
  - n<1K
---

# ReDesign Figma-909 Benchmark

📦 **Download:** [`Jintae-Park/ReDesign-Figma909` on HuggingFace](https://huggingface.co/datasets/Jintae-Park/ReDesign-Figma909)
— or run `python scripts/download_figma_dataset.py` to fetch it into `./figma_data`.

909 real-world graphic-design frames sourced from the **Figma Community**, used as
the Figma evaluation benchmark in the ReDesign project (recursive layer
decomposition of designs into editable elements).

> In a fresh clone this folder contains only this README; the actual dataset is
> hosted on HuggingFace (link above) and downloaded on demand.

Every frame is a self-contained episode with ground-truth layer decomposition
metadata and per-element images, enabling both **reconstruction-accuracy** and
**editability** evaluation.

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
[`ATTRIBUTIONS.csv`](./ATTRIBUTIONS.csv). When you use this dataset, please credit
the original authors and retain the CC BY 4.0 license and source links.

> If you are an author and would like a frame removed, please open an issue on the
> GitHub repository.

## Dataset structure

```
figma_data/
├── valid_frames/<episode_id>.json          # 909 GT metadata (layers, geometry, license, attribution)
├── unit_images/<figma_dir>/                # per-episode GT element images + reconstruction
│   ├── _original_<f>.png                    #   original frame render
│   ├── _reconstructed_<f>.png               #   GT reconstruction (agent input)
│   ├── _reconstructed_bbox_<f>.png          #   reconstruction with element bboxes
│   ├── _expanded_background.png             #   expanded background layer
│   └── <element>.png                        #   individual GT layer/element images
├── reconstructed_images/<episode_id>.png        # GT reconstruction, episode-id keyed (convenience)
├── reconstructed_images/<episode_id>_bbox.png   # + bbox variant
└── ATTRIBUTIONS.csv                        # per-episode author / source / license
```

`episode_id` is the `valid_frames` JSON filename stem (e.g.
`1002728450918630649_2_1898`). Inside each JSON, `unit_images_dir` and the
per-element `image_path` fields are paths **relative to the dataset root**, so the
GT reconstruction resolves to `<root>/<unit_images_dir>/<reconstructed_image_path>`.

> The 909 frames are the de-duplicated union of two difficulty subsets
> (`dino80_obj_5_60_char_25`: 435, `dino90_obj_5_25_char_50`: 474). Build-time
> intermediate artifacts (inpainting/segmentation byproducts) have been removed;
> only ground-truth assets remain.

## Usage with ReDesign

```bash
# Download
python scripts/download_figma_dataset.py        # -> ./figma_data

# Run the agent on all 909 episodes
python -m REDESIGN.run_agent_figma \
    --data_dir figma_data --output_dir outputs/figma_agent

# Evaluate reconstruction accuracy
python evaluation/eval_accuracy_baselines_figma.py \
    --figma-data figma_data --models agent \
    --exp-pairs outputs/figma_agent:outputs/figma_qwen:merged \
    --output outputs/eval_accuracy_figma
```

See the [ReDesign GitHub repository](https://github.com/sonjt00/ReDesign) for the
full pipeline (environment, checkpoints, inference, evaluation).

Complete per-episode attribution for all 288 original authors is provided in
[`ATTRIBUTIONS.csv`](./ATTRIBUTIONS.csv).
