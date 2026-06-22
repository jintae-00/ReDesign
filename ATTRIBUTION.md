# Attribution

## Figma-909 dataset

The 909 Figma frames redistributed with this project are each licensed under
**[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**
by their original Figma Community authors.

- **909 / 909 (100%)** of episodes are CC BY 4.0.
- **288** unique original authors; **389** unique Figma Community files.
- Full per-episode attribution — `author_name`, `author_url`, `source_url`,
  `license_type`, `license_url` — is preserved in **every**
  `figma_data/valid_frames/*.json` and aggregated in
  [`figma_data/ATTRIBUTIONS.csv`](figma_data/ATTRIBUTIONS.csv).

If you use this dataset, you must credit the original authors, indicate any
changes, and retain the CC BY 4.0 license notice and source links, in accordance
with CC BY 4.0. If you are an author and want a frame removed, please open an issue.

## Crello dataset

The Crello benchmark is **not** redistributed here. Download it from CyberAgent
AI Lab and follow its terms — see [`crello_data/README.md`](crello_data/README.md).

## Third-party tool code (`modules/`)

The agent bundles third-party model code under `modules/`, each retaining its own
upstream license (GroundingDINO, SAM 2 / Hi-SAM, LaMa, ObjectClear, PaddleOCR,
YOLO, vtracer, Qwen-Image-Layered). See [`modules/README.md`](modules/README.md)
for sources and licenses. Their checkpoints are downloaded separately via
`scripts/download_checkpoints.py` and remain under their respective licenses.
