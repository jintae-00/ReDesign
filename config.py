"""Repository-wide configuration for ReDesign.

Resolves the locations of the bundled ``modules/`` (tool source code) and the
``weights/`` checkpoint directory relative to this file, loads ``.env`` (API
keys), and puts every module package on ``sys.path`` so the agent tools can be
imported with their original module names.
"""
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

# Repository root (this file lives at the repo root).
SRC = Path(__file__).resolve().parent

# Load API keys / settings from <repo_root>/.env if present.
load_dotenv(SRC / ".env")

MODULES = SRC / "modules"
WEIGHTS = SRC / "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoints are downloaded here by scripts/download_checkpoints.py
WEIGHTS.mkdir(parents=True, exist_ok=True)

# Expose each tool module package on sys.path (mirrors the original layout).
if str(MODULES) not in sys.path:
    sys.path.insert(0, str(MODULES))
if MODULES.is_dir():
    for p in MODULES.iterdir():
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
