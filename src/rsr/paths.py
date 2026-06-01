"""
Single source of truth for every on-disk path the experiments use.

Everything is resolved relative to the repository root (the directory that
contains this `src/` folder), so scripts work no matter what the current
working directory is. This replaces the old bare `Path("data")` /
`Path("SimLex-999")` references that only worked when run from the repo root.
"""
from __future__ import annotations

from pathlib import Path

# src/rsr/paths.py  ->  repo root is two parents up from this file.
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- top-level dirs ---------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
THINGS_DIR = REPO_ROOT / "things_similarity"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
DOCS_DIR = REPO_ROOT / "docs"

# --- contextual-vector pipeline artifacts (generated, gitignored) -----------
# Stage 0/A/B outputs all live under artifacts/ so the repo root stays clean.
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"                       # Stage B: vectors.npz per model
SENTENCES_DIR = ARTIFACTS_DIR / "sentences"               # Stage A: <slug>.jsonl per target
SENTENCES_MANIFEST = ARTIFACTS_DIR / "sentences_manifest.csv"
COVERAGE_CSV = ARTIFACTS_DIR / "coverage_probe.csv"           # Stage 0a
COVERAGE_COMBINED_CSV = ARTIFACTS_DIR / "coverage_probe_combined.csv"  # Stage 0b
HARVEST_CHECKPOINT = ARTIFACTS_DIR / ".harvest_enwiki_checkpoint.pkl"
FALLBACK_CHECKPOINT = ARTIFACTS_DIR / ".enwiki_fallback_checkpoint.pkl"

# --- supervision datasets (RSR training signal) -----------------------------
MEN_NATURAL = DATA_DIR / "MEN" / "MEN" / "MEN_dataset_natural_form_full"
MEN_LEMMA = DATA_DIR / "MEN" / "MEN" / "MEN_dataset_lemma_form_full"
SIMVERB_FILE = DATA_DIR / "simverb-3500-data" / "data" / "SimVerb-3500.txt"
THINGS_WORDS = THINGS_DIR / "variables" / "unique_id.txt"
THINGS_SIM_MAT = THINGS_DIR / "data" / "spose_similarity.mat"

# --- evaluation datasets ----------------------------------------------------
# NOTE: SimLex-999 / SICK / STS now live under data/ after the restructure.
SIMLEX_FILE = DATA_DIR / "SimLex-999" / "SimLex-999.txt"
SICK_DIR = DATA_DIR / "SICK"
STS_DIR = DATA_DIR / "STS_Benchmark"

# --- large corpora for from-scratch / harvest -------------------------------
ALLCOMBINED = DATA_DIR / "AllCombined.txt"
ENWIKI_DIR = DATA_DIR / "enwiki_namespace_0"


def ensure_dirs() -> None:
    """Create the output dirs experiments write to, if missing."""
    for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
