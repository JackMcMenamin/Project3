"""Repository paths, anchored to the repo root so scripts run from anywhere."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
THINGS_DIR = REPO_ROOT / "things_similarity"
ARTIFACTS = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"

MEN_PATH = DATA_DIR / "MEN" / "MEN" / "MEN_dataset_natural_form_full"
SIMVERB_PATH = DATA_DIR / "simverb-3500-data" / "data" / "SimVerb-3500.txt"
THINGS_WORDS_PATH = THINGS_DIR / "variables" / "unique_id.txt"
THINGS_SIM_PATH = THINGS_DIR / "data" / "spose_similarity.mat"
SIMLEX_PATH = DATA_DIR / "SimLex-999" / "SimLex-999.txt"

# Every reported run uses the full-Wikipedia pool. The first harvest
# (artifacts/sentences, Simple English) is still on disk but is not the default.
DEFAULT_SENTENCE_POOL = ARTIFACTS / "sentences_strict"
_sentence_pool = DEFAULT_SENTENCE_POOL


def sentence_pool() -> Path:
    return _sentence_pool


def set_sentence_pool(path: str | Path) -> Path:
    """Point every loader at a different pool. Relative paths are taken from
    the repo root. Exits if the folder does not exist."""
    global _sentence_pool
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_dir():
        raise SystemExit(f"no such sentence pool: {p}")
    _sentence_pool = p
    return p
