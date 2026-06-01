"""
Dataset loading for the RSR experiments.

Supervision signal (pooled, min-max normalised, then concatenated):
    MEN  +  SimVerb-3500  +  THINGS
Evaluation:
    SimLex-999  (held out, never used for supervision)

`load_all_rsr_datasets()` returns the concatenated pair list, a small info
dict, and the set of all words that appear anywhere in the supervision pool
(`rsr_words`) — the latter drives the both/one/neither-in-RSR partition that
the paper reports.
"""
from __future__ import annotations

from pathlib import Path

import scipy.io as sio

from . import paths


def load_men_pairs(path: Path = paths.MEN_NATURAL) -> list[tuple[str, str, float]]:
    """MEN: lines of `word1 word2 score`; words may carry a `-pos` suffix."""
    pairs: list[tuple[str, str, float]] = []
    if not path.exists():
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                w1, w2 = parts[0].split("-")[0], parts[1].split("-")[0]
                pairs.append((w1, w2, float(parts[2])))
    return pairs


def load_simverb_pairs(path: Path = paths.SIMVERB_FILE) -> list[tuple[str, str, float]]:
    """SimVerb-3500: tab-separated `verb1 verb2 pos score ...`."""
    pairs: list[tuple[str, str, float]] = []
    if not path.exists():
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                pairs.append((parts[0], parts[1], float(parts[3])))
    return pairs


def load_things_pairs(
    words_path: Path = paths.THINGS_WORDS,
    sim_path: Path = paths.THINGS_SIM_MAT,
) -> list[tuple[str, str, float]]:
    """THINGS: all upper-triangle pairs of the SPoSE similarity matrix."""
    words: list[str] = []
    with open(words_path, "r", encoding="utf-8") as f:
        for line in f:
            words.append(line.strip().replace(" ", "_"))

    sim_matrix = sio.loadmat(str(sim_path))["spose_sim"]

    pairs: list[tuple[str, str, float]] = []
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((words[i], words[j], float(sim_matrix[i, j])))
    return pairs


def _normalize(pairs: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    """Min-max scale scores into [0, 1] so the three sources are comparable."""
    if not pairs:
        return pairs
    scores = [p[2] for p in pairs]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-8:
        return pairs
    return [(w1, w2, (s - lo) / (hi - lo)) for w1, w2, s in pairs]


def load_all_rsr_datasets() -> tuple[list[tuple[str, str, float]], dict, set[str]]:
    """Pooled, normalised supervision set + info dict + supervision vocabulary."""
    men = _normalize(load_men_pairs())
    simverb = _normalize(load_simverb_pairs())
    things = _normalize(load_things_pairs())

    all_pairs = men + simverb + things

    rsr_words: set[str] = set()
    for w1, w2, _ in all_pairs:
        rsr_words.add(w1)
        rsr_words.add(w2)

    info = {
        "men": len(men),
        "simverb": len(simverb),
        "things": len(things),
        "total": len(all_pairs),
        "unique_words": len(rsr_words),
    }
    return all_pairs, info, rsr_words


def load_simlex(path: Path = paths.SIMLEX_FILE) -> list[tuple[str, str, float]]:
    """SimLex-999 evaluation pairs: header row, then `w1\\tw2\\tpos\\tSimLex999...`."""
    pairs: list[tuple[str, str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            pairs.append((parts[0], parts[1], float(parts[3])))
    return pairs
