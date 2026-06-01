"""
Stage A — Sentence harvester.

Goal: for every retained target word/MWE in our supervision+evaluation pool,
collect up to MAX_OCCURRENCES sentences containing the target, drawn from
either AllCombined.txt or (for THINGS-only fallback targets) the enwiki dump.

Output:
  sentences/<target_slug>.jsonl
    one JSON object per sentence, with fields:
      {"target": "...", "sentence": "...", "source": "allcombined" | "enwiki"}

Targets:
  - Union of MEN, SimVerb, SimLex words and surviving THINGS concepts
  - THINGS sense-suffixed entries (baton1, bow2, ...) are DROPPED entirely
  - THINGS targets with <10 corpus hits in the combined probe are DROPPED

Design:
  - Aho-Corasick automaton over the lowercased, whitespace-normalised target set
  - Per-line scan: split into sentences with NLTK punkt, then for each sentence
    check membership; once we've collected MAX_OCCURRENCES for a target we stop
    accepting more (sentence-saturation early-exit per target)
  - Per-shard checkpointing for the enwiki pass

Run with:  python -u harvest_sentences.py
"""
from __future__ import annotations

import csv
import json
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import ahocorasick
import nltk
from nltk.tokenize import sent_tokenize

ROOT = Path(__file__).resolve().parents[2]  # experiments/contextual/ -> repo root
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
ALLCOMBINED = ROOT / "data" / "AllCombined.txt"
ENWIKI_DIR = ROOT / "data" / "enwiki_namespace_0"
COMBINED_CSV = ARTIFACTS / "coverage_probe_combined.csv"
SENTENCES_DIR = ARTIFACTS / "sentences"
ENWIKI_CHECKPOINT = ARTIFACTS / ".harvest_enwiki_checkpoint.pkl"
MANIFEST = ARTIFACTS / "sentences_manifest.csv"

MAX_OCCURRENCES = 50
MIN_SENT_TOKENS = 5
MAX_SENT_TOKENS = 64
FLOOR = 10
SUFFIX_RE = re.compile(r"\d+$")  # THINGS sense suffix
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
random.seed(42)


def slugify(target: str) -> str:
    """Filesystem-safe filename for a target."""
    return re.sub(r"[^A-Za-z0-9_]", "_", target.lower())


# ----- Stage 0: build the inventory of targets to harvest -----

def load_target_inventory() -> tuple[set[str], set[str]]:
    """Return (allcombined_targets, enwiki_fallback_targets).

    A target is in `enwiki_fallback_targets` if it is a THINGS target that fell
    below FLOOR in AllCombined.txt and (presumably) needs enwiki to reach floor.
    All other retained targets harvest from AllCombined.txt only.

    Sense-suffixed THINGS targets (baton1, bow2, ...) are dropped entirely.
    """
    allcombined_targets: set[str] = set()
    enwiki_fallback_targets: set[str] = set()
    dropped = {"sense_suffix": 0, "below_floor": 0}

    with COMBINED_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            tgt = row["target"]
            ac = int(row["count_allcombined"])
            tot = int(row["count_total"])

            if ds == "THINGS" and SUFFIX_RE.search(tgt):
                dropped["sense_suffix"] += 1
                continue
            if tot < FLOOR:
                dropped["below_floor"] += 1
                continue

            if ds == "THINGS" and ac < FLOOR:
                # Crossed the floor only with the enwiki fallback contribution.
                # We still want as many AllCombined sentences as exist, but the
                # primary harvesting source is enwiki for these targets.
                enwiki_fallback_targets.add(tgt)
            allcombined_targets.add(tgt)

    print(f"Inventory:", flush=True)
    print(f"  AllCombined targets: {len(allcombined_targets)}", flush=True)
    print(f"  of which need enwiki fallback (THINGS): {len(enwiki_fallback_targets)}", flush=True)
    print(f"  dropped — sense suffix: {dropped['sense_suffix']}", flush=True)
    print(f"  dropped — below floor: {dropped['below_floor']}", flush=True)
    return allcombined_targets, enwiki_fallback_targets


# ----- Stage A1: harvest from AllCombined.txt -----

def build_automaton(targets: set[str]) -> tuple[ahocorasick.Automaton, set[str]]:
    """Build automaton keyed on lowercased, whitespace-normalised targets.

    Returns the automaton and the set of single-token targets (for word-boundary
    discrimination at match time).
    """
    norm = {WHITESPACE_RE.sub(" ", t.lower().strip()): t for t in targets}
    A = ahocorasick.Automaton()
    for nt, orig in norm.items():
        A.add_word(nt, orig)
    A.make_automaton()
    singles = {nt for nt in norm if " " not in nt}
    return A, singles


def is_word_boundary(s: str, i: int) -> bool:
    if i < 0 or i >= len(s):
        return True
    c = s[i]
    return not (c.isalnum() or c == "_")


def matches_in_text(text: str, A: ahocorasick.Automaton) -> set[str]:
    """Return the set of (original) targets matched at any word-boundary position."""
    norm = WHITESPACE_RE.sub(" ", text.lower())
    hits: set[str] = set()
    for end_idx, orig in A.iter(norm):
        # Length of the matched key in the normalised text:
        key_len = len(WHITESPACE_RE.sub(" ", orig.lower().strip()))
        start_idx = end_idx - key_len + 1
        if is_word_boundary(norm, start_idx - 1) and is_word_boundary(norm, end_idx + 1):
            hits.add(orig)
    return hits


def acceptable_sentence(sent: str) -> bool:
    n = len(sent.split())
    return MIN_SENT_TOKENS <= n <= MAX_SENT_TOKENS


def harvest_allcombined(targets: set[str]) -> dict[str, list[dict]]:
    """Stream AllCombined.txt, return {target -> list of sentence dicts}."""
    A, _ = build_automaton(targets)
    out: dict[str, list[dict]] = defaultdict(list)
    saturated: set[str] = set()
    line_count = 0
    sent_count = 0
    accepted = 0
    t0 = time.time()

    with ALLCOMBINED.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue
            # Cheap reject: if no targets match the line, no need to sentence-split.
            if not matches_in_text(line, A):
                continue
            for sent in sent_tokenize(line):
                sent_count += 1
                if not acceptable_sentence(sent):
                    continue
                # Target match check on this specific sentence.
                hit_targets = matches_in_text(sent, A) - saturated
                if not hit_targets:
                    continue
                accepted += 1
                for t in hit_targets:
                    if len(out[t]) < MAX_OCCURRENCES:
                        out[t].append({"target": t, "sentence": sent, "source": "allcombined"})
                        if len(out[t]) >= MAX_OCCURRENCES:
                            saturated.add(t)
                # Early exit: if every target has saturated, stop scanning the corpus.
                if len(saturated) == len(targets):
                    print("All targets saturated — early exit.", flush=True)
                    break

            if line_count % 200_000 == 0:
                elapsed = time.time() - t0
                pct_sat = 100.0 * len(saturated) / len(targets)
                print(f"  AllCombined: {line_count:,} lines, "
                      f"{accepted:,} accepted sents, "
                      f"saturated {len(saturated)}/{len(targets)} ({pct_sat:.1f}%), "
                      f"{elapsed/60:.1f} min", flush=True)
            if len(saturated) == len(targets):
                break

    print(f"AllCombined harvest done: {sum(len(v) for v in out.values()):,} sentences, "
          f"{len(out):,}/{len(targets)} targets covered", flush=True)
    return dict(out)


# ----- Stage A2: harvest from enwiki for THINGS fallback targets -----

def iter_text_parts(parts):
    if not parts:
        return
    for p in parts:
        t = p.get("type")
        if t == "paragraph":
            v = p.get("value")
            if v:
                yield v
        elif t in ("section", "list"):
            yield from iter_text_parts(p.get("has_parts", []))
        elif t == "list_item":
            v = p.get("value")
            if v:
                yield v
            yield from iter_text_parts(p.get("has_parts", []))


def iter_article_text(obj: dict):
    for s in obj.get("sections", []):
        yield from iter_text_parts(s.get("has_parts", []))


def harvest_enwiki(
    targets: set[str],
    seed_counts: dict[str, int],
) -> dict[str, list[dict]]:
    """Stream enwiki shards, harvest sentences for `targets`.

    `seed_counts` is the per-target count already in the AllCombined harvest;
    we only need (MAX_OCCURRENCES - seed_counts[t]) more sentences per target.
    """
    A, _ = build_automaton(targets)
    out: dict[str, list[dict]] = defaultdict(list)

    needed: dict[str, int] = {t: MAX_OCCURRENCES - seed_counts.get(t, 0) for t in targets}
    saturated: set[str] = {t for t, n in needed.items() if n <= 0}
    print(f"  enwiki harvest: {len(targets) - len(saturated)} targets need more sentences",
          flush=True)
    if len(saturated) == len(targets):
        return dict(out)

    # Resume support.
    if ENWIKI_CHECKPOINT.exists():
        with ENWIKI_CHECKPOINT.open("rb") as f:
            ck = pickle.load(f)
        out_loaded = ck.get("out", {})
        for t, sents in out_loaded.items():
            out[t] = sents
            if seed_counts.get(t, 0) + len(sents) >= MAX_OCCURRENCES:
                saturated.add(t)
        shards_done = ck.get("shards_done", set())
        print(f"  resumed enwiki: {len(shards_done)} shards done, "
              f"{sum(len(v) for v in out.values()):,} sentences cached, "
              f"{len(saturated)} targets already saturated", flush=True)
    else:
        shards_done = set()

    enwiki_files = sorted(ENWIKI_DIR.glob("*.jsonl"))
    remaining = [p for p in enwiki_files if p.name not in shards_done]
    total_size = sum(p.stat().st_size for p in enwiki_files)
    bytes_done = sum(p.stat().st_size for p in enwiki_files if p.name in shards_done)
    print(f"  enwiki: {len(remaining)}/{len(enwiki_files)} shards remaining "
          f"({total_size / 1e9:.1f} GB total)", flush=True)

    t0 = time.time()
    for idx, fpath in enumerate(remaining, start=1):
        if len(saturated) == len(targets):
            print("  all enwiki targets saturated — early exit.", flush=True)
            break
        shard_t0 = time.time()
        f_articles = 0
        f_added = 0
        with fpath.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                f_articles += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for text in iter_article_text(obj):
                    if not text:
                        continue
                    if not matches_in_text(text, A):
                        continue
                    for sent in sent_tokenize(text):
                        if not acceptable_sentence(sent):
                            continue
                        hits = matches_in_text(sent, A) - saturated
                        if not hits:
                            continue
                        for t in hits:
                            already = seed_counts.get(t, 0) + len(out[t])
                            if already < MAX_OCCURRENCES:
                                out[t].append({"target": t, "sentence": sent, "source": "enwiki"})
                                f_added += 1
                                if seed_counts.get(t, 0) + len(out[t]) >= MAX_OCCURRENCES:
                                    saturated.add(t)
                if len(saturated) == len(targets):
                    break

        bytes_done += fpath.stat().st_size
        shards_done.add(fpath.name)
        elapsed = time.time() - t0
        pct_dump = 100.0 * bytes_done / total_size
        eta_min = (len(remaining) - idx) * (elapsed / idx) / 60 if idx else 0
        print(f"  [{idx:>2d}/{len(remaining)}] {fpath.name}: "
              f"{f_articles:>7,} articles, +{f_added:>5,} sents, "
              f"saturated {len(saturated)}/{len(targets)} | "
              f"{time.time()-shard_t0:.1f}s, dump {pct_dump:.1f}%, "
              f"ETA {eta_min:.1f} min", flush=True)
        # Checkpoint.
        tmp = ENWIKI_CHECKPOINT.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump({"out": dict(out), "shards_done": shards_done}, f)
        tmp.replace(ENWIKI_CHECKPOINT)

    return dict(out)


# ----- Stage A3: write per-target sentence files + manifest -----

def write_outputs(
    ac_sentences: dict[str, list[dict]],
    en_sentences: dict[str, list[dict]],
    all_targets: set[str],
) -> None:
    SENTENCES_DIR.mkdir(exist_ok=True)
    rows = []
    for t in sorted(all_targets):
        combined = ac_sentences.get(t, []) + en_sentences.get(t, [])
        # If we somehow over-collected, cap.
        combined = combined[:MAX_OCCURRENCES]
        slug = slugify(t)
        out_path = SENTENCES_DIR / f"{slug}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for s in combined:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        n_ac = sum(1 for s in combined if s["source"] == "allcombined")
        n_en = sum(1 for s in combined if s["source"] == "enwiki")
        rows.append({
            "target": t,
            "slug": slug,
            "n_total": len(combined),
            "n_allcombined": n_ac,
            "n_enwiki": n_en,
        })

    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "slug", "n_total",
                                           "n_allcombined", "n_enwiki"])
        w.writeheader()
        w.writerows(rows)

    saturated = sum(1 for r in rows if r["n_total"] >= MAX_OCCURRENCES)
    median_n = sorted(r["n_total"] for r in rows)[len(rows) // 2]
    print(f"\nWrote {len(rows)} per-target sentence files to {SENTENCES_DIR}", flush=True)
    print(f"  saturated (>= {MAX_OCCURRENCES}): {saturated}/{len(rows)}", flush=True)
    print(f"  median sentences/target: {median_n}", flush=True)
    print(f"  manifest: {MANIFEST}", flush=True)


def main() -> None:
    print("Loading target inventory ...", flush=True)
    ac_targets, en_fallback_targets = load_target_inventory()

    print("\n=== Stage A1: harvesting from AllCombined.txt ===", flush=True)
    ac_sentences = harvest_allcombined(ac_targets)

    seed_counts = {t: len(ac_sentences.get(t, [])) for t in en_fallback_targets}
    print("\n=== Stage A2: harvesting from enwiki (THINGS fallback only) ===", flush=True)
    en_sentences = harvest_enwiki(en_fallback_targets, seed_counts)

    print("\n=== Stage A3: writing per-target files + manifest ===", flush=True)
    write_outputs(ac_sentences, en_sentences, ac_targets)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
