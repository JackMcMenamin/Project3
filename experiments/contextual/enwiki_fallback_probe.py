"""
Fallback coverage probe over the full enwiki dump for THINGS targets that
fell below the >=10 floor in AllCombined.txt.

Design:
  - Aho-Corasick automaton over all targets in one linear pass per line.
    Single tokens use word-boundary checks at match time; MWEs allow
    flexible internal whitespace via a normalisation step.
  - Per-shard checkpointing: state (counts, shards_done) is pickled after
    each JSONL file completes. Re-running resumes where it left off.
  - Line-buffered stdout: per-shard progress is flushed immediately.

Run with: python -u enwiki_fallback_probe.py
"""
from __future__ import annotations

import csv
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import ahocorasick

ROOT = Path(__file__).resolve().parents[2]  # experiments/contextual/ -> repo root
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
ENWIKI_DIR = ROOT / "data" / "enwiki_namespace_0"
EXISTING_CSV = ARTIFACTS / "coverage_probe.csv"
OUT_CSV = ARTIFACTS / "coverage_probe_combined.csv"
CHECKPOINT = ARTIFACTS / ".enwiki_fallback_checkpoint.pkl"

FLOOR = 10  # only chase THINGS targets that fell below this in AllCombined.txt

# Pattern matches a maximal run of letters (with optional internal apostrophe).
# We use it to verify single-token matches sit on word boundaries.
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
# Whitespace runs inside MWEs are normalised to a single space so an automaton
# entry "ice cream" matches "ice  cream" or "ice\ncream" too.
WHITESPACE_RE = re.compile(r"\s+")


def load_existing_counts() -> tuple[dict[tuple[str, str], int], dict[str, set[str]]]:
    counts: dict[tuple[str, str], int] = {}
    by_dataset: dict[str, set[str]] = defaultdict(set)
    with EXISTING_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            counts[(row["dataset"], row["target"])] = int(row["count"])
            by_dataset[row["dataset"]].add(row["target"])
    return counts, dict(by_dataset)


def iter_text_parts(parts):
    """Recursively yield text from `has_parts` (paragraph, list, list_item, section)."""
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


def build_automaton(targets: set[str]) -> ahocorasick.Automaton:
    """Build an Aho-Corasick automaton keyed on lowercased, whitespace-normalised targets."""
    A = ahocorasick.Automaton()
    for t in targets:
        # MWE entries have their internal whitespace already normalised in load.
        A.add_word(t, t)
    A.make_automaton()
    return A


def is_word_boundary(s: str, i: int) -> bool:
    """True if position i in s is at a word boundary (re.\\b)."""
    if i < 0 or i >= len(s):
        return True
    c = s[i]
    return not (c.isalnum() or c == "_")


def count_matches(text: str, A: ahocorasick.Automaton, single_targets: set[str]) -> dict[str, int]:
    """Return per-target hit counts for one text block.

    For single-token targets we require word boundaries on both sides
    (so 'cat' doesn't match inside 'cathedral'). For MWE targets, the
    internal whitespace is already normalised and the boundaries are
    enforced by the entry containing leading/trailing word chars.
    """
    out: dict[str, int] = defaultdict(int)
    # Normalise whitespace once so MWEs with internal whitespace runs match.
    norm = WHITESPACE_RE.sub(" ", text.lower())
    n = len(norm)
    for end_idx, key in A.iter(norm):
        start_idx = end_idx - len(key) + 1
        # Word boundary check: pre-char and post-char must not be alnum/_.
        # For MWEs this is also correct because the first/last char of the key
        # is alphabetic, so the same check applies.
        left_ok = is_word_boundary(norm, start_idx - 1)
        right_ok = is_word_boundary(norm, end_idx + 1)
        if left_ok and right_ok:
            out[key] += 1
    return out


def load_checkpoint() -> tuple[dict[str, int], set[str]]:
    if CHECKPOINT.exists():
        with CHECKPOINT.open("rb") as f:
            data = pickle.load(f)
        return data["counts"], data["shards_done"]
    return defaultdict(int), set()


def save_checkpoint(counts: dict[str, int], shards_done: set[str]) -> None:
    tmp = CHECKPOINT.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump({"counts": dict(counts), "shards_done": shards_done}, f)
    tmp.replace(CHECKPOINT)


def main() -> None:
    print("Loading existing AllCombined.txt counts ...", flush=True)
    existing_counts, by_dataset = load_existing_counts()

    things = by_dataset.get("THINGS", set())
    needs_fallback = {
        t for t in things if existing_counts.get(("THINGS", t), 0) < FLOOR
    }
    # Normalise MWEs so the automaton entries match against normalised text.
    needs_fallback_norm = {WHITESPACE_RE.sub(" ", t.lower().strip()) for t in needs_fallback}
    single_targets = {t for t in needs_fallback_norm if " " not in t}
    print(f"THINGS total: {len(things)}", flush=True)
    print(f"Below floor (<{FLOOR}) needing fallback: {len(needs_fallback_norm)} "
          f"({len(single_targets)} single-token, "
          f"{len(needs_fallback_norm)-len(single_targets)} MWE)", flush=True)

    print("Building Aho-Corasick automaton ...", flush=True)
    A = build_automaton(needs_fallback_norm)
    print(f"  automaton has {len(needs_fallback_norm):,} patterns", flush=True)

    counts, shards_done = load_checkpoint()
    if shards_done:
        print(f"Resuming from checkpoint: {len(shards_done)} shards already done, "
              f"{sum(counts.values()):,} hits accumulated.", flush=True)

    enwiki_files = sorted(ENWIKI_DIR.glob("*.jsonl"))
    total_size = sum(p.stat().st_size for p in enwiki_files)
    remaining = [p for p in enwiki_files if p.name not in shards_done]
    print(f"Shards: {len(enwiki_files)} total, {len(remaining)} remaining "
          f"({total_size / 1e9:.1f} GB total)", flush=True)
    print(flush=True)

    t0 = time.time()
    bytes_done = sum(p.stat().st_size for p in enwiki_files if p.name in shards_done)
    articles_total = 0

    for shard_idx, fpath in enumerate(remaining, start=1):
        shard_t0 = time.time()
        f_articles = 0
        f_hits = 0
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
                    for key, n in count_matches(text, A, single_targets).items():
                        counts[key] += n
                        f_hits += n

        bytes_done += fpath.stat().st_size
        articles_total += f_articles
        shard_elapsed = time.time() - shard_t0
        total_elapsed = time.time() - t0
        pct_dump = 100.0 * bytes_done / total_size
        # Estimate remaining time based on bytes processed in this run only.
        run_bytes = bytes_done - sum(p.stat().st_size for p in enwiki_files if p.name in shards_done and p.name != fpath.name)
        # (Simple estimate: remaining shards * mean shard time so far in this run.)
        eta_min = (len(remaining) - shard_idx) * (total_elapsed / shard_idx) / 60

        print(f"[{shard_idx:>2d}/{len(remaining)}] {fpath.name}: "
              f"{f_articles:>7,} articles, "
              f"{f_hits:>7,} hits, "
              f"{shard_elapsed:>5.1f}s | "
              f"dump {pct_dump:5.1f}%, "
              f"elapsed {total_elapsed/60:5.1f} min, "
              f"ETA {eta_min:5.1f} min",
              flush=True)

        shards_done.add(fpath.name)
        save_checkpoint(counts, shards_done)

    print(f"\nFallback complete. Total hits: {sum(counts.values()):,} across "
          f"{len(counts):,} distinct targets.", flush=True)

    # Reverse the MWE-normalisation when writing so target strings match the
    # form stored in coverage_probe.csv ("ice cream", not "ice cream" with
    # different whitespace).
    fallback_by_orig: dict[str, int] = defaultdict(int)
    norm_to_orig = {WHITESPACE_RE.sub(" ", t.lower().strip()): t for t in needs_fallback}
    for k, c in counts.items():
        orig = norm_to_orig.get(k, k)
        fallback_by_orig[orig] += c

    print(f"\nWriting combined counts to {OUT_CSV}", flush=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "target", "is_mwe",
                    "count_allcombined", "count_enwiki_fallback", "count_total"])
        for (ds, tgt), c in sorted(existing_counts.items()):
            extra = fallback_by_orig.get(tgt, 0) if (ds == "THINGS" and tgt in needs_fallback) else 0
            w.writerow([ds, tgt, int(" " in tgt), c, extra, c + extra])

    # THINGS coverage table after fallback.
    floors = (1, 5, 10, 25, 50)
    new_totals = {
        t: existing_counts.get(("THINGS", t), 0)
           + (fallback_by_orig.get(t, 0) if t in needs_fallback else 0)
        for t in things
    }
    n = len(things)
    print("\n=== THINGS coverage (after enwiki fallback) ===", flush=True)
    print(f"{'floor':>6s} {'kept':>6s} {'pct':>7s}  (was AllCombined.txt only)", flush=True)
    for fl in floors:
        kept_after = sum(1 for c in new_totals.values() if c >= fl)
        kept_before = sum(
            1 for t in things if existing_counts.get(("THINGS", t), 0) >= fl
        )
        print(f"  >={fl:<3d} {kept_after:>5d} "
              f"{100.0*kept_after/n:>5.1f}%   "
              f"(was {kept_before}, {100.0*kept_before/n:.1f}%)",
              flush=True)

    still_dropped = sorted(t for t, c in new_totals.items() if c < FLOOR)
    print(f"\nTHINGS targets still <{FLOOR} after fallback: {len(still_dropped)}",
          flush=True)
    print(f"  examples: {still_dropped[:15]}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
