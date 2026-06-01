"""
Coverage probe: how many of our supervision/eval targets appear (and how often)
in AllCombined.txt? Streams the corpus once, counts whole-word/MWE occurrences
per target, writes a CSV summary and prints a coverage table.

Sources:
  - MEN          : data/MEN/MEN/MEN_dataset_natural_form_full
  - SimVerb-3500 : data/simverb-3500-data/data/SimVerb-3500.txt
  - THINGS       : things_similarity/variables/unique_id.txt    (underscored MWEs)
  - SimLex-999   : SimLex-999/SimLex-999.txt
"""
from __future__ import annotations

import csv
import re
import time
from collections import defaultdict
from pathlib import Path

# Repo root is three levels up: experiments/contextual/coverage_probe.py.
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
CORPUS = ROOT / "data" / "AllCombined.txt"
OUT_CSV = ARTIFACTS / "coverage_probe.csv"

MEN_FILE     = ROOT / "data" / "MEN" / "MEN" / "MEN_dataset_natural_form_full"
SIMVERB_FILE = ROOT / "data" / "simverb-3500-data" / "data" / "SimVerb-3500.txt"
THINGS_FILE  = ROOT / "things_similarity" / "variables" / "unique_id.txt"
SIMLEX_FILE  = ROOT / "data" / "SimLex-999" / "SimLex-999.txt"


def load_targets() -> dict[str, set[str]]:
    """Return {dataset_name: set(target_strings)}.

    Targets are stored in the form they will be matched against the corpus:
    lowercase, with MWEs as space-separated tokens (e.g. 'ice cream').
    """
    targets: dict[str, set[str]] = {}

    # MEN: "word1 word2 score"  (single tokens)
    men: set[str] = set()
    with MEN_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                men.add(parts[0].lower())
                men.add(parts[1].lower())
    targets["MEN"] = men

    # SimVerb: "verb1\tverb2\tPOS\tscore\tRELATION"
    sv: set[str] = set()
    with SIMVERB_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sv.add(parts[0].lower())
                sv.add(parts[1].lower())
    targets["SimVerb"] = sv

    # THINGS: one concept per line, MWEs joined by underscore
    things: set[str] = set()
    with THINGS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip().lower()
            if tok:
                things.add(tok.replace("_", " "))
    targets["THINGS"] = things

    # SimLex-999: header line then "word1\tword2\tPOS\t..."
    sl: set[str] = set()
    with SIMLEX_FILE.open("r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sl.add(parts[0].lower())
                sl.add(parts[1].lower())
    targets["SimLex"] = sl

    return targets


def build_matchers(all_targets: set[str]) -> tuple[set[str], list[str]]:
    """Split targets into single-token and multi-token sets for fast matching."""
    singles = {t for t in all_targets if " " not in t}
    multis = sorted((t for t in all_targets if " " in t), key=len, reverse=True)
    return singles, multis


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def count_occurrences(
    corpus_path: Path,
    singles: set[str],
    multis: list[str],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)

    # Pre-build a regex for MWEs: word boundaries, allow flexible whitespace.
    if multis:
        mwe_pattern = re.compile(
            r"\b(" + "|".join(re.escape(m).replace(r"\ ", r"\s+") for m in multis) + r")\b",
            re.IGNORECASE,
        )
    else:
        mwe_pattern = None

    line_count = 0
    t0 = time.time()
    with corpus_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            if not line.strip():
                continue
            lower = line.lower()

            # Single-token matches: tokenise and check membership.
            for tok in WORD_RE.findall(lower):
                if tok in singles:
                    counts[tok] += 1

            # MWE matches: regex over the line.
            if mwe_pattern is not None:
                for m in mwe_pattern.findall(lower):
                    counts[" ".join(m.split())] += 1

            if line_count % 200_000 == 0:
                elapsed = time.time() - t0
                print(f"  ... {line_count:,} lines  ({elapsed:.1f}s)")

    return dict(counts)


def summarise(
    name: str,
    targets: set[str],
    counts: dict[str, int],
    floors: tuple[int, ...] = (1, 5, 10, 25, 50),
) -> dict:
    n = len(targets)
    hits = {t: counts.get(t, 0) for t in targets}
    rows = sorted(hits.items(), key=lambda kv: kv[1])
    summary = {
        "dataset": name,
        "total_targets": n,
        "total_occurrences": sum(hits.values()),
    }
    for f in floors:
        summary[f"ge_{f}"] = sum(1 for c in hits.values() if c >= f)
        summary[f"pct_ge_{f}"] = 100.0 * summary[f"ge_{f}"] / n if n else 0.0
    summary["lowest_5"] = rows[:5]
    summary["highest_5"] = rows[-5:][::-1]
    return summary


def main() -> None:
    print("Loading target sets...")
    by_dataset = load_targets()
    for name, items in by_dataset.items():
        sample = sorted(items)[:5]
        print(f"  {name:8s} {len(items):>5d} unique items  (e.g. {sample})")

    all_targets: set[str] = set().union(*by_dataset.values())
    singles, multis = build_matchers(all_targets)
    print(f"\nUnion: {len(all_targets):,} unique targets "
          f"({len(singles):,} single-token, {len(multis):,} MWE)")

    print(f"\nStreaming corpus: {CORPUS}  ({CORPUS.stat().st_size / 1e6:.1f} MB)")
    counts = count_occurrences(CORPUS, singles, multis)
    print(f"  done: {sum(counts.values()):,} total target hits across "
          f"{len(counts):,} distinct targets")

    print("\n=== Coverage by dataset ===")
    summaries = []
    header = f"{'Dataset':10s} {'N':>5s} {'>=1':>6s} {'>=5':>6s} {'>=10':>6s} {'>=25':>6s} {'>=50':>6s}"
    print(header)
    print("-" * len(header))
    for name, items in by_dataset.items():
        s = summarise(name, items, counts)
        summaries.append(s)
        print(f"{name:10s} {s['total_targets']:>5d} "
              f"{s['pct_ge_1']:>5.1f}% {s['pct_ge_5']:>5.1f}% "
              f"{s['pct_ge_10']:>5.1f}% {s['pct_ge_25']:>5.1f}% "
              f"{s['pct_ge_50']:>5.1f}%")

    s_all = summarise("UNION", all_targets, counts)
    summaries.append(s_all)
    print(f"{'UNION':10s} {s_all['total_targets']:>5d} "
          f"{s_all['pct_ge_1']:>5.1f}% {s_all['pct_ge_5']:>5.1f}% "
          f"{s_all['pct_ge_10']:>5.1f}% {s_all['pct_ge_25']:>5.1f}% "
          f"{s_all['pct_ge_50']:>5.1f}%")

    print("\n=== Lowest-frequency examples (per dataset) ===")
    for s in summaries:
        if s["dataset"] == "UNION":
            continue
        zero = [t for t, c in s["lowest_5"] if c == 0]
        nonzero_low = [(t, c) for t, c in s["lowest_5"] if c > 0]
        print(f"  {s['dataset']:10s} zero-hit examples: {zero[:5]}")
        if nonzero_low:
            print(f"  {s['dataset']:10s} low-hit examples: {nonzero_low[:5]}")

    # Write per-target CSV (every target across every dataset, with its count).
    print(f"\nWriting per-target counts to {OUT_CSV}")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "target", "is_mwe", "count"])
        for name, items in by_dataset.items():
            for t in sorted(items):
                w.writerow([name, t, int(" " in t), counts.get(t, 0)])

    print("\nDone.")


if __name__ == "__main__":
    main()
