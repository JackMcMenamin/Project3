"""Re-harvest target-word sentences from the local enwiki dump, matching the original harvest.

harvest_sentences.py built the current pool from AllCombined.txt (Simple English
Wikipedia) with fairly loose acceptance. This applies the rules from the original
generate_wiki_targetwords.py to the full enwiki dump we already have on disk:

  - source is data/enwiki_namespace_0/ for every target, not just THINGS fallbacks
  - the target must appear as an all-lowercase token, so "apple" doesn't match
    Apple the company or a title like "The Ant and the Aardvark"
  - sentences must end in . ! ? and be 5-100 words
  - the "Notes  References  Sources" artifact is dropped
  - at most ONE sentence per article per target, so a word's contexts come from
    many documents instead of whichever article happened to mention it a lot

Always-capitalised targets (iPod, SIM card, Christmas tree) would be wiped out by
the lowercase rule, so every sentence is also kept in a relaxed case-insensitive
bucket. At write time a target uses its strict pool if that reached FLOOR, and
falls back to the relaxed one otherwise - recorded per target in the manifest so
the fallback is never silent.

Writes artifacts/sentences_strict/<slug>.jsonl, same format as the original
harvest, so the trainer can point at it with --sentences-dir.

    python -u experiments/contextual/harvest_sentences_strict.py
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import ahocorasick

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
ENWIKI_DIR = ROOT / "data" / "enwiki_namespace_0"
SRC_SENTENCES = ARTIFACTS / "sentences"          # defines the target inventory
OUT_DIR = ARTIFACTS / "sentences_strict"
CHECKPOINT = ARTIFACTS / ".harvest_strict_checkpoint.pkl"
MANIFEST = ARTIFACTS / "sentences_strict_manifest.csv"

MAX_OCCURRENCES = 50
FLOOR = 10                  # below this, fall back to the relaxed pool
MIN_WORDS, MAX_WORDS = 5, 100

SENT_SPLIT = re.compile(r"(?<=[.!?]) +")
WHITESPACE = re.compile(r"\s+")
HAS_UPPER = re.compile(r"[A-Z]")
TOKEN = re.compile(r"\S+")
BIBLIO = "Notes  References  Sources"


def slugify(target: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", target.lower())


def load_targets() -> dict[str, str]:
    """{normalised target -> slug}, taken from the existing harvest.

    The target is read back out of each file's first record rather than derived
    from the filename: slugify turns spaces into underscores, so the slug alone
    can't tell "baseball glove" from a hypothetical "baseball_glove".
    """
    out = {}
    for p in sorted(SRC_SENTENCES.glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            first = f.readline().strip()
        if not first:
            continue
        target = json.loads(first)["target"]
        out[WHITESPACE.sub(" ", target.lower().strip())] = p.stem
    return out


def build_automaton(keys) -> ahocorasick.Automaton:
    A = ahocorasick.Automaton()
    for k in keys:
        A.add_word(k, k)
    A.make_automaton()
    return A


def is_boundary(s: str, i: int) -> bool:
    if i < 0 or i >= len(s):
        return True
    c = s[i]
    return not (c.isalnum() or c == "_")


def matches(text: str, A: ahocorasick.Automaton) -> set[str]:
    hits = set()
    for end, key in A.iter(text):
        start = end - len(key) + 1
        if is_boundary(text, start - 1) and is_boundary(text, end + 1):
            hits.add(key)
    return hits


def lowercase_only(sentence: str) -> str:
    """Sentence with every capitalised token blanked out.

    Matching against this is how we enforce the original lowercase-token rule while
    still supporting multiword targets - the blanked tokens can't participate in
    a phrase match.
    """
    return TOKEN.sub(lambda m: "\x00" * len(m.group(0))
                     if HAS_UPPER.search(m.group(0)) else m.group(0).lower(),
                     sentence)


def iter_text_parts(parts):
    for p in parts or ():
        t = p.get("type")
        if t == "paragraph":
            if p.get("value"):
                yield p["value"]
        elif t in ("section", "list"):
            yield from iter_text_parts(p.get("has_parts", []))
        elif t == "list_item":
            if p.get("value"):
                yield p["value"]
            yield from iter_text_parts(p.get("has_parts", []))


def iter_article_text(obj: dict):
    for s in obj.get("sections", []):
        yield from iter_text_parts(s.get("has_parts", []))


def sentences_of(text: str):
    """The original splitter: newlines first so lists and headers don't merge."""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in SENT_SPLIT.split(line):
            s = s.strip()
            if not s.endswith((".", "!", "?")):
                continue
            if not (MIN_WORDS <= len(s.split()) <= MAX_WORDS):
                continue
            if BIBLIO in s:
                continue
            yield s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, default=0,
                    help="stop after N shards (0 = all); useful for a quick look")
    args = ap.parse_args()

    targets = load_targets()
    print(f"targets: {len(targets):,}", flush=True)
    A = build_automaton(targets)

    strict: dict[str, list[str]] = defaultdict(list)
    relaxed: dict[str, list[str]] = defaultdict(list)
    done_shards: set[str] = set()

    if CHECKPOINT.exists():
        with CHECKPOINT.open("rb") as f:
            ck = pickle.load(f)
        strict.update(ck["strict"])
        relaxed.update(ck["relaxed"])
        done_shards = ck["shards"]
        print(f"resumed: {len(done_shards)} shards, "
              f"{sum(map(len, strict.values())):,} strict sentences", flush=True)

    shards = sorted(ENWIKI_DIR.glob("*.jsonl"))
    remaining = [p for p in shards if p.name not in done_shards]
    if args.shards:
        remaining = remaining[:args.shards]
    print(f"shards: {len(remaining)} of {len(shards)} remaining", flush=True)

    t0 = time.time()
    for idx, shard in enumerate(remaining, 1):
        s0, added = time.time(), 0
        n_articles = 0
        with shard.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_articles += 1
                # one sentence per article per target
                claimed_strict: set[str] = set()
                claimed_relaxed: set[str] = set()
                for text in iter_article_text(obj):
                    if not text:
                        continue
                    low = WHITESPACE.sub(" ", text.lower())
                    if not matches(low, A):
                        continue                    # cheap reject on the paragraph
                    for sent in sentences_of(text):
                        flat = WHITESPACE.sub(" ", sent)
                        r_hits = matches(flat.lower(), A) - claimed_relaxed
                        if not r_hits:
                            continue
                        s_hits = matches(lowercase_only(flat), A) - claimed_strict
                        for t in r_hits:
                            if len(relaxed[t]) < MAX_OCCURRENCES:
                                relaxed[t].append(sent)
                                claimed_relaxed.add(t)
                        for t in s_hits:
                            if len(strict[t]) < MAX_OCCURRENCES:
                                strict[t].append(sent)
                                claimed_strict.add(t)
                                added += 1

        done_shards.add(shard.name)
        sat = sum(1 for t in targets if len(strict[t]) >= MAX_OCCURRENCES)
        el = time.time() - t0
        print(f"[{idx:>2}/{len(remaining)}] {shard.name}: {n_articles:,} articles, "
              f"+{added:,} strict, saturated {sat:,}/{len(targets):,}, "
              f"{time.time()-s0:.0f}s (total {el/60:.1f} min)", flush=True)

        tmp = CHECKPOINT.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump({"strict": dict(strict), "relaxed": dict(relaxed),
                         "shards": done_shards}, f)
        tmp.replace(CHECKPOINT)

        if sat == len(targets):
            print("all targets saturated - stopping early", flush=True)
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, slug in targets.items():
        pool, mode = strict[target], "strict"
        if len(pool) < FLOOR and len(relaxed[target]) > len(pool):
            pool, mode = relaxed[target], "relaxed"
        with (OUT_DIR / f"{slug}.jsonl").open("w", encoding="utf-8") as f:
            for s in pool[:MAX_OCCURRENCES]:
                f.write(json.dumps({"target": target, "sentence": s,
                                    "source": "enwiki"}, ensure_ascii=False) + "\n")
        rows.append({"target": target, "slug": slug, "n": len(pool[:MAX_OCCURRENCES]),
                     "mode": mode, "n_strict": len(strict[target]),
                     "n_relaxed": len(relaxed[target])})

    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "slug", "n", "mode",
                                          "n_strict", "n_relaxed"])
        w.writeheader()
        w.writerows(rows)

    total = sum(r["n"] for r in rows)
    fell_back = sum(1 for r in rows if r["mode"] == "relaxed")
    thin = sum(1 for r in rows if r["n"] < FLOOR)
    print(f"\nwrote {len(rows):,} targets, {total:,} sentences -> {OUT_DIR}")
    print(f"  fell back to relaxed: {fell_back}")
    print(f"  still under {FLOOR}:     {thin}")
    print(f"  manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
