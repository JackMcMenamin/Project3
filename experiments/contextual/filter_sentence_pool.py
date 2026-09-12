"""Filter the existing harvested pool without re-reading the dump.

The full re-harvest (harvest_sentences.py --strict) is the proper fix, but it
takes hours. This does the two filters that need no corpus access, so we can get
a number the same evening:

  - drop sentences that don't end in . ! ?
  - drop sentences where the target only appears capitalised mid-sentence, which
    means we matched a name or title (Apple, The Ant and the Aardvark) and not
    the common noun
  - drop exact duplicates under one target

Deliberately NOT matching the original filter here: it also drops sentences where the target is
only capitalised at the start, because its regex takes lowercase tokens only.
That's orthography rather than a wrong sense, so keeping those costs nothing and
saves ~3% of the pool.

Targets left with fewer than MIN_KEEP sentences are written anyway but reported,
since a thin pool is its own problem for the averaged word vector.

    python experiments/contextual/filter_sentence_pool.py
    python experiments/contextual/filter_sentence_pool.py --out artifacts/sentences_clean
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "sentences"
MIN_KEEP = 10


def keep(sentence: str, head: str) -> bool:
    if not sentence.rstrip().endswith((".", "!", "?")):
        return False
    # lowercase form present -> common noun sense, keep
    if re.search(rf"\b{re.escape(head)}\b", sentence):
        return True
    hits = list(re.finditer(rf"\b{re.escape(head)}\b", sentence, re.I))
    if not hits:
        return False          # target isn't really in here
    # only capitalised: fine at the very start, a name anywhere else
    return all(h.start() == 0 for h in hits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts/sentences_clean",
                    help="output directory, relative to the repo root")
    args = ap.parse_args()

    dst = ROOT / args.out
    dst.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    thin: list[tuple[str, int]] = []
    emptied: list[str] = []
    relaxed: list[str] = []
    files = sorted(SRC.glob("*.jsonl"))

    for path in files:
        head = path.stem.split("_")[0]
        recs, seen = [], set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_in += 1
                if rec["sentence"] in seen:
                    continue
                seen.add(rec["sentence"])
                recs.append(rec)

        punctuated = [r for r in recs if r["sentence"].rstrip().endswith((".", "!", "?"))]
        rows = [r for r in punctuated if keep(r["sentence"], head)]
        # Some words are always capitalised in real use - iPod, SIM card, Christmas
        # tree. For those the lowercase rule removes the word rather than the wrong
        # sense, so fall back to punctuation-only rather than emptying the target.
        if len(rows) < MIN_KEEP <= len(punctuated):
            rows = punctuated
            relaxed.append(path.stem)

        n_out += len(rows)
        with (dst / path.name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if not rows:
            emptied.append(path.stem)
        elif len(rows) < MIN_KEEP:
            thin.append((path.stem, len(rows)))

    print(f"targets      {len(files):,}")
    print(f"sentences    {n_in:,} -> {n_out:,}  ({100*(n_in-n_out)/n_in:.1f}% dropped)")
    print(f"relaxed      {len(relaxed)} always-capitalised targets kept on punctuation only")
    print(f"emptied      {len(emptied)} targets now have no sentences")
    print(f"thin         {len(thin)} targets left under {MIN_KEEP}")
    if relaxed:
        print("  relaxed:", ", ".join(relaxed[:12]) + (" ..." if len(relaxed) > 12 else ""))
    if emptied:
        print("  emptied:", ", ".join(emptied[:12]) + (" ..." if len(emptied) > 12 else ""))
    if thin:
        preview = ", ".join(f"{t}({n})" for t, n in thin[:12])
        print("  thin:   ", preview + (" ..." if len(thin) > 12 else ""))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
