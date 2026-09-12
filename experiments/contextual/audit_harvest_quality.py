"""Audit the harvested sentence pool against the original Wikipedia harvest filters.

harvest_sentences.py keeps any punkt sentence of 5-64 tokens that contains the
target. The original generate_wiki_targetwords.py is stricter, so this counts how
much of our pool would not survive its rules and flags a few things neither
pipeline checks for.

What gets counted, per target file in artifacts/sentences/:

  no_terminal      sentence doesn't end in . ! ? - the original hard filter, and the one
                   that drops headings, list items and table fragments
  capitalised_only the target only ever appears capitalised in the sentence, so
                   it's the proper noun (Apple, Turkey, Bow) not the common noun
  mojibake         UTF-8 read as latin-1 somewhere upstream (CosA-not-i etc.)
  biblio           the "Notes  References  Sources" artifact it strips
  duplicate        the exact same sentence twice under one target
  context_repeat   proxy for the original one-sentence-per-article rule: sentences that
                   share a rare content word with another sentence for the same
                   target, which usually means they came from the same document

    python experiments/contextual/audit_harvest_quality.py [--sample N]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SENTENCES = ROOT / "artifacts" / "sentences"

MOJIBAKE = re.compile(r"Ã.|â€|Â.")
WORD = re.compile(r"[A-Za-z]+")
# words too common to signal a shared source document
STOP = set("""the a an and or but of in on at to for with from by as is was are were be been
being this that these those it its he she they them his her their we you i not no than then
which who whom whose what when where why how all any both each few more most other some such
only own same so too very can will just should now had has have do does did done one two into
over after before between during about against also may would could""".split())


def load(path: Path) -> list[str]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["sentence"])
    return out


def rare_words(sent: str) -> set[str]:
    return {w.lower() for w in WORD.findall(sent)
            if len(w) > 4 and w.lower() not in STOP}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None,
                    help="pool to audit, relative to the repo root "
                         "(default artifacts/sentences)")
    ap.add_argument("--sample", type=int, default=0,
                    help="only audit the first N target files (0 = all)")
    ap.add_argument("--examples", type=int, default=4,
                    help="how many offending sentences to print per category")
    args = ap.parse_args()

    pool = (ROOT / args.dir) if args.dir else SENTENCES
    files = sorted(pool.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no sentence files in {pool}")
    if args.sample:
        files = files[:args.sample]
    print(f"pool: {pool}")

    n_sent = n_targets = 0
    counts = Counter()
    examples: dict[str, list[tuple[str, str]]] = {k: [] for k in
        ("no_terminal", "cap_midsentence", "cap_initial", "mojibake", "biblio",
         "duplicate", "context_repeat")}
    per_target_repeat = []
    lengths = []

    for path in files:
        sents = load(path)
        if not sents:
            continue
        target = path.stem.replace("_", " ")
        n_targets += 1
        n_sent += len(sents)
        seen: set[str] = set()
        sigs: list[set[str]] = []
        repeats = 0

        for s in sents:
            lengths.append(len(s.split()))

            if not s.rstrip().endswith((".", "!", "?")):
                counts["no_terminal"] += 1
                if len(examples["no_terminal"]) < args.examples:
                    examples["no_terminal"].append((target, s))

            # Does the target ever appear in lowercase form in this sentence?
            # Split the capitalised-only cases in two: a capital at the very
            # start of the sentence is just orthography, a capital mid-sentence
            # means we've matched a name or title. The original lowercase-token rule
            # drops both, but only the second is genuine sense contamination.
            head = target.split()[0]
            if head and re.search(rf"\b{re.escape(head)}\b", s) is None:
                hits = list(re.finditer(rf"\b{re.escape(head)}\b", s, re.I))
                if hits:
                    counts["capitalised_only"] += 1
                    key = "cap_midsentence" if any(h.start() > 0 for h in hits) \
                        else "cap_initial"
                    counts[key] += 1
                    if len(examples[key]) < args.examples:
                        examples[key].append((target, s))

            if MOJIBAKE.search(s):
                counts["mojibake"] += 1
                if len(examples["mojibake"]) < args.examples:
                    examples["mojibake"].append((target, s))

            if "Notes  References  Sources" in s:
                counts["biblio"] += 1
                if len(examples["biblio"]) < args.examples:
                    examples["biblio"].append((target, s))

            if s in seen:
                counts["duplicate"] += 1
                if len(examples["duplicate"]) < args.examples:
                    examples["duplicate"].append((target, s))
            seen.add(s)

            # shared rare vocabulary with an earlier sentence for this target
            sig = rare_words(s) - {target}
            if sig and any(len(sig & prev) >= 2 for prev in sigs):
                counts["context_repeat"] += 1
                repeats += 1
                if len(examples["context_repeat"]) < args.examples:
                    examples["context_repeat"].append((target, s))
            sigs.append(sig)

        per_target_repeat.append(repeats / len(sents))

    lengths.sort()
    print(f"\nAudited {n_targets:,} targets, {n_sent:,} sentences")
    print(f"length: median {lengths[len(lengths)//2]} words, "
          f"p90 {lengths[int(len(lengths)*0.9)]}, max {lengths[-1]}")
    print("=" * 62)
    print(f"{'check':<20}{'sentences':>12}{'share':>10}   fails original filter?")
    print("-" * 62)
    verdict = {
        "no_terminal": "yes - it requires . ! ?",
        "capitalised_only": "yes - it matches lowercase tokens only",
        "cap_midsentence": "  of which: a name or title, wrong word sense",
        "cap_initial": "  of which: sentence-initial, sense is fine",
        "biblio": "yes - it strips it",
        "mojibake": "no - neither of us checks",
        "duplicate": "no - but its 1-per-article rule prevents it",
        "context_repeat": "no - its 1-per-article rule prevents it",
    }
    for k in ("no_terminal", "capitalised_only", "cap_midsentence", "cap_initial",
              "biblio", "mojibake", "duplicate", "context_repeat"):
        print(f"{k:<20}{counts[k]:>12,}{100*counts[k]/n_sent:>9.1f}%   {verdict[k]}")

    per_target_repeat.sort()
    print(f"\ncontext_repeat per target: median "
          f"{100*per_target_repeat[len(per_target_repeat)//2]:.0f}%, "
          f"p90 {100*per_target_repeat[int(len(per_target_repeat)*0.9)]:.0f}%")

    for k, items in examples.items():
        if not items:
            continue
        print(f"\n--- {k} ---")
        for target, s in items:
            print(f"  [{target}]  {s[:150]}")


if __name__ == "__main__":
    main()
