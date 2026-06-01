"""
Stage B — Contextual vector extraction.

For each target with cached sentences (from harvest_sentences.py), produce
one 768-d contextual embedding per (model, target):

    1. Load all sentences for the target.
    2. For each sentence:
         a. Tokenise the sentence with the model's tokenizer.
         b. Locate the target span in the tokenisation by matching the
            target's own subword sequence (with leading space for GPT-2).
         c. Run the model, take the last hidden layer.
         d. Mean-pool over the target's subword positions.
       Skip the sentence on tokenisation mismatch.
    3. Mean-average the per-sentence target vectors → one cached 768-d vector.

Output:
    cache/{model}/vectors.npz       one array per target (keyed by slug)
    cache/{model}/metadata.csv      per-target: n_used, n_failed, sources

Run with:  python -u extract_contextual_vectors.py --model bert
           python -u extract_contextual_vectors.py --model gpt2
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]  # experiments/contextual/ -> repo root
ARTIFACTS = ROOT / "artifacts"
SENTENCES_DIR = ARTIFACTS / "sentences"
MANIFEST = ARTIFACTS / "sentences_manifest.csv"
CACHE_DIR = ARTIFACTS / "cache"

MIN_SENTENCES_PER_TARGET = 5  # below this, drop the target
BATCH_SIZE = 32

MODEL_CONFIG = {
    "bert": {
        "hf_name": "bert-base-uncased",
        "leading_space": False,
    },
    "gpt2": {
        "hf_name": "gpt2",
        "leading_space": True,
    },
}


def slugify(target: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", target.lower())


def find_subword_span(
    sent_ids: list[int],
    target_ids: list[int],
) -> tuple[int, int] | None:
    """Find the first occurrence of `target_ids` as a contiguous slice in `sent_ids`.

    Returns (start, end_exclusive) or None if not found.
    """
    n, m = len(sent_ids), len(target_ids)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if sent_ids[i:i + m] == target_ids:
            return i, i + m
    return None


def encode_target_variants(
    tokenizer, target: str, leading_space: bool
) -> list[list[int]]:
    """Return the encoding variants to try for span location, in priority order.

    For GPT-2 BPE we generate the cross-product of:
      * leading-space vs no-leading-space  (mid-sentence vs sentence-start/after-quote)
      * lowercase, Titlecase, Title For Each Word    (since BPE is case-sensitive)
    For BERT WordPiece, the tokenizer is uncased and leading space has no effect,
    so a single lowercase variant suffices.
    """
    if not leading_space:
        # BERT-style: tokenizer handles case; one variant covers all sentences.
        return [tokenizer(target, add_special_tokens=False)["input_ids"]]

    case_forms = [target, target.capitalize(), target.title(), target.upper()]
    # Handle camelCase brands ("iPod", "iPad"): if target is single-token and
    # all-lowercase, try the iX-style form.
    if target.islower() and len(target) >= 2 and " " not in target:
        camel = target[0] + target[1].upper() + target[2:]
        if camel not in case_forms:
            case_forms.append(camel)
    # Handle abbreviation-prefixed MWEs ("CD player", "SIM card", "TV show"):
    # for each word in a multi-word target, also try the form where that word
    # is uppercased and the others stay lowercase.
    if " " in target:
        words = target.split()
        for idx in range(len(words)):
            mixed = " ".join(w.upper() if i == idx else w.lower()
                             for i, w in enumerate(words))
            if mixed not in case_forms:
                case_forms.append(mixed)
    seen: list[list[int]] = []
    for prefix in (" ", ""):
        for form in case_forms:
            ids = tokenizer(prefix + form, add_special_tokens=False)["input_ids"]
            if ids and ids not in seen:
                seen.append(ids)
    return seen


def find_first_span(
    sent_ids: list[int],
    target_variants: list[list[int]],
) -> tuple[int, int] | None:
    """Try each target variant in order; return first match, or None."""
    for tids in target_variants:
        sp = find_subword_span(sent_ids, tids)
        if sp is not None:
            return sp
    return None


def load_sentences(target: str) -> list[dict]:
    path = SENTENCES_DIR / f"{slugify(target)}.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def extract_vector_for_target(
    model,
    tokenizer,
    sentences: list[dict],
    target: str,
    leading_space: bool,
    device: str,
    pad_id: int,
) -> tuple[np.ndarray | None, int, int]:
    """Return (mean_vector, n_used, n_failed_span) or (None, n_used, n_failed)
    if fewer than MIN_SENTENCES_PER_TARGET sentences could be successfully embedded.

    Tokenises each sentence once, locates the target span, then runs the model
    in batches over only the sentences whose span was found.
    """
    target_variants = encode_target_variants(tokenizer, target, leading_space)
    if not target_variants:
        return None, 0, len(sentences)

    raw_ids: list[list[int]] = []
    spans: list[tuple[int, int] | None] = []
    for s in sentences:
        ids = tokenizer(
            s["sentence"], add_special_tokens=True,
            truncation=True, max_length=128,
        )["input_ids"]
        raw_ids.append(ids)
        spans.append(find_first_span(ids, target_variants))

    keep_idx = [i for i, sp in enumerate(spans) if sp is not None]
    n_failed = len(sentences) - len(keep_idx)
    if not keep_idx:
        return None, 0, n_failed

    used_vecs: list[np.ndarray] = []
    with torch.no_grad():
        for batch_start in range(0, len(keep_idx), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(keep_idx))
            batch_idx = keep_idx[batch_start:batch_end]
            batch_ids = [raw_ids[i] for i in batch_idx]
            max_len = max(len(ids) for ids in batch_ids)
            input_ids = torch.full(
                (len(batch_ids), max_len), pad_id, dtype=torch.long, device=device,
            )
            attention = torch.zeros(
                (len(batch_ids), max_len), dtype=torch.long, device=device,
            )
            for j, ids in enumerate(batch_ids):
                input_ids[j, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                attention[j, :len(ids)] = 1

            outputs = model(input_ids=input_ids, attention_mask=attention)
            last_hidden = outputs.last_hidden_state  # (B, L, H)
            for j, orig_i in enumerate(batch_idx):
                start, end = spans[orig_i]
                seq_len = int(attention[j].sum().item())
                if start >= seq_len:
                    n_failed += 1
                    continue
                end_clamped = min(end, seq_len)
                vec = last_hidden[j, start:end_clamped].mean(dim=0).cpu().numpy()
                used_vecs.append(vec)

    if len(used_vecs) < MIN_SENTENCES_PER_TARGET:
        return None, len(used_vecs), n_failed
    mean_vec = np.mean(used_vecs, axis=0).astype(np.float32)
    return mean_vec, len(used_vecs), n_failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIG), required=True)
    args = parser.parse_args()

    cfg = MODEL_CONFIG[args.model]
    out_dir = CACHE_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg['hf_name']} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    if args.model == "gpt2" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    model = AutoModel.from_pretrained(cfg["hf_name"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  device: {device}", flush=True)

    targets: list[str] = []
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            targets.append(row["target"])
    print(f"Targets to embed: {len(targets):,}", flush=True)

    vectors: dict[str, np.ndarray] = {}
    metadata_rows: list[dict] = []
    dropped: list[str] = []

    t0 = time.time()
    for i, target in enumerate(targets, start=1):
        sentences = load_sentences(target)
        if len(sentences) < MIN_SENTENCES_PER_TARGET:
            dropped.append(target)
            metadata_rows.append({
                "target": target,
                "slug": slugify(target),
                "n_sentences_available": len(sentences),
                "n_used": 0,
                "n_failed_span_location": 0,
                "n_allcombined": sum(1 for s in sentences if s["source"] == "allcombined"),
                "n_enwiki": sum(1 for s in sentences if s["source"] == "enwiki"),
                "kept": False,
                "drop_reason": "too_few_sentences",
            })
            if i % 100 == 0 or i == len(targets):
                print(f"  [{i}/{len(targets)}] {target!r}: dropped (only {len(sentences)} sentences)", flush=True)
            continue

        vec, n_used, n_failed = extract_vector_for_target(
            model, tokenizer, sentences, target,
            leading_space=cfg["leading_space"], device=device,
            pad_id=pad_id,
        )
        n_ac = sum(1 for s in sentences if s["source"] == "allcombined")
        n_en = sum(1 for s in sentences if s["source"] == "enwiki")

        if vec is None:
            dropped.append(target)
            metadata_rows.append({
                "target": target,
                "slug": slugify(target),
                "n_sentences_available": len(sentences),
                "n_used": n_used,
                "n_failed_span_location": n_failed,
                "n_allcombined": n_ac,
                "n_enwiki": n_en,
                "kept": False,
                "drop_reason": "too_few_valid_after_tokenisation",
            })
        else:
            vectors[slugify(target)] = vec
            metadata_rows.append({
                "target": target,
                "slug": slugify(target),
                "n_sentences_available": len(sentences),
                "n_used": n_used,
                "n_failed_span_location": n_failed,
                "n_allcombined": n_ac,
                "n_enwiki": n_en,
                "kept": True,
                "drop_reason": "",
            })

        if i % 100 == 0 or i == len(targets):
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(targets) - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i:>4d}/{len(targets)}] {elapsed/60:.1f} min, "
                  f"{rate:.1f} targets/s, "
                  f"kept {len(vectors)}, dropped {len(dropped)}, "
                  f"ETA {eta:.1f} min", flush=True)

    # Save vectors and metadata.
    npz_path = out_dir / "vectors.npz"
    np.savez(npz_path, **vectors)
    print(f"\nSaved {len(vectors)} vectors to {npz_path}", flush=True)

    meta_path = out_dir / "metadata.csv"
    fields = ["target", "slug", "n_sentences_available", "n_used",
              "n_failed_span_location", "n_allcombined", "n_enwiki",
              "kept", "drop_reason"]
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(metadata_rows)
    print(f"Wrote metadata to {meta_path}", flush=True)

    print(f"\nSummary:", flush=True)
    print(f"  total targets: {len(targets)}", flush=True)
    print(f"  vectors cached: {len(vectors)}", flush=True)
    print(f"  dropped: {len(dropped)}", flush=True)
    if dropped:
        print(f"  dropped examples: {dropped[:10]}", flush=True)


if __name__ == "__main__":
    main()
