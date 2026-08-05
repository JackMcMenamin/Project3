"""
Extract General MLM Training Sentences from Wikipedia

This script pre-extracts a high-quality, fixed-size corpus of English Wikipedia
sentences for use as the MLM training pool during RSR experiments.

Rationale:
    The previous approach streamed Wikipedia articles live during training
    (one sentence per article, in Page ID order). This introduced three problems:
    1. Sequential topic bias: early batches were dominated by older, highly
       curated articles; later batches degraded in quality.
    2. No shuffling: batches were topically clustered.
    3. Network latency: streaming from Hugging Face shards introduced I/O stalls.

    By pre-extracting sentences to a local JSONL file, we can shuffle freely,
    guarantee uniform quality, and eliminate data-loading bottlenecks.

Extraction Strategy:
    - Stream Wikipedia articles sequentially.
    - For each article, extract all qualifying sentences (5-100 words, terminal
      punctuation, no bibliographic artifacts).
    - Discard any article that does not produce at least 5 qualifying sentences
      (filters out stubs, disambiguation pages, list-only pages, and redirects).
    - From each qualifying article, retain the first 5 qualifying sentences.
    - Continue streaming until exactly 40,000 qualifying articles have been
      collected, yielding exactly 200,000 sentences.

Output:
    data/wiki_mlm.jsonl — one JSON object per article, containing:
        - "title": the Wikipedia article title
        - "sentences": a list of exactly 5 qualifying sentences

Workflow Context:
    - Run once before training (similar to generate_wiki_targetwords.py).
    - Consumed by: train.py, which loads and shuffles the sentences at startup.
"""

import os
import re
import json
import time
from datasets import load_dataset

# Pre-compiled regex for splitting text into sentences by terminal punctuation.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?]) +')

# Number of qualifying articles to collect (each contributes exactly 5 sentences)
TARGET_ARTICLES = 120_000
SENTS_PER_ARTICLE = 5


def extract_qualifying_sentences(text, max_sentences=SENTS_PER_ARTICLE):
    """Extracts qualifying prose sentences from a Wikipedia article.

    Applies the same filtering criteria used across the codebase:
    split by newlines (to avoid merging lists/headers), split into sentences
    by punctuation, require sentence-ending punctuation, enforce 5-100 word
    count, and skip bibliographic artifacts.

    Args:
        text (str): Raw article text from the Wikipedia dump.
        max_sentences (int): Maximum number of sentences to extract.

    Returns:
        list of str: Up to max_sentences qualifying sentences, or fewer if
                     the article does not contain enough.
    """
    sentences = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        sents = _SENT_SPLIT_RE.split(line)
        for s in sents:
            s = s.strip()
            if not s:
                continue
            # Must end with sentence-terminating punctuation
            if not s.endswith(('.', '!', '?')):
                continue
            # Enforce minimum 5 words and maximum 100 words
            word_count = len(s.split())
            if word_count < 5 or word_count > 100:
                continue
            # Skip common Wikipedia bibliographic artifacts
            if "Notes  References  Sources" in s:
                continue
            sentences.append(s)
            if len(sentences) >= max_sentences:
                return sentences
    return sentences


def main():
    print(f"Target: {TARGET_ARTICLES} qualifying articles × {SENTS_PER_ARTICLE} sentences = {TARGET_ARTICLES * SENTS_PER_ARTICLE} total sentences")
    print("Streaming Wikipedia...")

    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    start_time = time.time()
    articles_scanned = 0
    articles_qualified = 0
    articles_discarded = 0
    all_records = []

    for article in dataset:
        articles_scanned += 1
        title = article.get('title', f'article_{articles_scanned}')
        text = article['text']

        sentences = extract_qualifying_sentences(text, max_sentences=SENTS_PER_ARTICLE)

        if len(sentences) < SENTS_PER_ARTICLE:
            articles_discarded += 1
        else:
            articles_qualified += 1
            all_records.append({
                "title": title,
                "sentences": sentences
            })

        if articles_scanned % 5000 == 0:
            elapsed = time.time() - start_time
            print(f"  Scanned {articles_scanned:,} articles | "
                  f"Qualified: {articles_qualified:,} | "
                  f"Discarded: {articles_discarded:,} | "
                  f"Elapsed: {elapsed:.1f}s")

        if articles_qualified >= TARGET_ARTICLES:
            break

    total_sentences = sum(len(r["sentences"]) for r in all_records)
    elapsed = time.time() - start_time

    print(f"\nExtraction complete:")
    print(f"  Articles scanned:    {articles_scanned:,}")
    print(f"  Articles qualified:  {articles_qualified:,}")
    print(f"  Articles discarded:  {articles_discarded:,}")
    print(f"  Total sentences:     {total_sentences:,}")
    print(f"  Elapsed time:        {elapsed:.1f}s")

    # Save to JSONL
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "wiki_mlm.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
