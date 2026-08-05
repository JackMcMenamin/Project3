"""
Extract Wikipedia Sentences for Target Words

This script streams Wikipedia, searches for sentences containing the target words 
from our similarity datasets, and saves them to `data/wiki_targetwords.jsonl`.
It collects 60 sentences per word, which are split equally into training (first 30) 
and evaluation (second 30) pools during downstream pipelines to prevent data leakage.
Word matching is restricted to strict lowercase tokens to match common noun/verb senses 
rather than sentence-initial or entity-specific capitalised proper nouns (e.g. "apple" vs "Apple").
"""

import os
import re
import json
import time
from datasets import load_dataset
import sys

# Pre-compiled regex for splitting text into sentences by terminal punctuation.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?]) +')

# Ensure src is in path to import data_prep
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_and_standardize_datasets, load_simlex999

def main():
    print("Loading similarity datasets to extract unique wordforms...")
    train_data = load_and_standardize_datasets()
    test_data = load_simlex999()

    unique_words = set()
    for w1, w2, _ in train_data + test_data:
        unique_words.add(w1)
        unique_words.add(w2)

    print(f"Total unique words to track: {len(unique_words)}")
    
    # Store collected sentences per word
    collected_sentences = {w: [] for w in unique_words}
    target_sentences_per_word = 60  # 60 sentences split equally into 30 train + 30 eval
    
    print("Streaming Wikipedia...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    start_time = time.time()
    articles_processed = 0
    words_completed = 0

    for article in dataset:
        text = article['text']
        
        # Split by newlines first to avoid merging lists and headers into prose
        raw_lines = text.split('\n')
        sentences = []
        for line in raw_lines:
            line = line.strip()
            if not line: continue
            # Split into actual sentences based on punctuation
            sents = _SENT_SPLIT_RE.split(line)
            for s in sents:
                s = s.strip()
                # Only keep strings that look like actual sentences (end with punctuation)
                if s and s.endswith(('.', '!', '?')):
                    # Enforce a minimum word count of 5 words
                    if len(s.split()) >= 5:
                        sentences.append(s)
        
        # Track words found in this specific article to enforce 1 sentence per article per word
        words_found_in_article = set()
        
        for s in sentences:
            s = s.strip()
            if not s: continue
            
            # Skip common Wikipedia bibliographic artifacts
            if "Notes  References  Sources" in s:
                continue
            
            # Enforce maximum 100 words (consistent with train.py quality filtering)
            if len(s.split()) > 100:
                continue
            
            # Extract only lowercase word tokens (including hyphenated forms) to ensure
            # we match the common noun sense (e.g., "apple") rather than proper nouns (e.g., "Apple").
            words_in_s = set(re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', s))
            found = words_in_s.intersection(unique_words)
            
            for w in found:
                # Only add if we haven't already grabbed a sentence for this word from this article
                if w not in words_found_in_article and len(collected_sentences[w]) < target_sentences_per_word:
                    collected_sentences[w].append(s)
                    words_found_in_article.add(w)
                    if len(collected_sentences[w]) == target_sentences_per_word:
                        words_completed += 1
                        
        articles_processed += 1
        
        if articles_processed % 5000 == 0:
            print(f"Processed {articles_processed} articles. Words completed ({target_sentences_per_word} sents): {words_completed}/{len(unique_words)}")
            
        # Stop early ONLY if all words have reached the target
        if words_completed == len(unique_words):
            break

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "wiki_targetwords.jsonl")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for w, sents in collected_sentences.items():
            if sents:
                json_line = json.dumps({"word": w, "sentences": sents})
                f.write(json_line + "\n")
                
    total_sentences = sum(len(s) for s in collected_sentences.values())
    print(f"Finished extracting {total_sentences} sentences across {len(unique_words)} target words.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
