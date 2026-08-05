"""
Wikipedia Sentence Counts Checker Module

This script streams Wikipedia to determine the absolute corpus frequency of all 
unique wordforms (including pluralized nouns and inflected verbs) present in our 
target similarity datasets. It processes a maximum of 200,000 articles and splits 
the final counts into 'rare' (<100 occurrences) and 'common' (>=100 occurrences) files.
"""

import os
import re
import time
from datasets import load_dataset
import sys

# Ensure src is in path to import data_prep
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_and_standardize_datasets, load_simlex999

def main():
    print("Loading similarity datasets to extract unique wordforms...")
    train_data = load_and_standardize_datasets()
    test_data = load_simlex999()

    unique_words = set()
    for w1, w2, _ in train_data + test_data:
        unique_words.add(w1.lower())
        unique_words.add(w2.lower())

    print(f"Total unique words to track (including all inflections): {len(unique_words)}")
    counts = {w: 0 for w in unique_words}
    
    # Target count used as the threshold between 'rare' and 'common' words
    target_count = 100

    print("Streaming Wikipedia...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    start_time = time.time()
    articles_processed = 0

    for article in dataset:
        text = article['text'].lower()
        # Roughly split into sentences
        sentences = re.split(r'[.!?]+', text)
        for s in sentences:
            # Extract words consisting of lowercase letters and hyphens
            words_in_s = set(re.findall(r'\b[a-z\-]+\b', s))
            found = words_in_s.intersection(unique_words)
            for w in found:
                counts[w] += 1
                    
        articles_processed += 1
        
        if articles_processed % 5000 == 0:
            remaining = sum(1 for w, c in counts.items() if c < target_count)
            print(f"Processed {articles_processed} articles. Words still needing sentences (<100): {remaining}/{len(unique_words)}")
                
        # Limit to processing at most 200,000 articles to avoid running forever
        # 200k articles is ~3% of English Wikipedia, which contains plenty of sentences.
        if articles_processed >= 200000:
            print("Reached limit of 200,000 articles.")
            break

    rare_words = {w: c for w, c in counts.items() if c < target_count}
    common_words = {w: c for w, c in counts.items() if c >= target_count}
    
    os.makedirs("results", exist_ok=True)
    
    # Save Rare Words
    rare_path = os.path.join("results", "rare_words_counts.txt")
    with open(rare_path, "w", encoding="utf-8") as f:
        f.write(f"Total unique words checked: {len(unique_words)}\n")
        f.write(f"Words with fewer than {target_count} sentences: {len(rare_words)}\n\n")
        # Sort by count ascending
        sorted_rare = sorted(rare_words.items(), key=lambda x: x[1])
        for w, c in sorted_rare:
            f.write(f"{w}: {c} sentences\n")
            
    # Save Common Words
    common_path = os.path.join("results", "common_words_counts.txt")
    with open(common_path, "w", encoding="utf-8") as f:
        f.write(f"Total unique words checked: {len(unique_words)}\n")
        f.write(f"Words with {target_count} or more sentences: {len(common_words)}\n\n")
        # Sort by count descending for common words
        sorted_common = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
        for w, c in sorted_common:
            f.write(f"{w}: {c} sentences\n")

    print(f"Finished. Found {len(rare_words)} rare words and {len(common_words)} common words.")
    print(f"Results saved to {rare_path} and {common_path}.")

if __name__ == "__main__":
    main()
