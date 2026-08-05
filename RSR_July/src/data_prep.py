"""
Data Preparation Module for Representational Similarity Regularisation (RSR)

This module handles the loading, standardization, and processing of both behavioral 
similarity datasets (e.g., WordSim-353, SimVerb-3500, THINGS, SimLex-999) and the 
Wikipedia corpora used for contextual modeling.

Workflow Context:
- The foundational data ingestion layer for the entire repository.
- Consumed by: `train.py` (to load the WordSim/THINGS/SimVerb human similarity datasets) 
  and `samplers.py` (which relies on `find_subword_span` to map strings back to RoBERTa tokens).
  
Functions:
- standardize_scores: Normalizes similarity ratings to a [0, 1] range.
- load_and_standardize_datasets: loading of training similarity datasets.
- load_simlex999: loading of the held-out SimLex-999 dataset.
- generate_wiki_corpora: Streams Wikipedia to extract sentences containing target words.
- find_subword_span: Locates token spans for specific words in tokenized sentences.
"""

import json
import os
import re
import random
from functools import lru_cache
import numpy as np
import pandas as pd
from datasets import load_dataset

def standardize_scores(data, min_score, max_score):
    """
    Standardizes a list of (word1, word2, score) tuples to the range [0, 1].

    Args:
        data (list of tuple): A list of tuples containing (word1, word2, score).
        min_score (float): The minimum possible score in the original dataset.
        max_score (float): The maximum possible score in the original dataset.

    Returns:
        list of tuple: The data list with scores normalized to [0, 1].
    """
    return [(w1, w2, (score - min_score) / (max_score - min_score)) for w1, w2, score in data]

def inspect_data(dataset_name, data):
    """
    Saves the 100 most similar and 100 least similar pairs from a dataset 
    to a text file in the results directory.
    
    Args:
        dataset_name (str): The name of the dataset.
        data (list of tuple): A list of tuples containing (word1, word2, score).
    """
    os.makedirs("results", exist_ok=True)
    # Sort data by score in descending order
    sorted_data = sorted(data, key=lambda x: x[2], reverse=True)
    
    output_file = os.path.join("results", f"{dataset_name}_inspection.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"--- {dataset_name} Data Inspection ---\n")
        f.write(f"Total pairs available: {len(data)}\n\n")
        
        f.write("Top 100 Most Similar Pairs:\n")
        for w1, w2, score in sorted_data[:100]:
            f.write(f"{w1} - {w2}: {score:.4f}\n")
            
        f.write("\nTop 100 Least Similar Pairs:\n")
        # To get least similar, we take the last 100 and reverse them so the very least is first
        for w1, w2, score in reversed(sorted_data[-100:]):
            f.write(f"{w1} - {w2}: {score:.4f}\n")
    
    print(f"Saved inspection results for {dataset_name} to {output_file}")

# A static set of common mass nouns to perfectly filter them out from pluralization.
# While inflect handles many standard rules, English mass nouns can technically have plurals 
# in specific contexts (e.g. "waters"). This explicit list prevents those edge cases.
MASS_NOUNS = {
    "rice", "water", "meat", "bread", "milk", "cheese", "butter", "sand", "dirt", 
    "mud", "snow", "ice", "wood", "gold", "silver", "iron", "steel", "glass", 
    "air", "gas", "oxygen", "smoke", "dust", "hair", "cotton", "wool", "silk", 
    "leather", "paper", "oil", "blood", "salt", "sugar", "coffee", "tea", "wine", 
    "beer", "juice", "soup", "honey", "information", "knowledge", "evidence", 
    "music", "art", "love", "happiness", "sadness", "anger", "fear", "courage", 
    "health", "wealth", "time", "money", "luggage", "baggage", "furniture", 
    "equipment", "software", "homework", "math", "history", "biology", "physics", 
    "chemistry", "garbage", "trash", "rubbish", "weather", "traffic", "news", 
    "advice", "jewelry", "clothing", "machinery", "scenery", "poetry", "prose",
    "cattle", "swine", "deer", "fish", "sheep", "moose", "salmon", "trout"
}

def augment_with_plurals(data):
    """
    Augments the dataset by adding pluralized versions of the word pairs.
    
    This function processes a list of similarity pairs and creates a parallel dataset 
    where the concept words are pluralized (e.g., 'rabbit' - 'groundhog' becomes 
    'rabbits' - 'groundhogs'), preserving the semantic similarity score.
    
    Rules applied:
    1. If a word is a recognized mass noun (e.g., 'rice'), it remains untouched.
    2. If a word is already plural (e.g., 'chips'), it remains untouched.
    3. Otherwise, the word is converted to plural using the `inflect` engine.
    
    Args:
        data (list of tuple): Original dataset of (word1, word2, score)
        
    Returns:
        list of tuple: The augmented dataset containing both original and plural pairs.
    """
    import inflect
    engine = inflect.engine()
    
    def get_plural(word):
        # 1. Protect explicit mass nouns and zero-plural nouns
        if word.lower() in MASS_NOUNS:
            return word
            
        # 2. Check if the word is already plural. 
        # `singular_noun` returns the singular string if it was plural, or False if it was singular.
        is_plural = engine.singular_noun(word)
        if is_plural is not False:
            return word # It was already plural, return it untouched
            
        # 3. Otherwise, it's a regular singular noun; pluralize it.
        return engine.plural(word)
        
    augmented_data = []
    for w1, w2, score in data:
        # Add the original singular pair
        augmented_data.append((w1, w2, score))
        
        # Determine the pluralized forms
        pw1 = get_plural(w1)
        pw2 = get_plural(w2)
        
        # Only append the new pair if at least one word actually changed.
        # This prevents identical duplicates when both words were already plural or mass nouns.
        if pw1 != w1 or pw2 != w2:
            augmented_data.append((pw1, pw2, score))
            
    return augmented_data

def augment_with_verbs(data):
    """
    Augments a dataset of verb pairs by adding their inflected forms.
    
    Generates and appends:
    1. Present singular (e.g., eat -> eats) via VBZ
    2. Past tense plural (e.g., eat -> ate, be -> were) via VBD (taking the last form)
    3. Gerundive form (e.g., eat -> eating) via VBG
    
    Args:
        data (list of tuple): Original dataset of (word1, word2, score)
        
    Returns:
        list of tuple: The augmented dataset.
    """
    import lemminflect
    
    augmented_data = []
    # Using a set to track unique pairs, avoiding duplicates if inflected forms collide
    seen = set()
    
    for w1, w2, score in data:
        # Original pair
        if (w1, w2) not in seen:
            augmented_data.append((w1, w2, score))
            seen.add((w1, w2))
            
        # Helper to extract inflection
        def get_inflection(w, tag, plural_past=False):
            # --- MANUAL NLP PATCHES ---
            # The lemminflect engine occasionally fails to generate certain
            # valid forms. For instance, when querying 'sting' for its gerund 
            # form ('VBG'), it returns None and defaults to the base form 'sting'.
            # We manually catch known edge cases here to ensure linguistic accuracy.
            if w.lower() == 'sting' and tag == 'VBG':
                return 'stinging'
                
            forms = lemminflect.getInflection(w, tag=tag)
            if not forms:
                return w
            return forms[-1] if plural_past else forms[0]
            
        # Present singular
        w1_vbz, w2_vbz = get_inflection(w1, 'VBZ'), get_inflection(w2, 'VBZ')
        if (w1_vbz, w2_vbz) not in seen:
            augmented_data.append((w1_vbz, w2_vbz, score))
            seen.add((w1_vbz, w2_vbz))
            
        # Past tense plural
        w1_vbd, w2_vbd = get_inflection(w1, 'VBD', True), get_inflection(w2, 'VBD', True)
        if (w1_vbd, w2_vbd) not in seen:
            augmented_data.append((w1_vbd, w2_vbd, score))
            seen.add((w1_vbd, w2_vbd))
            
        # Gerundive
        w1_vbg, w2_vbg = get_inflection(w1, 'VBG'), get_inflection(w2, 'VBG')
        if (w1_vbg, w2_vbg) not in seen:
            augmented_data.append((w1_vbg, w2_vbg, score))
            seen.add((w1_vbg, w2_vbg))
            
    return augmented_data

def load_wiki_word_counts():
    """
    Parses the generated Wikipedia diagnostic counting files to map 
    every augmented wordform to its observed frequency.
    """
    counts = {}
    files = [
        os.path.join("results", "rare_words_counts.txt"),
        os.path.join("results", "common_words_counts.txt")
    ]
    for f in files:
        if os.path.exists(f):
            with open(f, encoding='utf-8') as fh:
                for line in fh.readlines()[3:]:
                    if ':' in line:
                        w, c = line.split(':')
                        counts[w.strip()] = int(c.strip().split()[0])
    return counts

def filter_by_frequency(data, min_freq=10, max_freq=100000):
    """
    Excludes pairs where either word falls outside the specified frequency bounds
    across the sampled Wikipedia corpus.
    """
    counts = load_wiki_word_counts()
    if not counts:
        raise FileNotFoundError("Wikipedia word-frequency count files not found in results/. Run the counting step first.")
        
    filtered_data = []
    dropped = 0
    for w1, w2, score in data:
        c1 = counts.get(w1, 0)
        c2 = counts.get(w2, 0)
        
        if (min_freq <= c1 <= max_freq) and (min_freq <= c2 <= max_freq):
            filtered_data.append((w1, w2, score))
        else:
            dropped += 1
            
    if dropped > 0:
        print(f"Filtered {dropped} pairs out of bounds ({min_freq}-{max_freq}). Retained: {len(filtered_data)}")
    return filtered_data

def record_dataset_stats(dataset_name, original_data, expanded_data, filtered_data):
    """
    Records specific structural statistics for a dataset across the processing pipeline.
    """
    def count_unique(data):
        words = set()
        for w1, w2, _ in data:
            words.add(w1)
            words.add(w2)
        return len(words)
        
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{dataset_name}_stats.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"--- {dataset_name} Statistics ---\n")
        f.write(f"(a) Original pairs: {len(original_data)}\n")
        f.write(f"(b) Original unique words: {count_unique(original_data)}\n")
        f.write(f"(c) Inflection-expanded pairs: {len(expanded_data)}\n")
        f.write(f"(d) Inflection-expanded unique words: {count_unique(expanded_data)}\n")
        f.write(f"(e) Frequency-filtered pairs: {len(filtered_data)}\n")
        f.write(f"(f) Frequency-filtered unique words: {count_unique(filtered_data)}\n")
    print(f"Saved stats for {dataset_name} to {out_path}")

def load_and_standardize_datasets():
    """
    Loads, standardizes, augments, and filters the similarity datasets (WordSim-353, SimVerb-3500, THINGS).
    
    Processing Steps per dataset:
    1. Extract word pairs and scores.
    2. Normalize scores to [0.0, 1.0].
    3. Augment with plurals (for nouns) or inflections (for verbs).
    4. Filter out any pair containing a word with a Wikipedia frequency < 10 or > 100,000.
    5. Log expansion and filtering statistics to `results/<dataset>_stats.txt`.
    6. Combine into a unified training list.
    
    Returns:
        list of tuple: A unified list of (word1, word2, standardized_score) tuples.
    """
    print("Loading and standardizing similarity datasets...")
    training_data = []
    
    # 1. Load WordSim-353 dataset
    ws_path = os.path.join("data", "WordSim353", "win353.csv")
    if os.path.exists(ws_path):
        df_ws = pd.read_csv(ws_path)
        ws_data = df_ws[['Word 1', 'Word 2', 'Human (Mean)']].values.tolist()
        ws_std = standardize_scores(ws_data, 0.0, 10.0)
        ws_expanded = augment_with_plurals(ws_std)
        ws_filtered = filter_by_frequency(ws_expanded, min_freq=10, max_freq=100000)
        record_dataset_stats("WordSim353", ws_std, ws_expanded, ws_filtered)
        inspect_data("WordSim353", ws_filtered)
        training_data.extend(ws_filtered)
        print(f"Loaded {len(ws_filtered)} pairs from WordSim353.")
    else:
        raise FileNotFoundError(f"WordSim353 dataset not found at {ws_path}. Run download_datasets.py first.")
        
    # 2. Load SimVerb-3500 dataset
    simverb_path = os.path.join("data", "SimVerb-3500", "SimVerb-3500.txt")
    if os.path.exists(simverb_path):
        # SimVerb format: word1 \t word2 \t POS \t score \t REL
        df_simverb = pd.read_csv(simverb_path, sep='\t', header=None)
        # score is in column 3 (0-indexed)
        sv_data = df_simverb[[0, 1, 3]].values.tolist()
        sv_std = standardize_scores(sv_data, 0.0, 10.0)
        sv_expanded = augment_with_verbs(sv_std)
        sv_filtered = filter_by_frequency(sv_expanded, min_freq=10, max_freq=100000)
        record_dataset_stats("SimVerb-3500", sv_std, sv_expanded, sv_filtered)
        inspect_data("SimVerb-3500", sv_filtered)
        training_data.extend(sv_filtered)
        print(f"Loaded {len(sv_filtered)} pairs from SimVerb-3500.")
    else:
        raise FileNotFoundError(f"SimVerb-3500 dataset not found at {simverb_path}. Run download_datasets.py first.")
        
    # 3. Load THINGS dataset (SPOSE embeddings to pairwise similarity)
    things_embed_path = os.path.join("data", "THINGS", "spose_embedding_66d_sorted.txt")
    things_vocab_path = os.path.join("data", "THINGS", "unique_id.txt")
    if os.path.exists(things_embed_path) and os.path.exists(things_vocab_path):
        with open(things_vocab_path, 'r', encoding='utf-8') as f:
            raw_vocab = [line.strip() for line in f if line.strip()]
        
        raw_embeddings = np.loadtxt(things_embed_path)
        
        # Filter out polysemous words ending with a digit (e.g., "punch1")
        # Also replace underscores with hyphens for multi-word expressions (e.g., "egg_roll" -> "egg-roll")
        vocab = []
        embeddings_list = []
        for i, word in enumerate(raw_vocab):
            if not re.search(r'\d+$', word):
                vocab.append(word.replace('_', '-'))
                embeddings_list.append(raw_embeddings[i])
                
        embeddings = np.array(embeddings_list)
        
        # Calculate pairwise cosine similarity for a random sample of UNIQUE pairs.
        # Uses a fixed seed (42) independent of the per-replicate run_id so that every
        # replicate trains on identical THINGS data — inter-run variance should reflect
        # only optimisation stochasticity, not training-data stochasticity.
        num_concepts = len(vocab)
        num_samples = 10000
        things_rng = random.Random(42)  # fixed seed, independent of run_id
        all_pairs = [(i, j) for i in range(num_concepts) for j in range(i + 1, num_concepts)]
        sampled_pairs_idx = things_rng.sample(all_pairs, min(num_samples, len(all_pairs)))
        things_data = []
        for i, j in sampled_pairs_idx:
            v1, v2 = embeddings[i], embeddings[j]
            # Handle potential zero norms
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            cos_sim = np.dot(v1, v2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
            things_data.append((vocab[i], vocab[j], float(cos_sim)))
            
        things_std = standardize_scores(things_data, -1.0, 1.0)
        things_expanded = augment_with_plurals(things_std)
        things_filtered = filter_by_frequency(things_expanded, min_freq=10, max_freq=100000)
        record_dataset_stats("THINGS", things_std, things_expanded, things_filtered)
        inspect_data("THINGS", things_filtered)
        training_data.extend(things_filtered)
        print(f"Loaded {len(things_filtered)} sampled pairs from THINGS SPOSE embeddings.")
    else:
        raise FileNotFoundError(f"THINGS dataset files not found in data/THINGS/. Run download_datasets.py first.")
    
    return training_data

def load_simlex999():
    """
    Loads the real SimLex-999 dataset from the local data directory.
    
    SimLex-999 is used as the held-out evaluation dataset to test whether 
    the learned similarity geometry generalizes out of distribution.
    
    Returns:
        list of tuple: A list of (word1, word2, standardized_score) tuples.
    """
    print("Loading SimLex-999...")
    file_path = os.path.join("data", "SimLex-999", "SimLex-999.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SimLex-999 dataset not found at {file_path}. Run download_datasets.py first.")
        
    df = pd.read_csv(file_path, sep='\t')
    # SimLex-999 has columns: word1, word2, POS, SimLex999, ...
    # We extract word1, word2, and the similarity score (SimLex999) which is 0-10
    raw_data = df[['word1', 'word2', 'SimLex999']].values.tolist()
    
    # Standardize to 0-1
    held_out_data = standardize_scores(raw_data, 0.0, 10.0)
    
    # Global frequency filter
    held_out_filtered = filter_by_frequency(held_out_data, min_freq=10, max_freq=100000)
    
    record_dataset_stats("SimLex-999", held_out_data, held_out_data, held_out_filtered)
    inspect_data("SimLex-999", held_out_filtered)
    
    print(f"Successfully loaded {len(held_out_filtered)} pairs from SimLex-999.")
    return held_out_filtered

# NOTE: The actual Wikipedia corpus generation is implemented in
# generate_wiki_targetwords.py, which streams Wikipedia and extracts
# up to 60 sentences per target word with strict quality filtering.


@lru_cache(maxsize=8192)
def _compile_word_pattern(word):
    """Returns a compiled regex that matches `word` at word boundaries.
    
    Results are cached so that the ~6,700 unique target words are compiled
    only once across all calls to find_subword_span.
    """
    return re.compile(r'\b' + re.escape(word) + r'\b')


def find_subword_span(tokenizer, sentence, word):
    """
    Locates the token indices of a specific word inside a tokenized sentence.
    
    Since RoBERTa uses subword tokenization (BPE), a single word might be 
    split into multiple tokens. This function maps the character span of the 
    word back to the token sequence, allowing us to pool the correct tokens 
    during the model's forward pass.
    
    Crucially, word matching is performed strictly on lowercased input text using 
    word boundary boundaries (\b) to avoid:
    1. Substring contamination (e.g. matching "car" inside "cartoon").
    2. Capitalisation-induced entity drift (e.g. matching "apple" to "Apple" 
       at the beginning of a sentence, which captures the company/proper noun 
       sense rather than the common fruit noun sense).
    
    Args:
        tokenizer (RobertaTokenizer): The HuggingFace tokenizer.
        sentence (str): The full context sentence.
        word (str): The target word to find (case-insensitive).
        
    Returns:
        tuple or None: A tuple containing (start_token_idx, end_token_idx), 
                       where end_token_idx is exclusive. Returns None if 
                       the word is not found.
    """
    # Ask tokenizer to return character offsets for each token
    encoded = tokenizer(sentence, return_offsets_mapping=True)
    offsets = encoded['offset_mapping']
    
    # Find the character span of the exact word using word boundaries.
    # Search the ORIGINAL sentence (case-sensitive) so that we match a
    # naturally-lowercase occurrence (e.g. "apple") rather than a sentence-initial
    # or proper-noun capitalised form (e.g. "Apple").
    pattern = _compile_word_pattern(word.lower())
    match = pattern.search(sentence)          # case-sensitive: prefer lowercase
    if not match:
        return None
    start_char, end_char = match.span()
    
    start_token = None
    end_token = None
    
    # Map the character span to the token indices
    for idx, (start, end) in enumerate(offsets):
        # Skip special tokens like <s> and </s>
        if start == end == 0:
            continue
            
        # The first token that overlaps with the start of the word
        if start <= start_char and end > start_char:
            start_token = idx
            
        # The last token that overlaps with the end of the word
        if start < end_char and end >= end_char:
            end_token = idx
            break
            
    if start_token is not None and end_token is not None:
        return (start_token, end_token + 1) # Exclusive end index for slicing
    return None

if __name__ == "__main__":
    train_data = load_and_standardize_datasets()
    test_data = load_simlex999()
    
    wiki_path = os.path.join("data", "wiki_targetwords.jsonl")
    if os.path.exists(wiki_path):
        print(f"Target words corpus already exists at {wiki_path}.")
        # Quick validation
        try:
            with open(wiki_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line:
                    import json
                    sample = json.loads(first_line)
                    print(f"Validated sample word: '{sample.get('word')}' with {len(sample.get('sentences', []))} sentences.")
        except Exception as e:
            print(f"Warning: validation of existing corpus failed: {e}")
    else:
        print(f"Target words corpus not found. Please run src/generate_wiki_targetwords.py to generate it.")
