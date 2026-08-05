"""
Batch Sampling Module for Representational Similarity Regularisation (RSR)

This module handles the complex logic of constructing batches for RSR training. 
Traditional random batching is inefficient because it often results in batches 
with very few overlapping word pairs (edges). To compute similarity loss, we 
need an interconnected graph of words.

The `RSRBatchSampler` uses an "edge-first" strategy, picking known similarity 
pairs and expanding from there, ensuring the batch forms a dense, connected 
topology of semantic relationships.

Workflow Context:
- Consumed by: `train.py`. The `RSRBatchSampler` yields dictionary batches for the RSR update steps. 
- Consumed by: `train.py`. The `prepare_mlm_batch` dynamically masks regular Wikipedia sentences for standard RoBERTa pretraining updates.

Classes:
- RSRBatchSampler: A custom iterator for building edge-dense RSR batches.

Functions:
- prepare_mlm_batch: A utility for creating standard Masked Language Modeling batches.
"""

import random
import torch
from data_prep import find_subword_span

class RSRBatchSampler:
    """
    An iterator that yields batches designed specifically for RSR training.
    
    Rather than randomly sampling sentences, it randomly samples known human 
    similarity edges (word pairs), looks up sentences containing those words, 
    and packages them together. This guarantees a minimum density of valid 
    supervision signals per batch.
    
    Attributes:
        similarity_pairs (list of tuple): All known (word1, word2, score) tuples.
        tokenizer (RobertaTokenizer): The HuggingFace tokenizer.
        batch_size_pairs (int): Number of edges (pairs) to sample per batch.
        use_templates (bool): Whether to use synthetic templates instead of Wiki sentences.
        wiki_sentences (list of str): Pool of sentences to draw context from if not using templates.
    """
    def __init__(self, similarity_pairs, tokenizer, batch_size_pairs=24, use_templates=True, wiki_sentences_dict=None):
        self.similarity_pairs = similarity_pairs
        self.tokenizer = tokenizer
        self.batch_size_pairs = batch_size_pairs
        self.use_templates = use_templates
        self.wiki_sentences_dict = wiki_sentences_dict or {}
        
        # --- CONCEPTUAL FIX: Graph Densification Lookup ---
        # Rationale: When we randomly sample M pairs of words for a batch, we gather the 
        # unique words from those pairs. However, it is highly likely that there are OTHER 
        # valid human-annotated similarity scores between these unique words that we didn't 
        # explicitly sample in this specific draw.
        # By creating an O(1) lookup dictionary of ALL known similarity edges in the dataset,
        # we can later scan every possible pair combination within the current batch and harvest
        # any "incidental" or "free" supervision signals. This drastically increases the density
        # of the graph topology being evaluated by the Soft-Spearman loss without requiring any
        # additional forward passes through the model.
        self.sim_lookup = {}
        for w1, w2, score in self.similarity_pairs:
            self.sim_lookup[(w1, w2)] = score
            self.sim_lookup[(w2, w1)] = score
        
    def __iter__(self):
        return self
        
    def __next__(self):
        """
        Constructs and yields the next RSR batch.
        
        Process:
        1. Randomly selects M edges (pairs) from the similarity dataset to seed the batch vocabulary.
        2. Retrieves or generates a context sentence for each word.
        3. Tokenizes the sentences and maps the character spans of the target words to token indices.
        4. (Graph Densification) Scans all N(N-1)/2 possible pairings within the batch against an O(1) global dataset lookup to maximize the density of the supervision matrix.
        
        Returns:
            dict: A batch dictionary containing:
                  - input_ids, attention_mask: Tokenized tensors.
                  - target_spans: List of (start_idx, end_idx) tuples for pooling.
                  - human_scores_matrix: NxN tensor of known similarities.
                  - valid_pairs_mask: NxN boolean mask indicating where human scores exist.
        """
        # 1. Edge-first sampling: pick M random pairs from the training set
        k = min(self.batch_size_pairs, len(self.similarity_pairs))
        sampled_pairs = random.sample(self.similarity_pairs, k)
        
        # Determine the unique words present in these sampled edges
        words_in_batch = list(set(w1 for w1, w2, _ in sampled_pairs).union(set(w2 for w1, w2, _ in sampled_pairs)))
        N = len(words_in_batch)

        
        sentences = []
        target_spans = []
        
        # 2. Gather sentences and spans
        for word in words_in_batch:
            if self.use_templates:
                # Use a null-context template to probe isolated word geometry
                sentence = f"The word is {word}."
            else:
                valid_sentences = self.wiki_sentences_dict.get(word, [])
                sentence = f"The word is {word}."
                if valid_sentences:
                    # Copy and shuffle so we can pick the first one that doesn't exceed 512 tokens
                    shuffled = list(valid_sentences)
                    random.shuffle(shuffled)
                    for s in shuffled:
                        if len(self.tokenizer.encode(s, add_special_tokens=True)) <= 512:
                            sentence = s
                            break 
                
            # 3. Locate the subword token indices of the target word
            span = find_subword_span(self.tokenizer, sentence, word)
            if span is None:
                raise ValueError(f"find_subword_span failed to locate '{word}' in sentence: '{sentence}'")
                
            sentences.append(sentence)
            target_spans.append(span)
            
        # Tokenize the full list of sentences for this batch
        encoded = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        
        # 4. Construct the pairwise target matrix
        human_scores_matrix = torch.zeros(N, N)
        valid_pairs_mask = torch.zeros(N, N, dtype=torch.bool)
        
        # --- CONCEPTUAL FIX: Full Batch Graph Iteration ---
        # Instead of ONLY populating the matrix with the M edges we specifically sampled,
        # we iterate over ALL N(N-1)/2 possible pairings of the unique words present in the batch.
        # If any pair of words has a known human score in our global dataset lookup, we add it!
        # This converts a sparse set of M edges into a much denser graph topology, maximizing
        # the amount of supervision we can extract from the forward pass we already computed.
        for i in range(N):
            for j in range(i + 1, N):
                w1 = words_in_batch[i]
                w2 = words_in_batch[j]
                
                # Check if this pair exists in our O(1) global dataset lookup
                if (w1, w2) in self.sim_lookup:
                    score = self.sim_lookup[(w1, w2)]
                    human_scores_matrix[i, j] = score
                    human_scores_matrix[j, i] = score
                    valid_pairs_mask[i, j] = True
                    valid_pairs_mask[j, i] = True
            
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'target_spans': target_spans,
            'human_scores_matrix': human_scores_matrix,
            'valid_pairs_mask': valid_pairs_mask,
            'sentences': sentences,
            'words_in_batch': words_in_batch
        }

def prepare_mlm_batch(tokenizer, sentences):
    """
    Prepares a batch of sentences for standard Masked Language Modeling (MLM).
    
    Replaces 15% of the tokens with the [MASK] token. Labels are set to -100 for 
    unmasked tokens so the cross-entropy loss only computes gradients for the masked ones.
    Utilizes PyTorch vectorized boolean masking for zero-overhead special token filtering.
    
    Args:
        tokenizer (RobertaTokenizer): The HuggingFace tokenizer.
        sentences (list of str): The raw text sentences.
        
    Returns:
        dict: A dictionary containing input_ids, attention_mask, and labels.
    """
    # Tokenize the sentences
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded['input_ids'].clone()
    labels = input_ids.clone()
    
    # Create a random probability matrix for masking
    prob_matrix = torch.full(labels.shape, 0.15)
    
    # Don't mask special tokens (e.g. <s>, </s>, <pad>)
    # --- EFFICIENCY FIX: PyTorch Vectorized Boolean Masking ---
    # Rationale: Converting the tensor to a python list with .tolist() and using a
    # standard Python list comprehension over the entire batch tensor is incredibly slow
    # and breaks the PyTorch C++ execution graph. By replacing it with native PyTorch 
    # boolean tensor comparisons, we achieve zero-overhead masking.
    special_tokens_mask = (labels == tokenizer.cls_token_id) | \
                          (labels == tokenizer.sep_token_id) | \
                          (labels == tokenizer.pad_token_id)
    prob_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # Apply the mask
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # -100 tells PyTorch CrossEntropyLoss to ignore these targets
    input_ids[masked_indices] = tokenizer.mask_token_id
    
    return {
        'input_ids': input_ids,
        'attention_mask': encoded['attention_mask'],
        'labels': labels,
        'sentences': sentences
    }
