"""
Main Training Loop Module for Representational Similarity Regularisation (RSR)

This module handles the execution of the RSR fine-tuning process.
It interleaves standard Masked Language Modeling (MLM) updates with RSR updates at a 
defined ratio (e.g., 3:1) as specified in the original experimental design.

Workflow Context:
- The central orchestrator for a single training session.
- Invoked by: `run_experiments.py` (which runs this script across different modes/seeds, potentially concurrently).
- Outputs: Live console output and a plain-text `results/eval_log_*.txt` file tracking SimLex-999 evaluation over time.
- Note: Utilizes `os._exit(0)` for hard teardown to prevent hanging threads from HuggingFace dataset streaming.

Functions:
- main: Orchestrates the execution of different training modes.
"""

import os
import sys
import json
import time
import torch
import random
import argparse
from transformers import RobertaTokenizerFast as RobertaTokenizer  # PATCH: fast tokenizer required for return_offsets_mapping on this transformers version
from datasets import load_dataset

from model import RobertaRSRModel
from loss import rsr_loss_for_batch
from samplers import RSRBatchSampler, prepare_mlm_batch
from evaluate import evaluate_simlex
from data_prep import load_and_standardize_datasets, load_simlex999

import re as _re

# Pre-compiled regex for splitting text into sentences by terminal punctuation.
# Used by _extract_first_quality_sentence on every article in the Wikipedia stream.
_SENT_SPLIT_RE = _re.compile(r'(?<=[.!?]) +')

def _extract_first_quality_sentence(text):
    """Extracts the first quality prose sentence from a Wikipedia article.
    
    Applies the same filtering criteria as generate_wiki_targetwords.py:
    split by newlines (to avoid merging lists/headers), split into sentences
    by punctuation, require sentence-ending punctuation, enforce a minimum
    word count, and skip bibliographic artifacts.
    
    Args:
        text (str): Raw article text from the Wikipedia dump.
        
    Returns:
        str or None: The first qualifying sentence, or None if no sentence passes.
    """
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Split into sentences by punctuation followed by a space
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
            return s
    return None

def get_mlm_sentences(mode, wiki_iter, target_sentences_list, batch_size=16):
    """
    Retrieves a batch of quality sentences for Masked Language Modeling (MLM).
    
    Depending on the training mode:
    1. In targetwords modes (e.g. `vanilla_targetwords`, `rsr_targetwords`), it 
       randomly samples from `target_sentences_list`, which contains the training 
       half of the pre-extracted Wikipedia target-word sentences.
    2. In general modes (e.g. `vanilla_all`, `rsr_all`), it streams from general 
       Wikipedia, extracting the first sentence of consecutive articles that 
       passes the quality criteria (5-100 words, punctuation, etc.).
       
    Args:
        mode (str): The training mode (e.g. "rsr_targetwords").
        wiki_iter (iterator): Iterator over the general Wikipedia stream.
        target_sentences_list (list of str): Pool of training target-word sentences.
        batch_size (int): Number of sentences to retrieve.
        
    Returns:
        list of str: A list of qualifying context sentences.
    """
    sentences = []
    if "targetwords" in mode and target_sentences_list:
        sentences = random.sample(target_sentences_list, min(batch_size, len(target_sentences_list)))
    else:
        # Pull from general wikipedia stream with quality filtering
        while len(sentences) < batch_size:
            try:
                article = next(wiki_iter)
                sent = _extract_first_quality_sentence(article['text'])
                if sent:
                    sentences.append(sent)
            except StopIteration:
                break
    return sentences

def main():
    parser = argparse.ArgumentParser(description="Run RSR Training Loop")
    parser.add_argument("--mode", type=str, default="rsr_targetwords", 
                        choices=["vanilla_all", "vanilla_targetwords", "rsr_all", "rsr_targetwords"],
                        help="The training mode configuration to run.")
    parser.add_argument("--steps", type=int, default=8000, help="Total number of training steps.")
    parser.add_argument("--run_id", type=int, default=0, help="Run ID for statistical replicates (seeds randomness).")
    parser.add_argument("--inspect_batches", action="store_true", help="Log batch contents to results/batch_inspection_log.txt for debugging.")
    parser.add_argument("--eval_mode", type=str, default="bare", choices=["bare", "wiki_avg"], help="Evaluation context mode.")
    parser.add_argument("--rsr_layer", type=int, default=None, help="The specific RoBERTa layer to apply RSR to (e.g., 5). If None, defaults to the last layer.")
    parser.add_argument("--eval_steps", type=int, default=400, help="Evaluation and logging frequency in steps.")
    parser.add_argument("--combine", type=str, default="interleave", choices=["interleave", "weighted"],
                        help="interleave (default): 3 MLM steps then 1 RSR step. "
                             "weighted: both losses every step, (1-lambda)*MLM + lambda*RSR.")
    parser.add_argument("--rsr_lambda", type=float, default=0.25,
                        help="RSR weight in weighted mode. 0.25 gives RSR the same overall "
                             "share as the 1-in-4 interleave, if you want them comparable.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory where evaluation logs are saved.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory where timing logs are saved.")
    args = parser.parse_args()
    
    # Save exact command line arguments to log folder
    os.makedirs(args.log_dir, exist_ok=True)
    params_file = os.path.join(args.log_dir, f"run_params_{args.mode}_run{args.run_id}.log")
    with open(params_file, "w", encoding="utf-8") as f:
        f.write(f"Command line: {' '.join(sys.argv)}\n")
        f.write(f"Raw sys.argv: {sys.argv}\n")
        f.write(f"Parsed arguments: {vars(args)}\n")
    
    print("\n" + "="*60)
    print(" REPRESENTATIONAL SIMILARITY REGULARISATION (RSR)")
    print(f" Mode: {args.mode}")
    print(f" Eval Mode: {args.eval_mode}")
    print(f" Steps: {args.steps}")
    print(f" Run ID: {args.run_id}")
    print(f" RSR Layer: {args.rsr_layer if args.rsr_layer is not None else 'Last'}")
    print(f" Inspect Batches: {args.inspect_batches}")
    print("="*60)
    
    # Set seeds for reproducibility of this specific run
    random.seed(42 + args.run_id)
    torch.manual_seed(42 + args.run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + args.run_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Model and Tokenizer
    print("Loading Roberta-base...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaRSRModel("roberta-base", rsr_layer=args.rsr_layer).to(device)
    
    # All layers of RoBERTa are trainable.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # 2. Load Datasets
    training_data = load_and_standardize_datasets()
    simlex_data = load_simlex999()
    
    # Setup Wiki sources only if we need the general stream
    wiki_iter = None
    if "targetwords" not in args.mode:
        wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        wiki_iter = iter(wiki_dataset)
    
    target_sentences_list = []
    target_sentences_dict = {}
    eval_sentences_dict = {}
    if "targetwords" in args.mode or args.eval_mode == "wiki_avg":
        target_path = os.path.join("data", "wiki_targetwords.jsonl")
        if os.path.exists(target_path):
            with open(target_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    # Very fast heuristic: 100 words is typically <150 tokens, safely below 512.
                    filtered_sentences = [s for s in data['sentences'] if len(s.split()) <= 100]
                    # Split into equal halves: first half for training, second half for evaluation.
                    # This prevents data leakage between targetwords training and wiki_avg evaluation.
                    midpoint = len(filtered_sentences) // 2
                    train_sents = filtered_sentences[:midpoint]
                    eval_sents = filtered_sentences[midpoint:]
                    target_sentences_list.extend(train_sents)
                    target_sentences_dict[data['word']] = train_sents
                    eval_sentences_dict[data['word']] = eval_sents
            print(f"Loaded {len(target_sentences_list)} training sentences and {sum(len(v) for v in eval_sentences_dict.values())} evaluation sentences.")
        else:
            raise FileNotFoundError("data/wiki_targetwords.jsonl not found. Run generate_wiki_targetwords.py first.")
            
    # Extract the unique training words for zero-shot categorisation during evaluation
    training_words = set()
    for w1, w2, _ in training_data:
        training_words.add(w1)
        training_words.add(w2)
        
    # Setup RSR Sampler
    rsr_sampler = RSRBatchSampler(training_data, tokenizer, batch_size_pairs=24, use_templates=("all" in args.mode), wiki_sentences_dict=target_sentences_dict)
    rsr_iterator = iter(rsr_sampler)
    
    # 3. Training Loop
    start_time = time.time()
    model.train()
    
    # Create evaluation log file
    os.makedirs(args.results_dir, exist_ok=True)
    log_file = os.path.join(args.results_dir, f"eval_log_{args.mode}_run{args.run_id}.txt")
    with open(log_file, "w") as f:
        f.write(f"RSR Training Run: Mode={args.mode}, Steps={args.steps}, RSR Layer={args.rsr_layer if args.rsr_layer is not None else 'Last'}\n")
        f.write("="*60 + "\n")
        
    def run_evaluation(current_step, avg_mlm_loss=None):
        print(f"\n--- Evaluation at Step {current_step} ---")
        results = evaluate_simlex(model, tokenizer, simlex_data, training_words, eval_mode=args.eval_mode, wiki_sentences_dict=eval_sentences_dict, device=device)
        
        print("SimLex-999 Evaluation:")
        log_line = f"Step {current_step:<4} | "
        if avg_mlm_loss is not None:
            print(f"  > Avg MLM Loss   : {avg_mlm_loss:.4f}")
            log_line += f"MLM Loss: {avg_mlm_loss:.4f} | "
        else:
            log_line += "MLM Loss: N/A | "
            
        for cat, score in results.items():
            print(f"  > {cat:<25}: {score:.3f}")
            log_line += f"{cat}: {score:.3f} | "
        
        with open(log_file, "a") as f:
            f.write(log_line + "\n")
            
        print("Evaluation completed and logged.\n")
        model.train()

    # Initial zero-shot baseline evaluation at step 0
    run_evaluation(0, avg_mlm_loss=None)
    
    inspect_log = None
    if args.inspect_batches:
        inspect_log = open(os.path.join(args.results_dir, f"batch_inspection_{args.mode}_run{args.run_id}.txt"), "w", encoding='utf-8')
        inspect_log.write(f"Batch Inspection Log for Mode={args.mode}, Run={args.run_id}\n")
        inspect_log.write("="*80 + "\n\n")

    mlm_losses_since_eval = []

    # ---- helpers so the weighted mode can request either loss on demand -----
    def _rsr_loss():
        """One RSR batch -> its loss. Shares the sampler with the interleaved path."""
        nonlocal rsr_iterator
        try:
            b = next(rsr_iterator)
        except StopIteration:
            rsr_iterator = iter(rsr_sampler)
            b = next(rsr_iterator)
        return rsr_loss_for_batch(
            model.target_vectors(b['input_ids'].to(device),
                                 b['attention_mask'].to(device),
                                 b['target_spans']),
            b['human_scores_matrix'].to(device),
            b['valid_pairs_mask'].to(device), tau=0.1)

    def _mlm_loss():
        """One MLM batch -> its loss."""
        sents = get_mlm_sentences(args.mode, wiki_iter, target_sentences_list, batch_size=8)
        b = prepare_mlm_batch(tokenizer, sents)
        return model(b['input_ids'].to(device),
                     attention_mask=b['attention_mask'].to(device),
                     labels=b['labels'].to(device)).loss

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()

        # ---- WEIGHTED mode -------------------------------------------------
        # Both losses in every update: (1-lambda)*MLM + lambda*RSR. Separate
        # batches for each, so this puts no restriction on the MLM data.
        # The alternating default below gives some steps a pure RSR gradient.
        if args.combine == "weighted" and "rsr" in args.mode:
            rsr_l = _rsr_loss()
            mlm_l = _mlm_loss()
            loss = (1.0 - args.rsr_lambda) * mlm_l + args.rsr_lambda * rsr_l
            loss_type = f"MIX(L={args.rsr_lambda})"
            mlm_losses_since_eval.append(mlm_l.item())
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"Step {step}/{args.steps} | Type: {loss_type} | "
                      f"mlm={mlm_l.item():.4f} rsr={rsr_l.item():.4f}")
            if step % args.eval_steps == 0 or step == args.steps:
                avg = (sum(mlm_losses_since_eval) / len(mlm_losses_since_eval)
                       if mlm_losses_since_eval else 0.0)
                mlm_losses_since_eval = []
                run_evaluation(step, avg)
            continue

        # ---- ALTERNATING mode (Barry's original, unchanged) ----------------
        # 3:1 Ratio -> Steps 1,2,3 are MLM, Step 0 (modulo 4) is RSR
        is_rsr_step = ("rsr" in args.mode) and (step % 4 == 0)

        if is_rsr_step:
            # --- RSR Forward Pass ---
            try:
                batch = next(rsr_iterator)
            except StopIteration:
                rsr_iterator = iter(rsr_sampler)
                batch = next(rsr_iterator)
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_spans = batch['target_spans']
            human_scores = batch['human_scores_matrix'].to(device)
            valid_mask = batch['valid_pairs_mask'].to(device)
            
            vectors = model.target_vectors(input_ids, attention_mask, target_spans)
            loss = rsr_loss_for_batch(vectors, human_scores, valid_mask, tau=0.1)
            loss_type = "RSR"
            
            if inspect_log:
                inspect_log.write(f"--- Step {step} | TYPE: RSR ---\n")
                inspect_log.write(f"Target Words ({len(batch['words_in_batch'])}): {batch['words_in_batch']}\n")
                inspect_log.write("Sentences:\n")
                for s in batch['sentences']:
                    inspect_log.write(f" - {s}\n")
                inspect_log.write("Human Scores Matrix:\n")
                inspect_log.write(f"{human_scores}\n\n")
                inspect_log.flush()
        else:
            # --- MLM Forward Pass ---
            sentences = get_mlm_sentences(args.mode, wiki_iter, target_sentences_list, batch_size=8)
            batch = prepare_mlm_batch(tokenizer, sentences)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss_type = "MLM"
            mlm_losses_since_eval.append(loss.item())
            
            if inspect_log:
                inspect_log.write(f"--- Step {step} | TYPE: MLM ---\n")
                inspect_log.write(f"Masked Sentences ({len(batch['sentences'])}):\n")
                for s in batch['sentences']:
                    inspect_log.write(f" - {s}\n")
                inspect_log.write("\n")
                inspect_log.flush()
            
        # Backprop
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step}/{args.steps} | Type: {loss_type} | Loss: {loss.item():.4f}")
            
        # Evaluation every eval_steps
        if step % args.eval_steps == 0 or step == args.steps:
            avg_mlm_loss = sum(mlm_losses_since_eval) / len(mlm_losses_since_eval) if mlm_losses_since_eval else 0.0
            mlm_losses_since_eval = []
            run_evaluation(step, avg_mlm_loss)
            
    if inspect_log:
        inspect_log.close()
        
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")
    
    # Save timing information to log folder
    os.makedirs(args.log_dir, exist_ok=True)
    timing_file = os.path.join(args.log_dir, f"time_{args.mode}_run{args.run_id}.txt")
    with open(timing_file, "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Run ID: {args.run_id}\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
        
    # Flush stdout/stderr and exit immediately to prevent PyGILState_Release/PyArrow crash on interpreter shutdown
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
