"""
Downstream Evaluation Script

This script evaluates trained models (both baseline RoBERTa and RSR checkpoints)
on standard downstream fine-tuning tasks including GLUE (STS-B, RTE, CoLA), 
SuperGLUE (WiC), and NER (CoNLL-2003).

It instantiates a Hugging Face Trainer and performs sequence or token classification
fine-tuning to establish if the intrinsic semantic improvements of RSR transfer
to real-world performance metrics.

Workflow Context:
- Evaluates whether continued pretraining using RSR causes catastrophic forgetting of 
  general linguistic capabilities on downstream tasks compared to vanilla pretraining.
- Invoked manually or via orchestration scripts (e.g. run_all.ps1) after models have 
  completed continued pretraining.
- Saves results as JSON files under the results/downstream directory for reporting.
"""
import argparse
import os
import json
import torch
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    RobertaTokenizerFast, 
    RobertaForSequenceClassification, 
    RobertaForTokenClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification
)

def load_model(model_class, model_path, num_labels):
    """
    Load a pre-trained RoBERTa model or an RSR checkpoint and initialize it
    for sequence or token classification.

    When loading an RSR checkpoint, the `roberta_mlm.` prefix is stripped from
    the state dictionary keys to properly map weights to the classification head.

    Args:
        model_class (class): e.g., RobertaForSequenceClassification.
        model_path (str): 'roberta-base' or path to a .pt checkpoint.
        num_labels (int): Number of classification classes/labels.

    Returns:
        PreTrainedModel: An instantiated model ready for fine-tuning.
    """
    model = model_class.from_pretrained('roberta-base', num_labels=num_labels)
    if model_path != "roberta-base":
        state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('roberta_mlm.'):
                k = k[len('roberta_mlm.'):]
            if k.startswith('roberta.'):
                new_state_dict[k] = v
        
        # Load strict=False because we initialize a new classification head
        model.load_state_dict(new_state_dict, strict=False)
    return model

def compute_metrics_clf(metric_name):
    """
    Factory function that returns a compute_metrics function tailored for
    the specified classification metric.

    Args:
        metric_name (str): Name of the Hugging Face `evaluate` metric 
                           (e.g., 'pearsonr', 'accuracy', 'matthews_correlation').

    Returns:
        Callable: A function taking `eval_pred` and returning a metric dictionary.
    """
    metric = evaluate.load(metric_name)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if metric_name in ["pearsonr", "spearmanr"]:
            predictions = predictions.squeeze()
            return metric.compute(predictions=predictions, references=labels)
        elif metric_name == "matthews_correlation":
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)
        elif metric_name == "glue": # For tasks like rte or wic
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)
        else: # accuracy
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)
    return compute_metrics

label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
def compute_metrics_ner(eval_pred):
    """
    Compute seqeval F1 and Accuracy for Named Entity Recognition tasks.

    Args:
        eval_pred (tuple): A tuple containing predictions and labels.

    Returns:
        dict: A dictionary containing overall `f1` and `accuracy` scores.
    """
    metric = evaluate.load("seqeval")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    """
    Main execution pipeline for downstream task evaluation.

    Configures the dataset, tokenizer, metrics, and collator based on the specified
    task ("stsb", "wic", "rte", "cola", "ner"). Instantiates the model and executes
    the fine-tuning via Hugging Face `Trainer`. Results are saved as a JSON file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model or 'roberta-base'")
    parser.add_argument("--task", type=str, required=True, choices=["stsb", "wic", "rte", "cola", "ner"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
    
    if args.task == "stsb":
        dataset = load_dataset("nyu-mll/glue", "stsb", trust_remote_code=True)
        num_labels = 1
        model_class = RobertaForSequenceClassification
        def tokenize_function(examples):
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
        metric_name = "pearsonr"
        compute_metrics = compute_metrics_clf(metric_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
    elif args.task == "wic":
        dataset = load_dataset("aps/super_glue", "wic", trust_remote_code=True)
        num_labels = 2
        model_class = RobertaForSequenceClassification
        def tokenize_function(examples):
            # WiC gives word, sentence1, sentence2.
            # We can format it as: word: <word> s1: <sentence1> s2: <sentence2>
            # Or simply sentence1 and sentence2
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
        metric_name = "accuracy"
        compute_metrics = compute_metrics_clf(metric_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    elif args.task == "rte":
        dataset = load_dataset("nyu-mll/glue", "rte", trust_remote_code=True)
        num_labels = 2
        model_class = RobertaForSequenceClassification
        def tokenize_function(examples):
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
        metric_name = "accuracy"
        compute_metrics = compute_metrics_clf(metric_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    elif args.task == "cola":
        dataset = load_dataset("nyu-mll/glue", "cola", trust_remote_code=True)
        num_labels = 2
        model_class = RobertaForSequenceClassification
        def tokenize_function(examples):
            return tokenizer(examples["sentence"], truncation=True)
        metric_name = "matthews_correlation"
        compute_metrics = compute_metrics_clf(metric_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    elif args.task == "ner":
        dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
        num_labels = len(label_list)
        model_class = RobertaForTokenClassification
        def tokenize_function(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
            labels = []
            for i, label in enumerate(examples[f"ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        compute_metrics = compute_metrics_ner
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Tokenize
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Load Model
    model = load_model(model_class, args.model_path, num_labels)
    
    # Training args
    model_name_safe = args.model_path.replace("/", "_").replace(".pt", "")
    run_name = f"{args.task}_{model_name_safe}"
    
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints", run_name),
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"--- Starting Fine-tuning for {run_name} on {args.task} ---")
    trainer.train()
    
    print(f"--- Evaluating {run_name} ---")
    eval_results = trainer.evaluate()
    
    # Save results
    results_file = os.path.join(args.output_dir, f"{run_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=4)
        
    print(f"Results saved to {results_file}")
    print(eval_results)

if __name__ == "__main__":
    main()
