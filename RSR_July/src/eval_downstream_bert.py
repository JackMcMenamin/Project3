"""GLUE fine-tuning for the BERT RSR checkpoints.

Sanity check on the RSR models: we've deliberately reshaped the embedding
geometry, so does the thing still work as a language model? We want parity with
the controls here, not a win - a drop would mean catastrophic forgetting.

BERT version of eval_downstream.py. Note the small sets (RTE, CoLA, STS-B,
MRPC) are very unstable to fine-tune and will happily collapse to majority-class
predictions if you look at them funny. Warmup + more epochs + several seeds
keeps that under control; the frac_majority_pred metric flags it when it happens
anyway.

    python src/eval_downstream_bert.py \
        --models vanilla=bert-base-uncased \
                 mlm_control=../models/bert_baseline_w07_s1.pt \
                 rsr=../models/bert_rsr_w07_s1.pt \
        --tasks mrpc rte cola stsb sst2 --seeds 1 2 3
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

# task -> (hf config, text fields, n labels, metric, epochs)
# more epochs on the small sets, they need the extra passes to converge
TASKS = {
    "mrpc": ("mrpc", ("sentence1", "sentence2"), 2, "acc_f1", 5),
    "rte":  ("rte",  ("sentence1", "sentence2"), 2, "accuracy", 10),
    "cola": ("cola", ("sentence", None),         2, "matthews", 10),
    "stsb": ("stsb", ("sentence1", "sentence2"), 1, "pearson_spearman", 10),
    "sst2": ("sst2", ("sentence", None),         2, "accuracy", 3),
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_model(spec, num_labels):
    """HF id, or one of our BertForMaskedLM checkpoints."""
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels,
        problem_type="regression" if num_labels == 1 else None)
    if spec.endswith(".pt"):
        state = torch.load(spec, map_location="cpu")
        # checkpoints are plain BertForMaskedLM dicts (bert.* + cls.*) - take the
        # encoder only, the classifier head starts fresh
        enc = {k: v for k, v in state.items() if k.startswith("bert.")}
        missing, unexpected = model.load_state_dict(enc, strict=False)
        # pooler is absent from an MLM checkpoint by design, so it keeps the
        # bert-base weights. Same for every model, so it doesn't skew anything.
        # Anything else missing means the load went wrong - bail out.
        enc_missing = [k for k in missing
                       if k.startswith("bert.") and not k.startswith("bert.pooler.")]
        if enc_missing:
            raise RuntimeError(f"{len(enc_missing)} encoder weights did not load "
                               f"from {spec}: {enc_missing[:5]}")
    return model


def make_metrics(kind):
    def fn(eval_pred):
        preds, labels = eval_pred
        if kind == "pearson_spearman":
            p = preds.squeeze()
            return {"pearson": float(pearsonr(p, labels)[0]),
                    "spearman": float(spearmanr(p, labels)[0])}
        p = np.argmax(preds, axis=1)
        if kind == "matthews":
            return {"matthews": float(matthews_corrcoef(labels, p)),
                    "frac_majority_pred": float(max(np.bincount(p, minlength=2)) / len(p))}
        if kind == "acc_f1":
            return {"accuracy": float(accuracy_score(labels, p)),
                    "f1": float(f1_score(labels, p)),
                    "frac_majority_pred": float(max(np.bincount(p, minlength=2)) / len(p))}
        return {"accuracy": float(accuracy_score(labels, p)),
                "frac_majority_pred": float(max(np.bincount(p, minlength=2)) / len(p))}
    return fn


def headline(kind, m):
    return {"pearson_spearman": "pearson", "matthews": "matthews",
            "acc_f1": "accuracy"}.get(kind, "accuracy")


def run_one(model_spec, task, seed, tok, outdir):
    cfg, fields, n_lab, metric_kind, epochs = TASKS[task]
    set_seed(seed)
    ds = load_dataset("nyu-mll/glue", cfg)

    f1, f2 = fields
    def tok_fn(ex):
        return (tok(ex[f1], truncation=True, max_length=128) if f2 is None
                else tok(ex[f1], ex[f2], truncation=True, max_length=128))
    ds = ds.map(tok_fn, batched=True)

    model = load_model(model_spec, n_lab)
    args = TrainingArguments(
        output_dir=os.path.join(outdir, "ckpt", f"{task}_s{seed}"),
        eval_strategy="epoch", save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        num_train_epochs=epochs,
        warmup_ratio=0.1,            # without this the small sets collapse
        lr_scheduler_type="linear",
        weight_decay=0.01, seed=seed,
        logging_steps=200, report_to="none", disable_tqdm=True,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=ds["train"], eval_dataset=ds["validation"],
                      processing_class=tok,
                      data_collator=DataCollatorWithPadding(tokenizer=tok),
                      compute_metrics=make_metrics(metric_kind))
    trainer.train()
    res = trainer.evaluate()
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {k.replace("eval_", ""): v for k, v in res.items()
            if k.startswith("eval_") and isinstance(v, float)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="name=path_or_hf_id")
    ap.add_argument("--tasks", nargs="+", default=["mrpc", "rte", "cola", "stsb"],
                    choices=list(TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--output-dir", default="results_downstream_bert")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    models = dict(m.split("=", 1) for m in args.models)

    rows, path = [], os.path.join(args.output_dir, "downstream_results.json")
    for task in args.tasks:
        for name, spec in models.items():
            for seed in args.seeds:
                print(f"\n===== {task} | {name} | seed {seed} =====", flush=True)
                try:
                    m = run_one(spec, task, seed, tok, args.output_dir)
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                    continue
                key = headline(TASKS[task][3], m)
                # >95% one class = it gave up and guessed. Flag rather than
                # letting it drag the mean down unnoticed.
                degenerate = m.get("frac_majority_pred", 0) > 0.95
                rows.append({"task": task, "model": name, "seed": seed,
                             "score": m.get(key), "degenerate": degenerate, **m})
                print(f"  {key} = {m.get(key):.4f}"
                      f"{'   [DEGENERATE - predicts one class]' if degenerate else ''}",
                      flush=True)
                with open(path, "w") as f:
                    json.dump(rows, f, indent=2)

    # ---- summary -------------------------------------------------------
    print("\n" + "=" * 74, flush=True)
    print("GLUE SUMMARY  (mean +/- std over seeds)", flush=True)
    print("=" * 74, flush=True)
    hdr = f"{'task':<8}" + "".join(f"{n:>20}" for n in models)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for task in args.tasks:
        line = f"{task:<8}"
        for name in models:
            vals = [r["score"] for r in rows
                    if r["task"] == task and r["model"] == name and r["score"] is not None]
            bad = sum(1 for r in rows
                      if r["task"] == task and r["model"] == name and r["degenerate"])
            if vals:
                mean = np.mean(vals)
                sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                line += f"{mean:>13.4f} +/-{sd:.3f}" + ("*" if bad else " ")
            else:
                line += f"{'-':>20}"
        print(line, flush=True)
    print("\n* = at least one seed collapsed to a single class", flush=True)
    print(f"\nSaved to {path}", flush=True)


if __name__ == "__main__":
    main()
