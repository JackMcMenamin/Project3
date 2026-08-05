"""RSA against the Pereira (2018) fMRI data, for the BERT checkpoints.

BERT port of eval_pereira_rsa.py. Builds a similarity matrix over the 180
concepts from each model layer, correlates it with each subject's voxel
similarity matrix, and splits the pairs by whether the words were in our RSR
supervision set. The Neither bucket is the interesting one - those concepts
never got a human rating, so any gain there can't be memorisation.

Three things differ from the RoBERTa version:
  - BERT throughout, and our checkpoints are plain BertForMaskedLM state dicts
    so there's no roberta_mlm. prefix to strip
  - the Both/One/Neither masks come from OUR supervision vocab (SimVerb +
    THINGS), otherwise the buckets describe the wrong model
  - missing brain CSVs raise instead of silently falling back to a random
    matrix. Pass --allow-simulated if you actually want that for a smoke test.

    python src/eval_pereira_bert.py --models vanilla=bert-base-uncased \
        mlm_control=../models/bert_baseline_w07_s1.pt \
        rsr=../models/bert_rsr_w07_s1.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr, sem, ttest_rel
from transformers import BertForMaskedLM, BertTokenizerFast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import find_subword_span  # offset-based, tokenizer-agnostic

N_LAYERS = 13  # embeddings + 12 encoder layers


# ---------------------------------------------------------------------------
# Supervision vocabulary (for the Both/One/Neither partition)
# ---------------------------------------------------------------------------
def load_our_rsr_vocab(project_root: str) -> set:
    """The word set OUR BERT RSR model was supervised on.

    Mirrors load_supervision(use_men=False) in the BERT training script:
    SimVerb-3500 + THINGS, lowercased, underscores -> spaces.
    """
    sys.path.append(os.path.join(project_root, "experiments", "contextual_finetune"))
    vocab = set()

    simverb = os.path.join(project_root, "data", "simverb-3500-data", "data",
                           "SimVerb-3500.txt")
    if os.path.exists(simverb):
        with open(simverb, encoding="utf-8") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 4:
                    vocab.add(p[0].lower())
                    vocab.add(p[1].lower())

    things_words = os.path.join(project_root, "things_similarity", "variables",
                                "unique_id.txt")
    if os.path.exists(things_words):
        with open(things_words, encoding="utf-8") as f:
            for line in f:
                w = line.strip().replace("_", " ").lower()
                if w:
                    vocab.add(w)

    if not vocab:
        raise FileNotFoundError(
            "Could not build the RSR supervision vocabulary - check that "
            "data/simverb-3500-data and things_similarity/ exist in the project root."
        )
    return vocab


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------
def extract_layer_representations(model, tokenizer, words, device, wiki_sentences,
                                  max_sents=20, eval_half=True):
    """One vector per concept per layer, averaged over its Wikipedia contexts.

    Mirrors the `wiki_avg` protocol: take up to `max_sents` sentences from the
    HELD-OUT half of each word's sentence pool (index 30+), locate the target
    word's sub-word span, mean-pool it, then average across sentences.
    """
    model.eval()
    layer_reps = {i: [] for i in range(N_LAYERS)}
    missing = []

    with torch.no_grad():
        for word in words:
            available = wiki_sentences.get(word, [])
            pool = available[30:] if (eval_half and len(available) >= 30) else available

            sents = []
            for s in pool:
                if len(tokenizer.encode(s, add_special_tokens=True)) <= 512:
                    sents.append(s)
                if len(sents) == max_sents:
                    break
            if not sents:
                missing.append(word)
                sents = [word]  # fall back to the bare word

            per_layer = {i: [] for i in range(N_LAYERS)}
            for s in sents:
                span = find_subword_span(tokenizer, s, word)
                enc = tokenizer(s, return_tensors="pt", truncation=True,
                                max_length=512).to(device)
                if span is None:
                    span = (1, enc["input_ids"].size(1) - 1)  # whole sentence minus specials
                out = model(**enc, output_hidden_states=True)

                start, end = span
                for i, h in enumerate(out.hidden_states):
                    if start >= h.size(1) or end > h.size(1) or end <= start:
                        rep = h[0, 1:-1, :].mean(dim=0)
                    else:
                        rep = h[0, start:end, :].mean(dim=0)
                    per_layer[i].append(rep.cpu().numpy())

            for i in range(N_LAYERS):
                layer_reps[i].append(np.mean(per_layer[i], axis=0))

    if missing:
        print(f"    WARNING: no context sentences for {len(missing)} concepts "
              f"(used bare word): {missing[:10]}", flush=True)
    return {i: np.array(v) for i, v in layer_reps.items()}


def similarity_matrix(reps):
    """N x N Pearson correlation matrix over concept representation vectors."""
    r = np.corrcoef(reps)
    return np.nan_to_num(r, nan=0.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_rsa(M_D, M_B, mask=None):
    """Spearman correlation between the upper triangles of two similarity matrices."""
    n = M_D.shape[0]
    triu = np.triu_indices(n, k=1)
    if mask is not None:
        keep = np.zeros((n, n), dtype=bool)
        keep[triu] = True
        keep &= mask
        d, b = M_D[keep], M_B[keep]
    else:
        d, b = M_D[triu], M_B[triu]
    if len(d) < 2:
        return np.nan
    return spearmanr(d, b).correlation


def test_2vs2(M_D, M_B):
    """Proportion of concept pairs whose matched correlations beat the mismatched
    ones, with both items held out of the comparison (chance = 0.5)."""
    n = M_D.shape[0]
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            keep = np.ones(n, dtype=bool)
            keep[[i, j]] = False
            di, dj = M_D[i, keep], M_D[j, keep]
            bi, bj = M_B[i, keep], M_B[j, keep]
            r_ii = pearsonr(di, bi).statistic
            r_jj = pearsonr(dj, bj).statistic
            r_ij = pearsonr(di, bj).statistic
            r_ji = pearsonr(dj, bi).statistic
            if (r_ii + r_jj) > (r_ij + r_ji):
                correct += 1
            total += 1
    return correct / total if total else np.nan


def load_brain_matrix(path, words, allow_simulated=False):
    """Brain similarity matrix for one subject, concepts ordered like `words`."""
    if not os.path.exists(path):
        if not allow_simulated:
            raise FileNotFoundError(
                f"Brain data not found: {path}\n"
                "Provide the Pereira beta CSVs (columns: concept, voxel, beta), or "
                "pass --allow-simulated to run the pipeline on RANDOM data "
                "(for wiring checks only - the numbers are meaningless)."
            )
        print(f"    !! SIMULATED brain data for {os.path.basename(path)} - "
              f"numbers are meaningless", flush=True)
        rng = np.random.default_rng(0)
        return similarity_matrix(rng.standard_normal((len(words), 1000)))

    df = pd.read_csv(path)
    piv = (df.groupby(["concept", "voxel"])["beta"].mean().reset_index()
             .pivot(index="concept", columns="voxel", values="beta"))
    piv.index = piv.index.str.lower()
    piv = piv.reindex([w.lower() for w in words])
    n_missing = int(piv.isna().all(axis=1).sum())
    if n_missing:
        print(f"    WARNING: {n_missing} concepts absent from {os.path.basename(path)}",
              flush=True)
    return similarity_matrix(piv.fillna(0).to_numpy())


# ---------------------------------------------------------------------------
def load_model(spec, device):
    """`spec` is either a HF model id or a path to our saved BertForMaskedLM state dict."""
    if spec.endswith(".pt"):
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        state = torch.load(spec, map_location="cpu")
        # Our checkpoints are plain BertForMaskedLM state dicts; tolerate a
        # wrapper prefix just in case.
        state = {k[len("mlm."):] if k.startswith("mlm.") else k: v
                 for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"    note: {len(missing)} missing / {len(unexpected)} unexpected keys",
                  flush=True)
    else:
        model = BertForMaskedLM.from_pretrained(spec)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="name=path_or_hf_id, e.g. vanilla=bert-base-uncased "
                         "rsr=../models/bert_rsr_w07_s1.pt")
    ap.add_argument("--brain-dir", default="data/ryskina_repo/outputs/rsa",
                    help="directory of betas_sentences_<subject>.csv files")
    ap.add_argument("--words", default="data/pereira_2018/Pereira_Materials/stimuli_180concepts.txt")
    ap.add_argument("--wiki", default="data/wiki_pereira.jsonl")
    ap.add_argument("--output-dir", default="results_brain_bert")
    ap.add_argument("--layers", default="all",
                    help="'all' or a comma-separated list, e.g. 5,7,12")
    ap.add_argument("--allow-simulated", action="store_true",
                    help="run on RANDOM brain data if the CSVs are missing (wiring test only)")
    ap.add_argument("--skip-2v2", action="store_true", help="skip the (slow) 2vs2 test")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}", flush=True)

    words = [l.strip() for l in open(args.words, encoding="utf-8") if l.strip()]
    print(f"Concepts: {len(words)}", flush=True)

    wiki = {}
    if os.path.exists(args.wiki):
        with open(args.wiki, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                wiki[d["word"]] = d["sentences"]
    print(f"Concepts with context sentences: "
          f"{sum(1 for w in words if wiki.get(w.lower()))}/{len(words)}", flush=True)

    # Partition masks from OUR supervision vocabulary
    rsr_vocab = load_our_rsr_vocab(project_root)
    in_rsr = np.array([w.lower() in rsr_vocab for w in words], dtype=bool)
    counts = in_rsr[:, None].astype(int) + in_rsr[None, :].astype(int)
    masks = {"All pairs": None, "Both in RSR": counts == 2,
             "One in RSR": counts == 1, "Neither in RSR": counts == 0}
    print(f"RSR supervision vocab: {len(rsr_vocab)} words | "
          f"{int(in_rsr.sum())}/{len(words)} Pereira concepts supervised", flush=True)
    for k, m in masks.items():
        if m is not None:
            n = int(np.triu(m, k=1).sum())
            print(f"    {k}: {n} pairs", flush=True)

    # Brain data
    subjects = sorted(f.split("_")[-1].replace(".csv", "")
                      for f in os.listdir(args.brain_dir)
                      if f.startswith("betas_sentences_") and f.endswith(".csv")
                      ) if os.path.isdir(args.brain_dir) else []
    if not subjects:
        if not args.allow_simulated:
            raise FileNotFoundError(
                f"No betas_sentences_*.csv in {args.brain_dir}. See the README for how "
                "to obtain the Pereira betas, or pass --allow-simulated for a wiring test."
            )
        subjects = ["SIM01", "SIM02"]
    print(f"Subjects: {len(subjects)} {subjects}", flush=True)
    brains = {s: load_brain_matrix(os.path.join(args.brain_dir, f"betas_sentences_{s}.csv"),
                                   words, args.allow_simulated) for s in subjects}

    layers = list(range(N_LAYERS)) if args.layers == "all" else [
        int(x) for x in args.layers.split(",")]

    model_specs = dict(m.split("=", 1) for m in args.models)
    rows = []
    for name, spec in model_specs.items():
        print(f"\n=== {name} ({spec}) ===", flush=True)
        model = load_model(spec, device)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        reps = extract_layer_representations(model, tokenizer, words, device, wiki)
        for L in layers:
            M_D = similarity_matrix(reps[L])
            for subj in subjects:
                M_B = brains[subj]
                row = {"model": name, "layer": L, "subject": subj}
                for cat, mask in masks.items():
                    row[f"rsa_{cat}"] = compute_rsa(M_D, M_B, mask)
                if not args.skip_2v2:
                    row["acc_2v2"] = test_2vs2(M_D, M_B)
                rows.append(row)
            print(f"  layer {L:>2}: RSA(all)="
                  f"{np.mean([r['rsa_All pairs'] for r in rows if r['layer']==L and r['model']==name]):.4f}",
                  flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, "pereira_bert_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-subject results -> {out_csv}", flush=True)

    # Summary + paired t-tests (subjects as random effect)
    print("\n" + "=" * 78, flush=True)
    print("MEAN ACROSS SUBJECTS", flush=True)
    print("=" * 78, flush=True)
    summary = df.groupby(["model", "layer"]).mean(numeric_only=True).round(4)
    print(summary.to_string(), flush=True)

    names = list(model_specs)
    if "rsr" in names:
        for baseline in [n for n in names if n != "rsr"]:
            print(f"\nPaired t-tests: rsr vs {baseline} (n={len(subjects)} subjects)",
                  flush=True)
            for L in layers:
                a = df[(df.model == "rsr") & (df.layer == L)].set_index("subject")
                b = df[(df.model == baseline) & (df.layer == L)].set_index("subject")
                for cat in masks:
                    col = f"rsa_{cat}"
                    x, y = a[col].dropna(), b[col].dropna()
                    common = x.index.intersection(y.index)
                    if len(common) < 3:
                        # With n<3 the paired t-test is degenerate (and returns
                        # t=+/-inf on near-identical data), so report the raw
                        # difference only rather than a meaningless p-value.
                        if len(common) > 0:
                            d = float((x[common] - y[common]).mean())
                            print(f"  L{L:<3} {cat:<16} delta={d:+.4f}  "
                                  f"(n={len(common)} - too few subjects for a t-test)",
                                  flush=True)
                        continue
                    t = ttest_rel(x[common], y[common])
                    d = float((x[common] - y[common]).mean())
                    n_pos = int((x[common] - y[common] > 0).sum())
                    flag = " *" if t.pvalue < 0.05 else ""
                    print(f"  L{L:<3} {cat:<16} delta={d:+.4f}  t={t.statistic:+.2f}  "
                          f"p={t.pvalue:.4f}  ({n_pos}/{len(common)} subjects +){flag}",
                          flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
