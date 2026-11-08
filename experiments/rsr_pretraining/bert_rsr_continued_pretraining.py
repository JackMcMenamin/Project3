"""BERT continued pre-training with Representational Similarity Regularisation.

Every training step combines two losses on bert-base-uncased:

    loss = (1 - lambda) * MLM + lambda * (1 - soft_spearman)

The MLM term is standard masked language modelling on harvested Wikipedia
sentences and keeps the model a working language model. The RSR term takes 24
supervised words, each inside one of its real sentences, reads each word's
contextual vector (mean over its word pieces), and ranks the cosines of every
scored pair against the human ratings with a differentiable Spearman
correlation. Nothing is frozen and there is no projection head, so the
constraint has to reshape the encoder itself.

The control (--regime baseline) is the same loop with the RSR term removed.
--regime compare runs RSR and control from the same seed.

Supervision is SimVerb-3500 plus THINGS. SimLex-999 is the evaluation set,
split by how many of each pair's words are in the supervision (see
rsr_core.data.exposure), and further by whether a Both pair is itself a
supervision pair and by part of speech. MEN is held out entirely and scored
by eval_men_transfer.py.

The switches in rsr_core.cli change what is trained on and how it is
sampled. All of them default to the reported runs except early stopping,
which is off unless --early-stop is given. audit_supervision.py takes the
same switches and reports what a configuration trains on without a model.

    python experiments/rsr_pretraining/bert_rsr_continued_pretraining.py \\
        --regime compare --lambda 0.7 --seed 1

    # no SimLex pair in training, pair-first sampling, a third of pairs from SimVerb
    python experiments/rsr_pretraining/bert_rsr_continued_pretraining.py \\
        --exclude-simlex-pairs --sampler pairs --simverb-share 0.33 --seed 1

    # vanilla SimLex at every layer, no training
    python experiments/rsr_pretraining/bert_rsr_continued_pretraining.py --probe-layers

    # a few steps on a tiny batch, to check everything runs
    python experiments/rsr_pretraining/bert_rsr_continued_pretraining.py --smoke
"""
from __future__ import annotations

import os

# Use the local Hugging Face cache only. Otherwise every from_pretrained call
# checks the hub, and a dropped connection kills a long run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsr_core import paths
from rsr_core.cli import add_protocol_args, add_supervision_args, apply_args, describe
from rsr_core.data import (RSR_TEMPLATE, find_subword_span, load_sentences_for,
                           load_simlex, load_simlex_pos, partition, set_seed,
                           set_sentence_split)
from rsr_core.sampling import WordPicker, rsr_targets
from rsr_core.supervision import (SupervisionOptions, anchor_words, batch_composition,
                                  build_supervision)
from rsr_core.loss import HAS_TORCHSORT, pair_loss
from rsr_core.report import history_row, print_comparison, print_eval, score_partitions
from rsr_core.paths import MODELS_DIR, RESULTS_DIR
# The evaluation scripts import this module as B and read these from it.
from rsr_core.paths import ARTIFACTS, DATA_DIR, REPO_ROOT  # noqa: F401

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    model_name: str = "bert-base-uncased"
    rsr_lambda: float = 0.7
    rsr_batch_size: int = 24        # target words per RSR batch
    mlm_batch_size: int = 16        # sentences per MLM batch
    mlm_probability: float = 0.15
    learning_rate: float = 5e-5
    soft_rank_strength: float = 1.0
    min_pairs_per_rsr_batch: int = 8
    max_seq_len: int = 128
    steps: int = 2000
    eval_every: int = 100
    seed: int = 1
    # Hidden layer the word vectors are read from, for training and for
    # evaluation. None is the last layer (12). 0 is the embedding output.
    rsr_layer: int | None = None
    # What RSR words are embedded in during training: a real harvested sentence
    # per word, redrawn each batch, or the fixed frame RSR_TEMPLATE.
    rsr_context: str = "sentences"
    # Contexts SimLex is scored in. The first drives early stopping and keeps
    # the unprefixed CSV columns; the rest are logged with a prefix.
    # "sentences" averages up to 20 harvested sentences, "bare" is the word on
    # its own, "template" is RSR_TEMPLATE.
    eval_contexts: tuple = ("sentences",)
    # Keep the checkpoint with the best all-pairs SimLex. That selects on the
    # evaluation set, so it is off unless --early-stop is given.
    early_stop: bool = False
    early_stop_metric: str = "all"
    # Supervision and sampling (rsr_core.supervision, rsr_core.sampling).
    # The defaults are the reported runs.
    use_men: bool = False
    exclude_simlex: bool = False
    inflect_verbs: bool = False
    things_pairs: int | None = None
    sampler: str = "words"
    simverb_share: float | None = None
    split_sentences: bool = False


class BertRSR(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
        self.mlm = BertForMaskedLM.from_pretrained(cfg.model_name)
        self.to(DEVICE)

    def mlm_loss(self, input_ids, attention_mask, labels) -> torch.Tensor:
        return self.mlm(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    def target_vectors(self, input_ids, attention_mask, spans) -> torch.Tensor:
        """One vector per row: the mean of the target word's piece vectors at
        the configured layer. spans gives (start, end) token positions."""
        if self.cfg.rsr_layer is None:
            hidden = self.mlm.bert(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state
        else:
            out = self.mlm.bert(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
            hidden = out.hidden_states[self.cfg.rsr_layer]
        vecs = []
        for j, (start, end) in enumerate(spans):
            # keep the span inside the unpadded length in case of truncation
            seq_len = int(attention_mask[j].sum().item())
            end = min(end, seq_len)
            start = min(start, max(seq_len - 1, 0))
            vecs.append(hidden[j, start:end].mean(dim=0))
        return torch.stack(vecs, dim=0)


class RSRBatchSampler:
    """Takes candidate words from a WordPicker, puts each in one of its
    sentences, and keeps the batch only if enough pairs are scored."""

    def __init__(self, picker: WordPicker, sim_lookup: dict, tokenizer, cfg: Config):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.sim_lookup = sim_lookup
        self.picker = picker
        self.targets = picker.targets
        self._sent_cache: dict[str, list[str]] = {}

    def _sentences(self, word: str) -> list[str]:
        if word not in self._sent_cache:
            self._sent_cache[word] = load_sentences_for(word, part="train")
        return self._sent_cache[word]

    def sample_batch(self):
        N = self.cfg.rsr_batch_size
        if len(self.targets) < N:
            return None
        words, sents, spans = [], [], []
        for word in self.picker.candidates():
            if len(words) == N:
                break
            if self.cfg.rsr_context == "template":
                candidates = [RSR_TEMPLATE.format(word=word)]
            else:
                candidates = self._sentences(word)
                if not candidates:
                    continue
                random.shuffle(candidates)
            target_ids = self.tokenizer(word, add_special_tokens=False)["input_ids"]
            for sent in candidates[:8]:
                enc = self.tokenizer(sent, add_special_tokens=True, truncation=True,
                                     max_length=self.cfg.max_seq_len)["input_ids"]
                span = find_subword_span(enc, target_ids)
                if span is not None:
                    words.append(word)
                    sents.append(sent)
                    spans.append(span)
                    break
        if len(words) < N:
            return None
        scored = sum(1 for a, b in combinations(words, 2) if (a, b) in self.sim_lookup)
        if scored < self.cfg.min_pairs_per_rsr_batch:
            return None
        batch = self.tokenizer(sents, add_special_tokens=True, truncation=True,
                               max_length=self.cfg.max_seq_len, padding=True,
                               return_tensors="pt")
        return batch, spans, words


def rsr_loss_for_batch(model: BertRSR, batch, spans, words, sim_lookup, cfg) -> torch.Tensor | None:
    vecs = model.target_vectors(batch["input_ids"].to(DEVICE),
                                batch["attention_mask"].to(DEVICE), spans)
    return pair_loss(vecs, words, sim_lookup,
                     cfg.min_pairs_per_rsr_batch, cfg.soft_rank_strength)


class MLMBatcher:
    """Standard BERT masking: 15% of real tokens are targets; of those 80%
    become [MASK], 10% a random token, 10% stay as they are."""

    def __init__(self, tokenizer, cfg: Config, sentences: list[str]):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.sentences = sentences

    def sample_batch(self):
        sents = random.sample(self.sentences, k=min(len(self.sentences), self.cfg.mlm_batch_size))
        enc = self.tokenizer(sents, add_special_tokens=True, truncation=True,
                             max_length=self.cfg.max_seq_len, padding=True,
                             return_tensors="pt")
        input_ids = enc["input_ids"]
        labels = input_ids.clone()

        probability = torch.full(labels.shape, self.cfg.mlm_probability)
        special = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True)
             for row in input_ids.tolist()], dtype=torch.bool)
        probability.masked_fill_(special, 0.0)
        probability.masked_fill_(enc["attention_mask"] == 0, 0.0)
        masked = torch.bernoulli(probability).bool()
        labels[~masked] = -100

        to_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
        input_ids[to_mask] = self.tokenizer.mask_token_id
        to_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~to_mask
        input_ids[to_random] = torch.randint(len(self.tokenizer), labels.shape,
                                             dtype=torch.long)[to_random]
        return input_ids.to(DEVICE), enc["attention_mask"].to(DEVICE), labels.to(DEVICE)


def embed_words_in_context(model: BertRSR, words: list[str], cfg: Config,
                           sent_cache: dict[str, list[str]], max_sents: int = 20,
                           context: str = "sentences") -> dict[str, torch.Tensor]:
    """Mean vector per word over its contexts (see Config.eval_contexts).
    Words that cannot be found in any context are left out."""
    model.eval()
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for w in words:
            if context == "bare":
                sents = [w]
            elif context == "template":
                sents = [RSR_TEMPLATE.format(word=w)]
            else:
                if w not in sent_cache:
                    sent_cache[w] = load_sentences_for(w, part="eval")
                sents = sent_cache[w]
            if not sents:
                continue
            target_ids = model.tokenizer(w, add_special_tokens=False)["input_ids"]
            vecs = []
            for sent in sents[:max_sents]:
                enc = model.tokenizer(sent, add_special_tokens=True, truncation=True,
                                      max_length=cfg.max_seq_len, return_tensors="pt")
                span = find_subword_span(enc["input_ids"][0].tolist(), target_ids)
                if span is None:
                    continue
                vecs.append(model.target_vectors(enc["input_ids"].to(DEVICE),
                                                 enc["attention_mask"].to(DEVICE), [span])[0])
            if vecs:
                out[w] = torch.stack(vecs).mean(dim=0)
    return out


def evaluate_simlex(model: BertRSR, simlex, supervised_vocab, cfg, sent_cache,
                    context: str = "sentences", lookup=None, pos=None) -> dict:
    """Spearman against SimLex for all pairs, each exposure partition, and
    the breakdowns `lookup` and `pos` allow (see rsr_core.data.partition).
    Every word is embedded once and shared across partitions."""
    cats = partition(simlex, supervised_vocab, lookup=lookup, pos=pos)
    words = list({w for p in simlex for w in p[:2]})
    emb = embed_words_in_context(model, words, cfg, sent_cache, context=context)
    return score_partitions(emb, cats)


class Shared:
    """Data loaded once per process and shared by both arms of a comparison."""

    def __init__(self, tokenizer, cfg: Config):
        print(f"Loading supervision ({describe(cfg)}) ...", flush=True)
        self.sup = build_supervision(SupervisionOptions.from_config(cfg))
        self.sim_lookup, self.supervised_vocab = self.sup.lookup, self.sup.vocab
        self.simlex = load_simlex()
        self.simlex_pos = load_simlex_pos()
        print(f"  {self.sup.summary()}", flush=True)
        targets = rsr_targets(self.sup, cfg.rsr_context)
        picker = WordPicker(self.sup, targets, cfg.rsr_batch_size, cfg.sampler,
                            cfg.simverb_share)
        self.rsr_sampler = RSRBatchSampler(picker, self.sim_lookup, tokenizer, cfg)
        print(f"  usable RSR targets: {len(targets)}", flush=True)
        # The MLM corpus is the harvested sentences of 400 random supervised
        # words. They are drawn from the default supervision whatever the
        # switches say, so configurations differ in the RSR term only. The
        # caller seeds `random` first so this is reproducible.
        words = anchor_words(self.sup)
        self.mlm_sentences = []
        for w in random.sample(words, k=min(400, len(words))):
            self.mlm_sentences.extend(load_sentences_for(w, part="train"))
        print(f"  MLM sentences: {len(self.mlm_sentences)}", flush=True)


def _eval_row(step: int, by_ctx: dict, primary: str, **extra) -> dict:
    row = {"step": step, **extra}
    for c, r in by_ctx.items():
        pre = "" if c == primary else c + "_"
        row.update({f"{pre}{k}_{m}": r[k][m] for k in r for m in r[k]})
    return row


def train_one(cfg: Config, regime: str, model: BertRSR, shared: Shared,
              verbose: bool = True, partial_path=None) -> pd.DataFrame:
    """Train `model` as the RSR arm (regime "rsr") or the control ("baseline").

    The history is rewritten to `partial_path` after every evaluation, so a run
    that dies partway still leaves its curves."""
    shared.rsr_sampler.cfg = cfg
    mlm_batcher = MLMBatcher(model.tokenizer, cfg, shared.mlm_sentences)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    eval_cache: dict[str, list[str]] = {}
    contexts = tuple(cfg.eval_contexts) or ("sentences",)
    primary = contexts[0]

    def evaluate():
        return {c: evaluate_simlex(model, shared.simlex, shared.supervised_vocab,
                                   cfg, eval_cache, context=c, lookup=shared.sim_lookup,
                                   pos=shared.simlex_pos) for c in contexts}

    def mlm_term():
        input_ids, attn, labels = mlm_batcher.sample_batch()
        if (labels != -100).sum() == 0:
            return None
        return model.mlm_loss(input_ids, attn, labels)

    def rsr_term():
        sample = shared.rsr_sampler.sample_batch()
        if sample is None:
            return None, None
        batch, spans, words = sample
        return rsr_loss_for_batch(model, batch, spans, words, shared.sim_lookup, cfg), words

    # Step 0. Context coverage differs (bare and template embed every word,
    # sentences drops words with no harvest), so each context's curve has to be
    # read against its own step-0 row.
    by_ctx = evaluate()
    if verbose:
        print("\nBefore training:", flush=True)
        for c in contexts:
            print_eval("step 0" if c == primary else f"step 0 {c}", by_ctx[c])
    history = [_eval_row(0, by_ctx, primary)]

    res = by_ctx[primary]
    best_metric = res[cfg.early_stop_metric]["rho"]
    best_state = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                  if cfg.early_stop else None)
    best_step = 0

    t0 = time.time()
    run_mlm, run_rsr, n_mlm, n_rsr = 0.0, 0.0, 0, 0
    # scored pairs per RSR batch by source, logged with each evaluation
    composition: Counter = Counter()
    for step in range(1, cfg.steps + 1):
        model.train()
        if regime == "rsr":
            m, (r, words) = mlm_term(), rsr_term()
            if m is None or r is None:
                continue
            loss = (1.0 - cfg.rsr_lambda) * m + cfg.rsr_lambda * r
            run_mlm += m.item(); n_mlm += 1
            run_rsr += r.item(); n_rsr += 1
            composition.update(batch_composition(words, shared.sup.source))
        else:
            loss = mlm_term()
            if loss is None:
                continue
            run_mlm += loss.item(); n_mlm += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0)
        optimizer.step()

        if step % cfg.eval_every == 0:
            mlm_avg, rsr_avg = run_mlm / max(n_mlm, 1), run_rsr / max(n_rsr, 1)
            by_ctx = evaluate()
            res = by_ctx[primary]
            mix = {f"rsr_pairs_{s}": n / max(n_rsr, 1) for s, n in sorted(composition.items())}
            if verbose:
                print(f"step {step}/{cfg.steps}  mlm_loss={mlm_avg:.4f}  "
                      f"rsr_loss={rsr_avg:.4f}  ({time.time() - t0:.0f}s)", flush=True)
                if mix:
                    print("  scored pairs per RSR batch: " + ", ".join(
                        f"{k[10:]} {v:.1f}" for k, v in mix.items()), flush=True)
                for c in contexts:
                    print_eval(f"step {step}" if c == primary else f"step {step} {c}", by_ctx[c])
            history.append(_eval_row(step, by_ctx, primary, mlm_loss=mlm_avg, rsr_loss=rsr_avg,
                                     **mix))
            if partial_path is not None:
                pd.DataFrame(history).to_csv(partial_path, index=False)
            run_mlm, run_rsr, n_mlm, n_rsr = 0.0, 0.0, 0, 0
            composition.clear()

            if cfg.early_stop and res[cfg.early_stop_metric]["rho"] > best_metric:
                best_metric = res[cfg.early_stop_metric]["rho"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_step = step

    if cfg.early_stop and best_state is not None and best_step != cfg.steps:
        model.load_state_dict(best_state)
        if verbose:
            print(f"\nEarly stopping: restored step {best_step} "
                  f"({cfg.early_stop_metric} rho={best_metric:.4f})", flush=True)

    df = pd.DataFrame(history)
    df.attrs["best_step"] = best_step
    return df


def train(cfg: Config, regime: str, shared: Shared | None = None,
          save_tag: str | None = None) -> pd.DataFrame:
    """Fresh model, trained under `regime`. With `save_tag`, the weights (after
    any early-stopping rollback) go to models/bert_{regime}_{save_tag}.pt with
    the config alongside as JSON, for the brain and GLUE evaluations."""
    set_seed(cfg.seed)
    print(f"Device: {DEVICE} | torchsort: {HAS_TORCHSORT} | regime: {regime}", flush=True)
    model = BertRSR(cfg)
    if shared is None:
        shared = Shared(model.tokenizer, cfg)
    print("\nTraining ...", flush=True)
    slug = save_tag or f"s{cfg.seed}"
    df = train_one(cfg, regime, model, shared,
                   partial_path=RESULTS_DIR / f"partial_{regime}_{slug}.csv")
    if save_tag:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out = MODELS_DIR / f"bert_{regime}_{save_tag}.pt"
        # the bare BertForMaskedLM state dict, so it loads into a stock HF model
        torch.save(model.mlm.state_dict(), out)
        meta = {"regime": regime, **asdict(cfg), "eval_contexts": list(cfg.eval_contexts),
                "best_step": int(df.attrs.get("best_step", -1))}
        with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved checkpoint -> {out}", flush=True)
    return df


def probe_layers(cfg: Config, max_sents: int = 20) -> pd.DataFrame:
    """Vanilla SimLex at every hidden layer, single layers and running means of
    layers 0..n (the two read-outs in Vulic et al., 2020). Each sentence is
    forwarded once with all hidden states, so the full curve costs about one
    evaluation."""
    set_seed(cfg.seed)
    print(f"Layer probe: vanilla {cfg.model_name}, {max_sents} sentences per word", flush=True)
    model = BertRSR(cfg)
    model.eval()
    simlex = load_simlex()
    sup = build_supervision(SupervisionOptions.from_config(cfg))
    cats = partition(simlex, sup.vocab, lookup=sup.lookup, pos=load_simlex_pos())
    n_layers = model.mlm.config.num_hidden_layers

    words = sorted({w for p in simlex for w in p[:2]})
    emb: list[dict[str, torch.Tensor]] = [dict() for _ in range(n_layers + 1)]
    with torch.no_grad():
        for i, w in enumerate(words):
            target_ids = model.tokenizer(w, add_special_tokens=False)["input_ids"]
            total, count = None, 0
            for sent in load_sentences_for(w, part="eval")[:max_sents]:
                enc = model.tokenizer(sent, add_special_tokens=True, truncation=True,
                                      max_length=cfg.max_seq_len, return_tensors="pt")
                span = find_subword_span(enc["input_ids"][0].tolist(), target_ids)
                if span is None:
                    continue
                out = model.mlm.bert(input_ids=enc["input_ids"].to(DEVICE),
                                     attention_mask=enc["attention_mask"].to(DEVICE),
                                     output_hidden_states=True)
                hs = torch.stack(out.hidden_states, dim=0)[:, 0]
                vec = hs[:, span[0]:span[1]].mean(dim=1)
                total = vec if total is None else total + vec
                count += 1
            if count:
                mean = total / count
                for L in range(n_layers + 1):
                    emb[L][w] = mean[L]
            if (i + 1) % 200 == 0:
                print(f"  embedded {i + 1}/{len(words)} words", flush=True)

    def flat(res):
        return {f"{k}_{m}": res[k][m] for k in res for m in ("rho", "n")}

    rows = [{"mode": "Ln", "layer": L, **flat(score_partitions(emb[L], cats))}
            for L in range(n_layers + 1)]
    running: dict[str, torch.Tensor] = {}
    for L in range(n_layers + 1):
        for w, v in emb[L].items():
            running[w] = v if w not in running else running[w] + v
        mean = {w: v / (L + 1) for w, v in running.items()}
        rows.append({"mode": "AVG(L<=n)", "layer": L, **flat(score_partitions(mean, cats))})
    df = pd.DataFrame(rows)

    for mode in ("Ln", "AVG(L<=n)"):
        sub = df[df["mode"] == mode]
        print(f"\n  {mode}\n  {'layer':>5}{'all':>10}{'both':>10}{'one':>10}{'neither':>10}")
        for _, r in sub.iterrows():
            print(f"  {int(r['layer']):>5}{r['all_rho']:>10.4f}{r['both_rho']:>10.4f}"
                  f"{r['one_rho']:>10.4f}{r['neither_rho']:>10.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"bert_layer_probe_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}", flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=["compare", "rsr", "baseline"], default="compare",
                    help="compare runs RSR and the control from the same seed")
    ap.add_argument("--lambda", dest="rsr_lambda", type=float, default=None,
                    help="weight on the RSR term (default 0.7)")
    ap.add_argument("--combine", choices=["weighted"], default=None,
                    help="accepted for old command lines; weighted is the only mode")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rsr-batch-size", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--rsr-layer", type=int, default=None,
                    help="hidden layer to read word vectors from (0 = embeddings, "
                         "default the last layer)")
    ap.add_argument("--rsr-context", choices=["sentences", "template"], default=None,
                    help="embed RSR words in real sentences (default) or in the "
                         "fixed frame 'The word is X.'")
    ap.add_argument("--eval-contexts", default=None,
                    help="comma-separated, from sentences, bare, template; the first "
                         "drives early stopping")
    ap.add_argument("--sentences-dir", default=None,
                    help="sentence pool relative to the repo root "
                         "(default artifacts/sentences_strict)")
    ap.add_argument("--save-tag", default=None,
                    help="save weights to models/bert_{regime}_{tag}.pt and tag the CSVs")
    ap.add_argument("--probe-layers", action="store_true",
                    help="no training: vanilla SimLex at every layer")
    ap.add_argument("--smoke", action="store_true",
                    help="20 steps on small batches, to check the pipeline runs")
    add_supervision_args(ap)
    add_protocol_args(ap)
    args = ap.parse_args()

    if args.sentences_dir is not None:
        paths.set_sentence_pool(args.sentences_dir)
    pool = paths.sentence_pool()
    print(f"Sentence pool: {pool} ({sum(1 for _ in pool.glob('*.jsonl'))} files)", flush=True)

    cfg = Config()
    for name in ("rsr_lambda", "steps", "seed", "rsr_batch_size", "eval_every",
                 "rsr_layer", "rsr_context"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.eval_contexts:
        cfg.eval_contexts = tuple(c.strip() for c in args.eval_contexts.split(",") if c.strip())
    apply_args(cfg, args)
    set_sentence_split(cfg.split_sentences)
    print(f"Configuration: {describe(cfg)}", flush=True)
    if args.smoke:
        cfg.steps, cfg.eval_every, cfg.mlm_batch_size = 20, 10, 4

    if args.probe_layers:
        probe_layers(cfg)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.save_tag}" if args.save_tag else ""
    read = "best" if cfg.early_stop else "last"

    if args.regime == "compare":
        set_seed(cfg.seed)
        shared = Shared(BertTokenizerFast.from_pretrained(cfg.model_name), cfg)
        frames = {}
        for regime in ("rsr", "baseline"):
            print(f"\n########## {regime.upper()} (lambda={cfg.rsr_lambda}) ##########", flush=True)
            frames[regime] = train(cfg, regime, shared, save_tag=args.save_tag)
            out = RESULTS_DIR / f"bert_continued_{regime}{suffix}_{stamp}.csv"
            frames[regime].to_csv(out, index=False)
            print(f"Saved -> {out}", flush=True)
        print_comparison(history_row(frames["rsr"], "first"),
                         history_row(frames["rsr"], read),
                         history_row(frames["baseline"], read))
    else:
        df = train(cfg, args.regime, save_tag=args.save_tag)
        out = RESULTS_DIR / f"bert_continued_{args.regime}{suffix}_{stamp}.csv"
        df.to_csv(out, index=False)
        print(f"Saved -> {out}", flush=True)
        if len(df) >= 2:
            print_comparison(history_row(df, "first"), history_row(df, read))


if __name__ == "__main__":
    main()
