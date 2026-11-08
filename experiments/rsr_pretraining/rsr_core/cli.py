"""Command-line switches shared by the BERT and GPT-2 scripts and the audit.

Every switch defaults to the configuration of the reported runs, except early
stopping, which is now off unless asked for.
"""
from __future__ import annotations

import argparse

from .sampling import SAMPLERS


def add_supervision_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("supervision and sampling")
    g.add_argument("--use-men", action="store_true",
                   help="add MEN to the supervision (it is normally the held-out test)")
    g.add_argument("--exclude-simlex-pairs", dest="exclude_simlex", action="store_true",
                   help="drop every supervision pair that is also a SimLex pair")
    g.add_argument("--inflect-verbs", action="store_true",
                   help="add third person, past and gerund copies of each SimVerb pair")
    g.add_argument("--things-pairs", type=int, default=None,
                   help="keep a fixed random sample of this many THINGS pairs (default all)")
    g.add_argument("--sampler", choices=SAMPLERS, default=None,
                   help="words: uniform words (default, the reported runs); "
                        "pairs: draw scored pairs and take both words")
    g.add_argument("--simverb-share", type=float, default=None,
                   help="with --sampler pairs, the fraction of drawn pairs taken from SimVerb")


def add_protocol_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("protocol")
    g.add_argument("--early-stop", action="store_true",
                   help="keep the checkpoint with the best all-pairs SimLex. This selects "
                        "on the test set; off by default, the final model is reported")
    g.add_argument("--no-early-stop", action="store_true",
                   help="the default now; accepted so older command lines still run")
    g.add_argument("--split-sentences", action="store_true",
                   help="train on the first half of each word's harvested sentences and "
                        "evaluate on the second half, so no evaluation sentence is seen "
                        "in training")


def apply_args(cfg, args) -> None:
    """Copy the shared switches onto a Config."""
    cfg.use_men = args.use_men
    cfg.exclude_simlex = args.exclude_simlex
    cfg.inflect_verbs = args.inflect_verbs
    if args.things_pairs is not None:
        cfg.things_pairs = args.things_pairs
    if args.sampler is not None:
        cfg.sampler = args.sampler
    if args.simverb_share is not None:
        cfg.simverb_share = args.simverb_share
    if cfg.simverb_share is not None and cfg.sampler != "pairs":
        raise SystemExit("--simverb-share needs --sampler pairs")
    if hasattr(args, "early_stop"):
        if args.early_stop and args.no_early_stop:
            raise SystemExit("--early-stop and --no-early-stop together")
        cfg.early_stop = args.early_stop
        cfg.split_sentences = args.split_sentences


def describe(cfg) -> str:
    """One line naming every switch that differs from the reported runs."""
    on = []
    if cfg.exclude_simlex:
        on.append("SimLex pairs excluded")
    if cfg.inflect_verbs:
        on.append("verbs inflected")
    if cfg.things_pairs is not None:
        on.append(f"THINGS sample {cfg.things_pairs:,}")
    if cfg.sampler != "words":
        on.append(f"sampler {cfg.sampler}"
                  + (f" (SimVerb share {cfg.simverb_share})" if cfg.simverb_share is not None else ""))
    if cfg.use_men:
        on.append("MEN in supervision")
    if getattr(cfg, "split_sentences", False):
        on.append("sentences split train/eval")
    on.append("early stopping" if cfg.early_stop else "final model reported")
    return "; ".join(on)
