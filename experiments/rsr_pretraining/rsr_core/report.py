"""Scoring per partition and the printed comparison table."""
from __future__ import annotations

import pandas as pd
import torch.nn.functional as F
from scipy.stats import spearmanr

from .data import BREAKDOWNS, PARTITIONS

LABELS = {"all": "All pairs", "both": "Both in RSR",
          "one": "One in RSR", "neither": "Neither in RSR",
          "both_trained": "Both, pair trained", "both_unseen": "Both, pair unseen",
          "both_N": "Both, nouns", "both_V": "Both, verbs",
          "one_N": "One, nouns", "one_V": "One, verbs",
          "neither_N": "Neither, nouns", "neither_V": "Neither, verbs",
          "neither_A": "Neither, adjectives"}


def score_partitions(emb: dict, cats: dict, min_n: int = 2) -> dict:
    """Spearman between cosine and human score for each partition. Pairs with
    a word that could not be embedded are skipped."""
    out = {}
    for name, pairs in cats.items():
        model_s, human_s = [], []
        for w1, w2, s in pairs:
            if w1 in emb and w2 in emb:
                model_s.append(F.cosine_similarity(emb[w1].unsqueeze(0),
                                                   emb[w2].unsqueeze(0)).item())
                human_s.append(s)
        if len(model_s) < min_n:
            out[name] = {"n": len(model_s), "rho": float("nan")}
        else:
            rho, _ = spearmanr(human_s, model_s)
            out[name] = {"n": len(model_s), "rho": float(rho)}
    return out


def print_eval(tag: str, res: dict) -> None:
    print(f"  [{tag}] SimLex rho  "
          f"all={res['all']['rho']:.4f} (n={res['all']['n']})  "
          f"both={res['both']['rho']:.4f}  "
          f"one={res['one']['rho']:.4f}  "
          f"neither={res['neither']['rho']:.4f}", flush=True)


def history_row(df: pd.DataFrame, which: str = "best", metric: str = "all") -> dict:
    """Rebuild a result dict from a training history: the first row (step 0),
    the last row, or the best row by `metric`. Breakdown columns are included
    when the history has them."""
    if which == "first":
        row = df.iloc[0]
    elif which == "last":
        row = df.iloc[-1]
    else:
        row = df.loc[df[f"{metric}_rho"].idxmax()]
    names = PARTITIONS + tuple(b for b in BREAKDOWNS if f"{b}_rho" in row.index)
    return {p: {"rho": row[f"{p}_rho"], "n": int(row[f"{p}_n"])} for p in names}


def print_comparison(vanilla: dict, rsr: dict, control: dict | None = None,
                     title: str = "SimLex-999 after continued pre-training") -> None:
    def rho(d, p):
        return d[p]["rho"] if p in d else float("nan")

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if control is None:
        header = f"{'Category':<16}{'Vanilla':>10}{'RSR':>10}{'Delta':>10}"
        print(header)
        print("-" * len(header))
        for p in PARTITIONS:
            v, r = rho(vanilla, p), rho(rsr, p)
            print(f"{LABELS[p]:<16}{v:>10.3f}{r:>10.3f}{r - v:>+10.3f}")
    else:
        header = (f"{'Category':<16}{'Vanilla':>9}{'Control':>9}{'RSR':>9}"
                  f"{'RSR-Van':>10}{'RSR-Ctl':>10}")
        print(header)
        print("-" * len(header))
        for p in PARTITIONS:
            v, c, r = rho(vanilla, p), rho(control, p), rho(rsr, p)
            print(f"{LABELS[p]:<16}{v:>9.3f}{c:>9.3f}{r:>9.3f}"
                  f"{r - v:>+10.3f}{r - c:>+10.3f}")
    print(f"\n  pairs evaluated: {vanilla['all']['n']}/999", flush=True)
    print_breakdown(vanilla, rsr, control)


def print_breakdown(vanilla: dict, rsr: dict, control: dict | None = None,
                    min_n: int = 10) -> None:
    """The finer splits, with pair counts. Splits under min_n pairs are left
    out of the table (they are still in the CSV); the rest are noisy in
    proportion to n."""
    rows = [b for b in BREAKDOWNS if b in rsr and rsr[b]["n"] >= min_n]
    if not rows:
        return
    ref = control if control is not None else vanilla
    name = "RSR-Ctl" if control is not None else "RSR-Van"
    print(f"\n  {'Breakdown':<22}{'n':>5}{'Vanilla':>9}{'RSR':>9}{name:>10}")
    for b in rows:
        v, r = vanilla.get(b, {}).get("rho", float("nan")), rsr[b]["rho"]
        c = ref.get(b, {}).get("rho", float("nan"))
        print(f"  {LABELS[b]:<22}{rsr[b]['n']:>5}{v:>9.3f}{r:>9.3f}{r - c:>+10.3f}")
