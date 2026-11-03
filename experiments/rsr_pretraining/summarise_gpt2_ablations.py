"""Summarise the two remaining GPT-2 follow-ups, whichever have finished.

1. Templates vs real sentences, scored in three contexts. Same table as the
   BERT version in notes/findings.md: Neither at the final step, RSR minus the
   matched control, for each training context x evaluation context, mean and
   range across seeds, with a paired t-test on template-vs-sentence training
   within each evaluation context. The BERT numbers are printed alongside.

2. Layer 12 against layer 6 (the main runs). Early-stopped and endpoint deltas
   per partition, paired by seed, to test the BERT prediction that a worse
   starting layer gains MORE.

Both read only completed gpt2_continued_* files, never partials, so a half-run
seed cannot skew a mean.

Usage:
    python experiments/rsr_pretraining/summarise_gpt2_ablations.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
PARTS = ["all", "both", "one", "neither"]


def runs(regime, tag):
    """{seed: DataFrame} for every finished arm matching a tag prefix."""
    out = {}
    for f in glob.glob(os.path.join(RES, f"gpt2_continued_{regime}_{tag}_s*_*.csv")):
        m = re.search(r"_s(\d+)_\d{8}_\d{6}\.csv$", f)
        if m:
            out[int(m.group(1))] = pd.read_csv(f)
    return out


def fmt(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "   n/a   "
    return f"{v.mean():+.3f} [{v.min():+.3f},{v.max():+.3f}]" if len(v) > 1 else f"{v[0]:+.3f}"


# ===========================================================================
# 1. templates vs sentences
# ===========================================================================
print("=" * 78)
print("GPT-2: templates vs real sentences, Neither at the final step, RSR - control")
print("=" * 78)
ctl = runs("baseline", "ctx_sentences")
sent = runs("rsr", "ctx_sentences")
tmpl = runs("rsr", "ctx_template")
seeds = sorted(set(ctl) & set(sent) & set(tmpl))
if not seeds:
    print("  (no complete seed has all three arms yet: control %s, sentences %s, templates %s)"
          % (sorted(ctl), sorted(sent), sorted(tmpl)))
else:
    print("  seeds with all three arms:", seeds)
    BERT = {"sentences": ("+0.045", "+0.019", "0.037"),
            "bare": ("+0.129", "+0.116", "0.29"),
            "template": ("+0.184", "+0.223", "0.15")}
    print(f"  {'eval context':<12}{'sentence-trained':>26}{'template-trained':>26}{'p':>8}   BERT: sent / tmpl / p")
    for ctx, pre in [("sentences", ""), ("bare", "bare_"), ("template", "template_")]:
        col = pre + "neither_rho"
        ds, dt = [], []
        for s in seeds:
            last = lambda d: d.loc[d.step == d.step.max(), col].iloc[0]
            ds.append(last(sent[s]) - last(ctl[s]))
            dt.append(last(tmpl[s]) - last(ctl[s]))
        p = ttest_rel(ds, dt).pvalue if len(seeds) > 1 else float("nan")
        print(f"  {ctx:<12}{fmt(ds):>26}{fmt(dt):>26}{p:>8.3f}   {' / '.join(BERT[ctx])}")
    # what continued training alone does to each probe (the BERT finding that
    # out-of-context structure is destroyed by the control)
    print()
    print("  control alone, all pairs, step 0 -> end (does continued CLM wreck out-of-context probes?)")
    for ctx, pre in [("sentences", ""), ("bare", "bare_"), ("template", "template_")]:
        col = pre + "all_rho"
        v0 = np.mean([ctl[s].loc[ctl[s].step == 0, col].iloc[0] for s in seeds])
        v1 = np.mean([ctl[s].loc[ctl[s].step == ctl[s].step.max(), col].iloc[0] for s in seeds])
        print(f"    {ctx:<12}{v0:.3f} -> {v1:.3f}  ({v1 - v0:+.3f})")
    print("  mean RSR training loss:  sentences %.4f   templates %.4f"
          % (np.mean([sent[s].rsr_loss.mean() for s in seeds]),
             np.mean([tmpl[s].rsr_loss.mean() for s in seeds])))

# ===========================================================================
# 2. layer 12 vs layer 6
# ===========================================================================
print()
print("=" * 78)
print("GPT-2: read-out layer 12 vs layer 6 (main runs), RSR - control")
print("=" * 78)
arms = {"L6": (runs("rsr", "main"), runs("baseline", "main")),
        "L12": (runs("rsr", "L12"), runs("baseline", "L12"))}
have = {k: sorted(set(r) & set(b)) for k, (r, b) in arms.items()}
if not have["L12"]:
    print("  (no complete L12 seed yet)")
else:
    print("  seeds:  L6 %s   L12 %s" % (have["L6"], have["L12"]))
    for label, pick in [("early-stopped (best all-pairs)", "es"), ("endpoint (final step)", "end")]:
        print(f"\n  {label}")
        print(f"  {'partition':<10}{'L6':>26}{'L12':>26}{'L12 - L6 p':>12}")
        rows = {}
        for k, (r, b) in arms.items():
            for p in PARTS:
                vals = []
                for s in have[k]:
                    rr, bb = r[s], b[s]
                    if pick == "es":
                        i, j = rr.all_rho.idxmax(), bb.all_rho.idxmax()
                        vals.append(rr.loc[i, p + "_rho"] - bb.loc[j, p + "_rho"])
                    else:
                        vals.append(rr[p + "_rho"].iloc[-1] - bb[p + "_rho"].iloc[-1])
                rows[(k, p)] = vals
        common = sorted(set(have["L6"]) & set(have["L12"]))
        for p in PARTS:
            a, c = rows[("L6", p)], rows[("L12", p)]
            pv = (ttest_rel([rows[("L12", p)][have["L12"].index(s)] for s in common],
                            [rows[("L6", p)][have["L6"].index(s)] for s in common]).pvalue
                  if len(common) > 1 else float("nan"))
            print(f"  {p:<10}{fmt(a):>26}{fmt(c):>26}{pv:>12.3f}")
    # starting points, for the "worse start, bigger gain" framing
    v6 = np.mean([arms["L6"][1][s].loc[arms["L6"][1][s].step == 0, "neither_rho"].iloc[0] for s in have["L6"]])
    v12 = np.mean([arms["L12"][1][s].loc[arms["L12"][1][s].step == 0, "neither_rho"].iloc[0] for s in have["L12"]])
    print(f"\n  vanilla Neither at step 0:  L6 {v6:.3f}   L12 {v12:.3f}")
    print("  BERT for comparison: layer 7 read better than 12 (probe) but gained +0.027 vs +0.053.")
