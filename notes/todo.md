# To do

Open items as of 8 November 2026. Numbers checked against the current loaders.
Every switch named below is in both training scripts and in
`audit_supervision.py`, which reports what a configuration trains on without
loading a model.

## The loss is effectively Pearson, not Spearman

- torchsort is not installed, and every run log says `torchsort: False`. All
  reported runs used the sigmoid fallback soft rank at strength 1.0.
- Cosines and scaled ratings differ by small amounts, so a sigmoid at strength 1
  is close to linear and the soft ranks are a near-linear map of the values. On
  simulated batches the "soft Spearman" matches Pearson to within 0.006 and
  differs from the true Spearman by up to 0.12.
- So the method as run is a Pearson-correlation loss on cosines. Either describe
  it that way, or install torchsort (or raise the strength) and rerun. Worth a
  strength sweep on one seed either way. The hyperparameter lists that say
  "torchsort, strength 1.0" are wrong.

## Checkpoint selection

- Done in code: early stopping is off by default, `--early-stop` turns it on,
  and the reproduction runners pass it explicitly. Still to decide: the fixed
  step to report at, or a non-SimLex dev set.
- Drop early stopping. Report every run at a fixed step chosen in advance, or
  select on a dev set that is not SimLex (WordSim-353 is in
  `brain_and_glue/data` and unused). The no-early-stop runs already exist:
  GPT-2 Neither +0.021 at 2,000 steps, +0.023 at 8,000; BERT +0.065 at 8,000.
- Both keeps falling with training length, so the fixed step count shapes the
  Both result. Pick it before looking.

## Supervision balance and verbs

- SimVerb is 3,500 of 1,721,231 supervision pairs (0.20%). THINGS supplies the
  rest because every pair of its 1,854 words is scored. In a 24-word RSR batch
  roughly 7 words are SimVerb verbs, but verb pairs are only 1% dense, so well
  under 1% of the scored pairs in a batch are verb pairs. Verbs are supervised
  in name only.
- The SimLex partitions are confounded with part of speech (counts before the
  sentence-coverage filter, which removes 3 or 4 pairs from each):

  | partition | nouns | verbs | adjectives |
  |---|---|---|---|
  | Both | 105 | 192 | 0 |
  | One | 191 | 28 | 0 |
  | Neither | 370 | 2 | 111 |

  Both is two thirds verbs, Neither has almost none. Two consequences:
  1. The Both decline may be a verb effect. Split the Both change by POS on the
     existing result CSVs before claiming anything about supervised pairs.
  2. Part of the Neither gain could be noun and adjective structure carried over
     from THINGS rather than generalisation as such. Report Neither by POS too.
- Rebalance the sampler so SimVerb pairs make up a fixed share of scored pairs
  per batch (for example a verb batch every k steps, or stratified sampling).
  Expect Both to improve if the verb explanation is right.
- Switches: `--sampler pairs`, `--simverb-share F`, `--things-pairs N`,
  `--inflect-verbs`. Every run now scores SimLex by part of speech
  (`both_V`, `neither_A` and so on), so point 1 can be read off any new run.
  Audit, scored pairs per 24-word batch (SimVerb share in brackets):

  | configuration | pairs | SimVerb |
  |---|---|---|
  | reported runs (uniform words, all THINGS) | 139 | 0.26 (0.2%) |
  | `--sampler pairs` | 275 | 0.03 (0.0%) |
  | `--sampler pairs --simverb-share 0.33` | 131 | 4.8 (3.6%) |
  | `--sampler pairs --things-pairs 10000` (the original pipeline) | 13 | 3.5 (26%) |

  Pair-first sampling on its own makes the imbalance worse, because the
  pairs are drawn from 1.7M THINGS pairs. Fixing the SimVerb share does not
  help much either: the THINGS pairs that come along between the chosen
  words swamp it. Only cutting THINGS to a sample moves SimVerb to a real
  share, and the batch then carries a tenth of the scored pairs. So the
  per-step RSR signal changes size as well, which a run has to account for
  (lambda or batch size).
- Inflected verb forms have almost no harvested sentences (21 of 2,460), so
  `--inflect-verbs` does nothing in the sentence context until they are
  harvested (README, Switches). In the template context it works now.

## Train and test overlap

- 274 of the 297 Both pairs are scored pairs in the supervision itself: 170
  SimLex verb pairs are part of SimVerb, and 104 noun pairs get a THINGS score.
  Only 23 Both pairs are genuinely unseen as pairs. One and Neither are clean by
  construction. The code comments and README say SimLex is never trained on,
  which is wrong for Both and must be corrected wherever Both is described.
  Worth reporting as a result in itself: training on those exact pairs does not
  raise their SimLex score.
- Decide whether to drop the SimLex pairs from the supervision and rerun, which
  turns Both into a clean "words seen, pairs unseen" test.
  `--exclude-simlex-pairs` does this. It drops 274 pairs (170 SimVerb, 104
  THINGS) and no word leaves the supervision, so the partition is unchanged
  and the run compares directly with the reported ones. Every run also
  scores `both_trained` and `both_unseen` separately.
- The original RoBERTa pipeline expands SimVerb to inflected forms (eats, ate,
  eating, same score) with lemminflect and pluralises nouns with inflect, then
  filters by corpus frequency. This pipeline does neither. If inflection
  expansion is added, check that no expanded pair coincides with a SimLex pair
  or an inflection of one.
- Sentence leakage. Anchor sentences and evaluation sentences come from the
  same pool. Split the pool once and rerun one seed. `--split-sentences`
  trains on the first half of each word's sentences and evaluates on the
  second half, as the original pipeline does.

## Nouns-only experiment

- Train on THINGS only. Hold out a set of THINGS words entirely (no pair that
  contains them is trained on) and test on the held-out pairs, plus Pereira
  brain RSA, which is mostly concrete nouns. This removes the POS confound and
  the SimLex/SPoSE mismatch.

## Controls and statistics

- The anchor sentence pool (sentences of 400 random targets) was drawn before
  any seed was set, in both scripts, so no compare run is exactly
  reproducible. Both arms of a run share the pool, so the controls are still
  matched. Fixed in the refactor; runs from now on are reproducible.

- The RSR arm and its control do not see the same anchor batches after step 1.
  Both draw from the global `random` and torch generators, and the RSR arm also
  samples RSR batches and runs dropout on them. The control still differs only
  in having no RSR term, but the data stream is not identical. Give the anchor
  sampler its own seeded generator so both arms see the same batches. This
  changes numbers, so do it before the next full set of runs, not mid-set.
- GPT-2 seeds 4 and 5.
- Bootstrap confidence intervals instead of min/max ranges. State that Neither
  is the pre-specified test and the other partitions are secondary.

## fMRI teacher

- Decide the teacher (region, network average or per participant), estimate a
  noise ceiling, design the control, then write the sampler.
