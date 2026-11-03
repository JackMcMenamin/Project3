"""
rsr — shared library for the Representational Similarity Regularisation experiments.

This package holds the logic that used to be copy-pasted across the various
`run_*_seeds.py` scripts: the soft-Spearman RSR loss, dataset loading
(MEN / SimVerb / THINGS for supervision, SimLex-999 for evaluation), the
per-architecture word-embedding wrappers, and the train / evaluate / report
helpers.

Experiment entry points live under `experiments/` and import from here.
"""

from . import paths, seeds, losses, datasets, models, train_eval, reporting

__all__ = ["paths", "seeds", "losses", "datasets", "models", "train_eval", "reporting"]
