# brain_and_glue

Two evaluations of the RSR-trained BERT checkpoints that sit outside the main
pipeline, plus the replication that preceded them.

- **Brain RSA.** Representational similarity analysis against the Pereira et al.
  (2018) fMRI data: 180 concepts, 10 subjects, partitioned by supervision
  exposure the same way as SimLex.
- **GLUE.** MRPC, RTE, CoLA, STS-B and SST-2, fine-tuned from the vanilla,
  control and RSR checkpoints, three seeds each.
- **Replication.** The earlier RoBERTa RSR experiment, re-run in its four modes
  to confirm the baseline numbers before building on them.

## Layout

```
src/eval_pereira_bert.py      brain RSA for BERT checkpoints
src/eval_downstream_bert.py   GLUE fine-tuning for BERT checkpoints
src/build_pereira_betas.py    fetches and processes the fMRI betas, one subject at a time
results/brain/                three seeds
results/downstream/           three seeds
results/replication/          the four-mode grid
```

The three scripts run inside the RoBERTa RSR pipeline they were written for,
which is not included here. The fMRI betas need the Ryskina repository and
roughly 66 GB of GLMsingle archives; `build_pereira_betas.py` handles that,
slowly.

## Running

```bash
python src/eval_pereira_bert.py \
    --models vanilla=bert-base-uncased \
             mlm_control=../models/bert_baseline_w07_s1.pt \
             rsr=../models/bert_rsr_w07_s1.pt \
    --layers all --skip-2v2

python src/eval_downstream_bert.py \
    --models vanilla=bert-base-uncased \
             mlm_control=../models/bert_baseline_w07_s1.pt \
             rsr=../models/bert_rsr_w07_s1.pt \
    --tasks mrpc rte cola stsb sst2 --seeds 1 2 3
```

Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for anything long-running;
a dropped connection otherwise kills the job at a `from_pretrained` call.

Results are discussed in `../notes/findings.md`.
