# RSR_July

Replication of Barry's RoBERTa RSR work, plus the brain and downstream
evaluations run against our own BERT checkpoints.

`src/` is Barrys codebase (from `github.com/barry/RSR`) with a few additions of
mine. Everything of his is unchanged apart from one compatibility fix -
`RobertaTokenizer` -> `RobertaTokenizerFast` in `train.py` and
`eval_roberta_layers.py`, because `find_subword_span` needs
`return_offsets_mapping` and the slow tokenizer doesn't provide it on our
transformers version.

## Layout

```
src/                    his pipeline + my additions (marked below)
data/                   datasets, gitignored - see setup
results/
  replication/          his 4-mode grid, re-run here
  brain/                Pereira RSA for the BERT models
  downstream/           GLUE
notes/findings.md       what came out of it
```

## My additions to src/

| File | What it does |
|---|---|
| `eval_pereira_bert.py` | RSA against Pereira fMRI, BERT port of `eval_pereira_rsa.py` |
| `eval_downstream_bert.py` | GLUE fine-tuning, BERT version of `eval_downstream.py` |
| `build_pereira_betas.py` | Fetches and processes the fMRI betas one subject at a time |

Also added `--combine weighted --rsr_lambda` to `train.py`. Default is still
`interleave`, so nothing about his runs changes.

## Setup

```bash
pip install -r requirements.txt

python src/download_datasets.py          # WordSim353, SimVerb, THINGS, SimLex
python src/generate_wiki_targetwords.py  # streams Wikipedia, takes hours
python src/generate_wiki_pereira.py      # contexts for the 180 Pereira concepts
```

The fMRI betas (`data/ryskina_repo/outputs/rsa/betas_sentences_M*.csv`) came
from Barry directly. To rebuild them yourself you need the Ryskina repo and
~66 GB of GLMsingle archives:

```bash
git clone --depth 1 https://github.com/ryskina/concepts-brain-llms.git data/ryskina_src
python src/build_pereira_betas.py
```

Fair warning, the Drive downloads are slow and flaky. Getting the CSVs off
someone who already has them is a much better use of an afternoon.

## Running things

Replication grid (4 modes, his config):

```bash
python src/train.py --mode rsr_all --steps 8000 --eval_mode wiki_avg \
    --rsr_layer 5 --eval_steps 400 --run_id 1
```

Brain RSA for our BERT checkpoints:

```bash
python src/eval_pereira_bert.py \
    --models vanilla=bert-base-uncased \
             mlm_control=../models/bert_baseline_w07_s1.pt \
             rsr=../models/bert_rsr_w07_s1.pt \
    --layers all --skip-2v2
```

GLUE:

```bash
python src/eval_downstream_bert.py \
    --models vanilla=bert-base-uncased \
             mlm_control=../models/bert_baseline_w07_s1.pt \
             rsr=../models/bert_rsr_w07_s1.pt \
    --tasks mrpc rte cola stsb sst2 --seeds 1 2 3
```

Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for anything long-running -
a dropped connection otherwise kills the job at a `from_pretrained` call.

## Results

See `notes/findings.md`. Short version: his four modes all reproduce, our BERT
models improve brain fit on concepts that were never supervised, and GLUE comes
out at parity.
