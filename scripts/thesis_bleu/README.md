# Thesis BLEU re-evaluation

The logged thesis scores came from teacher-forced argmax validation outputs,
not decoded translations. These scripts decode the held-out test set with the
legacy beam search and score the output with the COCO BLEU implementation that
`nlg-eval` uses, plus sacreBLEU.

## Requirements

Python 3.10 and `uv sync`. The checkpoints and datasets are gitignored and live
in the main checkout; the commands below use absolute paths.

## Test set

```bash
uv run python scripts/thesis_bleu/build_testset.py
```

Applies `datasets/machine_translation/filter_gm_dataset.ipynb` verbatim to
`corpus_enfr_final.test.{en,fr}`. The filtered test set is 290,577 lines
(50,303 removed from 340,880); 5K and 20K seed-0 samples and their `.idx`
files are written to `results/thesis_bleu/data/`.

## Decode

```bash
uv run python scripts/thesis_bleu/decode.py \
  --encoder /Users/t/projects/specialk/models/enfr_bpe_gold_master_mk2/encoder_epoch_3.chkpt \
  --decoder /Users/t/projects/specialk/models/enfr_bpe_gold_master_mk2/decoder_epoch_3.chkpt \
  --vocab /Users/t/projects/specialk/models/enfr_bpe_gold_master_mk2/vocab_need_to_rebase.pt \
  --src /Users/t/projects/specialk/results/thesis_bleu/data/test_gm.5k.en \
  --out /Users/t/projects/specialk/results/thesis_bleu/hyp/5k/final_enfr.txt \
  --beam 5 --batch-size 128 --limit 5000
```

Output resumes from the existing line count, so long runs are restartable.

| Row | Encoder / decoder | Vocabulary |
|---|---|---|
| Final EN→FR | `enfr_bpe_gold_master_mk2/encoder_epoch_3.chkpt`, `decoder_epoch_3.chkpt` | `enfr_bpe_gold_master_mk2/vocab_need_to_rebase.pt` |
| Final FR→EN | `fren_bpe_gold_master_mk2/encoder_epoch_3.chkpt`, `decoder_epoch_3.chkpt` | `fren_bpe_gold_master_mk2/nmt_fren_goldmaster_mk2_bpe.pt` |
| Attempt-2 EN→FR | `enfr_bpe_gold_master/encoder_epoch_1.chkpt`, `decoder_epoch_1.chkpt` | `nmt_enfr_goldmaster_bpe.pt` |
| Attempt-2 FR→EN | `fren_goldmaster_attempt_2_democratic/encoder_epoch_2.chkpt`, `decoder_epoch_2.chkpt` | `nmt_fren_goldmaster_bpe.pt` |

- Use `vocab_need_to_rebase.pt` for final EN→FR. `vocab.pt` is a 227K-pair
  rebased vocabulary; teacher-forced validation loss is 2.18 with the former
  and 3.90 with the latter.
- The surviving attempt-2 EN→FR checkpoint is the full-data 11.4M-pair gold
  master, not the 3.5M-pair XL model. No XL checkpoint survives.
- The surviving attempt-2 FR→EN pair emits `t` / `a` / `i` for almost every
  line and is not usable. Its original teacher-forced eval output is degenerate
  too.

## Score

```bash
uv run python scripts/thesis_bleu/score.py \
  --hyp /Users/t/projects/specialk/results/thesis_bleu/hyp/5k/final_enfr.txt \
  --ref /Users/t/projects/specialk/results/thesis_bleu/data/test_gm.5k.fr \
  --lang fr --json
```

COCO BLEU is whitespace-tokenised, matching `nlg-eval`/`pycocoevalcap`;
sacreBLEU uses sacremoses-detokenised text and the default 13a tokenizer.

Checks: reference-vs-reference is 100.0, and the scorer reproduces the logged
teacher-forced BLEU-4 values (FR→EN mk2 epoch 2: `26.7739`; EN→FR gold-master
epoch 1: `22.1662`).

## Decoder change

`specialk/models/transformer/legacy/Beam.py` needs floor division for beam
indices and keeps beam bookkeeping on the CPU. Current PyTorch returns floats
from `/` on integer tensors, and reading MPS tensors token-by-token caused one
synchronisation per token per hypothesis. Output is byte-identical; decoding is
about 12× faster.
