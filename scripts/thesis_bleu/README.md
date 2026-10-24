# Thesis BLEU re-evaluation

Scripts for measuring the thesis-era machine-translation checkpoints with real
beam search rather than teacher-forced argmax outputs.

The old thesis scores were computed on teacher-forced validation outputs
(`save_eval_outputs` in the legacy `TransformerModel`). Those outputs are not
translations and contain repeated spans, so the resulting BLEU is inflated and
not comparable to normal MT evaluation. These scripts decode the held-out test
set with the legacy beam search, then score the result with the same COCO BLEU
implementation that `nlg-eval` wraps.

## Requirements

- Python 3.10 and the project dependencies (`uv sync`), including
  `pycocoevalcap`, `sacrebleu`, `sacremoses`, and PyTorch.
- The model checkpoints and datasets. They are gitignored and live in the main
  checkout (`/Users/t/projects/specialk/models` and
  `/Users/t/projects/specialk/datasets`), not in a worktree.
- On Apple silicon the decoder uses MPS automatically; pass `--device cpu` to
  force CPU.

Run from the repository root. The examples below use `python`; substitute
`.venv/bin/python` or `uv run python` as appropriate.

## 1. Build the filtered test set

```bash
python scripts/thesis_bleu/build_testset.py
```

This applies `datasets/machine_translation/filter_gm_dataset.ipynb` verbatim to
`corpus_enfr_final.test.{en,fr}` and writes to
`/Users/t/projects/specialk/results/thesis_bleu/data/`:

- `test_gm.{en,fr}` — 290,577 lines (50,303 removed from 340,880).
- `test_gm.5k.{en,fr}` and `test_gm.20k.{en,fr}` — seed-0 samples.
- `*.idx` — indices of the sampled lines in the full filtered set.

## 2. Decode

One row:

```bash
PYTHON=.venv/bin/python scripts/thesis_bleu/run_decode.sh final_enfr \
  /Users/t/projects/specialk/models/enfr_bpe_gold_master_mk2/encoder_epoch_3.chkpt \
  /Users/t/projects/specialk/models/enfr_bpe_gold_master_mk2/decoder_epoch_3.chkpt \
  /Users/t/projects/specialk/models/enfr_bpe_gold_master_mk2/vocab_need_to_rebase.pt \
  /Users/t/projects/specialk/results/thesis_bleu/data/test_gm.5k.en \
  /Users/t/projects/specialk/results/thesis_bleu/hyp/5k/final_enfr.txt \
  5000
```

The script passes `--beam 5 --batch-size 128`. Decoding resumes automatically:
if the output file already contains N lines, it starts from line N. This makes
long runs restartable.

Checkpoint and vocabulary matrix:

| Row | Encoder / decoder | Vocabulary |
|---|---|---|
| Final EN→FR | `enfr_bpe_gold_master_mk2/encoder_epoch_3.chkpt`, `decoder_epoch_3.chkpt` | `enfr_bpe_gold_master_mk2/vocab_need_to_rebase.pt` |
| Final FR→EN | `fren_bpe_gold_master_mk2/encoder_epoch_3.chkpt`, `decoder_epoch_3.chkpt` | `fren_bpe_gold_master_mk2/nmt_fren_goldmaster_mk2_bpe.pt` |
| Attempt-2 EN→FR | `enfr_bpe_gold_master/encoder_epoch_1.chkpt`, `decoder_epoch_1.chkpt` | `nmt_enfr_goldmaster_bpe.pt` |
| Attempt-2 FR→EN | `fren_goldmaster_attempt_2_democratic/encoder_epoch_2.chkpt`, `decoder_epoch_2.chkpt` | `nmt_fren_goldmaster_bpe.pt` |

Notes on the mapping:

- `/enfr_bpe_gold_master_mk2/vocab.pt` is a 227K-pair rebased vocabulary, not
  the NMT training vocabulary. The checkpoint matches
  `vocab_need_to_rebase.pt` (teacher-forced validation loss 2.18 vs 3.90).
- The surviving attempt-2 EN→FR model is the full-data gold master
  (11.4M pairs), not the 3.5M-pair XL model described in the thesis. No XL
  checkpoint survives.
- The surviving attempt-2 FR→EN pair does not decode. It emits `t` / `a` / `i`
  for almost every line and is not a usable model; its original teacher-forced
  eval output is also degenerate. That row is not reproducible.

## 3. Score

```bash
python scripts/thesis_bleu/score.py \
  --hyp /Users/t/projects/specialk/results/thesis_bleu/hyp/5k/final_enfr.txt \
  --ref /Users/t/projects/specialk/results/thesis_bleu/data/test_gm.5k.fr \
  --lang fr --json
```

`score.py` prints COCO BLEU-1..4 (×100), sacreBLEU, and the sacreBLEU signature
string. COCO BLEU is computed on whitespace tokens, matching the
`nlg-eval`/`pycocoevalcap` setup used in the thesis. sacreBLEU is computed on
sacremoses-detokenised text with the default 13a tokenizer.

Sanity checks:

- Scoring the reference against itself gives 100.0 for all four BLEU values and
  sacreBLEU.
- The scorer reproduces the old logged teacher-forced validation scores
  exactly: FR→EN mk2 epoch 2 gives `26.7739` BLEU-4, and EN→FR gold-master
  epoch 1 gives `22.1662` BLEU-4.

## 4. Run the whole matrix

The sample runs are sequential because they share one MPS device:

```bash
M=/Users/t/projects/specialk/models
D=/Users/t/projects/specialk/results/thesis_bleu/data
O=/Users/t/projects/specialk/results/thesis_bleu/hyp/20k
mkdir -p "$O"

scripts/thesis_bleu/run_decode.sh final_enfr \
  "$M/enfr_bpe_gold_master_mk2/encoder_epoch_3.chkpt" \
  "$M/enfr_bpe_gold_master_mk2/decoder_epoch_3.chkpt" \
  "$M/enfr_bpe_gold_master_mk2/vocab_need_to_rebase.pt" \
  "$D/test_gm.20k.en" "$O/final_enfr.txt" 20000

scripts/thesis_bleu/run_decode.sh final_fren \
  "$M/fren_bpe_gold_master_mk2/encoder_epoch_3.chkpt" \
  "$M/fren_bpe_gold_master_mk2/decoder_epoch_3.chkpt" \
  "$M/fren_bpe_gold_master_mk2/nmt_fren_goldmaster_mk2_bpe.pt" \
  "$D/test_gm.20k.fr" "$O/final_fren.txt" 20000

scripts/thesis_bleu/run_decode.sh attempt2_enfr \
  "$M/enfr_bpe_gold_master/encoder_epoch_1.chkpt" \
  "$M/enfr_bpe_gold_master/decoder_epoch_1.chkpt" \
  "$M/nmt_enfr_goldmaster_bpe.pt" \
  "$D/test_gm.20k.en" "$O/attempt2_enfr.txt" 20000
```

Throughput on a 16 GB Apple-silicon machine is about 12 lines/s with beam 5 and
batch 128, so 20K lines take roughly 27 minutes per row. The full filtered test
set takes about 6.5 hours per row.

## Legacy decoder changes

`specialk/models/transformer/legacy/Beam.py` needed two fixes for modern
PyTorch and accelerator use:

- `best_scores_id / num_words` now uses floor division. Current PyTorch returns
  a float from `/` on integer tensors, and the result is used as an index.
- Beam bookkeeping (`next_ys`, `prev_ks`) is kept on the CPU. Reading it token
  by token on MPS caused one device synchronisation per token per hypothesis;
  moving it to CPU made decoding about 12× faster with byte-identical output.

The legacy model code itself is unchanged.
