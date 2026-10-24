# Thesis BLEU re-evaluation

The logged thesis scores came from teacher-forced argmax validation outputs. They contain repeated spans and overstate translation quality. These scripts decode the held-out test set with the legacy beam search and score the output with the COCO BLEU implementation that `nlg-eval` uses, plus sacreBLEU.

## Requirements

- Python 3.11 to 3.13 and `uv sync`.
- The checkpoints and datasets are gitignored and live in the main checkout. The commands below use absolute paths.
- On Apple silicon the decoder uses MPS automatically. Pass `--device cpu` to force CPU.

## Test set

    uv run python scripts/thesis_bleu/build_testset.py

This applies `datasets/machine_translation/filter_gm_dataset.ipynb` verbatim to `corpus_enfr_final.test.{en,fr}`. The filtered test set is 290,577 lines, down from 340,880. It writes 5K and 20K seed-0 samples and their `.idx` files to `results/thesis_bleu/data/`.

## Decode

    uv run python scripts/thesis_bleu/decode.py \
      --encoder /Users/t/projects/specialk/models/enfr_bpe_gold_master_mk2/encoder_epoch_3.chkpt \
      --decoder /Users/t/projects/specialk/models/enfr_bpe_gold_master_mk2/decoder_epoch_3.chkpt \
      --vocab /Users/t/projects/specialk/models/enfr_bpe_gold_master_mk2/vocab_need_to_rebase.pt \
      --src /Users/t/projects/specialk/results/thesis_bleu/data/test_gm.5k.en \
      --out /Users/t/projects/specialk/results/thesis_bleu/hyp/5k/final_enfr.txt \
      --beam 5 --batch-size 128 --limit 5000

Output resumes from the existing line count, so long runs are restartable.

Models:

- Final EN→FR: `enfr_bpe_gold_master_mk2/encoder_epoch_3.chkpt` and `decoder_epoch_3.chkpt`, vocab `enfr_bpe_gold_master_mk2/vocab_need_to_rebase.pt`.
- Final FR→EN: `fren_bpe_gold_master_mk2/encoder_epoch_3.chkpt` and `decoder_epoch_3.chkpt`, vocab `fren_bpe_gold_master_mk2/nmt_fren_goldmaster_mk2_bpe.pt`.
- Attempt-2 EN→FR: `enfr_bpe_gold_master/encoder_epoch_1.chkpt` and `decoder_epoch_1.chkpt`, vocab `nmt_enfr_goldmaster_bpe.pt`.
- Attempt-2 FR→EN: `fren_goldmaster_attempt_2_democratic/encoder_epoch_2.chkpt` and `decoder_epoch_2.chkpt`, vocab `nmt_fren_goldmaster_bpe.pt`.

Notes:

- Use `vocab_need_to_rebase.pt` for final EN→FR. `vocab.pt` is a 227K-pair rebased vocabulary. Teacher-forced validation loss is 2.18 with the former and 3.90 with the latter.
- The surviving attempt-2 EN→FR checkpoint is the full-data 11.4M-pair gold master. The 3.5M-pair XL checkpoint is gone.
- The surviving attempt-2 FR→EN pair emits `t`, `a`, or `i` for almost every line and is not usable. Its original teacher-forced eval output is degenerate too.

## Score

    uv run python scripts/thesis_bleu/score.py \
      --hyp /Users/t/projects/specialk/results/thesis_bleu/hyp/5k/final_enfr.txt \
      --ref /Users/t/projects/specialk/results/thesis_bleu/data/test_gm.5k.fr \
      --lang fr --json

COCO BLEU is whitespace-tokenised, matching `nlg-eval` and `pycocoevalcap`. sacreBLEU uses sacremoses-detokenised text and the default 13a tokeniser.

Checks:

- Reference against itself is 100.0.
- The scorer reproduces the logged teacher-forced BLEU-4 values: FR→EN mk2 epoch 2 gives `26.7739`, and EN→FR gold-master epoch 1 gives `22.1662`.

## Decoder change

`specialk/models/transformer/legacy/Beam.py` needs floor division for beam indices and keeps beam bookkeeping on the CPU. Current PyTorch returns floats from `/` on integer tensors, and reading MPS tensors token by token caused one synchronisation per token per hypothesis. Output is byte-identical. Decoding is about 12 times faster.
