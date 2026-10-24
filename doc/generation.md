# Generating translations

> **Draft.** The Python API below works today. The `specialk.translate` CLI is
> planned: `specialk/translate.py` still targets the removed legacy models and
> will be rewritten to match this page.

## 1. Train a model

    uv run python -m specialk.trainers.train --dataset mt --prod

Use `--dataset dummy` for a small model that trains in minutes, and
`--back-translation` to swap source and target languages.

Checkpoints land in:

    tb_logs/<task>/<model>/version_<n>/checkpoints/*.ckpt

The three best checkpoints by `val_loss` are kept. Each one includes the
source and target tokenizers, so you don't need a separate vocab file.

## 2. Generate from Python

```python
from pathlib import Path

from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)

ckpt = next(Path("tb_logs/nmt_model_dummy/transformer_smol/version_0/checkpoints").glob("*.ckpt"))
module = PyTorchTransformerModule.load_from_checkpoint(ckpt, weights_only=False)
module.eval()  # required on MPS: dropout in train mode is rejected at inference

src = ["Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen"]
tokens = module.tokenizer.to_tensor(src).to("mps")

scores, out = module.beam_search(
    tokens,
    num_return_sequences=1,
    num_beams=5,
    max_new_tokens=50,
    no_repeat_ngram_size=3,
)
print(module.decoder_tokenizer.detokenize(out[:, 0, :], specials=False))
```

`out` has shape `batch × num_return_sequences × seq_len`, and `scores` holds
one log-probability per returned sequence. For sampling instead of beam
search, use `module.generate(tokens, top_p=..., top_k=..., temperature=...)`.

## 3. Generate from the command line (planned)

    uv run python -m specialk.translate \
        --checkpoint path/to/model.ckpt \
        --src input.txt \
        --output predictions.txt \
        --beam-size 5 --n-best 1 --batch-size 128 --max-new-tokens 100

| Flag | Meaning | Default |
|---|---|---|
| `--checkpoint` | Lightning `.ckpt` from step 1 | required |
| `--src` | input text, one sentence per line | required |
| `--output` | output file, one prediction per line (`n-best` lines per input) | `predictions.txt` |
| `--beam-size` | number of beams | 5 |
| `--n-best` | hypotheses kept per input, must be ≤ beam size | 1 |
| `--batch-size` | sentences per forward pass | 128 |
| `--max-new-tokens` | generation length cap | 100 |
| `--no-repeat-ngram-size` | block repeated n-grams | off |
| `--device` | `mps`, `cuda` or `cpu` | auto-detected |

Changes from the old `translate.py` flags:

- `-checkpoint_encoder` / `-checkpoint_decoder` / `-vocab` → one `--checkpoint`.
- `-model transformer|recurrent` → read from the checkpoint.
- `-cuda` / `-cuda_device` → `--device`.
- `-telegram`, `-copysrc` and `-override_max_token_seq_len` are dropped.

## 4. Where this fits in the pipeline

Translation is the back-translation step of style transfer:

1. Train EN→FR and FR→EN NMT models (`scripts/scripts/prod/train_nmt_models.sh`).
2. Translate the style corpora (political, publications) into French with
   the EN→FR model. This is where `translate` gets called.
3. Train the style-specific decoders on the result (`train_pol_st_models.sh`,
   `train_pub_st_models.sh`).

The scripts under `scripts/scripts/` still use the old flags. They will be
moved to the new CLI one by one, starting with the scripts in `prod/`.

## Open questions

- Should style-transfer decoding (encoder from the NMT model plus a
  style-specific decoder) go through the same CLI, or get its own entry point?
- Should we keep RNN checkpoints working, or only transformers?
