# Generating translations

## Training a model

    uv run python -m specialk.trainers.train --dataset mt --prod

Use `--dataset dummy` if you just want a small model that trains in minutes,
and `--back-translation` to swap the source and target languages.

Checkpoints land in:

    tb_logs/<task>/<model>/version_<n>/checkpoints/*.ckpt

I keep the three best checkpoints by `val_loss`. Each one carries its source
and target tokenizers, so you don't need to pass a separate vocab file around.

## Generating from Python

    from pathlib import Path

    from specialk.models.transformer.torch.pytorch_transformer import (
        PyTorchTransformerModule,
    )

    checkpoints = Path("tb_logs/nmt_model_dummy/transformer_smol/version_0/checkpoints")
    ckpt = next(checkpoints.glob("*.ckpt"))
    module = PyTorchTransformerModule.load_from_checkpoint(ckpt, weights_only=False)
    module.eval()  # required on MPS: train-mode dropout is rejected at inference

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

`out` is shape `batch × num_return_sequences × seq_len`, and `scores` holds one
log-probability per returned sequence. If you'd rather sample than search, use
`module.generate(tokens, top_p=..., top_k=..., temperature=...)`.

The same `beam_search` call works for recurrent checkpoints (`RNNModule`), too.
The architecture comes from the checkpoint, so there's nothing to change.

## Generating from the command line

    uv run python -m specialk.translate \
        --checkpoint path/to/model.ckpt \
        --src input.txt \
        --output predictions.txt \
        --beam-size 5 --n-best 1 --batch-size 128 --max-new-tokens 100

`--checkpoint` and `--src` are required; everything else has a default:

- `--output`: where the predictions go. Defaults to `predictions.txt`, with one line per input, or n-best lines per input when you ask for more.
- `--beam-size`: number of beams. Defaults to 5.
- `--n-best`: hypotheses kept per input, best first. Has to be at most `--beam-size`; defaults to 1.
- `--batch-size`: sentences per forward pass. Defaults to 128.
- `--max-new-tokens`: generation length cap. Defaults to 100.
- `--no-repeat-ngram-size`: blocks repeated n-grams. Off by default.
- `--device`: `mps`, `cuda`, `cuda:N` or `cpu`. Defaults to MPS, then CUDA, then CPU.

Both transformer and recurrent checkpoints work, and both support n-best.
There's no `--model` flag anymore; the checkpoint says which one to build.

The old flags map over like this:

- `-checkpoint_encoder` / `-checkpoint_decoder` / `-vocab` become one `--checkpoint`.
- `-model transformer|recurrent` is gone; the checkpoint decides.
- `-cuda` / `-cuda_device` become `--device`.
- `-telegram`, `-copysrc` and `-override_max_token_seq_len` are gone.

Style-transfer decoding uses the same command. The style-specific decoder is
fine-tuned from an NMT checkpoint and saved as an ordinary checkpoint, so a
single `--checkpoint` is enough. Split encoder/decoder checkpoints are a legacy
artefact and aren't supported.

## Where this fits in the pipeline

Translation is the back-translation step of style transfer:

1. Train the EN→FR and FR→EN NMT models (`scripts/scripts/prod/train_nmt_models.sh`).
2. Translate the style corpora (political, publications) into French with the EN→FR model. That's where `translate` gets called.
3. Train the style-specific decoders on the result (`train_pol_st_models.sh`, `train_pub_st_models.sh`).

The scripts are being moved to this CLI one by one, starting with the prod ones.
