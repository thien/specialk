# Plan: train the decoder with a style classifier

Code and small checks only. No full-scale training.

**What this plan assumes:**
- The data (`thien/political`, `thien/publications`) and the `--dataset mt` French→English checkpoint are ready to use.
- Milestone A is a CNN classifier plus our transformer, sharing the checkpoint's SentencePiece vocabulary.
- Milestone B is MarianMT with LoRA plus BERT.
- Nothing here runs more than a few batches. Every check runs on MPS.

**Rule for every step:** `NMTModule`, `PyTorchTransformerModule` and `MarianMTModule` keep their constructor arguments and weight names, so the saved checkpoint keeps loading. `test_sequence_models.py` and `test_generator.py` must stay green.

---

## Step 1: model fixes, `specialk/models/style_decoder.py`

### 1.1 Constructor
```python
StyleBackTranslationModel(
    mt_model, classifier,
    target_label: int = 1,
    use_mapping: bool = True,
    mapping_dim: int = 256,
    lambda_class: float = 1.0,      # thesis λ_c
    tau_start: float = 1.0,         # softmax temperature, decays linearly
    tau_end: float = 0.1,
    learning_rate: float = 1e-3,
)
```
- Call `self.save_hyperparameters(ignore=["mt_model", "classifier"])`.
- **Freezing:**
  - All classifier parameters get `requires_grad=False`.
  - Encoder parameters get the same, unless the seq2seq model uses LoRA, in which case PEFT has already frozen the non-LoRA weights.
  - Look the encoder up with `mt_model.model.encoder` for our transformer and `mt_model.model.model.encoder` for Marian, the same paths the tests use now.
- `self.seq_pad = PAD` for our tokenizer and `mt_model.tokenizer.pad_token_id` for Marian. Marian's padding id isn't 0.

### 1.2 Keep frozen parts in eval mode
Lightning calls `.train()` on the whole module during `fit`, which turns the classifier's dropout back on. Override:
```python
def train(self, mode=True):
    super().train(mode)
    self.classifier.eval()
    self._encoder().eval()
    return self
```

### 1.3 One forward pass with clear outputs
```python
def _forward(x, y) -> tuple[seq_logits, clf_out]
```
- `seq_logits` has shape `[batch, len, vocab_seq]` and is the **only** thing the reconstruction loss sees. This fixes the bug where the loss was computed on logits already projected into the classifier's vocabulary.
- The classifier input is `soft = softmax(seq_logits / τ, -1)`.
- With the mapping, `soft = softmax(clf_projection(seq_projection(soft)), -1)`, so the classifier still gets a distribution.
- The hook `_classify(soft, y) -> clf_out` is implemented per subclass. The base class raises `NotImplementedError`, and the base-class test is deleted.

### 1.4 Per-subclass classifier pass and loss

| | `_classify` | `classifier_loss(clf_out, target)` |
|---|---|---|
| CNN | `soft` → `[len, batch, vocab]`, then pad or truncate `len` to `classifier.sequence_length` (ConvNet's pooling window is fixed to that length), then `classifier.model(...)` → sigmoid output `[batch]` | binary cross-entropy (`TextClassifier.criterion`) |
| BERT | `embeds = soft @ classifier.model.get_input_embeddings().weight`, `mask = (y != self.seq_pad)` (flips the current inverted mask), returns `.logits` | cross-entropy (`BERTClassifier.criterion`) on `target.long()` |

`get_input_embeddings()` works for BERT, DistilBERT and LoRA-wrapped models. The current `.bert.embeddings` breaks on DistilBERT, which is what `train_classifier.py` produces.

### 1.5 Loss and steps
```python
target = torch.full_like(label, self.target_label)
recon  = mt_model.criterion(seq_logits.flatten(0, 1), y.flatten())
cls    = self.classifier_loss(clf_out, target)
loss   = recon + self.lambda_class * cls
```
- `training_step` and `_shared_eval_step` both go through one `_step(batch, stage)`. That fixes `batch["class"]` and `self.constants.PAD`.
- Logged under the shared constant names, with `batch_size=` passed:
  - `train_loss` / `eval_loss` (from `VAL_LOSS`)
  - `*_recon_loss`, `*_class_loss`
  - `*_acc` (token accuracy using `self.seq_pad`)
  - `tau`
- **Validation style accuracy:**
  - Run `argmax` on `seq_logits`, build a one-hot tensor and pass it through `_classify`.
  - Count the share predicted as `target_label`: threshold 0.5 for the CNN, argmax for BERT.
  - Log it as `eval_style_acc`. This is the number that shows whether the setup works.

### 1.6 Temperature schedule
- `on_train_batch_start`: τ moves linearly from `tau_start` to `tau_end` over `self.trainer.estimated_stepping_batches`.
- Validation uses the current τ.

### 1.7 Optimizer
- `AdamW([p for p in self.parameters() if p.requires_grad], lr=self.hparams.learning_rate)`.
- This replaces `_get_lora_params` and its fallback, which currently trains the encoder too.
- Remove the `log.info("x.shape", xhape=...)` line.

### 1.8 Subclasses
- `StyleBackTranslationModelWithCNN` and `...WithBERT` keep their `isinstance` asserts and implement only `_classify` and `classifier_loss`.
- BERT still forces `use_mapping=True`.

**Done when:** the step 5 tests pass on MPS.

---

## Step 2: data, `specialk/trainers/common.py`

```python
def load_style_datasets(name: str, target_label: Optional[int] = None,
                        num_proc: int = 8) -> tuple[Dataset, Dataset]
```
- `name` can be `"political"` or `"publications"`. Both load the `style` / default config.
- Rename the French column to `SOURCE` and `text` to `TARGET`. Remove the other text columns. Keep `label`.
- `class_encode_column("label")` for publications. Political already stores labels as integer classes (`republican=0`, `democratic=1`).
- If `target_label` is given, filter to those rows.
- Return `(train, eval)` using the existing `_single_split` helper.
- `init_dataloader` already passes extra columns through. A test confirms `batch["label"]` arrives as an integer tensor.

**Check:** the lightweight test loads a few rows from the local `en_fr_labels.parquet` fixture through the same rename and filter path. This needs a `path=` override so it doesn't download.

---

## Step 3: classifier trainer, `specialk/trainers/train_classifier.py`

- **CLI:**
  - `__main__` parses the arguments and calls `main_new`.
  - Rename `args.model` → `args.model_type`.
  - Add `--dataset_name`, `--config` (political `classifier`), `--max-seq-len`, `--epochs`, `--lr`, `--accelerator`, `--limit-train-batches`, `--limit-val-batches`.
  - Remove the hardcoded dataset name.
- **Tokenizer:** add `--mt-checkpoint`. When it's given, load `PyTorchTransformerModule.load_from_checkpoint(...)` and use its `tokenizer` and `sequence_length`, so the CNN shares the decoder's vocabulary and length.
- **`CNNClassifier` (`models/classifier/models.py`):**
  - Learning rate becomes an argument, default 0.001 (thesis value), replacing the hardcoded 0.02.
  - Call `self.save_hyperparameters(ignore=["tokenizer"])` so a new checkpoint loads without `hparams_file`. The legacy loads in `style_transfer.py` still pass `hparams_file`, which takes precedence. `tests/models/classifier/test_classifier.py` must stay green.
- `main_distilbert_peft` stays as it is, selected by `--model_type bert`.

**Check:** CLI smoke test with a tiny CNN and `--limit-train-batches 2`. Then reload the checkpoint with `CNNClassifier.load_from_checkpoint(path)` and no hparams file.

---

## Step 4: entry point, `specialk/trainers/train_style_decoder.py`

```
--dataset political|publications   --target-label INT
--mt-checkpoint PATH               --classifier-checkpoint PATH  --classifier cnn|bert
--lambda-class 1.0  --tau-start 1.0  --tau-end 0.1  --learning-rate 1e-3
--batch-size  --max-epochs 3  --limit-train-batches  --limit-val-batches
--accelerator auto  --precision  --n-workers  --seed
```
- Load the seq2seq with `load_from_checkpoint`, read-only; its tokenizer tokenizes the data.
- Load the classifier with `CNNClassifier.load_from_checkpoint` or `BERTClassifier.load_from_checkpoint`.
- Data: `load_style_datasets(..., target_label)` → `init_dataloader`.
- Trainer settings are copied from `train.py`:
  - TensorBoard logging to `tb_logs/style_decoder/<dataset>_<label>`
  - gradient clipping 0.5
  - `ModelCheckpoint(monitor=VAL_LOSS, save_top_k=-1)` to save every epoch. The thesis found 3 epochs before overfitting.
- `main(argv)` is testable, the same pattern as `train.py`.
- `train.py`'s checkpoint callback monitors `"val_loss"` while the code logs `"eval_loss"`. That mismatch exists already; noted here, left alone.

---

## Step 5: tests, `tests/models/test_style_decoder.py`

- **Delete:** the three copies of `test_model_inference_transformer_cnn_with_class`, including the stale `module.mapping` one.
- **Fixtures:** keep the tiny `PyTorchTransformerModule` and the CNN on `enfr_small.model`. The CNN's `sequence_length` equals the dataloader's length.
- **Accelerator:** `check_torch_device()` instead of `"cpu"`.
- **Markers:** BERT and Marian tests are `heavyweight`; CNN tests are `lightweight`.

| Test | What it checks |
|---|---|
| `test_cnn_fit_with_validation` | `fit` with train and val loaders runs; `eval_loss` and `eval_style_acc` are logged |
| `test_bert_fit_with_validation` | the same, for BERT |
| `test_frozen_parts_unchanged` | after `fit`, classifier and encoder weights match the originals and the decoder's changed; the classifier is in eval mode |
| `test_recon_loss_uses_seq_vocab` | with `use_mapping=True`, `seq_logits.shape[-1] == mt.vocabulary_size` |
| `test_target_label_changes_class_loss` | same batch, `target_label` 0 vs 1 → different `cls` |
| `test_tau_decays` | τ at the last step is about `tau_end`, and lower than at the start |
| `test_bert_mask` | the mask is 1 on non-padding tokens of `y` |
| `test_cnn_length_pad_truncate` | decoder lengths shorter and longer than the CNN's both work |
| `test_optimizer_params` | only `requires_grad` parameters, no encoder or classifier |
| `test_cli_smoke` (step 4) | `main([... "--limit-train-batches", "1", "--limit-val-batches", "1"])` with tiny saved checkpoints in `tmp_path` |
| `test_load_style_datasets` (step 2) | the fixture parquet gives `source`, `target` and an integer `label` |

---

## Step 6: small check that it learns (minutes, not a training run)

- **Overfit one batch:** a tiny transformer plus a CNN on a fixed batch of about 50 rows for about 200 steps on MPS.
- Expect `train_class_loss` to drop and `eval_style_acc` (on that batch) to rise toward 1 while `recon_loss` stays bounded.
- Written as a `benchmark`-marked test so it isn't in the default run.
- Then one smoke run of the step 4 CLI against the real `mt` checkpoint, with `--limit-train-batches 5 --limit-val-batches 2`, on political. This only proves the real checkpoint, tokenizer and data fit together.

---

## Milestone B: MarianMT with LoRA plus BERT (code only)

- `MarianMTModule` asserts `TaskType.SEQ_CLS`. Change it to `SEQ_2_SEQ_LM` and update the LoRA test's config to match. Check that the decoder cross-attention modules still get adapters, by counting `lora_` parameters.
- Step 1 already covers the BERT path: mapping, embeddings and mask.
- Step 4 gets `--seq2seq transformer|marian` and `--lora-r`.
- Check: the existing Marian+BERT LoRA trainer-loop test, now with a validation loader.

---

## Order and size

1 → 5 together, then 2 → 3 → 4, then 6. Each step is one commit on this branch.

**Out of scope:** full training, evaluation sweeps (intensity, preservation, naturalness), putting quotes back into full articles, KV cache and beam search changes, distillation, LLM baselines.

**Unconfirmed:**
- The column name and config of the French text in political. It isn't in `style` on HF, so step 2 takes the column name as a parameter.
- Whether `ConvNet`'s final layer outputs one value or two. The binary cross-entropy path assumes one.

---

## Follow-up: fixed maps and generated-text evaluation

- The BERT and CNN paths translate vocabularies through a deterministic surface-form token map. The trainable projection remains only when a caller asks for it explicitly.
- `specialk.trainers.evaluate_style_transfer` generates translations from French sources, classifies the generated English, and reports style accuracy and BLEU for the base and style-tuned checkpoints.
- The French political sources are the `.ts` files that `scripts/scripts/setup_political_dataset.sh` produced beside the `.en` files.
