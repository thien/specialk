import lightning.pytorch as pl
import pytest
import torch
from peft import LoraConfig
from peft.utils.peft_types import TaskType

from specialk.core.utils import check_torch_device, log
from specialk.models.classifier.models import BERTClassifier
from tests.models.fixtures import dataset  # noqa: F401; noqa: F402
from tests.models.fixtures import hf_bert_dataloader  # noqa: F401; noqa: F402
from tests.models.fixtures import hf_bert_tokenizer  # noqa: F401; noqa: F402
from tests.models.fixtures import hf_distilbert_dataloader  # noqa: F401; noqa: F402
from tests.models.fixtures import hf_distilbert_tokenizer  # noqa: F401; noqa: F402

ref_model_base_name = "bert-base-uncased"
ref_distil_model_base_name = "distilbert/distilbert-base-cased"


import logging

# supress bert renaming warnings (re. gamma, beta).
# see https://github.com/huggingface/transformers/pull/31654
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)

torch.manual_seed(1337)
torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session", autouse=True)
def pretrained_bert() -> BERTClassifier:
    return BERTClassifier(name="test", model_base_name=ref_model_base_name)


@pytest.fixture(scope="session", autouse=True)
def pretrained_peft_bert() -> BERTClassifier:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value"],
    )
    return BERTClassifier(
        name="test", model_base_name=ref_model_base_name, peft_config=peft_config
    )


@pytest.fixture(scope="session", autouse=True)
def pretrained_peft_distilbert() -> BERTClassifier:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    )
    return BERTClassifier(
        name="test_distilbert",
        model_base_name=ref_distil_model_base_name,
        peft_config=peft_config,
    )


@pytest.mark.heavyweight
def test_model_inference(hf_bert_dataloader, pretrained_bert):  # noqa: F811
    model = pretrained_bert
    model.eval()

    batch: dict = next(iter(hf_bert_dataloader))
    x: torch.Tensor
    y: torch.Tensor
    x, y = batch["text"], batch["label"]

    x = x.squeeze(2, 1)

    x_mask = x != model.tokenizer.pad_token_id
    log.info("input", x=x.shape)
    y_hat = model.model(x, attention_mask=x_mask, labels=y)
    log.info(y_hat)
    output = y_hat.logits


@pytest.mark.heavyweight
def test_model_inference_peft(hf_bert_dataloader, pretrained_peft_bert):  # noqa: F811
    model = pretrained_peft_bert
    model.eval()

    batch: dict = next(iter(hf_bert_dataloader))
    x: torch.Tensor
    y: torch.Tensor
    x, y = batch["text"], batch["label"]

    x = x.squeeze(2, 1)

    x_mask = x != model.tokenizer.pad_token_id
    log.info("input", x=x.shape)
    y_hat = model.model(x, attention_mask=x_mask, labels=y)
    log.info(y_hat)
    output = y_hat.logits


@pytest.mark.heavyweight
def test_model_inference_peft_distilbert(
    hf_distilbert_dataloader, pretrained_peft_distilbert
):  # noqa: F811
    model = pretrained_peft_distilbert
    model.eval()

    batch: dict = next(iter(hf_distilbert_dataloader))
    x: torch.Tensor
    y: torch.Tensor
    x, y = batch["text"], batch["label"]

    x = x.squeeze(2, 1)

    x_mask = x != model.tokenizer.pad_token_id
    log.info("input", x=x.shape)
    y_hat = model.model(x, attention_mask=x_mask, labels=y)
    log.info(y_hat)
    output = y_hat.logits


@pytest.mark.heavyweight
def test_training_step_with_embedding_noise(
    hf_distilbert_dataloader, pretrained_peft_distilbert
):  # noqa: F811
    model = pretrained_peft_distilbert
    model.embedding_noise = 0.1
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator=check_torch_device(),
    )

    trainer.fit(model, train_dataloaders=hf_distilbert_dataloader)
