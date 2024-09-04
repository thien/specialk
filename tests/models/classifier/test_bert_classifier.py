import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch import Trainer
from peft import LoraConfig
from peft.utils.peft_types import TaskType

from specialk.core.constants import PROJECT_DIR
from specialk.core.utils import log
from specialk.models.classifier.models import BERTClassifier, CNNClassifier
from tests.models.fixtures import dataset  # noqa: F401; noqa: F402
from tests.models.fixtures import hf_bert_dataloader  # noqa: F401; noqa: F402
from tests.models.fixtures import hf_bert_tokenizer  # noqa: F401; noqa: F402
from tests.models.fixtures import hf_distilbert_dataloader  # noqa: F401; noqa: F402
from tests.models.fixtures import hf_distilbert_tokenizer  # noqa: F401; noqa: F402

dirpath = "tests/tokenizer/test_files"


torch.manual_seed(1337)
torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session", autouse=True)
def pretrained_bert() -> BERTClassifier:
    return BERTClassifier(name="test", model_base_name="bert-base-uncased")


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
        name="test", model_base_name="bert-base-uncased", peft_config=peft_config
    )


@pytest.fixture(scope="session", autouse=True)
def pretrained_distilbert() -> BERTClassifier:
    return BERTClassifier(
        name="test_distilbert",
        model_base_name="distilbert/distilbert-base-cased",
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
        model_base_name="distilbert/distilbert-base-cased",
        peft_config=peft_config,
    )


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


def test_save_load_distilbert_checkpoint(
    hf_distilbert_dataloader, pretrained_distilbert, tmp_path
):
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    module = pretrained_distilbert
    trainer = Trainer(max_epochs=0, accelerator="cpu", logger=False)
    trainer.fit(module, train_dataloaders=hf_distilbert_dataloader)

    m = module.model

    # generate temporary path to save checkpoint and load from.
    trainer.save_checkpoint(checkpoint_path)

    ckpt_module = BERTClassifier.load_from_checkpoint(checkpoint_path)
    m2 = ckpt_module.model
    torch.testing.assert_close(
        m.distilbert.embeddings.word_embeddings.weight,
        m2.distilbert.embeddings.word_embeddings.weight,
    )


def test_save_load_peft_distilbert_checkpoint(
    hf_distilbert_dataloader, pretrained_peft_distilbert, tmp_path
):
    checkpoint_path = PROJECT_DIR / "peft_distilbert_checkpoint.ckpt"
    # checkpoint_path = tmp_path / "peft_distilbert_checkpoint.ckpt"
    module = pretrained_peft_distilbert

    trainer = Trainer(max_epochs=1, logger=False, limit_train_batches=1)

    trainer.fit(module, train_dataloaders=hf_distilbert_dataloader)

    # dump and load from checkpoint.
    trainer.save_checkpoint(checkpoint_path)
    ckpt_module = BERTClassifier.load_from_checkpoint(checkpoint_path)

    for (name1, param1), (_, param2) in zip(
        module.named_parameters(), ckpt_module.named_parameters()
    ):
        if not torch.allclose(param1, param2, rtol=1e-4, atol=1e-4):
            log.error(f"Mismatch in parameter {name1}", original=param1, loaded=param2)
            raise Exception
