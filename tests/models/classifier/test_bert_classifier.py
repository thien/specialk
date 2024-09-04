import pytest
import torch
from peft import LoraConfig
from peft.utils.peft_types import TaskType

from specialk.core.utils import log
from specialk.models.classifier.models import BERTClassifier, CNNClassifier
from tests.models.fixtures import dataset  # noqa: F401; noqa: F402
from tests.models.fixtures import hf_dataloader  # noqa: F401; noqa: F402
from tests.models.fixtures import hf_tokenizer  # noqa: F401; noqa: F402

dirpath = "tests/tokenizer/test_files"


torch.manual_seed(1337)


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


def test_model_inference(hf_dataloader, pretrained_bert):  # noqa: F811
    model = pretrained_bert
    model.eval()

    batch: dict = next(iter(hf_dataloader))
    x: torch.Tensor
    y: torch.Tensor
    x, y = batch["text"], batch["label"]

    x = x.squeeze(2, 1)

    x_mask = x != model.tokenizer.pad_token_id
    log.info("input", x=x.shape)
    y_hat = model.model(x, attention_mask=x_mask, labels=y)
    log.info(y_hat)
    output = y_hat.logits


def test_model_inference_peft(hf_dataloader, pretrained_peft_bert):  # noqa: F811
    model = pretrained_peft_bert
    model.eval()

    batch: dict = next(iter(hf_dataloader))
    x: torch.Tensor
    y: torch.Tensor
    x, y = batch["text"], batch["label"]

    x = x.squeeze(2, 1)

    x_mask = x != model.tokenizer.pad_token_id
    log.info("input", x=x.shape)
    y_hat = model.model(x, attention_mask=x_mask, labels=y)
    log.info(y_hat)
    output = y_hat.logits


# def test_load_model_from_checkpoint():
#     path_classifier = Path(
#         "/Users/t/Projects/specialk/assets/classifiers/legacy/cnn_classifier/"
#     )
#     for category in [
#         "adversarial_political",
#         "adversarial_publication",
#         "naturalness_political",
#     ]:
#         path_checkpoint = path_classifier / category / f"{category}.ckpt"
#         path_hyperparams = path_classifier / category / f"hyperparameters.yaml"
#         path_tok = path_classifier / category / f"tokenizer"

#         module = CNNClassifier.load_from_checkpoint(
#             path_checkpoint, hparams_file=path_hyperparams
#         )
#         module.tokenizer = WordVocabulary.from_file(path_tok)

#         text = ["Donald Trump!!!", "obama rules"] * 10
#         batch_size = 3
#         output = module.generate(text, batch_size)
#         log.info("output", out=output, shape=output.shape)
#         assert output.shape == torch.Size((len(text)))
#         assert isinstance(output, torch.Tensor)
