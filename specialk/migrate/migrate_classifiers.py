import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from specialk.core import log
from specialk.core.constants import LOGGING_DIR
from specialk.core.utils import namespace_to_dict, save_dict_to_yaml
from specialk.models.classifier.onmt import CNNModels
from specialk.models.classifier.trainer import CNNClassifier
from specialk.models.tokenizer import WordVocabulary

# Add the directory containing onmt to sys.path
onmt_dir = os.path.dirname("/Users/t/Projects/specialk/specialk/onmt")
sys.path.append(onmt_dir)

import onmt


def migrate_tokenizer(vocab, name):
    d = {}
    d["vocab"] = {
        "idxToLabel": vocab.idxToLabel,
        "labelToIdx": vocab.labelToIdx,
        "frequencies": vocab.frequencies,
        "special": vocab.special,
    }
    d["class"] = "WordVocabulary"
    d["kwargs"] = {
        "lower": True,
        "max_length": vocab.seq_length,
        "name": name,
        "vocab_size": len(vocab.idxToLabel),
    }
    return WordVocabulary.from_dict(d)


def migrate_model(component, tokenizer, name):
    model = component["model"]
    vocab_size, word_vec_size = model["word_lut.weight"].shape

    opt = component["opt"]

    # load model.
    new_model = CNNModels.ConvNet(
        vocab_size=vocab_size,
        num_filters=opt.num_filters,
        num_classes=opt.num_classes,
        filter_size=opt.filter_size,
        word_vec_size=opt.word_vec_size,
        sequence_length=opt.sequence_length,
    )

    if model["word_lut.weight"].shape == new_model.word_lut.weight.shape:
        new_model.word_lut.weight.data = model["word_lut.weight"]
    else:
        print("???")

    if model["conv1.weight"].shape == new_model.conv.weight.shape:
        new_model.conv.weight.data = model["conv1.weight"]
        new_model.conv.bias.data = model["conv1.bias"]
    else:
        print("???")

    if model["linear.weight"].shape == new_model.linear.weight.shape:
        new_model.linear.weight.data = model["linear.weight"]
        new_model.linear.bias.data = model["linear.bias"]
    else:
        print("???")

    # copy over model
    module = CNNClassifier(
        name,
        vocabulary_size=vocab_size,
        sequence_length=opt.sequence_length,
        tokenizer=tokenizer,
    )

    module.model = new_model
    module.save_hyperparameters(ignore="ipython_dir")

    return module


def migrate_asset(component, name):
    vocab = component["dicts"]["src"]
    tokenizer = migrate_tokenizer(vocab, name)

    model = migrate_model(component, tokenizer, name)
    return tokenizer, model


def deal_with_components(path: Path, tgt_dir: Path):
    name = path.name.split(".")[0]
    log.info("loading path", path=path, name=name)
    component = torch.load(
        path,
        map_location=torch.device("cpu"),
    )
    VOC, new_model = migrate_asset(component, name)
    text = VOC.to_tensor(["Donald Trump!!!", "obama rules"])
    log.info(
        "generated output", out=new_model.model(new_model.one_hot(text)), input=text
    )

    # dummy trainer so we can save checkpoint (it's not a checkpoint)
    # but it's easier than writing all this padding to save the model
    # and the packaged vocab.
    hyperparams = {
        **namespace_to_dict(component["opt"]),
        "name": name,
        "vocabulary_size": new_model.vocabulary_size,
        "sequence_length": new_model.sequence_length,
    }

    log.info("hyperparams", params=hyperparams)

    logger = TensorBoardLogger(LOGGING_DIR)
    logger.log_hyperparams(params=hyperparams)
    trainer = pl.Trainer(max_epochs=0, logger=logger)
    dataloader = DataLoader([])
    trainer.fit(new_model, train_dataloaders=dataloader)

    tgt_path = tgt_dir / name / f"{name}.ckpt"
    tgt_hpram_path = tgt_dir / name / f"hyperparameters.yaml"
    tgt_tok_path = tgt_dir / name / f"tokenizer"

    trainer.save_checkpoint(tgt_path)
    log.info(f"saved asset to {tgt_path}")
    save_dict_to_yaml(hyperparams, tgt_hpram_path)

    VOC.to_file(tgt_tok_path)

    log.info("attempting to load new model", tgt_path=tgt_path)
    model2 = CNNClassifier.load_from_checkpoint(tgt_path, **hyperparams)
    log.info("loaded new model")
    log.info("new_output", out=model2.model(model2.one_hot(text)))
    log.info("we're good.")
    print("\n\n\n")


if __name__ == "__main__":
    src_dir = Path("/Users/t/Projects/specialk/specialk/metrics/cnn_models")
    tgt_dir = Path(
        "/Users/t/Projects/specialk/assets/classifiers/legacy/cnn_classifier/"
    )
    tgt_dir.mkdir(exist_ok=True)
    for filename in [
        "adversarial_political.pt",
        "adversarial_publication.pt",
        "naturalness_political.pt",
    ]:
        deal_with_components(src_dir / filename, tgt_dir)
