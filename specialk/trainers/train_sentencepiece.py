import sentencepiece as spm

import specialk.core.constants as Constants
from specialk.core.utils import log

special_tokens = ["<blank>", "<unk>", "<s>", "</s>", "<p>", "<sep>", "<cls>"]
special_tokens = [i for i in special_tokens if i not in {"<unk>"}]

inp = f"{Constants.PROJECT_DIR}/datasets/machine_translation/corpus_enfr_final.en,{Constants.PROJECT_DIR}/datasets/machine_translation/corpus_enfr_final.fr"
log.info("Setting up SPM", src_data=inp)

DEFAULT_VOCAB_SIZE = 35000
SMALL_VOCAB_SIZE = 10000
spm.SentencePieceTrainer.train(
    input=inp,
    model_prefix="frenidentity_small",
    vocab_size=SMALL_VOCAB_SIZE,
    accept_language="en,fr",
    num_threads=14,
    control_symbols=special_tokens,
    bos_id=Constants.SOS,
    eos_id=Constants.EOS,
    unk_id=Constants.UNK,
    unk_piece=Constants.UNK_WORD,
    bos_piece=Constants.SOS_WORD,
    eos_piece=Constants.EOS_WORD,
    pad_piece=Constants.PAD_WORD,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=True,
    normalization_rule_name="identity",
)

print("hello")
