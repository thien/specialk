import sentencepiece as spm
import specialk.core.constants as Constants

special_tokens = ["<blank>", "<unk>", "<s>", "</s>", "<p>", "<sep>", "<cls>"]
special_tokens = [i for i in special_tokens if i not in {"<unk>"}]
special_tokens

inp = "/Users/t/Projects/specialk/datasets/machine_translation/corpus_enfr_final.en,/Users/t/Projects/specialk/datasets/machine_translation/corpus_enfr_final.fr"

spm.SentencePieceTrainer.train(
    input=inp,
    model_prefix="frenidentity",
    vocab_size=35000,
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
    train_extremely_large_corpus=True
    normalization_rule_name="identity"
)

print("hello")