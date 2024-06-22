#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from collections import namedtuple
import fileinput

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders
from tqdm import tqdm
import inspect
from core.utils import get_len


# In[2]:


Batch = namedtuple("Batch", "ids src_tokens src_lengths")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch["id"],
            src_tokens=batch["net_input"]["src_tokens"],
            src_lengths=batch["net_input"]["src_lengths"],
        )


# In[3]:


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not args.max_sentences or args.max_sentences <= args.buffer_size
    ), "--max-sentences/--batch-size cannot be larger than --buffer-size"

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print("| loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(":"),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    #     if args.buffer_size > 1:
    #         print('| Sentence buffer size:', args.buffer_size)
    #     print('| Type the input sentence and press return:')
    start_id = 0
    i = 0
    with open("sacrebleu_fr.txt", "w") as writer:
        for inputs in tqdm(
            buffered_read(args.input, args.buffer_size),
            total=int(get_len(args.input) / args.buffer_size),
        ):
            results = []
            for batch in make_batches(inputs, args, task, max_positions, encode_fn):
                src_tokens = batch.src_tokens
                src_lengths = batch.src_lengths
                if use_cuda:
                    src_tokens = src_tokens.cuda()
                    src_lengths = src_lengths.cuda()

                sample = {
                    "net_input": {
                        "src_tokens": src_tokens,
                        "src_lengths": src_lengths,
                    },
                }
                translations = task.inference_step(generator, models, sample)
                for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                    src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                    results.append((start_id + id, src_tokens_i, hypos))

            # sort output to match input order

            for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                #                 print('S-{}\t{}'.format(id, src_str))

                # Process top predictions
                for hypo in hypos[: min(len(hypos), args.nbest)]:
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=src_str,
                        alignment=hypo["alignment"].int().cpu()
                        if hypo["alignment"] is not None
                        else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )
                    hypo_str = decode_fn(hypo_str)
                    writer.write(hypo_str + "\n")
                    i += 1
                    if i % 1000 == 0:
                        torch.cuda.empty_cache()

        # update running id counter
        start_id += len(inputs)


# In[4]:


def parse_args_and_arch(parser, input_args=None, parse_known=False):
    parser.add_argument("-f")
    #     parser.add_argument("data")
    from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY

    # The parser doesn't know about model/criterion/optimizer-specific args, so
    # we parse twice. First we parse the model/criterion/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser.
    if hasattr(args, "arch"):
        model_specific_group = parser.add_argument_group(
            "Model-specific configuration",
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group)

    # Add *-specific args to parser.
    from fairseq.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        choice = getattr(args, registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            if hasattr(cls, "add_args"):
                cls.add_args(parser)
    if hasattr(args, "task"):
        from fairseq.tasks import TASK_REGISTRY

        TASK_REGISTRY[args.task].add_args(parser)
    if getattr(args, "use_bmuf", False):
        # hack to support extra args for block distributed data parallelism
        from fairseq.optim.bmuf import FairseqBMUF

        FairseqBMUF.add_args(parser)

    # Parse a second time.
    if parse_known:
        args, extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None

    # Post-process args.
    if hasattr(args, "max_sentences_valid") and args.max_sentences_valid is None:
        args.max_sentences_valid = args.max_sentences
    if hasattr(args, "max_tokens_valid") and args.max_tokens_valid is None:
        args.max_tokens_valid = args.max_tokens
    if getattr(args, "memory_efficient_fp16", False):
        args.fp16 = True

    # Apply architecture configuration.
    if hasattr(args, "arch"):
        ARCH_CONFIG_REGISTRY[args.arch](args)

    if parse_known:
        return args, extra
    else:
        return args


# In[ ]:


pathsrc = "models/wmt14.en-fr.joined-dict.transformer"
inp = "../datasets/newspapers/popular.atok.part2.en"
bsize = 50
input_args = [
    "--path",
    pathsrc + "/model.pt",
    "--max-len-a",
    "150",
    "--max-len-b",
    "150",
    #             "data", pathsrc,
    "--beam",
    "5",
    "--source-lang",
    "en",
    "--target-lang",
    "fr",
    "--tokenizer",
    "moses",
    "--bpe",
    "subword_nmt",
    "--bpe-codes",
    pathsrc + "/bpecodes",
    "--batch-size",
    str(bsize),
    "--buffer-size",
    str(bsize),
    "--remove-bpe",
    "--input",
    inp,
    "--quiet",
    pathsrc,
]


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = parse_args_and_arch(parser, input_args=input_args)
    main(args)


if __name__ == "__main__":
    cli_main()


# In[ ]:
