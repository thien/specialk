"""
Takes an NMT encoder and a style contingent decoder
to perform style transfer. It can also be used for
general machine translation. (i.e. it is contingent
on the encoder and decoders used.)
"""

import argparse
from tqdm import tqdm
from lib.TransformerModel import TransformerModel as transformer
from lib.RecurrentModel import RecurrentModel as recurrent

def load_args():
    parser = argparse.ArgumentParser(description="train.py")
    # data options

    parser.add_argument('-model', choices=['transformer', 'recurrent'], 
                        required=True, help="""
                        Either a recurrent (seq2seq model) or a transformer.
                        """)

    parser.add_argument('-src', default="", required=True, type=str,
                        help="""
                        Source sequence for decoding purposes (one line per sequence).
                        """)
    
    parser.add_argument('-vocab', required=True, help="""
                        Vocabulary (refer to training data pickle.
                        """)

    parser.add_argument('-checkpoint_encoder', required=True, default=None, type=str,
                        help="""
                        If training from a checkpoint, then this is the path to the pretrained encoder.
                        """)

    parser.add_argument('-checkpoint_decoder', required=True, default=None, type=str,
                        help="""
                        If training from a checkpoint, then this is the path to the pretrained model.
                        """)
    
    parser.add_argument('-output', default='predictions.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")

    parser.add_argument('-copysrc', action='store_true', help="""
                        If enabled, saves a copy of the source sequences into the model folder.
                        """)

    parser.add_argument('-cuda', action='store_true',
                        help="""
                        Determines whether to use CUDA or not. (You should.)
                        """)

    # debugging options
    parser.add_argument('-telegram_key', help="""
                        filepath to telegram API private key to send messages to.
                        """)

    # translate option
    parser.add_argument('-batch_size', type=int, default=128, help="""
                        Determines batch size of input data, for feeding into the models.
                        """)
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')

    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences"""
                        )
    # parser.add_argument("-lowercase", action="store_true", help="""
    #                     If enabled, sets input characters as lowercase.
    #                     """)

    parser.add_argument("-verbose", action="store_true")

    # debugging options
    parser.add_argument('-telegram', type=str, default="", help="""
                        filepath to telegram API private key
                        and chatID to send messages to.
                        """)

    opt = parser.parse_args()

    opt.save_model = False

    # validation.

    return opt

import torch
if __name__ == "__main__":
    opt = load_args()
    # load encoder and decoder

    # then load them into the model.
    model = transformer(opt) if opt.model == "transformer" else recurrent(opt)
    print("Setup model wrapper.")
    model.load(opt.checkpoint_encoder, opt.checkpoint_decoder)
    print("Initiated model and weights.")
    # print(model.model)
    print(model.model.encoder.src_word_emb)
    print(model.model.decoder.tgt_word_emb)
    # print(model.model)
    # load test data
    test_loader, max_token_seq_len = model.load_testdata(opt.src, opt.vocab)
    # # idx2word = test_loader.dataset.tgt_idx2word
    # # load output file.

    # print(test_loader)
    # print(test_loader)
    
    # mins = []
    # maxs = []
    # for ent in test_loader:
    #     tok, ent_len = ent
    #     mins.append(torch.min(ent_len).item())
    #     maxs.append(torch.max(ent_len).item())
    # print(min(mins))
    # print(max(maxs))

    # print(model.model.encoder.src_word_emb(torch.tensor([0,8573])))
    model.translate(test_loader, max_token_seq_len)
    with open(opt.output, 'w') as f:
        # setup test loader.
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            print(batch)
            hypotheses, scores = model.translate_batch(*batch)
            for sequences in hypotheses:
                for sequence in sequences:
                    line = " ".join([idx2word[t] for t in sequence])
                    f.write(line + "\n")
    print("Done.")