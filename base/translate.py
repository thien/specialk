"""
Takes an NMT encoder and a style contingent decoder
to perform style transfer.
"""

"""
Deals with training the models.

train.py will load some datasets, and will produce some results and save the encoder and decoders seperately.
"""

import argparse
from tqdm import tqdm
# import lib.RecurrentModel as recurrent
from lib.TransformerModel import TransformerModel as transformer

def load_args():
    parser = argparse.ArgumentParser(description="train.py")
    # data options

    parser.add_argument('-src', default="", required=True, type=str,
                        help="""
                        Source sequence for decoding purposes (one line per sequence).
                        """)
    
    parser.add_argument('-vocab', required=True, help="""
                        Vocabulary (refer to training data pickle.
                        """)

    parser.add_argument('-checkpoint_encoder', default="", required=True, type=str,
                        help="""
                        If training from a checkpoint, then this is the path to the pretrained encoder.
                        """)

    parser.add_argument('-checkpoint_decoder', default="", required=True, type=str,
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

    parser.add_argument('-batch_size', type=int, default=128, help="""
                        Determines batch size of input data, for feeding into the models.
                        """)

    # debugging options
    parser.add_argument('-telegram_key', help="""
                        filepath to telegram API private key to send messages to.
                        """)

    # translate option
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')

    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences"""
                        )

    opt = parser.parse_args()

    # validation.

    return opt


if __name__ == "__main__":
    opt = load_args()
    # load model encoder and decoder
    enc = torch.load(opt.checkpoint_encoder)
    dec = torch.load(opt.checkpoint_decoder)
    if enc['type'] != dec['type']:
        throw Exception("The encoder and decoder model components don't belong to the same group.")
    if enc['type'] == "transformer":
        model = transformer(enc['settings'])
        model.initiate()
        
        # overwrite model.opt properties with what is currently loaded
    # initiate model
    # load src
    # load vocabulary
    # translate
    # save responses
    model.load_dataset()
    print("Loaded data.")
    model.initiate()
    print("Initiated model and weights.")
    model.setup_optimiser()
    print("Training model.")
    train(model)
    print("Done.")