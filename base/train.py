"""
Deals with training the models.

train.py will load some datasets, and will produce some results and save the encoder and decoders seperately.
"""

import argparse
from tqdm import tqdm
from lib.RecurrentModel import RecurrentModel as recurrent
from lib.TransformerModel import TransformerModel as transformer

def load_args():
    parser = argparse.ArgumentParser(description="train.py")
    # data options
    parser.add_argument('-data', required=True,
                        help='path to the *.pt file that was computed through preprocess.py')

    # parser.add_argument('-model_name', default="model",
    #                     help="""
    #                     Model filename (the model will be saved as <model_name>_epochN_PPL.pt where PPL is the validation perplexity.
    #                     """)

    parser.add_argument('-checkpoint_encoder', default="", type=str,
                        help="""
                        If training from a checkpoint, then this is the path to the pretrained encoder.
                        """)

    parser.add_argument('-checkpoint_decoder', default="", type=str,
                        help="""
                        If training from a checkpoint, then this is the path to the pretrained model.
                        """)
    
    parser.add_argument('-log', action='store_true',
                        help="""
                        Determines whether to enable logs, which will save status into text files.
                        """)

    parser.add_argument('-save_model', action='store_true',
                        help="""
                        Determines whether to save the model or not.
                        """)

    parser.add_argument('-save_mode', default='all', choices=['all', 'best'],
                        help="""
                        Determines whether to save all versions of the model or keep the best version.
                        """)

    # model options.
    parser.add_argument('-model', choices=['transformer', 'recurrent'], 
                        required=True, help="""
                        Either a recurrent (seq2seq model) or a transformer.
                        """)

    parser.add_argument('-cuda', action='store_true',
                        help="""
                        Determines whether to use CUDA or not. (You should.)
                        """)

    parser.add_argument('-batch_size', type=int, default=64, help="""
                        Determines batch size of input data, for feeding into the models.
                        """)

    parser.add_argument('-layers', type=int, default=6, help="""
                        Number of layers for the model. (Recommended to have 6 for Transformer, 2 for recurrent.)
                        """)

    parser.add_argument("-d_word_vec", type=int, default=300, help="""
                        Dimension size of the token vectors representing words (or characters, or bytes).
                        """)

    # training options
    parser.add_argument('-epochs', type=int, required=True, default=10, help="""
                        Number of epochs for training. (Note that for transformers, the number of sequences become considerably longer.)
                        """)
    parser.add_argument('-dropout', type=float, default=0.1, help="""
                        Dropout probability' applied between self-attention layers/RNN Stacks.
                        """)

    # debugging options
    parser.add_argument('-telegram_key', help="""
                        filepath to telegram API private key to send messages to.
                        """)

    # transformer specific options
    parser.add_argument('-d_model', type=int, default=512, help="""
                        Dimension size of the model.
                        """)
    parser.add_argument('-d_inner_hid', type=int, default=2048, help="""
                        Dimension size of the hidden layers of the transformer.
                        """)
    parser.add_argument('-d_k', type=int, default=64, help="Key vector dimension size. ")
    parser.add_argument('-d_v', type=int, default=64, help="Value vector dimension size.")

    parser.add_argument('-n_head', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()

    # validation.
    assert opt.epochs > 0

    return opt

def setup_model(opt):
    return transformer(opt) if opt.model == "transformer" else recurrent(opt)

def train(model):
    """
    Deals with train and validation rounds.
    """
    model.init_logs()
    for epoch in tqdm(range(1, model.opt.epochs+1), desc='Epochs'):
        stats = model.train(epoch)
        model.save(epoch=epoch, note="epoch_" + str(epoch))
        model.update_logs(epoch)

if __name__ == "__main__":
    model = setup_model(load_args())
    print("Setup model wrapper.")
    model.load_dataset()
    print("Loaded data.")
    model.initiate()
    print("Initiated model and weights.")
    model.setup_optimiser()
    print("Training model.")
    train(model)
    print("Done.")