from model import Model
# from lib.TransformerModel import TransformerModel
import argparse

parser = argparse.ArgumentParser(description="train.py")
parser.add_argument('-cuda', default=True, type=bool, 
                    help='determines whether to use CUDA or not.')
parser.add_argument('-model', default="transformer",
                        help="""
                        Either a recurrent (seq2seq model) or a transformer.
                        """)
opt = parser.parse_args()
# print("loaded")

k = Model(opt)
