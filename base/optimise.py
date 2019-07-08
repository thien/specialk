import argparse
from tqdm import tqdm
from lib.RecurrentModel import RecurrentModel as recurrent
from lib.TransformerModel import TransformerModel as transformer

seed = 1337
np.random.seed(seed)
torch.manual_seed(seed)

# -d_model', type=int, default=512, help="""
#                         Dimension size of the model.
#                         """)
#     parser.add_argument('-d_inner_hid', type=int, default=2048, help="""
#                         Dimension size of the hidden layers of the transformer.
#                         """)
#     parser.add_argument('-d_k', type=int, default=64, help="""
#                         Key vector dimension size.
#                         """)
#     parser.add_argument('-d_v', type=int, default=64, help="""
#                         Value vector dimension size.
#                         """)
#     parser.add_argument('-n_head', type=int, default=8, help="""
#                         Number of attention heads.
#                         """)
#     parser.add_argument('-n_warmup_steps', type=int, default=4000, help="""
#                         Number of warmup steps.
#                         """)
#     parser.add_argument('-embs_share_weight', action='store_true', help="""
#                         If enabled, allows the embeddings of the encoder
#                         and the decoder to share weights.
#                         """)
#     parser.add_argument('-proj_share_weight', action='store_true', help="""
#                         If enabled, allows the projection/generator 
#                         to share weights.
#                         """)
#     parser.add_argument('-label_smoothing', action='store_true', help="""
#                         Enables label smoothing.
#                         """)
def tf_blackbox(d_model,
                d_inner_hid,
                d_k,
                d_v,
                n_head,
                n_warmup_steps,
                embs_share_weight,
                proj_share_weight,
                label_smoothing):
                