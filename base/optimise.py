# import argparse
# from tqdm import tqdm
# from lib.RecurrentModel import RecurrentModel as recurrent
# from lib.TransformerModel import TransformerModel as transformer

# seed = 1337
# np.random.seed(seed)
# torch.manual_seed(seed)

# # -d_model', type=int, default=512, help="""
# #                         Dimension size of the model.
# #                         """)
# #     parser.add_argument('-d_inner_hid', type=int, default=2048, help="""
# #                         Dimension size of the hidden layers of the transformer.
# #                         """)
# #     parser.add_argument('-d_k', type=int, default=64, help="""
# #                         Key vector dimension size.
# #                         """)
# #     parser.add_argument('-d_v', type=int, default=64, help="""
# #                         Value vector dimension size.
# #                         """)
# #     parser.add_argument('-n_head', type=int, default=8, help="""
# #                         Number of attention heads.
# #                         """)
# #     parser.add_argument('-n_warmup_steps', type=int, default=4000, help="""
# #                         Number of warmup steps.
# #                         """)
# #     parser.add_argument('-embs_share_weight', action='store_true', help="""
# #                         If enabled, allows the embeddings of the encoder
# #                         and the decoder to share weights.
# #                         """)
# #     parser.add_argument('-proj_share_weight', action='store_true', help="""
# #                         If enabled, allows the projection/generator 
# #                         to share weights.
# #                         """)
# #     parser.add_argument('-label_smoothing', action='store_true', help="""
# #                         Enables label smoothing.
# #                         """)

# class Hyper:
#     def __init__(self):
#         self.opt = opt
#         self.model = None
        
#     def tf_blackbox(self,
#                     d_model,
#                     d_inner_hid,
#                     d_k,
#                     d_v,
#                     n_head,
#                     n_warmup_steps,
#                     embs_share_weight,
#                     proj_share_weight,
#                     label_smoothing):
#         # since the hyperparameters are different
#         # they'll need their own function.

#         # initiate new instance of model
#         if self.model:
#             del self.model
        
#         self.model = 

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

train_X = torch.rand(10, 2)
Y = 1 - torch.norm(train_X - 0.5, dim=-1) + 0.1 * torch.rand(10)
train_Y = (Y - Y.mean()) / Y.std()

gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# create acquisition function

from botorch.acquisition import UpperConfidenceBound

UCB = UpperConfidenceBound(gp, beta=0.1)

# deal with optimiser

from botorch.optim import joint_optimize

bounds = torch.stack([torch.zeros(2), torch.ones(2)])
candidate = joint_optimize(
    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)
candidate  # tensor([0.4887, 0.5063])

print(candidate)