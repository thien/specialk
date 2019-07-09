import matplotlib
matplotlib.use('Agg')

import random
import numpy as np
import multiprocessing
import argparse
from tqdm import tqdm
import torch

from train import train_model as train_final_model

from lib.RecurrentModel import RecurrentModel as recurrent
from lib.TransformerModel import TransformerModel as transformer



from GPyOpt.methods import BayesianOptimization

seed = 1337
np.random.seed(seed)
torch.manual_seed(seed)


def load_args():
    parser = argparse.ArgumentParser(description="train.py")
    # data options
    parser.add_argument('-data', required=True,
                        help='path to the *.pt file that was computed through preprocess.py')
    parser.add_argument('-log', action='store_true',
                        help="""
                        Determines whether to enable logs, which will save status into text files.
                        """)
    parser.add_argument('-save_mode', default='all', choices=['all', 'best'],
                        help="""
                        Determines whether to save all versions of the model or keep the best version.
                        """)
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
    parser.add_argument('-epochs', type=int, required=True, default=10, 
                        help="""
                        Number of epochs for training. (Note
                        that for transformers, the number of
                        sequences become considerably longer.)
                        """)

    # debugging options
    parser.add_argument('-telegram', type=str, help="""
                        filepath to telegram API private key
                        and chatID to send messages to.
                        """)
    parser.add_argument("-verbose", action="store_true", help="""
                        If enabled, prints updates in terminal.
                        """)

    # transformer specific options
    parser.add_argument('-d_model', type=int, default=512, help="""
                        Dimension size of the model.
                        """)
    parser.add_argument('-d_inner_hid', type=int, default=2048, help="""
                        Dimension size of the hidden layers of the transformer.
                        """)
    parser.add_argument('-d_k', type=int, default=64, help="""
                        Key vector dimension size.
                        """)
    parser.add_argument('-d_v', type=int, default=64, help="""
                        Value vector dimension size.
                        """)
    parser.add_argument('-n_head', type=int, default=8, help="""
                        Number of attention heads.
                        """)

    parser.add_argument('-embs_share_weight', action='store_true', help="""
                        If enabled, allows the embeddings of the encoder
                        and the decoder to share weights.
                        """)
    parser.add_argument('-proj_share_weight', action='store_true', help="""
                        If enabled, allows the projection/generator 
                        to share weights.
                        """)

    opt = parser.parse_args()

    opt.checkpoint_encoder = ""
    opt.checkpoint_decoder = ""
    opt.save_model = False
    opt.bot = False
   
    return opt


opt = load_args()

domain_transformer = [
    {
        'name': 'learning_rate',
        'type': 'continuous',
        'domain': (0.0001, 0.1)
    },
    {
        'name': 'dropout',
        'type': 'continuous',
        'domain': (0.0, 0.6)
    },
    {
        'name': 'n_warmup_steps',
        'type': 'discrete',
        'domain': range(2000, 6000, 1000)
    },
    {
        'name': 'label_smoothing',
        'type': 'discrete',
        'domain': [0, 1]
    }
]


def allocate_var(v):
    """
    Changes hyperparameter variables (which are presently
    set as an ordered numpy array) to a python dict.
    """
    d = {}
    if len(v) == 1:
        v = v[0]
    for i in range(len(v)):
        value = v[i]
        name = domain_transformer[i]['name']
        if domain_transformer[i]['type'] == "discrete":
            value = int(value)
        d[name] = value
    return d


model = transformer(opt)
model.load_dataset()


def fit(f):
    """optimisation function"""

    # substitute hyperparameter values
    hyperparams = allocate_var(f)
    for key in hyperparams:
        setattr(opt, key, hyperparams[key])
    model.opt = opt

    # initiate model
    model.reset_metrics()
    model.initiate()
    model.setup_optimiser()

    for epoch in tqdm(range(1, opt.epochs+1), desc='Epochs'):
        stats = model.train(epoch)
 
            
    """
    Accuracy metric is easier to interpret. It however isn’t 
    differentiable so it can’t be used for back-propagation 
    by the learning algorithm. For training models themselves,
    we'd need a differentiable loss function to act as a good
    proxy for accuracy.

    However since we're only looking at the end outcome after
    training, we can optimise for accuracy for the
    hyperparameter search.
    """

    return model.valid_accs[-1]


myBopt = BayesianOptimization(
    f=fit,                   # function to optimize
    domain=domain_transformer,            # box-constrains of the problem
    initial_design_numdata=5,    # number data initial design
    model_type="GP_MCMC",
    acquisition_type='EI_MCMC',  # EI
    evaluator_type="predictive",  # Expected Improvement
    batch_size=1,
    num_cores=8,
    maximize=True,
    exact_feval=True)         # May not always give exact results

myBopt.run_optimization(max_iter=2)
# get the best parameters out
x_best = myBopt.x_opt  # myBopt.X[np.argmin(myBopt.Y)]

print(allocate_var(x_best))
# myBopt.plot_convergence()
# myBopt.plot_acquisition()


# save opt output notes

print("TRAINING BEST MODEL:")
opt.save_model = True
opt.log = True

del model

# substitute hyperparameter values
hyperparams = allocate_var(x_best)
for key in hyperparams:
    setattr(opt, key, hyperparams[key])

train_final_model(opt)