import torch
import core.constants as Constants
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


"""
Wrapper class for Transformer.

This does not necessarily contain the transformer
neural architecture code, but contains code for
training, saving and so on.
"""

# model is imported from code in parent directory
class Transformer(Model):
    def __init__(self, opt):
        """
        initiate() loads the model into memory,
        based on parameters from opt.

        opt: parser.parse_args() variable output.
             It'll be a class list type.
        """
        # init will store opt into the object.
        super().__init__(opt)
        if opt.checkpoint_encoder:
            self.load(opt.checkpoint_encoder)
        else:
            self.initiate(opt)

    def initiate(self, opt):
        """
        Setups transformer model and stores it into memory.
        """
        self.model = Transformer(
            opt.src_vocab_size,
            opt.tgt_vocab_size,
            opt.max_token_seq_len,
            tgt_emb_prj_weight_sharing=opt.proj_share_weight,
            emb_src_tgt_weight_sharing=opt.embs_share_weight,
            d_k=opt.d_k,
            d_v=opt.d_v,
            d_model=opt.d_model,
            d_word_vec=opt.d_word_vec,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            dropout=opt.dropout).to(self.device)
        )

    def load(self, encoder_path, decoder_path=None):
        """
        Loads the model encoder and decoders from file.
        """
        if encoder_path:
            enc = torch.load(encoder_path)
            self.model.encoder.load_state_dict(enc['model'])
        if decoder_path:
            dec = torch.load(decoder_path)
            # Note that the decoder file contains both the decoder and the 
            # target_word_projection.
            self.model.decoder.load_state_dict(dec['model'])
            self.model.generator.load_state_dict(dec['generator'])
    
    def setup_optimiser(self):
        """
        Setups gradient optimiser mechanism. We default to Adam.
        """
        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            self.opt.d_model, self.opt.n_warmup_steps)
        print("[Info] optimiser configured.")

    def train(self, train_data):
        """
        Trains model against some data.

        params:
        train_data: models representing training_data.
        """
        self.model.train()

        return self

    def validate(self, val_data):
        """
        Tests model based on validation data.
        """
        return None

    def translate(self, test_data):
        """
        Batch translates sequences.
        """
        return sequences

    def save(self, extra_info, save_path):
        """
        Saves model components into file.
        """
        
        checkpoint_encoder = {
        'type': "transformer",
        'model': self.model.encoder.state_dict(),
        'settings': self.opt,
        'extras': extra_info
        }

        checkpoint_decoder = {
        'type': "transformer",
        'model': self.model.decoder.state_dict(),
        'generator' : self.model.generator.state_dict(),
        'settings': self.opt,
        'extras': extra_info
        }


    # ---------------------------
    # Below the line represents transformer specific code.
    # ---------------------------
