import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from tqdm import tqdm
import time
import math
import os

from lib.nmtModel import NMTModel
from lib.transformer.Models import Transformer
from lib.transformer.Optim import ScheduledOptim
from lib.transformer.Translator import Translator
"""
Wrapper class for Transformer.

This does not necessarily contain the transformer
neural architecture code, but contains code for
training, saving and so on.
"""

# model is imported from code in parent directory
class TransformerModel(NMTModel):
    def __init__(self, opt):
        """
        initiate() loads the model into memory,
        based on parameters from self.opt.

        opt: parser.parse_args() variable output.
             It'll be a class list type.
        """
        # init will store opt into the object.
        super().__init__(opt)
        
        # variable is tripped once a model is requested to save.
        self.save_trip = False

    def initiate(self):
        """
        Setups transformer model and stores it into memory.
        """
        if self.opt.checkpoint_encoder:
            self.load(self.opt.checkpoint_encoder)
        else:
            # start fresh.
            self.model = Transformer(
                self.opt.src_vocab_size,
                self.opt.tgt_vocab_size,
                self.opt.max_token_seq_len,
                tgt_emb_prj_weight_sharing=self.opt.proj_share_weight,
                emb_src_tgt_weight_sharing=self.opt.embs_share_weight,
                d_k=self.opt.d_k,
                d_v=self.opt.d_v,
                d_model=self.opt.d_model,
                d_word_vec=self.opt.d_word_vec,
                d_inner=self.opt.d_inner_hid,
                n_layers=self.opt.layers,
                n_head=self.opt.n_head,
                dropout=self.opt.dropout).to(self.device)
    
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
        self.model.to(self.device) 
    
    def setup_optimiser(self):
        """
        Setups gradient optimiser mechanism. We default to Adam.
        """
        self.optimiser = ScheduledOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            self.opt.d_model, self.opt.n_warmup_steps)
        print("[Info] optimiser configured.")

    def train(self, epoch, evaluate=True):
        """
        Trains model against some data.
        This represents one round of epoch training.

        It's a wrapper function that calls self.compute_epoch();
        this function comes with additional stat management.

        params:
        epoch: epoch round (int)
        evaluate: boolean flag to determine whether to run model on
                  validation data.
        """

        # training data
        start = time.time()
        train_stats = self.compute_epoch(self.training_data, False)
        train_loss, train_acc = train_stats

        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)

        print('  - (Training)   perplexity: {perplexity: 8.5f}, accuracy: {accu:3.3f} %, '\
            'elapse: {elapse:3.3f} min'.format(
                perplexity=math.exp(min(train_loss, 100)), accu=100*train_acc,
                elapse=(time.time()-start)/60))

        if evaluate:
            # validation data
            with torch.no_grad():
                valid_stats = self.compute_epoch(self.validation_data, True)
            valid_loss, valid_acc = valid_stats

            self.valid_losses.append(valid_loss)
            self.valid_accs.append(valid_acc)

            print('  - (Validation) perplexity: {perplexity: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    perplexity=math.exp(min(valid_loss, 100)), accu=100*valid_acc,
                    elapse=(time.time()-start)/60))

        return self

    def translate(self, test_loader):
        """
        Batch translates sequences.

        Assumes test_data is a DataLoader.
        """
        self.model.word_prob_prj = nn.LogSoftmax(dim=1)
        self.model.eval()
        translator = Translator(opt, new=False)
        translator.model = self.model
        idx2word = test_loader.dataset.tgt_idx2word

        with open(opt.output, 'w') as f:
            for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
                # get sequences through beam search.
                all_hyp, _ = translator.translate_batch(*batch)
                # save outputs
                for idx_seqs in all_hyp:
                    for idx_seq in idx_seqs:
                        pred_line = ' '.join([idx2word[idx] for idx in idx_seq])
                        f.write(pred_line + '\n')
        print('[Info] Finished.')
        return sequences

    def save(self, epoch=None, note=None):
        """
        Saves model components into file.
        """
        
        checkpoint_encoder = {
        'type': "transformer",
        'model': self.model.encoder.state_dict(),
        'epoch' : epoch,
        'settings': self.opt
        }

        checkpoint_decoder = {
        'type': "transformer",
        'model': self.model.decoder.state_dict(),
        'generator' : self.model.generator.state_dict(),
        'epoch' : epoch,
        'settings': self.opt
        }

        if not note:
            note = ""

        # make sure a path is specified prior to saving the files.
        if self.opt.save_model:
            ready_to_save = False
            if self.opt.save_mode == "all":
                model_name = note + '_accu_{accu:3.3f}.chkpt'.format(accu=100*self.valid_accs[-1])
                ready_to_save = True
            else:
                # assumes self.opt.save_mode = "best"
                if self.valid_accs[-1] >= max(self.valid_accs):
                    model_name = note + ".chkpt"
                    ready_to_save = True
                    print('    - [Info] The checkpoint file has been updated.')
            if ready_to_save:
                encoder_name = "encoder_" + model_name
                decoder_name = "decoder_" + model_name
                # setup directory to save this at.
                encoder_filepath = os.path.join(self.opt.directory, encoder_name)
                decoder_filepath = os.path.join(self.opt.directory, decoder_name)
                torch.save(checkpoint_encoder, encoder_filepath)
                torch.save(checkpoint_decoder, decoder_filepath)
        else:
            if not self.save_trip:
                print("    - [Warning]: the model is not specified to save.")
                self.save_trip = True
    
    # ---------------------------
    # Below the line represents transformer specific code.
    # ---------------------------

    def performance(self, pred, gold, smoothing=False):
        """
        Calculates token level accuracy.
        Smoothing can be applied if needed.
        """
        loss = self.calculate_loss(pred, gold, smoothing)
        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(self.constants.PAD)
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()
        return loss, n_correct

    def calculate_loss(self, pred, gold, smoothing=False):
        """
        Computes cross entropy loss,
        apply label smoothing if needed.
        """
        gold = gold.contiguous().view(-1)
        if smoothing:
            epsilon = 0.1
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - epsilon) + (1 - one_hot) * epsilon / (n_class - 1)

            log_prb = F.log_softmax(pred, dim=1)
            # create non-padding mask with torch.ne()
            non_pad_mask = gold.ne(self.constants.PAD)
            loss = -(one_hot * log_prb).sum(dim=1)
            # losses are averaged later
            loss = loss.masked_select(non_pad_mask).sum()
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=self.constants.PAD, reduction='sum')
        return loss

    def compute_epoch(self, dataset, validation=False):
        """
        Performs forward pass on some data.
        """

        if validation:
            self.model.eval()
        else:
            self.model.train()

        total_loss, n_word_total, n_word_correct = 0,0,0

        for batch in tqdm(dataset, desc=' - Training', leave=False):
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(self.device), batch)
            gold = tgt_seq[:, 1:]
            
            if not validation:
                self.optimiser.zero_grad()
            # compute forward propagation
            pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos)
            # compute performance
            loss, n_correct = self.performance(pred, gold, smoothing=self.opt.label_smoothing)

            if not validation:
                # backwards propagation
                loss.backward()
                # update parameters
                self.optimiser.step_and_update_lr()

            # bartending outputs.
            total_loss += loss.item()
            n_word_total += gold.ne(self.constants.PAD).sum().item()
            n_word_correct += n_correct
    

        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total
        return loss_per_word, accuracy