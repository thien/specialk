import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from tqdm import tqdm
import time
import math
import os

import core.constants as Constants
from lib.nmtModel import NMTModel
from lib.recurrent.Models import Encoder, Decoder
from lib.recurrent.Models import NMTModel as Seq2Seq
from lib.recurrent.Optim import Optim
from lib.recurrent.Translator import Translator

class RecurrentModel(NMTModel):
    def __init__(self, opt):
        """
        initiate() loads the model into memory,
        based on parameters from self.opt.

        opt: parser.parse_args() variable output.
             It'll be a class list type.
        """
        super().__init__(opt)
        # variable is tripped once a model is requested to save.
        self.save_trip = False
    
    def load(self, encoder_path, decoder_path=None):
        """
        Loads models from file.
        """
        if encoder_path:
            enc = torch.load(encoder_path)
            # copy encoder weights
            opts_e = enc['settings']
            # replace parameters in opts.
            blacklist = {
                "checkpoint_encoder",
                "checkpoint_decoder",
                "cuda",
                "directory",
                "data",
                "log",
                "model",
                "save_mode",
                "telegram_key",
                "save_model",
                "train_from_state_dict",
                "batch_size"
            }
            for arg in dir(opts_e):
                if arg[0] == "_":
                    continue
                if arg in blacklist:
                    continue
                setattr(self.opt, arg, getattr(opts_e, arg))
            # initiate a new model
            self.initiate()
            self.model.encoder.load_state_dict(enc['model'])

        if decoder_path:
            dec = torch.load(decoder_path)
            opts_d = enc['settings']
            # Note that the decoder file contains both the decoder
            # and the target_word_projection.
            self.model.decoder.load_state_dict(dec['model'])
            # self.model.generator.load_state_dict(dec['generator'])
        self.model.to(self.device) 

        return self

    def initiate(self):
        """
        Setups seq2seq model and stores it into memory.
        """
    
        encoder = Encoder(self.opt, self.opt.src_vocab_size).to(self.device)
        decoder = Decoder(self.opt, self.opt.tgt_vocab_size).to(self.device)

        self.model = Seq2Seq(encoder, decoder).to(self.device)
        # self.model.decoder.generator = generator

        return self

    def setup_optimiser(self, opt=None):
        if not opt:
            opt = self.opt
        # based on the opt.
        self.optimiser = Optim(
            opt.optim,
            opt.learning_rate,
            opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
        self.optimiser.set_parameters(self.model.parameters())
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

    def translate(self, test_loader, max_token_seq_len):
        """
        Uses the models to perform inference/translation.
        """
        translator = Translator(self.opt, False)
        translator.model = self.model
        # max_token_seq_len depends on the vocabulary_loader.
        translator.max_token_seq_len = max_token_seq_len
        idx2word = test_loader.dataset.tgt_idx2word
        # setup run
        with open(self.opt.output, 'w') as f:
            for batch in tqdm(test_loader, desc='  - (Test)', leave=False):
                src_seq, src_pos, = map(lambda x: x.to(self.device), batch)
                src_pos = torch.sum(src_pos > 0, dim=1)
                
                # sort for pack_padded_sequences
                sorted_lengths, sorted_idx = torch.sort(src_pos, descending=True)
                src_seq = src_seq[sorted_idx]
                # swap batch relationship order.
                src_seq = src_seq.transpose(0, 1)

                src = (src_seq, sorted_lengths)

                pred, score, attn, _ = translator.translateBatch(src, None)

                # reverse tensor relationship order
                pred = pred.transpose(0, 1)
                # reverse order
                _, reversed_idx = torch.sort(sorted_idx)
                pred  = pred[reversed_idx]
                score = score[reversed_idx]
                attn  = attn[reversed_idx]

                pred, predScore, attn, goldScore = list(zip(*sorted(zip(pred, predScore, attn, goldScore, indices), key=lambda x: x[-1])))[:-1]

                #  (3) convert indexes to words
                predBatch = []
                for b in range(src[0].size(1)):
                    predBatch.append(
                        [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                                for n in range(self.opt.n_best)]
                    )

                print("COOL")
                break
        return self
    
    def save(self, epoch=None, note=None):
        """
        save model weights and parameters to file.
        """

        checkpoint_encoder = {
            'type': "recurrent",
            'model': self.model.encoder.state_dict(),
            'epoch': epoch,
            'optim': self.optimiser,
            'settings': self.opt
        }

        checkpoint_decoder = {
            'type': "recurrent",
            'model': self.model.decoder.state_dict(),
            'epoch': epoch,
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
        # save the optimiser (hmm)
        return self

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
        Performs forward pass on batches of data.
        """

        if validation:
            self.model.eval()
        else:
            self.model.train()

        total_loss, n_word_total, n_word_correct = 0, 0, 0

        label = "Training" if not validation else "Validation"
        for batch in tqdm(dataset, desc=' - '+label, leave=False):
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(self.device), batch)
            src_pos = torch.sum(src_pos > 0, dim=1)
            tgt_pos = torch.sum(tgt_pos > 0, dim=1)

            if not validation:
                self.model.zero_grad()

            pred = self.model((src_seq, src_pos), (tgt_seq, tgt_pos))
            pred = pred.view(-1, pred.size(2))
   
            loss, n_correct = self.performance(pred, tgt_seq, smoothing=self.opt.label_smoothing)

            if not validation:
                # backprop
                loss.backward()
                # outputs.backward(grads)
                # update parameters
                self.optimiser.step()

            total_loss += loss.item()
            n_word_total += tgt_seq.ne(self.constants.PAD).sum().item()
            n_word_correct += n_correct
    
        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total

        return loss_per_word, accuracy