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
import lib.recurrent as recurrent


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
            self.model.encoder.load_state_dict(enc['model'])
            # load optimiser
            self.setup_optimiser(enc['optim'])
        if decoder_path:
            dec = torch.load(decoder_path)
            # Note that the decoder file contains both the decoder
            # and the target_word_projection.
            self.model.decoder.load_state_dict(dec['model'])
            self.model.generator.load_state_dict(dec['generator'])
        self.model.to(self.device) 

        return self

    def initiate(self):
        """
        Setups seq2seq model and stores it into memory.
        """
    
        encoder = recurrent.Encoder(opt, self.opt.src_vocab_size)
        decoder = recurrent.Decoder(opt, self.opt.tgt_vocab_size)

        generator = nn.Sequential(
            nn.Linear(self.opt.rnn_size, self.opt.tgt_vocab_size),
            nn.LogSoftmax()
            )

        self.model = NMTModel(encoder, decoder)
        self.model.generator = generator

        return self

    def setup_optimiser(self, opt=None):
        if not opt:
            opt = self.opt
        # based on the opt.
        self.optimiser = recurrent.Optim(
            opt.optim,
            opt.learning_rate,
            opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
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

    def translate(self):
        """
        Uses the models to perform inference/translation.
        """
        print("[Warning]: translate() is not implemented.")
        return self
    
    def save(self, epoch=None, note=None):
        """
        save model weights and parameters to file.
        """
        print("[Warning]: save() is not implemented.")

        checkpoint_encoder = {
            'type': "recurrent",
            'model': self.model.encoder.state_dict(),
            'epoch': epoch,
            'optim': self.optimiser,
            'settings': self.opt
        }

        checkpoint_decoder = {
            'type': "recurrent",
            'model': self.model.encoder.state_dict(),
            'generator': self.model.generator.state_dict(),
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


    # def NMTCriterion(self,vocabSize):
    #     """
    #     Deals with criterion for each GPU (which you'll need to sort out.)
    #     """
    #     weight = torch.ones(vocabSize)
    #     weight[onmt.Constants.PAD] = 0
    #     crit = nn.NLLLoss(weight, size_average=False)
    #     if self.opt.gpus:
    #         crit.cuda()
    #     return crit

    @staticmethod
    def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
        # compute generations one piece at a time
        num_correct, loss = 0, 0
        outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

        batch_size = outputs.size(1)
        outputs_split = torch.split(outputs, opt.max_generator_batches)
        targets_split = torch.split(targets, opt.max_generator_batches)
        for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
            out_t = out_t.view(-1, out_t.size(2))
            scores_t = generator(out_t)
            loss_t = crit(scores_t, targ_t.view(-1))
            pred_t = scores_t.max(1)[1]
            num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(Constants.PAD).data).sum()
            num_correct += num_correct_t
            loss += loss_t.data[0]
            if not eval:
                loss_t.div(batch_size).backward()

        grad_output = None if outputs.grad is None else outputs.grad.data
        return loss, grad_output, num_correct

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

            # need to exclude <s> from targets.
            targets = tgt_seq[:, 1:]

            if not validation:
                self.optimiser.zero_grad()

            outputs = model((src_seq, src_pos), (tgt_seq, tgt_pos))
            # compute loss function
            loss, grads, n_correct = self.memoryEfficientLoss(outputs, targets, self.model.generator, nn.NLLLoss())

            if not validation:
                # backprop
                outputs.backward(grads)
                # update parameters
                self.optimiser.step()

            total_loss += loss.item()
            n_word_total += targets.ne(self.constants.PAD).sum().item()
            n_word_correct += n_correct
    
        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total

        return loss_per_word, accuracy