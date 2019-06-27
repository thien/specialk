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
        if decoder_path:
            dec = torch.load(decoder_path)
            # Note that the decoder file contains both the decoder and the 
            # target_word_projection.
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
            nn.Linear(opt.rnn_size, self.opt.tgt_vocab_size),
            nn.LogSoftmax()
            )

        self.model = onmt.Models.NMTModel(encoder, decoder)
        self.model.generator = generator

        return self

    def setup_optimiser(self):
        # based on the opt.
        self.optimiser = recurrent.Optim(
            self.opt.optim,
            self.opt.learning_rate,
            self.opt.max_grad_norm,
            lr_decay=self.opt.learning_rate_decay,
            start_decay_at=self.opt.start_decay_at
        )
        print("[Info] optimiser configured.")

    def train(self):
        """
        Trains models.
        """
        print("[Warning]: train() is not implemented.")
        return self

    def validate(self, val_data):
        """
        Validates model performance against validation data.
        """
        print("[Warning:] validate() is not implemented.")

        total_loss = 0
        total_words = 0
        total_num_correct = 0

        self.model.eval()
        for i in range(len(data)):
            batch = data[i][:-1] # exclude original indices
            outputs = self.model(batch)
            targets = batch[1][1:]  # exclude <s> from targets
            loss, _, num_correct = self.memoryEfficientLoss(
                    outputs, targets, self.model.generator, criterion, eval=True)
            total_loss += loss
            total_num_correct += num_correct
            total_words += targets.data.ne(onmt.Constants.PAD).sum()

        return total_loss / total_words, total_num_correct / total_words

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
            'epoch' : epoch,
            'settings': self.opt
        }

        checkpoint_decoder = {
            'type': "recurrent",
            'model': self.model.encoder.state_dict(),
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
        # save the optimiser (hmm)
        return self

    
    # ---------------------------
    # Below the line represents transformer specific code.
    # ---------------------------

    @staticmethod
    def NMTCriterion(vocabSize):
        """
        Deals with criterion for each GPU (which you'll need to sort out.)
        """
        weight = torch.ones(vocabSize)
        weight[onmt.Constants.PAD] = 0
        crit = nn.NLLLoss(weight, size_average=False)
        if opt.gpus:
            crit.cuda()
        return crit

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
            num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(onmt.Constants.PAD).data).sum()
            num_correct += num_correct_t
            loss += loss_t.data[0]
            if not eval:
                loss_t.div(batch_size).backward()

        grad_output = None if outputs.grad is None else outputs.grad.data
        return loss, grad_output, num_correct