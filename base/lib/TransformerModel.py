import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import time
from tqdm import tqdm

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

"""
Wrapper class for Transformer.

This does not necessarily contain the transformer
neural architecture code, but contains code for
training, saving and so on.
"""

# model is imported from code in parent directory
class TransformerModel(Model):
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
        
        # variable is tripped once a model is requested to save.
        self.save_trip = False

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
        self.optimiser = ScheduledOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            self.opt.d_model, self.opt.n_warmup_steps)
        print("[Info] optimiser configured.")

    def train(self, epoch):
        """
        Trains model against some data.
        This represents one round of epoch training.

        It's a wrapper function that calls self.compute_epoch();
        this function comes with additional stat management.

        params:
        train_data: models representing training_data.
        """

        valid_accs = []
        
        # training data
        start = time.time()
        train_stats = self.compute_epoch(self.training_data, False)
        train_loss, train_acc = train_stats

        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
            'elapse: {elapse:3.3f} min'.format(
                ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                elapse=(time.time()-start)/60))

        # validation data
        with torch.no_grad():
            valid_stats = self.compute_epoch(self.validation_data, True)
        valid_loss, valid_acc = valid_stats

        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
            'elapse: {elapse:3.3f} min'.format(
                ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                elapse=(time.time()-start)/60))

        valid_accs.append(valid_acc)
        extra_info = 
        self.save(extra_info)
        self.update_logs(train_stats, valid_stats, ep)

        return self

    def translate(self, test_data):
        """
        Batch translates sequences.

        Assumes test_data is a DataLoader.
        """
        with torch.no_grad():
            pass
        return sequences

    def save(self, extra_info):
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

        # make sure a path is specified prior to saving the files.
        if opt.save_model:
            if opt.save_model == "all":
                model_name = opt.save_model + '_accu_'
        else:
            if not self.save_trip:
                print("Note: the model is not specified to save.")
                self.save_trip = True
    
    # ---------------------------
    # Below the line represents transformer specific code.
    # ---------------------------

    def performance(self, pred, gold, smoothing=False):
        """
        Calculates token level accuracy.
        Smoothing can be applied if needed.
        """
        loss = calculate_loss(pred, gold, smoothing)
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

        for batch in tqdm(dataset, mininterval=2, desc=' - (Training)\t', leave=False):
            # prepare data
            src_seq, src_pos, tqt_seq, tgt_pos = map(lambda x: x.to(self.device), batch)
            gold = tgt_seq[:, 1:]
            
            if not self.validation:
                self.optimiser.zero_grad()
            # compute forward propagation
            predictions = self.model(src_seq, src_pos, tgt_seq, tgt_pos)
            # compute performance
            loss, n_correct = performance(pred, gold, smoothing=self.opt.label_smoothing)

            if not self.validation:
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