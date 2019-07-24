"""
Model 'interface';
Note that I don't follow this strictly since
init does have some logic in it.
"""

import torch
import math
import core.constants as constants
from core.dataset import TranslationDataset, collate_fn, paired_collate_fn
from core.bpe import Encoder as BPE
import os
import torch.utils.data
import datetime
import atexit
import telebot
import json
import numpy as np

from preprocess import load_file, seq2idx

class NMTModel:
    def __init__(self, opt, models_folder="models"):
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')
        self.constants = constants
        self.opt.directory = self.init_dir(stores=models_folder)
        
        # update variables.
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []

        if self.opt.telegram:
            self.init_telegram(self.opt.telegram)

        # bpe holders.
        self.src_bpe = None
        self.tgt_bpe = None

        atexit.register(self.exit_handler)
    # -------------------------
    # OVERLOADED FUNCTIONS
    # These functions are to be overwritten by whatever is declared
    # in their child functions.
    # -------------------------

    def load(self):
        """
        Loads models from file.
        """
        print("[Warning]: load() is not implemented.")
        return self
    
    def initiate(self):
        """
        Loads models into memory and initiate parameters.
        """
        print("[Warning]: initiate() is not implemented.")
        return self
    
    def setup_optimiser(self):
        # based on the opt.
        print("[Warning] setup_optimiser() is not implemented.")

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
        return self

    def translate(self, dataset):
        """
        Uses the models to perform inference/translation.
        """
        print("[Warning]: translate() is not implemented.")
        return self
    
    def save(self):
        """
        save model weights and parameters to file.
        """
        print("[Warning]: save() is not implemented.")
        return self

    # ----------------------------
    # BASE FUNCTIONS
    # ----------------------------

    def reset_metrics(self):
        """
        Resets metrics.
        """
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []

    def init_dir(self, stores):
        """
        initiates a directory model data will be in.
        If theres no preloaded encoder/decoder it'll 
        create a new folder.
        """
        if self.opt.checkpoint_encoder:
            return os.path.split(self.opt.checkpoint_encoder)[0]

        # setup parent directory
        basepath = os.path.abspath(os.getcwd())
        basepath = os.path.join(basepath, stores)

        if self.opt.directory_name:
            directory_name = self.opt.directory_name
        else:
            # setup current model directory
            directory_name = self.opt.model
            directory_name += "-" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        directory = os.path.join(basepath, directory_name)
        # create container for that folder.
        if not os.path.isdir(directory):
            if self.opt.save_model:
                os.mkdir(directory)
        return directory 

    def init_logs(self):
        """
        If called, saves output logs to file.
        """
        self.log_train_file = os.path.join(self.opt.directory,"train.log")
        self.log_valid_file = os.path.join(self.opt.directory,"valid.log")
        # check if log files exists already.
        # if it does, then we need to increment the log name
        step = 1
        while os.path.isfile(self.log_train_file):
            filename = "train_" + str(step) + ".log"
            self.log_train_file = os.path.join(self.opt.directory,filename)
            step += 1
        step = 1
        while os.path.isfile(self.log_valid_file):
            filename = "valid_" + str(step) + ".log"
            self.log_valid_file = os.path.join(self.opt.directory,filename)
            step += 1
        
        print('[Info] Training performance will be written to file: {} and {}'.format(
            self.log_train_file, self.log_valid_file))

        with open(self.log_train_file, 'w') as log_tf, open(self.log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    def update_logs(self, epoch_i):
        """
        called within train(). Updates results into log files.
        Assumes logs are enabled.
        """
        train_loss, train_acc = self.train_losses[-1], self.train_accs[-1]
        valid_loss, valid_acc = self.valid_losses[-1], self.valid_accs[-1]

        # deal with logs
        if self.log_train_file and self.log_valid_file:
            with open(self.log_train_file, 'a') as log_tf, open(self.log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_acc))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_acc))
        else:
            print("[Warning] log files are not initiated. No updates are kept into storage.")

    def load_dataset(self):
        """
        Loads PyTorch pickled training and validation dataset.
        """
        data = torch.load(self.opt.data)
        # the token sequence length is determined by `preprocess.py`
        self.opt.max_token_seq_len = data['settings'].max_token_seq_len
        # here we need to check whether the dataset is BPE or not.
        if 'byte_pairs' in data['dict']['src']:
            if '__sow' in data['dict']['src']['byte_pairs']:
                # self.src_bpe = BPE(vocab_size=4096, pct_bpe=0.8, ngram_min=1, UNK=constants.UNK_WORD, PAD=constants.PAD_WORD, word_tokenizer=self.parse)
                # self.tgt_bpe = BPE(vocab_size=4096, pct_bpe=0.8, ngram_min=1, UNK=constants.UNK_WORD, PAD=constants.PAD_WORD, word_tokenizer=self.parse)
                # self.src_bpe.from_dict(data['dict']['src'])
                # self.tgt_bpe.from_dict(data['dict']['tgt'])
                self.src_bpe = BPE.from_dict(data['dict']['src'])
                self.tgt_bpe = BPE.from_dict(data['dict']['tgt'])

        datasets = self.init_dataloaders(data, self.opt)
        self.training_data, self.validation_data = datasets

        # need to store vocabulary size for quick referencing
        self.opt.src_vocab_size = self.training_data.dataset.src_vocab_size
        self.opt.tgt_vocab_size = self.training_data.dataset.tgt_vocab_size
        return self

    def load_testdata(self, test_datapath, test_vocab):
        """
        Loads a text file representing sequences.

        params:
        test_datapath: some text file.
        test_vocab: it's the same PyTorch pickled training dataset.
        """
        
        # load vocabulary
        data = torch.load(test_vocab)
        settings = data['settings']
        # print(settings)
        # load test sequences
        token_instances = load_file(test_datapath, 
                                    settings.max_word_seq_len,
                                    settings.format,
                                    settings.case_sensitive)
        is_bpe = settings.format.lower() == "bpe"
        if is_bpe:
            # load test data
            # bpe_src = BPE(vocab_size=4096, pct_bpe=0.8, ngram_min=1, UNK=constants.UNK_WORD, PAD=constants.PAD_WORD, word_tokenizer=self.parse)
            bpe_src = BPE.from_dict(data['dict']['src'])
            # convert test sequences into IDx
            test_src_insts = bpe_src.transform(token_instances)
            test_src_insts = [i for i in test_src_insts]
            # setup data loader
            src_word2idx = data['dict']['src']
            tgt_word2idx = data['dict']['tgt']

            src_byte_pairs = {x+"_": y for x,y in src_word2idx['byte_pairs'].items()}
            tgt_byte_pairs = {x+"_": y for x,y in tgt_word2idx['byte_pairs'].items()}
            src_word2idx = {**src_byte_pairs, **src_word2idx['words']}
            tgt_word2idx = {**tgt_byte_pairs, **tgt_word2idx['words']}
            
            test_loader = torch.utils.data.DataLoader(
                TranslationDataset(
                    src_word2idx=src_word2idx,
                    tgt_word2idx=tgt_word2idx,
                    src_insts=test_src_insts),
                num_workers=2,
                batch_size=self.opt.batch_size,
                collate_fn=collate_fn)

        else:
            # convert test sequences into IDx
            test_src_insts = seq2idx(token_instances, data['dict']['src'])
            # setup data loaders.
            test_loader = torch.utils.data.DataLoader(
                TranslationDataset(
                    src_word2idx=data['dict']['src'],
                    tgt_word2idx=data['dict']['tgt'],
                    src_insts=test_src_insts),
                num_workers=2,
                batch_size=self.opt.batch_size,
                collate_fn=collate_fn)
        
        return test_loader, settings.max_token_seq_len, is_bpe


    @staticmethod
    def init_dataloaders(data, opt):
        """
        Initiates memory efficient vanilla dataloaders for feeding 
        into the models. (Assumes dataset is not BPE)
        """
        src_word2idx = data['dict']['src']
        tgt_word2idx = data['dict']['tgt']
        # check if we have BPE dictionaries
        is_bpe = False
        if 'byte_pairs' in src_word2idx:
            if '__sow' in src_word2idx['byte_pairs']:
                is_bpe = True
                # we have BPE
                src_byte_pairs = {x+"_": y for x,y in src_word2idx['byte_pairs'].items()}
                tgt_byte_pairs = {x+"_": y for x,y in tgt_word2idx['byte_pairs'].items()}
                src_word2idx = {**src_byte_pairs, **src_word2idx['words']}
                tgt_word2idx = {**tgt_byte_pairs, **tgt_word2idx['words']}

        train_loader = torch.utils.data.DataLoader(
            TranslationDataset(
                src_word2idx=src_word2idx,
                tgt_word2idx=tgt_word2idx,
                src_insts=data['train']['src'],
                tgt_insts=data['train']['tgt']),
            num_workers=2,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn,
            shuffle=True)

        valid_loader = torch.utils.data.DataLoader(
            TranslationDataset(
                src_word2idx=src_word2idx,
                tgt_word2idx=tgt_word2idx,
                src_insts=data['valid']['src'],
                tgt_insts=data['valid']['tgt']),
            num_workers=2,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn)
        
        # validate the tables if weight sharing flag is called.
        if opt.embs_share_weight:
            assert train_loader.dataset.src_word2idx == train_loader.dataset.tgt_word2idx, \
                'The src/tgt word2idx table are different but you asked to share the word embeddings.'

        return train_loader, valid_loader

    def exit_handler(self):
        """
        Handles anything when the code has finished running.
        """
        if self.opt.save_model:
            if len(os.listdir(self.opt.directory)) < 1:
                # delete the folder since nothing interesting happened.
                os.rmdir(self.opt.directory)
        return None

    def init_telegram(self, api_json_path):
        """
        Initiates telegram API bot.
        (Quite useful if you want to get notified of any
        status changes in the event that you're afk with model
        training.)
        """
        api_q = json.load(api_json_path)
        self.bot = telebot.TeleBot(api_q['api_private_key'])
        self.bot_chatid = api_q['chat_id']

    def t_msg(self, msg):
        """
        Sends messages to telegram chat ID.
        """
        if self.bot:
            self.bot.send_message(self.bot_chatid, msg)

    @staticmethod
    def parse(x):
        return x.split()