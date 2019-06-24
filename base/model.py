"""
Model 'interface';
Note that I don't follow this strictly since
init does have some logic in it.
"""

import torch
import core.constants as constants
import os

class Model:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')
        self.constants = constants
        self.opt.dir = self.mkdir(opt.model)
        
    @staticmethod
    def make_dir(self, path):
        """
        makes a directory model data will be in.
        """
        directory_name = ""
        os.mkdir(path)

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

    def translate(self):
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

    def load_data(self):
        """
        Loads PyTorch pickled file, representing the dataset.
        """
        data = torch.load(self.opt.data)
        # the token sequence length is determined by `preprocess.py`
        opt.max_token_seq_len = data['settings'].max_token_seq_len
        datasets = self.init_dataloaders(data, self.opt)
        self.training_data, self.validation_data = datasets
        return self

    @staticmethod
    def init_dataloaders(data, opt):
        """
        Creates memory efficient dataloaders for feeding into the models.
        """