#### implementation of Meta-Agnostic Meta-Learning, Finn, et.al
import numpy as np
import sys

import torch 
from torch.optim import Adam
import torch.nn as nn
from torch.autograd import Variable


class MAML():
    """
    Parameter optimization using meta-learning model, M, using MAML algorithm
    """
    def __init__(self, epochs=100, epsilon=1e-1, lr=0.003, weight_decay=1e-5, lr_decay=0.5):
        """
        Initialize model based on different hyper-parameters set forth by DAML paper.
        source: https://arxiv.org/pdf/1906.03520.pdf
        """
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.epsilon = epsilon #bounding condition
        self.lr_decay = lr_decay
    
    def train(self, feature_matrix):
        """ Train model using MAML params """
        for epoch in self.epochs:
            optim = Adam(lr=self.lr, weight_decay=self.weight_decay)
            meta_optim = Adam()
            
        
