#### implementation of Meta-Agnostic Meta-Learning, Finn, et.al
import numpy as np
import sys
from copy import deepcopy

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
        
        self.lr = lr
        self.meta_lr = lr
        self.lr_decay = lr_decay
        
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.epsilon = epsilon #bounding condition

        #define loss functions
        self.pr_loss = nn.NLLLoss(ignore_index=0)
        self.dec_loss = nn.NLLLoss(ignore_index=0)
        
        #define model
        self.model = TSD() # TODO - implement TSD class
        
    def train(self, feature_matrix):
        """ Train model using MAML params """
        for epoch in self.epochs:
            meta_lr = self.lr_decay #set meta lr based on decay rate
            ### Define optimizers
            optim = Adam(lr=self.lr, weight_decay=self.weight_decay)
            meta_optim = Adam(lr=meta_lr, weight_decay=self.weight_decay)
            
            loss_tasks = [] #defined in Finn. et.al maml #L95
            init_state = deepcopy(self.model.state_dict()) #transfer model state later in meta-update
            
            for observation in feature_matrix:
                for k in range(len(feature_matrix)):#run single-task gradient updates for inner optimizer
                    loss = self.model(observation) # update pseudocode.
                    loss.backward()
                    grad = nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                
                loss_tasks.append(loss)
            
            #meta step calculation
            loss = self.model(observation)
            self.model.load_state_dict(init_state)
            meta_optim.zero_grad()
            
            loss_meta = torch.stack(loss_tasks).sum(0) / len(feature_matrix) #implementation different from MAML in Finn's paper from DAML.
            loss_meta.backward()

            nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
            meta_optim.step() #perform meta-update
            
            init_state = deepcopy(self.model.state_dict())
            
            
            
                    
            
        
