# -*- coding: utf-8 -*-

import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

import matplotlib.pyplot as plt
import math
import tqdm
from tqdm import tqdm
import numpy as np

###########################################################
class LSTM(nn.Module):
  '''
  '''
  def __init__(self,config):
    super(LSTM,self).__init__()
    self.hidden_size = config['hidden_size']
    self.n_layer = config['lstm_layer']

    self.lstm = nn.LSTM(config['input_features'], 
                        config['hidden_size'], 
                        config['lstm_layer'], 
                        batch_first=False,
                        bidirectional=config['blstm']) 
    # self.fc1 = nn.Linear(config['hidden_size'], config['hidden_size'])
    in_features = config['hidden_size'] * (2 if config['blstm'] else 1)
    self.fc1 = nn.Linear(in_features, config['hidden_size'])

  def forward(self, x):      # l'activation softmax est mise dans l'appel à la loss CTC
    '''Forward pass'''       # il faut l'ajouter au moment du test si on le souhaite
    # B, C, H, W = x.size()
    # x = x.view(B, H, W)        # remove channel
    # x = x.permute(2, 0, 1)     # [W, B, 28]
    out, (hn, cn) = self.lstm(x)
    out = self.fc1(out)
    return out

#########################################################
def train_loop(dataloader, model, loss_fn, optimizer):
    print("Training loop...")
    size = len(dataloader.dataset)
    nb_batches = len(dataloader)
    epoch_loss = 0
    for batch, (X, y,X_l,y_l) in tqdm(enumerate(dataloader)):
        # Compute prediction and loss
        pred = torch.nn.functional.log_softmax(model(X.float()),dim = 2)
        loss = loss_fn(pred, y,X_l,y_l)
        # pred = torch.nn.functional.log_softmax(model(X.float()),dim = 1)
        # loss = loss_fn(pred.permute(2,0,1), y,X_l,y_l)
        epoch_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("\nTraining loss:",epoch_loss / nb_batches)
    return epoch_loss / nb_batches

#########################################################
def valid_loop(dataloader, model, loss_fn):
    print("Valid_loop...")
    size = len(list(dataloader.dataset))
    nb_batches = len(dataloader)
    valid_loss = 0
    with torch.no_grad():
        for X, y,X_l,y_l in dataloader:
            pred = torch.nn.functional.log_softmax(model(X.float()),dim = 2)
            # pred = torch.nn.functional.log_softmax(model(X.float()),dim = 1)
            valid_loss += loss_fn(pred, y,X_l,y_l).item()
            # valid_loss += loss_fn(pred.permute(2,0,1), y,X_l,y_l).item()

    valid_loss /= nb_batches
    print("Valid loss:",valid_loss)
    return valid_loss

###########################################################################
def StatesToSymbols(best_path,T,config):
    last_symbol = best_path[0]
    best_symbols = [last_symbol]
    
    for t in range(1,T):
       if best_path[t] != last_symbol:
            last_symbol = best_path[t]
            best_symbols.append(last_symbol)

    bs = ''.join([str(best_symbols[i]) for i in range(len(best_symbols)) if best_symbols[i] != config['blank_label']])
    
    return bs

class CNN (nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        #premier bloc conv ReLU Pooling à 16 canaux = sortie de 16 X 14 X T/2 
        self.DEVICE = config['device']
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn. ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn. ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=11, kernel_size=(7,1), stride=1, padding=0), 
            nn. Flatten (2) # on élimine la dimension verticale déjà ramenée à 1 par le kernel (7,1)
        )
        
    def forward(self, x):
        output = self.cnn(x)
        return output