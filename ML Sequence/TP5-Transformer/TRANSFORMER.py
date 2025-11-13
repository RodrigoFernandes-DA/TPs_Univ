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

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        # add the batch dimension
        pos_encoding = pos_encoding.unsqueeze(0) #.transpose(0, 1) 
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        #return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
        #print("token_embedding.shape",token_embedding.shape)
        #print("pos_encoding.shape",self.pos_encoding.shape)
        return self.dropout(token_embedding + self.pos_encoding[:,:token_embedding.size(1), :])

    
###########################################################

# -------------------------
# CNN encoder (2D per-window)
# -------------------------
class CNNEncoder(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers=10):
        super(CNNEncoder, self).__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        
        # Create 10 CNN layers
        layers = []
        
        # First layer: from input_features to hidden_size//4
        layers.extend([
            nn.Conv1d(in_channels=input_features, out_channels=hidden_size//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//4),
            nn.Dropout(0.1)
        ])
        
        # Intermediate layers
        current_channels = hidden_size//4
        for i in range(num_layers - 2):
            layers.extend([
                nn.Conv1d(in_channels=current_channels, out_channels=current_channels * 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(current_channels * 2),
                nn.Dropout(0.1)
            ])
            current_channels = current_channels * 2
            
            # If we exceed hidden_size, maintain it
            if current_channels > hidden_size:
                current_channels = hidden_size
        
        # Final layer to match transformer hidden_size
        layers.extend([
            nn.Conv1d(in_channels=current_channels, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1)
        ])
        
        self.cnn_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_features)
        batch_size, seq_len, features = x.shape
        
        # Reshape for CNN: (batch_size, input_features, seq_len)
        # Treat each feature dimension as a channel
        x = x.transpose(1, 2)  # (batch_size, input_features, seq_len)
        
        # Apply CNN layers
        x = self.cnn_layers(x)
        
        # Reshape back to transformer expected format: (batch_size, seq_len, hidden_size)
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        # add the batch dimension
        pos_encoding = pos_encoding.unsqueeze(0) #.transpose(0, 1) 
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:,:token_embedding.size(1), :])

class Transformer(nn.Module):
    def __init__(self, config, device):
        super(Transformer, self).__init__()
        self.input_features = config['input_features']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']
        self.num_classes = config['num_classes']
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.pad_idx = config['pad_idx']
        self.max_length = config['max_length']
        self.START_TOKEN = config['START_TOKEN']
        self.END_TOKEN = config['END_TOKEN']
        self.DEVICE = device

        # Add CNN encoder before transformer
        self.cnn_encoder = CNNEncoder(
            input_features=config['input_features'],
            hidden_size=config['hidden_size'],
            num_layers=10
        )
        
        self.positional_encoding_layer = PositionalEncoding(
            dim_model=self.hidden_size, 
            dropout_p=self.dropout, 
            max_len=self.max_length
        )
        
        # Since CNN already outputs hidden_size, we can use a simple projection or identity
        self.x_embedding = nn.Linear(config['hidden_size'], config['hidden_size'])        
        
        self.y_embedding = nn.Embedding(config['num_classes'], config['hidden_size'])         
        
        self.transformer = nn.Transformer(
            d_model=config['hidden_size'], 
            nhead=config['num_heads'], 
            num_encoder_layers=config['num_layers'],
            num_decoder_layers=config['num_layers'],
            dim_feedforward=config['hidden_size'],
            dropout=config['dropout'],
            batch_first=True
        )
        
        # Output layer for text prediction
        self.output_layer = nn.Linear(config['hidden_size'], config['num_classes'])

    def forward(self, x, y, y_output_mask, src_key_padding_mask, tgt_key_padding_mask):
        # Apply CNN encoding first
        x = self.cnn_encoder(x)
        
        # Then apply the original transformer processing
        x = self.x_embedding(x)
        x = self.positional_encoding_layer(x)
        
        y = self.y_embedding(y) * math.sqrt(self.hidden_size)
        y = self.positional_encoding_layer(y)

        # Transformer blocks
        transformer_out = self.transformer(
            x, y,
            tgt_mask=y_output_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        output = self.output_layer(transformer_out)
        return output
    
    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

#########################################################
def train_loop(dataloader, model, loss_fn, optimizer):
    #print("Training loop...")
    size = len(dataloader.dataset)
    nb_batches = len(dataloader)
    epoch_loss = 0
    
    model.train()
    
    for batch, (X, y,X_l,y_l) in enumerate(dataloader):
        
        X = X.to(model.DEVICE)
        y = y.to(model.DEVICE)

        #y_input = y[:-1,:] # on retire le END_TOKEN car nest jamais traité en entrée
        #y_output = y[1:,:] # on ne cherche jamais à predire le START_TOKEN
        
        y_input = y[:,:-1] 
        y_output = y[:,1:] 
        
        l = y_input.size(1) # la longueur maximale d'un élément du batch
        #print("y_input.size(0)",y_input.size(0),y_input.size(1))
        y_output_mask = model.get_tgt_mask(l).to(model.DEVICE)

        #x_padding_mask = (X[:,:,0] == model.pad_idx).transpose(0, 1).to(model.DEVICE)
        #y_input_padding_mask = (y_input == model.pad_idx).transpose(0, 1).to(model.DEVICE)
        
        x_padding_mask = (X[:,:,0] == model.pad_idx).to(model.DEVICE)
        y_input_padding_mask = (y_input == model.pad_idx).to(model.DEVICE)
        
        # Compute prediction and loss
        pred = model(X.float(), y_input,
                     y_output_mask,
                     x_padding_mask, 
                     y_input_padding_mask) #tgt_is_causal = True)
        pred = pred.permute(1, 2, 0) 
        y_output = y_output.permute(1, 0) 

        loss = loss_fn(pred, y_output)
        epoch_loss += loss.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #print("\nTraining loss:",epoch_loss / nb_batches)
    return epoch_loss / nb_batches

#########################################################
def valid_loop(dataloader, model, loss_fn):

    size = len(list(dataloader.dataset))
    nb_batches = len(dataloader)
    valid_loss = 0
    
    with torch.no_grad():
        for batch, (X, y,X_l,y_l) in enumerate(dataloader):
            
            X = X.to(model.DEVICE)
            y = y.to(model.DEVICE)
            
            #y_input = y[:-1,:] # on retire le END_TOKEN car nest jamais traité en entrée
            #y_output = y[1:,:] # on ne cherche jamais à predire le START_TOKEN
            y_input = y[:,:-1] 
            y_output = y[:,1:] 
            
            l = y_input.size(1) # la longueur maximale d'un élément du batch
            y_output_mask = model.get_tgt_mask(l).to(model.DEVICE)
            
            #x_padding_mask = (X[:,:,0] == model.pad_idx).transpose(0, 1).to(model.DEVICE)
            #y_input_padding_mask = (y_input == model.pad_idx).transpose(0, 1).to(model.DEVICE)
            
            x_padding_mask = (X[:,:,0] == model.pad_idx).to(model.DEVICE)
            y_input_padding_mask = (y_input == model.pad_idx).to(model.DEVICE)
            
            # Compute prediction and loss
            pred = model(X.float(),y_input,
                         y_output_mask,
                         x_padding_mask,
                         tgt_key_padding_mask = y_input_padding_mask)
            
            pred = pred.permute(1, 2, 0)      
            y_output = y_output.permute(1, 0) 
            
            loss = loss_fn(pred, y_output)
            valid_loss += loss.item()
    

    valid_loss /= nb_batches
    
    return valid_loss



