# -*- coding: utf-8 -*-

import numpy as np

import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
from tqdm import tqdm
import os
import editdistance

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import CTC

def Load_MNISTSequences(file_name):
    ###########################################################################
    # on charge les dataset des séquence de train et de test
    print("Loading training and test data...")
    pkl_file = open('MNIST_5digitsDifficile.pkl', 'rb') 
    x_train,yy_train, x_test,yy_test = pickle.load(pkl_file) 
    pkl_file.close()
    
    y_train=[]
    y_test=[]
    #####################################################################
    # reformatage de la ground truth en string
    for n in range(len(yy_train)):
        GT = yy_train[n].T[0][:]
        y_train.append(torch.from_numpy(GT))
        x_train[n] = torch.from_numpy(x_train[n])
    
    for n in range(len(yy_test)):
        GT = yy_test[n].T[0][:]
        y_test.append(torch.from_numpy(GT)) 
        x_test[n] = torch.from_numpy(x_test[n])
    ###############################################################
    
    return x_train,x_test,y_train,y_test

##############################################################################
class DigitSequenceDataset(Dataset):
    """5 digits sequences MNIST datset."""

    def __init__(self, DATA, LABELS,):
        """
        Arguments:
        """
        self.data = DATA            
        self.labels = LABELS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx],self.labels[idx]
                                                                                         
###############################################################
def pad_collate(batch): # recebe um batch de train
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  x_lens = torch.LongTensor(x_lens) # todas as len dos exemplos
  y_lens = torch.LongTensor(y_lens)
  xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=False, padding_value=-1)
  yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=-1)

  return xx_pad, yy_pad, x_lens, y_lens 

#######################################
if __name__ == '__main__':
    
    TRAINING = True  # Training if True Testing otherwise
    SHOW = True
    
    device="cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    
    config = {
        'input_features':28,
        'batch_size':64,
        'num_epochs':60,
        'learning_rate':1e-3,
        'num_classes':11,
        'blank_label':10,
        'hidden_size':128, #256
        'lstm_layer':1,#2
        'blstm': False, #True
        #'device': device,
    }

    x_train,x_test,y_train,y_test = Load_MNISTSequences('MNIST_5digitsDifficile.pkl')
    
    ############################################################
    N_train = len(y_train)
    
    l_seq_digits = 5
    D = 28 # 28 X 28
    
    N_max = 60000 # digits
    N_max_seq = int(N_max / l_seq_digits) # 60000 / 5 digits
    
    N_train = 10000 # digits
    N_train_seq = int(N_train / l_seq_digits)
    END_TRAIN = N_train_seq # 90% 10 000 digits = 5 X 2000
    
    N_batch = int(N_train_seq / config['batch_size'])
    N_valid = 1000
    N_valid_seq = int(N_valid / l_seq_digits)
    START_VALID = N_max_seq - N_valid_seq  # 10% for validation 1000 digits = 5 X 200
    
    model_name = "LSTM_"+str(config['hidden_size'])+"_"+str(N_train)
    

    if TRAINING:               
        # On crée les dataloarder de pytorch pour géréer les lots de séquences
        # car on ne peut plus mettre les séquences bout à bout puisqu'on entaine
        # des réseaux récurrentsou des CNN qui exploitent le contexte
        # pour le TRAIN
        train = x_train[:N_train_seq]
        gt_train = y_train[:N_train_seq]
        Train_seq_dataset = DigitSequenceDataset(train,gt_train)
        train_dataloader = torch.utils.data.DataLoader(Train_seq_dataset,
                                                        batch_size = config['batch_size'],
                                                        shuffle=True, 
                                                        collate_fn = pad_collate)
        
        ############### pour la VALID #################
        valid = x_train[N_max_seq - N_valid_seq:]
        gt_valid = y_train[N_max_seq - N_valid_seq:]
        Valid_seq_dataset = DigitSequenceDataset(valid,gt_valid)
        valid_dataloader = torch.utils.data.DataLoader(Valid_seq_dataset,
                                                       batch_size = config['batch_size'],
                                                       collate_fn = pad_collate)
                                                    
        ###########################################################
        if SHOW:
            for batch in (train_dataloader):
                for i in range(5):
                    plt.imshow(np.flip(batch[0][:,i,:].numpy().T,axis=0), cmap='gray')
                    plt.title(batch[1][i])
                    plt.show()    
              
                break
        ############################################################
 
        my_lstm = CTC.LSTM(config)
        loss_fn = torch.nn.CTCLoss(blank=config['blank_label'], reduction='mean')
        optimizer = torch.optim.Adam(my_lstm.parameters(),lr=config['learning_rate'])
        #optimizer = torch.optim.RMSprop(my_lstm.parameters(),lr=config['learning_rate'])
        
        train_loss = []
        valid_loss = []
        for e in range(config['num_epochs']):
            train_loss.append(CTC.train_loop(train_dataloader,my_lstm,loss_fn, optimizer))
            if e == 0:
                valid_loss.append(train_loss[0])
            valid_loss.append(CTC.valid_loop(valid_dataloader,my_lstm,loss_fn))

            ###################################################################
            if SHOW and (e<10 or e==config['num_epochs']-1):
                plt.plot(valid_loss,color = 'red', label = "valid")
                plt.plot(train_loss, color = 'blue', label = " train")
                plt.title("CTC loss over epochs")
                plt.legend()
                plt.show()
            ###################################################################
            if e == 0:
                best_valid_loss = valid_loss[0]
            else:
                if valid_loss[-1] < best_valid_loss:
                    best_valid_loss = valid_loss[-1]
                    # on mémorise ce modèle
                    best_iteration = e
                    torch.save(my_lstm.state_dict(), model_name)
            
        print('Embedded Training Ended successfully')
    
    else:
        ###########################################################################
        N_test_seq = len(x_test)
        Test_seq_dataset = DigitSequenceDataset(x_test,y_test)
        test_dataloader = torch.utils.data.DataLoader(Test_seq_dataset,
                                                        batch_size = 1,
                                                        collate_fn = pad_collate)
        
        my_lstm = CTC.LSTM(config)
        my_lstm.load_state_dict(torch.load(model_name))
    
        TOTAL = 0
        FP =0
        String_FP = 0
        # loop testing every test sample and computing the Character Rrror Rate (CER)
        print("Recognition in progress...")
        
        with torch.no_grad():
            for X, y,X_l,y_l in test_dataloader:
                ###### formate la gt en string ##########################
                y = y.numpy()[0]
                y = ''.join([str(y[i]) for i in range(y.shape[0])])

                pred = my_lstm(X.float())
                prediction = pred[:,0,:].numpy()
                y_pred = np.argmax(prediction,axis=1) 
                best_sequence = CTC.StatesToSymbols(y_pred,X_l,config)

                print("y :",y,"best_sequence :",best_sequence)
                # if SHOW:
                #     plt.imshow(np.flip(x_test[n].T,axis=0), cmap='gray')
                #     plt.title(gt+" "+bs)
                #     plt.show()       
                #     print("GT:",gt,"recognized",bs)
                # print("best path",best_path)
                Edit_dist = editdistance.eval(y,best_sequence)
                FP += Edit_dist
                TOTAL += len(y)
                if Edit_dist !=0:
                    String_FP +=1
                    
        print("FP",FP)
        print("TOTAL characters",TOTAL)
        print("String FP",String_FP)
        print("TOTAL strings",N_test_seq)
        print('Recognition ended successfully, Character Error Rate = ',FP/TOTAL*100,'%')
        print('                              , Character Recognition Rate = ',(1-FP/TOTAL)*100,'%')
        print('                              , String Error Rate    = ',String_FP/N_test_seq*100,'%')
        print('                              , String Recognition Rate    = ',(1-String_FP/N_test_seq)*100,'%')