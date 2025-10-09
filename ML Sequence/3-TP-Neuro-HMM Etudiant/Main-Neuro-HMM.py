# -*- coding: utf-8 -*-

import numpy as np

# pour charger les données MNIST avec python https://pypi.org/project/python-mnist/
# pip install python-mnist 
#from mnist import MNIST
#from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
from tqdm import tqdm
import os
# import editdistance

import torch
from torch import nn

import mlp
import neuro_hmm


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
        gt = ''.join([str(yy_train[n].T[0][i]) for i in range(len(yy_train[n].T[0]))])
        y_train.append(gt)    
    
    for n in range(len(yy_test)):
        gt = ''.join([str(yy_test[n].T[0][i]) for i in range(len(yy_test[n].T[0]))])
        y_test.append(gt) 
    ###############################################################
    return x_train,x_test,y_train,y_test
##############################################################################
##############################################################################
# # extraction d'une fenetre glissante sur les images d'entrée
def Apply_SlidingWindow(x,config,SHOW=False):
    D = x[0].shape[1]
    xx=[]
    for n in range(len(x)):
        T = x[n].shape[0]
        TT = int(T / config['w_stride'])-1
        
        feature = np.zeros((TT,config['input_features']))
        begin = 0
        for t in range(TT):
            end = begin + config['w_width']
            feature[t,:] = x[n][begin:end,:].reshape((config['input_features']))
            begin += config['w_stride']
        xx = xx + [feature]
    
    if SHOW:
        for i in range(5):
            plt.imshow(np.flip(x[i].T,axis=0), cmap='gray')
            plt.show()
            plt.imshow(np.flip(xx[i].T,axis=0), cmap='gray')
            plt.show()

    return xx
 
if __name__ == '__main__':
    N_max = 60000 # digits    
    D = 28
    l_seq_digits = 5
    
    TRAINING = True  # Training if True Testing otherwise
    SHOW = True
    batch_size = 128

    N_train = 10000 # digits
    N_valid = 1000
    # Sliding window parameters
    w_width = 3
    stride = 2

    x_train,x_test,y_train,y_test = Load_MNISTSequences('MNIST_5digitsDifficile.pkl')
    N_train = len(y_train)
    
    N_max_seq = int(N_max / l_seq_digits) # 60000 / 5 digits
    N_train_seq = int(N_train / l_seq_digits)
    END_TRAIN = N_train_seq # 90% 10 000 digits = 5 X 2000
    N_batch = int(N_train_seq / batch_size)
    N_valid_seq = int(N_valid / l_seq_digits)
    START_VALID = N_max_seq - N_valid_seq  # 10% for validation 1000 digits = 5 X 200
 
    config = {
        'N_max_seq':N_max_seq,
        'l_seq_digits':l_seq_digits,
        'N_train':N_train,
        'N_valid':N_valid,
        'N_train_seq':N_train_seq,
        'N_valid_seq':N_valid_seq,
        'w_width':w_width,
        'w_stride':stride,
        'input_features':D*w_width, # w_width * 28
        'batch_size':batch_size,
        'num_epochs':40,
        'hidden_size':128,
        'learning_rate':1e-3,
        'N_classes':10,
        'pad_idx':-1,
        'max_length':140,
        'END_TRAIN': END_TRAIN,
        'START_VALID':START_VALID,
        'n_states':5,
    }
    
    model_name = "MLP_"+str(config['input_features'])+"_"+"_"+str(config['hidden_size'])+"_"+str(config['n_states'])+"_"+str(config['N_train'])
    
    x_train = Apply_SlidingWindow(x_train,config,SHOW=False)
    
    if TRAINING:        
        train = x_train[:config['N_train_seq']]
        gt_train = y_train[:config['N_train_seq']]
        
        valid = x_train[config['N_max_seq'] - config['N_valid_seq']:]
        gt_valid = y_train[config['N_max_seq'] - config['N_valid_seq']:]
        
        ###########################################################
        if SHOW:
            for i in range(5):
                plt.imshow(np.flip(train[i].T,axis=0), cmap='gray')
                plt.title(gt_train[i])
                plt.show()     
            
        # create the list of HMM model of digits with equi probable intialization
        Models = []    
        for classe in range(config['N_classes']):
            Models.append( neuro_hmm.Neuro_HMM(config['n_states'],TRAIN=TRAINING))
                  
        ################ launch embedded training #########################
        LL_train_history, LL_valid_history, best_iteration = neuro_hmm.Embedded_Training(
                Models,train,valid,gt_train,gt_valid,
                config,model_name)
        
        ###################################################################
        plt.plot( LL_valid_history,color = 'red', marker='o', label = "valid")
        LL_train = [LL_train_history[i*N_batch] for i in range(config['num_epochs'])]
        plt.plot( LL_train, color = 'blue', marker='o', label = " train")
        plt.title("Embedded training LL over EM iterations")
        plt.legend()
        plt.show()
            
        print('Embedded Training Ended successfully')
    
    else:
        ###########################################################################
        N_test_seq = len(x_test)
        x_test = Apply_SlidingWindow(x_test,config,SHOW=False)
        ###########################################################################
        print("Create digit models: n_states =",config['n_states'],"hydden size =",
              config['hidden_size'],"N_train = ",config['N_train'])

        Models = []
        for classe in range(config['N_classes']):
            model = neuro_hmm.Neuro_HMM(config['n_states'],TRAIN=TRAINING)
            Models.append(model)
        
        print("Build a Left-Right ergodic digit sequence model")
        Ergodic_Model = neuro_hmm.Ergodic_HMM(config,Models)
    
        my_mlp = mlp.MLP(config)
        my_mlp.load_state_dict(torch.load(model_name))
    
        TOTAL = 0
        FP =0
        String_FP = 0
        # loop testing every test sample and counting True and False positive
        print("Recognition in progress...")
        for n in tqdm(range(N_test_seq)):

            gt = y_test[n]
            
            LL_best_path, best_path = Ergodic_Model.Viterbi(x_test[n],my_mlp,BACKTRACK = True)
            
            # retrieve best symbol sequence from best states sequence
            bs = neuro_hmm.StatesToSymbols(best_path,config['n_states'])
    
            if SHOW:
                plt.imshow(np.flip(x_test[n].T,axis=0), cmap='gray')
                plt.title(gt+" "+bs)
                plt.show()       
                print("GT:",gt,"recognized",bs)
            print("best path",best_path)
            # Edit_dist = editdistance.eval(gt,bs)
            # FP += Edit_dist  
            TOTAL += len(gt)       
            # if Edit_dist !=0:
                # String_FP +=1
        print("FP",FP)
        print("TOTAL characters",TOTAL)
        print("String FP",String_FP)
        print("TOTAL strings",N_test_seq)
        print('Recognition ended successfully, Character Error Rate = ',FP/TOTAL*100,'%')
        print('                              , Character Recognition Rate = ',(1-FP/TOTAL)*100,'%')
        print('                              , String Error Rate    = ',String_FP/N_test_seq*100,'%')
        print('                              , String Recognition Rate    = ',(1-String_FP/N_test_seq)*100,'%')
