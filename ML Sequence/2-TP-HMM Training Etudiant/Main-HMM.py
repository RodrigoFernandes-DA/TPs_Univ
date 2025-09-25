# -*- coding: utf-8 -*-

import numpy as np

# pour charger les données MNIST avec pytorch
# pip install python-mnist 
from mnist import MNIST
from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
from tqdm import tqdm
import os

import hmm

color_names = list(mcolors.CSS4_COLORS)

TRAINING = True        # Training if True Testing otherwise
FB_TRAINING = True     # Training with Forward Backward algorithm
LOAD_CODEBOOK = True   # will load a pre-trained kmeans when starting training
KMEANS_ONLY = False    # will run only kmeans if training is set to True

D = 28           # 28 X 28
n_clusters = 100  # number of clusters used to discretize each pixel columns
n_states = 15
Max_EM_iter = 20 # max number of iteration for EM
N_classes = 10
###############################################################################
# let's work with only N_train first samples of MNIST and N_valid samples
N_train = 10000
N_valid = 1000

dir_name = "Model_"+str(n_clusters)+"_"+str(n_states)

if TRAINING == True:
    ###########################################################################
    # load the digit MNIST dataset
    data = MNIST('./samples')
    X, y = data.load_training()
    X = np.asarray(X)
    y = np.asarray(y)

    #print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
    #print('labels: %s' % np.unique(y))
    #effectif = np.bincount(y)
    #print('Class distribution: %s' % effectif)
    N_max = X.shape[0]   
    if N_train + N_valid > N_max:
        print("Not enough data: N_train + N_valid >", N_max)
        sys.exit()
        
    train_population = np.bincount(y[range(N_train)])
    # print('Training set class distribution: %s' % train_population)

    codebook = hmm.Codebook(X, N_train, D, n_clusters, LOAD_CODEBOOK = LOAD_CODEBOOK)
    train_discretes = codebook.labels_
        
    if KMEANS_ONLY == True:
        sys.exit("Kmeans succesfully ended")
        
    # prepare validation data
    valid_population = np.bincount(y[range(N_max-N_valid,N_max)])
    print('Valid set class distribution: %s' % valid_population)
    valid_discretes = hmm.Discretize(X[N_max-N_valid:N_max],N_valid,codebook)
        
    #  LOOP TRAINING FOR TRAINING EVERY CLASSES
    suffix_file_name = "_"+str(n_clusters)+"_"+str(n_states)+"_"\
        +str(N_train)+"_"+str(N_valid)
    
    # check whether directory already exists
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Folder ",dir_name," created!")
            
    for classe in range(N_classes):
        file_name = "class_"+str(classe)+suffix_file_name
        #######################################################################
        # let's prepare the data for training
        # we need to have them put togather label wise
        print("Prepare training data for class ",classe, "with ",
              train_population[classe],"training samples")
        print("                                         with ",
              valid_population[classe],"valid samples")

        train_data = hmm.Get_data_class(train_discretes,y[:N_train],
                                   classe,train_population[classe]) 
        valid_data = hmm.Get_data_class(valid_discretes,y[N_max-N_valid:],
                                   classe,valid_population[classe])
        
        model = hmm.HMM(n_states,n_clusters)
                
        if FB_TRAINING:
            ##### TRAIN with FORWARD-BACKWARD
            LL_train_history, LL_valid_history, best_iteration = model.TrainFB(
                classe,train_data,valid_data,Max_EM_iter,dir_name,file_name)
        else:
            ##### TRAIN with VITERBI
            LL_train_history, LL_valid_history, best_iteration = model.TrainViterbi(
                classe,train_data,valid_data,Max_EM_iter,dir_name,file_name)
        
        plt.plot(range(1,Max_EM_iter), LL_valid_history[1:Max_EM_iter],
                 color = 'red', marker='o', label = str(classe)+" valid")
        plt.plot(range(1,Max_EM_iter), LL_train_history[1:Max_EM_iter],
                 color = 'blue', marker='o', label = str(classe)+" train")
        plt.title("Log Likelihood over EM iterations")
        plt.legend()
        plt.savefig(dir_name+'/class_'+str(classe)+'.png')
        plt.show()
        
        print('Training Ended successfully')

else:
    #  EVALUATION ON THE MNIST TEST DATASET
    # loading the test dataset
    ###########################################################################
    # load the digit MNIST dataset
    print("Loading MNIST test data...")
    data = MNIST('./samples')
    X_test, y_test = data.load_testing()
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    N_test = X_test.shape[0]

    D = 28
    effectif = np.bincount(y_test)
    print('Dimensions: %s x %s' % (X_test.shape[0], X_test.shape[1]))
    print('labels: %s' % np.unique(y_test))
    print('Class distribution: %s' % effectif)

    ###########################################################################
    print("Loading codebook: n_cluster =",n_clusters,"N_train = ",N_train)
    file_name = "Codebook_"+str(n_clusters)+"_"+str(N_train)
    codebook = hmm.load_codebook(file_name)
    print("Discretize test dataset...")
    test_discretes = hmm.Discretize(X_test,N_test,codebook)

    ###########################################################################
    print("Loading models: n_states =",n_states,"n_cluster =",
          n_clusters,"N_train = ",N_train)

    models = []
    suffix = "_"+str(n_clusters)+"_"+str(n_states)+"_"+str(N_train)+"_"\
        +str(N_valid)
    for classe in range(N_classes):
        file_name = 'class_'+str(classe)+suffix
        model = hmm.HMM(n_states,n_clusters,dir_name,file_name)
        models.append(model)
    
    LL = []
    TP = 0
    FP = 0
    # loop testing every test sample and counting True and False positive
    print("Recognition in progress...")
    for n in tqdm(range(N_test)):
        ll_max = -100000000000.0
        max_class = -1
        for classe in range(N_classes):
            ll = models[classe].One_sample_log_likelihood(
                test_discretes[n*D:(n+1)*D])  
            if ll > ll_max:
                ll_max = ll
                max_class = classe

        if y_test[n]==max_class:
            TP+=1
        else:
            FP+=1
    print('Evaluation ended successfully, Recognition rate = ',TP/N_test*100,'%')

