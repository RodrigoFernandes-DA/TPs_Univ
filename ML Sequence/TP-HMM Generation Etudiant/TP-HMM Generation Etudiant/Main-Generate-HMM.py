# -*- coding: utf-8 -*-

import numpy as np

# pour charger les données MNIST avec pytorch
# pip install python-mnist 
#from mnist import MNIST
from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
from tqdm import tqdm

import hmm

#color_names = list(mcolors.CSS4_COLORS)

D = 28           # 28 X 28
N_train = 10000
N_valid = 1000

n_clusters = 100  # number of clusters used to discretize each pixel columns
n_states = 15

N_classes = 10
N_samples = 10
length = D

dir_name = "Model_"+str(n_clusters)+"_"+str(n_states)

###########################################################################
print("Loading codebook: n_cluster =",n_clusters,"N_train = ",N_train)
file_name = "Codebook_"+str(n_clusters)+"_"+str(N_train)
codebook = hmm.load_codebook(file_name)

###########################################################################
print("Loading models: n_states =",n_states,"n_cluster =",n_clusters,"N_train = ",N_train)

models = []
suffix = "_"+str(n_clusters)+"_"+str(n_states)+"_"+str(N_train)+"_"+str(N_valid)
    
for classe in range(N_classes):
    file_name = 'class_'+str(classe)+suffix
    model = hmm.HMM(n_states,n_clusters,dir_name,file_name)
    models.append(model)

LL_scores = np.zeros((N_classes,))
print("Generation in progress...")

for classe in range(N_classes):
    seq_states, seq_observations = models[classe].GenerateSamples(N_samples,length)  
    N_samples = 10
    length = 28
    hmm.GenerateSamples(N_samples,length) 
    centers = codebook.cluster_centers_
    XX_discretes = codebook.labels_
    ###################################################################
    # View the corresponding image of the generated observation sequences
    image = np.zeros((D,N_samples*D))
    for sample in range(N_samples):
        for t in range(D):
            image[:,sample * D + t] = centers[seq_observations[sample,t],:]
        
        # ###################################################################
        # # Compute Loglikelihood of the best path with Viterbi
        # for c in range(N_classes):
        #     LL_scores[c] = models[c].Viterbi(seq_observations[sample])
        # print("\nLL_scores",LL_scores)
        # print("Classse:",classe,"meilleur model :",np.argmax(LL_scores),"LL:",np.max(LL_scores))
    plt.figure(figsize=(9,1.1))
    plt.imshow(image, cmap='gray')
    plt.show()
    

    
print("Generation ended successfully.")

