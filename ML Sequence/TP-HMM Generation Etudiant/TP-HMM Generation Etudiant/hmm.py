###############################################################################
#
# Adaptation of the cython source code _hmmc.pyx from the hmmlearn toolkit
# compatible with the sckitlearn earlier version
# https://hmmlearn.readthedocs.io/en/latest/
#
# Thierry Paquet (c) 2019-2023 Université de Rouen Normandie
#
# 

from math import exp, log, log1p, isinf, fabs
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans

    
class HMM: # initialisation d'un HMM Gauche Droite discret
    def __init__(self,n_states,n_clusters,dir_name="none",file_name="none"):   # pour un HMM discret
        self.log_epsilon = -14 # natural logarithm lower than 1e-6
        if file_name == "none":
            self.n_states = n_states
            self.n_clusters = n_clusters

            self.A = np.identity(n_states)
            for i in range(self.n_states-1):
                self.A[i,i+1] = 1        

            # initialisation max d'entropie
            self.B = np.ones((n_states,n_clusters))/n_clusters 
        
            self.log_A = np.log(self.A)
            self.log_B = np.log(self.B)
        else: # Load a model from file
            model_file = open(dir_name+"/"+file_name,'rb')
            mode = pickle.load(model_file)
            if mode == "log":
                self.log_A = pickle.load(model_file)
                self.log_B = pickle.load(model_file)

            elif mode == "dec":
                self.A = pickle.load(model_file)
                self.B = pickle.load(model_file)

            model_file.close()

        #### cette partie est fixe car on travaille exclusivement sur 
        ###  des modèles Gauche Droite dont les probab des états initiaux 
        ###  et finaux sont 1 ou 0
        self.n_states = n_states
        self.n_clusters = n_clusters
        self.Pi = np.zeros((n_states,1))
        self.Pf = np.zeros((n_states,1))

        self.log_Pi = [float('-inf') for i in range(n_states)]  # proba initiales
        self.log_Pf = [float('-inf') for i in range(n_states)]  # proba finales
        self.Pi[0] = 1                  # pour un modèle gauche droite
        self.Pf[n_states - 1] = 1       # idem
        self.log_Pi[0] = 0.0
        self.log_Pf[n_states-1] = 0.0

    ###########################################################################
    def GenerateSamples(self,N_samples,length):
    
        seq_states = np.zeros((N_samples,length,),dtype=int) # cria tabelas vazias
        seq_observations = np.zeros((N_samples,length),dtype=int)      
        generator = np.random.default_rng() # inicializa generator
 
        Intervals_A = np.roll(np.cumsum(np.exp(self.log_A),axis=1),1) # transforma em log / transforma em matiz de intervalos
        Intervals_A[:,0] = 0
        Intervals_B = np.roll(np.cumsum(np.exp(self.log_B),axis=1),1)
        Intervals_B[:,0] = 0

        n=0
        while n < N_samples:
                    # generer sequence d'indices
            draw_state = generator.random((length,))
            
            for t in range(1,length-1):
                # draw the next state
                I = Intervals_A[seq_states[n,t-1],:] # get the distribuition
                # probabilities of current state knowing the previous one

                test = I[I<draw_state[t]] #get the values that pass the test
                seq_states[n,t] = np.size(test)-1

            if seq_states[n, length-2] > self.n_states -3:

                seq_states[n, length-1] = self.n_states -1

                draw_obs = generator.random((length,))
                for t in range(length-1):
                    I = Intervals_B[seq_states[n,t],:]
                    test = I[I<draw_obs[t]]
                    seq_observations[n,t] = np.size(test)-1
            n +=1
                
        return seq_states, seq_observations

    
    ###########################################################################
    def Viterbi(self,obs_seq):
        T = obs_seq.shape[0]
        Viterbi_lattice = np.zeros((T,self.n_states))
        work_buffer = np.zeros(self.n_states)
        
        Viterbi_lattice[0,:] = self.log_Pi + self.log_B[:,obs_seq[0]]
        for t in range(1, T):
            for j in range(self.n_states):
                work_buffer = Viterbi_lattice[t-1, :] +self.log_A[:,j]
                Viterbi_lattice[t,j] = np.max(work_buffer) + self.log_B[j,obs_seq[t]]

        Viterbi_lattice[T-1,:] += self.log_Pf

        LL_best_path = np.max(Viterbi_lattice[T-1,:])

        return LL_best_path
    
    ###########################################################################
    # dump the model to a file A and B probability Matrices
    # assuming initial and final states are known for Left right models
    def dump_model(self,dir_name,file_name,mode):
        model_file = open(dir_name+"/"+file_name,'wb')
        pickle.dump(mode,model_file)
        if mode == "log":
            pickle.dump(self.log_A,model_file)
            pickle.dump(self.log_B,model_file)
        elif mode =="dec":
            pickle.dump(self.A,model_file)
            pickle.dump(self.B,model_file)
        model_file.close()   
    
    
###############################################################################
def dump_codebook(codebook,file_name):
    print("dump_codebook",file_name,"...")
    model_file = open(file_name,'wb')
    pickle.dump(codebook,model_file)
    model_file.close()   
###############################################################################
def load_codebook(file_name):
    print("load_codebook",file_name,"...")
    model_file = open(file_name,'rb')
    codebook = pickle.load(model_file)
    model_file.close()
    return codebook
###############################################################################
        
###############################################################################
#                 Kmeans clustering to compute the codebook
# USAGE:
# 1- Compute a codebook store it with discretized training data
#                         or
# 2- Load a codebook and the discrete training data
# return: the discrete training data and the codebook (scikit learn KMeans class)
def Codebook(X, N, D, n_clusters, LOAD_CODEBOOK=False,VERBOSE=True):
    file_name = "Codebook_"+str(n_clusters)+"_"+str(N)

    if LOAD_CODEBOOK:
        codebook = load_codebook(file_name)

    else:
        XX = X[0].reshape((D,D))
        for n in range(1,N):
            XX = np.concatenate((XX, X[n].reshape((D,D))),axis=1)
        # plot the first five training samples
        plt.figure()
        plt.imshow(XX[:,0:5*D], cmap='gray')
        plt.show()
        
        print("Running kmeans: n_clusters =",n_clusters,"N_train = ",N)
        XX = np.transpose(XX)    
        codebook = KMeans(n_clusters, random_state=0).fit(XX)

        dump_codebook(codebook,file_name)
        print('codebook dumped',file_name,'...')
    
        if VERBOSE:
            centers = codebook.cluster_centers_
            XX_discretes = codebook.labels_
            ###################################################################
            # View the discretization effect of pixel columns
            approximation = np.zeros((D,5*D))
            for column in range(5*D):
                approximation[:,column] = centers[XX_discretes[column],:]
                
            plt.figure()
            plt.imshow(XX[:,0:5*D], cmap='gray')
            plt.show()
            plt.imshow(approximation[:,0:5*D], cmap='gray')
            plt.show()
        
    return codebook

###############################################################################
#      Discretize each observation to its nearest codebook centroid
def Discretize(X,N,codebook):
    # prepare  data
    D=28
    data = X[0].reshape((D,D))
    for n in range(N):
        data = np.concatenate((data, X[n].reshape((D,D))),axis=1)
    
    data = np.transpose(data[:,D:])
    data_discretes = codebook.predict(data)

    return data_discretes

