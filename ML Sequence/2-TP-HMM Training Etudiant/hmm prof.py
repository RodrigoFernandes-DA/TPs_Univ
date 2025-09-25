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

# computes the Log (Sum(P(Xi)) when we pass the LogP(X) and not X
def logsumexp(X):
    X_max = max(X)
    if isinf(X_max):
        return -float('inf')

    acc = 0
    for i in range(X.shape[0]):
        acc += exp(X[i] - X_max)

    return log(acc) + X_max

# computes Log(P(a)+P(b)) when we pass LogP(a) and LogP(b) and not a and b
def logaddexp(a, b):

    if isinf(a) and a < 0:
        return b
    elif isinf(b) and b < 0:
        return a
    else:
        return max(a, b) + log1p(exp(-fabs(a - b)))
    
class HMM: # initialisation d'un HMM Gauche Droite discret
    def __init__(self,n_states,n_clusters,dir_name="none",file_name="none"):   # pour un HMM discret
        self.log_epsilon = -14 # natural logarithm on a value lower than 1e-6
        rng = np.random.default_rng()
        if file_name == "none":
            self.n_states = n_states
            self.n_clusters = n_clusters

            self.A = np.identity(n_states)/2
            for i in range(self.n_states-1):
                self.A[i,i+1] = 0.5        

            # initialisation déterministe max d'entropie
            #self.B = np.ones((n_states,n_clusters))/n_clusters 
            
            # initialisation aléatoire uniforme
            B = rng.random((self.n_states,self.n_clusters))
            den = np.sum(B,axis=1)
            DEN = np.repeat(den,self.n_clusters).reshape(self.n_states,self.n_clusters)
            self.B = B/DEN
            
            self.log_A = np.log(self.A)
            self.log_B = np.log(self.B)
        else:
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
    ###########################################################################
    def forward(self,train_data):
        T = train_data.shape[0]
        fwdlattice = np.zeros((T,self.n_states))
        
        fwdlattice[0, :] = self.log_Pi + self.log_B[:,train_data[0]]
        
        for t in range(1, T):
            for j in range(self.n_states):
                work_buffer = fwdlattice[t - 1, :] + self.log_A[:,j]               
                fwdlattice[t, j] = logsumexp(work_buffer) + self.log_B[j,train_data[t]]
            
        return fwdlattice

    ###########################################################################
    def backward(self,train_data): 
        T = train_data.shape[0]
        bwdlattice = np.zeros((T,self.n_states))
           
        bwdlattice[T-1, :] = self.log_Pf

        for t in range(T-2, -1,-1):
            for i in range(self.n_states):
                work_buffer = bwdlattice[t+1,:] + self.log_B[:,train_data[t+1]] + self.log_A[i,:]
                bwdlattice[t, i] = logsumexp(work_buffer)
            
        return bwdlattice

    ###########################################################################   
    def cumulate_B_num(self,train_data,fwd,bwd, sum_B_num,log_likelihood):
        T = train_data.shape[0]
        work_buffer = np.full((self.n_states,self.n_clusters),float('-inf'))
        
        work_buffer[:,train_data[0]] = fwd[0,:] + bwd[0,:]

        for t in range(1,T-1): # we miss T-1 because we need to be consistent 
            c = train_data[t]  # with cumulate_den()
            buffer = fwd[t,:] + bwd[t,:]
            for i in range(self.n_states):
                work_buffer[i,c] = logaddexp(work_buffer[i,c], buffer[i] )
        
        for i in range(self.n_states):
            for c in range(self.n_clusters):
                sum_B_num[i,c] = logaddexp(sum_B_num[i,c], work_buffer[i,c] 
                                        - log_likelihood)
     
    ###########################################################################
    def cumulate_A_num(self,train_data,fwd_lattice,
                       bwd_lattice,sum_A_num, log_likelihood):
        T = train_data.shape[0] 
        
        fwd_l = np.repeat(fwd_lattice[0,:],self.n_states).reshape(
            self.n_states,self.n_states)
        bwd_l = np.repeat(bwd_lattice[1,:],self.n_states).reshape(
            self.n_states,self.n_states)
        frame_ll = np.repeat(self.log_B[:,train_data[1]],self.n_states).reshape(
            self.n_states,self.n_states)
        
        work_buffer = fwd_l + self.log_A + frame_ll.T + bwd_l.T
        
        for t in range(1,T-1):
            fwd_l = np.repeat(fwd_lattice[t,:],self.n_states).reshape(
                self.n_states,self.n_states)
            bwd_l = np.repeat(bwd_lattice[t+1,:],self.n_states).reshape(
                self.n_states,self.n_states)
            frame_ll = np.repeat(self.log_B[:,train_data[t+1]],self.n_states).reshape(
                self.n_states,self.n_states)
            
            buffer = fwd_l + self.log_A + frame_ll.T + bwd_l.T
    
            for i in range (self.n_states):
                for j in range(self.n_states):
                    work_buffer[i,j] = logaddexp(work_buffer[i,j],buffer[i,j]) 
        
        for i in range (self.n_states):
            for j in range(self.n_states):
                sum_A_num[i,j] = logaddexp(sum_A_num[i,j],
                                           work_buffer[i,j] - log_likelihood) 

    ###########################################################################  
    def cumulate_den(self,T,fwd,bwd, sum_den,log_likelihood):  
        
        work_buffer = fwd[0,:] + bwd[0,:]
        
        for t in range(1,T-1):
            buffer = fwd[t,:] + bwd[t,:]
                
            for i in range(self.n_states):
                work_buffer[i] = logaddexp(work_buffer[i], buffer[i])
        
        for i in range(self.n_states):
            sum_den[i] = logaddexp(sum_den[i], work_buffer[i] - log_likelihood)
            
    ###########################################################################            
    def Mstep(self,sum_B_num, sum_den, sum_A_num):
        
        SUM_DEN_A = np.repeat(sum_den,self.n_states).reshape(
            self.n_states,self.n_states)
        SUM_DEN_B = np.repeat(sum_den,self.n_clusters).reshape(
            self.n_states,self.n_clusters)
        
        self.log_A = sum_A_num - SUM_DEN_A
        self.log_B = sum_B_num - SUM_DEN_B
        
        # impossible observations on the train dataset (not seen on train) 
        # recieve a non zero probability to allow the model to run on valid and test
        # where unseen observations during training may occur
        self.log_B[np.equal(self.log_B,float('-inf'))] = self.log_epsilon
        

    ###########################################################################
    def Viterbi(self,data,BACKTRACK = False):
        T = data.shape[0]

        Viterbi_lattice = np.zeros((T,self.n_states))
        work_buffer = np.zeros(self.n_states)
        
        best_path = []
        if BACKTRACK:
            Path_lattice = np.zeros((T,self.n_states))
            best_path = np.zeros(T,dtype=int)
        
        Viterbi_lattice[0, :] = self.log_Pi + self.log_B[:,data[0]]
        for t in range(1, T):
            for j in range(self.n_states):
                work_buffer = Viterbi_lattice[t - 1, :] + self.log_A[:,j]
                if t == T-1:
                    work_buffer += self.log_Pf[j]
                    
                Viterbi_lattice[t, j] = np.max(work_buffer) + self.log_B[j,data[t]]
                if BACKTRACK:
                    Path_lattice[t,j] = np.argmax(work_buffer)
        
        LL_best_path = np.max(Viterbi_lattice[T-1, :])
        if BACKTRACK:
            best_path[T-1] = np.argmax(Viterbi_lattice[T-1, :])
            # then backtrack the best path
            for t in range(T-2,-1,-1):
                best_path[t] = Path_lattice[t+1,best_path[t+1]]
            
        return LL_best_path, best_path

    ###########################################################################
    # compute frames log Likelihood for every training or validation samples
    def Log_likelihood(self,data):     
        LL = 0.0
        (n_sample,T) = data.shape
                    
        #######################################################################      
        # LOOP for each sample of the class starts here
        for n in range(n_sample):
            ##################################################################
            # Compute forward variables 
            fwd_lattice = self.forward(data[n,:])# frame_log_lik[n,:,:])

            log_likelihood = logsumexp(fwd_lattice[T-1,:] + self.log_Pf)
            LL += log_likelihood 
            del fwd_lattice   

        return LL
    ###########################################################################
    def One_sample_log_likelihood(self,data):    
        T = np.shape(data)[0]
        #######################################################################
        # Compute forward variables 
        fwd_lattice = self.forward(data)
        log_likelihood = logsumexp(fwd_lattice[T-1,:] + self.log_Pf)
        del fwd_lattice
        
        return log_likelihood

    ###########################################################################
    # Training a HMM
    def TrainFB(self,classe,train_data,valid_data,Max_EM_iter,dir_name,file_name):
        #######################################################################
        #   Intialization
        #   Create the random model for the class considered        
        #   Create a n_states, n_vocab discrete observations left-right HMM
        print("hmm train with Forward Backward",classe,"...")
 
        n_train = np.shape(train_data)[0]
        n_valid = np.shape(valid_data)[0]
        T = np.shape(train_data)[1]
        
        #######################################################################
        ## EM LOOP start here for training one single model
        ##
        LL_train_history = np.zeros(Max_EM_iter)
        LL_valid_history = np.zeros(Max_EM_iter)
        LL_train = -1e+10 
        LL_valid_prev = -1e+10#
        
        iteration = 0
        best_iteration = 0
        
        for iteration in tqdm(range(Max_EM_iter)):
            LL_train_history[iteration] = LL_train
            LL_train = 0.0
            ###################################################################
            # Create the variables required for the EM STEPS
            sum_B_num = np.full((self.n_states,self.n_clusters),float('-inf'))
            sum_A_num = np.full((self.n_states,self.n_states),float('-inf'))
            sum_den = np.full(self.n_states,float('-inf'))

            ###################################################################
            # START EM
            ###################################################################    
            # LOOP for each sample of the class starts here
            for n in range(n_train):
                ###############################################################
                # Compute forward variables 
                fwd_lattice = self.forward(train_data[n,:])

                ###############################################################
                # Compute backward variables
                bwd_lattice = self.backward(train_data[n,:])
 
                log_likelihood = logsumexp(fwd_lattice[T-1,:]+bwd_lattice[T-1,:])
                LL_train += log_likelihood 
                
                ###############################################################
                # acumulate the statistics required for denominator of B and A 
                self.cumulate_den(train_data[n,:].shape[0],
                                  fwd_lattice,bwd_lattice,sum_den,log_likelihood)
                ###############################################################
                # acumulate the statistics required for numerator of B 
                self.cumulate_B_num(train_data[n,:],fwd_lattice,
                                    bwd_lattice,sum_B_num,log_likelihood)
                
                ###############################################################
                # acumulate the  statistics required for numerator of A
                self.cumulate_A_num(train_data[n,:],fwd_lattice,
                                    bwd_lattice,sum_A_num, log_likelihood)
                del(fwd_lattice)
                del(bwd_lattice)

                ###############################################################
                # End loop for E STEP over n_samples
            
            LL_train_history[iteration] = LL_train
            
            ###################################################################
            # call reestimation M_STEP  here
            self.Mstep(sum_B_num, sum_den, sum_A_num)
            
            del(sum_B_num)
            del(sum_den)
            del(sum_A_num)
            
            LL_valid = self.Log_likelihood(valid_data)
            LL_valid_history[iteration] = LL_valid

            if LL_valid > LL_valid_prev:
                # dump the model to file
                self.dump_model(dir_name,file_name,"log")
                LL_valid_prev = LL_valid
                best_iteration = iteration
                
            ###################################################################
            # End EM loop 
        
        #LL_train_history à t-1    et    LL_valid_history à t
        LL_train_history = np.roll(LL_train_history, -1)
        LL_train_history[Max_EM_iter-1] =  LL_train_history[Max_EM_iter-2]
        
        print("Best_iteration:",best_iteration,"best LL_valid:",
              LL_valid_history[best_iteration]/n_valid)
        return LL_train_history/n_train, LL_valid_history/n_valid, best_iteration 
    #####################################################
    def MstepViterbi(self,sum_B_num, sum_den, sum_A_num):
        # sum_A_num[np.equal(sum_A_num,0)] = 1
        # sum_B_num[np.equal(sum_B_num,0)] = 1
        
        SUM_DEN_A = np.repeat(sum_den,self.n_states).reshape(
            self.n_states,self.n_states)
        
        SUM_DEN_B = np.repeat(sum_den,self.n_clusters).reshape(
            self.n_states,self.n_clusters)
        # SUM_DEN_B[np.equal(SUM_DEN_B,0)] = 1

        
        self.A = sum_A_num / SUM_DEN_A
        self.B = sum_B_num / SUM_DEN_B

        self.log_A = np.log(self.A)
        self.log_B = np.log(self.B)
        
    ###########################################################################
    # Training a HMM
    def TrainViterbi(self,classe,train_data,valid_data,Max_EM_iter,dir_name,file_name):
        #######################################################################
        #   Intialization
        #   Create the random model for the class considered        
        #   Create a n_states, n_vocab discrete observations left-right HMM
        print("hmm train with Viterbi ",classe,"...")
    
        n_train = np.shape(train_data)[0]
        n_valid = np.shape(valid_data)[0]
        T = np.shape(train_data)[1]
        
        #######################################################################
        ## EM LOOP start here for training one single model
        ##
        LL_train_history = np.zeros(Max_EM_iter)
        LL_valid_history = np.zeros(Max_EM_iter)
        LL_train = -1e+10 
        LL_valid_prev = -1e+10#
        
        iteration = 0
        best_iteration = 0
        
        for iteration in tqdm(range(Max_EM_iter)):
            LL_train_history[iteration] = LL_train
            LL_train = 0.0
            ###################################################################
            # Create the variables required for the EM STEPS
            sum_B_num = np.ones((self.n_states,self.n_clusters),dtype=int)
            sum_A_num = np.ones((self.n_states,self.n_states),dtype=int)
            sum_den = np.ones(self.n_states,dtype=int)
    
            ###################################################################
            # START EM
            ###################################################################    
            # LOOP for each sample of the class starts here
            for n in range(n_train):
                LL, best_path = self.Viterbi(train_data[n,:],True)
                LL_train += LL 
                # print("train_data:",train_data[n,:].shape[0])
                # print("best path:",best_path)
                # print("best path:",best_path.T.shape[0])
                T = train_data[n,:].shape[0]
                for t in range(T-1):
                    sum_B_num[best_path[t],train_data[n,t]] +=1
                    sum_den[best_path[t]] += 1
                    sum_A_num[best_path[t],best_path[t+1]] +=1

                sum_B_num[best_path[T-1],train_data[n,T-1]] +=1
                sum_den[best_path[T-1]] += 1

                ###############################################################
                # End loop for E STEP over n_samples
            
            LL_train_history[iteration] = LL_train
            
            ###################################################################
            # call reestimation M_STEP  here
            self.MstepViterbi(sum_B_num, sum_den, sum_A_num)
            
            del(sum_B_num)
            del(sum_den)
            del(sum_A_num)
            
            LL_valid = 0
            for n in range(n_valid):
                LL, best_path = self.Viterbi(valid_data[n,:])
                LL_valid += LL 
            
            LL_valid_history[iteration] = LL_valid
    
            if LL_valid > LL_valid_prev:
                # dump the model to file
                self.dump_model(dir_name,file_name,"log")
                LL_valid_prev = LL_valid
                best_iteration = iteration
                
            ###################################################################
            # End EM loop 
        
        #LL_train_history à t-1    et    LL_valid_history à t
        LL_train_history = np.roll(LL_train_history, -1)
        LL_train_history[Max_EM_iter-1] =  LL_train_history[Max_EM_iter-2]
        
        print("Best_iteration:",best_iteration,"best LL_valid:",
              LL_valid_history[best_iteration]/n_valid)
        return LL_train_history/n_train, LL_valid_history/n_valid, best_iteration 
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
###############################################################################
def Get_data_class(data_discretes,y,classe,size):
    M = np.shape(data_discretes)[0]
    print(M)
    D = 28
    N = M//D
    classe_data = np.zeros((size,D),dtype='int')
    n_train = 0
    for n in range(N):
        if y[n]==classe:
            classe_data[n_train,:] = data_discretes[n*D:(n+1)*D]
            n_train +=1
            
    return classe_data
