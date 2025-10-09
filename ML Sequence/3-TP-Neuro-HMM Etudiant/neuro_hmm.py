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
import random

import torch
from torch import nn

#from torch.utils.data import DataLoader
# from torch.utils.data import random_split
# from torchvision import transforms

import mlp

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
    
class Neuro_HMM: # initialisation d'un HMM Gauche Droite discret
    def __init__(self,n_states,TRAIN = False, dir_name="none",file_name="none"):   # pour un HMM discret
        self.log_epsilon = -14 # natural logarithm lower than 1e-6
        self.n_states = n_states
        
        if file_name == "none":
            
            self.A = np.identity(self.n_states)/2
            for i in range(self.n_states-1):
                self.A[i,i+1] = 0.5        

            # initialisation max d'entropie
            try:
                self.log_A = np.log(self.A)
            except RuntimeWarning:
                print("log 0 = -inf...")
        else:
            model_file = open(dir_name+"/"+file_name,'rb')
            mode = pickle.load(model_file)
            if mode == "log":
                self.log_A = pickle.load(model_file)

            elif mode == "dec":
                self.A = pickle.load(model_file)
            model_file.close()

        
        #### cette partie est fixe car on travaille exclusivement sur 
        ###  des modèles Gauche Droite dont les probab des états initiaux 
        ###  et finaux sont 1 ou 0
        self.Pi = np.zeros((self.n_states,1))
        self.Pf = np.zeros((self.n_states,1))

        self.log_Pi = [float('-inf') for i in range(self.n_states)]  # proba initiales
        self.log_Pf = [float('-inf') for i in range(self.n_states)]  # proba finales
        self.Pi[0] = 1                  # pour un modèle gauche droite
        self.Pf[self.n_states - 1] = 1       # idem
        self.log_Pi[0] = 0.0
        self.log_Pf[self.n_states-1] = 0.0
        
        self.TRAIN = TRAIN
        # if training create the variables required to cumulate the statistics during E STEP
        if TRAIN:
            self.sum_A_num = np.full((self.n_states,self.n_states),float('-inf'))
            self.sum_den = np.full(self.n_states,float('-inf'))
    ###################################################################################
    def reset_statistics(self):
        if self.TRAIN:
            self.sum_A_num[:,:] = -float('inf')
            self.sum_den[:] = -float('inf')
    ###########################################################################
    # dump the model to a file A and B probability Matrices
    # assuming initial and final states are known for Left right models
    def dump_model(self,dir_name,file_name,mode):
        model_file = open(dir_name+"/"+file_name,'wb')
        pickle.dump(mode,model_file)
        if mode == "log":
            pickle.dump(self.log_A,model_file)
        elif mode =="dec":
            pickle.dump(self.A,model_file)
        model_file.close() 

    ###########################################################################
    def forward(self,T,log_B):
        fwdlattice = np.zeros((T,self.n_states))
        fwdlattice[0, :] = self.log_Pi + log_B[0,:]
        for t in range(1, T):
            for j in range(self.n_states):
                work_buffer = fwdlattice[t - 1, :] + self.log_A[:,j]               
                fwdlattice[t, j] = logsumexp(work_buffer) + log_B[t,j]
        return fwdlattice

    ###########################################################################
    def backward(self,T,log_B): 
        bwdlattice = np.zeros((T,self.n_states))
        bwdlattice[T-1, :] = self.log_Pf
        for t in range(T-2, -1,-1):
            for i in range(self.n_states):
                work_buffer = bwdlattice[t+1,:] + log_B[t+1,:] + self.log_A[i,:]
                bwdlattice[t, i] = logsumexp(work_buffer)
        return bwdlattice
    ###########################################################################  
    def gamma(self,T,fwd,bwd,log_likelihood):  
        log_posteriors = np.zeros((T,self.n_states))
        for t in range(T-1):
            for s in range(self.n_states):
                log_posteriors[t,s] = fwd[t,s] + bwd[t,s]
            log_posteriors[t,:] -= logsumexp(log_posteriors[t,:])
        pred= np.exp(log_posteriors)
        return pred

    ###########################################################################
    def Viterbi(self,data,my_mlp,BACKTRACK = False):
        B = my_mlp.pred_on_seq(data,self.n_states)
        log_B = np.log(B)
        
        T = data.shape[0]
        Viterbi_lattice = np.zeros((T,self.n_states))
        work_buffer = np.zeros(self.n_states)
        
        if BACKTRACK:
            Path_lattice = np.zeros((T,self.n_states))
            best_path = np.zeros(T,dtype=int)
        
        Viterbi_lattice[0, :] = self.log_Pi + log_B[0,:]
        for t in range(1, T):
            for j in range(self.n_states):
                work_buffer = Viterbi_lattice[t - 1, :] + self.log_A[:,j]
                if t == T-1:
                    work_buffer += self.log_Pf[j]
                
                Viterbi_lattice[t, j] = np.max(work_buffer) + log_B[t,j]

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
# create the sequence HMM of the gt sequene of symbols 
# using the symbols HMM Models
def GT_HMM(n_states,gt,Models):
    GT_Model = Neuro_HMM(n_states*len(gt),TRAIN=True)
    
    for symbol in range(len(gt)):
        # copy the transition matrix
        start = symbol*n_states
        end = start + n_states
        
        GT_Model.log_A[start : end, start : end] = Models[int(gt[symbol])].log_A
        
        if symbol < len(gt)-1:
            # transition from each symbol possible with prob = 0.5
            GT_Model.log_A[end-1,end-1] = np.log(0.5)
            GT_Model.log_A[end-1,end] = np.log(0.5)
    return GT_Model
###########################################################################
# create the Ergodic sequence HMM 
# using the symbols HMM Models
def Ergodic_HMM(config,Models):
     
    Ergodic_Model = Neuro_HMM(config['N_classes']*config['n_states'])
    for symbol in range(config['N_classes']):
        # copy the transition matrix
        start = symbol*config['n_states']
        end = start + config['n_states']
        
        Ergodic_Model.log_A[start : end, start : end] = Models[symbol].log_A
        for next_symbol in range(config['N_classes']): 
            start_next = next_symbol*config['n_states']

            Ergodic_Model.log_A[end-1, start_next] = np.log(1 - np.exp(Ergodic_Model.log_A[end-1, end-1])/config['N_classes'])
        
        Ergodic_Model.log_Pi[symbol*config['n_states']] = np.log(1/config['N_classes'])
        Ergodic_Model.log_Pf[symbol*config['n_states']+config['n_states']-1] = np.log(1/config['N_classes'])
        
    return Ergodic_Model
    
###########################################################################
def GT_statesToSymbol_states(GT_Model,gt,state_posteriors, n_symbols):
    n_states = int(GT_Model.n_states / len(gt)) # n_states per symbol
    T =state_posteriors.shape[0]
    symbol_posteriors = np.zeros((T,n_states*n_symbols))
    
    for i in range(len(gt)):

        symbol_begin = int(gt[i]) * n_states
        symbol_end = symbol_begin + n_states

        state_begin = i*n_states
        state_end = state_begin + n_states
        
        symbol_posteriors[:,symbol_begin:symbol_end] = state_posteriors[:,state_begin:state_end]
                          
    return symbol_posteriors      
#######################################################################
def pred_on_batch(batch_list,list_gt,Models,my_mlp, config,iteration):

    pred =[] # list of return predictions
    LL_train = 0
    for n in range(len(batch_list)):
        T = batch_list[n].shape[0]
        gt = list_gt[n] # The GT sequence
        ###############################################################
        # build the HMM digit sequence model of the GT
        GT_Model = GT_HMM(config['n_states'],gt,Models)
        nn = config['n_states']*config['N_classes']
        if iteration == 0: # equiprobable initialisation 
            B = np.ones((T,nn))/nn
            #B = np.ones((T,nn))
        else:
            B = my_mlp.pred_on_seq(batch_list[n], config['N_classes']*config['n_states'])
            
        mask=[] 
        for s in range(len(gt)):
            begin = int(gt[s])*config['n_states']
            end = begin + config['n_states']
            mask = mask+[i for i in range(begin,end)] 
        B = B[:,mask]
        
        sum_prob = np.sum(B,axis=1)
        for t in range(T):
            B[t,:] /=sum_prob[t]
   
        try:
            log_B = np.log(B)
        except RuntimeWarning:
            print("log 0 = -inf...")
        ###############################################################
        # Compute forward variables 
        fwd_lattice = GT_Model.forward(T,log_B)

        ###############################################################
        # Compute backward variables
        bwd_lattice = GT_Model.backward(T,log_B)

        log_likelihood = logsumexp(fwd_lattice[T-1,:]+bwd_lattice[T-1,:])

        LL_train += log_likelihood 
        ###############################################################
        # Compute the gamma posteriors
        gamma = GT_Model.gamma(T,fwd_lattice,bwd_lattice,log_likelihood)

        # then distribute posterior probabilities of the GT states 
        #  to the symbol states involved in the current GT
        symbol_states_posteriors = GT_statesToSymbol_states(GT_Model,gt,
                                                            gamma, config['N_classes'])

        pred.append(symbol_states_posteriors)

        del GT_Model
        del fwd_lattice
        del bwd_lattice
        del gamma
        
    return pred, LL_train

#################################################
def Concat(data,pred):
    n_sample = len(data)

    DATA = np.array(data[0])
    PRED = np.array(pred[0])
        
    for n in range(1,n_sample):
        DATA = np.concatenate((DATA,data[n]))
        PRED = np.concatenate((PRED,pred[n]))
        
    return DATA, PRED

###########################################################################
def StatesToSymbols(best_path,n_states):
    T = best_path.shape[0]
    last_symbol = best_path[0]//n_states
    best_symbols = [last_symbol]
    begin = best_path[0]
    end = best_path[0]+n_states - 1

    for t in range(1,T):
        if best_path[t] < begin or best_path[t]>end:
            last_symbol = best_path[t]//n_states
            best_symbols.append(last_symbol)
            begin = best_path[t]
            end = best_path[t]+n_states - 1
    
    bs = ''.join([str(best_symbols[i]) for i in range(len(best_symbols))])
    
    return bs

###########################################################################  
def Embedded_Training(Models,train_data,valid_data,gt_train,gt_valid,config,Model_name):
    # Models     : la liste des modèle HMM
    # train_data : le tableau des données de train
    # valid_data : idem en valid
    print("neuro hmm embedded training",len(Models)," HMM models...")
    
    n_symbols = len(Models)
    n_train = len(train_data)
    n_valid = len(valid_data)
    
    #######################################################################
    ## EM LOOP start here for training one single model
    LL_train = -1e+10 
    LL_valid_prev = -1e+10#
    
    iteration = 0
    best_iteration = 0
    
    N_batch = int(config['N_train_seq'] / config['batch_size'])

    OPTIM = "ADAM" # "SGD" "ADAM"
    mlp_loss = []
    LL_train_history = []
    LL_valid_history =[]
    #########################################################
    #################  Instanciation du MLP #################
    my_mlp = mlp.MLP(config)
    my_loss = nn.CrossEntropyLoss()
    if OPTIM == "SGD":
        my_optimizer = torch.optim.SGD(my_mlp.parameters(), lr=config['learning_rate'])
    elif OPTIM == "ADAM":
        my_optimizer = torch.optim.Adam(my_mlp.parameters(), lr=config['learning_rate'])
    else:
        print("Optimiseur inconnu !!!!")
      
    # LOOP over epoch
    print("TRAIN !!!")
    for iteration in tqdm(range(config['num_epochs'])):
        LL_train = 0.0
 
        # shuffle the training data at each iteration
        l=list(zip(train_data,gt_train))
        random.shuffle(l)        
        train_data,gt_train = list(zip(*l))
        
        begin_batch = 0
        for batch in range(N_batch):
            ##################################################################
            #    Expectation STEP
            begin = begin_batch
            end = begin + config['batch_size']
            train_pred, LL_batch = pred_on_batch(train_data[begin:end],
                                                     gt_train[begin:end],
                                                     Models,
                                                     my_mlp, 
                                                     config,
                                                     iteration)
            begin_batch = end
            LL_train_history.append(LL_batch/config['batch_size'])
            ##################################################################
            #    Maximisation STEP
            DATA, PRED = Concat(train_data[begin:end],train_pred)
            mlp_loss.append(mlp.train_on_batch(DATA,PRED, my_mlp,my_loss, my_optimizer))
        ################################################################
        if iteration == 40:
            plt.plot(mlp_loss,color = 'blue', label = " MLP train loss")
            plt.legend()
            plt.title("Embedded training over batch iteration "+str(iteration))
            plt.show()
            
            plt.plot(LL_train_history,color = 'red', label = "HMM train LL")
            plt.legend()
            plt.title("Embedded training over batch iterations "+str(iteration))
            plt.show()

        ###############################################################
        valid_pred, LL_valid = pred_on_batch(valid_data,
                                              gt_valid,
                                              Models,
                                              my_mlp,  
                                              config,
                                              1)
        LL_valid_history.append( LL_valid/n_valid)

        if iteration == 40:
            plt.plot(LL_valid_history,color = 'blue', label = "HMM valid LL")
            plt.legend()
            plt.title("Embedded training over epochs "+str(iteration))
            plt.show()

        # Early Stopping
        if iteration == 0:
            best_valid_loss = LL_valid_history[0]
        else:
            if LL_valid_history[-1] > best_valid_loss:
                best_valid_loss = LL_valid_history[-1]
                # on mémorise ce modèle
                best_iteration = iteration
                torch.save(my_mlp.state_dict(), Model_name)
            
    ###################################################################
    # End EPOCH loop 

    print("Best_iteration:",best_iteration,"best LL_valid:",
          LL_valid_history[best_iteration])
    return LL_train_history, LL_valid_history, best_iteration 



