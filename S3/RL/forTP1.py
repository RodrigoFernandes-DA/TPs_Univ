#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:48:05 2020

@author: berar
"""

import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    
    def __init__(self):
        self.K=10
        np.random.seed(6)
        self.variance = np.random.randint(-1,2,size=self.K)
        self.biais = np.random.poisson(3,self.K)
        while sum(self.biais == self.biais.max()) != 1 :
            self.biais = np.random.poisson(3,self.K)
  
    def get_arm(self,k):   
        reward  = np.random.poisson(6+self.variance[k]) + self.biais[k]  
        return reward


if __name__ == "__main__":
    K=10
    T=500
    bandits = [[]]
    for k in range(K-1):
     bandits.append([]) 

    myBandit = Bandit()

    for t in range(T):
        k = np.random.randint(0,K)
        reward = myBandit.get_arm(k)
        print(k,reward)
        bandits[k].append(reward)
    

    fig = plt.figure()
    plt.violinplot(bandits)
    plt.show()
