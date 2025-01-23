#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# M1 Science et Ingénieurie des données
# Université de Rouen Normandie
# T. Paquet
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from numpy.linalg import norm

colors =['r','b','g','c','m','o']
n_colors = 6

###########################################################################
# initialisation ++ le kmeans
#
def initPlusPlus(X,K):
    N,p = np.shape(X)
    C = np.zeros((p,K))
    generator = np.random.default_rng()
    
    index = np.random.choice(N, 1,replace = False)
    liste_index = [index]
    C[:,0] = X[index,:]
    X = np.delete(X,index,0)
    print("k=0 C[k]=",C[:,0],"index=",index)
    k=1
    while k < K:
        #y = np.zeros(X.shape[0])
        NN = X.shape[0]
        dist = np.zeros(NN)
        for n in range(NN):
            D = C[:,:k] - np.repeat(X[n,:],k).reshape(p,k)
            D = np.diag(D @ D.T)
            #y[n] = np.argmin(D)
            dist[n] = np.min(D)

        # calcul des probabilités
        proba = dist/np.sum(dist)
        rand_value = generator.random((1))[0]
        intervals = np.cumsum(proba)
        index =0
        while index < NN:
            if intervals[index]> rand_value:
                break
            index += 1
        # tirage aléatoire selon proba
        C[:,k] = X[index,:]
        X = np.delete(X,index,0)
        print("k=",k,"C[k]=",C[:,k],"index=",index)
        k += 1

    return C

def my_kmeans(X,K,Visualisation=False,Seuil=0.001,Max_iterations = 100):
    
    N,p = np.shape(X)
    iteration = 0        
    Dist=np.zeros((K,N))

    J=np.zeros(Max_iterations+1)
    J[0] = 10000000
    # Initialisation des clusters
    # par tirage de K exemples, pour tomber dans les données     
    Index_init = np.random.choice(N, K,replace = False)
    # C = np.zeros((p,K))
    # for k in range(K):
    #     C[:,k] = X[Index_init[k],:].T  
        
    C = initPlusPlus(X,K)
    
    if Visualisation: 
        fig = plt.figure(3, figsize=(8, 6))

        plt.plot(X[:,0],X[:,1],'ro')
        plt.plot(C[0,:],C[1,:],'kx')
        plt.title('Initilisation k-means++ ('+str(K)+')')
        plt.show()
    
    while iteration < Max_iterations:
        iteration +=1
        #################################################################
        # E step : estimation des données manquantes 
        #          affectation des données aux clusters les plus proches
        for k in range(K):
            Dist[k,:] = np.square(norm(X - C[:,k],axis=1))

        y = np.argmin(Dist,axis=0)
        
#        for k in range(K):
#            Nk = np.shape(X[y==k,:])[0]
#            J[iteration] += Nk / N * (np.sum(np.min(Dist[k,y==k],axis=0))/Nk)  # Crirère somme des variances intra

        J[iteration] += np.sum(np.min(Dist[y,:],axis=0))/N # Critière variance intra totale
        
        if Visualisation:
            fig = plt.figure(iteration+10, figsize=(8, 6))
            for k in range(K):
                plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o')
            plt.plot(C[0, :], C[1, :],'kx')
            plt.show()
        
        ################################################################
        # M Step : calcul des meilleurs centres          
        for k in range(K):
            Cluster = X[y==k,:]
            C[:,k] = np.mean(Cluster,axis=0)

        if np.abs(J[iteration]-J[iteration-1])/J[iteration-1] < Seuil:
            break
            
    if Visualisation:
        fig = plt.figure(figsize=(8, 6))
        plt.plot(J[1:iteration], 'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.show()
            
    return C, y, J[1:iteration]


if __name__ == '__main__':

#########################################################
#''' K means '''
    iris = datasets.load_iris()
    digit = datasets.load_digits() 
    X = iris.data#[:, :2]  # we only take the first two features.
    y = iris.target
    K= 4

    # digits = datasets.load_digits() 
    # X = digits.data#[:, :2]  # we only take the first two features.
    # y = digits.target
    # K= 10
    
    # fig = plt.figure(2, figsize=(8, 6))
    # plt.clf()
    # plt.scatter(X[0:50, 0], X[0:50, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[0])
    # plt.scatter(X[50:100, 0], X[50:100, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[1])
    # plt.scatter(X[100:150, 0], X[100:150, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[2])
    
    # plt.xlabel('Sepal length')
    # plt.ylabel('Sepal width')
    # plt.legend(scatterpoints=1)


    Cluster, y, Critere = my_kmeans(X,K,Visualisation = False)
    
    
    fig = plt.figure(3, figsize=(8, 6))
    for k in range(K):
        plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o')
    plt.plot(Cluster[0, :], Cluster[1, :],'kx')
    plt.title('K moyennes ('+str(K)+')')
    plt.show()
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(Critere, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Evolution du critère')
    plt