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
def my_kmedoides(X,K,Visualisation=False,Seuil=0.001,Max_iterations = 100):
    
    N,p = np.shape(X)
    iteration = 0        
    Normes=np.zeros((K,N))

    J=np.zeros(Max_iterations+1)

    # Initialisation des clusters
    # par tirage de K exemples, pour tomber dans les données     
    Index_init = np.random.choice(N, K,replace = False)
    C = np.zeros((K,p))
    for k in range(K):
        C[k,:] = X[Index_init[k],:]


    while iteration < Max_iterations:
        print("Iteration :",iteration)
        #################################################################
        #          affectation des données aux médoïde le plus proches
        for k in range(K):
            Normes[k,:] = norm(X - C[k,:],axis=1)

        y = np.argmin(Normes,axis=0)
        
        # if Visualisation:
        #     fig = plt.figure(iteration+10, figsize=(8, 6))
        #     for k in range(K):
        #         plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o')
        #     plt.plot(C[0, :], C[1, :],'kx')
        #     plt.show()
        
        ################################################################
        # Calcul des meilleurs médoïdes
        for k in range(K):
            print("k :",k)
            Cluster = X[y==k,:]
            liste_dist = []
            Nk = Cluster.shape[0]
            print("Nk =",Nk)
            for i in range(Nk):
                D = Cluster[i,:]- Cluster #np.delete(Cluster, i,0)
                sum_dist = np.sum(np.sqrt(np.diag(D @ D.T)))
                liste_dist.append(sum_dist)
            
            min_dist = min(liste_dist)
            indice_min = liste_dist.index(min_dist) 
            C[k,:] = Cluster[indice_min,:]
            print("min_dist =",min_dist,"indice_mi =",indice_min)
            J[iteration] += min_dist
            
        if iteration > 0:
            if np.absolute(J[iteration-1]-J[iteration])/J[iteration-1] < Seuil:
                print(J[iteration-1])
                print(J[iteration])
                print("Stop value %.9f"%((J[iteration]-J[iteration-1])/J[iteration-1]))
                break
        iteration +=1
            
    if Visualisation:
        fig = plt.figure(figsize=(8, 6))
        plt.plot(J[1:iteration], 'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.show()
            
    return C, y, J[:iteration+1]


if __name__ == '__main__':

#########################################################
#''' K means '''
    iris = datasets.load_iris()
    X = iris.data#[:, :2]  # we only take the first two features.
    y = iris.target
    K= 3


    fig = plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[0:50, 0], X[0:50, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[0])
    plt.scatter(X[50:100, 0], X[50:100, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[1])
    plt.scatter(X[100:150, 0], X[100:150, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[2])
    
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend(scatterpoints=1)


    Cluster, y, Critere = my_kmedoides(X,K,Visualisation = False)
    
    
    fig = plt.figure(3, figsize=(8, 6))
    for k in range(K):
        plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o')
    plt.plot(Cluster[:,0], Cluster[:,1],'kx')
    plt.title('K medoïdes ('+str(K)+')')
    plt.show()
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(Critere, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Critère des k-médoïdes')
    plt.show()
    
    print("Critere:",Critere)
        
