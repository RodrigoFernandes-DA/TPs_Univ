#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# M1 Science et Ingénieurie des données
# Université de Rouen Normandie
# T. Paquet 2022 Thierry.Paquet@univ-rouen.fr

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

import numpy as np
import scipy as sp #right=True.
from operator import itemgetter, attrgetter

def TriVP(Valp,Vectp):
    # trie dans l'ordre décroisant les valeurs propres
    # en cas de valeurs propres complexes on trie  selon leu module
    liste1 = Vectp.tolist()
    liste2 = Valp.tolist()
    norme = np.abs(Valp)
    liste3 = norme.tolist()

    result = zip(liste1, liste2,liste3)
    result_trie =sorted(result,key =itemgetter(2), reverse=True)
    liste1, liste2, liste3 =  zip(*result_trie)
    Vectp = np.asarray(liste1)
    Valp = np.asarray(liste2)
    
    return Valp,Vectp

def Kernel(XC,kernel='linear',gamma=0,degre=3):
    # Calcule de la matrice de Gram, sélection du noyau
    # valeurs par défaut : 
    # rbf :gamma = 1/n
    # polynomial : degre = 3,    c=1
    n = XC.shape[1]
    m = XC.shape[0]
    if kernel == 'linear':
        K = XC @ XC.T
    elif kernel == 'rbf':
        # valeur par défaut comme dans scikitlearn
        if gamma == 0:
            gamma = 1/n
            
        K = np.ones((m,m))
        for i in range(m):
            for j in range(i+1,m):
                K[i,j] = np.exp(-np.linalg.norm(XC[i,:]-XC[j,:])**2 * gamma)
                K[j,i] = K[i,j]
    elif kernel =='poly':
        PS = XC @ XC.T + np.ones((m,m))
        K = np.power(PS,degre)
        
    return K

def myACP(X):
    n = X.shape[1]
    m = X.shape[0]
    moy = np.sum(X,0)/m # axe de la matrice selon lequel on somme
    np.reshape(moy,(n,1))

    # données centrées
    XC = X - moy.T
    
    # covariance
    S = XC.T @ XC / m

    # calcule des valeurs propres et vecteurs propres
    # vecteurs propres de norme 1 rangés en colonnes
    Valp, Vectp = np.linalg.eig(S) 

    # il faut ordonner dans l'ordre des valeurs propres décroissantes
    Valp,Vectp = TriVP(Valp,Vectp)

    # on projette sur les deux premiers axes principaux
    Projection = XC @ Vectp[:,:2]
    #print("my_ACP")
    #print(Vectp)
    
    #print("produits scalaires",Vectp.T @ Vectp)
    return Projection, Valp, Vectp

###############################################################################
def n_composantes(Val_P,p):
    total = np.sum(Val_P)
    somme = np.cumsum(Val_P/total)

    i = 0
    while somme[i]< p:
        i+= 1

    return somme, i
################################################
def approximation( X, Vect_P, n_p):
    # calcul de la moyenne
    m , d = X.shape

    moy = np.sum(X,0)/m
    XC = X - moy

    X_app = np.zeros((m,d))
    for i in range(m):
        # ajout des n valeurs projetées
        projections = (XC[i,:]@Vect_P[:,0:n_p]).reshape((1,n_p))
        projections = np.repeat(projections,d,axis =0)
        X_app[i,:]= moy + np.sum(projections*Vect_P[:,0:n_p],axis=1)
        
    return X_app

def myKernelPCA(X,n_components,kernel='linear',gamma=0,degre=3):
    n = X.shape[1]
    m = X.shape[0]
    moy = np.sum(X,0)/m # axe de la matrice selon lequel on somme
    np.reshape(moy,(n,1))

    # Etape 1: centrer les données
    XC = X - moy.T
    
    # Etape 2: calcule de la matrice de Gram, sélection du noyau
    K = Kernel(XC, kernel = kernel, gamma = gamma, degre = degre)
    print(K.shape)
    # Etape 3: centrage des produits scalaires
    UN = np.ones((m,m))/m
    Ktild = K - UN @ K - K @ UN + UN @ K @ UN
    
    # Etape 4: calcule les vecteurs propres de Ktild 
    Valp, Vectp = np.linalg.eig(Ktild) 

    # Etape 5: il faut ordonner dans l'ordre des valeurs propres décroissantes
    Valp,Vectp = TriVP(Valp,Vectp)
    
    # Etape 6: Extraction des coordonnées des deux premiers vecteurs propres dans l'espac de départ
    aj = Vectp[:,:n_components]
    
    # Etape 7: Normalisation de pour avoir des vecteurs propres de l'espace projeté soient normée
    for i in range(n_components):
        norm_aj = np.linalg.norm(aj[:,[i]])
        aj[:,[i]] = aj[:,[i]] / np.sqrt( Valp[i].real) / norm_aj
    
    # Etape 8: calcul des données projetées
    Y = aj.T @ Ktild
    
    return Y.T
    
if __name__ == '__main__':    

    p_var_exp = 0.95
    digits = datasets.load_digits(n_class=6)
    X = digits.data
    y = digits.target
    (N,d) = X.shape 
    print("Données MNIST N=",N,", d =",d)

    ###############################################################
    #              ACP simple 
    Y, Val_P, Vect_P = myACP(X.data)    
        
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
    plt.xlabel('CP 1')
    plt.ylabel('CP 2')
    plt.title('myACP des données MNIST')
    plt.show()

    ###############################################################
    somme, n_c_p = n_composantes(Val_P,p_var_exp)
    
    ###############################################################
    X_app = approximation( X, Vect_P, n_c_p)
    
    fig = plt.figure(2, figsize=(8, 6))
    plt.plot(somme,'ro')    
    plt.title('Variance expliquée à '+str(int(p_var_exp*100))+'%, '+str(n_c_p)+' composantes')
    plt.plot(somme[0:n_c_p],'bx')   
    plt.show() 
    
    fig = plt.figure(3, figsize=(6,14))    
    for i in range(3):
        plt.subplot(10,3,3*i+1)
        #plt.title(y[i])
        plt.imshow(1 - X[i].reshape((8,8)), cmap='gray')
        plt.subplot(10,3,3*i+2)
        plt.imshow(1-X_app[i].reshape((8,8)),cmap='gray')
        plt.subplot(10,3,3*i+3)
        plt.imshow((X[i]-X_app[i]).reshape((8,8)),cmap='gray')
        plt.show()


