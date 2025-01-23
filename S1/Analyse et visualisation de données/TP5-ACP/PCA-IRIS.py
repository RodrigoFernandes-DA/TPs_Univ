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
    
    # on calcule la varience explique
    VarExp = Valp / np.sum(Valp)
    
    # on calcule la varience explique
    print("Variance (MyACP):", Valp)
    print("Variance expliquée (MyACP):", VarExp)
    print("Directions Propres (MyACP):")
    for i, vecteur in enumerate(Vectp.T):  # Chaque colonne est une direction propre
        print(f"Composante {i + 1}: {vecteur}")
    
    return Projection, VarExp

def myKernelPCA(X, kernel='linear', gamma=0, degre=3):
    """
    Implementation of Kernel PCA (ACP à noyaux)
    """
    n = X.shape[0]  # Number of samples

    # Step 1: Center the data (mean = 0)
    moy = np.mean(X, axis=0)
    XC = X - moy

    # Step 2: Compute the Kernel matrix (Gram matrix)
    K = Kernel(XC, kernel=kernel, gamma=gamma, degre=degre)

    # Step 3: Center the Kernel matrix
    ONE_N = np.ones((n, n)) / n
    K_centered = K - ONE_N @ K - K @ ONE_N + ONE_N @ K @ ONE_N

    # Step 4: Compute eigenvalues and eigenvectors of the centered Kernel matrix
    Valp, Vectp = np.linalg.eigh(K_centered)

    # Step 5: Sort eigenvalues and eigenvectors in descending order
    Valp, Vectp = TriVP(Valp, Vectp)

    # Step 6: Extract and normalize the first two eigenvectors
    indices = np.argsort(-Valp.real)  # Sort indices by descending eigenvalues
    Vectp = Vectp[:, indices]
    Valp = Valp[indices]

    # Normalize eigenvectors to project the data correctly
    for i in range(2):
        Vectp[:, i] /= np.sqrt(Valp[i].real)

    # Step 7: Project the data onto the first two components
    Y = Vectp[:, :2] * np.sqrt(Valp[:2])

    return Y.real


# def myKernelPCA(X, kernel='linear', gamma=0, degre=3):
#     n = X.shape[1]
#     m = X.shape[0]
#     moy = np.sum(X, 0) / m  # Moyenne pour centrer les données
#     moy = np.reshape(moy, (1, n))
    
#     # Étape 1: Centrage des données
#     XC = X - moy

#     # Étape 2: Calcul de la matrice de Gram (noyau sélectionné)
#     K = Kernel(XC, kernel=kernel, gamma=gamma, degre=degre)

#     # Étape 3: Centrage de la matrice de Gram
#     UN = np.ones((m, m)) / m
#     Ktild = K - UN @ K - K @ UN + UN @ K @ UN

#     # Étape 4: Décomposition en valeurs propres et vecteurs propres
#     Valp, Vectp = np.linalg.eigh(Ktild)  # eigh car matrice symétrique définie
#     Valp, Vectp = TriVP(Valp, Vectp)     # Trier les valeurs propres

#     # Étape 5: Extraction des deux premières composantes principales
#     aj = Vectp[:, :2]  # Les deux premiers vecteurs propres

#     # Étape 6: Normalisation des vecteurs propres
#     for i in range(2):
#         norm_aj = np.linalg.norm(aj[:, [i]])
#         aj[:, [i]] = aj[:, [i]] / np.sqrt(Valp[i].real) / norm_aj

#     # Étape 7: Projection des données dans le nouvel espace
#     Y = Ktild @ aj

#     return Y.real  # Retourne uniquement la partie réelle


# def myKernelPCA(X,kernel='linear',gamma=0,degre=3):
#     n = X.shape[1]
#     m = X.shape[0]
#     moy = np.sum(X,0)/m # axe de la matrice selon lequel on somme
#     np.reshape(moy,(n,1))

#     # Etape 1: centrer les données
#     XC = 
    
#     # Etape 2: calcule de la matrice de Gram, sélection du noyau
#     K = Kernel(XC, kernel = kernel, gamma = gamma, degre = degre)

#     # Etape 3: centrage des produits scalaires
#     UN = np.ones((m,m))/m
#     Ktild = 
    
#     # Etape 4: calcule les vecteurs propres de Ktild 
#     Valp, Vectp = np.linalg.eig(Ktild) 

#     # Etape 5: il faut ordonner dans l'ordre des valeurs propres décroissantes
#     Valp,Vectp = TriVP(Valp,Vectp)
    
#     # Etape 6: Extraction des coordonnées des deux premiers vecteurs propres dans l'espac de départ
#     aj = 
    
#     # Etape 7: Normalisation de pour avoir des vecteurs propres de l'espace projeté soient normée
#     for i in range(2):
#         norm_aj = np.linalg.norm(aj[:,[i]])
#         aj[:,[i]] = aj[:,[i]] / np.sqrt( Valp[i].real) / norm_aj
    
#     # Etape 8: calcul des données projetées
#     Y = 
    
#     return Y.T
    
if __name__ == '__main__':    

    #######################################################################
    ########### DATASET IRIS ############################
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    fig = plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[0:50, 0], X[0:50, 1], edgecolor='k',label=iris.target_names[0])
    plt.scatter(X[50:100, 0], X[50:100, 1], edgecolor='k',label=iris.target_names[1])
    plt.scatter(X[100:150, 0], X[100:150, 1], edgecolor='k',label=iris.target_names[2])
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend(scatterpoints=1)
    plt.show()

    # #######################################################################
    # ########### ACP simple ########### 
    # Y, VarExp = myACP(iris.data)    
    
    # # Représentation graphique des variances expliquées
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # # Scree plot (graphique des variances expliquées)
    # ax[0].bar(range(1, len(VarExp) + 1), VarExp, alpha=0.7, align='center')
    # ax[0].step(range(1, len(VarExp) + 1), np.cumsum(VarExp), where='mid', label='Variance cumulée')
    # ax[0].set_title('Variance expliquée par composante principale')
    # ax[0].set_xlabel('Composante principale')
    # ax[0].set_ylabel('Proportion de variance expliquée')
    # ax[0].legend()
    
    # # Visualisation des données projetées
    # ax[1].scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    # ax[1].set_xlabel('CP 1')
    # ax[1].set_ylabel('CP 2')
    # ax[1].set_title('myACP des données IRIS')

    # plt.tight_layout()
    # plt.show()
    
    
    # #######################################################################
    # ########### ACP ScikitLearn ########### 
    # # on vérifie nos résultats de scikitlearn
    # acp = PCA(n_components = 4, copy=True, iterated_power='auto', \
    #           random_state=None, svd_solver='full', tol=0.0, whiten=False)
    # YY = acp.fit_transform(iris.data)
    
    # # on obtient les directions propres 
    # directions = acp.components_
    
    # # on calcule la varience explique
    # VarExpSkit = acp.explained_variance_ratio_
    # cumulative_variance = np.cumsum(VarExpSkit)
    
    # print("Variance Expliquée (SckitLearn):", VarExp)
    # print("Directions Propres (SckitLearn):")
    # for i, direction in enumerate(directions):
    #     print(f"Composante {i + 1}: {direction}")
    
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # # Scree plot (gráfico das variâncias explicadas)
    # ax[0].bar(range(1, len(VarExpSkit) + 1), VarExpSkit, alpha=0.7, align='center')
    # ax[0].step(range(1, len(VarExpSkit) + 1), cumulative_variance, where='mid', label='Variance cumulée')
    # ax[0].set_title('Variance expliquée par composante principale')
    # ax[0].set_xlabel('Composante principale')
    # ax[0].set_ylabel('Proportion de variance expliquée')
    # ax[0].legend()

    # # ax[1].plt.clf()
    # ax[1].scatter(YY[:, 0], YY[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
    # ax[1].set_xlabel('CP 1')
    # ax[1].set_ylabel('CP 2')
    # ax[1].set_title('scikit ACP des données IRIS')
    # plt.tight_layout()
    # plt.show()
    
#    #######################################################################
#    # mon ACP à noyaux
#    #                    'linear'   'rbf'   'poly'
#    Y = myKernelPCA(iris.data,kernel='rbf')
#    
#    fig = plt.figure(1, figsize=(8, 6))
#    plt.clf()
#    plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
#    plt.xlabel('1st KPC')
#    plt.ylabel('2nd KPC')
#    plt.title('my kernelPCA (rbf) données IRIS')
#    plt.show()
#    
#    # on fait une kernel PCA avec scikit learn
#    kernelpca = KernelPCA(n_components=2, kernel='rbf')
#    Y = kernelpca.fit_transform(iris.data)
#    
#    fig = plt.figure(1, figsize=(8, 6))
#    plt.clf()
#    plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
#    plt.xlabel('1st KPC')
#    plt.ylabel('2nd KPC')
#    plt.title('scikit kernelPCA (rbf) données IRIS')
#    plt.show()

    #######################################################################
    # mon ACP à noyaux
    #                    'linear'   'rbf'   'poly'
    Y = myKernelPCA(iris.data, kernel='rbf')
    
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel('1st KPC')
    plt.ylabel('2nd KPC')
    plt.title('my kernelPCA (rbf) données IRIS')
    plt.show()
    
    # Kernel PCA avec scikit-learn
    kernelpca = KernelPCA(n_components=2, kernel='rbf')
    Y = kernelpca.fit_transform(iris.data)
    
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel('1st KPC')
    plt.ylabel('2nd KPC')
    plt.title('scikit kernelPCA (rbf) données IRIS')
    plt.show()
