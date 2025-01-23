#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importação de bibliotecas necessárias
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import math
from numpy.linalg import norm

# Definição de cores e número de cores para visualização
colors = ['r', 'b', 'g', 'c', 'm', 'o']
n_colors = 6

#####################################################################

# Função para calcular log-sum-exp de forma numericamente estável
def logsumexp(X):
    # Obtém o valor máximo em X para estabilizar o cálculo
    X_max = max(X)
    if math.isinf(X_max):  # Se o valor for infinito, retorna -inf diretamente
        return -float('inf')

    # Acumula a soma das exponenciais de X - X_max
    acc = 0
    for i in range(X.shape[0]):
        acc += math.exp(X[i] - X_max)

    # Retorna o log da soma com o ajuste de X_max para estabilidade numérica
    return math.log(acc) + X_max

####################################################################

# Função para calcular a soma logarítmica da verossimilhança
def LogSumExp(Log_Vrais_Gauss):
    K, N = np.shape(Log_Vrais_Gauss)  # Dimensões da matriz de log-verossimilhança

    logsomme = np.zeros(N)
    for n in range(N):
        # Calcula a soma logarítmica para cada ponto de dados
        logsomme[n] = logsumexp(Log_Vrais_Gauss[:, n])
        
    return logsomme  # Retorna a soma logarítmica para cada ponto

####################################################################

# Geração de N amostras de uma mistura de Gaussianas
def my_GMM_generate(P, Mean, Cov, N, Visualisation=False):
    K, p = np.shape(Mean)  # Número de clusters e dimensão dos dados
    
    eff = np.asarray(N * P, dtype=int)  # Número de amostras por cluster
    # Gera amostras do primeiro cluster
    X = np.random.multivariate_normal(Mean[0, :], Cov[0, :, :], eff[0])
    y = [0 for i in range(eff[0])]

    # Gera amostras para clusters restantes e concatena
    for k in range(1, K):
        Xk = np.random.multivariate_normal(Mean[k, :], Cov[k, :, :], eff[k])
        X = np.concatenate((X, Xk), axis=0)
        yk = [k for i in range(eff[k])]
        y = np.concatenate((y, yk), axis=0)
    
    # Visualiza os dados, se especificado
    if Visualisation:
        plt.figure(figsize=(8, 8))
        debut = 0
        for k in range(K):
            fin = debut + eff[k]
            plt.plot(X[debut:fin, 0], X[debut:fin, 1], colors[k] + 'o', markersize=4, markeredgewidth=3)
            plt.plot(Mean[k, 0], Mean[k, 1], 'kx', markersize=10, markeredgewidth=3)
            debut = fin
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.show()

    return X, y  # Retorna as amostras e seus rótulos

##############################################################

# Calcula o log da verossimilhança para um ponto dado sua média e covariância
def my_G_LogVraisemblance(X, mean, cov):
    N, p = np.shape(X)
    
    covinv = np.linalg.inv(cov)  # Inverte a matriz de covariância
    det = np.linalg.det(cov)  # Determina o determinante da covariância
    log_factor = np.log((2 * np.pi) ** (p / 2) * math.sqrt(det))
    
    # Calcula o log da verossimilhança para cada ponto
    Res = X - mean
    Ex = -np.diag(Res @ covinv @ Res.T) / 2
    logvrais = Ex - log_factor
    
    return logvrais

##############################################################

# Função para inicializar parâmetros do modelo GMM
def my_GMM_init(X, K):
    N, p = np.shape(X)
    
    # Inicializa probabilidades a priori aleatoriamente e normaliza
    P = np.random.random_sample(K)
    P = P / np.sum(P)
    
    # Inicializa centros dos clusters escolhendo K exemplos aleatórios de X
    Index_init = np.random.choice(N, K, replace=False)
    Mean = np.zeros((K, p))
    for k in range(K):
        Mean[k, :] = X[Index_init[k], :]

    # Define os clusters a partir da distancia de cada centro
    Dist = np.zeros((K, N))
    for k in range(K):
        Dist[k, :] = np.square(norm(X - Mean[k, :], axis=1))
    y = np.argmin(Dist, axis=0)
    
    # Inicializa matrizes de covariância com base nos clusters iniciais
    Cov = np.zeros((K, p, p))
    for k in range(K):
        Cluster = X[y == k, :]
        Nk = np.shape(Cluster)[0]
        Res = Cluster - Mean[k, :]
        Cov[k, :, :] = Res.T @ Res / Nk
        
    return P, Mean, Cov  # Retorna probabilidades, médias e covariâncias

###########################################################################

# Função para calcular as probabilidades a posteriori
def my_GMM_p_a_posteriori(X, K, P, Mean, Cov):
    N, p = np.shape(X)
    Log_Vrais_Gauss = np.zeros((K, N))

    # Calcula log-verossimilhança para cada componente Gaussiano
    for k in range(K):
        Log_Vrais_Gauss[k, :] = math.log(P[k]) + my_G_LogVraisemblance(X, Mean[k, :], Cov[k, :, :])

    # Soma logarítmica das probabilidades
    LogDen = LogSumExp(Log_Vrais_Gauss)
    Proba_Clusters = np.exp(Log_Vrais_Gauss - LogDen)  # Probabilidades a posteriori
    LogVrais = np.sum(LogDen)

    return Proba_Clusters, LogVrais  # Retorna as probabilidades e log-verossimilhança total

###########################################################################

# Função para prever o cluster mais provável para cada ponto
def my_GMM_predict(X, K, P, Mean, Cov):
    Proba_Clusters, LogVrais = my_GMM_p_a_posteriori(X, K, P, Mean, Cov)
    y = np.argmax(Proba_Clusters, axis=0)  # Cluster mais provável para cada ponto
    return y, LogVrais  # Retorna as previsões e log-verossimilhança

##########################################################################

# Função para ajustar o modelo GMM aos dados usando Expectation-Maximization (EM)
def my_GMM_fit(X, K, Visualisation, Seuil=1e-7, Max_iterations=100):
    N, p = np.shape(X)

    # Inicializa parâmetros do modelo
    P, Mean, Cov = my_GMM_init(X, K)

    iteration = 0    
    LOGVRAIS = np.zeros(Max_iterations + 1)
    LOGVRAIS[0] = -100000
    
    # Loop principal do algoritmo EM
    while iteration < Max_iterations:
        iteration += 1

        # E-step: calcula as responsabilidades para cada ponto
        Proba_Clusters, LOGVRAIS[iteration] = my_GMM_p_a_posteriori(X, K, P, Mean, Cov)
        
        # Condição de parada: mudança relativa na log-verossimilhança
        if np.abs(LOGVRAIS[iteration] - LOGVRAIS[iteration - 1]) / np.abs(LOGVRAIS[iteration]) < Seuil:
            print("iteração =", iteration, "BREAK")
            break

        # M-step: atualiza os parâmetros do modelo com base nas responsabilidades
        Nk = np.sum(Proba_Clusters, axis=1)
        Mean = np.dot(Proba_Clusters, X) / Nk[:, None]
        
        for k in range(K):
            Res_gauche = (X - Mean[k, :]).T * Proba_Clusters[k, :]
            Res_droite = X - Mean[k, :]
            Cov[k, :, :] = (Res_gauche @ Res_droite) * np.identity(p) / Nk[k]
            
        P = Nk / N  # Atualiza probabilidades dos clusters
    
    return P, Mean, Cov, LOGVRAIS[1:iteration]

#############################################################################

# Código principal para geração de dados e ajuste do modelo GMM
if __name__ == '__main__':
    # Parâmetros para geração de dados sintéticos
    PROB = np.array([0.6, 0.2, 0.1, 0.1])
    MEAN = np.array([[0, 0], [5, 5], [-5, 0], [2, 2]])
    COV = np.array([[[2, 0], [0, 1]], [[5, 0], [0, 
