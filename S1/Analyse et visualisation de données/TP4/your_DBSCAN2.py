import numpy as np
from sklearn import datasets
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

colors = ['k', 'r', 'b', 'g', 'c', 'm']
n_colors = len(colors)

###########################################################################
def EpsilonVoisinage(i, X, Dist, eps):
    Voisins = [v for v in range(len(X)) if Dist[v, i] < eps]
    return Voisins

###########################################################################
def etendre_cluster(X, y, Dist, Cluster, no_cluster, Voisins, Visite, eps, minpts):
    for v in Voisins:
        if not Visite[v]:
            Visite[v] = True
            if y[v] == -1:
                y[v] = no_cluster
                Cluster.append(v)

            Voisins2 = EpsilonVoisinage(v, X, Dist, eps)
            if len(Voisins2) >= minpts:
                for vv in Voisins2:
                    if vv not in Voisins:
                        Voisins.append(vv)

    return Cluster, y, Visite

###########################################################################
def estime_EPS(Dist):
    # estimation du rayon du epsilon voisinage
    N = Dist.shape[0]
    Diag = np.eye(N)*1000
    EPS = np.percentile(np.min(Dist+Diag,axis=0),95)
    return EPS

def estime_MINPTS(X,Dist,eps):
    # estimation de minpts dans le epsilon voisinage
    NVoisins = []
    N,pp =np.shape(X)
    for p in range(N):
        NVoisins = NVoisins+[len(EpsilonVoisinage(p,X,Dist,eps))]
        MINPTS = math.ceil(np.percentile(np.asarray(NVoisins,dtype=np.float64),5))

    return MINPTS

###########################################################################
def my_DBSCAN(X, eps, minpts, Visualisation=False):
    N, pp = np.shape(X)
    no_cluster = 0
    Dist = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
    # eps = estime_EPS(Dist)
    # minpts = estime_MINPTS(X,Dist,eps)


    Visite = [False] * N
    y = -np.ones(N)  # tableau des labels des données, initialisé bruit (-1)
    Clusters = []

    for p in range(N):
        if not Visite[p]:
            Visite[p] = True
            Voisins = EpsilonVoisinage(p, X, Dist, eps)
            if len(Voisins) >= minpts:
                no_cluster += 1
                cluster = [p]
                y[p] = no_cluster
                cluster, y, Visite = etendre_cluster(X, y, Dist, cluster, no_cluster, Voisins, Visite, eps, minpts)
                Clusters.append(cluster)

    return y

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data

    eps_values = [0.3, 0.5, 0.7]
    minpts_values = [3, 5, 7]

    # fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    # axs = axs.flatten()

    # for i, minpts in enumerate(minpts_values):
    #     for j, eps in enumerate(eps_values):
    #         ax = axs[i * 3 + j]
    #         my_y = my_DBSCAN(X, eps, minpts)

    #         for k in range(1, len(set(my_y))):
    #             ax.scatter(X[my_y == k, 0], X[my_y == k, 1], color=colors[k % n_colors], label=f'Cluster {k}', marker='o')
    #         ax.scatter(X[my_y == -1, 0], X[my_y == -1, 1], color='k', label='Noise', marker='^')

    #         ax.set_title(f'DBSCAN: eps={eps}, minpts={minpts}')
    #         ax.legend()

    # # plt.tight_layout()
    # plt.tight_layout(h_pad=2.0)  # Increase vertical padding between subplots
    # plt.subplots_adjust(top=0.95)  # Optionally adjust the top spacing if needed
    # plt.show()



###########################################

    eps_values = [0.3, 0.5, 0.7]
    minpts_values = [5, 7, 10]

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # X, y = make_circles(n_samples=150, factor=0.3, noise=0.1)
    # X = StandardScaler().fit_transform(X)
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()

    for i, minpts in enumerate(minpts_values):
        for j, eps in enumerate(eps_values):
            ax = axs[i * 3 + j]
            my_y = my_DBSCAN(X, eps, minpts)
            statistiques = np.unique(my_y,return_counts=True)
            K = len(statistiques[0])-(1 if -1 in statistiques[0] else 0)
            Bruit = [p for p in range(len(my_y)) if my_y[p]==-1]
            
            if len(set(my_y)) == 1:
                a = 2
            else:
                a = len(set(my_y))

            for k in range(1, a):
                ax.scatter(X[my_y == k, 0], X[my_y == k, 1], color=colors[k % n_colors], label=f'Cluster {k}', marker='o')
            ax.scatter(X[my_y == -1, 0], X[my_y == -1, 1], color='k', label='Noise', marker='^')

            ax.set_title(f'DBSCAN: eps={eps}, minpts={minpts} ('+str(K)+' clusters, '+str(len(Bruit))+' noise)')

    # plt.tight_layout()
    plt.tight_layout(h_pad=2.0)  # Increase vertical padding between subplots
    plt.subplots_adjust(top=0.95)  # Optionally adjust the top spacing if needed
    plt.show()