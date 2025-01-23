from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from scipy.linalg import eigh
from sklearn import manifold
from scipy.sparse import diags, linalg

iris = datasets.load_iris()
X = iris.data
y = iris.target

# digits = datasets.load_digits()
# X = digits.data
# y = digits.target

N = X.shape[0]

"""
**Etape 1: Construire un graphe des voisins, pour chaque point 𝑋𝑖 , déterminer ses kPPV**
1. chaque nœud est connecté à ses kPPV
2. les arrêtes de poids non-nuls prennent comme valeurs la distance euclidienne entre les 2 nœuds
3. symétriser le graphe pour obtenir la matrice d’adjacence W
"""
n_neighbors = 4 # Aumentar o valor de K nao impacta muito o resultado. A distribuicao segue com mesma geometria, mas um pouco mais acentuado
kng = kneighbors_graph(X, n_neighbors,mode='distance')
kng.nnz

# simetrizar (pode aumentar o numero de vizinhos para k+(k-1))
W = 0.5*(kng+kng.T)  

plt.figure(figsize=(5,5))
plt.imshow(kng.todense())
# plt.imshow(kng.todense()[:50,:50])


plt.figure(figsize=(5,5))
plt.imshow(W.todense())

"""
**Etape 2: Calculer la matrice des degrés D**
"""
# matriz de degree ou matrice de adjacences (soma dos valores das colunas)
D = diags(np.asarray(W.sum(axis=0)).flatten())

"""
**Etape 3: Calculer la matrice Laplacienne**
"""
Laplacian = D-W

"""
**Etape 4: Déterminer les plus faibles valeurs propres non nulles de la matrice Laplacienne**
"""
[yl, YL] = linalg.eigsh(Laplacian, n_neighbors, which='SM')
print("Val propres Laplacian non  normalisé =  ", yl)

fig,ax = plt.subplots(figsize=(5,5))
scatter = ax.scatter(YL[:,2],YL[:,3], c = y[0:N],cmap = plt.cm.Set1)

# Si on normalise le laplacian
Dinv = linalg.inv(D)
Dinv = Dinv.sqrt()
NormLaplacian  =  Dinv @ Laplacian @ Dinv

[yln, YLn] = linalg.eigsh(NormLaplacian, n_neighbors, which='SM')
print("Val propres Laplacian non  normalisé =  ", yln)

fig,ax = plt.subplots(figsize=(5,5))
scatter = ax.scatter(YLn[:,2],YLn[:,3], c = y[0:N],cmap = plt.cm.Set1)

"""
**Comparer à scikitlearn**
"""
C = y[0:N].astype(float)

X_iso = manifold.SpectralEmbedding(n_neighbors = n_neighbors , n_components = 3, affinity='nearest_neighbors',random_state=0,eigen_solver="arpack").fit_transform(X)
fig,ax = plt.subplots(figsize=(5,5))
scatter = ax.scatter(X_iso[:,1],X_iso[:,2], c = C,cmap = plt.cm.Set1)
plt.title("Representation en 2 dimensions ({} voisins) - ScikitLearn".format(n_neighbors))
legend1 = ax.legend(*scatter.legend_elements(),loc = "upper right",title= "nuage de points")
ax.add_artist(legend1)
plt.show()
