Travaux Pratiques - Master Sciences des Donnees



4 exercicios, tem que fazer 3
Avalia o que é feito nos TPs



Como posso obter exatamente os mesmos resultados, mas sem usar a biblioteca mnist ou semelhantes? saiba que eu ja tenho os arquivos "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "train-images-idx3-ubyte" e "train-labels-idx1-ubyte" na pasta samples

from mnist import MNIST

data = MNIST('./samples')
X, y = data.load_training()
X = np.asarray(X)
y = np.asarray(y)

X_test, y_test = data.load_testing()
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

