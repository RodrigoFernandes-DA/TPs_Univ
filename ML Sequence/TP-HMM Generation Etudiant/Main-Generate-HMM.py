# -*- coding: utf-8 -*-

import numpy as np

# pour charger les données MNIST avec pytorch
# pip install python-mnist 
# from mnist import MNIST
# from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
from tqdm import tqdm
import hmm

from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

#color_names = list(mcolors.CSS4_COLORS)

D = 28           # 28 X 28
N_train = 10000
N_valid = 1000

n_clusters = 100  # number of clusters used to discretize each pixel columns
n_states = 15

N_classes = 10
N_samples = 10
length = D

dir_name = "Model_"+str(n_clusters)+"_"+str(n_states)

###########################################################################
print("Loading codebook: n_cluster =",n_clusters,"N_train = ",N_train)
file_name = "Codebook_"+str(n_clusters)+"_"+str(N_train)
codebook = hmm.load_codebook(file_name)

###########################################################################
print("Loading models: n_states =",n_states,"n_cluster =",n_clusters,"N_train = ",N_train)

models = []
suffix = "_"+str(n_clusters)+"_"+str(n_states)+"_"+str(N_train)+"_"+str(N_valid)
    
for classe in range(N_classes):
    file_name = 'class_'+str(classe)+suffix
    model = hmm.HMM(n_states,n_clusters,dir_name,file_name)
    models.append(model)

LL_scores = np.zeros((N_classes,))
print("Generation in progress...")

total_characters = 0
correct_characters = 0
total_errors = 0

# for classe in range(N_classes):
for classe in range(3,4):
    seq_states, seq_observations = models[classe].GenerateSamples(N_samples,length)  

    centers = codebook.cluster_centers_
    XX_discretes = codebook.labels_
    ###################################################################
    # View the corresponding image of the generated observation sequences
    image = np.zeros((D,N_samples*D))
    for sample in range(N_samples):
        for t in range(D):
            image[:,sample * D + t] = centers[seq_observations[sample,t],:]
        
        ###################################################################
        # Compute Loglikelihood of the best path with Viterbi
        for c in range(N_classes):
            LL_scores[c] = models[c].Viterbi(seq_observations[sample])

        predicted_class = np.argmax(LL_scores)
        print("\nLL_scores",LL_scores)
        print("Classse:",classe,"meilleur model :",predicted_class,"LL:",np.max(LL_scores))

        if predicted_class == classe:
            correct_characters += 1
        
        # Aumente o número total de caracteres
        total_characters += 1

        # Verifique se a previsão da classe é incorreta para calcular o CER
        if predicted_class != classe:
            total_errors += 1

    
    # plt.figure(figsize=(9,1.1))
    # plt.imshow(image, cmap='gray')
    # plt.show()


    
CRR = correct_characters / total_characters
CER = total_errors / total_characters

print(f"Character Recognition Rate (CRR): {CRR * 100:.2f}%")
print(f"Character Error Rate (CER): {CER * 100:.2f}%")
    
print("Generation ended successfully.")




###########################################################################
# MNIST Test Dataset Evaluation
###########################################################################
print("\n" + "="*50)
print("EVALUATING ON MNIST TEST DATASET")
print("="*50)

# Load MNIST test data
# Assuming you have a function to load MNIST data
# If not, you'll need to implement this part based on your data loading method
try:
    # Example of how you might load MNIST data
    # You may need to adjust this based on your actual data loading method
    from tensorflow.keras.datasets import mnist
    (_, _), (X_test, y_test) = mnist.load_data()
    
    # Alternatively, if you're using the python-mnist package:
    # from mnist import MNIST
    # mndata = MNIST('path_to_mnist_data')
    # X_test, y_test = mndata.load_testing()
    
    print(f"Loaded MNIST test data: {len(X_test)} samples")
    
except ImportError:
    print("Please install tensorflow or provide MNIST test data")
    sys.exit(1)

# Discretize the test data using the codebook
print("Discretizing test data...")
X_test_discrete = np.zeros((len(X_test), D), dtype=int)

for i in tqdm(range(len(X_test))):
    # Reshape and discretize each column (time step)
    image = X_test[i].reshape(D, D)
    for t in range(D):
        column = image[:, t].reshape(1, -1)
        X_test_discrete[i, t] = codebook.predict(column)[0]

# Evaluate on MNIST test data
print("Evaluating on MNIST test data...")
true_labels = []
predicted_labels = []

total_test_characters = min(1000, len(X_test))
correct_test_characters = 0
total_test_errors = 0

LL_scores_test = np.zeros((N_classes,))

for i in tqdm(range(min(1000, len(X_test)))):  # Limit to 1000 samples for faster testing
    observation_sequence = X_test_discrete[i]
    true_class = y_test[i]
    
    # Compute log-likelihood for each class
    for c in range(N_classes):
        LL_scores_test[c] = models[c].Viterbi(observation_sequence)
    
    predicted_class = np.argmax(LL_scores_test)
    
    true_labels.append(true_class)
    predicted_labels.append(predicted_class)
    
    if predicted_class == true_class:
        correct_test_characters += 1
    else:
        total_test_errors += 1

# Calculate CRR and CER for test data
CRR_test = correct_test_characters / total_test_characters
CER_test = total_test_errors / total_test_characters

print(f"\nMNIST Test Results:")
print(f"Character Recognition Rate (CRR): {CRR_test * 100:.2f}%")
print(f"Character Error Rate (CER): {CER_test * 100:.2f}%")

# Create confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix for MNIST Test Predictions')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_mnist.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, digits=4))

print("MNIST test evaluation completed successfully.")