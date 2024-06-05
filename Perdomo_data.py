import numpy as np 
import matplotlib.pyplot as plt
from scripts.data_prep_GMSC import load_data
from scripts.optimization import logistic_regression, evaluate_loss
from scripts.strategic import best_response
from scripts.DANN_training import GDANN, train_architecture
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import random

from sklearn.linear_model import LogisticRegression

path = ".\GiveMeSomeCredit\cs-training.csv"
X,Y, data  = load_data(path)
n = X.shape[0]
d = X.shape[1] - 1

strat_features = np.array([1, 6, 8]) - 1 # for later indexings


# Defining constants 
num_iters = 1
eps = 10
method = "RRM"



# Define stuff for training the DANN 
model = GDANN(no_features = 11, no_domain_classes = num_iters)
scaler = StandardScaler()
# scaler = MinMaxScaler(feature_range=(-10,10))
pca = PCA(n_components = 2)
logreg = LogisticRegression(C = 0.01)
retrained_logreg = LogisticRegression(C = 0.01)


# fit logistic regression model we treat as the truth
# lam = 1.0/n
# theta_true, loss_list, smoothness = logistic_regression(X, Y, lam, 'Exact')
# baseline_accuracy = ((X.dot(theta_true) > 0)  == Y).mean()

logreg.fit(X,Y)
theta =  logreg.coef_[0]
baseline_accuracy = accuracy_score(Y, logreg.predict(X))
#Scale the data to a range (-1,1)
#X = scaler.fit_transform(X)


#Lists for storing differnt data distributions
X_iterations = []
Y_iterations = []

X_strat = X

X_iterations.append(X)
X_strat = best_response(X_strat, theta, eps, strat_features)
X_iterations.append(X_strat)
Y_iterations.append(Y)
Y_iterations.append(Y)


# train_architecture(model, X, X_iterations, Y_iterations)
# model.load_weights("GDANN_arch.weights.h5")
model.load_weights("./model_weights/GDANN_28_05.weights.h5")
accuracies_og_model = []
accuracies_ret_model = []
accuracies_gen_rep = []
accuracies_DANN_model = []


num_test_iters = 10
for i in range(num_test_iters):
    X_strat = best_response(X_strat, theta, eps, strat_features)

    selected_indices = np.random.choice(X_strat.shape[0], size = 5, replace = False)
    selected_points = X_strat[selected_indices]
    selected_labels = Y[selected_indices]

    feature_rep = model.feature_extractor(selected_points)
    generated_rep = model.generator([feature_rep, tf.ones_like(selected_labels)])


    for j,point in enumerate(selected_points):
        if i == 0 or i ==9:
            plt.scatter(np.arange(11), point, label='Influenced by the drift', marker='o')
            plt.scatter(np.arange(11), X[selected_indices[j]], label='Original', marker='s')
            plt.scatter(np.arange(11), generated_rep[j], label='Generated', marker='^')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(f'Generated vs Drifted vs OG iteration {i}')
            plt.legend()
            plt.show()

    feature_rep_entire_df = model.feature_extractor(X_strat)
    generated_rep_entire_df = model.generator([feature_rep_entire_df, tf.ones_like(Y)])

    normalized_drift = scaler.fit_transform(X_strat)
    normalized_generated = scaler.fit_transform(generated_rep_entire_df)

    #Check the mean and STDS 

    mean_drift = np.mean(normalized_drift, axis =0)
    mean_gen = np.mean(normalized_generated, axis =0)
    mean_og = np.mean(X, axis =0)

    std_drift = np.std(normalized_drift, axis = 0)
    std_gen = np.std(normalized_generated, axis = 0)
    std_og = np.std(X, axis = 0)

    normalized_X = scaler.fit_transform(X)
    mean_scaled_X = np.mean(normalized_X, axis = 0)
    std_scaled_X = np.std(normalized_X, axis = 0)

    print("\n")
    print("Check the mean and variance")
    print("\n")

    print("Drift :", mean_drift, std_drift)
    print("Gen :", mean_gen, std_gen)
    print("OG :", mean_og, std_og)
    print("Norm OG :", mean_scaled_X, std_scaled_X)

    pca_drifted = pca.fit_transform(normalized_drift)
    pca_original = pca.fit_transform(normalized_X)
    pca_generated = pca.fit_transform(normalized_generated)
    # pca_drifted = pca.fit_transform(X_strat)
    # pca_original = pca.fit_transform(X)
    # pca_generated = pca.fit_transform(generated_rep_entire_df)
    if i ==0 or i ==9:
    # Plot PCA results
        plt.figure(figsize=(10, 6))

        # Plot drifted data
        plt.scatter(pca_drifted[:, 0], pca_drifted[:, 1], c='blue', marker = '^', label=f'Data that has been influenced by the drift')

        # Plot original data
        plt.scatter(pca_original[:, 0], pca_original[:, 1], c='red', marker = 'o', label='Original Data')

        # Plot generated data
        plt.scatter(pca_generated[:, 0], pca_generated[:, 1], c='green', marker = 's', label='Generated Data')

        # Add labels and legend
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'Principal Components iteration {i}')
        plt.legend()
        plt.grid(True)

        # Show plot
        plt.show()


    #Retraining 
    retrained_logreg.fit(X_strat, Y)

    #Predictions for class labels by DANN
    predicted_class_labels_probabilities = model.label_classifier(feature_rep_entire_df)
    predicted_class_labels = tf.cast(predicted_class_labels_probabilities >= 0.5, tf.int32)

    #Assesment of the accuracies 
    accuracies_og_model.append(accuracy_score(Y,logreg.predict(X_strat)))
    accuracies_ret_model.append(accuracy_score(Y, retrained_logreg.predict(X_strat)))
    accuracies_gen_rep.append(accuracy_score(Y, logreg.predict(generated_rep_entire_df)))
    accuracies_DANN_model.append(accuracy_score(Y, predicted_class_labels))

iterations = np.arange(num_test_iters)


print("Iteration", iterations, iterations.shape)
print("Accuracy", accuracies_og_model, len(accuracies_og_model))
plt.plot(iterations, accuracies_og_model, marker='o', label='Original LR Model')
plt.plot(iterations, accuracies_ret_model, marker='o', label='Retrained LR Model')
plt.plot(iterations, accuracies_gen_rep, marker='o', label='Generated representation with original LR model')
plt.plot(iterations, accuracies_DANN_model, marker='o', label='DANN Model')

plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Model Accuracies Over Iterations')
plt.legend()
plt.grid(True)
plt.show()
