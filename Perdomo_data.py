import numpy as np 
import matplotlib.pyplot as plt
from scripts.data_prep_GMSC import load_data, print_Perdomo_table
from scripts.strategic import best_response
from scripts.DANN_training import GDANN, train_architecture
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import os
import random

from sklearn.linear_model import LogisticRegression



#Assign indexes to strategic features
strat_features = np.array([1, 6, 8]) - 1 # for later indexings

# Defining constants 
num_iters = 1
eps = 10
method = "RRM"
output_dir = "complex_model_plots"

# Define objects for training the architecture 
model = GDANN(no_features = 11, no_domain_classes = num_iters)
pca = PCA(n_components = 2)
logreg = LogisticRegression()
retrained_logreg = LogisticRegression()

#Run the training of the architecture
# train_architecture(model, X, X_iterations, Y_iterations, num_epochs=2)
# When running training after exectuting train_architecture function execute sys.exit() 
# So the rest of the experiment does not get exectured

# model.load_weights("GDANN_arch.weights.h5")
# # # # #The model below might be the one that provides best results 
# model.load_weights("./model_weights/GDANN_22_06.weights.h5")
model.load_weights("./cluster_results/GDANN_arch_21_09.weights.h5")


## lists for plotting of the distance
drifted_list = []
generated_list = []
lr_coefficients_list = []


## For multiple experiments
num_experiments = 2
num_test_iters = 2

accuracies_og_model_all_exp = []
accuracies_ret_model_all_exp = []
accuracies_gen_rep_all_exp = []
accuracies_DANN_model_all_exp = []
distances_og_drift_all_exp = []
distances_og_gen_all_exp = []
for elo in range(num_experiments):
    #Load data 
    path = ".\GiveMeSomeCredit\cs-training.csv"
    X,Y, data  = load_data(path)
    n = X.shape[0]
    d = X.shape[1] - 1

    #Fit a baseline classifier and calculate its accuracy
    logreg.fit(X,Y)
    theta =  logreg.coef_[0]
    baseline_accuracy = accuracy_score(Y, logreg.predict(X))

    #Lists for storing differnt data distributions
    X_iterations = []
    Y_iterations = []

    X_strat = X
    X_iterations.append(X)

    # #Generate the distribution t1 by moving inducing drift on t0
    # X_strat = best_response(X_strat, theta, eps, strat_features)
    # X_iterations.append(X_strat)
    # Y_iterations.append(Y)
    # Y_iterations.append(Y)

    accuracies_og_model = []
    accuracies_ret_model = []
    accuracies_gen_rep = []
    accuracies_DANN_model = []

    distances_og_gen = []
    distances_og_drift = []
        
    # SELECT 10K points 
    num_points = 10000
    selected_indices = np.random.choice(n, size=num_points, replace=False)
    X_strat = X_strat[selected_indices]
    Y = Y[selected_indices]
    X = X[selected_indices]
    
    
    for i in range(num_test_iters):
        print("Experiment :", elo, "Iteration :", i)
        #Generate the distribution ti by moving inducing drift on t_(i-1)
        X_strat = best_response(X_strat, theta, eps, strat_features)      

        #Generate the domain-invariant representation and the mapping back to t0 for the entire df
        feature_rep_entire_df = model.feature_extractor(X_strat)
        generated_rep_entire_df = model.generator([feature_rep_entire_df])

        ## Append for the distance calculation 
        drifted_list.append(X_strat)
        generated_list.append(generated_rep_entire_df)

        #Retrain the logistic regression on t_i
        retrained_logreg.fit(X_strat, Y)

        #Updating theta - depending on type of the experiment
        theta = retrained_logreg.coef_[0]
        lr_coefficients_list.append(theta)
        # print("New reatrained theta", theta)

        #Predictions for class labels by DANN
        predicted_class_labels_probabilities = model.label_classifier(feature_rep_entire_df)
        predicted_class_labels = tf.cast(predicted_class_labels_probabilities >= 0.5, tf.int32)

        #Assesment of the accuracies 
        accuracies_og_model.append(accuracy_score(Y,logreg.predict(X_strat)))
        accuracies_ret_model.append(accuracy_score(Y, retrained_logreg.predict(X_strat)))
        accuracies_gen_rep.append(accuracy_score(Y, logreg.predict(generated_rep_entire_df)))
        accuracies_DANN_model.append(accuracy_score(Y, predicted_class_labels))

        #Calculate the distances
        distances_og_drift.append(np.mean(np.abs(X_strat - X)))
        distances_og_gen.append(np.mean(np.abs(generated_rep_entire_df- X)))
        print("dist og drift", distances_og_drift)
        print("dist og gen", distances_og_gen)

    iterations = np.arange(num_test_iters)
    accuracies_og_model_all_exp.append(accuracies_og_model)
    accuracies_ret_model_all_exp.append(accuracies_ret_model)
    accuracies_gen_rep_all_exp.append(accuracies_gen_rep)
    accuracies_DANN_model_all_exp.append(accuracies_DANN_model)
    distances_og_drift_all_exp.append(distances_og_drift)
    distances_og_gen_all_exp.append(distances_og_gen)



########################Results for all experiments ###################################
# Store results in a dictionary for easier access
print ("Distance og gen", distances_og_gen_all_exp)
accuracy_dict_all_runs = {
    "Acc $M_0$": np.array(accuracies_og_model_all_exp),
    "Acc $M_{ret}$": np.array(accuracies_ret_model_all_exp),
    "Acc $M_g$": np.array(accuracies_gen_rep_all_exp),
    "Acc $M_{LC}$": np.array(accuracies_DANN_model_all_exp),
    "distance_og_drift": np.array(distances_og_drift_all_exp),
    "distance_og_gen": np.array(distances_og_gen_all_exp)
}

print_Perdomo_table(accuracy_dict_all_runs)