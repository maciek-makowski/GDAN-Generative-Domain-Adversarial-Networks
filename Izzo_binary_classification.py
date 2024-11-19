import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scripts.DANN_training import GDANN, train_architecture
from scripts.Izzo_utils import shift_dist, generate_Izzo_latex_table
from sklearn.metrics import accuracy_score


###Define constants
num_iters = 10
no_samples = 10000
no_features = 11

###Define strategic features, which will be modified by the drift
strat_features = np.zeros(no_features)
strat_features[0:int(0.5*no_features)] = np.ones(int(0.5*no_features)) 
theta = np.ones(no_features)

### Define logistic regression models and other objects
logreg = LogisticRegression(C = 0.01, penalty='l2')
retrained_logreg = LogisticRegression(C = 0.01, penalty='l2')
model = GDANN(no_features = 11, no_domain_classes = 1)
scaler = StandardScaler()
pca = PCA(n_components = 2)

###Create distribution t_0
X,Y = shift_dist(no_samples, theta, no_features, strat_features)
X_og = X
Y_og = Y

### Fit logistic regression to t_0 and measure baseline accuracy 
logreg.fit(X,Y)
theta = logreg.coef_[0].T
baseline_accuracy = accuracy_score(Y, logreg.predict(X))
print("Baseline accuracy",baseline_accuracy)

### Create distribution t_1
X,Y = shift_dist(no_samples, theta, no_features, strat_features)

### Define lists 
X_iterations = []
Y_iterations = []
X_iterations.append(X_og)
Y_iterations.append(Y_og)
X_iterations.append(X)
Y_iterations.append(Y)

###Train the architecture 
# train_architecture(model, X_og, X_iterations, Y_iterations)

###Load model weights
# model.load_weights("GDANN_arch.weights.h5")
model.load_weights("./cluster_results/GDANN_arch_Izzo_5k.weights.h5")

#Change the no_points not to overload the memory
no_samples = 10000

accuracies_og_model_all_exp = []
accuracies_ret_model_all_exp = []
accuracies_gen_rep_all_exp = []
accuracies_DANN_model_all_exp = []
distances_og_drift_all_exp = []
distances_og_gen_all_exp = []
for _ in range(20):

    X,Y = X_og, Y_og
    #define lists for plotting
    accuracies_og_model = []
    accuracies_ret_model = []
    accuracies_gen_rep = []
    accuracies_DANN_model = []
    pca_drifted_list = []
    pca_drifted_non_norm_list = []
    drifted_list = []
    generated_list = []

    distances_og_gen = []
    distances_og_drift = []

    ### Create the testing loop
    num_test_iters = 20
    for i in range(num_test_iters):
        ### Create distribution t_i 
        X,Y = shift_dist(no_samples, theta, no_features, strat_features)

        feature_rep_entire_df = model.feature_extractor(X)
        generated_rep_entire_df = model.generator([feature_rep_entire_df])

        #Store the data for distance calculations
        drifted_list.append(X)
        generated_list.append(generated_rep_entire_df)

        #Retrain the logistic regression every third iteration
        retrained_logreg.fit(X, Y)
        if i % 3 == 0 and i != 0: theta = retrained_logreg.coef_[0].T
        print("RETRAINED THETA", theta, "Iteration", i)

        #Predictions for class labels by DANN
        predicted_class_labels_probabilities = model.label_classifier(feature_rep_entire_df)
        predicted_class_labels = tf.cast(predicted_class_labels_probabilities >= 0.5, tf.int32)

        #Assesment of the accuracies 
        accuracies_og_model.append(accuracy_score(Y,logreg.predict(X)))
        accuracies_ret_model.append(accuracy_score(Y, retrained_logreg.predict(X)))
        accuracies_gen_rep.append(accuracy_score(Y, logreg.predict(generated_rep_entire_df)))
        accuracies_DANN_model.append(accuracy_score(Y, predicted_class_labels))

        distances_og_drift.append(np.mean(np.abs(X - X_og)))
        distances_og_gen.append(np.mean(np.abs(generated_rep_entire_df- X_og)))
        print("dist og drift", distances_og_drift)
        print("dist og gen", distances_og_gen)

    ###Plotting and printing for evaluation 
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

# Calculate and print the mean and standard deviation for each set of accuracies
for name, values in accuracy_dict_all_runs.items():
    mean_val = np.mean(values, axis =0)
    std_val = np.std(values, axis = 0)
    print(f"{name}: Mean = {mean_val}, Std Dev = {std_val}")
    print(f"{name}:{values}")

generate_Izzo_latex_table(accuracy_dict_all_runs)
