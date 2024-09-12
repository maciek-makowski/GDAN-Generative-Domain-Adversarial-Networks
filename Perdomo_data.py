import numpy as np 
import matplotlib.pyplot as plt
from scripts.data_prep_GMSC import load_data
from scripts.optimization import logistic_regression, evaluate_loss
from scripts.strategic import best_response
from scripts.DANN_training import GDANN, train_architecture
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import random

from sklearn.linear_model import LogisticRegression

#Load data 
path = ".\GiveMeSomeCredit\cs-training.csv"
X,Y, data  = load_data(path)
n = X.shape[0]
d = X.shape[1] - 1

#Assign indexes to strategic features
strat_features = np.array([1, 6, 8]) - 1 # for later indexings

# Defining constants 
num_iters = 1
eps = 10
method = "RRM"

# Define objects for training the architecture 
model = GDANN(no_features = 11, no_domain_classes = num_iters)
pca = PCA(n_components = 2)
logreg = LogisticRegression()
retrained_logreg = LogisticRegression()

#Fit a baseline classifier and calculate its accuracy
logreg.fit(X,Y)
theta =  logreg.coef_[0]
baseline_accuracy = accuracy_score(Y, logreg.predict(X))

#Lists for storing differnt data distributions
X_iterations = []
Y_iterations = []

X_strat = X
X_iterations.append(X)

#Generate the distribution t1 by moving inducing drift on t0
X_strat = best_response(X_strat, theta, eps, strat_features)
X_iterations.append(X_strat)
Y_iterations.append(Y)
Y_iterations.append(Y)

#Run the training of the architecture
# train_architecture(model, X, X_iterations, Y_iterations, num_epochs=2)

# model.load_weights("GDANN_arch.weights.h5")
# # # # #The model below might be the one that provides best results 
# model.load_weights("./model_weights/GDANN_22_06.weights.h5")
model.load_weights("./cluster_results/GDANN_arch_11_09.weights.h5")
accuracies_og_model = []
accuracies_ret_model = []
accuracies_gen_rep = []
accuracies_DANN_model = []

## lists for plotting of the distance
drifted_list = []
generated_list = []
lr_coefficients_list = []

## SELECT 10K points 
num_points = 10000
selected_indices = np.random.choice(n, size=num_points, replace=False)
X_strat = X_strat[selected_indices]
Y = Y[selected_indices]
X = X[selected_indices]
num_test_iters = 10

for i in range(num_test_iters):
    #Generate the distribution ti by moving inducing drift on t_(i-1)
    X_strat = best_response(X_strat, theta, eps, strat_features)      

    #Select random points for testing (hardware limitation)
    selected_indices = np.random.choice(X_strat.shape[0], size = 5, replace = False)
    selected_points = X_strat[selected_indices]
    selected_labels = Y[selected_indices]

    #Generate the domain-invariant representation and the mapping back to t0 for selected points
    feature_rep = model.feature_extractor(selected_points)
    generated_rep = model.generator([feature_rep])

    #Plotting of individual points
    # for j,point in enumerate(selected_points):
    #     if i == 0 or i==1 or i ==9:
    #         plt.scatter(np.arange(11), point, label='Influenced by the drift', marker='o')
    #         plt.scatter(np.arange(11), X[selected_indices[j]], label='Original', marker='s')
    #         plt.scatter(np.arange(11), generated_rep[j], label='Generated', marker='^')
    #         plt.xlabel('Index')
    #         plt.ylabel('Value')
    #         plt.title(f'Generated vs Drifted vs OG iteration {i}')
    #         plt.legend()
    #         plt.show()
    
    #Generate the domain-invariant representation and the mapping back to t0 for the entire df
    feature_rep_entire_df = model.feature_extractor(X_strat)
    generated_rep_entire_df = model.generator([feature_rep_entire_df])

    ## Append for the distance calculation 
    drifted_list.append(X_strat)
    generated_list.append(generated_rep_entire_df)

    # #Check the mean and STDS 
    # mean_drift = np.mean(normalized_drift, axis =0)
    # mean_gen = np.mean(normalized_generated, axis =0)
    # mean_og = np.mean(X, axis =0)

    # std_drift = np.std(normalized_drift, axis = 0)
    # std_gen = np.std(normalized_generated, axis = 0)
    # std_og = np.std(X, axis = 0)

    # normalized_X = scaler.fit_transform(X)
    # mean_scaled_X = np.mean(normalized_X, axis = 0)
    # std_scaled_X = np.std(normalized_X, axis = 0)

    # pca_drifted = pca.fit_transform(normalized_drift)
    # pca_original = pca.fit_transform(normalized_X)
    # pca_generated = pca.fit_transform(normalized_generated)

    #Create PCA clusters visualization 
    if i == 0 or i == 9 or i == 5: 
        pca_drifted = pca.fit_transform(X_strat)
        pca_original = pca.fit_transform(X)
        pca_generated = pca.fit_transform(generated_rep_entire_df)

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


    #Retrain the logistic regression on t_i
    retrained_logreg.fit(X_strat, Y)

    #Updating theta - depending on type of the experiment
    # theta = retrained_logreg.coef_[0]
    lr_coefficients_list.append(theta)
    print("New reatrained theta", theta)

    #Predictions for class labels by DANN
    predicted_class_labels_probabilities = model.label_classifier(feature_rep_entire_df)
    predicted_class_labels = tf.cast(predicted_class_labels_probabilities >= 0.5, tf.int32)

    #Assesment of the accuracies 
    accuracies_og_model.append(accuracy_score(Y,logreg.predict(X_strat)))
    accuracies_ret_model.append(accuracy_score(Y, retrained_logreg.predict(X_strat)))
    accuracies_gen_rep.append(accuracy_score(Y, logreg.predict(generated_rep_entire_df)))
    accuracies_DANN_model.append(accuracy_score(Y, predicted_class_labels))

iterations = np.arange(num_test_iters)

######################## PLOTTING AND VERYFING RESULTS ################################

accuracy_dict = {
    "accuracies_og_model": accuracies_og_model,
    "accuracies_ret_model": accuracies_ret_model,
    "accuracies_gen_rep": accuracies_gen_rep,
    "accuracies_DANN_model": accuracies_DANN_model
}

# Printing the names and values
for name, values in accuracy_dict.items():
    print(f"{name}: {values}")

### THE ACCURACY PLOT ###############################3

print("Iteration", iterations, iterations.shape)
print("Accuracy", accuracies_og_model, len(accuracies_og_model))
plt.plot(iterations, np.ones_like(iterations)*baseline_accuracy, marker='o',  linestyle=':', color='grey', label = 'Baseline accuracy')
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

#########PLOT OF THE LR COEFFICIENTS #####################
# Plot each iteration as a grouped bar
fig, ax = plt.subplots(figsize=(10, 6))
data = np.vstack(lr_coefficients_list)
data_transposed = data.T
x = np.arange(11)  # Index positions
width = 0.8 / 10
for i in range(10):
    ax.bar(x + i * width, data_transposed[:, i], width=width, label=f'Iteration {i+1}')

ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.set_title('Evolution of each index over 10 iterations')
ax.set_xticks(x + 0.4)
ax.set_xticklabels(x)
ax.legend()

plt.tight_layout()
plt.show()

### BOXPLOTS TO SHOW DIFFERENCES BETWEEN THE GENERATED AND DRIFTED ################################################
for iter in [0,9]:
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), sharey=True)
    axes = axes.flatten()

    for feature in range(11):
        # Collect data for the current feature
        feature_data = [X[:, feature], drifted_list[iter][:, feature], generated_list[iter][:, feature]]
        # Create a boxplot in the respective subplot
        axes[feature].boxplot(feature_data, patch_artist=True, labels=['Distribution t0', 'Distribution ti', 'Generated'], vert = False)
        axes[feature].set_title(f'Feature {feature + 1}')
        axes[feature].grid(True)

    # Remove any unused subplots (in this case, the last one)
    for i in range(11, 12):
        fig.delaxes(axes[i])

    # Adjust layout
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.suptitle(f"Boxplot comparison of distributions iteration - {iter}", fontsize = 16)
    plt.show()

### PLOT OF THE DISTANCES BETWEEN DISTRIBUTIONS ##################################

distances_og_gen = []
distances_og_drift = []

for i,drifted_data in enumerate(drifted_list):
    distances_og_drift.append(np.mean(np.abs(drifted_data - X)))
    distances_og_gen.append(np.mean(np.abs(generated_list[i] - X)))

print("Distances drifted original", distances_og_drift)
print("Differences from i to i+1", distances_og_drift - distances_og_drift[0])
fig, ax1 = plt.subplots()

# Plot the accuracies on the primary y-axis
color1 = 'tab:blue'
color2 = 'tab:green'
color3 = 'tab:orange'
ax1.plot(iterations, accuracies_og_model, marker='o', label='Accuracy original LR Model', color=color1)
ax1.plot(iterations, accuracies_gen_rep, marker='o', label='Accuracy generated representation with original LR model', color=color2)
ax1.plot(iterations, accuracies_DANN_model, marker='o', label='Accuracy with the DANN Label classifier', color=color3)
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Accuracy', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.legend(loc='upper left')
ax1.grid(True, which='both', axis='y', linestyle='--', color=color1, alpha=0.5)

# Create a second y-axis to plot the distances
ax2 = ax1.twinx()
color3 = 'tab:red'
color4 = 'tab:purple'
ax2.plot(iterations, distances_og_drift, marker='o', label='Distance between original and influenced by the drift', color=color3)
ax2.plot(iterations, distances_og_gen, marker='o', label='Distance between original and generated', color=color4)
ax2.set_ylabel('Distance', color=color3)
ax2.tick_params(axis='y', labelcolor=color3)
ax2.legend(loc='upper right')
ax2.grid(True, which='both', axis='y', linestyle='--', color=color3, alpha=0.5)

# Add a title for the entire plot
plt.title('Accuracies and Distances over Iterations')

# Show the plot
plt.show()