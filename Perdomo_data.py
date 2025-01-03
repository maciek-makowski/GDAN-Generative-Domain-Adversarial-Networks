import random
import sys
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from scripts.data_prep_GMSC import load_data
from scripts.strategic import best_response
from scripts.DANN_training import GDANN, train_architecture
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
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

# Define classes 
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
#In case training without evaluation should be performed uncomment
#sys.exit()

#Load the model
model.load_weights("./cluster_results/GDANN_arch_Perdomo_10k_v3.weights.h5")

#Define lists for storing 
accuracies_og_model = []
accuracies_ret_model = []
accuracies_gen_rep = []
accuracies_DANN_model = []

## lists for plotting of the distance
drifted_list = []
feature_extracted_list = []
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

    #Generate the domain-invariant representation and the mapping back to t0 for the entire df
    feature_rep_entire_df = model.feature_extractor(X_strat)
    generated_rep_entire_df = model.generator([feature_rep_entire_df])

    ## Append for the distance calculation 
    feature_extracted_list.append(feature_rep_entire_df)
    drifted_list.append(X_strat)
    generated_list.append(generated_rep_entire_df)

    # Generate a PCA plot demonstrating the outputs of the feature extractor    
    if i == (num_test_iters - 1):
        plt.figure(figsize=(10, 6))
        for i,dist in enumerate(feature_extracted_list):
            pca_feature_extracted = pca.fit_transform(dist)
            plt.scatter(pca_feature_extracted[:,0], pca_feature_extracted[:,1], c = [random.random() for _ in range(3)], label=f"Iter {i}")

        plt.legend()
        plt.title("PCA of the feature extracted representation")
        plt.grid(True)
        plt.show()

    # Create PCA clusters visualization 
    if i == 0 or i == 9 or i == 5: 
        pca_drifted = pca.fit_transform(X_strat)
        pca_original = pca.fit_transform(X)
        pca_generated = pca.fit_transform(generated_rep_entire_df)

        # Plot PCA results
        plt.figure(figsize=(10, 6))
        plt.scatter(pca_drifted[:, 0], pca_drifted[:, 1], c='blue', marker = '^', label=f'Data that has been influenced by the drift')
        plt.scatter(pca_original[:, 0], pca_original[:, 1], c='red', marker = 'o', label='Original Data')
        plt.scatter(pca_generated[:, 0], pca_generated[:, 1], c='green', marker = 's', label='Generated Data')

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'Principal Components iteration {i}')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Generate KDE plots of the performative features
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns, figure size
        sns.kdeplot(X_strat[:, 0], ax=axes[0], fill=True, color="blue", label='influenced by the drift')
        sns.kdeplot(X[:, 0], ax=axes[0], fill=True, color="red", label="original")
        sns.kdeplot(generated_rep_entire_df[:, 0], ax=axes[0], fill=True, color="green", label="generated")
        axes[0].set_title('KDE Plot for feature 0')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Density')
        axes[0].grid(True)
        axes[0].legend()
        axes[0].set_xlim(-1,1)

        sns.kdeplot(X_strat[:, 5], ax=axes[1], fill=True, color="blue", label='influenced by the drift')
        sns.kdeplot(X[:, 5], ax=axes[1], fill=True, color="red", label="original")
        sns.kdeplot(generated_rep_entire_df[:, 5], ax=axes[1], fill=True, color="green", label="generated")
        axes[1].set_title('KDE Plot for feature 6')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Density')
        axes[1].grid(True)
        axes[1].legend()
        axes[1].set_xlim(-8,8)

        sns.kdeplot(X_strat[:, 7], ax=axes[2], fill=True, color="blue", label='influenced by the drift')
        sns.kdeplot(X[:, 7], ax=axes[2], fill=True, color="red", label="original")
        sns.kdeplot(generated_rep_entire_df[:, 7], ax=axes[2], fill=True, color="green", label="generated")
        axes[2].set_title('KDE Plot for feature 8')
        axes[2].set_xlabel('Value')
        axes[2].set_ylabel('Density')
        axes[2].grid(True)
        axes[2].legend()
        axes[2].set_xlim(-14,14)

        fig.suptitle(f"Kernel Density Estimation of the Performative Features - Iteration {i}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
        plt.show()


    #Retrain the logistic regression on t_i
    retrained_logreg.fit(X_strat, Y)

    #Updating theta - depending on type of the experiment
    # theta = retrained_logreg.coef_[0]
    lr_coefficients_list.append(theta)

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
    for feature in range(d+1):
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