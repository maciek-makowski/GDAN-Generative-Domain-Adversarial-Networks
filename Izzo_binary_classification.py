import sys 
import numpy as np 
import pygad
import matplotlib.pyplot as plt
from scripts.Izzo_utils import shift_dist, fit 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
# from scripts.DANN_training import DANN, train_dann_model
from scripts.DANN_training import GDANN, train_architecture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


num_iters = 10
no_samples = 20000
no_features = 11
# strat_features = np.ones(no_features)
strat_features = np.zeros(no_features)
strat_features[0:int(0.5*no_features)] = np.ones(int(0.5*no_features)) 
#theta = np.random.randn(no_features)
theta = np.ones(no_features)
print(theta)

### Try to specify the models better, cause there is divergence in accuracies between methods
logreg = LogisticRegression(C = 0.01, penalty='l2')
retrained_logreg = LogisticRegression(C = 0.01, penalty='l2')
model = GDANN(no_features = 11, no_domain_classes = 1)
scaler = StandardScaler()
pca = PCA(n_components = 2)

X,Y = shift_dist(no_samples, theta, no_features, strat_features)
X_og = X
Y_og = Y

logreg.fit(X,Y)
theta = logreg.coef_[0].T

baseline_accuracy = accuracy_score(Y, logreg.predict(X))
print("Baseline accuracy",baseline_accuracy)

X,Y = shift_dist(no_samples, theta, no_features, strat_features)

X_iterations = []
Y_iterations = []

X_iterations.append(X_og)
Y_iterations.append(Y_og)
X_iterations.append(X)
Y_iterations.append(Y)

# train_architecture(model, X_og, X_iterations, Y_iterations)
model.load_weights("GDANN_arch.weights.h5")
# # # model.load_weights("./model_weights/GDANN_IZZO_28_05.weights.h5")

#Change the no_points not to overload the memory
no_samples = 10000

#define lists for plotting
accuracies_og_model = []
accuracies_ret_model = []
accuracies_gen_rep = []
accuracies_DANN_model = []

pca_drifted_list = []
pca_drifted_non_norm_list = []

drifted_list = []
generated_list = []

#testing loop 
num_test_iters = 10
for i in range(num_test_iters):
    X,Y = shift_dist(no_samples, theta, no_features, strat_features)

    feature_rep_entire_df = model.feature_extractor(X)
    generated_rep_entire_df = model.generator([feature_rep_entire_df, tf.zeros_like(Y)])
    # generated_rep_entire_df = model.generator([feature_rep_entire_df])

    # normalized_drift = scaler.fit_transform(X)
    # normalized_generated = scaler.fit_transform(generated_rep_entire_df)
    # normalized_X = scaler.fit_transform(X_og)

    # pca_drifted = pca.fit_transform(normalized_drift)
    # pca_original = pca.fit_transform(normalized_X)
    # pca_generated = pca.fit_transform(normalized_generated)

    pca_drifted = pca.fit_transform(X)
    pca_original = pca.fit_transform(X_og)
    pca_generated = pca.fit_transform(generated_rep_entire_df)

    if i % 3 == 0:
        pca_drifted_list.append(pca_drifted)
        pca_drifted_non_norm_list.append(pca.fit_transform(X))

    # Plot PCA results
    plt.figure(figsize=(10, 6))

    # Plot drifted data
    plt.scatter(pca_drifted[:, 0], pca_drifted[:, 1], c='blue', marker = '^', label=f'Data that has been influenced by the drift')

    # Plot original data
    plt.scatter(pca_original[:, 0], pca_original[:, 1], c='red', marker = 'o', label='Original Data')

    # Plot generated data
    plt.scatter(pca_generated[:, 0], pca_generated[:, 1], c='green', marker = 's', label='Generated Data', alpha = 0.1)

    # Add labels and legend
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title(f"Principal Component Analysis iter {i}")
    plt.grid(True)

    # Show plot
    plt.show()

    #Store the data for distance calculations
    drifted_list.append(X)
    generated_list.append(generated_rep_entire_df)

    #Retraining 
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

iterations = np.arange(num_test_iters)


print("Iteration", iterations, iterations.shape)
print("Accuracy OG", accuracies_og_model)
print("Accuracy ret", accuracies_ret_model)
print("Accuracy gen", accuracies_gen_rep)
print("Accuracy DANN", accuracies_DANN_model)
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

distances_og_gen = []
distances_og_drift = []

for i,drifted_data in enumerate(drifted_list):
    distances_og_drift.append(np.mean(np.abs(drifted_data - X_og[:no_samples])))
    distances_og_gen.append(np.mean(np.abs(generated_list[i] - X_og[:no_samples])))

fig, ax1 = plt.subplots()

# Plot the accuracies on the primary y-axis
color1 = 'tab:blue'
color2 = 'tab:green'
ax1.plot(iterations, accuracies_og_model, marker='o', label='Accuracy original LR Model', color=color1)
ax1.plot(iterations, accuracies_gen_rep, marker='o', label='Accuracy generated representation with original LR model', color=color2)
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



colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#0000ff', '#ffa500', '#808000']
for i, pca_drifted in enumerate(pca_drifted_list):
    print("Iter ", i)
    plt.scatter(pca_drifted[:, 0], pca_drifted[:, 1], c=colors[i], marker = 'o', label=f'Iter {3*i}')

plt.legend()
plt.title("Differences in data distributions influenced by the drift")
plt.show()
