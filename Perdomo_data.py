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
import tensorflow as tf

from sklearn.linear_model import LogisticRegression

path = ".\GiveMeSomeCredit\cs-training.csv"
X,Y, data  = load_data(path)
n = X.shape[0]
d = X.shape[1] - 1

strat_features = np.array([1, 6, 8]) - 1 # for later indexings

# fit logistic regression model we treat as the truth
lam = 1.0/n
theta_true, loss_list, smoothness = logistic_regression(X, Y, lam, 'Exact')
baseline_accuracy = ((X.dot(theta_true) > 0)  == Y).mean()

# Defining constants 
num_iters = 5
eps = 10
method = "RRM"
# initial theta
theta = np.copy(theta_true)

# Define stuff for training the DANN 
model = GDANN(no_features = 11, no_domain_classes = num_iters)
scaler = StandardScaler()
pca = PCA(n_components = 2)
logreg = LogisticRegression()

#Lists for storing differnt data distributions
X_iterations = []
Y_iterations = []

X_strat = X

for t in range(num_iters):
    X_strat = best_response(X_strat, theta, eps, strat_features)
    print("X_strat", X_strat[:5])
    #X_iterations.append(scaler.fit_transform(X_strat))
    X_iterations.append(X_strat)
    Y_iterations.append(Y)
    


#train_architecture(model, X, X_iterations, Y_iterations)
#model.load_weights("GDANN_arch.weights.h5")
model.load_weights("GDANN_15_05.weights.h5")

domain_labels = []
len_single_domain = len(X_iterations[0])
for i, datapoints in enumerate(X_iterations):
    domain_labels.extend([i] * len(datapoints))
    
    combined_domain_labels = np.array(domain_labels)   
    combined_data = np.concatenate(X_iterations)
    combined_class_labels = np.concatenate(Y_iterations) 





# # for i in set(domain_labels):
# #     random_index = np.random.randint(0, len(combined_data))

# #     random_data_point = combined_data[random_index].reshape(-1,1)
# #     random_domain_label = combined_domain_labels[random_index]
# #     random_class_label = combined_class_labels[random_index]
    
# #     actual_index = random_index % int(0.2 * len(combined_data))
# #     print("Random index", random_index)
# #     print("Actual index", actual_index)


# #     feature_rep = model.feature_extractor(random_data_point.T)
# #     generated_rep = model.generator([feature_rep, tf.convert_to_tensor(random_domain_label.reshape(-1,1))])

# #     print("drifted", random_data_point.T)
# #     print("X0", X[actual_index])
# #     print("generated", generated_rep.numpy())

# #     plt.scatter(np.arange(11), random_data_point, label='drifted')
# #     plt.scatter(np.arange(11), X[actual_index], label='OG')
# #     plt.scatter(np.arange(11), generated_rep, label='Generated') 
# #     plt.xlabel('Index')
# #     plt.ylabel('Value')
# #     plt.title(f"Differences betweeen generated, origianal, and drift-influenced points - iteration {random_domain_label}")
# #     plt.legend()
# #     plt.show()


selected_points = []
#fig, axs = plt.subplots(5, 1, figsize=(20, 4))

for i, domain_label in enumerate(set(combined_domain_labels)):
    # Select a random index with the current domain label
    domain_indices = np.where(combined_domain_labels == domain_label)[0]
    random_index = np.random.choice(domain_indices)
    original_index = random_index % int(0.2 * len(combined_data))
    
    random_data_point = combined_data[random_index].reshape(-1, 1)
    random_domain_label = combined_domain_labels[random_index]
    random_class_label = combined_class_labels[random_index]

    feature_rep = model.feature_extractor(random_data_point.T)
    generated_rep = model.generator([feature_rep, tf.convert_to_tensor(random_domain_label.reshape(-1, 1))])

    plt.scatter(np.arange(11), random_data_point, label='Drifted', marker='o')
    plt.scatter(np.arange(11), X[original_index], label='Original', marker='s')
    plt.scatter(np.arange(11), generated_rep, label='Generated', marker='^')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Domain Label: {random_domain_label}')
    plt.legend()
    plt.show()
    # axs[i].scatter(np.arange(11), random_data_point, label='Drifted', marker='o')
    # axs[i].scatter(np.arange(11), X[original_index], label='Original', marker='s')
    # axs[i].scatter(np.arange(11), generated_rep, label='Generated', marker='^')
    # axs[i].set_xlabel('Index')
    # axs[i].set_ylabel('Value')
    # axs[i].set_title(f'Domain Label: {random_domain_label}')
    # axs[i].legend()
    

domain_label = 0
indices_domain = np.where(combined_domain_labels == domain_label)[0]
drifted_datapoints = combined_data[indices_domain]
selected_domain_labels = combined_domain_labels[indices_domain]
original_indicies = indices_domain % int(0.2 * len(combined_data))
original_datapoints = X[original_indicies]
feature_rep = model.feature_extractor(drifted_datapoints)
generated_rep = model.generator([feature_rep, tf.convert_to_tensor(selected_domain_labels)])

normalized_drift = scaler.fit_transform(drifted_datapoints)
normalized_generated = scaler.fit_transform(generated_rep)

pca_drifted = pca.fit_transform(normalized_drift)
pca_original = pca.fit_transform(original_datapoints)
pca_generated = pca.fit_transform(normalized_generated)

# Plot PCA results
plt.figure(figsize=(10, 6))

# Plot drifted data
plt.scatter(pca_drifted[:, 0], pca_drifted[:, 1], c='blue', label=f'Data that has been influenced by the drift iter: {domain_label}')

# Plot original data
plt.scatter(pca_original[:, 0], pca_original[:, 1], c='red', label='Original Data')

# Plot generated data
plt.scatter(pca_generated[:, 0], pca_generated[:, 1], c='green', label='Generated Data')

# Add labels and legend
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
