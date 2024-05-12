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
    


train_architecture(model, X, X_iterations, Y_iterations)
# model.load_weights("GDANN_arch.weights.h5")

# domain_labels = []
# len_single_domain = len(X_iterations[0])
# for i, datapoints in enumerate(X_iterations):
#     domain_labels.extend([i] * len(datapoints))
    
#     combined_domain_labels = np.array(domain_labels)   
#     combined_data = np.concatenate(X_iterations)
#     combined_class_labels = np.concatenate(Y_iterations) 


# for _ in range(100):
#     random_index = np.random.randint(0, len(combined_data))

#     random_data_point = combined_data[random_index].reshape(-1,1)
#     random_domain_label = combined_domain_labels[random_index]
#     random_class_label = combined_class_labels[random_index]

#     print("RDP", random_data_point)

#     feature_rep = model.feature_extractor(random_data_point.T)
#     generated_rep = model.generator([feature_rep, tf.convert_to_tensor(random_domain_label.reshape(-1,1))])

#     plt.scatter(np.arange(11), random_data_point, label='OG')
#     plt.scatter(np.arange(11), generated_rep, label='Generated')  # Adding an offset for better visualization
#     plt.xlabel('Index')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.show()


