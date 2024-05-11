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
scaler = MinMaxScaler()
logreg = LogisticRegression()

#Lists for storing differnt data distributions
X_iterations = []
Y_iterations = []

X = scaler.fit_transform(X)
X_strat = X

for t in range(num_iters):
    X_strat = best_response(X_strat, theta, eps, strat_features)
    X_iterations.append(scaler.fit_transform(X_strat))
    Y_iterations.append(Y)
    

train_architecture(model, X, X_iterations, Y_iterations)