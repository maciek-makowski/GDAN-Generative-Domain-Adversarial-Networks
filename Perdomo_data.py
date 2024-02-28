import numpy as np 
from scripts.data_prep_GMSC import load_data
from scripts.optimization import logistic_regression, evaluate_loss
from scripts.strategic import best_response

path = ".\GiveMeSomeCredit\cs-training.csv"

X,Y, data  = load_data(path)
n = X.shape[0]
d = X.shape[1] - 1

strat_features = np.array([1, 6, 8]) - 1 # for later indexing

# Description of the strategic features 
# print('Strategic Features: \n')
# for i, feature in enumerate(strat_features):
#     print(i, data.columns[feature + 1])

# fit logistic regression model we treat as the truth
lam = 1.0/n
theta_true, loss_list, smoothness = logistic_regression(X, Y, lam, 'Exact')

print('Accuracy: ', ((X.dot(theta_true) > 0)  == Y).mean())
print('Loss: ', loss_list[-1])


# Defining constants 
num_iters = 10
eps = 100
method = "RRM"
# initial theta
theta = np.copy(theta_true)

print('Running epsilon =  {}\n'.format(eps))

#X_strat = X

for t in range(num_iters):
    eps = np.random.uniform(0,1000)
    
    print("t", t, "\n")
    # adjust distribution to current theta
    X_strat = best_response(X, theta, eps, strat_features)
    
    # performative loss value of previous theta
    loss_start = evaluate_loss(X_strat, Y, theta, lam, strat_features)
    acc = ((X_strat.dot(theta) > 0) == Y).mean()
    print("ACC with old theta", acc)
    
    # learn on induced distribution
    theta_init = np.zeros(d+1) if method == 'Exact' else np.copy(theta)
    
    theta_new, ll, logistic_smoothness = logistic_regression(X_strat, Y, lam, 'Exact', tol=1e-7, 
                                                                theta_init=theta_init)
    
    

    # evaluate final loss on the current distribution
    loss_end = evaluate_loss(X_strat, Y, theta_new, lam, strat_features)
    acc = ((X_strat.dot(theta_new) > 0) == Y).mean()
    print("ACC with new theta", acc, "\n")

    theta = np.copy(theta_new)