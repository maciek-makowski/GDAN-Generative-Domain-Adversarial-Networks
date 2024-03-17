import numpy as np 
import matplotlib.pyplot as plt
from scripts.data_prep_GMSC import load_data
from scripts.optimization import logistic_regression, evaluate_loss
from scripts.strategic import best_response
from scripts.DANN_training import DANN, train_dann_model
from sklearn.metrics import accuracy_score
from adapt.feature_based import TCA



path = ".\GiveMeSomeCredit\cs-training.csv"

X,Y, data  = load_data(path)

# X = X[:5]
# Y = Y[:5]

n = X.shape[0]
d = X.shape[1] - 1


strat_features = np.array([1, 6, 8]) - 1 # for later indexings

# Description of the strategic features 
# print('Strategic Features: \n')
# for i, feature in enumerate(strat_features):
#     print(i, data.columns[feature + 1])

# fit logistic regression model we treat as the truth
lam = 1.0/n
theta_true, loss_list, smoothness = logistic_regression(X, Y, lam, 'Exact')

baseline_accuracy = ((X.dot(theta_true) > 0)  == Y).mean()

print('Accuracy: ', baseline_accuracy)
print('Loss: ', loss_list[-1])

# Defining constants 
num_iters = 85
eps = 100
method = "RRM"
# initial theta
theta = np.copy(theta_true)

# Define stuff for training the DANN 
model = DANN(11, 0.01)


X_strat = X

#Define lists for plotting 
accuracy_list = []
retrained_accuracy = []
new_rep_accuracy = []
new_model_accuracy = []

for t in range(num_iters):
    #eps = np.random.uniform(0,100)
    eps = 10

    print("t", t, "\n")
    # adjust distribution to current theta
    X_strat = best_response(X_strat, theta, eps, strat_features)
    
    if t ==0:
        #train_dann_model(model, X, Y, X_strat, Y, num_epochs=6)
        model.load_weights('feature_extractor_weights.weights.h5')

    # performative loss value of previous theta
    loss_start = evaluate_loss(X_strat, Y, theta, lam, strat_features)
    acc = ((X_strat.dot(theta) > 0) == Y).mean()
    print("ACC with old theta", acc)
    accuracy_list.append(acc)
    
    # learn on induced distribution
    theta_init = np.zeros(d+1) if method == 'Exact' else np.copy(theta)
    
    theta_new, ll, logistic_smoothness = logistic_regression(X_strat, Y, lam, 'Exact', tol=1e-7, 
                                                                theta_init=theta_init)
    
    

    # evaluate final loss on the current distribution
    loss_end = evaluate_loss(X_strat, Y, theta_new, lam, strat_features)
    acc = ((X_strat.dot(theta_new) > 0) == Y).mean()
    print("ACC with new theta", acc)
    retrained_accuracy.append(acc)

    # evalute on new feature representation 
    X_modified = model.feature_extractor(X_strat)
    #print("X_modififed_before", X_strat[:5])
    X_modified = model.feature_extractor.predict(X_strat)
    #print("X_modified", X_modified[:5], "X mod shape", X_modified.shape)
    acc = ((X_modified.dot(theta) > 0) == Y).mean()
    #acc = accuracy_score(Y, np.where(X_modi > 0.5, 1, 0)))
    print("ACC with modified_rep", acc)
    new_rep_accuracy.append(acc)

    # evaluate new features with the classifier leart by the DANN 
    predictions_of_new_classifier = np.where(model.label_classifier.predict(X_modified) > 0.5, 1, 0)
    acc = accuracy_score(Y, predictions_of_new_classifier)
    print("ACC with different classifier", acc)
    new_model_accuracy.append(acc)

    #theta = np.copy(theta_new)
        

#new_rep_accuracy.insert(0, baseline_accuracy)
#print("First, retrained, modified", accuracy_list, retrained_accuracy, new_rep_accuracy)
for elements in zip(accuracy_list, retrained_accuracy, new_rep_accuracy, new_model_accuracy):
    print(*elements)

# Plotting the lists
plt.plot(accuracy_list, label='Accuracy with first model')
plt.plot(retrained_accuracy, label='Retrained accuracy')
plt.plot(new_rep_accuracy, label = 'Accuracy with modified representation')
plt.plot(new_model_accuracy, label = 'Accuracy with DANN model')


# Adding a horizontal line
plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline accuracy')

# Adding titles for the axes
plt.xlabel('Iterations of new data being generated')
plt.ylabel('Accuracy values')
plt.title(f'Performance on data from Perdomo et al. 2020 Epsilon :{eps}')
# Adding a legend
plt.legend()

# Turning grid on
plt.grid(True)

# Displaying the plot
plt.show()