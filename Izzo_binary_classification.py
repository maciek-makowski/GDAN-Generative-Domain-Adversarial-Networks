import sys 
import numpy as np 
import pygad
import matplotlib.pyplot as plt
from scripts.Izzo_utils import shift_dist, fit 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from scripts.DANN_training import DANN, train_dann_model
from sklearn.decomposition import PCA


num_iters = 10
no_samples = 10000
no_features = 11
strat_features = np.ones(no_features)
#theta = np.random.randn(no_features)
theta = np.ones(no_features)
print(theta)

### Try to specify the models better, cause there is divergence in accuracies between methods
model = LogisticRegression(C = 0.01, penalty='l2')
retrained_model = LogisticRegression(C = 0.01, penalty='l2')
DANN_model = DANN(11, 0.01)

X,Y = shift_dist(no_samples, theta, no_features, strat_features)
X_og = X
# ## Grid search resulted in l2 and 0.001 for C you can do it more extensively later
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 0.5, 1.0],  # Regularization parameter
#     'penalty': ['l2']  # Penalty type
# }


# grid_search = GridSearchCV(retrained_model, param_grid, cv=5, scoring='accuracy')

# grid_search.fit(X, Y)

# ##Get the best parameters and score
# best_estimator = grid_search.best_estimator_
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print("Best Parameters:", best_estimator.get_params())
# print("Best Estimator coefficients", best_estimator.coef_)
# print("Best Score:", best_score)

# test_score = grid_search.score(X, Y)
# print("Test Score:", test_score)

model.fit(X,Y)
theta = model.coef_[0].T

baseline_accuracy = accuracy_score(Y, model.predict(X))
print("Baseline accuracy",baseline_accuracy)

new_theta = theta.copy()

# Lists for plotting
regular_accuracies = []
retrained_accuracies = []
new_rep_accuracies = []
new_rep_DANN_model = []

# Lists for PCA plotting
X_modified_list = []
X_drifted_list = []

for t in range(num_iters):
    old_accuracy = accuracy_score(Y, model.predict(X))
    regular_accuracies.append(old_accuracy)
    print("accuracy with old model", old_accuracy)

    X_prev = X
    Y_prev = Y

    X,Y = shift_dist(no_samples, theta, no_features, strat_features)
    retrained_model.fit(X,Y)

    X_drift_scaled = (X - np.mean(X, axis = 0)) / np.std(X, axis=0)
    X_drifted_list.append(X_drift_scaled)

    if t == 0:
        #train_dann_model(DANN_model, X_prev, Y_prev, X, Y, num_epochs=6, data_generator="Izzo")
        DANN_model.load_weights('feature_extractor_weights_Izzo.weights.h5')
        

  

    retrained_accuracy = accuracy_score(Y, retrained_model.predict(X))
    retrained_accuracies.append(retrained_accuracy)
    print("accuracy after retraining", retrained_accuracy)
    
    modified_X = DANN_model.feature_extractor.predict(X)
    
    X_modified_scaled = (modified_X - np.mean(modified_X, axis = 0))/np.std(modified_X, axis = 0)
    X_modified_list.append(X_modified_scaled)

    new_rep_accuracy = accuracy_score(Y, model.predict(modified_X))
    new_rep_accuracies.append(new_rep_accuracy)
    print("accuracy with new representation", new_rep_accuracy)

    predictions_DANN_model = np.where(DANN_model.label_classifier.predict(modified_X) > 0.5, 1,0)
    acc = accuracy_score(Y, predictions_DANN_model)
    new_rep_DANN_model.append(acc)
    print("accuracy new rep DANN model",acc)

    


# Plotting the lists
plt.plot(regular_accuracies, label='Accuracy with first model')
plt.plot(retrained_accuracies, label='Retrained accuracy')
plt.plot(new_rep_accuracies, label = 'Accuracy with modified representation')
plt.plot(new_rep_DANN_model, label = 'Accuracies with DANN model')


# Adding a horizontal line
plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline accuracy')

# Adding titles for the axes
plt.xlabel('Iterations of new data being generated')
plt.ylabel('Accuracy values')
plt.title('Performance with generation of a DANN transformation on data from Izzo et al. 2022')
# Adding a legend
plt.legend()

# Turning grid on
plt.grid(True)

# Displaying the plot
plt.show()

pca = PCA(n_components=2)
pca_data = {"mod":[],"drift":[]}
for i in range(len(X_modified_list)):
    pca_data['mod'].append(pca.fit_transform(X_modified_list[i]))
    pca_data['drift'].append(pca.fit_transform(X_drifted_list[i]))

pca_og = pca.fit_transform(X_og)

fig, axs = plt.subplots(1,2, figsize=(10, 10))

axs[0].scatter(pca_data['drift'][0][:,0], pca_data['drift'][0][:,1], alpha=0.5, label=f'First iteration drift')
axs[0].scatter(pca_data['drift'][-1][:,0], pca_data['drift'][-1][:,1], alpha=0.5, label=f'Last iteration drift')
axs[0].scatter(pca_og[:,0], pca_og[:,1], alpha = 0.5, label = f'Original data distribution')

axs[1].scatter(pca_data['mod'][0][:,0], pca_data['mod'][0][:,1], alpha=0.5, label=f'First iteration modified')
axs[1].scatter(pca_data['drift'][0][:,0], pca_data['drift'][0][:,1], alpha=0.5, label=f'First iteration drift')
#axs[1].scatter(pca.fit_transform(X)[:,0], pca.fit_transform(X)[:,1], alpha=0.5, label=f'Distribution og')


axs[0].set_xlabel('PCA 1')
axs[0].set_ylabel('PCA 2')
axs[0].set_title(f'Drift effects on the distribution', loc = 'center')
axs[1].set_xlabel('PCA 1')
axs[1].set_ylabel('PCA 2')
axs[1].set_title(f'DANN transformation effects on the distribution', loc = 'center')
axs[0].legend()
axs[1].legend()

plt.suptitle("Principal component analysis - data Izzo 2022")
plt.tight_layout()
plt.show()