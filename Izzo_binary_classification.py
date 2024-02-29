import sys 
import numpy as np 
import pygad
import matplotlib.pyplot as plt
from scripts.Izzo_utils import shift_dist, fit 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def fitness_func(ga_instance, solution, solution_idx):
    normalized_solution = solution / np.sum(solution)

    modified_features = ([row.T * normalized_solution for row in X]) 
    accuracy = accuracy_score(Y, model.predict(modified_features))
    if baseline_accuracy - accuracy == 0:
        fitness = float('inf')
    else: 
        fitness = (1.0 / np.abs(baseline_accuracy - accuracy)) + (1.0/(1.0 - np.sum(solution)))
    return fitness


num_iters = 10
no_samples = 10000
no_features = 2

#theta = np.random.randn(no_features)
theta = [0,0]
print(theta)

### Try to specify the models better, cause there is divergence in accuracies between methods
model = LogisticRegression(C = 0.01, penalty='l2')
retrained_model = LogisticRegression(C = 0.01, penalty='l2')

X,Y = shift_dist(no_samples, theta)

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

# Define genetic algorithm to lern the mappings 
sol_per_pop = 40
num_genes = no_features
initial_weight_matrix = np.ones((sol_per_pop, no_features))
fitness_function = fitness_func


num_generations = 40
num_parents_mating = 15
gene_space = {'low': 0, 'high': 1}
parent_selection_type = "sss"
keep_parents = 4
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 20

ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        num_genes=num_genes,
                        gene_space=gene_space,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        save_solutions = True,
                        initial_population= initial_weight_matrix
                        )


# Lists for plotting
regular_accuracies = []
retrained_accuracies = []
new_rep_accuracies = []

for t in range(num_iters):
    old_accuracy = accuracy_score(Y, model.predict(X))
    regular_accuracies.append(old_accuracy)
    print("accuracy with old model", old_accuracy)


    if t == 0:
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        ga_instance.plot_fitness()


        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    X,Y = shift_dist(no_samples, theta)
    retrained_model.fit(X,Y)

    retrained_accuracy = accuracy_score(Y, retrained_model.predict(X))
    retrained_accuracies.append(retrained_accuracy)
    print("accuracy after retraining", retrained_accuracy)
    
    modified_X = np.array([solution * x for x in X])
    print("X", X[0:2,:])
    print("Modified X", modified_X[0:2,:])

    new_rep_accuracy = accuracy_score(Y, model.predict(modified_X))
    new_rep_accuracies.append(new_rep_accuracy)
    print("accuracy with new representation", new_rep_accuracy)

    


# Plotting the lists
plt.plot(regular_accuracies, label='Accuracy with first model')
plt.plot(retrained_accuracies, label='Retrained accuracy')
plt.plot(new_rep_accuracies, label = 'Accuracy with modified representation')


# Adding a horizontal line
plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline accuracy')

# Adding titles for the axes
plt.xlabel('Iterations of new data being generated')
plt.ylabel('Accuracy values')
plt.title('Performance with generation of a linear transformation on data from Izzo et al. 2022')
# Adding a legend
plt.legend()

# Turning grid on
plt.grid(True)

# Displaying the plot
plt.show()