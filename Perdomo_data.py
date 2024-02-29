import pygad
import numpy as np 
import matplotlib.pyplot as plt
from scripts.data_prep_GMSC import load_data
from scripts.optimization import logistic_regression, evaluate_loss
from scripts.strategic import best_response


def perform_multiplication(importance_matrix, features):
    result_matrix = ([row.T * importance_matrix for row in features]) 
    return result_matrix


def calc_accuracy(features, labels, classifier):
    accuracy =((np.dot(features, classifier) > 0)  == labels).mean()
    return accuracy

def fitness_func(ga_instance, solution, solution_idx):
    modified_features = ([row.T * solution for row in X_strat]) 
    accuracy = calc_accuracy(modified_features, Y, theta)
    if baseline_accuracy - accuracy == 0:
        fitness = float('inf')
    else: 
        fitness = 1.0 / np.abs(baseline_accuracy - accuracy)
    return fitness


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

baseline_accuracy = ((X.dot(theta_true) > 0)  == Y).mean()

print('Accuracy: ', baseline_accuracy)
print('Loss: ', loss_list[-1])


# Defining constants 
num_iters = 10
eps = 100
method = "RRM"
# initial theta
theta = np.copy(theta_true)


# Define genetic algorithm to lern the mappings 
sol_per_pop = 20
num_genes = d + 1
initial_weight_matrix = np.ones((sol_per_pop, d+1))
fitness_function = fitness_func


num_generations = 20
num_parents_mating = 15
gene_space = {'low': 0, 'high': 1}
parent_selection_type = "sss"
keep_parents = 4
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

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



X_strat = X

#Define lists for plotting 
accuracy_list = []
retrained_accuracy = []
new_rep_accuracy = []

for t in range(num_iters):
    eps = np.random.uniform(0,100)

    if t==1:
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        ga_instance.plot_fitness()

        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    
    print("t", t, "\n")
    # adjust distribution to current theta
    X_strat = best_response(X, theta, eps, strat_features)
    
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
    if t > 0: 
        modified_accuracy = calc_accuracy(perform_multiplication(solution, X_strat), Y, theta)
        print("ACC wth different feature rep", modified_accuracy, "\n")
        new_rep_accuracy.append(modified_accuracy)


    #theta = np.copy(theta_new)
        

new_rep_accuracy.insert(0, baseline_accuracy)
#print("First, retrained, modified", accuracy_list, retrained_accuracy, new_rep_accuracy)
for elements in zip(accuracy_list, retrained_accuracy, new_rep_accuracy):
    print(*elements)
# Plotting the lists
plt.plot(accuracy_list, label='Accuracy with first model')
plt.plot(retrained_accuracy, label='Retrained accuracy')
plt.plot(new_rep_accuracy, label = 'Accuracy with modified representation')


# Adding a horizontal line
plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline accuracy')

# Adding titles for the axes
plt.xlabel('Iterations of new data being generated')
plt.ylabel('Accuracy values')
plt.title('Performance with generation of a linear transformation on data from Perdomo et al. 2020')
# Adding a legend
plt.legend()

# Turning grid on
plt.grid(True)

# Displaying the plot
plt.show()