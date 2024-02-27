import whynot.gym as gym
import numpy as np
import scripts.utils as utils
import pygad
import sys
import matplotlib.pyplot as plt

def perform_multiplication(importance_matrix, features):
    result_matrix = ([row.T * importance_matrix for row in features]) 
    return result_matrix


def calc_accuracy(features, labels, classifier):
    accuracy =((np.dot(features, classifier) > 0)  == labels).mean()
    return accuracy

env = gym.make('Credit-v0')
env.seed(1)

base_dataset = env.initial_state.values()
base_features, base_labels = base_dataset["features"], base_dataset["labels"]
num_agents, num_features = base_features.shape

l2_penalty = 1.0 / num_agents
baseline_theta = utils.fit_logistic_regression(base_features, base_labels, l2_penalty)
baseline_acc = ((base_features.dot(baseline_theta) > 0)  == base_labels).mean()

desired_accuracy = baseline_acc
print(f"Baseline logistic regresion model accuracy: {100 * baseline_acc}%")

theta = np.copy(baseline_theta)
env.config.epsilon = 150
env.config.l2_penalty = l2_penalty
env.reset()

observation, _, _, _ = env.step(theta)
features_strat, labels = observation["features"], observation["labels"]

baseline_acc = ((features_strat.dot(theta) > 0)  == labels).mean()

print(f"After env influence logistic regresion model accuracy: {100 * baseline_acc:.2f}%")

##### DEFINE GA PARAMETERS 

# initial_weight_matrix = np.ones((11, 20))

initial_weight_matrix = np.ones((20, 11))
# print(initial_weight_matrix)
print(initial_weight_matrix.shape)
# print(list(initial_weight_matrix))
# print(list(initial_weight_matrix.shape))
# initial_weight_matrix = initial_weight_matrix.reshape(-1,1)
# print(initial_weight_matrix.shape)
# print(initial_weight_matrix)
# print(initial_weight_matrix.T)
# print(initial_weight_matrix.T.shape)
# print(list(initial_weight_matrix))
# print(list(initial_weight_matrix.T.shape))
# sys.exit()

def fitness_func(ga_instance, solution, solution_idx):
    modified_features = ([row.T * solution for row in features_strat]) 
    accuracy = calc_accuracy(modified_features, labels, theta, desired_accuracy)
    if desired_accuracy - accuracy == 0:
        fitness = float('inf')
    else: 
        fitness = 1.0 / np.abs(desired_accuracy - accuracy)
    return fitness


fitness_function = fitness_func

num_generations = 20
num_parents_mating = 5

num_genes = 11
sol_per_pop = 20
gene_space = {'low': 0, 'high': 1}

parent_selection_type = "sss"
keep_parents = 2

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

best_solution = []

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

for _ in range(1000):
    # print("START POP",ga_instance.population)
    # print("START POP SHAPE",ga_instance.population.shape)
    ga_instance.run()
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Type solution", type(solution))
    print("shape of that shit", solution.shape)
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    prediction = calc_accuracy(perform_multiplication(solution, features_strat), labels, theta, desired_accuracy)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    best_solution.append(solution)

    # ga_instance.plot_fitness()
    # #ga_instance.plot_new_solution_rate()
    # ga_instance.plot_genes(graph_type = 'histogram')

np.savez('best_soluton_list.npz', *best_solution)

    
averaged_best_sol = np.zeros(11)
for i in best_solution:
    for index in range(len(i)):
        averaged_best_sol[index] += i[index]

averaged_best_sol = averaged_best_sol / 11
print("Average best soluton ", averaged_best_sol)

