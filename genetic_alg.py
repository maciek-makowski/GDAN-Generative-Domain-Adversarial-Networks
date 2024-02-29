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

def fitness_func(ga_instance, solution, solution_idx):
    modified_features = ([row.T * solution for row in features_strat]) 
    accuracy = calc_accuracy(modified_features, labels, theta)
    if desired_accuracy - accuracy == 0:
        fitness = float('inf')
    else: 
        fitness = 1.0 / np.abs(desired_accuracy - accuracy)
    return fitness

if __name__ == "__main__":

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

    #for _ in range(10):
    # print("START POP",ga_instance.population)
    # print("START POP SHAPE",ga_instance.population.shape)
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    prediction = calc_accuracy(perform_multiplication(solution, features_strat), labels, theta)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    best_solution.append(solution)


    normal_accuracies = []
    modified_accuracies = []
    for _ in range(10): 
        observation, _, _, _ = env.step(theta)
        features_strat, labels = observation["features"], observation["labels"]
        print("Strat features", features_strat[0:2,:])
        print("Labels", labels)

        normal_accuracy = calc_accuracy(features_strat, labels, theta)
        print("Normal accyracy", normal_accuracy)
        modified_accuracy = calc_accuracy(perform_multiplication(solution, features_strat), labels, theta)
        print("Modified accuracy", modified_accuracy)

        normal_accuracies.append(normal_accuracy)
        modified_accuracies.append(modified_accuracy)
        theta =  np.random.rand(11)


    # Plotting the lists
    plt.plot(normal_accuracies, label='non modified feature representation')
    plt.plot(modified_accuracies, label='modififed feature representation')

    # Adding a horizontal line
    plt.axhline(y=baseline_acc, color='r', linestyle='--', label='Baseline accuracy')

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
    # ga_instance.plot_fitness()
    # #ga_instance.plot_new_solution_rate()
    # ga_instance.plot_genes(graph_type = 'histogram')

    #np.savez('best_soluton_list.npz', *best_solution)

        
    # averaged_best_sol = np.zeros(11)
    # for i in best_solution:
    #     for index in range(len(i)):
    #         averaged_best_sol[index] += i[index]

    # averaged_best_sol = averaged_best_sol / 11
    # print("Average best soluton ", averaged_best_sol)

