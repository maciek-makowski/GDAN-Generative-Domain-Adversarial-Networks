import numpy as np
import pygad
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def generate_data(parameter_vector, num_samples, mu, covariance_matrix):
    d = len(parameter_vector)

    X = np.random.multivariate_normal(np.zeros(d), cov= covariance_matrix, size = num_samples)

    noise_variance = 0.01
    noise = np.random.normal(0, np.sqrt(noise_variance), num_samples)

    # print("noise", noise)
    # print("Noise shape", noise.shape)

    beta = np.random.multivariate_normal(np.zeros(d), cov = np.eye(d))

    # print("beta", beta)
    # print("Beta shape", beta.shape)

    # print("mu", mu)
    # print("Mu shape", mu.shape)
    
    Y = X @ beta + parameter_vector @ mu + noise
    return X, Y

def generate_covariance_matrix(d, norm=0.01):
    ## Generate random matrix
    A = np.random.randn(d, d)
    ## Multiply it by its transpose so it's symmetric
    covariance_matrix = A @ A.T
    ## Calculate the norm to normalize the matrix and then multiply by the desired value of the norm 
    covariance_matrix /= np.linalg.norm(covariance_matrix, ord=2) / norm

    return covariance_matrix

def generate_vector_with_norm(d, norm):
    # Generate a random vector
    vector = np.random.randn(d)
    
    # Calculate the current norm of the vector
    current_norm = np.linalg.norm(vector, ord = 2)
    
    # Scale the vector to achieve the desired norm
    scaled_vector = vector * (norm / current_norm)
    
    return scaled_vector

def perform_elementwise_multiplication(data, weight_vector):
    result = np.array([row.T * weight_vector for row in data])
    return result

def fitness_func(ga_instance, solution, solution_idx):
    normalized_solution = solution / np.sum(solution)

    modified_data = perform_elementwise_multiplication(X, normalized_solution)
    modified_predictions = model.predict(modified_data)
    new_MSE = mean_squared_error(Y, modified_predictions)
    if new_MSE == mean_squared_error_:
        fitness = float('inf')
    else: 
        fitness = 1.0 / np.abs(new_MSE - mean_squared_error_)
    return fitness
##Define the number of features
no_features = 10
## Define starting parameter vector of a model
theta = np.ones(no_features)
## Define the sensitivity parameter 
epsilon = 0.9
## Define the num_samples
num_samples = 1000

## Define the linear regression model 
model = LinearRegression()
retrained_model = LinearRegression()

mu = generate_vector_with_norm(no_features, epsilon)
covariance_matrix = generate_covariance_matrix(no_features)
################### INITIALZING GA ########################################
sol_per_pop = 40
num_genes = no_features 
initial_weight_matrix = np.ones((sol_per_pop, no_features))
fitness_function = fitness_func


num_generations = 100
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

## Definition of arrays for plotting 
best_MSE = []
best_modified_MSE = []
best_retrained_MSE = []
## The looping proces  
for i in range(10):
    X,Y = generate_data(theta, num_samples, mu, covariance_matrix)
    # print(X,Y)
    if i ==0 :
        model.fit(X,Y)
        y_pred = model.predict(X)
        mean_squared_error_ = mean_squared_error(Y, y_pred)
        print("First MSE ",mean_squared_error_)
    elif i == 1:
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        ga_instance.plot_fitness()

        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        

        temp_new_MSE = mean_squared_error(Y, model.predict(X))
        temp_new_modified_MSE =  mean_squared_error(Y, model.predict(perform_elementwise_multiplication(X, solution)))
        print("MSE before modification of the featues", temp_new_MSE)
        print("MSE after modification of the features", temp_new_modified_MSE)

        retrained_model.fit(X,Y)
        retrained_MSE = mean_squared_error(Y, retrained_model.predict(X))
        
        best_MSE.append(temp_new_MSE)
        best_modified_MSE.append(temp_new_modified_MSE)
        best_retrained_MSE.append(retrained_MSE)

        print("")
        #theta = model.coef_
    else: 
        temp_new_MSE = mean_squared_error(Y, model.predict(X))
        temp_new_modified_MSE =  mean_squared_error(Y, model.predict(perform_elementwise_multiplication(X, solution)))
    
        print("MSE before modification of the featues", temp_new_MSE )
        print("MSE after modification of the features", temp_new_modified_MSE)

        retrained_model.fit(X,Y)
        retrained_MSE = mean_squared_error(Y, retrained_model.predict(X))

        print("MSE after retraining", retrained_MSE)

        best_MSE.append(temp_new_MSE)
        best_modified_MSE.append(temp_new_modified_MSE)
        best_retrained_MSE.append(retrained_MSE)

        

        print("")



### PLOTTING 
        
# Plotting the lists
plt.plot(best_MSE, label='non modified feature representation')
plt.plot(best_modified_MSE, label='modififed feature representation')
plt.plot(best_retrained_MSE, label='retrained model')

# Adding a horizontal line
plt.axhline(y=mean_squared_error_, color='r', linestyle='--', label='Baseline MSE')

# Adding titles for the axes
plt.xlabel('Iterations of new data being generated')
plt.ylabel('Values of the MSE')
plt.title('Performance with generation of a linear transformation on data from Miller et al. 2021')
# Adding a legend
plt.legend()

# Turning grid on
plt.grid(True)

# Displaying the plot
plt.show()