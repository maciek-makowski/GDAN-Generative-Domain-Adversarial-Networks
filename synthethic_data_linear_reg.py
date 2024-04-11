import numpy as np
import pygad
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scripts.DANN_training import DANN, train_dann_model


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

##Define the number of features
no_features = 11
## Define starting paraeter vector of a model
theta = np.ones(no_features)
## Define the sensitivity parameter 
epsilon = 0.9
## Define the num_samples
num_samples = 1000
num_iter = 10

## Define the linear regression model 
model = LinearRegression()
retrained_model = LinearRegression()

mu = generate_vector_with_norm(no_features, epsilon)
covariance_matrix = generate_covariance_matrix(no_features)
## Initialize DANN model
DANN_model = DANN(no_features, task='regression')


## Definition of arrays for plotting 
best_MSE = []
best_modified_MSE = []
best_retrained_MSE = []
## The looping proces  
for i in range(num_iter):
    X,Y = generate_data(theta, num_samples, mu, covariance_matrix)

    # print(X,Y)
    if i ==0 :
        X_prev = X
        Y_prev = Y 
        model.fit(X,Y)
        y_pred = model.predict(X)
        mean_squared_error_ = mean_squared_error(Y, y_pred)
        print("First MSE ",mean_squared_error_)
    elif i == 1:
        train_dann_model(DANN_model, X_prev,Y_prev, X,Y, num_epochs=6, data_generator='linreg')
        DANN_model.load_weights('feature_extractor_weights_linreg.weights.h5')

        temp_new_MSE = mean_squared_error(Y, model.predict(X))
        temp_new_modified_MSE =  mean_squared_error(Y, DANN_model.label_classifier.predict(X))
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
        temp_new_modified_MSE =  mean_squared_error(Y, DANN_model.label_classifier.predict(X))
    
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