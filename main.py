import whynot as wn
import whynot.gym as gym
import scripts.utils as utils
import numpy as np
import do_mpc
import casadi as ca
import matplotlib.pyplot as plt
import logging
import random

# Configure the logger
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO

# Create a logger instance
logger = logging.getLogger(__name__)

def perform_multiplication(importance_matrix, features, x,y):
    # result_matrix = ca.MX()
    # for row in features: 
    #     new_row = row.T * importance_matrix       
    #     result_matrix = ca.vertcat(result_matrix, new_row)

    result_matrix = ([row.T * importance_matrix for row in features]) 

    print(result_matrix[0])

    print("LENGTH", len(result_matrix))
    num_rows = 18357
    num_cols = 11
    sx_matrix = ca.SX.zeros(num_rows * num_cols)

    # Reshape the matrix to the desired size
    reshaped_matrix = ca.reshape(sx_matrix, num_rows, num_cols)

    for index, row in enumerate(result_matrix):
        reshaped_matrix[index, :]= row 
    # print("Tyep", type(result_matrix), result_matrix[0])   
    # # result_matrix = ca.vertcat(*[row.T * importance_matrix for row in features]) 
    # # transform_function = ca.Function('trans', [importance_matrix], [result_matrix])
    # # final_result_matrix = transform_function(importance_matrix)

    # print(final_result_matrix[0:11])
    # #final_result_matrix = ca.reshape(final_result_matrix, 18357, 11)

    # matrix_reshaped = ca.horzsplit(ca.vertcat(final_result_matrix), [0,11,1])
    # # matrix_result = ca.horzcat(*matrix_reshaped)
    # # final_result_matrix = matrix_result[:18357, :]
    # print(matrix_reshaped[0,:])
    # print("ELOOOO")

    return reshaped_matrix

def calc_accuracy(features, labels, classifier):
    ### Evaluates model performance
    #temp = ca.dot(features, classifier) > 0

    #print("Featues type", type(features), features.size1(), features.size2())
    length = labels.shape[0]
    classifier = classifier.reshape(-1,1)
    logger.info(f"Features 0,5 : {features[0,:]} {type(features)} {type(classifier)}")

    temp = ca.mtimes(features, classifier)    
    logger.info(f"temp 0,5  {temp[0,:]}")

    pred = temp > 0 
    logger.info(f"Pred 0,5 {pred[0,:]}")
    corr_pred = ca.eq(pred, labels)
    logger.info(f"Corr_pred {corr_pred[0,:]}")
    # pdb.set_trace()
    #logger.info(f"acc {ca.sum1(corr_pred)}")
    accuracy = ca.sum1(corr_pred)/ length
    
    return accuracy

env = gym.make('Credit-v0')
env.seed(1)

base_dataset = env.initial_state.values()
#print(base_dataset)
base_features, base_labels = base_dataset["features"], base_dataset["labels"]
num_agents, num_features = base_features.shape
#print(f"The dataset has {num_agents} agents and {num_features} features.")
print("Shape features", base_features.shape)
print("Shape labels", base_labels.shape)


l2_penalty = 1.0 / num_agents
baseline_theta = utils.fit_logistic_regression(base_features, base_labels, l2_penalty)

print("Baseline theta", baseline_theta)
print("Base features", base_features[0,:])
print("result", base_features.dot(baseline_theta)[0])
print("label 0 ", base_labels[0])

baseline_acc = ((base_features.dot(baseline_theta) > 0)  == base_labels).mean()

desired_accuracy = baseline_acc

print(f"Baseline logistic regresion model accuracy: {100 * baseline_acc:.2f}%")

theta = np.copy(baseline_theta)

env.config.epsilon = 150
env.config.l2_penalty = l2_penalty
env.reset()

observation, _, _, _ = env.step(theta)
features_strat, labels = observation["features"], observation["labels"]

baseline_acc = ((features_strat.dot(theta) > 0)  == labels).mean()

print(f"After env influence logistic regresion model accuracy: {100 * baseline_acc:.2f}%")

model = do_mpc.model.Model('discrete')

weight_importance_0 = model.set_variable(var_type = '_u', var_name = 'weight_importance_0', shape=(1,1))
weight_importance_1 = model.set_variable(var_type = '_u', var_name = 'weight_importance_1', shape=(1,1))
weight_importance_2 = model.set_variable(var_type = '_u', var_name = 'weight_importance_2', shape=(1,1))
weight_importance_3 = model.set_variable(var_type = '_u', var_name = 'weight_importance_3', shape=(1,1))
weight_importance_4 = model.set_variable(var_type = '_u', var_name = 'weight_importance_4', shape=(1,1))
weight_importance_5 = model.set_variable(var_type = '_u', var_name = 'weight_importance_5', shape=(1,1))
weight_importance_6 = model.set_variable(var_type = '_u', var_name = 'weight_importance_6', shape=(1,1))
weight_importance_7 = model.set_variable(var_type = '_u', var_name = 'weight_importance_7', shape=(1,1))
weight_importance_8 = model.set_variable(var_type = '_u', var_name = 'weight_importance_8', shape=(1,1))
weight_importance_9 = model.set_variable(var_type = '_u', var_name = 'weight_importance_9', shape=(1,1))
weight_importance_10 = model.set_variable(var_type = '_u', var_name = 'weight_importance_10', shape=(1,1))

current_accuracy = model.set_variable(var_type = '_x', var_name='current_accuracy')
#features = model.set_variable(var_type = '_x', var_name = 'features', shape = (18357,11))

weight_importance = ca.vertcat(
                weight_importance_0,
                weight_importance_1,
                weight_importance_2,
                weight_importance_3,
                weight_importance_4,
                weight_importance_5,
                weight_importance_6,
                weight_importance_7,
                weight_importance_8,
                weight_importance_9,
                weight_importance_10

)
#model.set_rhs('features', features)
model.set_rhs('current_accuracy', calc_accuracy(perform_multiplication(weight_importance, features_strat, 1, 18357), labels, theta))

model.setup()

print("Model setup")

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 100,
    't_step': 1,
    'n_robust': 20,
    'store_full_solution': True
}
mpc.set_param(**setup_mpc)

for i in range(11):
    mpc.bounds['lower', '_u', f'weight_importance_{i}'] = 0
    mpc.bounds['upper', '_u', f'weight_importance_{i}'] = 1


print("Desired accuracy before", desired_accuracy)
lterm = (desired_accuracy - current_accuracy) * (desired_accuracy - current_accuracy)
#lterm = ca.vertcat([0])

#mterm = ca.vertcat([0])
mterm = (desired_accuracy - current_accuracy) * (desired_accuracy - current_accuracy)

mpc.set_objective(mterm = mterm, lterm = lterm)

mpc.set_rterm(
    weight_importance_0=random.random() * 1e-3,
    weight_importance_1=random.random() * 1e-3,
    weight_importance_2=random.random() * 1e-3,
    weight_importance_3=random.random() * 1e-3,
    weight_importance_4=random.random() * 1e-3,
    weight_importance_5=random.random() * 1e-3,
    weight_importance_6=random.random() * 1e-3,
    weight_importance_7=random.random() * 1e-3,
    weight_importance_8=random.random() * 1e-3,
    weight_importance_9=random.random() * 1e-3,
    weight_importance_10=random.random() * 1e-3,
)

mpc.setup()

print("MPC setup")

simulator = do_mpc.simulator.Simulator(model)

simulator.set_param(t_step = 1)

simulator.setup()

print("Simulator setup")

mpc.x0['current_accuracy'] = baseline_acc

simulator.x0['current_accuracy'] = baseline_acc

mpc.set_initial_guess()

#u0 = np.ones(11)
u0 = np.random.rand(11).reshape(-1,1)
x0 = mpc.x0

print("Everything before for setup")

print("initialized value accuracy", mpc.x0['current_accuracy'])
print("U0 ", u0)




for i in range(5):
    x0 = simulator.make_step(u0)
    u0 = mpc.make_step(x0)
    print("Current accuracy", mpc.x0['current_accuracy'])
    print("U0", u0)
    print("X0", x0)

    print("Verification")
    new_features = np.empty_like(features_strat)
    for index, row in enumerate(features_strat):
        row = row.reshape(-1,1)
        new_features[index, :] = (row * u0).reshape(11)


    baseline_acc = ((new_features.dot(theta) > 0)  == labels).mean()
    print("ACC", baseline_acc)


#### GRAPHICS 
mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

fig, ax = plt.subplots(1, figsize=(16,9))
for g in [sim_graphics]: ##mpc_graphics
    # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
    g.add_line(var_type='_x', var_name='current_accuracy', axis=ax)

ax.set_ylabel('angle position [rad]')
# mpc_graphics.plot_predictions()
sim_graphics.plot_results()

ax.legend
plt.show()