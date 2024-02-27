import whynot as wn
import whynot.gym as gym
import scripts.utils as utils
import numpy as np
import do_mpc
import casadi as ca
import matplotlib.pyplot as plt
import os
import logging
from do_mpc.data import save_results, load_results
import random
import sys


# Configure the logger
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO

# Create a logger instance
logger = logging.getLogger(__name__)

def perform_multiplication(importance_matrix, features, x,y):
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

    return reshaped_matrix

def calc_accuracy(features, labels, classifier):
    ### Evaluates model performance
    #temp = ca.dot(features, classifier) > 0

    #print("Featues type", type(features), features.size1(), features.size2())
    length = labels.shape[0]
    classifier = classifier.reshape(-1,1)
    temp = ca.mtimes(features, classifier)    
    pred = temp > 0 
    corr_pred = ca.eq(pred, labels)
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


UP_0 = model.set_variable('_p', 'UP_0')
UP_1 = model.set_variable('_p', 'UP_1')
UP_2 = model.set_variable('_p', 'UP_2')
UP_3 = model.set_variable('_p', 'UP_3')
UP_4 = model.set_variable('_p', 'UP_4')
UP_5 = model.set_variable('_p', 'UP_5')
UP_6 = model.set_variable('_p', 'UP_6')
UP_7 = model.set_variable('_p', 'UP_7')
UP_8 = model.set_variable('_p', 'UP_8')
UP_9 = model.set_variable('_p', 'UP_9')
UP_10 = model.set_variable('_p', 'UP_10')

# TVP_0 = model.set_variable('_tvp', 'TVP_0')
# TVP_1 = model.set_variable('_tvp', 'TVP_1')
# TVP_2 = model.set_variable('_tvp', 'TVP_2')
# TVP_3 = model.set_variable('_tvp', 'TVP_3')
# TVP_4 = model.set_variable('_tvp', 'TVP_4')
# TVP_5 = model.set_variable('_tvp', 'TVP_5')
# TVP_6 = model.set_variable('_tvp', 'TVP_6')
# TVP_7 = model.set_variable('_tvp', 'TVP_7')
# TVP_8 = model.set_variable('_tvp', 'TVP_8')
# TVP_9 = model.set_variable('_tvp', 'TVP_9')
# TVP_10 = model.set_variable('_tvp', 'TVP_10')


weight_imp_state = model.set_variable(var_type= '_x', var_name = 'weight_imp_state', shape=(11,1))
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

# time_params = ca.vertcat(
#                 TVP_0,
#                 TVP_1,
#                 TVP_2,
#                 TVP_3,
#                 TVP_4,
#                 TVP_5,
#                 TVP_6,
#                 TVP_7,
#                 TVP_8,
#                 TVP_9,
#                 TVP_10

# )

uncertain_params = ca.vertcat(
                UP_0,
                UP_1,
                UP_2,
                UP_3,
                UP_4,
                UP_5,
                UP_6,
                UP_7,
                UP_8,
                UP_9,
                UP_10

)
#model.set_rhs('features', features)
model.set_rhs('current_accuracy', calc_accuracy(perform_multiplication(weight_imp_state, features_strat, 1, 18357), labels, theta))
model.set_rhs('weight_imp_state', weight_imp_state + 0.001 * (weight_importance * uncertain_params ))


model.setup()

print("Model setup")

mpc = do_mpc.controller.MPC(model)

prediction_horizon = 3

setup_mpc = {
    'n_horizon': prediction_horizon,
    't_step': 1,
    'n_robust': 1,
    'store_full_solution': True
}
mpc.set_param(**setup_mpc)

mpc.bounds['lower', '_x', 'weight_imp_state'] = 0
mpc.bounds['upper', '_x', 'weight_imp_state'] = 1

for i in range(11):
    mpc.bounds['lower', '_u', f'weight_importance_{i}'] = -1
    mpc.bounds['upper', '_u', f'weight_importance_{i}'] = 1

### UNCERTAIN PARAMETERS 

n_combinations = 20

p_template = mpc.get_p_template(n_combinations)

print(p_template)

def p_fun(t_now):
    for i in range(n_combinations):
        p_template['_p', i] = np.random.normal(1, 0.1, 11)
    return p_template

mpc.set_p_fun(p_fun)

###     TIME VARYING PARAMERTERS 
# tvp_template = mpc.get_tvp_template()

# print(tvp_template['_tvp'])

# def tvp_fun(t_now):
#     for i in  range(prediction_horizon):
#         tvp_template['_tvp',i] = np.random.uniform(0,2,11)
#     return tvp_template
 
# # p_template['_p',0] = np.random.rand(11)


# mpc.set_tvp_fun(tvp_fun)



print("Desired accuracy before", desired_accuracy)
norm_result = ca.norm_2(
                ca.vertcat(
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

)

lterm = 1 - norm_result
#lterm = (desired_accuracy - current_accuracy) * (desired_accuracy - current_accuracy)
#lterm = ca.vertcat([0])

# mterm = ca.vertcat([0])
mterm = 10 * (desired_accuracy - current_accuracy)

mpc.set_objective(mterm = mterm, lterm = lterm)
# mpc.set_rterm(
#     weight_importance_0=random.random() * 1e-3,
#     weight_importance_1=random.random() * 1e-3,
#     weight_importance_2=random.random() * 1e-3,
#     weight_importance_3=random.random() * 1e-3,
#     weight_importance_4=random.random() * 1e-3,
#     weight_importance_5=random.random() * 1e-3,
#     weight_importance_6=random.random() * 1e-3,
#     weight_importance_7=random.random() * 1e-3,
#     weight_importance_8=random.random() * 1e-3,
#     weight_importance_9=random.random() * 1e-3,
#     weight_importance_10=random.random() * 1e-3,
# )
# mpc.set_rterm(
#     weight_importance_0= 1e-2,
#     weight_importance_1= 1e-2,
#     weight_importance_2= 1e-2,
#     weight_importance_3= 1e-2,
#     weight_importance_4= 1e-2,
#     weight_importance_5= 1e-2,
#     weight_importance_6= 1e-2,
#     weight_importance_7= 1e-2,
#     weight_importance_8= 1e-2,
#     weight_importance_9= 1e-2,
#     weight_importance_10= 1e-2,
# )

mpc.setup()

print("MPC setup")

simulator = do_mpc.simulator.Simulator(model)

simulator.set_param(t_step = 1)

### UNCERTAIN PPARMS
p_template_sim = simulator.get_p_template()

print(p_template_sim)

def p_fun_sim(t_now):
    for key in p_template_sim.keys():
        #print("key", key, "value", p_template_sim[key])
        p_template_sim[key] = 1 #np.random.uniform(0,2,1) 
    return p_template_sim


simulator.set_p_fun(p_fun_sim)

### TIME VARYING PARAMS

# tvp_template_sim = simulator.get_tvp_template()

# print(tvp_template_sim)

# #print(p_template_sim)

# def tvp_fun_sim(t_now):
#     for key in tvp_template_sim.keys():
#         #print("key", key, "value", p_template_sim[key])
#         tvp_template_sim[key] = 1 #np.random.uniform(0,2,1) 
#     return tvp_template_sim


# simulator.set_tvp_fun(tvp_fun_sim)

simulator.setup()

print("Simulator setup")

mpc.x0['current_accuracy'] = baseline_acc
#mpc.x0['weight_imp_state'] = np.ones(11)
mpc.x0['weight_imp_state'] = np.random.rand(11)

simulator.x0['current_accuracy'] = baseline_acc
simulator.x0['weight_imp_state'] = np.random.rand(11)
#simulator.x0['weight_imp_state'] = np.ones(11)

mpc.set_initial_guess()

#u0 = np.ones(11)
u0 = np.random.rand(11).reshape(-1,1)
x0 = mpc.x0

print("Everything before for setup")

print("initialized value accuracy", mpc.x0['current_accuracy'])
print("U0 ", u0)




for i in range(20):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    print("Current accuracy", mpc.x0['current_accuracy'])
    print("Current weight imp", mpc.x0['weight_imp_state'], type(np.array(mpc.x0['weight_imp_state'])))
    weight_imp_for_test = np.array(mpc.x0['weight_imp_state'])
    print("U0", u0)


    print("Verification")
    new_features = np.empty_like(features_strat)
    for index, row in enumerate(features_strat):
        row = row.reshape(-1,1)
        new_features[index, :] = (row * weight_imp_for_test).reshape(11)


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

folder_path = "./results"
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed: {filename}")
    except Exception as e:
        print(f"Error while removing {filename}: {e}")


save_results([mpc, simulator])

results = load_results('./results/results.pkl')

print("results", results)
keys = ['_time', '_x', '_y', '_u', '_z', '_tvp', '_p', '_aux']


print("Results MPC", results['mpc'])
print("Results simulation", results['simulator'])
for key in keys:
    print(key, results['mpc'][key])

for key in keys:
    print(key, results['simulator'][key])

