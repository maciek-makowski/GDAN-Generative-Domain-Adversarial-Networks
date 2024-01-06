import whynot as wn
import whynot.gym as gym
import scripts.utils as utils
import numpy as np
import do_mpc
import casadi as ca
import matplotlib.pyplot as plt

def perform_multiplication(importance_matrix, features, x,y):
    
    result_matrix = ca.repmat(importance_matrix, x, y).T * features

    return result_matrix

def calc_accuracy(features, labels, classifier):
    ### Evaluates model performance
    #temp = ca.dot(features, classifier) > 0
    length = labels.shape[0]
    classifier = classifier.reshape(-1,1)

    temp = ca.mtimes(features, classifier)     

    pred = temp > 0 
    corr_pred = ca.eq(pred, labels)
    print(corr_pred)
    print("corr pred shape", corr_pred.size1(), corr_pred.size2())
    accuracy = ca.sum1(corr_pred)/ length

    #print("Accuracy inside func", accuracy, type(accuracy))
    
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
features = model.set_variable(var_type = '_x', var_name = 'features', shape = (18357,11))

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
model.set_rhs('features', features)
model.set_rhs('current_accuracy', calc_accuracy(perform_multiplication(weight_importance, features, 1, 18357), labels, theta))

model.setup()

print("Model setup")

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 2,
    't_step': 1,
    'n_robust': 1,
    'store_full_solution': True
}
mpc.set_param(**setup_mpc)

for i in range(11):
    mpc.bounds['lower', '_u', f'weight_importance_{i}'] = 0
    mpc.bounds['upper', '_u', f'weight_importance_{i}'] = 1


print("Desired accuracy before", desired_accuracy)
lterm = desired_accuracy - current_accuracy
mterm = (desired_accuracy - current_accuracy) * (desired_accuracy - current_accuracy)

mpc.set_objective(mterm = mterm, lterm = lterm)

mpc.set_rterm(
    weight_importance_0=1e-7,
    weight_importance_1=1e-8,
    weight_importance_2=1e-2,
    weight_importance_3=1e-4,
    weight_importance_4=1e-2,
    weight_importance_5=0.5,
    weight_importance_6=1e-4,
    weight_importance_7=1e-3,
    weight_importance_8=1e-1,
    weight_importance_9=1e-4,
    weight_importance_10=1e-5,
)

mpc.setup()

print("MPC setup")

simulator = do_mpc.simulator.Simulator(model)

simulator.set_param(t_step = 1)

simulator.setup()

print("Simulator setup")

print("Type of strat_features", type(features_strat))
print("Shape of strat_features", features_strat.shape)

mpc.x0['current_accuracy'] = baseline_acc
mpc.x0['features'] = features_strat

simulator.x0['current_accuracy'] = baseline_acc
simulator.x0['features'] = features_strat

mpc.set_initial_guess()

u0 = np.random.rand(11,1)
x0 = mpc.x0

print("Everything before for setup")

print("initialized value accuracy", mpc.x0['current_accuracy'])
print("U0 ", u0)

for i in range(20):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    print("Current accuracy", mpc.x0['current_accuracy'])
    print("U0", u0)
    print("X0", x0)

