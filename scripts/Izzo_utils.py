import numpy as np 

# Defining constants 
no_features = 2
g = 0.5
s0 = 0.5 #square root of standard deviation 0
s1 = 0.5  #square root of standard deviation 1 

mu0 = 1
mu1 = -1
eps = 10.0

reg = 1e-2

def mean0(theta):
    return mu0

def mean1(theta, strat_features, feature_index):
    return mu1 - eps * theta[feature_index] * strat_features[feature_index]


def shift_dist(n, theta, no_features, strat_features):
    X = np.ones((n, no_features))
    Y = np.ones(n)
    for i in range(n):
        if np.random.rand() <= g:
            for j in range(no_features):           
                X[i, j] = s1 * np.random.randn() + mean1(theta, strat_features, j)
            Y[i] = 1
        else:
            for j in range(no_features):
                X[i, j] = s0 * np.random.randn() + mean0(theta)
            Y[i] = 0
    return X, Y
## Functions for fitting logistic regression 
def h(x, theta):
    """
    Logistic model output on x with params theta.
    x should have a bias term appended in the 0-th coordinate.
    1 / (1 + exp{-x^T theta})
    """
    return 1. / (1. + np.exp(-np.dot(x, theta)))


def fit(X, Y, theta0 = None, reg = reg, lr = 1, tol = 0.001, max_iter = 1000):
    """
    Fits a logistic model to X, Y via Newton's method, without the aid of Pytorch.
    """
    if theta0 is None:
        theta0 = np.random.randn(len(X[0, :]))
        
    theta = theta0.copy()
    grad = gradient(X, Y, theta, reg)
    grad_norm = np.linalg.norm(grad)
    count = 1
    while grad_norm > tol:
        if count % 500 == 0:
            print(f'Iteration {count}: |grad| = {grad_norm}, lr = {lr}')
        hess = hessian(X, theta, reg)
        step = np.linalg.solve(hess, grad)
        theta -= lr * step
        grad = gradient(X, Y, theta, reg)
        old_grad_norm = grad_norm
        grad_norm = np.linalg.norm(grad)
        if grad_norm > old_grad_norm:
            lr *= 0.9
        count += 1
        if count > max_iter:
            print(f'Warning: Optimization failed to converge. Aborting with |grad| = {grad_norm}.')
            break
    return theta


## Utility functions 

def gradient(X, Y, theta, reg):
    """
    Computes the gradient of the loss on X, Y at theta with ridge regularization reg.
    """
    n = len(Y)
    h_vec = np.array([h(x, theta) for x in X])
    grad = X.T @ (h_vec - Y) + n * reg * theta
    return grad


def hessian(X, theta, reg):
    """
    Computes the Hessian of the loss on X at theta with ridge regularization reg.
    """
    n = len(X)
    d = len(X[0, :])
    h_vec = np.array([h(x, theta) for x in X])
    w = h_vec * (1 - h_vec)
    
    hess = np.zeros((d, d))
    for i in range(n):
        hess += np.outer(w[i] * X[i], X[i])
    hess += n * reg * np.eye(d)
    return hess


def generate_Izzo_latex_table(accuracy_dict_all_runs):
    num_iterations = len(next(iter(accuracy_dict_all_runs.values()))[0])  # Get the number of iterations (columns)
    already_printed = 0
    # Print the LaTeX table header
    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\begin{tabular}{@{}p{2.5cm}|" + "p{1cm}" * min(10, num_iterations) + "@{}}")
    print("    \\toprule")
    print("        Metric/iter & " + "&".join([str(i+1) for i in range(min(10, num_iterations))]) + " \\\\")
    print("        \\midrule")

    for name, values in accuracy_dict_all_runs.items():
        mean_vals = np.mean(values, axis=0)
        std_vals = np.std(values, axis=0)
        
        # Print the first set of up to 10 iterations
        print(f"        \\makecell{{\\raggedright {name}}} ", end="")
        for mean, std in zip(mean_vals[:10], std_vals[:10]):
            if "Acc" in name: 
                print(f"& \\makecell{{\\textbf{{{(100*mean):.2f}}} \\\\ \\pm{(100*std):.2f}}} ", end="")
            else:
                 print(f"& \\makecell{{\\textbf{{{mean:.2f}}} \\\\ \\pm{std:.3f}}} ", end="")
    
        print("\\\\")

        # If there are more than 10 iterations, print the second set in a new row
        if num_iterations > 10:
            if already_printed == 0:
                print("        \\midrule")
                print("        Metric/iter & " + "&".join([str(i+1) for i in range(11, num_iterations + 1)]) + " \\\\")
                print("        \\midrule")
                already_printed = 1
            
            # Print the metric name again and the next set of iterations
            print(f"        \\makecell{{\\raggedright {name}}} ", end="")
            for mean, std in zip(mean_vals[10:], std_vals[10:]):
                if "Acc" in name: 
                    print(f"& \\makecell{{\\textbf{{{(100*mean):.2f}}} \\\\ \\pm{(100*std):.2f}}} ", end="")
                else:
                    print(f"& \\makecell{{\\textbf{{{mean:.2f}}} \\\\ \\pm{std:.3f}}} ", end="")
            print("\\\\")

    print("    \\bottomrule")
    print("    \\end{tabular}")
    print("    \\label{tab:results}")
    print("\\end{table}")