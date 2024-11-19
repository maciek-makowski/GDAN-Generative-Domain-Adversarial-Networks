"""Loading data from file"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os 


def load_data(file_loc):
    """Load data from cvs file.

    Parameters
    ----------
        file_loc: string
            path to the '.csv' training data file
    Returns
    -------
        X_full: np.array
            balances data matrix     
        Y_full: np.array
            corresponding labels (0/1) 
        data: DataFrame
            raw data     
    """

    data = pd.read_csv(file_loc, index_col=0)
    data.dropna(inplace=True)

    # full data set
    X_all = data.drop('SeriousDlqin2yrs', axis=1)

    # zero mean, unit variance
    X_all = preprocessing.scale(X_all)

    # add bias term
    X_all = np.append(X_all, np.ones((X_all.shape[0], 1)), axis=1)

    # outcomes
    Y_all = np.array(data['SeriousDlqin2yrs'])

    # balance classes
    default_indices = np.where(Y_all == 1)[0]
    other_indices = np.where(Y_all == 0)[0][:10000]
    indices = np.concatenate((default_indices, other_indices))

    X_balanced = X_all[indices]
    Y_balanced = Y_all[indices]

    # shuffle arrays
    p = np.random.permutation(len(indices))
    X_full = X_balanced[p]
    Y_full = Y_balanced[p]
    return X_full, Y_full, data



def print_Perdomo_table(metric_dict):
    latex_table = r"""
\begin{table}[!h] 
\centering
\caption{Comparison of metrics for experiments with Perdomo generator. Each experiment has been performed 10 times, and each time we subsample 10000 points from the \emph{GiveMeSomeCredit} and then evolve them by inducing drift. The values presented in the table are means and standard deviations.}
\resizebox{\textwidth}{!}{
\begin{tabular}{@{}p{1.8cm}|p{3cm}|p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}@{}}
\toprule
\textbf{Method} & \textbf{Metric} 
& \textbf{1} & \textbf{2} & \textbf{3} & \textbf{4} & \textbf{5} & \textbf{6} & \textbf{7} & \textbf{8} & \textbf{9} & \textbf{10} \\ \midrule
"""
    
    for name, values in metric_dict.items():
        latex_table += f"\\multirow{{1}}{{*}}{{\\parbox{{2.5cm}}{{\\raggedright \\textbf{{{name}}}}}}}\n"
        

        print ("NAME", name)
        print("Values ", values)
        # Assuming 'values' is a 2D numpy array where each row is a set of 10 runs
        
        mean_val = np.mean(values, axis=0)
        std_val = np.std(values, axis =0)

        print("MEan val", mean_val)
        print("Std val", std_val)
        
        metric_name = f"{name}"  # Placeholder, adjust as needed
        latex_table += f"& {metric_name} "
        
        for j,run_value in enumerate(mean_val):
            if 'Acc' in metric_name:  # Adjust based on context if needed
                mean_str = f"{(100*run_value):.2f}"
                std_str = f"{(100*std_val[j]):.2f}"
            else:
                mean_str = f"{run_value:.3f}"
                std_str = f"{std_val[j]:.3f}"

            latex_table += f"& \\makecell{{\\textbf{{{mean_str}}} \\\\ \\pm{std_str}}} "
        latex_table += r"\\"
        latex_table += "\n"
        
    latex_table += r"""
\bottomrule
\end{tabular}}
\end{table}
"""

    print(latex_table)