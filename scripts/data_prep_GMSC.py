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

