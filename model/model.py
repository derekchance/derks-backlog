from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .series import SERIES
from .core import load_dataset
from .core import MODEL_DIR
from .core import TARGET
from .core import richard_curve
from .xgb import main as xgb
from .pytabkit import main as realmlp



param_grid = {
    "n_components": range(1, 21),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


def update_models():
    print('Updating models')
    #print('SVR (radial basis)...')
    #rbf()
    #print('SVR (linear)...')
    #linear_svr()
    #print('Ridge...')
    #ridge()
    #print('ElasticNet...')
    #elasticnet()
    #print('XGB...')
    #xgb()
    #print('Stacking Models (like legos)')
    #stacking()
    realmlp()
    print('Done.')
