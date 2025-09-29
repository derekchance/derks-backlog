import json
import warnings

import numpy as np
import pytabkit as ptk
from joblib import dump
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .core import (
    PandasCountVectorizer,
    CategoricalEncoder,
    LIST_FEATURES,
    NUMERICAL_MEANED_FEATURES,
    NUMERICAL_ZEROED_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    load_Xy,
)


set_config(transform_output='default')
SEARCH_ITERS = 50
SEARCH_SCORING_METRIC = 'neg_mean_squared_error'
SEARCH_CV_SPLIT = 5
N_BEST = 1

# categorize features by preprocessing
numerical_meaned_features = NUMERICAL_MEANED_FEATURES.copy()
numerical_zeroed_features = NUMERICAL_ZEROED_FEATURES.copy()
categorical_features = CATEGORICAL_FEATURES.copy()
list_features = LIST_FEATURES.copy()


SEARCH_PARAMS = {
    'rgr__gamma': [0, 1, 100],
    'rgr__learning_rate': [0.01],
    'rgr__max_depth': range(3, 22, 1),
    'rgr__min_child_weight': [0, 1, 2, 5, 10],
    'rgr__n_estimators': range(500, 1001, 100),
    'rgr__subsample': [0.6, 0.7, 0.8],
    'preprocess__categorical__encode__min_frequency': [None, 3],
}

# define preprocessors
list_transformers = []
for n in list_features:
    list_transformers.append((n, PandasCountVectorizer(input='content', lowercase=False, token_pattern=r'^[0-9]*'), n))

categorical_transformer = Pipeline([
    ('encode', CategoricalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
])

nonlist_transformers = [
    ('categorical', categorical_transformer, categorical_features),
    ('numerical', SimpleImputer(fill_value='mean'), numerical_meaned_features+numerical_zeroed_features),
]


column_transformer = ColumnTransformer(
    transformers=nonlist_transformers+list_transformers,
    sparse_threshold=0
)

column_transformer.set_output(transform='pandas')
categorical_columns = [f'categorical__{n}' for n in categorical_features]

# Define model
model = Pipeline([
        ('preprocess', column_transformer),
        ('rgr', ptk.RealMLP_HPO_Regressor(
            n_cv=8, hpo_space_name='tabarena', use_caruana_ensembling=True, n_hyperopt_steps=50,
            val_metric_name='rmse', n_epochs=1024
        ))
])


def main():
    X, y = load_Xy()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model.fit(X, y, rgr__cat_col_names=categorical_columns)

    dump(model, MODEL_DIR / f'./models/realmlp_model.joblib')


if __name__ == '__main__':
    main()
