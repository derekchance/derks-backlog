import json

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from joblib import dump
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

from .core import (
    PandasDictVectorizer,
    PandasCountVectorizer,
    CategoricalEncoder,
    SummarizeSimilar,
    LIST_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DICT_FEATURES,
    MODEL_DIR,
    load_Xy,
    SIMILAR_FEATURES,
)


set_config(transform_output='pandas')
SEARCH_ITERS = 50
SEARCH_SCORING_METRIC = 'neg_mean_squared_error'
SEARCH_CV_SPLIT = 5
N_BEST = 1

# categorize features by preprocessing
numerical_features = NUMERICAL_FEATURES.copy()
categorical_features = CATEGORICAL_FEATURES.copy()
dict_features = DICT_FEATURES.copy()
list_features = LIST_FEATURES.copy()
similar_features = SIMILAR_FEATURES.copy()

dict_features.remove('Genres')
list_features.remove('genre_mc_clean')


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
    list_transformers.append((n, PandasCountVectorizer(input='content'), n))

dict_transformers = []
for n in dict_features:
    dict_transformers.append((n, PandasDictVectorizer(sparse=False), n))

categorical_transformer = Pipeline([
    ('encode', CategoricalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
])

nonlist_transformers = [
    ('categorical', categorical_transformer, categorical_features),
    ('numerical', StandardScaler(), numerical_features),
    ('similar', SummarizeSimilar(), similar_features),
]


column_transformer = ColumnTransformer(
    transformers=nonlist_transformers+list_transformers+dict_transformers,
    sparse_threshold=0
)


# Define model
model = Pipeline([
    ('preprocess', column_transformer),
    ('rgr', xgb.XGBRegressor(
        device='cuda',
        enable_categorical=True,
    ))
])


def main(search=False):
    X, y = load_Xy()

    if search:
        search = RandomizedSearchCV(
            model,
            param_distributions=SEARCH_PARAMS,
            n_iter=SEARCH_ITERS,
            scoring=SEARCH_SCORING_METRIC,
            cv=SEARCH_CV_SPLIT,
            verbose=1,
        )

        search.fit(X, y)

        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.set_index('rank_test_score', inplace=True)
        cv_df.sort_index(inplace=True)
        cv_df.to_csv(MODEL_DIR / './model_results/xgb_search.csv')

        cv_df = pd.read_csv(MODEL_DIR / './model_results/xgb_search.csv')
        cv_df.reset_index()

        best_model = search.best_estimator_
        joblib.dump(best_model.named_steps['rgr'].get_params(), MODEL_DIR / './model_results/xgb_best_params.joblib')
    else:
        best_params = joblib.load(MODEL_DIR / './model_results/xgb_best_params.joblib')
        best_model = Pipeline([
            ('preprocess', column_transformer),
            ('rgr', xgb.XGBRegressor(**best_params))
        ])
        best_model.fit(X, y)

    dump(best_model, MODEL_DIR / f'./models/xgb_model.joblib')


if __name__ == '__main__':
    main()
