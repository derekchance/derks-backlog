import pandas as pd
import numpy as np
import xgboost as xgb
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV

from .core import LIST_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DICT_FEATURES, MODEL_DIR, load_Xy

SEARCH_ITERS = 50
SEARCH_SCORING_METRIC = 'neg_mean_squared_error'
SEARCH_CV_SPLIT = 5
N_BEST = 1

# categorize features by preprocessing
numerical_features = NUMERICAL_FEATURES
categorical_features = CATEGORICAL_FEATURES
dict_features = DICT_FEATURES
list_features = LIST_FEATURES


SEARCH_PARAMS = {
    'rgr__gamma': [1],
    'rgr__learning_rate': [0.05, 0.1],
    'rgr__max_depth': range(2, 11, 2),
    'rgr__min_child_weight': [0, 0.1, 1],
    'rgr__n_estimators': range(300, 1001, 100),
    'rgr__subsample': [0.5, 0.6, 0.7],
}

# define preprocessors
list_transformers = []
for n in list_features:
    list_transformers.append((n, CountVectorizer(input='content'), n))

dict_transformers = []
for n in dict_features:
    dict_transformers.append((n, DictVectorizer(), n))

categorical_transformer = Pipeline([
    ('impute', SimpleImputer(fill_value='MISSING_VALUE', strategy='constant')),
    ('encode', OneHotEncoder(handle_unknown='ignore')),
])

nonlist_transformers = [
    ('categorical', categorical_transformer, categorical_features),
    ('numerical', 'passthrough', numerical_features),
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
    ))
])


def main():
    X, y = load_Xy()

    search = RandomizedSearchCV(
        model,
        param_distributions=SEARCH_PARAMS,
        n_iter=SEARCH_ITERS,
        scoring=SEARCH_SCORING_METRIC,
        cv=SEARCH_CV_SPLIT,
        verbose=0,
    )

    search.fit(X, y)
    
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.set_index('rank_test_score', inplace=True)
    cv_df.sort_index(inplace=True)
    cv_df.to_csv(MODEL_DIR / './model_results/xgb_search.csv')

    cv_df = pd.read_csv(MODEL_DIR / './model_results/xgb_search.csv')
    cv_df.reset_index()

    best_model = Pipeline([
        ('preprocess', column_transformer),
        ('rgr', xgb.XGBRegressor(
            gamma=cv_df.loc[0, 'param_rgr__gamma'],
            learning_rate=cv_df.loc[0, 'param_rgr__learning_rate'],
            max_depth=cv_df.loc[0, 'param_rgr__max_depth'],
            min_child_weight=cv_df.loc[0, 'param_rgr__min_child_weight'],
            n_estimators=cv_df.loc[0, 'param_rgr__n_estimators'],
        ))
    ])

    dump(best_model, MODEL_DIR / f'./models/xgb_model{n}.joblib')


if __name__ == '__main__':
    main()
