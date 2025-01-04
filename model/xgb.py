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

SEARCH_ITERS = 200
SEARCH_SCORING_METRIC = 'neg_mean_squared_error'
SEARCH_CV_SPLIT = 5
N_BEST = 3

FEATURES = [
    'developer_metacritic',
    'medianScore',
    'topCriticScore',
    'percentile',
    'aggregated_rating_igdb',
    'aggregated_rating_count_igdb',
    'rating_igdb',
    'rating_count_igdb',
    'age',
    'metaScore_metacritic',
    'userScore_metacritic',
    'AM',
    'BR',
    'Brandon',
    'Buried Treasure',
    'EC',
    'Jackie',
    'Nick',
    'Sterling',
    'Yahtzee',
    'Classic',
    'Companies',
    'Genres',
    'genre_mc_clean',
    'game_engines_igdb',
    'genres_igdb',
]

# categorize features by preprocessing
numerical_features = [n for n in FEATURES if n in NUMERICAL_FEATURES]
categorical_features = [n for n in FEATURES if n in CATEGORICAL_FEATURES]
dict_features = [n for n in FEATURES if n in DICT_FEATURES]
list_features = [n for n in FEATURES if n in LIST_FEATURES]


SEARCH_PARAMS = {
    'rgr__gamma': np.logspace(-2, 2, 5),
    'rgr__learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    'rgr__max_depth': range(2, 14),
    'rgr__min_child_weight': [0, 0.001, 0.01, 0.1, 1],
    'rgr__n_estimators': range(100, 1001, 50),
    'rgr__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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

    for n in range(1, N_BEST+1):
        n_model = Pipeline([
            ('preprocess', column_transformer),
            ('rgr', xgb.XGBRegressor(
                gamma=cv_df.loc[n, 'param_rgr__gamma'],
                learning_rate=cv_df.loc[n, 'param_rgr__learning_rate'],
                max_depth=cv_df.loc[n, 'param_rgr__max_depth'],
                min_child_weight=cv_df.loc[n, 'param_rgr__min_child_weight'],
                n_estimators=cv_df.loc[n, 'param_rgr__n_estimators'],
            ))
        ])

        dump(n_model, MODEL_DIR / f'./models/xgb_model{n}.joblib')


if __name__ == '__main__':
    main()
