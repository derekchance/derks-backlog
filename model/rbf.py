from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from .core import LIST_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DICT_FEATURES, MODEL_DIR, load_Xy

SEARCH_SCORING_METRIC = 'neg_mean_squared_error'
SEARCH_CV_SPLIT = 5
PARAMS = {
    'rgr__C': np.logspace(-2,2, 5),
    'rgr__shrinking': [True],
    'rgr__gamma': ['auto', 'scale', 0.01, 0.1, 1, 10],
    'rgr__epsilon': np.logspace(-8,0,9),
}

FEATURES = [
    'developer_metacritic',
    'percentRecommended',
    'rating_igdb',
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
    'Companies',
    'Genres',
    'genre_mc_clean',
]

# categorize features by preprocessing
numerical_features = [n for n in FEATURES if n in NUMERICAL_FEATURES]
categorical_features = [n for n in FEATURES if n in CATEGORICAL_FEATURES]
dict_features = [n for n in FEATURES if n in DICT_FEATURES]
list_features = [n for n in FEATURES if n in LIST_FEATURES]

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
    ('numerical', SimpleImputer(), numerical_features),
]

column_transformer = ColumnTransformer(transformers=nonlist_transformers+list_transformers+dict_transformers, sparse_threshold=0)

model = Pipeline([
    ('preprocess', column_transformer),
    ('rgr', SVR(max_iter=10000000, tol=1e-4, kernel='rbf')
     )
])


def main():
    X, y = load_Xy()
    search = GridSearchCV(
        model,
        param_grid=PARAMS,
        scoring=SEARCH_SCORING_METRIC,
        cv=SEARCH_CV_SPLIT,
        verbose=0,
    )

    search.fit(X, y)

    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.set_index('rank_test_score', inplace=True)
    cv_df.sort_index(inplace=True)
    cv_df.to_csv(MODEL_DIR / './model_results/rbf_search.csv')

    best_model = Pipeline([
        ('preprocess', column_transformer),
        ('rgr', SVR(
            max_iter=10000000,
            tol=1e-4,
            kernel='rbf',
            C=cv_df.loc[1, 'param_rgr__C'],
            shrinking=cv_df.loc[1, 'param_rgr__shrinking'],
            gamma=cv_df.loc[1, 'param_rgr__gamma'],
            epsilon=cv_df.loc[1, 'param_rgr__epsilon'],
        )
         )
    ])

    dump(best_model, MODEL_DIR / './models/rbf_model.joblib')


if __name__ == '__main__':
    main()
