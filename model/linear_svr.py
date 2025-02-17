from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR

from .core import LIST_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DICT_FEATURES, MODEL_DIR, load_Xy

SEARCH_SCORING_METRIC = 'neg_mean_squared_error'
SEARCH_CV_SPLIT = 5
PARAMS = {
    'rgr__C': np.logspace(-5,2, 8),
    'rgr__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    'rgr__epsilon': np.append(np.logspace(-2,0,3), 0),
}

# categorize features by preprocessing
numerical_features = NUMERICAL_FEATURES
categorical_features = CATEGORICAL_FEATURES
dict_features = DICT_FEATURES
list_features = LIST_FEATURES

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

numeric_transformer = Pipeline([
    ('impute', SimpleImputer()),
    ('scale', StandardScaler()),
])

nonlist_transformers = [
    ('categorical', categorical_transformer, categorical_features),
    ('numerical', numeric_transformer, numerical_features),
]

column_transformer = ColumnTransformer(transformers=nonlist_transformers+list_transformers+dict_transformers, sparse_threshold=0)

model = Pipeline([
    ('preprocess', column_transformer),
    ('rgr', LinearSVR(max_iter=10000000, tol=1e-4,)
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
    cv_df.to_csv(MODEL_DIR / './model_results/linear_svr_search.csv')

    best_model = Pipeline([
        ('preprocess', column_transformer),
        ('rgr', LinearSVR(
            max_iter=10000000,
            tol=1e-4,
            C=cv_df.loc[1, 'param_rgr__C'],
            loss=cv_df.loc[1, 'param_rgr__loss'],
            epsilon=cv_df.loc[1, 'param_rgr__epsilon'],
        )
         )
    ])

    dump(best_model, MODEL_DIR / './models/linear_svr_model.joblib')


if __name__ == '__main__':
    main()
