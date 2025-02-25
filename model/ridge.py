import pandas as pd
from joblib import dump
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

from model.linear_svr import similar_features
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
ALPHA_SPACE = [0.01, 0.01, 0.05, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 400, 500, 1000]
SEARCH_SCORING_METRIC = 'neg_mean_squared_error'
SEARCH_CV_SPLIT = 5

# categorize features by preprocessing
numerical_features = NUMERICAL_FEATURES.copy()
categorical_features = CATEGORICAL_FEATURES.copy()
dict_features = DICT_FEATURES.copy()
list_features = LIST_FEATURES.copy()
similar_features = SIMILAR_FEATURES.copy()

categorical_features.remove('genre_metacritic')

# define preprocessors
list_transformers = []
for n in list_features:
    list_transformers.append((n, PandasCountVectorizer(input='content'), n))

dict_transformers = []
for n in dict_features:
    dict_transformers.append((n, PandasDictVectorizer(sparse=False), n))

categorical_transformer = Pipeline([
    ('impute', SimpleImputer(fill_value='MISSING_VALUE', strategy='constant')),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

numeric_transformer = Pipeline([
    ('impute', SimpleImputer()),
    ('scale', StandardScaler()),
])

similar_transformer = Pipeline([
    ('encode', SummarizeSimilar()),
    ('impute', SimpleImputer()),
    ('scale', StandardScaler())
])

nonlist_transformers = [
    ('categorical', categorical_transformer, categorical_features),
    ('numerical', numeric_transformer, numerical_features),
    ('similar', similar_transformer, similar_features),
]


column_transformer = ColumnTransformer(
    transformers=nonlist_transformers+list_transformers+dict_transformers,
    sparse_threshold=0
)

model = Pipeline([
    ('preprocess', column_transformer),
    ('rgr', Ridge(solver='svd')
     )
])


def main():
    X, y = load_Xy()
    search = GridSearchCV(
        model,
        param_grid={
            'rgr__alpha': ALPHA_SPACE,
            'preprocess__categorical__encode__min_frequency': [None, 2, 3, 4, 5],
        },
        scoring=SEARCH_SCORING_METRIC,
        cv=SEARCH_CV_SPLIT,
        verbose=1,
    )

    search.fit(X, y)

    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.set_index('rank_test_score', inplace=True)
    cv_df.sort_index(inplace=True)
    cv_df.to_csv(MODEL_DIR / './model_results/ridge_search.csv')

    best_model = Pipeline([
        ('preprocess', column_transformer),
        ('rgr', Ridge(solver='svd', random_state=42)
         )
    ])
    best_params = cv_df.loc[1, cv_df.columns.str.contains('param_')].rename(lambda x: x.replace('param_', '')).to_dict()
    best_model = best_model.set_params(**best_params)

    dump(best_model, MODEL_DIR / './models/ridge_model.joblib')


if __name__ == '__main__':
    main()
