import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

from .core import LIST_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DICT_FEATURES, MODEL_DIR, load_Xy

ALPHA_SPACE = [0.01, 0.01, 0.05, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
SEARCH_SCORING_METRIC = 'neg_mean_squared_error'
SEARCH_CV_SPLIT = 5

FEATURES = [
    'developer_metacritic',
    'aggregated_rating_igdb',
    'aggregated_rating_count_igdb',
    'rating_igdb',
    'rating_count_igdb',
    'age',
    'metaScore_metacritic',
    'userScore_metacritic',
    'BR',
    'Brandon',
    'Sterling',
    'Yahtzee',
    'Classic',
    'Companies',
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
    ('rgr', Ridge(solver='svd', random_state=42)
     )
])


def main():
    X, y = load_Xy()
    search = GridSearchCV(
        model,
        param_grid={
            'rgr__alpha': ALPHA_SPACE
        },
        scoring=SEARCH_SCORING_METRIC,
        cv=SEARCH_CV_SPLIT,
        verbose=0,
    )

    search.fit(X, y)

    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.set_index('rank_test_score', inplace=True)
    cv_df.sort_index(inplace=True)
    cv_df.to_csv(MODEL_DIR / './model_results/ridge_search.csv')

    best_model = Pipeline([
        ('preprocess', column_transformer),
        ('rgr', Ridge(solver='svd', random_state=42, alpha=cv_df.loc[1, 'param_rgr__alpha'])
         )
    ])

    dump(best_model, MODEL_DIR / './models/ridge_model.joblib')


if __name__ == '__main__':
    main()
