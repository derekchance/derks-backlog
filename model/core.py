from pathlib import Path
from datetime import date
import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

MODEL_DIR = Path(__file__).parent
CATEGORICAL_FEATURES = [
    'genre',
    'category',
    'franchise',
    'game_type',
]

NUMERICAL_MEANED_FEATURES = [
    #metacritic
    'userScore',
    'metaScore',

    #igdb
    'rating',
    'aggregated_rating',

    #HowLongToBeeat
    'retire_rate',
    'comp_main_rate',
    'comp_plus_rate',
    'comp_100_rate',
    'comp_all_rate',
    'review_rate',
    'speedrun_rate',
    'review_score',
]

NUMERICAL_ZEROED_FEATURES = [
    'rating_count',
    'aggregated_rating_count',
    'count_comp',
    'count_review',
    'count_speedrun',

    # Recs Features
    'AM',
    'BR',
    'Buried Treasure',
    'Jackie',
    'Nick',
    'Fabio',
    'Kaleb',
    'Sterling',
    'Yahtzee',
    'hbomberguy',
    'fightincowboy',

    # Self-Engineered Features
    'classic',
    'soulslike',
]

LIST_FEATURES = [
    'game_engines',
    'genres',
    'themes',
    'keywords',
    'game_modes',
    'involved_companies',
    'similar_games',
]

TARGET = 'glicko'

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_MEANED_FEATURES + NUMERICAL_ZEROED_FEATURES + LIST_FEATURES


class CategoricalEncoder(OrdinalEncoder):
    def _safe_convert(self, X):
        for i, j in enumerate(X):
            cats = [n for n in range(len(self.categories_[i]))]
            X[j] = pd.Categorical(X[j], categories=cats)
        return X

    def transform(self, X):
        X_trans = super().transform(X)
        return self._safe_convert(X=X_trans)


class PandasCountVectorizer(CountVectorizer):
    def fit_transform(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents=raw_documents, y=y)
        return pd.DataFrame.sparse.from_spmatrix(X, columns=self.get_feature_names_out(), index=raw_documents.index).astype(float)

    def transform(self, raw_documents):
        X = super().transform(raw_documents=raw_documents)
        return pd.DataFrame.sparse.from_spmatrix(X, columns=self.get_feature_names_out(), index=raw_documents.index).astype(float)

    def set_output(self, *, transform=None):
        pass


class PandasDictVectorizer(DictVectorizer):
    def transform(self, X):
        idx = X.index
        X = super().transform(X=X)
        return X.set_index(idx)


    def fit_transform(self, X, y=None):
        idx = X.index
        X = super().fit_transform(X=X, y=y)
        return X.set_index(idx)


class SummarizeSimilar(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.map = None

    def fit(self, X, y=None):
        pool = set()
        for n in X.itertuples():
            try:
                pool = pool | set(json.loads(n.similar_games_igdb))
            except:
                pass
        self.map = X.loc[X.id_igdb.isin(pool) & (X.Finished == 1), ['id_igdb', 'glicko']]
        self.map['id_igdb'] = self.map['id_igdb'].astype(int)
        self.map.set_index('id_igdb', inplace=True)

    def transform(self, X):
        results = []
        for n in X.itertuples():
            try:
                results.append(self.map.reindex(json.loads(n.similar_games_igdb)).mean().iloc[0])
            except:
                results.append(np.nan)
        return pd.Series(data=results, index=X.index, name='similar_games_mean_glicko').to_frame()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X=X, y=y)
        return self.transform(X=X)

    def set_output(self, *, transform=None):
        pass

    def get_feature_names_out(self, input_features=None):
        return np.array(['similar_games_mean_glicko'])

def _safe_loads(x):
    """for reading json format Series"""
    try:
        return json.loads(x.replace("'", '"'))[0]
    except:
        return {}


def load_dataset(game_ids='all'):
    if game_ids == 'all':
        game_ids = '(SELECT id FROM games)'
    elif isinstance(game_ids, int):
        game_ids = f'({game_ids})'

    with sqlite3.connect('games.db') as con:
        with open('input.sql') as f:
            query = f.read().format(game_ids=game_ids)

        return pd.read_sql(query, con)


def load_Xy():
    df = load_dataset()

    X = df.loc[df.glicko.notna(), FEATURES].copy()
    y = df.loc[df.glicko.notna(), TARGET].values
    return X, y


def richard_curve(t, a=0, k=1, b=0.0035, v=1, q=100, c=1):
    return a + (k-a)/((c + q * np.e ** (-b * t)) ** (1/v))
