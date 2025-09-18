from pathlib import Path
from datetime import date
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

MODEL_DIR = Path(__file__).parent
CATEGORICAL_FEATURES = [
    #'tier',
    'genre_metacritic',
    'developer_metacritic',
    #'platform_metacritic',
    #'franchise_igdb',
]

NUMERICAL_FEATURES = [
    #igdb
    'aggregated_rating_igdb',
    'aggregated_rating_count_igdb',
    'rating_igdb',
    'rating_count_igdb',
    'age_comb',

    #metacritic
    'metaScore_metacritic',
    'userScore_metacritic',

    #HowLongToBeeat
    #'count_review_hltb',
    #'count_retired_hltb',
    'count_comp_hltb',
    'review_score_hltb',
    'retire_rate_hltb',

    #Recs Features
    'AM',
    'BR',
    'Brandon',
    'Buried Treasure',
    'EC',
    'Jackie',
    'Nick',
    'Fabio',
    'Kaleb',
    'Sterling',
    'Yahtzee',
    'hbomberguy',
    'Tyler',
    'fightincowboy',

    # Self-Engineered Features
    'Classic',
    'soulslike',
]

DICT_FEATURES = [
    #'Companies',
    'Genres',
]

LIST_FEATURES = [
    'genre_mc_clean',
    #'game_engines_igdb',
    'genres_igdb',
    #'keywords_igdb',
    #'similar_games_igdb',
    #'involved_companies_igdb',
    'themes_igdb',
    #'platform_hltb_clean',
]

SIMILAR_FEATURES = ['id_igdb', 'similar_games_igdb', 'Finished', 'glicko']

TARGET = 'glicko'

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + DICT_FEATURES + LIST_FEATURES + SIMILAR_FEATURES


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


def _prepare_dataframe(df):
    # load recommendations and append to dataset
    rec_df = pd.read_csv('rec_log.csv')
    rec_df = rec_df.groupby('Title').Recommender.value_counts().unstack(level=1).fillna(0)
    df = df.merge(rec_df, how='left', left_on='Title', right_index=True)
    df['release_calendar_date_igdb'] = pd.to_datetime(df.first_release_date_igdb, unit='s')
    df['release_date'] = df.release_calendar_date_igdb.fillna(df.releaseDate_metacritic)
    df['age'] = (pd.to_datetime(date.today()) - df['release_date']).dt.days
    df['Companies'] = df['Companies'].apply(_safe_loads)
    df['Genres'] = df['Genres'].apply(_safe_loads)
    df['genre_mc_clean'] = df.loc[:, 'genre_metacritic'].fillna('').str.replace("-","")
    df['platform_hltb_clean'] = df['profile_platform_hltb'].astype(str).str.split(', ').astype(str)
    df['retire_rate_hltb'] = df['count_retired_hltb'] / df['count_comp_hltb']
    df['retire_rate_hltb'] = df['retire_rate_hltb'].where(df['count_comp_hltb'] > 5)
    df['review_score_hltb'] = df['review_score_hltb'].where(df['count_review_hltb'] > 5)
    df['age_hltb'] = (2025 - df['release_world_hltb']) * 365
    df['age_comb'] = df['age'].fillna(df['age_hltb'])
    df.loc[:, 'AM':] = df.loc[:, 'AM':].fillna(0.)
    df.loc[:, LIST_FEATURES] = df.loc[:, LIST_FEATURES].fillna('[]')
    return df


def load_dataset():
    df = pd.read_csv('game_log.csv')
    df = _prepare_dataframe(df)

    return df.loc[:, FEATURES]


def load_Xy():
    df = pd.read_csv('game_log.csv')
    df = _prepare_dataframe(df)

    X = df.loc[df.Finished == 1, FEATURES]
    y = df.loc[df.Finished == 1, TARGET]
    return X, y


def richard_curve(t, a=0, k=1, b=0.0035, v=1, q=100, c=1):
    return a + (k-a)/((c + q * np.e ** (-b * t)) ** (1/v))
