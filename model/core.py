from pathlib import Path
from datetime import date
import json

import pandas as pd

MODEL_DIR = Path(__file__).parent
CATEGORICAL_FEATURES = [
    'tier',
    'genre_metacritic',
    'developer_metacritic',
]

NUMERICAL_FEATURES = [
    'percentRecommended',
    'numReviews',
    'numTopCriticReviews',
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
]

DICT_FEATURES = [
    'Companies',
    'Genres',
]

LIST_FEATURES = [
    'genre_mc_clean',
    'game_engines_igdb',
    'genres_igdb',
    'keywords_igdb',
    'similar_games_igdb',
    'themes_igdb',
]

TARGET = 'My Rating'

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + DICT_FEATURES + LIST_FEATURES


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

    X = df.loc[df[TARGET].notna(), FEATURES]
    y = df.loc[df[TARGET].notna(), TARGET]
    return X, y

