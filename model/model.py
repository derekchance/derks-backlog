from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from .series import SERIES
from .core import load_dataset
from .core import MODEL_DIR
from .core import TARGET
from .rbf import main as rbf
from .ridge import main as ridge
from .xgb import main as xgb
from .stacking import main as stacking


def adjust_sequels(df):
    """Gives preference to earlier titles in series with better scoring sequels"""
    for i in SERIES:
        series_df = df.loc[df.Title.isin(i)]
        if len(series_df) > 1:
            model_scores = []
            titles = []
            for j in i:
                if j in series_df.Title.unique():
                    model_scores.append(series_df.loc[series_df.Title == j, 'raw_score'].values[0])
                    titles.append(j)
            if np.argmax(model_scores) != 0:
                first = model_scores[0]
                means = []
                for k, j in enumerate(model_scores[1:]):
                    if j > first:
                        means.append(0.8 * j + 0.2 * first)
                        model_scores[k+1] = 0.8 * first + 0.2 * j
                model_scores[0] = np.mean(means)

                for k, j in enumerate(model_scores):
                    df.loc[df.Title == titles[k], 'model_score'] = j
    return df.model_score


def update_model_scores():
    df = pd.read_csv(MODEL_DIR.parent / 'game_log.csv')
    original_columns = set(df.columns.to_list())
    test_df = load_dataset()

    model = joblib.load(MODEL_DIR / 'models/stacking_model.joblib')
    df['raw_score'] = model.predict(test_df)
    original_columns = original_columns | {'raw_score'}
    df.loc[:, list(original_columns)].to_csv('game_log.csv', index=False)

    backlog_cols = ['Title', 'raw_score', 'release_date', 'genre_metacritic', 'developer_metacritic']
    backlog_df = df.loc[df['My Rating'].isnull(), backlog_cols]
    backlog_df['model_score'] = df['raw_score'].copy()
    backlog_df['model_score'] = adjust_sequels(df)
    backlog_df \
        .sort_values('model_score', ascending=False) \
        .to_csv(MODEL_DIR.parent / 'backlog.csv', index=False)

    log_cols = ['Title', 'My Rating', 'raw_score', 'release_date', 'genre_metacritic', 'developer_metacritic']

    df.loc[df['My Rating'].notna(), log_cols] \
        .sort_values('raw_score', ascending=False) \
        .to_csv(MODEL_DIR.parent / 'simple_log.csv', index=False)


def update_models():
    print('Updating models')
    print('SVR (radial basis)...')
    rbf()
    print('Ridge...')
    ridge()
    print('XGB...')
    xgb()
    print('Stacking Models (like legos)')
    stacking()
    print('Done.')
