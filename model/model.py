from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .series import SERIES
from .core import load_dataset
from .core import MODEL_DIR
from .core import TARGET
from .core import richard_curve
from .xgb import main as xgb
from .pytabkit import main as realmlp



param_grid = {
    "n_components": range(1, 21),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

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


def update_model_scores(model='stacking'):
    df = pd.read_csv(MODEL_DIR.parent / 'game_log.csv')
    original_columns = set(df.columns.to_list())
    test_df = load_dataset()

    model = joblib.load(MODEL_DIR / f'models/{model}_model.joblib')
    df['raw_score'] = model.predict(test_df)
    original_columns = original_columns | {'raw_score'}
    df.loc[:, list(original_columns)].to_csv('game_log.csv', index=False)

    df['time_est'] = df['comp_all_hltb'] / 3600

    backlog_cols = ['Title', 'raw_score', 'time_est', 'release_date', 'genre_metacritic', 'developer_metacritic']
    backlog_df = df.loc[df.Finished == 0, backlog_cols]

    backlog_df['model_score'] = backlog_df['raw_score'].copy()
    backlog_df['model_score'] = adjust_sequels(backlog_df)

    #grid_search = GridSearchCV(
    #    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    #)
    #grid_search.fit(backlog_df['model_score'].to_frame())
    #backlog_df['tier'] = grid_search.predict(backlog_df['model_score'].to_frame())

    final_backlog_df = backlog_df.loc[:, ['Title', 'model_score', 'raw_score', 'time_est', 'release_date', 'genre_metacritic',
                       'developer_metacritic']] \
        .sort_values('model_score', ascending=False)

    final_backlog_df.to_csv(MODEL_DIR.parent / 'backlog.csv', index=False)

    played_df = df.loc[df['Finished'] == 1, :].copy()
    played_df['Err'] = played_df[TARGET] - played_df['raw_score']
    played_df['Err_z'] = (played_df['Err'].abs() - played_df['Err'].abs().mean()) / played_df['Err'].abs().std()
    played_df['raw_score_z'] = (played_df['raw_score'] - played_df['raw_score'].mean()) / played_df['raw_score'].std()
    played_df[f'{TARGET}_z'] = (played_df[TARGET] - played_df[TARGET].mean()) / played_df[TARGET].std()
    played_df['replay_score'] = played_df[['raw_score_z', f'{TARGET}_z']].mean(axis=1)
    played_df['replay_score'] = MinMaxScaler().fit_transform(played_df['replay_score'].to_frame())

    played_df['last_played'] = pd.to_datetime(df.last_played, errors='coerce', format='mixed')
    played_df['last_played_weight'] = richard_curve(
        (pd.Timestamp.now() - played_df.last_played).dt.days).fillna(1)

    played_df['replay_score'] *= played_df['last_played_weight']

    replay_bl = [
        'Super Smash Bros.',
        'Tetris',
        'Super Smash Bros. Melee',
    ]
    played_df['replay_score'] = played_df.replay_score.where(~played_df['Title'].isin(replay_bl), 0)

    log_cols = ['Title', 'glicko', 'raw_score', 'replay_score', 'last_played', 'last_played_weight', 'release_date', 'genre_metacritic', 'developer_metacritic']

    played_df.loc[:, log_cols] \
        .sort_values('raw_score', ascending=False) \
        .to_csv(MODEL_DIR.parent / 'simple_log.csv', index=False)


def update_models():
    print('Updating models')
    #print('SVR (radial basis)...')
    #rbf()
    #print('SVR (linear)...')
    #linear_svr()
    #print('Ridge...')
    #ridge()
    #print('ElasticNet...')
    #elasticnet()
    #print('XGB...')
    #xgb()
    #print('Stacking Models (like legos)')
    #stacking()
    realmlp()
    print('Done.')
