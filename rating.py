import argparse
import pandas as pd
import numpy as np


def reset_ratings():
    df = pd.read_csv('game_log.csv')
    df['elo'] = 1000
    df['Trials'] = 0
    df['glicko_rd'] = 350
    df['glicko'] = 1200
    df.to_csv('game_log.csv', index=False)


def trial(game_a=None, game_b=None, primary_rating='glicko'):
    df = pd.read_csv('game_log.csv')
    df['elo'] = df['elo'].fillna(1000)
    df['Trials'] = df['Trials'].fillna(0)
    df['glicko_rd'] = df['glicko_rd'].fillna(350)
    df['glicko'] = df['glicko'].fillna(1200)
    played_df = df.loc[df.Finished == 1].copy()

    print(game_a)
    if game_a is None:
        weights = (1 + played_df['Trials'].max() - played_df['Trials']) ** 2
        weights = np.where(
            played_df.Title == game_b,
            0,
            weights
        )
        game_a = played_df.loc[:, 'Title'].sample(n=1, weights=weights).iloc[0]
    game_a_elo = df.loc[df.Title == game_a, primary_rating].iloc[0]

    if game_b is None:
        dist = (game_a_elo - played_df[primary_rating]).abs()
        weights = 1/(dist + 1)**3
        weights = np.where(played_df.Title == game_a,
                           0,
                           weights,
                           )
        game_b = played_df.loc[:, 'Title'].sample(n=1, weights=weights).iloc[0]


    print(f'1: {game_a}')
    print(f'2: {game_b}')
    print('')
    winner = input('1 or 2?')
    if winner.lower() == '1':
        score_a = 1
        score_b = 0
    elif winner.lower() == '2':
        score_a = 0
        score_b = 1
    else:
        score_a = 0.5
        score_b = 0.5

    df.loc[df.Title == game_a, 'Trials'] += 1
    df.loc[df.Title == game_b, 'Trials'] += 1

    df.loc[df.Title == game_a, 'Finished'] = 1
    df.loc[df.Title == game_b, 'Finished'] = 1
    df.to_csv('game_log.csv', index=False)

    elo(games=(game_a, game_b), result=(score_a, score_b))
    glicko(games=(game_a, game_b), result=(score_a, score_b))
    return [game_a, game_b][int(winner) - 1]


def elo(games, result, k=32):
    df = pd.read_csv('game_log.csv')
    game_a, game_b = games
    score_a, score_b = result

    game_a_elo = df.loc[df.Title == game_a, 'elo'].iloc[0]
    game_b_elo = df.loc[df.Title == game_b, 'elo'].iloc[0]

    expected_a = (1+10**((game_b_elo-game_a_elo)/400.0)) ** -1
    expected_b = (1+10**((game_a_elo-game_b_elo)/400.0)) ** -1

    game_a_new_elo = game_a_elo + k * (score_a - expected_a)
    game_b_new_elo = game_b_elo + k * (score_b - expected_b)

    df.loc[df.Title == game_a, 'elo'] = game_a_new_elo
    df.loc[df.Title == game_b, 'elo'] = game_b_new_elo
    df.to_csv('game_log.csv', index=False)

    print('')
    print(f'{game_a}: {game_a_elo} -> {game_a_new_elo}')
    print(f'{game_b}: {game_b_elo} -> {game_b_new_elo}')


def _glicko(ratings, rds, s_i):
    r_0, r_i = ratings
    rd, rd_i = rds

    q = np.log(10)/400
    g = 1 / np.sqrt(1 + ((3 * q**2 * rd_i**2) / (np.pi ** 2)))
    E = 1 / (1 + (10 ** ((g * (r_0-r_i))/-400)))

    d2 = 1 / (q**2 * g**2 * E * (1-E))
    r = r_0 + (q / ((1/rd**2) + (1/d2))) * g * (s_i - E)

    rd_prime = np.sqrt(((1/rd**2) + (1/d2))**-1)
    return r, rd_prime


def glicko(games, result):
    df = pd.read_csv('game_log.csv')
    game_a, game_b = games
    score_a, score_b = result

    game_a_glicko = df.loc[df.Title == game_a, 'glicko'].iloc[0]
    game_b_glicko = df.loc[df.Title == game_b, 'glicko'].iloc[0]

    game_a_rd = df.loc[df.Title == game_a, 'glicko_rd'].iloc[0]
    game_b_rd = df.loc[df.Title == game_b, 'glicko_rd'].iloc[0]

    a_glicko, a_rd = _glicko((game_a_glicko, game_b_glicko), (game_a_rd, game_b_rd), score_a)
    b_glicko, b_rd = _glicko((game_b_glicko, game_a_glicko), (game_b_rd, game_a_rd), score_b)

    df.loc[df.Title == game_a, 'glicko'] = a_glicko
    df.loc[df.Title == game_b, 'glicko'] = b_glicko
    df.loc[df.Title == game_a, 'glicko_rd'] = a_rd
    df.loc[df.Title == game_b, 'glicko_rd'] = b_rd

    df.to_csv('game_log.csv', index=False)

    print('')
    print(f'{game_a}: {game_a_glicko} ({game_a_rd}) -> {a_glicko} ({a_rd})')
    print(f'{game_b}: {game_b_glicko} ({game_b_rd}) -> {b_glicko} ({b_rd})')
    print('')


def trials(game=None, n=1):
    count = 0
    while count < n:
        trial(game_a=game, game_b=None)
        count += 1


def tournament(n):
    competitors = 2**n_trials
    df = pd.read_csv('game_log.csv')
    played_df = df.loc[df.Finished == 1].copy()

    competitors = played_df.loc[:, 'Title'].sample(n=competitors).tolist()

    for n in range(n_trials):
        winners = []
        comp_df = df.loc[df.Title.isin(competitors), ['Title', 'elo', 'glicko']].T
        while comp_df.shape[1] > 1:
            low, high = comp_df.loc['glicko'].idxmin(), comp_df.loc['glicko'].idxmax()
            low_game = comp_df.pop(low).Title
            high_game = comp_df.pop(high).Title
            winner = trial(game_a=low_game, game_b=high_game)
            winners.append(winner)
        competitors = winners


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_trials')
    parser.add_argument('--game', default=None)
    args = parser.parse_args()

    n_trials = int(args.n_trials)
    game = args.game
    print(game)
    trial(game_a=game)
