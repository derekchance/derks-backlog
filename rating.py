import argparse
import pandas as pd
import numpy as np
import sqlite3


def trial(game_a=None, game_b=None, primary_rating='glicko'):
    with sqlite3.connect('games.db') as con:
        query = '''
          SELECT
              games.id,
              games.Title,
              last_played,
              Trials,
              elo,
              glicko,
              glicko_rd
          FROM games
                   LEFT JOIN last_played
                        ON games.id = last_played.game_id
                   LEFT JOIN ratings
                        ON games.id = ratings.game_id
          '''
        df = pd.read_sql(query, con, index_col='id')
    played_df = df.loc[df.last_played.notna()].copy()

    if game_a is None:
        weights = (1 + played_df['Trials'].max() - played_df['Trials']) ** 2
        weights = np.where(
            played_df.Title == game_b,
            0,
            weights
        )
        game_a = played_df.sample(n=1, weights=weights)
    else:
        game_a = played_df.loc[played_df.Title == game_a]
    game_a_elo = df.loc[game_a.index[0], primary_rating]

    if game_b is None:
        dist = (game_a_elo - played_df[primary_rating]).abs()
        weights = 1/(dist + 1)**3
        weights = np.where(played_df.Title == game_a['Title'].iloc[0],
                           0,
                           weights,
                           )
        game_b = played_df.sample(n=1, weights=weights)
    else:
        game_b = played_df.loc[played_df.Title == game_b]


    print(f'1: {game_a['Title'].iloc[0]}')
    print(f'2: {game_b['Title'].iloc[0]}')
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

    df.loc[game_a.index[0], 'Trials'] += 1
    df.loc[game_b.index[0], 'Trials'] += 1

    with sqlite3.connect('games.db') as con:
        df.drop(columns=['last_played'])\
          .to_sql('ratings',
                  con,
                  if_exists='replace',
                  index='game_id'
                  )

    elo(games=(game_a, game_b), result=(score_a, score_b))
    glicko(games=(game_a, game_b), result=(score_a, score_b))
    return [game_a, game_b][int(winner) - 1]


def elo(games, result, k=32):
    game_a, game_b = games
    score_a, score_b = result

    with sqlite3.connect('games.db') as con:
        query = "SELECT elo FROM ratings WHERE game_id IN (?, ?)"
        cur = con.execute(query, (int(game_a.index[0]), int(game_b.index[0])))
        game_a_elo, game_b_elo = [n[0] for n in cur.fetchall()]

    expected_a = (1+10**((game_b_elo-game_a_elo)/400.0)) ** -1
    expected_b = (1+10**((game_a_elo-game_b_elo)/400.0)) ** -1

    game_a_new_elo = game_a_elo + k * (score_a - expected_a)
    game_b_new_elo = game_b_elo + k * (score_b - expected_b)

    with sqlite3.connect('games.db') as con:
        statement = "UPDATE ratings SET elo=? WHERE game_id=?"
        cur = con.cursor()
        cur.execute(statement, (game_a_new_elo, int(game_a.index[0])))
        cur.execute(statement, (game_b_new_elo, int(game_b.index[0])))
        con.commit()

    print('')
    print(f'{game_a['Title'].iloc[0]}: {game_a_elo} -> {game_a_new_elo}')
    print(f'{game_b['Title'].iloc[0]}: {game_b_elo} -> {game_b_new_elo}')


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
    game_a, game_b = games
    score_a, score_b = result

    with sqlite3.connect('games.db') as con:
        query = "SELECT glicko FROM ratings WHERE game_id IN (?, ?)"
        cur = con.execute(query, (int(game_a.index[0]), int(game_b.index[0])))
        game_a_glicko, game_b_glicko = [n[0] for n in cur.fetchall()]

        query = "SELECT glicko_rd FROM ratings WHERE game_id IN (?, ?)"
        cur = con.execute(query, (int(game_a.index[0]), int(game_b.index[0])))
        game_a_rd, game_b_rd = [n[0] for n in cur.fetchall()]

    a_glicko, a_rd = _glicko((game_a_glicko, game_b_glicko), (game_a_rd, game_b_rd), score_a)
    b_glicko, b_rd = _glicko((game_b_glicko, game_a_glicko), (game_b_rd, game_a_rd), score_b)

    with sqlite3.connect('games.db') as con:
        statement = "UPDATE ratings SET glicko=?, glicko_rd=? WHERE game_id=?"
        cur = con.cursor()
        cur.execute(statement, (a_glicko, a_rd, int(game_a.index[0])))
        cur.execute(statement, (b_glicko, b_rd, int(game_b.index[0])))
        con.commit()


    print('')
    print(f'{game_a['Title'].iloc[0]}: {game_a_glicko} ({game_a_rd}) -> {a_glicko} ({a_rd})')
    print(f'{game_b['Title'].iloc[0]}: {game_b_glicko} ({game_b_rd}) -> {b_glicko} ({b_rd})')
    print('')


def trials(game=None, n=1):
    count = 0
    while count < n:
        trial(game_a=game, game_b=None)
        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_trials')
    parser.add_argument('--game', default=None)
    args = parser.parse_args()

    n_trials = int(args.n_trials)
    game = args.game
    trial(game_a=game)
