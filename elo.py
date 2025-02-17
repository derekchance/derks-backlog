import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('n_trials')
args = parser.parse_args()

n_trials = int(args.n_trials)


def trial(game_a=None, game_b=None):
    df = pd.read_csv('game_log.csv')
    played_df = df.loc[df.Finished == 1].copy()

    if game_a is None and game_b is None:
        game_a, game_b = played_df.Title.sample(n=2)
    elif game_a is None:
        game_a = played_df.loc[played_df.Title != game_b, 'Title'].sample(n=1).iloc[0]
    elif game_b is None:
        game_b = played_df.loc[played_df.Title != game_a, 'Title'].sample(n=1).iloc[0]

    game_a_elo = df.loc[df.Title == game_a, 'elo'].iloc[0]
    game_b_elo = df.loc[df.Title == game_b, 'elo'].iloc[0]
    df.loc[df.Title == game_a, 'Trials'] += 1
    df.loc[df.Title == game_b, 'Trials'] += 1
    k = 24

    expected_a = (1+10**((game_b_elo-game_a_elo)/400.0)) ** -1
    expected_b = (1+10**((game_a_elo-game_b_elo)/400.0)) ** -1

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

    game_a_new_elo = game_a_elo + k * (score_a - expected_a)
    game_b_new_elo = game_b_elo + k * (score_b - expected_b)

    df.loc[df.Title == game_a, 'elo'] = game_a_new_elo
    df.loc[df.Title == game_b, 'elo'] = game_b_new_elo

    df.loc[df.Title == game_a, 'Trials'] += 1
    df.loc[df.Title == game_b, 'Trials'] += 1

    print('')
    print(f'{game_a}: {game_a_elo} -> {game_a_new_elo}')
    print(f'{game_b}: {game_b_elo} -> {game_b_new_elo}')

    df.to_csv('game_log.csv', index=False)


def trials(game=None, n=1):
    count = 0
    while count < n:
        trial(game_a=game, game_b=None)
        count += 1


if __name__ == '__main__':
    trials(n=n_trials)
