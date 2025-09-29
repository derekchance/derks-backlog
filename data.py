from datetime import datetime
import http.client
import json
import sqlite3
import warnings

from fuzzywuzzy import fuzz
import joblib
import pandas as pd
import numpy as np
from howlongtobeatpy import HowLongToBeat, SearchModifiers
from sheets import update_backlog_values, update_log_values

from model.core import MODEL_DIR, load_dataset

igdb_client_id = 'w7mohm48mdexvwnjg1dayuku9vuu0h'
igdb_client_secret = 'fwb7dt2nucfyrvd9mrwaa2ge3oh0cz'

twitch_conn = http.client.HTTPSConnection("id.twitch.tv")
twitch_headers = {
    'client_id': igdb_client_id,
    'client_secret': igdb_client_secret,
    'grant_type': 'client_credentials',
}
twitch_conn.request(
    'POST',
    f'/oauth2/token?client_id={igdb_client_id}&client_secret={igdb_client_secret}&grant_type=client_credentials',
    headers=twitch_headers)
res = twitch_conn.getresponse()
data = res.read()
twitch_access_token = json.loads(data.decode('utf-8'))['access_token']


def safe_loads(x):
    try:
        return json.loads(x.replace("'", '"'))[0]
    except:
        return {}


def _find_game_idx(title):
    try:
        with sqlite3.connect('games.db') as con:
            cur = con.cursor()
            cur.execute(f"SELECT id FROM games WHERE Title Like ?", (title,))
            game_idx = cur.fetchone()
            assert game_idx is not None
            return game_idx[0]
    except AssertionError as e:
        raise KeyError('Title Not Found')



def get_metacritic_info(metacritic_url):
    mc_conn = http.client.HTTPSConnection("metacriticapi.p.rapidapi.com")
    mc_headers = {
        'x-rapidapi-key': "3c8fa7e8cdmsh21fc03a2acc587ep1277bejsn309b6f765d3d",
        'x-rapidapi-host': "metacriticapi.p.rapidapi.com"
    }

    metacritic_url = metacritic_url.replace('https://www.metacritic.com/game/', '/games/')
    mc_conn.request("GET", f"{metacritic_url}", headers=mc_headers)

    res = mc_conn.getresponse()
    data = res.read()
    return json.loads(data.decode('utf-8'))


def _get_igdb_query(title, strict=False, alt_title=None, limit=None):
    query = f'''
    search "{title}";
    fields name,
    age_ratings,
    aggregated_rating,
    aggregated_rating_count,
    category,
    first_release_date,
    franchise,
    game_engines,
    game_modes,
    genres,
    hypes,
    involved_companies,
    keywords,
    rating,
    rating_count,
    similar_games,
    themes;
    '''
    if strict | (alt_title is not None):
        if alt_title is not None:
            title = alt_title
        query += f' where name = "{title}";'
    if limit is not None:
        query += f' limit {limit};'
    return query

def _get_igdb_info(title, strict=False, alt_title=None, limit=None):
    igdb_conn = http.client.HTTPSConnection("api.igdb.com")
    igdb_headers = {
        'Client-ID': igdb_client_id,
        'Authorization': f'Bearer {twitch_access_token}',
    }
    query = _get_igdb_query(title, strict=strict, alt_title=alt_title, limit=limit)
    igdb_conn.request('POST', '/v4/games/', headers=igdb_headers, body=query)
    res = igdb_conn.getresponse()
    data = res.read()
    return json.loads(data.decode('utf-8'))


def get_igdb_info(title):
    try:
        response = _get_igdb_info(title, strict=True, limit=1)[0]
        response['ratio'] = 100
        return _get_igdb_info(title, strict=True, limit=1)[0]
    except:
        responses = _get_igdb_info(title)
        ratio = [fuzz.ratio(title, n['name']) for n in responses]
        matched_response = responses[np.argmax(ratio)]
        matched_response['ratio'] = np.max(ratio)
        return matched_response


def update_igdb_info(title, game_idx=None):
    if game_idx is None:
        game_idx = _find_game_idx(title)
    try:
        igdb_results = get_igdb_info(title)
    except ValueError:
        print('igdb lookup failed')
        igdb_results = {'id': np.nan}

    for i, j in igdb_results.items():
        if isinstance(j, list):
            igdb_results[i] = str(j)
    igdb_results['game_id'] = game_idx

    for i in ['id', 'age_ratings', 'aggregated_rating', 'aggregated_rating_count', 'first_release_date', 'game_modes',
              'genres', 'hypes', 'involved_companies', 'keywords', 'name', 'rating', 'rating_count', 'similar_games',
              'themes']:
        igdb_results[i] = igdb_results.get(i, np.nan)
    with sqlite3.connect('games.db') as con:
        statement = '''
                    INSERT INTO igdb (id, age_ratings, aggregated_rating, aggregated_rating_count, first_release_date, \
                                      game_modes, genres, hypes, involved_companies, keywords, name, rating, \
                                      rating_count, similar_games, themes, game_id, dlu)
                    VALUES (:id, :age_ratings, :aggregated_rating, :aggregated_rating_count, :first_release_date, \
                            :game_modes, :genres, :hypes, :involved_companies, :keywords, :name, :rating, :rating_count, \
                            :similar_games, :themes, :game_id, CURRENT_TIMESTAMP)
                    ON CONFLICT(game_id) DO UPDATE SET id=:id, \
                                                       age_ratings=:age_ratings, \
                                                       aggregated_rating=:aggregated_rating, \
                                                       aggregated_rating_count=:aggregated_rating_count, \
                                                       first_release_date=:first_release_date, \
                                                       game_modes=:game_modes, \
                                                       genres=:genres, \
                                                       hypes=:hypes, \
                                                       involved_companies=:involved_companies, \
                                                       keywords=:keywords, \
                                                       name=:name, \
                                                       rating=:rating, \
                                                       rating_count=:rating_count, \
                                                       similar_games=:similar_games, \
                                                       themes=:themes, \
                                                       dlu=CURRENT_TIMESTAMP
                    WHERE game_id = :game_id \
                    '''
        cur = con.cursor()
        cur.executemany(statement, (igdb_results,))



def update_metacritic(metacritic_url, game_idx=None, title=None):
    if game_idx is None:
        game_idx = _find_game_idx(title)
    metacritic_result = get_metacritic_info(metacritic_url=metacritic_url)
    metacritic_result['game_id'] = game_idx
    metacritic_result['url'] = metacritic_url
    with sqlite3.connect('games.db') as con:
        statement = '''
                    INSERT INTO metacritic (game_id, genre, releaseDate, developer, userScore, metaScore, url, platform,
                                            description, dlu)
                    VALUES (:game_id, :genre, :releaseDate, :developer, :userScore, :metaScore, :url, :platform,
                            :description, CURRENT_TIMESTAMP)
                    ON CONFLICT(game_id) DO UPDATE SET genre=:genre, 
                                                       releaseDate=:releaseDate, 
                                                       developer=:developer, 
                                                       userScore=:userScore,
                                                       metaScore=:metaScore,
                                                       url=:url,
                                                       platform=:platform,
                                                       description=:description,
                                                       dlu=CURRENT_TIMESTAMP
                        
                    WHERE game_id = :game_id 
                    '''
        cur = con.cursor()
        cur.executemany(statement, (metacritic_result,))



def fetch_hltb(title, hltb_id=None, distance_threshold=0.15):
    hltb_fields = [
        'game_id',
        'game_name',
        'game_name_date',
        'game_alias',
        'game_type',
        'game_image',
        'comp_lvl_combine',
        'comp_lvl_sp',
        'comp_lvl_co',
        'comp_lvl_mp',
        'comp_main',
        'comp_plus',
        'comp_100',
        'comp_all',
        'comp_main_count',
        'comp_plus_count',
        'comp_100_count',
        'comp_all_count',
        'invested_co',
        'invested_mp',
        'invested_co_count',
        'invested_mp_count',
        'count_comp',
        'count_speedrun',
        'count_backlog',
        'count_review',
        'review_score',
        'count_playing',
        'count_retired',
        'profile_platform',
        'profile_popular',
        'release_world'
    ]

    if hltb_id is not None:
        result = HowLongToBeat().search_from_id(hltb_id).json_content
    else:
        results_list = HowLongToBeat(distance_threshold).search(title, search_modifiers=SearchModifiers.HIDE_DLC)
        if results_list is not None and len(results_list) > 0:
            result = max(results_list, key=lambda element: element.similarity)
            result = result.json_content
        else:
            results_list = HowLongToBeat(distance_threshold).search(title, search_modifiers=SearchModifiers.ISOLATE_DLC)
            if results_list is not None and len(results_list) > 0:
                result = max(results_list, key=lambda element: element.similarity)
                result = result.json_content
            else:
                result = {n: np.nan for n in hltb_fields}
    return result


def update_hltb(title, game_idx=None, hltb_id=None, distance_threshold=0.15):
    if game_idx is None:
        game_idx = _find_game_idx(title=title)

    game_data = fetch_hltb(title=title, hltb_id=hltb_id, distance_threshold=distance_threshold)
    game_data['hltb_id'] = game_data['game_id']
    game_data['game_id'] = game_idx
    with sqlite3.connect('games.db') as con:
        statement = '''
                    INSERT INTO hltb (game_id, hltb_id, game_name, game_name_date, game_alias, game_type, game_image,
                                      comp_lvl_combine, comp_lvl_sp, comp_lvl_co, comp_lvl_mp, comp_main, comp_plus,
                                      comp_100, comp_all, comp_main_count, comp_plus_count, comp_100_count,
                                      comp_all_count, invested_co, invested_mp, invested_co_count, invested_mp_count,
                                      count_comp, count_speedrun, count_backlog, count_review, review_score,
                                      count_playing, count_retired, profile_platform, profile_popular, release_world, dlu)
                    VALUES (:game_id, :hltb_id, :game_name, :game_name_date, :game_alias, :game_type, :game_image,
                            :comp_lvl_combine, :comp_lvl_sp, :comp_lvl_co, :comp_lvl_mp, :comp_main, :comp_plus,
                            :comp_100, :comp_all, :comp_main_count, :comp_plus_count, :comp_100_count, :comp_all_count,
                            :invested_co, :invested_mp, :invested_co_count, :invested_mp_count, :count_comp,
                            :count_speedrun, :count_backlog, :count_review, :review_score, :count_playing,
                            :count_retired, :profile_platform, :profile_popular, :release_world, CURRENT_TIMESTAMP)
                    ON CONFLICT(game_id) DO UPDATE SET hltb_id=:hltb_id,
                                                       game_name=:game_name,
                                                       game_name_date=:game_name_date,
                                                       game_alias=:game_alias,
                                                       game_type=:game_type,
                                                       game_image=:game_image,
                                                       comp_lvl_combine=:comp_lvl_combine,
                                                       comp_lvl_sp=:comp_lvl_sp,
                                                       comp_lvl_co=:comp_lvl_co,
                                                       comp_lvl_mp=:comp_lvl_mp,
                                                       comp_main=:comp_main,
                                                       comp_plus=:comp_plus,
                                                       comp_100=:comp_100,
                                                       comp_all=:comp_all,
                                                       comp_main_count=:comp_main_count,
                                                       comp_plus_count=:comp_plus_count,
                                                       comp_100_count=:comp_100_count,
                                                       comp_all_count=:comp_all_count,
                                                       invested_co=:invested_co,
                                                       invested_mp=:invested_mp,
                                                       invested_co_count=:invested_co_count,
                                                       invested_mp_count=:invested_mp_count,
                                                       count_comp=:count_comp,
                                                       count_speedrun=:count_speedrun,
                                                       count_backlog=:count_backlog,
                                                       count_review=:count_review,
                                                       review_score=:review_score,
                                                       count_playing=:count_playing,
                                                       count_retired=:count_retired,
                                                       profile_platform=:profile_platform,
                                                       profile_popular=:profile_popular,
                                                       release_world=:release_world,
                                                       dlu=CURRENT_TIMESTAMP

                    WHERE game_id = :game_id \
                    '''
        cur = con.cursor()
        cur.executemany(statement, (game_data,))


def update_game(title, metacritic_url=None, hltb_distance_threshold=0.15, classic=False, soulslike=False):
    try:
        game_idx = _find_game_idx(title=title)
    except KeyError:
        with sqlite3.connect('games.db') as con:
            statement = 'INSERT INTO games (Title) VALUES (?)'
            cur = con.cursor()
            cur.execute(statement, (title, ))

        game_idx = _find_game_idx(title=title)

    if metacritic_url is None:
        with sqlite3.connect('games.db') as con:
            cur = con.cursor()
            cur.execute('SELECT url FROM metacritic WHERE game_id = ?', (game_idx,))
            try:
                metacritic_url = cur.fetchone()[0]
                assert isinstance(metacritic_url, str)
            except (TypeError, AssertionError):
                raise 'Metacritic URL not found. Include URL as keyword argument.'

    update_metacritic(metacritic_url=metacritic_url, title=title, game_idx=game_idx)


    if classic:
        with sqlite3.connect('games.db') as con:
            statement = 'INSERT INTO classics (game_id, Title) VALUES (?, ?) ON CONFLICT (game_id) DO NOTHING'
            cur = con.cursor()
            cur.execute(statement, (title,))

    if soulslike:
        with sqlite3.connect('games.db') as con:
            statement = 'INSERT INTO soulslikes (game_id, Title) VALUES (?, ?) ON CONFLICT (game_id) DO NOTHING'
            cur = con.cursor()
            cur.execute(statement, (title,))

    print(metacritic_url, title)
    update_igdb_info(title=title, game_idx=game_idx)
    update_hltb(title=title, game_idx=game_idx, distance_threshold=hltb_distance_threshold)
    update_model_scores(game_id=game_idx)
    update_backlog_values()
    update_log_values()



def mark_played(game_id=None, title=None, date_played=None):
    assert (game_id is not None) | (title is not None), 'Must provide either game_id or title'
    if game_id is None:
        game_id = _find_game_idx(title=title)

    if date_played is None:
        date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        date_str = datetime.strptime('2025-08-29', '%Y-%m-%d').strftime('%Y-%m-%d %H:%M:%S')


    with sqlite3.connect('games.db') as con:
        statement = '''
            INSERT INTO last_played (game_id, last_played) VALUES (:game_id, :date_str)
            ON CONFLICT (game_id) DO UPDATE SET last_played=:date_str WHERE game_id=:game_id
                    '''
        cur = con.cursor()
        cur.executemany(statement, ({'game_id': game_id, 'date_str': date_str}, ))


def mark_dropped(game_id=None, title=None):
    assert (game_id is not None) | (title is not None), 'Must provide either game_id or title'
    if game_id is None:
        game_id = _find_game_idx(title=title)

    with sqlite3.connect('games.db') as con:
        statement = '''
            INSERT INTO dropped (game_id) VALUES (?)
            ON CONFLICT (game_id) DO NOTHING
                    '''
        cur = con.cursor()
        cur.executemany(statement, (game_id,))


def update_raw_model_scores(game_id='all'):
    model_input = load_dataset(game_ids=game_id)
    model = joblib.load(MODEL_DIR / f'models/realmlp_model.joblib')
    with warnings.catch_warnings(action='ignore'):
        raw_scores = model.predict(model_input)
    if game_id == 'all':
        rs_df = pd.Series(index=model_input['id'], data=raw_scores, name='raw_score').to_frame()
        with sqlite3.connect('games.db') as con:
            cur = con.cursor()
            cur.execute('DROP TABLE model_scores')
            cur.execute('CREATE TABLE model_scores(game_id INTEGER PRIMARY KEY, raw_score REAL)')
            rs_df.to_sql('model_scores', con, if_exists='append', index='game_id')
    elif isinstance(game_id, int):
        with sqlite3.connect('games.db') as con:
            statement = '''
                        INSERT INTO model_scores (game_id, raw_score) VALUES (:game_id, :raw_score)
                        ON CONFLICT (game_id) DO UPDATE SET raw_score=:raw_score WHERE game_id=:game_id
                        '''
            model_data=(
                {'game_id': int(game_id), 'raw_score': float(raw_scores[0])},
            )
            #return model_data
            cur = con.cursor()
            cur.executemany(statement, model_data)
    else:
        raise('Valid inputs are "all" or the game_id (game.id)')


def update_sequel_adjusted_scores():
    """Gives preference to earlier titles in series with better scoring sequels"""
    with sqlite3.connect('games.db') as con:
        with open('series_adj.sql') as f:
            statement = f.read()

        cur = con.cursor()
        cur.execute(statement)


def update_model_scores(game_id='all'):
    update_raw_model_scores(game_id=game_id)
    update_sequel_adjusted_scores()
