import http.client
import json
from datetime import date

from fuzzywuzzy import fuzz
import joblib
import pandas as pd
import numpy as np
from howlongtobeatpy import HowLongToBeat, SearchModifiers

from model.model import update_model_scores

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


def _find_game_idx(df, title):
    try:
        return df.index[df.Title.str.lower() == title.lower()][0]
    except:
        raise('Could not find game. No index provided.')


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


def fetch_opencritic_game(title, distance_threshold=0.1):
    oc_conn = http.client.HTTPSConnection("opencritic-api.p.rapidapi.com")

    oc_headers = {
        'x-rapidapi-key': "3c8fa7e8cdmsh21fc03a2acc587ep1277bejsn309b6f765d3d",
        'x-rapidapi-host': "opencritic-api.p.rapidapi.com"
    }
    game = title.replace(' ', '%20')

    oc_conn.request("GET", f"/game/search?criteria={game}", headers=oc_headers)

    res = oc_conn.getresponse()
    data = res.read()
    try:
        game_data = json.loads(data.decode('utf-8'))[0]
        if game_data['dist'] <= distance_threshold:
            return game_data
        else:
            print(f'closet match: {game_data}')
            print('Change distance threshold and rerun if want to keep above match')
            return {key: np.nan for key in game_data}
    except:
        return {'id': np.nan, 'name': np.nan, 'dist': np.nan}


def get_opencritic_info(oc_id):
    oc_conn = http.client.HTTPSConnection("opencritic-api.p.rapidapi.com")

    oc_headers = {
        'x-rapidapi-key': "3c8fa7e8cdmsh21fc03a2acc587ep1277bejsn309b6f765d3d",
        'x-rapidapi-host': "opencritic-api.p.rapidapi.com"
    }

    oc_conn.request("GET", f"/game/{oc_id}", headers=oc_headers)

    res = oc_conn.getresponse()
    data = res.read()
    return json.loads(data.decode('utf-8'))


def update_opencritic(title, game_idx=None, distance_threshold=0.1, dry_run=False):
    df = pd.read_csv('game_log.csv')
    if game_idx is None:
        game_idx = _find_game_idx(df, title)

    game_data = fetch_opencritic_game(title=title, distance_threshold=distance_threshold)
    if ~np.isnan(game_data['id']):
        oc_info = get_opencritic_info(oc_id=game_data['id'])
    else:
        oc_info = {}
    for key in oc_info:
        game_data[key] = oc_info[key]
    oc_df = pd.DataFrame.from_dict(game_data, orient='index').T
    oc_df.index = [game_idx]
    oc_cols = oc_df.columns[oc_df.columns.isin(df.columns)]
    df.loc[game_idx, oc_cols] = oc_df.loc[game_idx, oc_cols]
    if dry_run:
        df.to_csv('cache/game_log_oc_test.csv', index=False)
    else:
        df.to_csv('game_log.csv', index=False)


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


def update_igdb_info(title, game_idx=None, dry_run=False):
    df = pd.read_csv('game_log.csv')
    if game_idx is None:
        game_idx = _find_game_idx(df, title)
    try:
        igdb_results = get_igdb_info(title)
    except(ValueError):
        print('igdb lookup failed')
        igdb_results = {'id': np.nan}
    igdb_df = pd.DataFrame.from_dict(igdb_results, orient='index').T
    igdb_df.index = [game_idx]
    igdb_df.rename(columns=lambda x: f'{x}_igdb', inplace=True)
    igdb_cols = igdb_df.columns[igdb_df.columns.isin(df.columns)]
    df.loc[game_idx, igdb_cols] = igdb_df.loc[game_idx, igdb_cols]
    if dry_run:
        df.to_csv('cache/game_log_igdb_test.csv', index=False)
    else:
        df.to_csv('game_log.csv', index=False)


def update_metacritic(metacritic_url, title=None, dry_run=False):
    df = pd.read_csv('game_log.csv')
    game_df = df.loc[df.metacritic_url == metacritic_url]
    if len(game_df) == 0:
        game_idx = df.index.max() + 1
        df.loc[game_idx, 'metacritic_url'] = metacritic_url
        if title is None:
            title = metacritic_url.split('/')[-2].replace('-', ' ').title()
        df.loc[game_idx, 'Title'] = title
    else:
        if len(game_df) > 1:
            print('Duplicate Entries Present.')
        game_idx = game_df.index[0]
        if title is None:
            title = df.loc[game_idx, 'Title']
        else:
            df.loc[game_idx, 'Title'] = title
    metacritic_result = get_metacritic_info(metacritic_url=metacritic_url)
    metacritic_df = pd.DataFrame.from_dict(metacritic_result, orient='index').T
    metacritic_df.index = [game_idx]
    metacritic_df.rename(columns=lambda x: f'{x}_metacritic', inplace=True)
    metacritic_cols = metacritic_df.columns[metacritic_df.columns.isin(df.columns)]
    df.loc[game_idx, metacritic_cols] = metacritic_df.loc[game_idx, metacritic_cols]
    if dry_run:
        df.to_csv('cache/game_log_mc_test.csv', index=False)
    else:
        df.to_csv('game_log.csv', index=False)
    return game_idx, title


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


def update_hltb(title, game_idx=None, hltb_id=None, distance_threshold=0.15, dry_run=False):
    df = pd.read_csv('game_log.csv')
    if game_idx is None:
        game_idx = _find_game_idx(df, title)

    game_data = fetch_hltb(title=title, hltb_id=hltb_id, distance_threshold=distance_threshold)
    hltb_df = pd.DataFrame(index=[int(game_idx)], data=game_data)
    hltb_df.rename(columns=lambda x: f'{x}_hltb', inplace=True)
    df.loc[int(game_idx), hltb_df.columns] = hltb_df.loc[int(game_idx), hltb_df.columns]

    if dry_run:
        df.to_csv('cache/game_log_hltb_test.csv', index=False)
    else:
        df.to_csv('game_log.csv', index=False)


def update_game(
        metacritic_url=None, title=None, oc_distance_threshold=0.1, hltb_distance_threshold=0.15,
        dry_run=False, classic=False, rating=None):
    df = pd.read_csv('game_log.csv')
    print(metacritic_url, title)
    if metacritic_url is None:
        try:
            metacritic_url = df.set_index('Title').loc[title, 'metacritic_url']
            df.to_csv('game_log.csv', index=False)
        except:
            raise(Exception('No Metacritic URL and Title not found'))

    game_idx, title = update_metacritic(metacritic_url=metacritic_url, title=title, dry_run=dry_run)
    df = pd.read_csv('game_log.csv')
    if classic:
        df.loc[game_idx, 'Classic'] = 1
    else:
        df.loc[game_idx, 'Classic'] = 0
    if rating is not None:
        df.loc[game_idx, 'My Rating'] = rating
    df.to_csv('game_log.csv', index=False)
    print(metacritic_url, title)
    update_opencritic(title=title, game_idx=game_idx, distance_threshold=oc_distance_threshold, dry_run=dry_run)
    update_igdb_info(title=title, game_idx=game_idx, dry_run=dry_run)
    update_hltb(title=title, game_idx=game_idx, dry_run=dry_run, distance_threshold=hltb_distance_threshold)
    update_model_scores()
    df = pd.read_csv('game_log.csv')
    return df.loc[game_idx, ['Title', 'My Rating', 'raw_score', 'model_score']]

