SELECT
    games.id,
    REPLACE(REPLACE(metacritic.genre, ' ','-'), '-','') AS genre,
    IFNULL(DATETIME(igdb.first_release_date, 'unixepoch'),
            DATETIME(metacritic.releaseDate)) AS release_date,
    JULIANDAY(CURRENT_DATE) - JULIANDAY(IFNULL(DATETIME(igdb.first_release_date, 'unixepoch'),
            DATETIME(metacritic.releaseDate))) AS age,
    metacritic.developer,
    metacritic.userScore,
    metacritic.metaScore,
    igdb.category,
    IFNULL(igdb.game_engines, '[]') AS game_engines,
    igdb.franchise,
    IFNULL(igdb.similar_games, '[]') AS similar_games,
    IFNULL(igdb.genres, '[]') AS genres,
    IFNULL(igdb.themes, '[]') AS themes,
    igdb.rating_count,
    igdb.aggregated_rating_count,
    IFNULL(igdb.keywords, '[]') AS keywords,
    IFNULL(igdb.game_modes, '[]') as game_modes,
    IFNULL(igdb.involved_companies, '[]') as involved_companies,
    igdb.rating,
    igdb.aggregated_rating,
    hltb.count_comp,
    hltb.count_retired / hltb.count_comp AS retire_rate,
    hltb.comp_main_count / hltb.count_comp AS comp_main_rate,
    hltb.comp_plus_count / hltb.count_comp AS comp_plus_rate,
    hltb.comp_100_count / hltb.count_comp AS comp_100_rate,
    hltb.comp_all_count / hltb.count_comp AS comp_all_rate,
    hltb.count_review / hltb.count_comp AS review_rate,
    hltb.count_review,
    hltb.count_speedrun / hltb.count_comp AS speedrun_rate,
    hltb.count_speedrun,
    hltb.review_score,
    hltb.game_type,
    IFNULL(recommendations.AM, 0) AS AM,
    IFNULL(recommendations.BR, 0) AS BR,
    IFNULL(recommendations."Buried Treasure", 0) AS "Buried Treasure",
    IFNULL(recommendations.Fabio, 0) AS Fabio,
    IFNULL(recommendations.Jackie, 0) AS Jackie,
    IFNULL(recommendations.Kaleb, 0) AS Kaleb,
    IFNULL(recommendations.Nick, 0) AS Nick,
    IFNULL(recommendations.Sterling, 0) AS Sterling,
    IFNULL(recommendations.Yahtzee, 0) AS Yahtzee,
    IFNULL(recommendations.fightincowboy, 0) AS fightincowboy,
    IFNULL(recommendations.hbomberguy, 0) AS hbomberguy,
    ros.series_glicko_mean,
    ros.series_peak_glicko,
    ros.series_valley_glicko,
    prequel_rating.prequel_glicko,
    IIF(classics.id IS NULL, 0 , 1) AS classic,
    IIF(soulslikes.id IS NULL, 0 , 1) AS soulslike,
    ratings.glicko
FROM games
LEFT OUTER JOIN metacritic ON games.id = metacritic.game_id
LEFT OUTER JOIN igdb ON games.id = igdb.game_id
LEFT OUTER JOIN  hltb ON games.id = hltb.game_id
LEFT OUTER JOIN recommendations ON games.Title = recommendations.Title
LEFT OUTER JOIN (SELECT games.id,
                        AVG(glicko) As series_glicko_mean,
                        MAX(glicko) AS series_peak_glicko,
                        MIN(glicko) AS series_valley_glicko
                 FROM games
                     JOIN series s On s.Title = games.Title
                     JOIN series o ON o.id = s.id AND o.Title != s.Title
                     JOIN ratings ON ratings.Title = o.Title
                 GROUP BY games.Title) ros ON games.id = ros.id
LEFT OUTER JOIN (SELECT games.id, ratings.glicko AS prequel_glicko
                 FROM games
                     JOIN series s On s.Title = games.Title
                     JOIN series o ON o.id = s.id AND o.sub_id = s.sub_id - 1
                     JOIN ratings ON ratings.Title = o.Title
                 GROUP BY games.Title) prequel_rating ON games.id = prequel_rating.id
LEFT OUTER JOIN classics ON games.id = classics.game_id
LEFT OUTER JOIN soulslikes ON games.id = soulslikes.game_id
LEFT OUTER JOIN ratings ON games.id = ratings.game_id
WHERE games.id IN {game_ids}
