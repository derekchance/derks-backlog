SELECT
    games.Title,
    model_scores.adj_score,
    model_scores.raw_score,
    metacritic.metaScore,
    hltb.comp_all / 3600 AS time_est,
    metacritic.genre,
    metacritic.developer,
    metacritic.releaseDate
FROM games
JOIN model_scores ON games.id = model_scores.game_id
LEFT OUTER JOIN hltb ON games.id = hltb.game_id
LEFT OUTER JOIN metacritic ON metacritic.game_id = games.id
WHERE games.id NOT IN (SELECT game_id FROM last_played)
ORDER BY adj_score DESC