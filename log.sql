SELECT
    games.Title,
    ratings.glicko,
    model_scores.raw_score,
    model_scores.raw_score * POWER(1 + 100 * exp(-0.0035 * (JULIANDAY(CURRENT_TIMESTAMP) - JULIANDAY(last_played.last_played))), -1) AS replay_score,
    last_played.last_played
FROM games
JOIN model_scores ON games.id = model_scores.game_id
JOIN last_played ON games.id = last_played.game_id
LEFT OUTER JOIN hltb ON games.id = hltb.game_id
LEFT OUTER JOIN metacritic ON metacritic.game_id = games.id
LEFT OUTER JOIN ratings ON games.id = ratings.game_id
ORDER BY raw_score DESC