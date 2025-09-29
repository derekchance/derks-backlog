UPDATE model_scores
SET adj_score = series_adj.adj_score
FROM (SELECT
    games.id AS game_id,
    CASE
        WHEN series.sub_id = series2.min_series_sub_id
        AND model_scores.raw_score < series2.max_series_raw_score
        THEN 0.8 * series2.max_series_raw_score + 0.2 * model_scores.raw_score
        WHEN series.sub_id > series2.min_series_sub_id
        AND model_scores2.raw_score < model_scores.raw_score
        THEN 0.8 * model_scores2.raw_score + 0.2 * model_scores.raw_score
        ELSE model_scores.raw_score
    END AS adj_score
FROM games
JOIN model_scores ON model_scores.game_id = games.id
LEFT OUTER JOIN series ON series.Title = games.Title
LEFT OUTER JOIN (
    SELECT
    series.id,
    MIN(series.sub_id) AS min_series_sub_id,
    MAX(model_scores.raw_score) as max_series_raw_score
    FROM games
        JOIN model_scores ON model_scores.game_id = games.id
        JOIN series ON series.Title = games.Title
    WHERE games.id NOT IN (SELECT game_id FROM last_played)
    GROUP BY series.id
    HAVING COUNT(series.Title) > 1) series2
        ON series.id = series2.id
LEFT OUTER JOIN series series3 ON series3.sub_id = series2.min_series_sub_id
    AND series2.id = series3.id
LEFT OUTER JOIN model_scores model_scores2 ON model_scores2.game_id = (SELECT id FROM games WHERE Title = series3.Title)) series_adj
WHERE model_scores.game_id = series_adj.game_id