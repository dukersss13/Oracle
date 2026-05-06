export default function Results({
  data,
  batchResults = [],
  homeTeam,
  awayTeam,
  homeTeamId,
  awayTeamId,
  onReset,
}) {
  const logoUrl = (teamId) => `https://cdn.nba.com/logos/nba/${teamId}/global/L/logo.svg`;

  const singleGame = data ? {
    game_id: "single",
    away_team: awayTeam,
    home_team: homeTeam,
    away_team_id: awayTeamId,
    home_team_id: homeTeamId,
    status: data?.status || "completed",
    result: data,
  } : null;

  const games = batchResults.length > 0 ? batchResults : (singleGame ? [singleGame] : []);

  return (
    <div>
      <div className="scoreboard-grid">
        {games.map((g) => {
          const awayScore = g?.result?.away_forecast?.total_forecasted;
          const homeScore = g?.result?.home_forecast?.total_forecasted;
          const hasScore = Number.isFinite(awayScore) && Number.isFinite(homeScore);

          return (
            <div key={g.game_id || `${g.away_team}-${g.home_team}`} className="scoreboard-card">
              <div className="scoreboard-teams">
                <div className="score-team-row">
                  {g.away_team_id ? (
                    <img src={logoUrl(g.away_team_id)} alt={`${g.away_team} logo`} className="team-logo" />
                  ) : (
                    <div className="team-logo-placeholder" />
                  )}
                  <span>{g.away_team}</span>
                  <strong>{hasScore ? awayScore : "-"}</strong>
                </div>
                <div className="score-vs">@</div>
                <div className="score-team-row">
                  {g.home_team_id ? (
                    <img src={logoUrl(g.home_team_id)} alt={`${g.home_team} logo`} className="team-logo" />
                  ) : (
                    <div className="team-logo-placeholder" />
                  )}
                  <span>{g.home_team}</span>
                  <strong>{hasScore ? homeScore : "-"}</strong>
                </div>
              </div>
              {g.status === "error" && <p className="error-text">{g.error || "Forecast failed"}</p>}
              {hasScore && (
                <p className="muted score-subtext">
                  Forecasted score: {g.away_team} {awayScore} - {homeScore} {g.home_team}
                </p>
              )}
            </div>
          );
        })}
      </div>
      <button className="cta" onClick={onReset}>Run Another Forecast</button>
    </div>
  );
}
