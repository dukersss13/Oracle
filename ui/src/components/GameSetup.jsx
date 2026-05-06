import { useEffect, useState } from "react";

export default function GameSetup({ onDone, runTodaySignal }) {
  const [todayGames, setTodayGames] = useState([]);
  const [todayGamesLoading, setTodayGamesLoading] = useState(true);
  const [todayGamesError, setTodayGamesError] = useState("");
  const [model] = useState("TRANSFORMER");
  const [error, setError] = useState("");

  const loadTodayGames = async () => {
    setTodayGamesLoading(true);
    setTodayGamesError("");
    try {
      const res = await fetch("/api/games/today");
      if (!res.ok) {
        setTodayGames([]);
        setTodayGamesError("Today's matchups are temporarily unavailable.");
        return;
      }
      const data = await res.json();
      setTodayGames(Array.isArray(data) ? data : []);
    } catch {
      setTodayGames([]);
      setTodayGamesError("Today's matchups are temporarily unavailable.");
    } finally {
      setTodayGamesLoading(false);
    }
  };

  useEffect(() => {
    loadTodayGames();
  }, []);

  const pullRoster = async (nickname) => {
    const res = await fetch(`/api/roster/${nickname}`);
    if (!res.ok) {
      throw new Error("Failed to fetch roster");
    }
    return await res.json();
  };

  const logoUrl = (teamId) => `https://cdn.nba.com/logos/nba/${teamId}/global/L/logo.svg`;

  const configureLineupForGame = async (game) => {
    setError("");
    try {
      const [homeRoster, awayRoster] = await Promise.all([pullRoster(game.home_team), pullRoster(game.away_team)]);
      onDone({
        homeTeam: game.home_team,
        awayTeam: game.away_team,
        gameDate: game.game_date,
        model,
        holdout,
        homeRoster,
        awayRoster,
      });
    } catch {
      setError("Failed to load roster from API.");
    }
  };

  const runForecastForTodayMatchups = () => {
    setError("");
    if (!todayGames.length) {
      setError("No matchups available to forecast.");
      return;
    }
    onDone({
      runAllToday: true,
      todayGames,
      model,
        holdout: true,
    });
  };

  useEffect(() => {
    if (!runTodaySignal) {
      return;
    }
    if (!todayGamesLoading && todayGames.length > 0) {
      runForecastForTodayMatchups();
    }
  }, [runTodaySignal, todayGamesLoading, todayGames]);

  return (
    <div className="card">
      <div className="today-games-wrap">
        <small>Today's Matchups</small>
        {todayGamesLoading ? (
          <p className="muted">Loading matchups...</p>
        ) : todayGamesError ? (
          <div>
            <p className="error-text">{todayGamesError}</p>
            <button type="button" className="secondary game-run-btn" onClick={loadTodayGames}>
              Retry Matchups
            </button>
          </div>
        ) : todayGames.length > 0 ? (
          <>
            <div className="today-games-grid">
              {todayGames.map((game) => (
                <div key={game.game_id} className="game-card">
                  <div className="game-card-teams">
                    <div className="team-row">
                      <img src={logoUrl(game.away_team_id)} alt={`${game.away_team} logo`} className="team-logo" />
                      <span>{game.away_team}</span>
                    </div>
                    <div className="vs">vs</div>
                    <div className="team-row">
                      <img src={logoUrl(game.home_team_id)} alt={`${game.home_team} logo`} className="team-logo" />
                      <span>{game.home_team}</span>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="secondary game-run-btn"
                    onClick={() => configureLineupForGame(game)}
                  >
                    Adjust Lineup
                  </button>
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="muted">No games found for today.</p>
        )}
      </div>
        <p className="muted">Use the top Forecast button to run all today's matchups.</p>
      {error && <p className="error-text">{error}</p>}
      </div>
  );
}
