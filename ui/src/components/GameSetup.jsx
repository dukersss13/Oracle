import { useEffect, useState } from "react";

export default function GameSetup({ onDone }) {
  const [teams, setTeams] = useState([]);
  const [todayGames, setTodayGames] = useState([]);
  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");
  const [dateMode, setDateMode] = useState("today");
  const [gameDate, setGameDate] = useState("");
  const [model, setModel] = useState("XGBOOST");
  const [holdout, setHoldout] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([fetch("/api/teams"), fetch("/api/games/today")])
      .then(async ([teamsRes, gamesRes]) => {
        const [teamsData, gamesData] = await Promise.all([teamsRes.json(), gamesRes.json()]);
        const sortedTeams = [...teamsData].sort((a, b) => a.nickname.localeCompare(b.nickname));
        setTeams(sortedTeams);
        setTodayGames(Array.isArray(gamesData) ? gamesData : []);
      })
      .catch(() => setError("Could not load teams/games."));
  }, []);

  const pullRoster = async (nickname) => {
    const res = await fetch(`/api/roster/${nickname}`);
    if (!res.ok) {
      throw new Error("Failed to fetch roster");
    }
    return await res.json();
  };

  const logoUrl = (teamId) => `https://cdn.nba.com/logos/nba/${teamId}/global/L/logo.svg`;

  const formatAsMmDdYyyy = (dateObj) => {
    const mm = String(dateObj.getMonth() + 1).padStart(2, "0");
    const dd = String(dateObj.getDate()).padStart(2, "0");
    const yyyy = String(dateObj.getFullYear());
    return `${mm}-${dd}-${yyyy}`;
  };

  const normalizeGameDate = (rawValue) => {
    const value = rawValue.trim();
    if (!value) return "";

    if (/^\d{2}-\d{2}-\d{4}$/.test(value)) {
      return value;
    }

    if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
      const [y, m, d] = value.split("-");
      return `${m}-${d}-${y}`;
    }

    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return "";
    }
    return formatAsMmDdYyyy(parsed);
  };

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    if (!homeTeam || !awayTeam) {
      setError("Select both teams and a game date mode.");
      return;
    }
    if (homeTeam === awayTeam) {
      setError("Home and away teams must be different.");
      return;
    }

    const normalizedDate =
      dateMode === "today" ? formatAsMmDdYyyy(new Date()) : normalizeGameDate(gameDate);
    if (!normalizedDate) {
      setError("Use a valid custom date.");
      return;
    }

    try {
      const [homeRoster, awayRoster] = await Promise.all([pullRoster(homeTeam), pullRoster(awayTeam)]);
      onDone({ homeTeam, awayTeam, gameDate: normalizedDate, model, holdout, homeRoster, awayRoster });
    } catch {
      setError("Failed to load roster from API.");
    }
  };

  const runForecastForGame = async (game) => {
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
        autoRun: true,
      });
    } catch {
      setError("Failed to load roster from API.");
    }
  };

  return (
    <form className="card" onSubmit={submit}>
      <div className="today-games-wrap">
        <small>Today's Matchups</small>
        {todayGames.length > 0 ? (
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
                  onClick={() => runForecastForGame(game)}
                >
                  Run Forecast
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted">No matchups returned right now.</p>
        )}
      </div>
      <div className="grid">
        <div className="field">
          <small>Home Team</small>
          <select value={homeTeam} onChange={(e) => setHomeTeam(e.target.value)}>
            <option value="">Select</option>
            {teams.map((t) => (
              <option key={t.id} value={t.nickname}>{t.nickname}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <small>Away Team</small>
          <select value={awayTeam} onChange={(e) => setAwayTeam(e.target.value)}>
            <option value="">Select</option>
            {teams.map((t) => (
              <option key={t.id} value={t.nickname}>{t.nickname}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <small>Date Mode</small>
          <select value={dateMode} onChange={(e) => setDateMode(e.target.value)}>
            <option value="today">Today</option>
            <option value="custom">Custom Date</option>
          </select>
        </div>
        {dateMode === "custom" && (
          <div className="field">
            <small>Custom Game Date</small>
            <input
              value={gameDate}
              onChange={(e) => setGameDate(e.target.value)}
              placeholder="MM-DD-YYYY or YYYY-MM-DD"
            />
          </div>
        )}
        <div className="field">
          <small>Model</small>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="XGBOOST">XGBoost</option>
            <option value="NN">Neural Network</option>
          </select>
        </div>
      </div>
      <label className="check-row">
        <input type="checkbox" checked={holdout} onChange={(e) => setHoldout(e.target.checked)} />
        Holdout mode
      </label>
      {error && <p className="error-text">{error}</p>}
      <button className="cta form-cta" type="submit">Continue to Lineup</button>
    </form>
  );
}
