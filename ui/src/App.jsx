import { useEffect, useState } from "react";
import ForecastProgress from "./components/ForecastProgress";
import GameSetup from "./components/GameSetup";
import History from "./components/History";
import LineupEditor from "./components/LineupEditor";
import Results from "./components/Results";

const STEPS = { SETUP: 0, LINEUP: 1, RUNNING: 2, RESULTS: 3 };

export default function App() {
  const [step, setStep] = useState(STEPS.SETUP);
  const [tab, setTab] = useState("forecast");
  const [runError, setRunError] = useState("");
  const [history, setHistory] = useState([]);
  const [runTodaySignal, setRunTodaySignal] = useState(0);
  const [teamIdByNickname, setTeamIdByNickname] = useState({});
  const [batchProgress, setBatchProgress] = useState(null);
  const [state, setState] = useState({
    homeTeam: "",
    awayTeam: "",
    gameDate: "",
    model: "TRANSFORMER",
    holdout: true,
    homeRoster: [],
    awayRoster: [],
    lineup: {},
    forecastId: null,
    forecastResult: null,
    batchResults: []
  });

  useEffect(() => {
    const loadTeams = async () => {
      try {
        const res = await fetch("/api/teams");
        if (!res.ok) return;
        const data = await res.json();
        const map = {};
        for (const t of data || []) {
          map[t.nickname] = t.id;
        }
        setTeamIdByNickname(map);
      } catch {
        // Best-effort for logos in results.
      }
    };
    loadTeams();
  }, []);

  const buildDefaultLineup = (homeTeam, awayTeam, homeRoster, awayRoster) => ({
    [homeTeam]: Object.fromEntries(homeRoster.map((p) => [p.player_name, null])),
    [awayTeam]: Object.fromEntries(awayRoster.map((p) => [p.player_name, null]))
  });

  const pullRoster = async (nickname) => {
    const res = await fetch(`/api/roster/${nickname}`);
    if (!res.ok) {
      throw new Error(`Failed to fetch roster for ${nickname}`);
    }
    return await res.json();
  };

  const waitForForecastCompletion = (forecastId) => new Promise((resolve, reject) => {
    const stream = new EventSource(`/api/forecast/${forecastId}/stream`);
    stream.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "completed") {
        stream.close();
        resolve();
      }
      if (msg.type === "error") {
        stream.close();
        reject(new Error(msg.message || "Forecast failed."));
      }
    };
    stream.onerror = () => {
      stream.close();
      reject(new Error("Forecast stream disconnected."));
    };
  });

  const createForecast = async (payload) => {
    const res = await fetch("/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      throw new Error(`Failed to create forecast (${res.status})`);
    }
    const data = await res.json();
    if (!data?.forecast_id) {
      throw new Error("Forecast service did not return an id");
    }
    return data.forecast_id;
  };

  const fetchForecastResult = async (forecastId) => {
    const res = await fetch(`/api/forecast/${forecastId}`);
    if (!res.ok) {
      throw new Error(`Failed to load forecast result (${res.status})`);
    }
    return await res.json();
  };

  const runAllTodayForecasts = async (games, cfg) => {
    setRunError("");
    setStep(STEPS.RUNNING);
    setBatchProgress({ totalGames: games.length, completedGames: 0, currentLabel: "Starting..." });
    setState((prev) => ({ ...prev, forecastId: null, forecastResult: null, batchResults: [] }));

    const results = [];
    for (let i = 0; i < games.length; i += 1) {
      const game = games[i];
      setBatchProgress({
        totalGames: games.length,
        completedGames: i,
        currentLabel: `Running ${game.away_team} @ ${game.home_team}`
      });

      try {
        const [homeRoster, awayRoster] = await Promise.all([
          pullRoster(game.home_team),
          pullRoster(game.away_team)
        ]);

        const lineup = buildDefaultLineup(game.home_team, game.away_team, homeRoster, awayRoster);
        const forecastId = await createForecast({
          home_team: game.home_team,
          away_team: game.away_team,
          game_date: game.game_date,
          model: cfg.model || "TRANSFORMER",
          holdout: cfg.holdout ?? true,
          lineup,
        });

        await waitForForecastCompletion(forecastId);
        const result = await fetchForecastResult(forecastId);

        results.push({
          ...game,
          forecast_id: forecastId,
          status: result?.status || "completed",
          result,
        });
      } catch (err) {
        results.push({
          ...game,
          status: "error",
          error: err?.message || "Forecast failed",
        });
      }

      setBatchProgress({
        totalGames: games.length,
        completedGames: i + 1,
        currentLabel: `Completed ${i + 1} of ${games.length}`
      });
    }

    setBatchProgress(null);
    setState((prev) => ({ ...prev, batchResults: results, forecastId: null, forecastResult: null }));
    setStep(STEPS.RESULTS);
  };

  const setupDone = (cfg) => {
    if (cfg.runAllToday && Array.isArray(cfg.todayGames)) {
      runAllTodayForecasts(cfg.todayGames, cfg);
      return;
    }

    const lineup = buildDefaultLineup(cfg.homeTeam, cfg.awayTeam, cfg.homeRoster, cfg.awayRoster);
    const nextState = { ...state, ...cfg, lineup };
    setState(nextState);
    if (cfg.autoRun) {
      runForecast(nextState);
      return;
    }
    setStep(STEPS.LINEUP);
  };

  const runForecast = async (snapshot = state) => {
    setRunError("");
    setBatchProgress(null);
    setStep(STEPS.RUNNING);
    setState((prev) => ({ ...prev, forecastId: null, forecastResult: null, batchResults: [] }));
    try {
      const forecastId = await createForecast({
        home_team: snapshot.homeTeam,
        away_team: snapshot.awayTeam,
        game_date: snapshot.gameDate,
        model: snapshot.model,
        holdout: snapshot.holdout,
        lineup: snapshot.lineup
      });
      setState((prev) => ({ ...prev, forecastId }));
    } catch (err) {
      setRunError(err?.message || "Unable to start forecast.");
      setStep(STEPS.LINEUP);
    }
  };

  const forecastDone = async (id) => {
    const res = await fetch(`/api/forecast/${id}`);
    const data = await res.json();
    setState((prev) => ({ ...prev, forecastResult: data }));
    setHistory((prev) => [
      {
        id,
        homeTeam: state.homeTeam,
        awayTeam: state.awayTeam,
        gameDate: state.gameDate,
        model: state.model,
        status: data?.status || "completed",
        result: data,
        createdAt: new Date().toISOString(),
      },
      ...prev,
    ]);
    setStep(STEPS.RESULTS);
  };

  const forecastError = (message) => {
    setRunError(message || "Forecast failed.");
    setStep(STEPS.LINEUP);
  };

  const reset = () => {
    setRunError("");
    setBatchProgress(null);
    setState((prev) => ({ ...prev, forecastId: null, forecastResult: null, batchResults: [] }));
    setTab("forecast");
    setStep(STEPS.SETUP);
  };

  const runFromHeader = () => {
    setTab("forecast");
    if (step !== STEPS.SETUP) {
      setStep(STEPS.SETUP);
    }
    setRunTodaySignal((prev) => prev + 1);
  };

  const runningPct = batchProgress && batchProgress.totalGames > 0
    ? Math.round((batchProgress.completedGames / batchProgress.totalGames) * 100)
    : 0;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <p className="eyebrow">Game Night Studio</p>
          <h1 className="logo">Oracle Forecast</h1>
        </div>
        <div className="tab-row">
          <button className={tab === "forecast" ? "tab active" : "tab"} onClick={runFromHeader}>Forecast</button>
          <button className={tab === "history" ? "tab active" : "tab"} onClick={() => setTab("history")}>History</button>
          <button className="tab" onClick={reset}>Main Menu</button>
        </div>
      </header>

      {tab === "history" && <History items={history} />}
      {tab !== "history" && step === STEPS.SETUP && <GameSetup onDone={setupDone} runTodaySignal={runTodaySignal} />}
      {tab !== "history" && step === STEPS.LINEUP && (
        <>
          {runError && <p className="error-text">{runError}</p>}
          <LineupEditor
            homeTeam={state.homeTeam}
            awayTeam={state.awayTeam}
            homeRoster={state.homeRoster}
            awayRoster={state.awayRoster}
            lineup={state.lineup}
            setLineup={(lineupFn) => setState((prev) => ({ ...prev, lineup: lineupFn(prev.lineup) }))}
            onBack={() => setStep(STEPS.SETUP)}
            onRun={() => runForecast()}
          />
        </>
      )}
      {tab !== "history" && step === STEPS.RUNNING && state.forecastId && (
        <ForecastProgress
          forecastId={state.forecastId}
          totalPlayers={state.model === "TRANSFORMER" ? 2 : state.homeRoster.length + state.awayRoster.length}
          onDone={forecastDone}
          onError={forecastError}
        />
      )}
      {tab !== "history" && step === STEPS.RUNNING && !state.forecastId && batchProgress && (
        <div className="card">
          <h3>Running Forecasts</h3>
          <div className="progress-wrap">
            <div className="progress-meta">
              <span>Games completed: {batchProgress.completedGames}/{batchProgress.totalGames}</span>
              <span>{runningPct}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${runningPct}%` }} />
            </div>
          </div>
          <p className="muted">{batchProgress.currentLabel}</p>
        </div>
      )}
      {tab !== "history" && step === STEPS.RESULTS && (state.forecastResult || state.batchResults.length > 0) && (
        <Results
          data={state.forecastResult}
          batchResults={state.batchResults}
          homeTeam={state.homeTeam}
          awayTeam={state.awayTeam}
          homeTeamId={teamIdByNickname[state.homeTeam]}
          awayTeamId={teamIdByNickname[state.awayTeam]}
          onReset={reset}
        />
      )}
    </div>
  );
}
