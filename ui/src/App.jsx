import { useState } from "react";
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
  const [state, setState] = useState({
    homeTeam: "",
    awayTeam: "",
    gameDate: "",
    model: "NN",
    holdout: true,
    homeRoster: [],
    awayRoster: [],
    lineup: {},
    forecastId: null,
    forecastResult: null
  });

  const setupDone = (cfg) => {
    const lineup = {
      [cfg.homeTeam]: Object.fromEntries(cfg.homeRoster.map((p) => [p.player_name, null])),
      [cfg.awayTeam]: Object.fromEntries(cfg.awayRoster.map((p) => [p.player_name, null]))
    };
    const nextState = { ...state, ...cfg, lineup };
    setState(nextState);
    setStep(STEPS.LINEUP);
  };

  const runForecast = async (snapshot = state) => {
    setRunError("");
    setStep(STEPS.RUNNING);
    setState((prev) => ({ ...prev, forecastId: null, forecastResult: null }));
    try {
      const res = await fetch("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          home_team: snapshot.homeTeam,
          away_team: snapshot.awayTeam,
          game_date: snapshot.gameDate,
          model: snapshot.model,
          holdout: snapshot.holdout,
          lineup: snapshot.lineup
        })
      });
      if (!res.ok) {
        throw new Error(`Failed to create forecast (${res.status})`);
      }
      const data = await res.json();
      if (!data?.forecast_id) {
        throw new Error("Forecast service did not return an id");
      }
      setState((prev) => ({ ...prev, forecastId: data.forecast_id }));
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
    setState((prev) => ({ ...prev, forecastId: null, forecastResult: null }));
    setTab("forecast");
    setStep(STEPS.SETUP);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <p className="eyebrow">Game Night Studio</p>
          <h1 className="logo">Oracle Forecast</h1>
        </div>
        <div className="tab-row">
          <button className={tab === "forecast" ? "tab active" : "tab"} onClick={() => setTab("forecast")}>Forecast</button>
          <button className={tab === "history" ? "tab active" : "tab"} onClick={() => setTab("history")}>History</button>
          <button className="tab" onClick={reset}>Main Menu</button>
        </div>
      </header>

      {tab === "history" && <History items={history} />}
      {tab !== "history" && step === STEPS.SETUP && <GameSetup onDone={setupDone} />}
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
          totalPlayers={state.homeRoster.length + state.awayRoster.length}
          onDone={forecastDone}
          onError={forecastError}
        />
      )}
      {tab !== "history" && step === STEPS.RESULTS && state.forecastResult && (
        <Results data={state.forecastResult} onReset={reset} />
      )}
    </div>
  );
}
