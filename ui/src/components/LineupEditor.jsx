import { useEffect, useState } from "react";

export default function LineupEditor({
  homeTeam,
  awayTeam,
  homeRoster,
  awayRoster,
  lineup,
  setLineup,
  onBack,
  onRun
}) {
  const [injuries, setInjuries] = useState([]);
  const [injuriesLoading, setInjuriesLoading] = useState(true);

  useEffect(() => {
    const fetchInjuries = async () => {
      setInjuriesLoading(true);
      try {
        const res = await fetch("/api/injuries");
        if (res.ok) {
          const data = await res.json();
          setInjuries(Array.isArray(data) ? data : []);

          // Auto-mark injured players as DNP (minutes = 0)
          if (Array.isArray(data) && data.length > 0) {
            const injuredNames = new Set(data.map((e) => e.name));
            setLineup((prev) => {
              const next = { ...prev };
              for (const team of [homeTeam, awayTeam]) {
                if (!next[team]) continue;
                next[team] = { ...next[team] };
                for (const playerName of Object.keys(next[team])) {
                  if (injuredNames.has(playerName)) {
                    next[team][playerName] = 0;
                  }
                }
              }
              return next;
            });
          }
        }
      } catch {
        // Injuries are best-effort; don't block the UI
      } finally {
        setInjuriesLoading(false);
      }
    };
    fetchInjuries();
  }, [homeTeam, awayTeam, setLineup]);

  const changeMins = (team, player, value) => {
    const parsed = value === "" ? null : Number(value);
    setLineup((prev) => ({
      ...prev,
      [team]: {
        ...prev[team],
        [player]: Number.isNaN(parsed) ? null : parsed
      }
    }));
  };

  const injuredNames = new Set(injuries.map((e) => e.name));

  const rosterCard = (team, roster) => (
    <div className="card tight-card">
      <h3>{team} Lineup</h3>
      {roster.map((p) => {
        const isInjured = injuredNames.has(p.player_name);
        const injury = isInjured ? injuries.find((e) => e.name === p.player_name) : null;
        return (
          <div key={p.player_id} className="lineup-row">
            <small>
              {p.player_name}
              {isInjured && (
                <span className="injury-tag" title={injury?.description || "Out"}>
                  {" "}DNP
                </span>
              )}
            </small>
            <input
              value={lineup[team]?.[p.player_name] ?? ""}
              placeholder={isInjured ? "0 (injured)" : "auto"}
              onChange={(e) => changeMins(team, p.player_name, e.target.value)}
            />
          </div>
        );
      })}
    </div>
  );

  return (
    <div>
      {injuriesLoading && <p className="muted">Loading injury report...</p>}
      {!injuriesLoading && injuries.length > 0 && (
        <p className="muted" style={{ marginBottom: "0.5rem" }}>
          {injuries.length} player(s) ruled out — auto-marked as DNP. Override minutes to include them.
        </p>
      )}
      <div className="grid">
        {rosterCard(homeTeam, homeRoster)}
        {rosterCard(awayTeam, awayRoster)}
      </div>
      <div className="button-row">
        <button className="secondary action-btn" onClick={onBack}>Back to Game</button>
        <button className="cta action-btn" onClick={() => onRun()}>Run Forecast</button>
      </div>
    </div>
  );
}
