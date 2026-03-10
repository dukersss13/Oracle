import { useEffect, useState } from "react";

export default function ForecastProgress({ forecastId, totalPlayers, onDone, onError }) {
  const [events, setEvents] = useState([]);

  const completedPlayers = new Set(
    events
      .filter((e) => e.type === "player_complete" && e.player)
      .map((e) => e.player)
  ).size;
  const progressPct = totalPlayers > 0 ? Math.min(100, Math.round((completedPlayers / totalPlayers) * 100)) : 0;

  useEffect(() => {
    const stream = new EventSource(`/api/forecast/${forecastId}/stream`);
    stream.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      setEvents((prev) => [...prev, msg]);
      if (msg.type === "completed") {
        stream.close();
        onDone(forecastId);
      }
      if (msg.type === "error") {
        stream.close();
        if (onError) {
          onError(msg.message || "Forecast failed.");
        }
      }
    };
    stream.onerror = () => {
      stream.close();
      if (onError) {
        onError("Forecast stream disconnected.");
      }
    };
    return () => stream.close();
  }, [forecastId, onDone]);

  return (
    <div className="card">
      <h3>Running Forecast</h3>
      <div className="progress-wrap">
        <div className="progress-meta">
          <span>Players completed: {completedPlayers}/{totalPlayers || "?"}</span>
          <span>{progressPct}%</span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progressPct}%` }} />
        </div>
      </div>
      {events.length === 0 && <p className="muted">Starting model run...</p>}
      {events.map((e, i) => (
        <p className="event-item" key={`${e.type}-${i}`}>
          {e.type === "player_complete" ? `Completed: ${e.player}` : e.type}
        </p>
      ))}
    </div>
  );
}
