import { useEffect, useState } from "react";

export default function ForecastProgress({ forecastId, totalPlayers, onDone, onError }) {
  const [progressPct, setProgressPct] = useState(0);
  const [completedPlayers, setCompletedPlayers] = useState(0);

  useEffect(() => {
    const stream = new EventSource(`/api/forecast/${forecastId}/stream`);
    stream.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "step") {
        if (msg.progress != null) {
          setProgressPct((prev) => Math.max(prev, msg.progress));
          if (totalPlayers > 0) {
            const done = Math.min(totalPlayers, Math.floor((Number(msg.progress) / 100) * totalPlayers));
            setCompletedPlayers((prev) => Math.max(prev, done));
          }
        }
      }
      if (msg.type === "player_complete") {
        setCompletedPlayers((prev) => {
          const next = prev + 1;
          if (totalPlayers > 0) {
            setProgressPct(Math.min(100, Math.round((next / totalPlayers) * 100)));
          }
          return next;
        });
      }
      if (msg.type === "completed") {
        setProgressPct(100);
        if (totalPlayers > 0) {
          setCompletedPlayers(totalPlayers);
        }
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
          <span>
            {totalPlayers > 0
              ? `Players completed: ${Math.min(completedPlayers, totalPlayers)}/${totalPlayers}`
              : "Running forecast"}
          </span>
          <span>{progressPct}%</span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progressPct}%` }} />
        </div>
      </div>
    </div>
  );
}
