const formatTime = (iso) => {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
};

export default function History({ items = [] }) {
  return (
    <div className="card">
      <h2>History</h2>
      {items.length === 0 ? (
        <p className="muted">No forecasts yet in this session.</p>
      ) : (
        <div>
          {items.map((entry) => {
            const homeTotal = entry?.result?.home_forecast?.total_forecasted ?? "-";
            const awayTotal = entry?.result?.away_forecast?.total_forecasted ?? "-";
            return (
              <div className="card tight-card" key={entry.id}>
                <h3>{entry.awayTeam} @ {entry.homeTeam}</h3>
                <p><strong>Date:</strong> {entry.gameDate}</p>
                <p><strong>Model:</strong> {entry.model}</p>
                <p><strong>Status:</strong> {entry.status}</p>
                <p><strong>Forecast totals:</strong> {awayTotal} - {homeTotal}</p>
                <p><strong>Created:</strong> {formatTime(entry.createdAt)}</p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
