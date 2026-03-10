export default function Results({ data, onReset }) {
  const renderTeam = (team) => {
    if (!team) return null;
    return (
      <div className="card tight-card">
        <h3>{team.team_name}</h3>
        <table className="table">
          <thead>
            <tr>
              <th>Player</th>
              <th>Forecast</th>
              <th>Actual</th>
            </tr>
          </thead>
          <tbody>
            {team.players.map((p) => (
              <tr key={p.player_name}>
                <td>{p.player_name}</td>
                <td>{p.forecasted_points}</td>
                <td>{p.actual_points}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p><strong>Total:</strong> {team.total_forecasted} forecasted, {team.total_actual} actual</p>
      </div>
    );
  };

  return (
    <div>
      <div className="grid">
        {renderTeam(data.home_forecast)}
        {renderTeam(data.away_forecast)}
      </div>
      <button className="cta" onClick={onReset}>Run Another Forecast</button>
    </div>
  );
}
