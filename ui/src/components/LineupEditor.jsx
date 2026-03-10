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

  const rosterCard = (team, roster) => (
    <div className="card tight-card">
      <h3>{team} Lineup</h3>
      {roster.map((p) => (
        <div key={p.player_id} className="lineup-row">
          <small>{p.player_name}</small>
          <input
            value={lineup[team]?.[p.player_name] ?? ""}
            placeholder="auto"
            onChange={(e) => changeMins(team, p.player_name, e.target.value)}
          />
        </div>
      ))}
    </div>
  );

  return (
    <div>
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
