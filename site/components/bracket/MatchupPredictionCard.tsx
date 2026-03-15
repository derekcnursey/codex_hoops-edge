import { CSSProperties } from "react";
import { displayTeam } from "../../lib/data";
import { GameComparison } from "../../lib/bracket/comparison";
import { BracketTeam, GradedGameResult, MatchupPrediction } from "../../lib/bracket/types";

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

function pct(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function metricValue(value: number | null | undefined, digits = 1): string {
  return value == null ? "--" : value.toFixed(digits);
}

function metricGrid(team: BracketTeam) {
  return [
    { label: "Adj Pace", value: metricValue(team.adjTempo) },
    { label: "Adj OE", value: metricValue(team.adjOe) },
    { label: "Adj DE", value: metricValue(team.adjDe) },
    { label: "Adj Net", value: metricValue(team.adjNet) },
  ];
}

function statusPill(label: string, tone: "blue" | "green" | "amber" | "red" | "slate") {
  const toneMap = {
    blue: { background: "#dbeafe", color: "#1d4ed8" },
    green: { background: "#dcfce7", color: "#166534" },
    amber: { background: "#fffbeb", color: "#b45309" },
    red: { background: "#fef2f2", color: "#b91c1c" },
    slate: { background: "#e2e8f0", color: "#334155" },
  }[tone];

  return (
    <span
      style={{
        ...mono,
        fontSize: 10,
        padding: "2px 6px",
        borderRadius: 999,
        background: toneMap.background,
        color: toneMap.color,
      }}
    >
      {label}
    </span>
  );
}

function teamTone(isFavorite: boolean, isSelected: boolean, isActualWinner: boolean) {
  if (isSelected) return { border: "#0f172a", background: "#f8fafc" };
  if (isActualWinner) return { border: "#86efac", background: "#f0fdf4" };
  if (isFavorite) return { border: "#93c5fd", background: "#eff6ff" };
  return { border: "#e2e8f0", background: "#ffffff" };
}

export default function MatchupPredictionCard({
  prediction,
  loading,
  error,
  selectedWinnerId,
  comparison,
  grading,
  teamA,
  teamB,
}: {
  prediction?: MatchupPrediction;
  loading?: boolean;
  error?: string;
  selectedWinnerId?: number;
  comparison?: GameComparison;
  grading?: GradedGameResult;
  teamA?: BracketTeam | null;
  teamB?: BracketTeam | null;
}) {
  const teams = [
    prediction && teamA
      ? {
          key: "A",
          team: teamA,
          winProb: prediction.winProbA,
          isFavorite: prediction.favoredTeamId === teamA.id,
          isSelected: selectedWinnerId === teamA.id,
          isActualWinner: grading?.actualWinnerId === teamA.id,
        }
      : null,
    prediction && teamB
      ? {
          key: "B",
          team: teamB,
          winProb: prediction.winProbB,
          isFavorite: prediction.favoredTeamId === teamB.id,
          isSelected: selectedWinnerId === teamB.id,
          isActualWinner: grading?.actualWinnerId === teamB.id,
        }
      : null,
  ].filter(Boolean) as {
    key: string;
    team: BracketTeam;
    winProb: number;
    isFavorite: boolean;
    isSelected: boolean;
    isActualWinner: boolean;
  }[];

  return (
    <div
      style={{
        marginTop: 8,
        padding: "10px 12px",
        borderRadius: 10,
        border: "1px solid #dbe4ef",
        background: "#f8fafc",
      }}
    >
      <div
        style={{
          ...mono,
          display: "flex",
          justifyContent: "space-between",
          gap: 8,
          fontSize: 11,
          color: "#64748b",
          textTransform: "uppercase",
          letterSpacing: "0.04em",
          marginBottom: 8,
        }}
      >
        <span>Game Details</span>
        {loading ? <span>Loading...</span> : null}
        {error ? <span style={{ color: "#b91c1c" }}>{error}</span> : null}
      </div>

      {prediction ? (
        <>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 6,
              marginBottom: 8,
            }}
          >
            {statusPill(`Model: ${displayTeam(prediction.modelWinnerName)}`, "blue")}
            {statusPill(`${displayTeam(prediction.favoredTeamName)} -${prediction.projectedSpread.toFixed(1)}`, "slate")}
            {comparison?.selectedWinnerName
              ? statusPill(comparison.agreesWithModel ? "Agree" : "Fade", comparison.agreesWithModel ? "green" : "amber")
              : null}
            {comparison?.confidenceLabel ? statusPill(comparison.confidenceLabel, "slate") : null}
            {grading?.isFinal
              ? statusPill(
                  grading.status === "correct" ? "Correct" : grading.status === "incorrect" ? "Missed" : "Actual final",
                  grading.status === "correct" ? "green" : grading.status === "incorrect" ? "red" : "slate",
                )
              : null}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
              gap: 8,
            }}
          >
            {teams.map(({ key, team, winProb, isFavorite, isSelected, isActualWinner }) => {
              const tone = teamTone(isFavorite, isSelected, isActualWinner);
              return (
                <div
                  key={key}
                  style={{
                    borderRadius: 8,
                    border: `1px solid ${tone.border}`,
                    background: tone.background,
                    padding: "8px 9px",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 6, marginBottom: 6 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "#0f172a" }}>
                      ({team.seed}) {displayTeam(team.name)}
                    </div>
                    <div style={{ display: "flex", gap: 4, flexWrap: "wrap", justifyContent: "flex-end" }}>
                      {isFavorite ? statusPill("Fav", "blue") : null}
                      {isSelected ? statusPill("Your pick", "slate") : null}
                      {isActualWinner ? statusPill("Actual", "green") : null}
                    </div>
                  </div>

                  <div style={{ ...mono, fontSize: 11, color: "#334155", marginBottom: 6 }}>
                    Win prob {pct(winProb, 1)}
                  </div>
                  <div style={{ ...mono, fontSize: 11, color: "#475569", marginBottom: 6 }}>
                    Rank {team.rank} | {team.conference || "--"} | {team.record}
                  </div>

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                      gap: 5,
                    }}
                  >
                    {metricGrid(team).map((metric) => (
                      <div
                        key={metric.label}
                        style={{
                          borderRadius: 6,
                          background: "#ffffff",
                          border: "1px solid #e2e8f0",
                          padding: "5px 6px",
                        }}
                      >
                        <div style={{ ...mono, fontSize: 10, color: "#64748b", marginBottom: 2 }}>{metric.label}</div>
                        <div style={{ ...mono, fontSize: 11, color: "#0f172a" }}>{metric.value}</div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          <div
            style={{
              ...mono,
              fontSize: 11,
              color: "#475569",
              marginTop: 8,
              display: "flex",
              flexWrap: "wrap",
              gap: 10,
            }}
          >
            <span>Favorite {displayTeam(prediction.favoredTeamName)}</span>
            <span>Underdog {displayTeam(prediction.underdogTeamName)}</span>
            {prediction.projectedScoreA != null && prediction.projectedScoreB != null ? (
              <span>
                Score {displayTeam(prediction.teamAName)} {prediction.projectedScoreA.toFixed(0)} - {prediction.projectedScoreB.toFixed(0)} {displayTeam(prediction.teamBName)}
              </span>
            ) : null}
            {grading?.actualWinnerName ? <span>Actual winner {displayTeam(grading.actualWinnerName)}</span> : null}
          </div>
        </>
      ) : null}

      {!prediction && !loading && !error ? (
        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
          Prediction loads when both teams are known.
        </div>
      ) : null}
    </div>
  );
}
