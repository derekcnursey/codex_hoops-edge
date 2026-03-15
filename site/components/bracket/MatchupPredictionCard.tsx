import { CSSProperties } from "react";
import { displayTeam } from "../../lib/data";
import { MatchupPrediction } from "../../lib/bracket/types";
import { GameComparison } from "../../lib/bracket/comparison";

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function spreadLabel(prediction: MatchupPrediction): string {
  return `${displayTeam(prediction.favoredTeamName)} -${prediction.projectedSpread.toFixed(1)}`;
}

export default function MatchupPredictionCard({
  prediction,
  loading,
  error,
  selectedWinnerId,
  comparison,
}: {
  prediction?: MatchupPrediction;
  loading?: boolean;
  error?: string;
  selectedWinnerId?: number;
  comparison?: GameComparison;
}) {
  const comparisonTone =
    comparison?.agreesWithModel === false
      ? { border: "#f59e0b", background: "#fffbeb", color: "#b45309", label: "Fading model" }
      : comparison?.agreesWithModel === true
        ? { border: "#86efac", background: "#f0fdf4", color: "#15803d", label: "Aligned with model" }
        : null;

  return (
    <div
      style={{
        marginTop: 10,
        padding: "10px 12px",
        borderRadius: 8,
        border: "1px solid #e2e8f0",
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
        <span>Model Matchup</span>
        {loading ? <span>Loading...</span> : null}
        {error ? <span style={{ color: "#b91c1c" }}>{error}</span> : null}
      </div>

      {prediction ? (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr auto",
              gap: 8,
              alignItems: "center",
              marginBottom: 8,
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, color: "#0f172a" }}>
              Favorite: {displayTeam(prediction.favoredTeamName)}
            </div>
            <div style={{ ...mono, fontSize: 12, color: "#0f172a" }}>
              Spread: {spreadLabel(prediction)}
            </div>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 8,
              marginBottom: 8,
            }}
          >
            <div
              style={{
                borderRadius: 6,
                border: `1px solid ${prediction.favoredTeamId === prediction.teamAId ? "#93c5fd" : "#dbeafe"}`,
                background: prediction.favoredTeamId === prediction.teamAId ? "#dbeafe" : "#eff6ff",
                padding: "8px 10px",
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 600, color: "#0f172a", display: "flex", gap: 6, alignItems: "center" }}>
                {displayTeam(prediction.teamAName)}
                {prediction.favoredTeamId === prediction.teamAId ? (
                  <span style={{ ...mono, fontSize: 10, color: "#1d4ed8" }}>FAV</span>
                ) : null}
                {selectedWinnerId === prediction.teamAId ? (
                  <span style={{ ...mono, fontSize: 10, color: "#0f172a" }}>YOUR PICK</span>
                ) : null}
              </div>
              <div style={{ ...mono, fontSize: 12, color: "#334155" }}>
                Win %: {pct(prediction.winProbA)}
              </div>
            </div>
            <div
              style={{
                borderRadius: 6,
                border: `1px solid ${prediction.favoredTeamId === prediction.teamBId ? "#93c5fd" : "#dbeafe"}`,
                background: prediction.favoredTeamId === prediction.teamBId ? "#dbeafe" : "#eff6ff",
                padding: "8px 10px",
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 600, color: "#0f172a", display: "flex", gap: 6, alignItems: "center" }}>
                {displayTeam(prediction.teamBName)}
                {prediction.favoredTeamId === prediction.teamBId ? (
                  <span style={{ ...mono, fontSize: 10, color: "#1d4ed8" }}>FAV</span>
                ) : null}
                {selectedWinnerId === prediction.teamBId ? (
                  <span style={{ ...mono, fontSize: 10, color: "#0f172a" }}>YOUR PICK</span>
                ) : null}
              </div>
              <div style={{ ...mono, fontSize: 12, color: "#334155" }}>
                Win %: {pct(prediction.winProbB)}
              </div>
            </div>
          </div>

          <div style={{ ...mono, fontSize: 11, color: "#475569" }}>
            Model winner: {displayTeam(prediction.modelWinnerName)}
          </div>
          {comparison?.selectedWinnerName ? (
            <div
              style={{
                marginTop: 8,
                padding: "8px 10px",
                borderRadius: 6,
                border: `1px solid ${comparisonTone?.border ?? "#e2e8f0"}`,
                background: comparisonTone?.background ?? "#ffffff",
                color: comparisonTone?.color ?? "#334155",
              }}
            >
              <div
                style={{
                  ...mono,
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 8,
                  fontSize: 11,
                  lineHeight: 1.5,
                }}
              >
                <span>Your pick: {displayTeam(comparison.selectedWinnerName)}</span>
                <span>Model: {displayTeam(comparison.modelWinnerName ?? "--")}</span>
                <span>{comparisonTone?.label ?? "Awaiting model compare"}</span>
                <span>{comparison.confidenceLabel}</span>
              </div>
            </div>
          ) : null}
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
