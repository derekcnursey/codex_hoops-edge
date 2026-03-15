import { CSSProperties } from "react";
import { displayTeam } from "../../lib/data";
import { BracketTeam, BracketSource, GradedGameResult, MatchupPrediction, ResolvedBracketGame } from "../../lib/bracket/types";
import { GameComparison, MAJOR_UPSET_SEED_GAP } from "../../lib/bracket/comparison";
import MatchupPredictionCard from "./MatchupPredictionCard";

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

function formatMetric(label: string, value: number | null | undefined, digits = 1): string {
  if (value == null) return `${label}: --`;
  return `${label}: ${value.toFixed(digits)}`;
}

function teamMetricSummary(team: BracketTeam) {
  return [
    formatMetric("Adj Net", team.adjNet),
    formatMetric("Adj OE", team.adjOe),
    formatMetric("Adj DE", team.adjDe),
    formatMetric("Adj Pace", team.adjTempo),
  ].join(" | ");
}

function TeamRow({
  team,
  source,
  isSelected,
  isClickable,
  isFavorite,
  isUpset,
  isMajorUpset,
  isFadingModel,
  isActualWinner,
  isCorrectPick,
  isMissedPick,
  onSelect,
}: {
  team: BracketTeam | null;
  source: BracketSource;
  isSelected: boolean;
  isClickable: boolean;
  isFavorite: boolean;
  isUpset: boolean;
  isMajorUpset: boolean;
  isFadingModel: boolean;
  isActualWinner: boolean;
  isCorrectPick: boolean;
  isMissedPick: boolean;
  onSelect: () => void;
}) {
  if (!team) {
    return (
      <div
        style={{
          padding: "10px 12px",
          borderRadius: 8,
          border: "1px dashed #cbd5e1",
          background: "#f8fafc",
          color: "#94a3b8",
        }}
      >
        <div style={{ ...mono, fontSize: 11 }}>Awaiting: {source.label}</div>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={!isClickable}
      style={{
        width: "100%",
        textAlign: "left",
        padding: "10px 12px",
        borderRadius: 8,
        border: `1px solid ${
          isSelected
            ? isCorrectPick
              ? "#16a34a"
              : isMissedPick
                ? "#dc2626"
                : isFadingModel
                  ? "#f59e0b"
                  : "#0f172a"
            : isActualWinner
              ? "#86efac"
              : isFavorite
                ? "#93c5fd"
                : "#e2e8f0"
        }`,
        background: isSelected ? "#0f172a" : isActualWinner ? "#f0fdf4" : "#ffffff",
        color: isSelected ? "#ffffff" : "#0f172a",
        cursor: isClickable ? "pointer" : "default",
        transition: "all 0.15s",
        boxShadow:
          isSelected && isCorrectPick
            ? "0 0 0 1px rgba(22,163,74,0.28)"
            : isSelected && isMissedPick
              ? "0 0 0 1px rgba(220,38,38,0.28)"
              : isSelected && isFadingModel
                ? "0 0 0 1px rgba(245,158,11,0.28)"
                : "none",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 8,
          marginBottom: 5,
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 700 }}>
          ({team.seed}) {displayTeam(team.name)}
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {isFavorite ? (
            <span
              style={{
                ...mono,
                fontSize: 10,
                padding: "2px 6px",
                borderRadius: 999,
                background: isSelected ? "rgba(255,255,255,0.15)" : "#dbeafe",
                color: isSelected ? "#e2e8f0" : "#1d4ed8",
              }}
            >
              FAVORITE
            </span>
          ) : null}
          {isSelected ? (
            <span
              style={{
                ...mono,
                fontSize: 10,
                padding: "2px 6px",
                borderRadius: 999,
                background: isSelected ? "rgba(255,255,255,0.15)" : "#e2e8f0",
                color: isSelected ? "#ffffff" : "#0f172a",
              }}
            >
              PICKED
            </span>
          ) : null}
          {isActualWinner ? (
            <span
              style={{
                ...mono,
                fontSize: 10,
                padding: "2px 6px",
                borderRadius: 999,
                background: isSelected ? "rgba(134,239,172,0.18)" : "#dcfce7",
                color: isSelected ? "#bbf7d0" : "#166534",
              }}
            >
              ACTUAL
            </span>
          ) : null}
          {isSelected && isCorrectPick ? (
            <span
              style={{
                ...mono,
                fontSize: 10,
                padding: "2px 6px",
                borderRadius: 999,
                background: "rgba(134,239,172,0.18)",
                color: "#bbf7d0",
              }}
            >
              CORRECT
            </span>
          ) : null}
          {isSelected && isMissedPick ? (
            <span
              style={{
                ...mono,
                fontSize: 10,
                padding: "2px 6px",
                borderRadius: 999,
                background: "rgba(248,113,113,0.18)",
                color: "#fecaca",
              }}
            >
              MISSED
            </span>
          ) : null}
          {isSelected && isFadingModel ? (
            <span
              style={{
                ...mono,
                fontSize: 10,
                padding: "2px 6px",
                borderRadius: 999,
                background: "rgba(251,191,36,0.18)",
                color: "#fde68a",
              }}
            >
              FADE
            </span>
          ) : null}
          {isUpset ? (
            <span
              style={{
                ...mono,
                fontSize: 10,
                padding: "2px 6px",
                borderRadius: 999,
                background: isSelected ? "rgba(251,191,36,0.18)" : "rgba(245, 158, 11, 0.12)",
                color: isSelected ? "#fde68a" : "#b45309",
              }}
            >
              {isMajorUpset ? "MAJOR UPSET" : "UPSET"}
            </span>
          ) : null}
        </div>
      </div>
      <div style={{ ...mono, fontSize: 11, opacity: isSelected ? 0.9 : 1 }}>
        {teamMetricSummary(team)}
      </div>
      <div style={{ ...mono, fontSize: 11, opacity: isSelected ? 0.9 : 1, marginTop: 4 }}>
        Rank {team.rank} | {team.conference} | {team.record}
      </div>
    </button>
  );
}

export default function BracketGame({
  game,
  prediction,
  comparison,
  grading,
  predictionLoading,
  predictionError,
  onSelectWinner,
}: {
  game: ResolvedBracketGame;
  prediction?: MatchupPrediction;
  comparison?: GameComparison;
  grading?: GradedGameResult;
  predictionLoading?: boolean;
  predictionError?: string;
  onSelectWinner: (gameId: string, teamId: number) => void;
}) {
  const teamA = game.teamA;
  const teamB = game.teamB;
  const isResolved = Boolean(teamA && teamB);
  const selectedWinnerId = game.selectedWinnerId;

  return (
    <div
      style={{
        background: "#ffffff",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        padding: 14,
        boxShadow: "0 1px 3px rgba(0, 0, 0, 0.04)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 10, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: "#0f172a" }}>{game.title}</div>
          <div style={{ ...mono, fontSize: 11, color: "#64748b" }}>
            {game.region ? `${game.region} • ` : ""}
            {game.roundLabel}
          </div>
        </div>
      </div>

      {comparison?.selectedWinnerName && comparison?.modelWinnerName ? (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 6,
            marginBottom: 10,
          }}
        >
          <span
            style={{
              ...mono,
              fontSize: 10,
              padding: "3px 7px",
              borderRadius: 999,
              background: comparison.agreesWithModel ? "#dcfce7" : "#fffbeb",
              color: comparison.agreesWithModel ? "#166534" : "#b45309",
            }}
          >
            {comparison.agreesWithModel ? "AGREE" : "FADE"}
          </span>
          <span
            style={{
              ...mono,
              fontSize: 10,
              padding: "3px 7px",
              borderRadius: 999,
              background: "#eff6ff",
              color: "#1d4ed8",
            }}
          >
            {comparison.confidenceLabel}
          </span>
          {comparison.isMajorUpset ? (
            <span
              style={{
                ...mono,
                fontSize: 10,
                padding: "3px 7px",
                borderRadius: 999,
                background: "#fff7ed",
                color: "#c2410c",
              }}
            >
              MAJOR UPSET
            </span>
          ) : null}
        </div>
      ) : null}

      {grading?.isFinal && grading.actualWinnerName ? (
        <div
          style={{
            marginBottom: 10,
            padding: "8px 10px",
            borderRadius: 8,
            border: `1px solid ${grading.status === "incorrect" ? "#fecaca" : "#bbf7d0"}`,
            background: grading.status === "incorrect" ? "#fef2f2" : "#f0fdf4",
            color: grading.status === "incorrect" ? "#991b1b" : "#166534",
          }}
        >
          <div style={{ ...mono, fontSize: 11, lineHeight: 1.5 }}>
            Actual winner: {displayTeam(grading.actualWinnerName)}
            {grading.status === "correct" ? " | Pick graded correct" : grading.status === "incorrect" ? " | Pick missed" : ""}
          </div>
        </div>
      ) : null}

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <TeamRow
          team={teamA}
          source={game.sourceA}
          isSelected={selectedWinnerId === teamA?.id}
          isClickable={isResolved}
          isFavorite={prediction?.favoredTeamId === teamA?.id}
          isUpset={Boolean(teamA && teamB && selectedWinnerId === teamA.id && teamA.seed > teamB.seed)}
          isMajorUpset={Boolean(
            teamA && teamB && selectedWinnerId === teamA.id && teamA.seed - teamB.seed >= MAJOR_UPSET_SEED_GAP,
          )}
          isFadingModel={Boolean(
            teamA && selectedWinnerId === teamA.id && prediction && prediction.modelWinnerId !== teamA.id,
          )}
          isActualWinner={grading?.actualWinnerId === teamA?.id}
          isCorrectPick={grading?.status === "correct" && selectedWinnerId === teamA?.id}
          isMissedPick={grading?.status === "incorrect" && selectedWinnerId === teamA?.id}
          onSelect={() => teamA && onSelectWinner(game.id, teamA.id)}
        />
        <TeamRow
          team={teamB}
          source={game.sourceB}
          isSelected={selectedWinnerId === teamB?.id}
          isClickable={isResolved}
          isFavorite={prediction?.favoredTeamId === teamB?.id}
          isUpset={Boolean(teamA && teamB && selectedWinnerId === teamB.id && teamB.seed > teamA.seed)}
          isMajorUpset={Boolean(
            teamA && teamB && selectedWinnerId === teamB.id && teamB.seed - teamA.seed >= MAJOR_UPSET_SEED_GAP,
          )}
          isFadingModel={Boolean(
            teamB && selectedWinnerId === teamB.id && prediction && prediction.modelWinnerId !== teamB.id,
          )}
          isActualWinner={grading?.actualWinnerId === teamB?.id}
          isCorrectPick={grading?.status === "correct" && selectedWinnerId === teamB?.id}
          isMissedPick={grading?.status === "incorrect" && selectedWinnerId === teamB?.id}
          onSelect={() => teamB && onSelectWinner(game.id, teamB.id)}
        />
      </div>

      <MatchupPredictionCard
        prediction={prediction}
        loading={predictionLoading}
        error={predictionError}
        selectedWinnerId={selectedWinnerId}
        comparison={comparison}
      />
    </div>
  );
}
