import { CSSProperties, MouseEvent, useState } from "react";
import { displayTeam } from "../../lib/data";
import { GameComparison, MAJOR_UPSET_SEED_GAP } from "../../lib/bracket/comparison";
import { BracketSource, BracketTeam, GradedGameResult, MatchupPrediction, ResolvedBracketGame } from "../../lib/bracket/types";
import MatchupPredictionCard from "./MatchupPredictionCard";

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

function compactPct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function compactFavoriteSummary(prediction?: MatchupPrediction): string | null {
  if (!prediction) return null;
  const favoriteWinProb = prediction.favoredTeamId === prediction.teamAId ? prediction.winProbA : prediction.winProbB;
  return `${displayTeam(prediction.favoredTeamName)} -${prediction.projectedSpread.toFixed(1)} • ${compactPct(favoriteWinProb)}`;
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
          padding: "8px 10px",
          borderRadius: 8,
          border: "1px dashed #cbd5e1",
          background: "#f8fafc",
          color: "#94a3b8",
        }}
      >
        <div style={{ ...mono, fontSize: 10 }}>Awaiting {source.label}</div>
      </div>
    );
  }

  const borderColor = isSelected
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
        : "#e2e8f0";

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={!isClickable}
      style={{
        width: "100%",
        textAlign: "left",
        padding: "8px 10px",
        borderRadius: 8,
        border: `1px solid ${borderColor}`,
        background: isSelected ? "#0f172a" : isActualWinner ? "#f0fdf4" : "#ffffff",
        color: isSelected ? "#ffffff" : "#0f172a",
        cursor: isClickable ? "pointer" : "default",
        transition: "all 0.15s",
        boxShadow:
          isSelected && isCorrectPick
            ? "0 0 0 1px rgba(22,163,74,0.24)"
            : isSelected && isMissedPick
              ? "0 0 0 1px rgba(220,38,38,0.22)"
              : isSelected && isFadingModel
                ? "0 0 0 1px rgba(245,158,11,0.22)"
                : "none",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
        <div style={{ fontSize: 12, fontWeight: 700, lineHeight: 1.25 }}>
          ({team.seed}) {displayTeam(team.name)}
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, justifyContent: "flex-end" }}>
          {isFavorite ? (
            <span
              style={{
                ...mono,
                fontSize: 9,
                padding: "2px 5px",
                borderRadius: 999,
                background: isSelected ? "rgba(255,255,255,0.14)" : "#dbeafe",
                color: isSelected ? "#e2e8f0" : "#1d4ed8",
              }}
            >
              FAV
            </span>
          ) : null}
          {isSelected ? (
            <span
              style={{
                ...mono,
                fontSize: 9,
                padding: "2px 5px",
                borderRadius: 999,
                background: isSelected ? "rgba(255,255,255,0.14)" : "#e2e8f0",
                color: isSelected ? "#ffffff" : "#0f172a",
              }}
            >
              PICK
            </span>
          ) : null}
          {isActualWinner ? (
            <span
              style={{
                ...mono,
                fontSize: 9,
                padding: "2px 5px",
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
                fontSize: 9,
                padding: "2px 5px",
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
                fontSize: 9,
                padding: "2px 5px",
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
                fontSize: 9,
                padding: "2px 5px",
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
                fontSize: 9,
                padding: "2px 5px",
                borderRadius: 999,
                background: isSelected ? "rgba(251,191,36,0.18)" : "rgba(245,158,11,0.12)",
                color: isSelected ? "#fde68a" : "#b45309",
              }}
            >
              {isMajorUpset ? "MAJOR" : "UPSET"}
            </span>
          ) : null}
        </div>
      </div>
      <div style={{ ...mono, fontSize: 10, opacity: isSelected ? 0.88 : 1, marginTop: 4 }}>
        Rank {team.rank} | {team.record}
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
  const [detailsOpen, setDetailsOpen] = useState(false);
  const favoriteSummary = compactFavoriteSummary(prediction);
  const showInfoButton = isResolved || Boolean(predictionLoading || predictionError || prediction);

  function handleToggleDetails(event: MouseEvent<HTMLButtonElement>) {
    event.stopPropagation();
    setDetailsOpen((current) => !current);
  }

  return (
    <div
      style={{
        background: "#ffffff",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        padding: 10,
        boxShadow: "0 1px 3px rgba(0, 0, 0, 0.04)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, marginBottom: 8, alignItems: "flex-start" }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#0f172a", lineHeight: 1.2 }}>{game.title}</div>
          <div style={{ ...mono, fontSize: 10, color: "#64748b", marginTop: 2 }}>
            {game.region ? `${game.region} • ` : ""}
            {game.roundLabel}
          </div>
        </div>
        {showInfoButton ? (
          <button
            type="button"
            aria-label={detailsOpen ? "Hide matchup details" : "Show matchup details"}
            aria-expanded={detailsOpen}
            onClick={handleToggleDetails}
            style={{
              ...mono,
              width: 22,
              height: 22,
              borderRadius: 999,
              border: "1px solid #cbd5e1",
              background: detailsOpen ? "#e2e8f0" : "#ffffff",
              color: "#475569",
              cursor: "pointer",
              flexShrink: 0,
            }}
          >
            i
          </button>
        ) : null}
      </div>

      {(comparison?.selectedWinnerName && comparison?.modelWinnerName) || comparison?.confidenceLabel ? (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
          {comparison?.selectedWinnerName && comparison?.modelWinnerName ? (
            <span
              style={{
                ...mono,
                fontSize: 9,
                padding: "2px 6px",
                borderRadius: 999,
                background: comparison.agreesWithModel ? "#dcfce7" : "#fffbeb",
                color: comparison.agreesWithModel ? "#166534" : "#b45309",
              }}
            >
              {comparison.agreesWithModel ? "AGREE" : "FADE"}
            </span>
          ) : null}
          {comparison?.confidenceLabel ? (
            <span
              style={{
                ...mono,
                fontSize: 9,
                padding: "2px 6px",
                borderRadius: 999,
                background: "#eff6ff",
                color: "#1d4ed8",
              }}
            >
              {comparison.confidenceLabel}
            </span>
          ) : null}
          {comparison?.isMajorUpset ? (
            <span
              style={{
                ...mono,
                fontSize: 9,
                padding: "2px 6px",
                borderRadius: 999,
                background: "#fff7ed",
                color: "#c2410c",
              }}
            >
              {MAJOR_UPSET_SEED_GAP}+ SEED UPSET
            </span>
          ) : null}
        </div>
      ) : null}

      <div style={{ ...mono, fontSize: 10, color: predictionError ? "#b91c1c" : "#475569", marginBottom: 8 }}>
        {favoriteSummary
          ? `Model favorite: ${favoriteSummary}`
          : predictionLoading
            ? "Loading prediction..."
            : predictionError
              ? predictionError
              : "Prediction appears when both teams are known."}
      </div>

      <div style={{ display: "grid", gap: 7 }}>
        <TeamRow
          team={teamA}
          source={game.sourceA}
          isSelected={selectedWinnerId === teamA?.id}
          isClickable={Boolean(teamA)}
          isFavorite={prediction?.favoredTeamId === teamA?.id}
          isUpset={selectedWinnerId === teamA?.id ? comparison?.isUpset ?? false : false}
          isMajorUpset={selectedWinnerId === teamA?.id ? comparison?.isMajorUpset ?? false : false}
          isFadingModel={selectedWinnerId === teamA?.id ? comparison?.agreesWithModel === false : false}
          isActualWinner={grading?.actualWinnerId === teamA?.id}
          isCorrectPick={grading?.status === "correct" && selectedWinnerId === teamA?.id}
          isMissedPick={grading?.status === "incorrect" && selectedWinnerId === teamA?.id}
          onSelect={() => teamA && onSelectWinner(game.id, teamA.id)}
        />
        <TeamRow
          team={teamB}
          source={game.sourceB}
          isSelected={selectedWinnerId === teamB?.id}
          isClickable={Boolean(teamB)}
          isFavorite={prediction?.favoredTeamId === teamB?.id}
          isUpset={selectedWinnerId === teamB?.id ? comparison?.isUpset ?? false : false}
          isMajorUpset={selectedWinnerId === teamB?.id ? comparison?.isMajorUpset ?? false : false}
          isFadingModel={selectedWinnerId === teamB?.id ? comparison?.agreesWithModel === false : false}
          isActualWinner={grading?.actualWinnerId === teamB?.id}
          isCorrectPick={grading?.status === "correct" && selectedWinnerId === teamB?.id}
          isMissedPick={grading?.status === "incorrect" && selectedWinnerId === teamB?.id}
          onSelect={() => teamB && onSelectWinner(game.id, teamB.id)}
        />
      </div>

      {detailsOpen ? (
        <MatchupPredictionCard
          prediction={prediction}
          loading={predictionLoading}
          error={predictionError}
          selectedWinnerId={game.selectedWinnerId}
          comparison={comparison}
          grading={grading}
          teamA={teamA}
          teamB={teamB}
        />
      ) : null}
    </div>
  );
}
