import { CSSProperties, MouseEvent, useEffect, useState } from "react";
import { displayTeam } from "../../lib/data";
import { GameComparison } from "../../lib/bracket/comparison";
import { BracketSource, BracketTeam, GradedGameResult, MatchupPrediction, ResolvedBracketGame } from "../../lib/bracket/types";
import MatchupPredictionCard from "./MatchupPredictionCard";

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

function mlPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function gameHeading(game: ResolvedBracketGame): string {
  if (game.roundId === "final-four" || game.roundId === "national-championship") {
    return game.title;
  }
  if (game.roundId === "first-four") {
    return `${game.region} play-in`;
  }
  return game.roundLabel;
}

function TeamRow({
  team,
  source,
  prediction,
  compact,
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
  prediction?: MatchupPrediction;
  compact?: boolean;
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
          padding: compact ? "6px 8px" : "8px 10px",
          borderRadius: 8,
          border: "1px dashed #cbd5e1",
          background: "#f8fafc",
          color: "#94a3b8",
        }}
      >
        <div style={{ ...mono, fontSize: compact ? 9 : 10 }}>Awaiting {source.label}</div>
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
        padding: compact ? "5px 8px" : "8px 10px",
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
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: 8, alignItems: "center" }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: compact ? 11 : 12, fontWeight: 700, lineHeight: 1.2, display: "flex", gap: 4, alignItems: "baseline", minWidth: 0 }}>
            <span
              style={{
                ...mono,
                fontSize: compact ? 9 : 10,
                opacity: isSelected ? 0.82 : 0.95,
                flexShrink: 0,
              }}
            >
              ({team.seed})
            </span>
            <span
              style={{
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                lineHeight: 1.2,
                minWidth: 0,
              }}
            >
              {displayTeam(team.name)}
            </span>
            <span style={{ ...mono, fontSize: compact ? 9 : 10, opacity: isSelected ? 0.82 : 0.95, flexShrink: 0 }}>
              #{team.rank}
            </span>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
          {prediction && isFavorite ? (
            <div style={{ ...mono, fontSize: compact ? 9 : 10, fontWeight: 600, whiteSpace: "nowrap" }}>
              -{prediction.projectedSpread.toFixed(1)} • {team.id === prediction.teamAId ? mlPct(prediction.winProbA) : mlPct(prediction.winProbB)}
            </div>
          ) : null}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4, justifyContent: "flex-end" }}>
          {isSelected && !isCorrectPick && !isMissedPick && !compact ? (
            <span
              style={{
                ...mono,
                fontSize: compact ? 8 : 9,
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
                fontSize: compact ? 8 : 9,
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
                fontSize: compact ? 8 : 9,
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
                fontSize: compact ? 8 : 9,
                padding: "2px 5px",
                borderRadius: 999,
                background: "rgba(248,113,113,0.18)",
                color: "#fecaca",
              }}
            >
              MISSED
            </span>
          ) : null}
          {isSelected && isFadingModel && !compact ? (
            <span
              style={{
                ...mono,
                fontSize: compact ? 8 : 9,
                padding: "2px 5px",
                borderRadius: 999,
                background: "rgba(251,191,36,0.18)",
                color: "#fde68a",
              }}
            >
              FADE
            </span>
          ) : null}
          {isUpset && !compact ? (
            <span
              style={{
                ...mono,
                fontSize: compact ? 8 : 9,
                padding: "2px 5px",
                borderRadius: 999,
                background: isSelected ? "rgba(251,191,36,0.18)" : "rgba(245,158,11,0.12)",
                color: isSelected ? "#fde68a" : "#b45309",
              }}
            >
              {isMajorUpset ? "UPSET+" : "UPSET"}
            </span>
          ) : null}
        </div>
      </div>
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
  compact = false,
  fixedHeight,
}: {
  game: ResolvedBracketGame;
  prediction?: MatchupPrediction;
  comparison?: GameComparison;
  grading?: GradedGameResult;
  predictionLoading?: boolean;
  predictionError?: string;
  onSelectWinner: (gameId: string, teamId: number) => void;
  compact?: boolean;
  fixedHeight?: number;
}) {
  const teamA = game.teamA;
  const teamB = game.teamB;
  const isResolved = Boolean(teamA && teamB);
  const selectedWinnerId = game.selectedWinnerId;
  const [detailsOpen, setDetailsOpen] = useState(false);
  const showInfoButton = isResolved || Boolean(predictionLoading || predictionError || prediction);

  function handleToggleDetails(event: MouseEvent<HTMLButtonElement>) {
    event.stopPropagation();
    setDetailsOpen((current) => !current);
  }

  useEffect(() => {
    if (!detailsOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setDetailsOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [detailsOpen]);

  return (
    <div
      style={{
        background: "#ffffff",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        padding: compact ? 8 : 10,
        boxShadow: "0 1px 3px rgba(0, 0, 0, 0.04)",
        height: fixedHeight,
        minHeight: compact ? 74 : undefined,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, marginBottom: compact ? 6 : 8, alignItems: "center" }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: compact ? 11 : 12, fontWeight: 700, color: "#0f172a", lineHeight: 1.2 }}>{gameHeading(game)}</div>
          <div style={{ ...mono, fontSize: compact ? 9 : 10, color: "#64748b", marginTop: 2 }}>
            {game.region ?? "National"}
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
              width: compact ? 20 : 22,
              height: compact ? 20 : 22,
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

      <div style={{ display: "grid", gap: 7 }}>
        <TeamRow
          team={teamA}
          source={game.sourceA}
          prediction={prediction}
          compact={compact}
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
          prediction={prediction}
          compact={compact}
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
        <div
          onClick={() => setDetailsOpen(false)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(15, 23, 42, 0.42)",
            zIndex: 60,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 20,
          }}
        >
          <div
            onClick={(event) => event.stopPropagation()}
            style={{
              width: "min(720px, 100%)",
              maxHeight: "min(85vh, 860px)",
              overflowY: "auto",
              borderRadius: 14,
              background: "#ffffff",
              boxShadow: "0 16px 40px rgba(15,23,42,0.22)",
              padding: 14,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center", marginBottom: 10 }}>
              <div>
                <div style={{ fontSize: 15, fontWeight: 700, color: "#0f172a" }}>{gameHeading(game)}</div>
                <div style={{ ...mono, fontSize: 11, color: "#64748b", marginTop: 2 }}>
                  {game.region ?? "National"} • {game.roundLabel}
                </div>
              </div>
              <button
                type="button"
                onClick={() => setDetailsOpen(false)}
                style={{
                  ...mono,
                  borderRadius: 999,
                  border: "1px solid #cbd5e1",
                  background: "#ffffff",
                  color: "#475569",
                  cursor: "pointer",
                  width: 28,
                  height: 28,
                }}
              >
                x
              </button>
            </div>
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
          </div>
        </div>
      ) : null}
    </div>
  );
}
