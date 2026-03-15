import { GradedGameResult, ResolvedBracketGame, MatchupPrediction } from "../../lib/bracket/types";
import { GameComparison } from "../../lib/bracket/comparison";
import BracketGame from "./BracketGame";

export default function BracketRound({
  label,
  games,
  predictions,
  comparisons,
  grading,
  loadingGames,
  errorGames,
  onSelectWinner,
  compact,
}: {
  label: string;
  games: ResolvedBracketGame[];
  predictions: Record<string, MatchupPrediction | undefined>;
  comparisons: Record<string, GameComparison | undefined>;
  grading: Record<string, GradedGameResult | undefined>;
  loadingGames: Record<string, boolean | undefined>;
  errorGames: Record<string, string | undefined>;
  onSelectWinner: (gameId: string, teamId: number) => void;
  compact?: boolean;
}) {
  return (
    <section
      style={{
        minWidth: compact ? "auto" : 300,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div
        style={{
          position: "sticky",
          top: 0,
          zIndex: 1,
          background: "#f8fafc",
          paddingBottom: 4,
        }}
      >
        <h2
          style={{
            fontSize: 15,
            fontWeight: 700,
            letterSpacing: "-0.02em",
            margin: 0,
            color: "#0f172a",
          }}
        >
          {label}
        </h2>
      </div>

      {games.map((game) => (
        <BracketGame
          key={game.id}
          game={game}
          prediction={predictions[game.id]}
          comparison={comparisons[game.id]}
          grading={grading[game.id]}
          predictionLoading={loadingGames[game.id]}
          predictionError={errorGames[game.id]}
          onSelectWinner={onSelectWinner}
        />
      ))}
    </section>
  );
}
