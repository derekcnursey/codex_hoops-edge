import { buildNcaaBracketGames, getBracketTeams } from "./ncaaBracket";
import { buildPredictionFromCacheEntry, canonicalMatchupKey, canonicalizePrediction } from "./predictions";
import {
  BracketSource,
  BracketTeam,
  MarchBettingGame,
  MatchupPrediction,
  MatchupPredictionCache,
  NcaaBracketField,
} from "./types";

function resolveSource(source: BracketSource, teamById: Record<number, BracketTeam>): BracketTeam | null {
  if (source.type !== "team") return null;
  return teamById[source.teamId] ?? null;
}

export function buildScheduledNcaaMarchData(
  field: NcaaBracketField,
  cache: MatchupPredictionCache,
): {
  initialPredictionCache: Record<string, MatchupPrediction>;
  marchGames: MarchBettingGame[];
} {
  const initialPredictionCache: Record<string, MatchupPrediction> = {};
  const marchGames: MarchBettingGame[] = [];
  const teamById = Object.fromEntries(
    getBracketTeams(field).map((team) => [team.id, team]),
  ) as Record<number, BracketTeam>;

  const scheduledGames = buildNcaaBracketGames(field)
    .filter((game) => game.roundId === "first-four" || game.roundId === "round-of-64")
    .sort((a, b) => a.roundOrder - b.roundOrder || a.matchupOrder - b.matchupOrder);

  for (const game of scheduledGames) {
    const teamA = resolveSource(game.sourceA, teamById);
    const teamB = resolveSource(game.sourceB, teamById);
    if (!teamA || !teamB) continue;

    const matchupKey = canonicalMatchupKey(teamA.id, teamB.id);
    const entry = cache.predictions[matchupKey];
    if (!entry) continue;

    const prediction = buildPredictionFromCacheEntry(entry, teamA.id, teamB.id);
    initialPredictionCache[matchupKey] = canonicalizePrediction(prediction);

    const displayFavoredTeamId = prediction.displayFavoredTeamId ?? prediction.favoredTeamId;
    const displayFavoredTeamName = prediction.displayFavoredTeamName ?? prediction.favoredTeamName;
    const favoriteWinProb =
      displayFavoredTeamId === teamA.id
        ? (prediction.displayWinProbA ?? prediction.winProbA)
        : (prediction.displayWinProbB ?? prediction.winProbB);
    const rawDisplaySpreadHome =
      prediction.modelSpreadHome == null ? null : -prediction.modelSpreadHome;
    const displayDisplaySpreadHome =
      prediction.displayModelSpreadHome == null ? null : -prediction.displayModelSpreadHome;
    const rawDiffAbs =
      rawDisplaySpreadHome != null && prediction.marketSpreadHome != null
        ? Math.abs(rawDisplaySpreadHome - prediction.marketSpreadHome)
        : null;
    const displayDiffAbs =
      displayDisplaySpreadHome != null && prediction.marketSpreadHome != null
        ? Math.abs(displayDisplaySpreadHome - prediction.marketSpreadHome)
        : null;
    const spreadDiffAbs =
      prediction.displayProjectedSpread != null && prediction.marketProjectedSpread != null
        ? Math.abs(prediction.displayProjectedSpread - prediction.marketProjectedSpread)
        : null;

    marchGames.push({
      gameId: game.id,
      roundId: game.roundId,
      roundLabel: game.roundLabel,
      region: game.region ?? null,
      matchupOrder: game.matchupOrder,
      startTime: prediction.scheduledStartTime ?? null,
      teamAId: teamA.id,
      teamAName: teamA.name,
      teamASeed: teamA.seed,
      teamBId: teamB.id,
      teamBName: teamB.name,
      teamBSeed: teamB.seed,
      homeTeamId: prediction.scheduledHomeTeamId ?? null,
      homeTeamName: prediction.scheduledHomeTeamName ?? null,
      awayTeamId: prediction.scheduledAwayTeamId ?? null,
      awayTeamName: prediction.scheduledAwayTeamName ?? null,
      favoriteTeamId: prediction.modelWinnerId,
      favoriteTeamName: prediction.modelWinnerName,
      favoriteWinProb,
      rawProjectedSpread: prediction.rawProjectedSpread ?? prediction.projectedSpread,
      displayProjectedSpread: prediction.displayProjectedSpread ?? prediction.projectedSpread,
      modelSpreadHome: prediction.modelSpreadHome ?? null,
      displayModelSpreadHome: prediction.displayModelSpreadHome ?? null,
      predSigma: prediction.predSigma ?? null,
      edgeHomePoints: prediction.edgeHomePoints ?? null,
      displayEdgeHomePoints: prediction.displayEdgeHomePoints ?? null,
      pickSide: prediction.pickSide ?? null,
      pickCoverProb: prediction.pickCoverProb ?? null,
      pickProbEdge: prediction.pickProbEdge ?? null,
      displayFavoredTeamId,
      displayFavoredTeamName,
      marketSpreadHome: prediction.marketSpreadHome ?? null,
      marketProjectedSpread: prediction.marketProjectedSpread ?? null,
      marketFavoredTeamId: prediction.marketFavoredTeamId ?? null,
      marketFavoredTeamName: prediction.marketFavoredTeamName ?? null,
      marketLineSource: prediction.marketLineSource ?? null,
      rawDiffAbs,
      displayDiffAbs,
      spreadDiffAbs,
    });
  }

  return { initialPredictionCache, marchGames };
}
