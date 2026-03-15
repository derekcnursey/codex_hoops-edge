import { MatchupPrediction, MatchupPredictionCacheEntry } from "./types";

export function canonicalMatchupKey(teamAId: number, teamBId: number): string {
  return teamAId < teamBId ? `${teamAId}::${teamBId}` : `${teamBId}::${teamAId}`;
}

export function flipPrediction(prediction: MatchupPrediction): MatchupPrediction {
  return {
    teamAId: prediction.teamBId,
    teamAName: prediction.teamBName,
    teamBId: prediction.teamAId,
    teamBName: prediction.teamAName,
    favoredTeamId: prediction.favoredTeamId,
    favoredTeamName: prediction.favoredTeamName,
    underdogTeamId: prediction.underdogTeamId,
    underdogTeamName: prediction.underdogTeamName,
    winProbA: prediction.winProbB,
    winProbB: prediction.winProbA,
    projectedSpread: prediction.projectedSpread,
    modelWinnerId: prediction.modelWinnerId,
    modelWinnerName: prediction.modelWinnerName,
    projectedScoreA: prediction.projectedScoreB ?? null,
    projectedScoreB: prediction.projectedScoreA ?? null,
  };
}

export function orientPrediction(
  prediction: MatchupPrediction,
  teamAId: number,
  teamBId: number,
): MatchupPrediction {
  if (prediction.teamAId === teamAId && prediction.teamBId === teamBId) return prediction;
  if (prediction.teamAId === teamBId && prediction.teamBId === teamAId) {
    return flipPrediction(prediction);
  }
  throw new Error("Cached prediction does not match requested matchup");
}

export function canonicalizePrediction(prediction: MatchupPrediction): MatchupPrediction {
  if (prediction.teamAId < prediction.teamBId) return prediction;
  return flipPrediction(prediction);
}

export function buildPredictionFromCacheEntry(
  entry: MatchupPredictionCacheEntry,
  teamAId: number,
  teamBId: number,
): MatchupPrediction {
  const directOrder = entry.team1_id === teamAId && entry.team2_id === teamBId;
  const muForTeamA = directOrder ? entry.mu_team1_minus_team2 : -entry.mu_team1_minus_team2;
  const winProbA = directOrder ? entry.win_prob_team1 : 1 - entry.win_prob_team1;
  const winProbB = 1 - winProbA;
  const teamAName = directOrder ? entry.team1_name : entry.team2_name;
  const teamBName = directOrder ? entry.team2_name : entry.team1_name;
  const favoredTeamId = muForTeamA >= 0 ? teamAId : teamBId;
  const favoredTeamName = favoredTeamId === teamAId ? teamAName : teamBName;
  const underdogTeamId = favoredTeamId === teamAId ? teamBId : teamAId;
  const underdogTeamName = underdogTeamId === teamAId ? teamAName : teamBName;

  return {
    teamAId,
    teamAName,
    teamBId,
    teamBName,
    favoredTeamId,
    favoredTeamName,
    underdogTeamId,
    underdogTeamName,
    winProbA,
    winProbB,
    projectedSpread: Math.abs(muForTeamA),
    modelWinnerId: favoredTeamId,
    modelWinnerName: favoredTeamName,
    projectedScoreA: null,
    projectedScoreB: null,
  };
}
