import {
  formatAmericanOddsFromProb,
  getSiteHomeWinProbFromValues,
} from "../data";
import { buildNcaaBracketGames, getBracketTeams } from "./ncaaBracket";
import {
  BracketRoundId,
  BracketSource,
  MatchupPredictionCache,
  NcaaBracketField,
} from "./types";

const DEFAULT_TOURNAMENT_START = "2026-03-20T00:00:00.000Z";

export type NcaaOddsRoundKey =
  | "round-of-64"
  | "round-of-32"
  | "sweet-16"
  | "elite-8"
  | "final-four"
  | "national-championship"
  | "champion";

export type NcaaOddsRow = {
  teamId: number;
  team: string;
  seed: number;
  region: string | null;
  conference: string;
  record: string;
  confRecord: string;
  rank: number;
  roundProbabilities: Record<NcaaOddsRoundKey, number>;
};

export type NcaaOddsSummary = {
  titleFavorite: NcaaOddsRow | null;
  finalFourLocks: NcaaOddsRow[];
};

export type NcaaOddsData = {
  generatedAt: string;
  season: number;
  methodology: {
    type: "exact_bracket";
    note: string;
  };
  rows: NcaaOddsRow[];
  summary: NcaaOddsSummary;
};

type ProbabilityMap = Map<number, number>;

function roundKeyFromGame(
  roundId: BracketRoundId,
): Exclude<NcaaOddsRoundKey, "champion"> | null {
  if (roundId === "first-four") return null;
  return roundId;
}

function emptyRoundProbabilities(): Record<NcaaOddsRoundKey, number> {
  return {
    "round-of-64": 0,
    "round-of-32": 0,
    "sweet-16": 0,
    "elite-8": 0,
    "final-four": 0,
    "national-championship": 0,
    champion: 0,
  };
}

function addProbability(
  target: ProbabilityMap,
  teamId: number,
  probability: number,
): void {
  target.set(teamId, (target.get(teamId) ?? 0) + probability);
}

function probabilityForMatchup(
  cache: MatchupPredictionCache,
  teamAId: number,
  teamBId: number,
): number {
  const canonicalA = Math.min(teamAId, teamBId);
  const canonicalB = Math.max(teamAId, teamBId);
  const key = `${canonicalA}::${canonicalB}`;
  const entry = cache.predictions[key];
  if (!entry) {
    throw new Error(`Missing NCAA matchup cache entry for ${key}`);
  }

  const directOrder = entry.team1_id === teamAId && entry.team2_id === teamBId;
  const marginA = directOrder
    ? (entry.display_mu_team1_minus_team2 ?? entry.mu_team1_minus_team2)
    : -(entry.display_mu_team1_minus_team2 ?? entry.mu_team1_minus_team2);
  const sigma = entry.pred_sigma ?? null;
  const startTime = entry.start_time ?? DEFAULT_TOURNAMENT_START;
  const probability = getSiteHomeWinProbFromValues(marginA, sigma, startTime);
  if (probability == null) {
    return directOrder ? entry.win_prob_team1 : 1 - entry.win_prob_team1;
  }
  return probability;
}

function sourceDistribution(
  source: BracketSource,
  winnerByGame: Map<string, ProbabilityMap>,
): ProbabilityMap {
  if (source.type === "team") {
    return new Map([[source.teamId, 1]]);
  }
  const distribution = winnerByGame.get(source.gameId);
  if (!distribution) {
    throw new Error(`Missing feeder winner distribution for ${source.gameId}`);
  }
  return distribution;
}

function sortedRows(rows: NcaaOddsRow[]): NcaaOddsRow[] {
  return [...rows].sort((a, b) => {
    const champDiff =
      b.roundProbabilities.champion - a.roundProbabilities.champion;
    if (champDiff !== 0) return champDiff;
    const titleDiff =
      b.roundProbabilities["national-championship"] -
      a.roundProbabilities["national-championship"];
    if (titleDiff !== 0) return titleDiff;
    return a.seed - b.seed || a.team.localeCompare(b.team);
  });
}

export function buildNcaaOddsData(
  field: NcaaBracketField,
  cache: MatchupPredictionCache,
): NcaaOddsData {
  const teams = getBracketTeams(field);
  const games = buildNcaaBracketGames(field).sort(
    (a, b) => a.roundOrder - b.roundOrder || a.matchupOrder - b.matchupOrder,
  );
  const rowsByTeamId = new Map<number, NcaaOddsRow>(
    teams.map((team) => [
      team.id,
      {
        teamId: team.id,
        team: team.name,
        seed: team.seed,
        region: team.region ?? null,
        conference: team.conference,
        record: team.record,
        confRecord: team.confRecord,
        rank: team.rank,
        roundProbabilities: emptyRoundProbabilities(),
      },
    ]),
  );
  const winnerByGame = new Map<string, ProbabilityMap>();

  for (const game of games) {
    const distA = sourceDistribution(game.sourceA, winnerByGame);
    const distB = sourceDistribution(game.sourceB, winnerByGame);
    const reachRound = roundKeyFromGame(game.roundId);
    if (reachRound) {
      for (const [teamId, probability] of distA.entries()) {
        const row = rowsByTeamId.get(teamId);
        if (row) row.roundProbabilities[reachRound] += probability;
      }
      for (const [teamId, probability] of distB.entries()) {
        const row = rowsByTeamId.get(teamId);
        if (row) row.roundProbabilities[reachRound] += probability;
      }
    }

    const winners = new Map<number, number>();
    for (const [teamAId, probAReach] of distA.entries()) {
      for (const [teamBId, probBReach] of distB.entries()) {
        const meetingProb = probAReach * probBReach;
        const probAWin = probabilityForMatchup(cache, teamAId, teamBId);
        addProbability(winners, teamAId, meetingProb * probAWin);
        addProbability(winners, teamBId, meetingProb * (1 - probAWin));
      }
    }
    winnerByGame.set(game.id, winners);
  }

  const championDistribution =
    winnerByGame.get("national-championship") ?? new Map<number, number>();
  for (const [teamId, probability] of championDistribution.entries()) {
    const row = rowsByTeamId.get(teamId);
    if (row) row.roundProbabilities.champion = probability;
  }

  const rows = sortedRows(Array.from(rowsByTeamId.values()));
  const titleFavorite = rows[0] ?? null;
  const finalFourLocks = rows
    .filter((row) => row.roundProbabilities["final-four"] >= 0.5)
    .slice(0, 8);

  return {
    generatedAt: cache.generated_at,
    season: field.season,
    methodology: {
      type: "exact_bracket",
      note: "Exact NCAA bracket advancement probabilities using display-adjusted matchup margins run through the site moneyline transform with the cached sigma model.",
    },
    rows,
    summary: {
      titleFavorite,
      finalFourLocks,
    },
  };
}

export function formatRoundOdds(probability: number): string | null {
  return formatAmericanOddsFromProb(probability);
}
