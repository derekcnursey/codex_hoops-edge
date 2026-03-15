export type NcaaFieldTeam = {
  team_id: number;
  team: string;
  rank: number;
  conference: string;
  record: string;
  conf_record: string;
  adj_oe: number;
  adj_de: number;
  adj_margin: number;
  adj_tempo: number;
  model_index: number | null;
};

export type NcaaRegionEntry =
  | ({
      seed: number;
      source: "team";
    } & NcaaFieldTeam)
  | {
      seed: number;
      source: "play_in";
      play_in_game_id: string;
    };

export type NcaaRegion = {
  name: string;
  entries: NcaaRegionEntry[];
};

export type NcaaFirstFourGame = {
  id: string;
  label: string;
  region: string;
  seed: number;
  teams: NcaaFieldTeam[];
};

export type NcaaBracketField = {
  generated_at: string;
  season: number;
  source: string;
  note: string;
  regions: NcaaRegion[];
  first_four: NcaaFirstFourGame[];
};

export type MatchupPredictionCacheEntry = {
  team1_id: number;
  team1_name: string;
  team2_id: number;
  team2_name: string;
  mu_team1_minus_team2: number;
  win_prob_team1: number;
};

export type MatchupPredictionCache = {
  generated_at: string;
  season: number;
  neutral_site: boolean;
  source: string;
  note: string;
  predictions: Record<string, MatchupPredictionCacheEntry>;
};

export type NcaaResultsStatus = "pending" | "in_progress" | "final";

export type NcaaTournamentResultGame = {
  winner_team_id?: number | null;
  loser_team_id?: number | null;
  status: NcaaResultsStatus;
};

export type NcaaTournamentResults = {
  version: number;
  season: number;
  games: Record<string, NcaaTournamentResultGame>;
};

export type BracketRoundId =
  | "first-four"
  | "round-of-64"
  | "round-of-32"
  | "sweet-16"
  | "elite-8"
  | "final-four"
  | "national-championship";

export type BracketTeam = {
  id: number;
  name: string;
  seed: number;
  region?: string;
  rank: number;
  conference: string;
  record: string;
  confRecord: string;
  adjOe: number;
  adjDe: number;
  adjNet: number;
  adjTempo: number;
  modelIndex: number | null;
};

export type BracketSource =
  | {
      type: "team";
      teamId: number;
      label: string;
    }
  | {
      type: "winner";
      gameId: string;
      label: string;
    };

export type BracketGameDefinition = {
  id: string;
  roundId: BracketRoundId;
  roundLabel: string;
  roundOrder: number;
  title: string;
  region?: string;
  matchupOrder: number;
  sourceA: BracketSource;
  sourceB: BracketSource;
};

export type ResolvedBracketGame = BracketGameDefinition & {
  teamA: BracketTeam | null;
  teamB: BracketTeam | null;
  selectedWinnerId?: number;
};

export type MatchupPrediction = {
  teamAId: number;
  teamAName: string;
  teamBId: number;
  teamBName: string;
  favoredTeamId: number;
  favoredTeamName: string;
  underdogTeamId: number;
  underdogTeamName: string;
  winProbA: number;
  winProbB: number;
  projectedSpread: number;
  modelWinnerId: number;
  modelWinnerName: string;
  projectedScoreA?: number | null;
  projectedScoreB?: number | null;
};

export type NcaaValidationResult = {
  valid: boolean;
  errors: string[];
};

export type GradedGameStatus = "pending" | "correct" | "incorrect";

export type GradedGameResult = {
  gameId: string;
  roundId: BracketRoundId;
  roundLabel: string;
  actualWinnerId?: number;
  actualLoserId?: number;
  actualWinnerName?: string;
  actualLoserName?: string;
  status: GradedGameStatus;
  isFinal: boolean;
  isScored: boolean;
  pointsEarned: number;
  pointsPossible: number;
};

export type GradedRoundSummary = {
  roundId: BracketRoundId;
  roundLabel: string;
  correct: number;
  incorrect: number;
  pending: number;
  score: number;
  possibleScore: number;
};

export type BracketGradingSummary = {
  correct: number;
  incorrect: number;
  pending: number;
  totalGraded: number;
  score: number;
  possibleScore: number;
  rounds: GradedRoundSummary[];
  byGame: Record<string, GradedGameResult>;
};
