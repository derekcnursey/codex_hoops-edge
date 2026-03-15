import fs from "fs";
import path from "path";

export type ProfitabilityRow = {
  group: string;
  family?: string | null;
  slice: string;
  bets: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
  roi_per_1_at_minus_110: number | null;
  p_value_gt_breakeven?: number | null;
  avg_pick_prob_edge?: number | null;
  avg_filter_score?: number | null;
  avg_abs_he_vs_market_edge?: number | null;
  new_transient_rate?: number | null;
  persistent_rate?: number | null;
  neutral_rate?: number | null;
};

export type BySeasonProfitabilityRow = ProfitabilityRow & {
  season: number;
};

export type RobustnessRow = {
  group: string;
  slice: string;
  season_count_with_bets: number;
  profitable_seasons: number;
  losing_seasons: number;
  breakeven_or_better_seasons: number;
  median_season_roi: number | null;
  mean_season_roi: number | null;
  season_roi_std: number | null;
  min_season_roi: number | null;
  max_season_roi: number | null;
  median_season_hit_rate: number | null;
  median_bets_per_season: number | null;
};

export type SignalDriverRow = {
  signal_driver: string;
  slice: string;
  bets: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
  roi_per_1_at_minus_110: number | null;
};

export type BettingTabMappingRow = {
  page_label: string;
  group: string;
  slice: string;
  note: string;
  bets: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
  roi_per_1_at_minus_110: number | null;
};

export type BettingPerformancePayload = {
  overall: ProfitabilityRow[];
  bySeason: BySeasonProfitabilityRow[];
  signalDrivers: SignalDriverRow[];
  robustness: RobustnessRow[];
  tabMapping: BettingTabMappingRow[];
  memo: string | null;
};

function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      const next = line[i + 1];
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  out.push(current);
  return out;
}

function coerceCell(raw: string): string | number | boolean | null {
  if (raw === "") return null;
  if (raw === "True") return true;
  if (raw === "False") return false;
  if (/^-?\d+(\.\d+)?([eE]-?\d+)?$/.test(raw)) return Number(raw);
  return raw;
}

function readCsvRows<T extends Record<string, unknown>>(filePath: string): T[] {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, "utf-8").trim();
  if (!raw) return [];
  const lines = raw.split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = parseCsvLine(line);
    const row: Record<string, unknown> = {};
    headers.forEach((header, idx) => {
      row[header] = coerceCell(cells[idx] ?? "");
    });
    return row as T;
  });
}

function getBettingPerformanceRoot(): string {
  const candidates = [
    path.join(process.cwd(), "artifacts", "market_bet_profitability_v1"),
    path.join(process.cwd(), "..", "artifacts", "market_bet_profitability_v1"),
    path.join(process.cwd(), "site", "..", "artifacts", "market_bet_profitability_v1"),
    path.join(process.cwd(), "site", "public", "data", "betting_performance"),
    path.join(process.cwd(), "public", "data", "betting_performance"),
    path.join(__dirname, "..", "..", "artifacts", "market_bet_profitability_v1"),
    path.join(__dirname, "..", "..", "..", "artifacts", "market_bet_profitability_v1"),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return candidates[0];
}

export function readBettingPerformancePayload(): BettingPerformancePayload | null {
  const root = getBettingPerformanceRoot();
  if (!fs.existsSync(root)) return null;
  const overall = readCsvRows<ProfitabilityRow>(path.join(root, "overall_profitability.csv"));
  const bySeason = readCsvRows<BySeasonProfitabilityRow>(path.join(root, "by_season_profitability.csv"));
  const signalDrivers = readCsvRows<SignalDriverRow>(path.join(root, "signal_driver_profitability.csv"));
  const robustness = readCsvRows<RobustnessRow>(path.join(root, "robustness_summary.csv"));
  const tabMapping = readCsvRows<BettingTabMappingRow>(path.join(root, "betting_tab_mapping_profitability.csv"));
  const memoPath = path.join(root, "memo.md");
  const memo = fs.existsSync(memoPath) ? fs.readFileSync(memoPath, "utf-8") : null;
  if (!overall.length) return null;
  return {
    overall,
    bySeason,
    signalDrivers,
    robustness,
    tabMapping,
    memo,
  };
}
