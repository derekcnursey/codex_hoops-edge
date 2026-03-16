import { NcaaOddsData, formatRoundOdds } from "./ncaaOdds";

const HARD_ROCK_TITLE_FEED_URL =
  "https://hardrock-hub-web.isportgenius.com.au/seo/basketball/ncaab/champion";
const HARD_ROCK_NCAAM_PAGE_URL = "https://www.hardrock.bet/sportsbook/basketball/ncaam/";

type HardRockFeedRow = {
  team: string;
  odds: string;
};

export type HardRockComparisonRow = {
  team: string;
  hrbTeamName: string;
  seed: number;
  region: string | null;
  hrbChampOdds: string;
  hrbChampProb: number;
  modelChampProb: number;
  modelChampOdds: string | null;
  modelFinalFourProb: number;
  modelFinalFourOdds: string | null;
  deltaPctPoints: number;
};

type HardRockRegionWinnerInput = {
  region: string;
  team: string;
  odds: string;
};

export type HardRockRegionWinnerRow = {
  region: string;
  team: string;
  hrbTeamName: string;
  seed: number;
  hrbRegionOdds: string;
  hrbRegionProb: number;
  hrbRegionFairProb: number;
  modelRegionProb: number;
  modelRegionOdds: string | null;
  deltaPctPoints: number;
};

export type HardRockRegionWinnerReport = {
  source: "manual_snapshot";
  snapshotLabel: string;
  note: string;
  regionHoldPct: Record<string, number>;
  rows: HardRockRegionWinnerRow[];
  topOverlays: HardRockRegionWinnerRow[];
  topUnderlays: HardRockRegionWinnerRow[];
  topFavorites: HardRockRegionWinnerRow[];
  unmatchedTeams: string[];
};

export type HardRockComparisonData = {
  fetchedAt: string;
  sourceUrl: string;
  sportsbookPageUrl: string;
  status: "live" | "unavailable";
  note: string;
  rows: HardRockComparisonRow[];
  topOverlays: HardRockComparisonRow[];
  topUnderlays: HardRockComparisonRow[];
  topByHardRock: HardRockComparisonRow[];
  unmatchedTeams: string[];
  matchedCount: number;
  finalFourMarketStatus: "unavailable_public_feed";
  regionWinnerReport: HardRockRegionWinnerReport | null;
};

const TEAM_ALIASES: Record<string, string> = {
  "connecticut huskies": "UConn",
  "penn quakers": "Pennsylvania",
  "hawaii rainbow warriors": "Hawai'i",
  "liu sharks": "Long Island University",
  "miami redhawks": "Miami (OH)",
  "mcneese state cowboys": "McNeese",
  "saint marys gaels": "Saint Mary's",
  "saint johns red storm": "St. John's",
  "st johns red storm": "St. John's",
  "saint louis billikens": "Saint Louis",
  "southern methodist university mustangs": "SMU",
  "virginia commonwealth rams": "VCU",
  "brigham young cougars": "BYU",
  "miami florida hurricanes": "Miami",
  "louisville cardinals": "Louisville",
  "texas a m aggies": "Texas A&M",
  "prairie view a m panthers d1": "Prairie View A&M",
  "howard bison d1": "Howard",
  "li u sharks": "Long Island University",
};

const HARD_ROCK_REGION_WINNER_SNAPSHOT: HardRockRegionWinnerInput[] = [
  { region: "East", team: "Duke Blue Devils", odds: "-165" },
  { region: "East", team: "Connecticut Huskies", odds: "+500" },
  { region: "East", team: "Michigan State Spartans", odds: "+700" },
  { region: "East", team: "Saint John's Red Storm", odds: "+900" },
  { region: "East", team: "Kansas Jayhawks", odds: "+1000" },
  { region: "East", team: "Louisville Cardinals", odds: "+1400" },
  { region: "East", team: "UCLA Bruins", odds: "+2250" },
  { region: "East", team: "Ohio State Buckeyes", odds: "+3000" },
  { region: "East", team: "TCU Horned Frogs", odds: "+10000" },
  { region: "East", team: "UCF Knights", odds: "+20000" },
  { region: "East", team: "South Florida Bulls", odds: "+20000" },
  { region: "East", team: "Northern Iowa Panthers", odds: "+20000" },
  { region: "East", team: "California Baptist Lancers", odds: "+50000" },
  { region: "East", team: "North Dakota State Bison", odds: "+50000" },
  { region: "East", team: "Furman Paladins", odds: "+50000" },
  { region: "East", team: "Siena Saints", odds: "+50000" },
  { region: "South", team: "Florida Gators", odds: "+125" },
  { region: "South", team: "Houston Cougars", odds: "+200" },
  { region: "South", team: "Illinois Fighting Illini", odds: "+325" },
  { region: "South", team: "Vanderbilt Commodores", odds: "+1200" },
  { region: "South", team: "Nebraska Cornhuskers", odds: "+1400" },
  { region: "South", team: "North Carolina Tar Heels", odds: "+4000" },
  { region: "South", team: "Iowa Hawkeyes", odds: "+5000" },
  { region: "South", team: "Saint Mary's Gaels", odds: "+7500" },
  { region: "South", team: "Clemson Tigers", odds: "+7500" },
  { region: "South", team: "Texas A&M Aggies", odds: "+10000" },
  { region: "South", team: "Virginia Commonwealth Rams", odds: "+15000" },
  { region: "South", team: "McNeese State Cowboys", odds: "+25000" },
  { region: "South", team: "Troy Trojans", odds: "+25000" },
  { region: "South", team: "Pennsylvania Quakers", odds: "+25000" },
  { region: "South", team: "Idaho Vandals", odds: "+25000" },
  { region: "South", team: "Lehigh Mountain Hawks", odds: "+25000" },
  { region: "South", team: "Prairie View A&M Panthers (D1)", odds: "+25000" },
  { region: "Midwest", team: "Michigan Wolverines", odds: "-155" },
  { region: "Midwest", team: "Iowa State Cyclones", odds: "+300" },
  { region: "Midwest", team: "Virginia Cavaliers", odds: "+800" },
  { region: "Midwest", team: "Tennessee Volunteers", odds: "+1400" },
  { region: "Midwest", team: "Texas Tech Red Raiders", odds: "+1750" },
  { region: "Midwest", team: "Alabama Crimson Tide", odds: "+2250" },
  { region: "Midwest", team: "Kentucky Wildcats", odds: "+4000" },
  { region: "Midwest", team: "Georgia Bulldogs", odds: "+6000" },
  { region: "Midwest", team: "Santa Clara Broncos", odds: "+7500" },
  { region: "Midwest", team: "Saint Louis Billikens", odds: "+10000" },
  { region: "Midwest", team: "Southern Methodist University Mustangs", odds: "+20000" },
  { region: "Midwest", team: "Akron Zips", odds: "+20000" },
  { region: "Midwest", team: "Miami Ohio Redhawks", odds: "+25000" },
  { region: "Midwest", team: "Hofstra Pride", odds: "+25000" },
  { region: "Midwest", team: "Wright State Raiders", odds: "+25000" },
  { region: "Midwest", team: "Tennessee State Tigers", odds: "+25000" },
  { region: "Midwest", team: "Howard Bison (D1)", odds: "+25000" },
  { region: "Midwest", team: "UMBC Retrievers", odds: "+25000" },
  { region: "West", team: "Arizona Wildcats", odds: "-155" },
  { region: "West", team: "Purdue Boilermakers", odds: "+400" },
  { region: "West", team: "Gonzaga Bulldogs", odds: "+500" },
  { region: "West", team: "Arkansas Razorbacks", odds: "+1000" },
  { region: "West", team: "Wisconsin Badgers", odds: "+1600" },
  { region: "West", team: "Brigham Young Cougars", odds: "+3000" },
  { region: "West", team: "Miami Florida Hurricanes", odds: "+5000" },
  { region: "West", team: "Missouri Tigers", odds: "+6000" },
  { region: "West", team: "Villanova Wildcats", odds: "+6000" },
  { region: "West", team: "Utah State Aggies", odds: "+7500" },
  { region: "West", team: "North Carolina State Wolfpack", odds: "+10000" },
  { region: "West", team: "Texas Longhorns", odds: "+10000" },
  { region: "West", team: "High Point Panthers", odds: "+25000" },
  { region: "West", team: "Hawaii Rainbow Warriors", odds: "+25000" },
  { region: "West", team: "Kennesaw State Owls", odds: "+25000" },
  { region: "West", team: "Queens University Royals", odds: "+25000" },
  { region: "West", team: "LIU Sharks", odds: "+25000" },
];

function normalizeTeamName(value: string): string {
  return value
    .toLowerCase()
    .replace(/&amp;/g, "&")
    .replace(/[’']/g, "")
    .replace(/saint/g, "st")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function americanToProb(odds: string): number | null {
  const parsed = Number(odds.replace("+", ""));
  if (!Number.isFinite(parsed) || parsed === 0) return null;
  return parsed > 0 ? 100 / (parsed + 100) : -parsed / (-parsed + 100);
}

function extractFeedRows(html: string): HardRockFeedRow[] {
  const rows = (html.match(/<tr>[\s\S]*?<\/tr>/g) ?? []).slice(1);
  return rows
    .map((row) => {
      const team = row.match(/data-first="([^"]+)"/)?.[1]?.trim() ?? null;
      const odds = row.match(/data-odds="([+-]\d+)"/)?.[1] ?? null;
      if (!team || !odds) return null;
      return { team, odds };
    })
    .filter((row): row is HardRockFeedRow => row !== null);
}

function matchTeam(team: string, ncaaData: NcaaOddsData) {
  const normalized = normalizeTeamName(team);
  const aliased = TEAM_ALIASES[normalized];
  if (aliased) {
    return ncaaData.rows.find((row) => row.team === aliased) ?? null;
  }

  const exact = ncaaData.rows.find(
    (row) => normalizeTeamName(row.team) === normalized,
  );
  if (exact) return exact;

  const partialMatches = ncaaData.rows
    .map((row) => ({ row, normalized: normalizeTeamName(row.team) }))
    .filter(
      ({ normalized: local }) =>
        normalized.startsWith(`${local} `) ||
        normalized.endsWith(` ${local}`) ||
        normalized.includes(` ${local} `),
    )
    .sort((a, b) => b.normalized.length - a.normalized.length);

  return partialMatches[0]?.row ?? null;
}

function buildComparisonRows(
  ncaaData: NcaaOddsData,
  feedRows: HardRockFeedRow[],
): Pick<
  HardRockComparisonData,
  "rows" | "topOverlays" | "topUnderlays" | "topByHardRock" | "unmatchedTeams" | "matchedCount"
> {
  const rows: HardRockComparisonRow[] = [];
  const unmatchedTeams: string[] = [];

  for (const feedRow of feedRows) {
    const matchedTeam = matchTeam(feedRow.team, ncaaData);
    if (!matchedTeam) {
      unmatchedTeams.push(feedRow.team);
      continue;
    }
    const hrbChampProb = americanToProb(feedRow.odds);
    if (hrbChampProb == null) continue;

    const modelChampProb = matchedTeam.roundProbabilities.champion;
    const modelFinalFourProb = matchedTeam.roundProbabilities["final-four"];
    rows.push({
      team: matchedTeam.team,
      hrbTeamName: feedRow.team,
      seed: matchedTeam.seed,
      region: matchedTeam.region,
      hrbChampOdds: feedRow.odds,
      hrbChampProb,
      modelChampProb,
      modelChampOdds: formatRoundOdds(modelChampProb),
      modelFinalFourProb,
      modelFinalFourOdds: formatRoundOdds(modelFinalFourProb),
      deltaPctPoints: (modelChampProb - hrbChampProb) * 100,
    });
  }

  rows.sort((a, b) => b.deltaPctPoints - a.deltaPctPoints);
  return {
    rows,
    topOverlays: rows.slice(0, 10),
    topUnderlays: [...rows].sort((a, b) => a.deltaPctPoints - b.deltaPctPoints).slice(0, 10),
    topByHardRock: [...rows].sort((a, b) => b.hrbChampProb - a.hrbChampProb).slice(0, 10),
    unmatchedTeams,
    matchedCount: rows.length,
  };
}

function buildRegionWinnerReport(
  ncaaData: NcaaOddsData,
): HardRockRegionWinnerReport {
  const rows: HardRockRegionWinnerRow[] = [];
  const unmatchedTeams: string[] = [];
  const regionHoldRaw = new Map<string, number>();

  for (const input of HARD_ROCK_REGION_WINNER_SNAPSHOT) {
    const matchedTeam = matchTeam(input.team, ncaaData);
    const hrbRegionProb = americanToProb(input.odds);
    if (!matchedTeam || hrbRegionProb == null) {
      unmatchedTeams.push(input.team);
      continue;
    }
    regionHoldRaw.set(input.region, (regionHoldRaw.get(input.region) ?? 0) + hrbRegionProb);
    rows.push({
      region: input.region,
      team: matchedTeam.team,
      hrbTeamName: input.team,
      seed: matchedTeam.seed,
      hrbRegionOdds: input.odds,
      hrbRegionProb,
      hrbRegionFairProb: 0,
      modelRegionProb: matchedTeam.roundProbabilities["final-four"],
      modelRegionOdds: formatRoundOdds(matchedTeam.roundProbabilities["final-four"]),
      deltaPctPoints: 0,
    });
  }

  const regionHoldPct: Record<string, number> = {};
  for (const [region, totalProb] of regionHoldRaw.entries()) {
    regionHoldPct[region] = (totalProb - 1) * 100;
  }

  const finalizedRows = rows.map((row) => {
    const holdTotal = regionHoldRaw.get(row.region) ?? 1;
    const hrbRegionFairProb = holdTotal > 0 ? row.hrbRegionProb / holdTotal : row.hrbRegionProb;
    return {
      ...row,
      hrbRegionFairProb,
      deltaPctPoints: (row.modelRegionProb - hrbRegionFairProb) * 100,
    };
  });

  return {
    source: "manual_snapshot",
    snapshotLabel: "Manual Hard Rock Region Winner snapshot · March 16, 2026",
    note:
      "Region Winner prices were manually entered from the current Hard Rock board you provided. Comparison uses no-vig fair probability by region against Hoops Edge Final Four probability.",
    regionHoldPct,
    rows: finalizedRows,
    topOverlays: [...finalizedRows]
      .sort((a, b) => b.deltaPctPoints - a.deltaPctPoints)
      .slice(0, 10),
    topUnderlays: [...finalizedRows]
      .sort((a, b) => a.deltaPctPoints - b.deltaPctPoints)
      .slice(0, 10),
    topFavorites: [...finalizedRows]
      .sort((a, b) => b.hrbRegionFairProb - a.hrbRegionFairProb)
      .slice(0, 12),
    unmatchedTeams,
  };
}

export async function fetchHardRockComparisonData(
  ncaaData: NcaaOddsData | null,
): Promise<HardRockComparisonData | null> {
  if (!ncaaData) return null;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 4000);

  try {
    const response = await fetch(HARD_ROCK_TITLE_FEED_URL, {
      signal: controller.signal,
      headers: {
        "User-Agent": "Mozilla/5.0",
      },
      cache: "no-store",
    });
    if (!response.ok) {
      return {
        fetchedAt: new Date().toISOString(),
        sourceUrl: HARD_ROCK_TITLE_FEED_URL,
        sportsbookPageUrl: HARD_ROCK_NCAAM_PAGE_URL,
        status: "unavailable",
        note: `Hard Rock title feed returned HTTP ${response.status}.`,
        rows: [],
        topOverlays: [],
        topUnderlays: [],
        topByHardRock: [],
        unmatchedTeams: [],
        matchedCount: 0,
        finalFourMarketStatus: "unavailable_public_feed",
        regionWinnerReport: buildRegionWinnerReport(ncaaData),
      };
    }

    const html = await response.text();
    const feedRows = extractFeedRows(html);
    const comparison = buildComparisonRows(ncaaData, feedRows);
    return {
      fetchedAt: new Date().toISOString(),
      sourceUrl: HARD_ROCK_TITLE_FEED_URL,
      sportsbookPageUrl: HARD_ROCK_NCAAM_PAGE_URL,
      status: "live",
      note:
        "Hard Rock's public college-basketball page lists 'To Make the Final Four', but no public team-by-team Final Four odds feed was exposed at fetch time. Title prices below are live from Hard Rock's public championship feed.",
      ...comparison,
      finalFourMarketStatus: "unavailable_public_feed",
      regionWinnerReport: buildRegionWinnerReport(ncaaData),
    };
  } catch (error) {
    const reason =
      error instanceof Error ? error.message : "Unknown Hard Rock fetch failure";
    return {
      fetchedAt: new Date().toISOString(),
      sourceUrl: HARD_ROCK_TITLE_FEED_URL,
      sportsbookPageUrl: HARD_ROCK_NCAAM_PAGE_URL,
      status: "unavailable",
      note: `Hard Rock title feed unavailable: ${reason}.`,
      rows: [],
      topOverlays: [],
      topUnderlays: [],
      topByHardRock: [],
      unmatchedTeams: [],
      matchedCount: 0,
      finalFourMarketStatus: "unavailable_public_feed",
      regionWinnerReport: buildRegionWinnerReport(ncaaData),
    };
  } finally {
    clearTimeout(timeout);
  }
}
