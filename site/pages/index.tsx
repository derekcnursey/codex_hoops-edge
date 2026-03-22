import { GetServerSideProps } from "next";
import { CSSProperties, useMemo, useState } from "react";
import Layout from "../components/Layout";
import {
  PredictionRow,
  displayTeam,
  formatAmericanOddsFromProb,
  getSiteHomeWinProbFromValues,
} from "../lib/data";
import {
  getLatestPredictionFile,
  getPredictionRowsByFilename,
  getPredictionRowsByDate,
  todayET,
} from "../lib/server-data";
import { getTeamRank, getTeamRankMapForDate } from "../lib/team-rankings";

type RankedPredictionRow = PredictionRow & {
  away_team_rank: number | null;
  home_team_rank: number | null;
};

type PredictionSurface = "he" | "torvik";

type HomeProps = {
  todayDate: string | null;
  todayRows: RankedPredictionRow[];
  tomorrowDate: string;
  tomorrowRows: RankedPredictionRow[];
};

function nextDateIso(dateStr: string): string {
  const [year, month, day] = dateStr.split("-").map(Number);
  const dt = new Date(Date.UTC(year, month - 1, day));
  dt.setUTCDate(dt.getUTCDate() + 1);
  return dt.toISOString().slice(0, 10);
}

function rankRows(date: string, rows: PredictionRow[]): RankedPredictionRow[] {
  const teamRanks = getTeamRankMapForDate(date);
  return rows.map((row) => ({
    ...row,
    away_team_rank: getTeamRank(str(row.away_team), teamRanks),
    home_team_rank: getTeamRank(str(row.home_team), teamRanks),
  }));
}

export const getServerSideProps: GetServerSideProps<HomeProps> = async () => {
  const today = todayET();
  const tomorrow = nextDateIso(today);
  const latest = getLatestPredictionFile();
  if (!latest) {
    return {
      props: {
        todayDate: null,
        todayRows: [],
        tomorrowDate: tomorrow,
        tomorrowRows: [],
      },
    };
  }
  const todayRows = rankRows(latest.date, getPredictionRowsByFilename(latest.filename));
  const tomorrowRows = rankRows(tomorrow, getPredictionRowsByDate(tomorrow));
  return {
    props: {
      todayDate: latest.date,
      todayRows,
      tomorrowDate: tomorrow,
      tomorrowRows,
    },
  };
};

/* -- helpers -- */

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace"
};

function formatSpread(v: number): string {
  return v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1);
}

function formatDateDisplay(dateStr: string): string {
  const [year, month, day] = dateStr.split("-");
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
  ];
  return `${months[Number(month) - 1]} ${Number(day)}, ${year}`;
}

function str(v: unknown): string {
  return typeof v === "string" ? v : String(v ?? "");
}

function num(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    return Number.isNaN(n) ? null : n;
  }
  return null;
}

function sourceMuHome(row: PredictionRow, source: PredictionSurface): number | null {
  const sourceKey = `model_mu_home_${source}`;
  const fallback =
    source === "he"
      ? row.model_mu_home_he ?? row.model_mu_home_team_ab_internal ?? row.model_mu_home
      : source === "torvik"
        ? row.model_mu_home_torvik ?? row.model_mu_home
        : null;
  return num((row as Record<string, unknown>)[sourceKey] ?? fallback);
}

function getPickSide(row: PredictionRow, source: PredictionSurface): string {
  const sourceKey = `pick_side_${source}`;
  const fallback = source === "he" ? row.pick_side : null;
  return str((row as Record<string, unknown>)[sourceKey] ?? fallback).toUpperCase();
}

function getPickProbEdge(row: PredictionRow, source: PredictionSurface): number {
  const sourceKey = `pick_prob_edge_${source}`;
  const fallback = source === "he" ? row.pick_prob_edge : null;
  return num((row as Record<string, unknown>)[sourceKey] ?? fallback) ?? 0;
}

function getPickTeam(row: PredictionRow, source: PredictionSurface): string {
  const side = getPickSide(row, source);
  return displayTeam(side === "HOME" ? str(row.home_team) : str(row.away_team));
}

function renderRankedTeam(teamName: string, rank: number | null) {
  return (
    <>
      {rank !== null && (
        <span
          style={{
            ...mono,
            fontSize: 11,
            color: "#64748b",
            marginRight: 2,
          }}
        >
          {rank}
        </span>
      )}
      {displayTeam(teamName)}
    </>
  );
}

function renderMatchupSeparator() {
  return (
    <span
      style={{
        fontSize: 11,
        color: "#64748b",
        margin: "0 4px",
        textTransform: "lowercase",
      }}
    >
      at
    </span>
  );
}

function formatGameTime(row: PredictionRow): string | null {
  const raw = row.start_time ?? row.startDate;
  if (!raw || typeof raw !== "string") return null;
  try {
    const d = new Date(raw);
    if (isNaN(d.getTime())) return null;
    return d.toLocaleTimeString("en-US", {
      timeZone: "America/New_York",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return null;
  }
}

function hasBook(row: PredictionRow): boolean {
  return row.market_spread_home !== null && row.market_spread_home !== undefined && row.market_spread_home !== "";
}

function bookSpread(row: PredictionRow): number | null {
  return num(row.market_spread_home);
}

function modelSpread(row: PredictionRow, source: PredictionSurface): number | null {
  const v = sourceMuHome(row, source);
  return v !== null ? -v : null;
}

function torvikModelSpread(row: PredictionRow): number | null {
  return modelSpread(row, "torvik");
}

function heModelSpread(row: PredictionRow): number | null {
  return modelSpread(row, "he");
}

function avgModelSpread(row: PredictionRow): number | null {
  const he = heModelSpread(row);
  const torvik = torvikModelSpread(row);
  if (he === null || torvik === null) return null;
  return (he + torvik) / 2;
}

function sigma(row: PredictionRow): number | null {
  return num(row.pred_sigma);
}

function edge(row: PredictionRow, source: PredictionSurface): number {
  return getPickProbEdge(row, source);
}

function homeMlFair(row: PredictionRow, source: PredictionSurface): string | null {
  const raw = sourceMuHome(row, source);
  const sigmaValue = sigma(row);
  const startTime = typeof row.start_time === "string" ? row.start_time : typeof row.startDate === "string" ? row.startDate : null;
  const neutralSite = row.neutral_site === true || row.neutral_site === 1 || row.neutralSite === true || row.neutralSite === 1;
  const tournament = typeof row.tournament === "string" ? row.tournament : null;
  const gameType = typeof row.gameType === "string" ? row.gameType : typeof row.game_type === "string" ? row.game_type : null;
  const homeProb = getSiteHomeWinProbFromValues(raw, sigmaValue, startTime, neutralSite, tournament, gameType);
  if (homeProb === null) return null;
  return formatAmericanOddsFromProb(homeProb);
}

function diff(row: PredictionRow, source: PredictionSurface): number | null {
  const m = modelSpread(row, source);
  const b = bookSpread(row);
  if (m === null || b === null) return null;
  return Math.abs(m - b);
}

function pickSpread(row: PredictionRow): number | null {
  const b = bookSpread(row);
  if (b === null) return null;
  return str(row.pick_side).toUpperCase() === "HOME" ? b : -b;
}

/* -- sort -- */

type SortKey = "matchup" | "time" | "book" | "he" | "torvik" | "avg" | "sigma" | "diff" | "edge";

type SortState = { key: SortKey; dir: "asc" | "desc" };

function sortVal(row: PredictionRow, key: SortKey, source: PredictionSurface): string | number {
  switch (key) {
    case "matchup":
      return `${displayTeam(str(row.away_team))} @ ${displayTeam(str(row.home_team))}`;
    case "time": {
      const raw = row.start_time ?? row.startDate;
      if (typeof raw !== "string" || !raw) return Number.POSITIVE_INFINITY;
      const ms = new Date(raw).getTime();
      return Number.isNaN(ms) ? Number.POSITIVE_INFINITY : ms;
    }
    case "book":
      return bookSpread(row) ?? -Infinity;
    case "he":
      return heModelSpread(row) ?? -Infinity;
    case "torvik":
      return torvikModelSpread(row) ?? -Infinity;
    case "avg":
      return avgModelSpread(row) ?? -Infinity;
    case "sigma":
      return sigma(row) ?? -Infinity;
    case "diff":
      return diff(row, source) ?? -Infinity;
    case "edge":
      return edge(row, source);
  }
}

/* -- column defs -- */

const columns: { key: SortKey; label: string; align: "left" | "center" }[] = [
  { key: "matchup", label: "MATCHUP", align: "left" },
  { key: "time", label: "TIME", align: "center" },
  { key: "book", label: "MARKET", align: "center" },
  { key: "he", label: "HE", align: "center" },
  { key: "torvik", label: "TORVIK", align: "center" },
  { key: "avg", label: "AVG", align: "center" },
  { key: "sigma", label: "SIGMA", align: "center" },
  { key: "diff", label: "DIFF", align: "center" },
  { key: "edge", label: "EDGE", align: "center" }
];

/* -- component -- */

export default function Home({ todayDate, todayRows, tomorrowDate, tomorrowRows }: HomeProps) {
  const [activeTab, setActiveTab] = useState<"today" | "tomorrow">("today");
  const [predictionSource, setPredictionSource] = useState<PredictionSurface>("he");
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "edge10">("all");
  const [diffMin, setDiffMin] = useState(0);
  const [sort, setSort] = useState<SortState>({ key: "edge", dir: "desc" });
  const activeDate = activeTab === "today" ? todayDate : tomorrowDate;
  const activeRows = activeTab === "today" ? todayRows : tomorrowRows;
  const tomorrowReady = tomorrowRows.length > 0;
  const title = activeTab === "today" ? "Today\u2019s Picks" : "Tomorrow\u2019s Board";
  const slateLabel = activeTab === "today" ? "Today" : tomorrowReady ? "Tomorrow" : "Tomorrow TBA";

  const maxDiff = useMemo(() => {
    if (!activeRows.length) return 20;
    const diffs = activeRows.map((r) => diff(r, predictionSource)).filter((d): d is number => d !== null);
    return diffs.length > 0 ? Math.ceil(Math.max(...diffs)) : 20;
  }, [activeRows, predictionSource]);

  const tableRows = useMemo(() => {
    let list = [...activeRows];

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter((r) => {
        const a = str(r.away_team).toLowerCase();
        const h = str(r.home_team).toLowerCase();
        const p = getPickTeam(r, predictionSource).toLowerCase();
        return a.includes(q) || h.includes(q) || p.includes(q);
      });
    }

    if (filter === "edge10") {
      list = list.filter((r) => hasBook(r) && edge(r, predictionSource) >= 0.10);
    }

    if (diffMin > 0) {
      list = list.filter((r) => {
        const d = diff(r, predictionSource);
        return d === null || d >= diffMin;
      });
    }

    list.sort((a, b) => {
      const aHas = hasBook(a);
      const bHas = hasBook(b);
      if (aHas !== bHas) return aHas ? -1 : 1;

      const av = sortVal(a, sort.key, predictionSource);
      const bv = sortVal(b, sort.key, predictionSource);
      if (typeof av === "number" && typeof bv === "number") {
        return sort.dir === "asc" ? av - bv : bv - av;
      }
      const cmp = String(av).localeCompare(String(bv));
      return sort.dir === "asc" ? cmp : -cmp;
    });

    return list;
  }, [activeRows, search, filter, diffMin, sort, predictionSource]);

  function handleSort(key: SortKey) {
    setSort((prev) =>
      prev.key === key
        ? { key, dir: prev.dir === "desc" ? "asc" : "desc" }
        : { key, dir: "desc" }
    );
  }

  if (!activeRows.length) {
    return (
      <Layout>
        <div style={{ padding: 24, color: "#94a3b8", textAlign: "center" }}>
          {activeTab === "today" ? "No games found for today." : "Tomorrow TBA."}
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      {/* single wrapper so .content gap doesn't add extra spacing */}
      <div>
        <div
          style={{
            display: "inline-flex",
            gap: 8,
            padding: 4,
            background: "#f8fafc",
            border: "1px solid #e2e8f0",
            borderRadius: 10,
            marginBottom: 16,
          }}
        >
          {([
            { key: "today", label: "Today" },
            { key: "tomorrow", label: tomorrowReady ? "Tomorrow" : "Tomorrow TBA" },
          ] as const).map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => {
                setActiveTab(tab.key);
                setSearch("");
                setFilter("all");
                setDiffMin(0);
              }}
              style={{
                ...mono,
                fontSize: 12,
                fontWeight: 600,
                padding: "7px 12px",
                borderRadius: 8,
                border: "none",
                background: activeTab === tab.key ? "#0f172a" : "transparent",
                color: activeTab === tab.key ? "#fff" : "#475569",
                cursor: "pointer",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* -- Title Row -- */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: 24
          }}
          >
            <h1
            style={{
              fontSize: 24,
              fontWeight: 700,
              letterSpacing: "-0.02em",
              margin: 0,
              color: "#0f172a"
            }}
          >
            {title}
          </h1>
          <span style={{ ...mono, fontSize: 13, color: "#64748b" }}>
            {activeDate ? formatDateDisplay(activeDate) : ""} · {activeRows.length} games
          </span>
        </div>

        <div
          style={{
            display: "inline-flex",
            gap: 8,
            padding: 4,
            background: "#f8fafc",
            border: "1px solid #e2e8f0",
            borderRadius: 10,
            marginBottom: 14,
          }}
        >
          {([
            { key: "he", label: "HE" },
            { key: "torvik", label: "Torvik" },
          ] as const).map((source) => (
            <button
              key={source.key}
              type="button"
              onClick={() => setPredictionSource(source.key)}
              style={{
                ...mono,
                fontSize: 12,
                fontWeight: 600,
                padding: "7px 12px",
                borderRadius: 8,
                border: "none",
                background: predictionSource === source.key ? "#0f172a" : "transparent",
                color: predictionSource === source.key ? "#fff" : "#475569",
                cursor: "pointer",
              }}
            >
              {source.label}
            </button>
          ))}
        </div>

        {/* -- All Games Table -- */}
        <div>
          {/* Controls row */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 10
            }}
          >
            <span style={{ fontSize: 13, fontWeight: 500, color: "#64748b" }}>
              {slateLabel} · Active surface {predictionSource.toUpperCase()}
            </span>

            <input
              type="text"
              placeholder="Search team..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{
                ...mono,
                width: 180,
                padding: "6px 10px",
                border: "1px solid #e2e8f0",
                borderRadius: 6,
                fontSize: 13,
                outline: "none",
                background: "#fff",
                color: "#334155"
              }}
            />

            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ ...mono, fontSize: 10, color: "#94a3b8", fontWeight: 500 }}>DIFF</span>
              <input type="range" min={0} max={maxDiff} step={1} value={diffMin} onChange={(e) => setDiffMin(Number(e.target.value))} style={{ width: 100, accentColor: "#0f172a" }} />
              <span style={{ ...mono, fontSize: 12, fontWeight: 700, color: "#0f172a", minWidth: 30 }}>{diffMin}</span>
            </div>

            <div style={{ display: "flex", gap: 6 }}>
              {(["all", "edge10"] as const).map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => setFilter(f)}
                  style={{
                    ...mono,
                    fontSize: 12,
                    fontWeight: 500,
                    padding: "5px 12px",
                    borderRadius: 6,
                    border: `1px solid ${filter === f ? "#0f172a" : "#e2e8f0"}`,
                    background: filter === f ? "#0f172a" : "#fff",
                    color: filter === f ? "#fff" : "#64748b",
                    cursor: "pointer"
                  }}
                >
                  {f === "all" ? "All" : "Edge \u2265 10%"}
                </button>
              ))}
            </div>
          </div>

          {/* Table container */}
          <div
            style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 10,
              overflow: "hidden",
              boxShadow: "0 1px 3px rgba(0,0,0,0.04)"
            }}
          >
            <div style={{ overflowX: "auto" }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontVariantNumeric: "tabular-nums"
                }}
              >
                <thead>
                  <tr>
                    {columns.map((col) => {
                      const active = sort.key === col.key;
                      return (
                        <th
                          key={col.key}
                          onClick={() => handleSort(col.key)}
                          style={{
                            ...mono,
                            fontSize: 10,
                            fontWeight: 600,
                            letterSpacing: "0.08em",
                            padding: "10px 14px",
                            textAlign: col.align,
                            background: "#fafbfc",
                            color: active ? "#0f172a" : "#64748b",
                            borderBottom: "1px solid #e2e8f0",
                            cursor: "pointer",
                            userSelect: "none",
                            whiteSpace: "nowrap",
                            ...(col.key === "matchup" ? { width: "1%" } : {})
                          }}
                        >
                          {col.label}
                          {active && (
                            <span style={{ marginLeft: 4 }}>
                              {sort.dir === "desc" ? "\u2193" : "\u2191"}
                            </span>
                          )}
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody>
                  {tableRows.length === 0 ? (
                    <tr>
                      <td
                        colSpan={columns.length}
                        style={{
                          padding: 24,
                          textAlign: "center",
                          color: "#94a3b8",
                          borderBottom: "none"
                        }}
                      >
                        No games found
                      </td>
                    </tr>
                  ) : (
                    tableRows.map((row, i) => {
                      const bk = bookSpread(row);
                      const he = heModelSpread(row);
                      const torvik = torvikModelSpread(row);
                      const avg = avgModelSpread(row);
                      const sg = sigma(row);
                      const df = diff(row, predictionSource);
                      const eg = edge(row, predictionSource);
                      const pickSide = getPickSide(row, predictionSource);
                      const hb = hasBook(row);

                      return (
                        <tr
                          key={`${str(row.away_team)}-${str(row.home_team)}-${i}`}
                          style={{
                            borderBottom: "1px solid #f1f5f9",
                            animation: `fadeIn 0.3s ease ${i * 0.02}s both`
                          }}
                        >
                          {/* MATCHUP — picked side is bold */}
                          <td
                            style={{
                              padding: "10px 14px",
                              textAlign: "left",
                              fontSize: 14,
                              color: "#334155",
                              whiteSpace: "nowrap",
                              width: "1%",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            <span style={{ fontWeight: pickSide === "AWAY" ? 700 : 400 }}>
                              {renderRankedTeam(str(row.away_team), row.away_team_rank)}
                            </span>
                            {renderMatchupSeparator()}
                            <span style={{ fontWeight: pickSide === "HOME" ? 700 : 400 }}>
                              {renderRankedTeam(str(row.home_team), row.home_team_rank)}
                              {homeMlFair(row, predictionSource) ? (
                                <span
                                  style={{
                                    ...mono,
                                    marginLeft: 6,
                                    fontSize: 11,
                                    fontWeight: 500,
                                    color: "#64748b"
                                  }}
                                >
                                  ({homeMlFair(row, predictionSource)})
                                </span>
                              ) : null}
                            </span>
                          </td>

                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 12,
                              color: "#64748b",
                              borderBottom: "1px solid #f1f5f9",
                              whiteSpace: "nowrap"
                            }}
                          >
                            {formatGameTime(row) ?? "—"}
                          </td>

                          {/* MARKET */}
                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 13,
                              color: "#334155",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {hb && bk !== null ? formatSpread(bk) : "\u2014"}
                          </td>

                          {/* HE */}
                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 14,
                              fontWeight: 700,
                              color: "#0f172a",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {he !== null ? formatSpread(he) : "\u2014"}
                          </td>

                          {/* TORVIK */}
                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 13,
                              color: "#334155",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {torvik !== null ? formatSpread(torvik) : "\u2014"}
                          </td>

                          {/* AVG */}
                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 13,
                              fontWeight: 600,
                              color: "#334155",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {avg !== null ? formatSpread(avg) : "\u2014"}
                          </td>

                          {/* SIGMA */}
                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 13,
                              color: "#64748b",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {sg !== null ? sg.toFixed(1) : "\u2014"}
                          </td>

                          {/* DIFF */}
                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 13,
                              fontWeight: 600,
                              color: "#334155",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {hb && df !== null ? df.toFixed(1) : "\u2014"}
                          </td>

                          {/* EDGE */}
                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 13,
                              fontWeight: 700,
                              color: hb
                                ? eg >= 0
                                  ? "#16a34a"
                                  : "#dc2626"
                                : "#94a3b8",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {hb
                              ? `${eg >= 0 ? "+" : ""}${(eg * 100).toFixed(1)}%`
                              : "\u2014"}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

      </div>
    </Layout>
  );
}
