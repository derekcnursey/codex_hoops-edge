import { CSSProperties, useMemo, useState } from "react";
import {
  HardRockComparisonData,
  HardRockComparisonRow,
  HardRockRegionWinnerRow,
} from "../../lib/bracket/hardRockComparison";
import { formatRoundOdds, NcaaOddsData, NcaaOddsRoundKey } from "../../lib/bracket/ncaaOdds";

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

const ROUND_COLUMNS: Array<{ key: NcaaOddsRoundKey; label: string }> = [
  { key: "round-of-64", label: "R64" },
  { key: "round-of-32", label: "R32" },
  { key: "sweet-16", label: "S16" },
  { key: "elite-8", label: "E8" },
  { key: "final-four", label: "F4" },
  { key: "national-championship", label: "Title" },
  { key: "champion", label: "Champ" },
];

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function deltaColor(delta: number): string {
  if (delta >= 1) return "#15803d";
  if (delta > 0) return "#16a34a";
  if (delta <= -1) return "#b91c1c";
  if (delta < 0) return "#dc2626";
  return "#475569";
}

function ComparisonSummaryCard({
  label,
  primary,
  secondary,
}: {
  label: string;
  primary: string;
  secondary: string;
}) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 12,
        padding: "14px 16px",
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      }}
    >
      <div style={{ ...mono, fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700, color: "#0f172a", marginBottom: 6 }}>
        {primary}
      </div>
      <div style={{ ...mono, fontSize: 11, color: "#64748b", lineHeight: 1.5 }}>{secondary}</div>
    </div>
  );
}

function TopList({
  title,
  rows,
}: {
  title: string;
  rows: HardRockComparisonRow[];
}) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 12,
        overflow: "hidden",
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      }}
    >
      <div
        style={{
          padding: "14px 16px",
          borderBottom: "1px solid #f1f5f9",
          fontWeight: 700,
          color: "#0f172a",
        }}
      >
        {title}
      </div>
      <div style={{ display: "flex", flexDirection: "column" }}>
        {rows.map((row) => (
          <div
            key={`${title}-${row.team}`}
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1fr) auto",
              gap: 12,
              padding: "12px 16px",
              borderBottom: "1px solid #f8fafc",
            }}
          >
            <div>
              <div style={{ fontWeight: 600, color: "#0f172a" }}>
                ({row.seed}) {row.team}
              </div>
              <div style={{ ...mono, fontSize: 11, color: "#64748b" }}>
                HRB {row.hrbChampOdds} · Model {row.modelChampOdds ?? "--"}
              </div>
            </div>
            <div
              style={{
                ...mono,
                fontWeight: 700,
                color: deltaColor(row.deltaPctPoints),
                whiteSpace: "nowrap",
              }}
            >
              {row.deltaPctPoints > 0 ? "+" : ""}
              {row.deltaPctPoints.toFixed(2)} pp
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function RegionTopList({
  title,
  rows,
}: {
  title: string;
  rows: HardRockRegionWinnerRow[];
}) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 12,
        overflow: "hidden",
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      }}
    >
      <div
        style={{
          padding: "14px 16px",
          borderBottom: "1px solid #f1f5f9",
          fontWeight: 700,
          color: "#0f172a",
        }}
      >
        {title}
      </div>
      <div style={{ display: "flex", flexDirection: "column" }}>
        {rows.map((row) => (
          <div
            key={`${title}-${row.region}-${row.team}`}
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1fr) auto",
              gap: 12,
              padding: "12px 16px",
              borderBottom: "1px solid #f8fafc",
            }}
          >
            <div>
              <div style={{ fontWeight: 600, color: "#0f172a" }}>
                {row.region} · ({row.seed}) {row.team}
              </div>
              <div style={{ ...mono, fontSize: 11, color: "#64748b" }}>
                HRB {row.hrbRegionOdds} fair {formatPercent(row.hrbRegionFairProb)} · Model{" "}
                {row.modelRegionOdds ?? "--"}
              </div>
            </div>
            <div
              style={{
                ...mono,
                fontWeight: 700,
                color: deltaColor(row.deltaPctPoints),
                whiteSpace: "nowrap",
              }}
            >
              {row.deltaPctPoints > 0 ? "+" : ""}
              {row.deltaPctPoints.toFixed(2)} pp
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function MarchAnalysisTab({
  ncaaData,
  hardRockReport,
}: {
  ncaaData: NcaaOddsData | null;
  hardRockReport: HardRockComparisonData | null;
}) {
  const [search, setSearch] = useState("");
  const [region, setRegion] = useState<string>("all");
  const [sortKey, setSortKey] = useState<NcaaOddsRoundKey>("champion");

  const regions = useMemo(
    () =>
      Array.from(new Set((ncaaData?.rows ?? []).map((row) => row.region).filter(Boolean))).sort() as string[],
    [ncaaData?.rows],
  );

  const filteredOddsRows = useMemo(() => {
    if (!ncaaData) return [];
    const query = search.trim().toLowerCase();
    return [...ncaaData.rows]
      .filter((row) => region === "all" || row.region === region)
      .filter((row) => {
        if (!query) return true;
        return (
          row.team.toLowerCase().includes(query) ||
          row.conference.toLowerCase().includes(query) ||
          (row.region ?? "").toLowerCase().includes(query)
        );
      })
      .sort((a, b) => {
        const diff = b.roundProbabilities[sortKey] - a.roundProbabilities[sortKey];
        if (diff !== 0) return diff;
        return a.seed - b.seed || a.team.localeCompare(b.team);
      });
  }, [ncaaData, region, search, sortKey]);

  const filteredComparisonRows = useMemo(() => {
    if (!hardRockReport) return [];
    const query = search.trim().toLowerCase();
    return hardRockReport.rows.filter((row) => {
      if (!query) return true;
      return (
        row.team.toLowerCase().includes(query) ||
        row.hrbTeamName.toLowerCase().includes(query) ||
        (row.region ?? "").toLowerCase().includes(query)
      );
    });
  }, [hardRockReport, search]);

  const filteredRegionWinnerRows = useMemo(() => {
    const rows = hardRockReport?.regionWinnerReport?.rows ?? [];
    const query = search.trim().toLowerCase();
    return rows.filter((row) => {
      if (region !== "all" && row.region !== region) return false;
      if (!query) return true;
      return (
        row.team.toLowerCase().includes(query) ||
        row.hrbTeamName.toLowerCase().includes(query) ||
        row.region.toLowerCase().includes(query)
      );
    });
  }, [hardRockReport, region, search]);

  if (!ncaaData) {
    return (
      <div
        style={{
          padding: 24,
          borderRadius: 12,
          border: "1px solid #e2e8f0",
          background: "#fff",
          color: "#475569",
        }}
      >
        NCAA tournament odds data is unavailable.
      </div>
    );
  }

  const bestOverlay = hardRockReport?.topOverlays[0] ?? null;
  const biggestUnderlay = hardRockReport?.topUnderlays[0] ?? null;
  const bestRegionOverlay = hardRockReport?.regionWinnerReport?.topOverlays[0] ?? null;
  const biggestRegionUnderlay = hardRockReport?.regionWinnerReport?.topUnderlays[0] ?? null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 12,
        }}
      >
        <ComparisonSummaryCard
          label="Method"
          primary="Display ML"
          secondary="Exact bracket advancement odds using display-adjusted spreads for ML conversion."
        />
        <ComparisonSummaryCard
          label="Hard Rock Feed"
          primary={hardRockReport?.status === "live" ? "Live title feed" : "Unavailable"}
          secondary={hardRockReport?.note ?? "No Hard Rock comparison loaded."}
        />
        <ComparisonSummaryCard
          label="Best Overlay"
          primary={bestOverlay ? bestOverlay.team : "--"}
          secondary={
            bestOverlay
              ? `Model ${bestOverlay.modelChampOdds ?? "--"} vs HRB ${bestOverlay.hrbChampOdds}`
              : "No overlay data"
          }
        />
        <ComparisonSummaryCard
          label="Biggest Underlay"
          primary={biggestUnderlay ? biggestUnderlay.team : "--"}
          secondary={
            biggestUnderlay
              ? `Model ${biggestUnderlay.modelChampOdds ?? "--"} vs HRB ${biggestUnderlay.hrbChampOdds}`
              : "No underlay data"
          }
        />
        <ComparisonSummaryCard
          label="Best Region Overlay"
          primary={bestRegionOverlay ? `${bestRegionOverlay.team} (${bestRegionOverlay.region})` : "--"}
          secondary={
            bestRegionOverlay
              ? `Model ${bestRegionOverlay.modelRegionOdds ?? "--"} vs HRB ${bestRegionOverlay.hrbRegionOdds}`
              : "No region data"
          }
        />
        <ComparisonSummaryCard
          label="Biggest Region Underlay"
          primary={
            biggestRegionUnderlay
              ? `${biggestRegionUnderlay.team} (${biggestRegionUnderlay.region})`
              : "--"
          }
          secondary={
            biggestRegionUnderlay
              ? `Model ${biggestRegionUnderlay.modelRegionOdds ?? "--"} vs HRB ${biggestRegionUnderlay.hrbRegionOdds}`
              : "No region data"
          }
        />
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 10,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {["all", ...regions].map((value) => (
            <button
              key={value}
              onClick={() => setRegion(value)}
              style={{
                ...mono,
                padding: "6px 14px",
                border: `1px solid ${region === value ? "#0f172a" : "#e2e8f0"}`,
                borderRadius: 6,
                fontSize: 12,
                fontWeight: region === value ? 600 : 400,
                background: region === value ? "#0f172a" : "#fff",
                color: region === value ? "#fff" : "#64748b",
                cursor: "pointer",
              }}
            >
              {value === "all" ? "All Regions" : value}
            </button>
          ))}
        </div>
        <input
          type="text"
          placeholder="Search team, region, conference..."
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          style={{
            ...mono,
            width: 300,
            maxWidth: "100%",
            padding: "6px 10px",
            border: "1px solid #e2e8f0",
            borderRadius: 6,
            fontSize: 13,
            outline: "none",
            background: "#fff",
            color: "#334155",
          }}
        />
      </div>

      {hardRockReport && (
        <>
          {hardRockReport.regionWinnerReport && (
            <>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                  gap: 12,
                }}
              >
                <RegionTopList
                  title="Top Region Overlays vs Hard Rock"
                  rows={hardRockReport.regionWinnerReport.topOverlays}
                />
                <RegionTopList
                  title="Top Region Underlays vs Hard Rock"
                  rows={hardRockReport.regionWinnerReport.topUnderlays}
                />
              </div>

              <div
                style={{
                  background: "#fff",
                  border: "1px solid #e2e8f0",
                  borderRadius: 12,
                  overflow: "hidden",
                  boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
                }}
              >
                <div
                  style={{
                    padding: "14px 16px",
                    borderBottom: "1px solid #f1f5f9",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 12,
                    flexWrap: "wrap",
                  }}
                >
                  <div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>
                      Hard Rock Region Winner Comparison
                    </div>
                    <div style={{ ...mono, fontSize: 11, color: "#64748b", marginTop: 4 }}>
                      {hardRockReport.regionWinnerReport.snapshotLabel}
                    </div>
                    <div style={{ ...mono, fontSize: 11, color: "#94a3b8", marginTop: 4 }}>
                      {hardRockReport.regionWinnerReport.note}
                    </div>
                  </div>
                  <div style={{ ...mono, fontSize: 11, color: "#64748b", textAlign: "right" }}>
                    {Object.entries(hardRockReport.regionWinnerReport.regionHoldPct).map(
                      ([regionName, holdPct], index, entries) => (
                        <span key={regionName}>
                          {regionName} hold {holdPct.toFixed(2)}%
                          {index < entries.length - 1 ? " · " : ""}
                        </span>
                      ),
                    )}
                  </div>
                </div>

                <div style={{ overflowX: "auto" }}>
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      fontSize: 13,
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    <thead>
                      <tr style={{ background: "#fafbfc", borderBottom: "1px solid #e2e8f0" }}>
                        <th style={thStyle}>Region</th>
                        <th style={thStyle}>Seed</th>
                        <th style={{ ...thStyle, textAlign: "left" }}>Team</th>
                        <th style={thStyle}>HRB Region</th>
                        <th style={thStyle}>HRB Fair %</th>
                        <th style={thStyle}>Model F4</th>
                        <th style={thStyle}>Delta</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredRegionWinnerRows.map((row) => (
                        <tr
                          key={`${row.region}-${row.team}`}
                          style={{ borderBottom: "1px solid #f8fafc" }}
                        >
                          <td style={{ ...tdStyle, ...mono, color: "#64748b" }}>{row.region}</td>
                          <td
                            style={{
                              ...tdStyle,
                              ...mono,
                              color: "#64748b",
                              fontWeight: 600,
                            }}
                          >
                            {row.seed}
                          </td>
                          <td style={{ ...tdStyle, textAlign: "left" }}>
                            <div style={{ fontWeight: 600, color: "#0f172a" }}>{row.team}</div>
                            <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                              {row.hrbTeamName}
                            </div>
                          </td>
                          <td style={tdStyle}>
                            <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                              {row.hrbRegionOdds}
                            </div>
                            <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                              {formatPercent(row.hrbRegionProb)}
                            </div>
                          </td>
                          <td style={tdStyle}>
                            <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                              {formatPercent(row.hrbRegionFairProb)}
                            </div>
                          </td>
                          <td style={tdStyle}>
                            <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                              {row.modelRegionOdds ?? "--"}
                            </div>
                            <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                              {formatPercent(row.modelRegionProb)}
                            </div>
                          </td>
                          <td style={tdStyle}>
                            <div
                              style={{
                                ...mono,
                                fontWeight: 700,
                                color: deltaColor(row.deltaPctPoints),
                              }}
                            >
                              {row.deltaPctPoints > 0 ? "+" : ""}
                              {row.deltaPctPoints.toFixed(2)} pp
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: 12,
            }}
          >
            <TopList title="Top Overlays vs Hard Rock" rows={hardRockReport.topOverlays} />
            <TopList title="Top Underlays vs Hard Rock" rows={hardRockReport.topUnderlays} />
          </div>

          <div
            style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 12,
              overflow: "hidden",
              boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
            }}
          >
            <div
              style={{
                padding: "14px 16px",
                borderBottom: "1px solid #f1f5f9",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 12,
                flexWrap: "wrap",
              }}
            >
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>
                  Hard Rock Championship Comparison
                </div>
                <div style={{ ...mono, fontSize: 11, color: "#64748b", marginTop: 4 }}>
                  {hardRockReport.matchedCount} matched teams · fetched {new Date(hardRockReport.fetchedAt).toLocaleString()}
                </div>
              </div>
              <div style={{ ...mono, fontSize: 11, color: "#64748b" }}>
                <a href={hardRockReport.sourceUrl} target="_blank" rel="noreferrer">
                  title feed
                </a>
                {" · "}
                <a href={hardRockReport.sportsbookPageUrl} target="_blank" rel="noreferrer">
                  sportsbook page
                </a>
              </div>
            </div>

            <div style={{ overflowX: "auto" }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: 13,
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                <thead>
                  <tr style={{ background: "#fafbfc", borderBottom: "1px solid #e2e8f0" }}>
                    <th style={thStyle}>Seed</th>
                    <th style={{ ...thStyle, textAlign: "left" }}>Team</th>
                    <th style={thStyle}>HRB Champ</th>
                    <th style={thStyle}>Model Champ</th>
                    <th style={thStyle}>Delta</th>
                    <th style={thStyle}>Model F4</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredComparisonRows.map((row) => (
                    <tr key={row.team} style={{ borderBottom: "1px solid #f8fafc" }}>
                      <td style={{ ...tdStyle, ...mono, color: "#64748b", fontWeight: 600 }}>
                        {row.seed}
                      </td>
                      <td style={{ ...tdStyle, textAlign: "left" }}>
                        <div style={{ fontWeight: 600, color: "#0f172a" }}>{row.team}</div>
                        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                          {row.region ?? "--"}
                        </div>
                      </td>
                      <td style={tdStyle}>
                        <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                          {row.hrbChampOdds}
                        </div>
                        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                          {formatPercent(row.hrbChampProb)}
                        </div>
                      </td>
                      <td style={tdStyle}>
                        <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                          {row.modelChampOdds ?? "--"}
                        </div>
                        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                          {formatPercent(row.modelChampProb)}
                        </div>
                      </td>
                      <td style={tdStyle}>
                        <div
                          style={{
                            ...mono,
                            fontWeight: 700,
                            color: deltaColor(row.deltaPctPoints),
                          }}
                        >
                          {row.deltaPctPoints > 0 ? "+" : ""}
                          {row.deltaPctPoints.toFixed(2)} pp
                        </div>
                      </td>
                      <td style={tdStyle}>
                        <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                          {row.modelFinalFourOdds ?? "--"}
                        </div>
                        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                          {formatPercent(row.modelFinalFourProb)}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      <div
        style={{
          background: "#fff",
          border: "1px solid #e2e8f0",
          borderRadius: 12,
          overflow: "hidden",
          boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
        }}
      >
        <div
          style={{
            padding: "14px 16px",
            borderBottom: "1px solid #f1f5f9",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>
              Hoops Edge Round Advancement Odds
            </div>
            <div style={{ ...mono, fontSize: 11, color: "#64748b", marginTop: 4 }}>
              Exact bracket advancement odds using the NCAA display-adjusted spread path for the ML calc.
            </div>
          </div>
        </div>

        <div style={{ overflowX: "auto" }}>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: 13,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            <thead>
              <tr style={{ background: "#fafbfc", borderBottom: "1px solid #e2e8f0" }}>
                <th style={thStyle}>Seed</th>
                <th style={{ ...thStyle, textAlign: "left" }}>Team</th>
                <th style={thStyle}>Region</th>
                {ROUND_COLUMNS.map((column) => (
                  <th key={column.key} style={thStyle}>
                    <button
                      onClick={() => setSortKey(column.key)}
                      style={{
                        ...mono,
                        fontSize: 11,
                        fontWeight: sortKey === column.key ? 700 : 600,
                        color: sortKey === column.key ? "#0f172a" : "#64748b",
                        background: "transparent",
                        border: "none",
                        cursor: "pointer",
                        textTransform: "uppercase",
                        letterSpacing: "0.05em",
                      }}
                    >
                      {column.label}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredOddsRows.map((row) => (
                <tr key={row.teamId} style={{ borderBottom: "1px solid #f8fafc" }}>
                  <td style={{ ...tdStyle, ...mono, fontWeight: 600, color: "#64748b" }}>
                    {row.seed}
                  </td>
                  <td style={{ ...tdStyle, textAlign: "left" }}>
                    <div style={{ fontWeight: 600, color: "#0f172a" }}>{row.team}</div>
                    <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>{row.conference}</div>
                  </td>
                  <td style={{ ...tdStyle, ...mono, color: "#64748b" }}>{row.region ?? "--"}</td>
                  {ROUND_COLUMNS.map((column) => {
                    const probability = row.roundProbabilities[column.key];
                    return (
                      <td key={column.key} style={tdStyle}>
                        <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                          {formatPercent(probability)}
                        </div>
                        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                          {formatRoundOdds(probability) ?? "--"}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

const thStyle: CSSProperties = {
  ...mono,
  padding: "8px 12px",
  fontSize: 11,
  fontWeight: 600,
  color: "#64748b",
  textAlign: "center",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  whiteSpace: "nowrap",
};

const tdStyle: CSSProperties = {
  padding: "10px 12px",
  textAlign: "center",
  whiteSpace: "nowrap",
};
