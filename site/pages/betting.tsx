import { GetServerSideProps } from "next";
import Link from "next/link";
import { useRouter } from "next/router";
import { CSSProperties } from "react";
import Layout from "../components/Layout";
import {
  InternalBettingPayload,
  InternalBettingRow,
  readInternalBettingPayload,
} from "../lib/internal-betting-data";

type Props = {
  payload: InternalBettingPayload | null;
};

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

export const getServerSideProps: GetServerSideProps<Props> = async (context) => {
  const qDate = typeof context.query.date === "string" ? context.query.date : null;
  return {
    props: {
      payload: readInternalBettingPayload(qDate),
    },
  };
};

function fmtSpread(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "—";
  return v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1);
}

function displayModelSpread(v: number | null): number | null {
  if (v === null || Number.isNaN(v)) return null;
  // Internal report stores home margin; site displays book-style home spread.
  return -v;
}

function fmtPct(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtScore(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "—";
  return v.toFixed(3);
}

function formatDateDisplay(dateStr: string): string {
  const [year, month, day] = dateStr.split("-");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${months[Number(month) - 1]} ${Number(day)}, ${year}`;
}

function formatSliceLabel(slice: string | null): string {
  if (!slice) return "regular";
  return slice
    .replace(/_/g, " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function badgeStyle(kind: "internal" | "caution" | "raw" | "neutral"): CSSProperties {
  switch (kind) {
    case "internal":
      return { background: "rgba(22, 163, 74, 0.10)", color: "#166534", border: "1px solid rgba(22, 163, 74, 0.2)" };
    case "caution":
      return { background: "rgba(245, 158, 11, 0.12)", color: "#92400e", border: "1px solid rgba(245, 158, 11, 0.25)" };
    case "raw":
      return { background: "rgba(37, 99, 235, 0.10)", color: "#1d4ed8", border: "1px solid rgba(37, 99, 235, 0.2)" };
    default:
      return { background: "#f8fafc", color: "#475569", border: "1px solid #e2e8f0" };
  }
}

function contextLabel(row: InternalBettingRow): string {
  const parts: string[] = [];
  if (row.persistence_label) parts.push(row.persistence_label);
  if (row.neutral_site_flag) parts.push("neutral");
  if (row.slice) parts.push(formatSliceLabel(row.slice));
  return parts.join(" · ");
}

function RowCard({ row, mode }: { row: InternalBettingRow; mode: "internal" | "raw" | "caution" }) {
  const disagreementLed = Boolean(row.flagged_mainly_by_disagreement);
  return (
    <div
      style={{
        background: "#fff",
        border: mode === "caution" ? "1px solid rgba(245, 158, 11, 0.25)" : "1px solid #e2e8f0",
        borderRadius: 12,
        padding: 16,
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
        display: "grid",
        gap: 12,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 17, fontWeight: 700, color: "#0f172a" }}>
            {row.awayTeam} at {row.homeTeam}
          </div>
          <div style={{ ...mono, fontSize: 12, color: "#64748b", marginTop: 4 }}>
            {contextLabel(row)}
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
          <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, ...badgeStyle(mode) }}>
            {mode === "internal" ? "Internal candidate" : mode === "raw" ? "Raw-edge watchlist" : "NCAA caution"}
          </span>
          <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, ...badgeStyle(disagreementLed ? "internal" : "raw") }}>
            {disagreementLed ? "Disagreement-led" : "Raw-edge-led"}
          </span>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
          gap: 10,
        }}
      >
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>MARKET LINE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{fmtSpread(row.book_spread)}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>HOOPS EDGE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{fmtSpread(displayModelSpread(row.predicted_spread))}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>RAW EDGE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{fmtPct(row.pick_prob_edge)}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>BET SCORE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{fmtScore(row.filter_score)}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>PICK SIDE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{row.model_pick_side ?? "—"}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>DISAGREEMENT EDGE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{fmtScore(row.he_market_edge_for_pick)}</div>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: 10,
          color: "#334155",
          fontSize: 13,
        }}
      >
        <div>
          <strong>Signal driver:</strong> {row.signal_driver ?? "—"}
        </div>
        <div>
          <strong>Usage:</strong> {row.usage_label ?? "—"}
        </div>
        <div>
          <strong>Same-sign 21d:</strong> {row.pick_team_recent_same_sign_count_21d ?? "—"}
        </div>
        <div>
          <strong>Prior streak:</strong> {row.pick_team_prior_same_sign_streak ?? "—"}
        </div>
      </div>
    </div>
  );
}

function Section({
  title,
  subtitle,
  rows,
  mode,
  emptyText,
}: {
  title: string;
  subtitle: string;
  rows: InternalBettingRow[];
  mode: "internal" | "raw" | "caution";
  emptyText: string;
}) {
  return (
    <section style={{ display: "grid", gap: 14 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#0f172a" }}>{title}</h2>
        <div style={{ ...mono, fontSize: 12, color: "#64748b", marginTop: 4 }}>{subtitle}</div>
      </div>
      {rows.length ? (
        <div style={{ display: "grid", gap: 12 }}>
          {rows.map((row) => (
            <RowCard
              key={`${row.gameId ?? `${row.homeTeam}-${row.awayTeam}`}-${row.signal_driver ?? "row"}`}
              row={row}
              mode={mode}
            />
          ))}
        </div>
      ) : (
        <div style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12, padding: 18, color: "#64748b" }}>
          {emptyText}
        </div>
      )}
    </section>
  );
}

export default function BettingPage({ payload }: Props) {
  const router = useRouter();

  if (!payload) {
    return (
      <Layout>
        <div style={{ padding: 24, color: "#64748b", textAlign: "center" }}>
          No internal betting artifacts found.
        </div>
      </Layout>
    );
  }

  const manifest = payload.manifest;

  return (
    <Layout>
      <div style={{ display: "grid", gap: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
          <div style={{ display: "grid", gap: 10 }}>
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, ...badgeStyle("internal") }}>
                Internal betting aid
              </span>
              <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, ...badgeStyle("caution") }}>
                Separate from public predictions
              </span>
            </div>
            <div>
              <h1 style={{ margin: 0, fontSize: 28, fontWeight: 800, letterSpacing: "-0.03em", color: "#0f172a" }}>
                Betting Workflow
              </h1>
              <div style={{ ...mono, fontSize: 13, color: "#64748b", marginTop: 6 }}>
                {formatDateDisplay(payload.currentDate)} · Season {manifest?.season ?? "—"} · {manifest?.slate_games ?? payload.slateScores.length} scored games
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <select
              value={payload.currentDate}
              onChange={(e) => router.push(`/betting?date=${e.target.value}`)}
              style={{
                ...mono,
                padding: "8px 10px",
                border: "1px solid #e2e8f0",
                borderRadius: 8,
                background: "#fff",
                color: "#334155",
              }}
            >
              {payload.availableDates.map((date) => (
                <option key={date} value={date}>
                  {date}
                </option>
              ))}
            </select>
            <Link
              href="/"
              style={{
                ...mono,
                fontSize: 12,
                padding: "8px 10px",
                borderRadius: 8,
                border: "1px solid #e2e8f0",
                background: "#fff",
                color: "#334155",
              }}
            >
              Public predictions
            </Link>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: 12,
          }}
        >
          {[
            ["Threshold", manifest?.filter_threshold !== undefined ? `disagreement_logit ≥ ${manifest.filter_threshold}` : "—"],
            ["Raw baseline", manifest?.raw_edge_threshold !== undefined ? `pick_prob_edge ≥ ${manifest.raw_edge_threshold}` : "—"],
            ["Shortlist", String(manifest?.non_ncaa_shortlist_games ?? payload.shortlist.length)],
            ["Raw watchlist", String(manifest?.raw_watchlist_games ?? payload.rawWatchlist.length)],
            ["NCAA caution", String(manifest?.ncaa_caution_games ?? payload.ncaaCaution.length)],
            ["Disagreement-led", String(manifest?.flagged_mainly_by_disagreement ?? payload.shortlist.filter((r) => r.flagged_mainly_by_disagreement).length)],
          ].map(([label, value]) => (
            <div
              key={label}
              style={{
                background: "#fff",
                border: "1px solid #e2e8f0",
                borderRadius: 12,
                padding: 14,
                boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
              }}
            >
              <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>{label.toUpperCase()}</div>
              <div style={{ marginTop: 6, fontSize: 18, fontWeight: 700, color: "#0f172a" }}>{value}</div>
            </div>
          ))}
        </div>

        <div
          style={{
            background: "rgba(15, 23, 42, 0.03)",
            border: "1px solid #e2e8f0",
            borderRadius: 12,
            padding: 16,
            color: "#334155",
            fontSize: 14,
            lineHeight: 1.6,
          }}
        >
          This tab is the internal betting workflow. It uses the disagreement-aware filter on top of the public Hoops Edge slate, but it does not change the public model spread or win probability product. NCAA caution rows are separated intentionally and should not be treated like the main shortlist.
        </div>

        <Section
          title="Internal Candidate Bets"
          subtitle="Promoted disagreement-aware shortlist for daily use."
          rows={payload.shortlist}
          mode="internal"
          emptyText="No non-NCAA internal candidates on this slate."
        />

        <Section
          title="Raw Edge Watchlist"
          subtitle="Games that clear the raw edge baseline but not the promoted disagreement-aware rule."
          rows={payload.rawWatchlist}
          mode="raw"
          emptyText="No raw-edge-only watchlist rows on this slate."
        />

        <Section
          title="NCAA Caution"
          subtitle="Tracked separately as caution / diagnostic context only."
          rows={payload.ncaaCaution}
          mode="caution"
          emptyText="No NCAA caution rows on this slate."
        />
      </div>
    </Layout>
  );
}
