import { GetServerSideProps } from "next";
import { CSSProperties, useState } from "react";
import BracketBuilder from "../components/bracket/BracketBuilder";
import ConferenceBrackets, {
  ConferenceBracketsData,
} from "../components/bracket/ConferenceBrackets";
import Layout from "../components/Layout";
import { readJsonFile } from "../lib/server-data";
import { NcaaBracketField, NcaaTournamentResults } from "../lib/bracket/types";
import { buildNcaaBracketGames } from "../lib/bracket/ncaaBracket";
import { validateBracketGraph, validateNcaaField, validateNcaaResults } from "../lib/bracket/validation";

type Props = {
  conferenceData: ConferenceBracketsData | null;
  ncaaField: NcaaBracketField | null;
  ncaaErrors: string[];
  ncaaResults: NcaaTournamentResults | null;
  ncaaResultsErrors: string[];
};

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

export const getServerSideProps: GetServerSideProps<Props> = async () => {
  const conferenceRaw = readJsonFile("brackets_2026.json");
  const ncaaRaw = readJsonFile("ncaa_bracket_builder_2026.json");
  const ncaaField = ncaaRaw as NcaaBracketField | null;
  const bracketGames = ncaaField ? buildNcaaBracketGames(ncaaField) : [];
  const ncaaResultsRaw = readJsonFile("ncaa_results_2026.json");
  const ncaaResultsPayload = ncaaResultsRaw as NcaaTournamentResults | null;
  const errors = ncaaField
    ? [
        ...validateNcaaField(ncaaField).errors,
        ...validateBracketGraph(bracketGames).errors,
      ]
    : ["NCAA bracket-builder data unavailable"];
  const resultsErrors = ncaaField
    ? validateNcaaResults(ncaaResultsPayload, ncaaField, bracketGames).errors
    : [];

  return {
    props: {
      conferenceData: conferenceRaw as ConferenceBracketsData | null,
      ncaaField,
      ncaaErrors: errors,
      ncaaResults: resultsErrors.length === 0 ? ncaaResultsPayload : null,
      ncaaResultsErrors: resultsErrors,
    },
  };
};

export default function Brackets({ conferenceData, ncaaField, ncaaErrors, ncaaResults, ncaaResultsErrors }: Props) {
  const [tab, setTab] = useState<"ncaa" | "conference">("ncaa");

  return (
    <Layout>
      <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div>
            <div style={{ ...mono, fontSize: 12, color: "#64748b", marginBottom: 6 }}>
              Brackets
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.03em", color: "#0f172a" }}>
              Tournament bracket tools
            </div>
          </div>

          <div
            style={{
              display: "flex",
              gap: 8,
              padding: 4,
              borderRadius: 10,
              border: "1px solid #e2e8f0",
              background: "#ffffff",
            }}
          >
            <button
              type="button"
              onClick={() => setTab("ncaa")}
              style={{
                ...mono,
                padding: "10px 14px",
                borderRadius: 8,
                border: "none",
                background: tab === "ncaa" ? "#0f172a" : "transparent",
                color: tab === "ncaa" ? "#ffffff" : "#475569",
                cursor: "pointer",
              }}
            >
              NCAA Builder
            </button>
            <button
              type="button"
              onClick={() => setTab("conference")}
              style={{
                ...mono,
                padding: "10px 14px",
                borderRadius: 8,
                border: "none",
                background: tab === "conference" ? "#0f172a" : "transparent",
                color: tab === "conference" ? "#ffffff" : "#475569",
                cursor: "pointer",
              }}
            >
              Conference Brackets
            </button>
          </div>
        </div>

        {tab === "ncaa" ? (
          ncaaField && ncaaErrors.length === 0 ? (
            <BracketBuilder field={ncaaField} results={ncaaResults} resultsErrors={ncaaResultsErrors} />
          ) : (
            <div
              style={{
                padding: 24,
                borderRadius: 12,
                border: "1px solid #fecaca",
                background: "#fef2f2",
                color: "#991b1b",
              }}
            >
              <div style={{ fontWeight: 700, marginBottom: 8 }}>NCAA bracket-builder data unavailable</div>
              <div style={{ ...mono, fontSize: 12 }}>{ncaaErrors[0] ?? "Unknown validation error"}</div>
            </div>
          )
        ) : (
          <ConferenceBrackets data={conferenceData} />
        )}
      </div>
    </Layout>
  );
}
