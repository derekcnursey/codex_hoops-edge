import { GetServerSideProps } from "next";
import { CSSProperties, useState } from "react";
import BracketBuilder from "../components/bracket/BracketBuilder";
import ConferenceBrackets, {
  ConferenceBracketsData,
} from "../components/bracket/ConferenceBrackets";
import MarchBettingTab from "../components/bracket/MarchBettingTab";
import Layout from "../components/Layout";
import { readJsonFile } from "../lib/server-data";
import { buildScheduledNcaaMarchData } from "../lib/bracket/marchBetting";
import { MatchupPrediction, MatchupPredictionCache, MarchBettingGame, NcaaBracketField, NcaaTournamentResults } from "../lib/bracket/types";
import { buildNcaaBracketGames } from "../lib/bracket/ncaaBracket";
import { validateBracketGraph, validateMatchupCache, validateNcaaField, validateNcaaResults } from "../lib/bracket/validation";

type Props = {
  conferenceData: ConferenceBracketsData | null;
  ncaaField: NcaaBracketField | null;
  ncaaErrors: string[];
  ncaaResults: NcaaTournamentResults | null;
  ncaaResultsErrors: string[];
  initialPredictionCache: Record<string, MatchupPrediction>;
  marchGames: MarchBettingGame[];
};

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

export const getServerSideProps: GetServerSideProps<Props> = async () => {
  const conferenceRaw = readJsonFile("brackets_2026.json");
  const ncaaRaw = readJsonFile("ncaa_bracket_builder_2026.json");
  const matchupRaw = readJsonFile("ncaa_matchup_predictions_2026.json");
  const ncaaField = ncaaRaw as NcaaBracketField | null;
  const matchupCache = matchupRaw as MatchupPredictionCache | null;
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
  const matchupValidation =
    ncaaField && matchupCache ? validateMatchupCache(matchupCache, ncaaField) : null;
  const { initialPredictionCache, marchGames } =
    ncaaField && matchupCache && matchupValidation?.valid
      ? buildScheduledNcaaMarchData(ncaaField, matchupCache)
      : { initialPredictionCache: {}, marchGames: [] };

  return {
    props: {
      conferenceData: conferenceRaw as ConferenceBracketsData | null,
      ncaaField,
      ncaaErrors: errors,
      ncaaResults: resultsErrors.length === 0 ? ncaaResultsPayload : null,
      ncaaResultsErrors: resultsErrors,
      initialPredictionCache,
      marchGames,
    },
  };
};

export default function Brackets({
  conferenceData,
  ncaaField,
  ncaaErrors,
  ncaaResults,
  ncaaResultsErrors,
  initialPredictionCache,
  marchGames,
}: Props) {
  const [tab, setTab] = useState<"ncaa" | "march" | "conference">("ncaa");

  return (
    <Layout wide={tab === "ncaa" || tab === "march"}>
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
              onClick={() => setTab("march")}
              style={{
                ...mono,
                padding: "10px 14px",
                borderRadius: 8,
                border: "none",
                background: tab === "march" ? "#0f172a" : "transparent",
                color: tab === "march" ? "#ffffff" : "#475569",
                cursor: "pointer",
              }}
            >
              March Betting
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
            <BracketBuilder
              field={ncaaField}
              results={ncaaResults}
              resultsErrors={ncaaResultsErrors}
              initialPredictionCache={initialPredictionCache}
            />
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
        ) : tab === "march" ? (
          <MarchBettingTab games={marchGames} />
        ) : (
          <ConferenceBrackets data={conferenceData} />
        )}
      </div>
    </Layout>
  );
}
