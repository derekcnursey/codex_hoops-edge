import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { displayTeam } from "../../lib/data";
import { buildNcaaBracketGames, getBracketTeams, getRoundOrder } from "../../lib/bracket/ncaaBracket";
import {
  BracketGradingSummary,
  BracketSource,
  BracketRoundId,
  BracketTeam,
  MatchupPrediction,
  NcaaBracketField,
  NcaaTournamentResults,
  NcaaValidationResult,
  ResolvedBracketGame,
} from "../../lib/bracket/types";
import {
  MAJOR_UPSET_SEED_GAP,
  buildGameComparison,
} from "../../lib/bracket/comparison";
import { canonicalMatchupKey, canonicalizePrediction, orientPrediction } from "../../lib/bracket/predictions";
import { gradeBracketPicks } from "../../lib/bracket/grading";
import { buildFinalResultsMap } from "../../lib/bracket/results";
import {
  SHARE_QUERY_PARAM,
  buildShareUrl,
  clearShareStateFromUrl,
  deserializeBracketImport,
  readShareStateFromUrl,
  serializeBracketExport,
} from "../../lib/bracket/state";
import { validateBracketGraph, validateNcaaField } from "../../lib/bracket/validation";
import BracketRound from "./BracketRound";

type BracketRoundSection = {
  label: string;
  games: ResolvedBracketGame[];
};

function gradingBreakdown(summary: BracketGradingSummary): string {
  return summary.rounds
    .filter((round) => round.possibleScore > 0 || round.correct > 0 || round.incorrect > 0)
    .map((round) => `${round.roundLabel}: ${round.correct}-${round.incorrect}-${round.pending}`)
    .join(" | ");
}

function normalizeRegionName(name: string): string {
  return name.trim().toLowerCase().replace(/\s+/g, "-");
}

function canonicalRegionName(field: NcaaBracketField, target: string, fallbackIndex: number): string {
  return (
    field.regions.find((region) => normalizeRegionName(region.name) === target)?.name ??
    field.regions[fallbackIndex]?.name ??
    target
  );
}

export default function BracketBuilder({
  field,
  results,
  resultsErrors = [],
}: {
  field: NcaaBracketField;
  results?: NcaaTournamentResults | null;
  resultsErrors?: string[];
}) {
  const games = useMemo(() => buildNcaaBracketGames(field), [field]);
  const teams = useMemo(() => getBracketTeams(field), [field]);
  const orderedGames = useMemo(
    () => [...games].sort((a, b) => a.roundOrder - b.roundOrder || a.matchupOrder - b.matchupOrder),
    [games],
  );
  const teamById = useMemo(
    () => Object.fromEntries(teams.map((team) => [team.id, team])) as Record<number, BracketTeam>,
    [teams],
  );
  const storageKey = `hoops-edge-ncaa-bracket-${field.season}`;
  const gameIds = useMemo(() => orderedGames.map((game) => game.id), [orderedGames]);
  const [validation, setValidation] = useState<NcaaValidationResult | null>(null);

  const [selectedWinners, setSelectedWinners] = useState<Record<string, number>>({});
  const [predictionCache, setPredictionCache] = useState<Record<string, MatchupPrediction>>({});
  const [loadingMatchups, setLoadingMatchups] = useState<Record<string, boolean>>({});
  const [errorMatchups, setErrorMatchups] = useState<Record<string, string>>({});
  const [autoFillMode, setAutoFillMode] = useState<"seed" | "model" | null>(null);
  const [isCompactLayout, setIsCompactLayout] = useState(false);
  const [shareStatus, setShareStatus] = useState<{ tone: "success" | "error" | "info"; text: string } | null>(null);
  const [modelBracketPicks, setModelBracketPicks] = useState<Record<string, number> | null>(null);
  const [modelBracketLoading, setModelBracketLoading] = useState(false);

  const inFlight = useRef(new Set<string>());
  const hasHydrated = useRef(false);
  const hasUrlState = useRef(false);
  const importInputRef = useRef<HTMLInputElement | null>(null);

  function resolveSource(source: BracketSource, winners: Record<string, number>): BracketTeam | null {
    if (source.type === "team") return teamById[source.teamId] ?? null;
    const winnerId = winners[source.gameId];
    return winnerId ? teamById[winnerId] ?? null : null;
  }

  function sanitizeWinnerMap(winners: Record<string, number>): Record<string, number> {
    const next = { ...winners };
    // Re-walk the bracket from earliest to latest so any upstream change drops
    // invalid downstream picks automatically.
    for (const game of orderedGames) {
      const teamA = resolveSource(game.sourceA, next);
      const teamB = resolveSource(game.sourceB, next);
      const selected = next[game.id];
      if (!teamA || !teamB || (selected !== teamA.id && selected !== teamB.id)) {
        delete next[game.id];
      }
    }
    return next;
  }

  const resolvedGames = useMemo(() => {
    return [...games]
      .sort((a, b) => a.roundOrder - b.roundOrder || a.matchupOrder - b.matchupOrder)
      .map((game) => ({
        ...game,
        teamA: resolveSource(game.sourceA, selectedWinners),
        teamB: resolveSource(game.sourceB, selectedWinners),
        selectedWinnerId: selectedWinners[game.id],
      }));
  }, [games, selectedWinners, teamById]);

  const championshipGame = resolvedGames.find((game) => game.id === "national-championship");
  const champion = championshipGame?.selectedWinnerId
    ? teamById[championshipGame.selectedWinnerId] ?? null
    : null;
  const picksMade = Object.keys(selectedWinners).length;
  const totalPickableGames = games.length;
  const predictionsByGame = useMemo(() => {
    return Object.fromEntries(
      resolvedGames.map((game) => {
        const key = game.teamA && game.teamB ? canonicalMatchupKey(game.teamA.id, game.teamB.id) : "";
        const cached = key ? predictionCache[key] : undefined;
        const oriented =
          cached && game.teamA && game.teamB
            ? orientPrediction(cached, game.teamA.id, game.teamB.id)
            : undefined;
        return [game.id, oriented];
      }),
    ) as Record<string, MatchupPrediction | undefined>;
  }, [predictionCache, resolvedGames]);
  const comparisonsByGame = useMemo(() => {
    return Object.fromEntries(
      resolvedGames.map((game) => [game.id, buildGameComparison(game, predictionsByGame[game.id])]),
    );
  }, [predictionsByGame, resolvedGames]);
  const finalResultsByGame = useMemo(() => buildFinalResultsMap(results ?? null), [results]);
  const gradingActive = Object.keys(finalResultsByGame).length > 0;
  const userGrade = useMemo(
    () => (gradingActive ? gradeBracketPicks(games, selectedWinners, results ?? null, teamById) : null),
    [games, gradingActive, results, selectedWinners, teamById],
  );
  const modelGrade = useMemo(
    () => (gradingActive && modelBracketPicks ? gradeBracketPicks(games, modelBracketPicks, results ?? null, teamById) : null),
    [games, gradingActive, modelBracketPicks, results, teamById],
  );

  function applyValidatedPicks(next: Record<string, number>): boolean {
    const sanitized = sanitizeWinnerMap(next);
    if (Object.keys(sanitized).length !== Object.keys(next).length) {
      return false;
    }
    setSelectedWinners(sanitized);
    return true;
  }

  function clearUrlShareState() {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    if (!url.searchParams.has(SHARE_QUERY_PARAM)) return;
    const nextUrl = clearShareStateFromUrl(window.location.href);
    window.history.replaceState({}, "", nextUrl);
    hasUrlState.current = false;
  }

  function disconnectFromSharedUrlIfNeeded() {
    if (!hasUrlState.current) return;
    clearUrlShareState();
    setShareStatus({
      tone: "info",
      text: "Detached from shared link. Current picks are now local to this browser until you share again.",
    });
  }

  useEffect(() => {
    if (typeof window === "undefined") return;
    const shared = readShareStateFromUrl(window.location.href, field.season, gameIds);
    if (shared.picks) {
      if (applyValidatedPicks(shared.picks)) {
        hasUrlState.current = true;
        hasHydrated.current = true;
        setShareStatus({
          tone: "info",
          text: "Loaded bracket from share link. URL picks override saved local picks for this session.",
        });
        window.localStorage.setItem(storageKey, JSON.stringify(shared.picks));
        return;
      }
      setShareStatus({
        tone: "error",
        text: "Shared bracket link was invalid for this field. Restored local picks if available.",
      });
    } else if (shared.error) {
      setShareStatus({
        tone: "error",
        text: shared.error,
      });
    }

    const raw = window.localStorage.getItem(storageKey);
    if (raw) {
      try {
        const parsed = JSON.parse(raw) as Record<string, number>;
        if (!applyValidatedPicks(parsed)) {
          window.localStorage.removeItem(storageKey);
        }
      } catch {
        window.localStorage.removeItem(storageKey);
      }
    }
    hasHydrated.current = true;
  }, [field.season, gameIds, storageKey]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!hasHydrated.current) return;
    window.localStorage.setItem(storageKey, JSON.stringify(selectedWinners));
  }, [selectedWinners, storageKey]);

  useEffect(() => {
    setModelBracketPicks(null);
    setModelBracketLoading(false);
  }, [field.season, gradingActive]);

  useEffect(() => {
    const validationResults = [validateNcaaField(field), validateBracketGraph(games)];
    const errors = validationResults.flatMap((item) => item.errors);
    setValidation({ valid: errors.length === 0, errors });
  }, [field, games]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mediaQuery = window.matchMedia("(max-width: 960px)");
    const update = () => setIsCompactLayout(mediaQuery.matches);
    update();
    mediaQuery.addEventListener("change", update);
    return () => mediaQuery.removeEventListener("change", update);
  }, []);

  async function fetchPrediction(teamAId: number, teamBId: number): Promise<MatchupPrediction> {
    const key = canonicalMatchupKey(teamAId, teamBId);
    // Cache canonically by ids, then orient to the slot order currently shown.
    if (predictionCache[key]) return orientPrediction(predictionCache[key], teamAId, teamBId);

    const response = await fetch(`/api/predict-matchup?teamAId=${teamAId}&teamBId=${teamBId}`);
    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { error?: string } | null;
      throw new Error(payload?.error || "Prediction lookup failed");
    }

    const prediction = (await response.json()) as MatchupPrediction;
    setPredictionCache((current) => ({ ...current, [key]: canonicalizePrediction(prediction) }));
    setErrorMatchups((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
    return orientPrediction(canonicalizePrediction(prediction), teamAId, teamBId);
  }

  useEffect(() => {
    let cancelled = false;
    for (const game of resolvedGames) {
      if (!game.teamA || !game.teamB) continue;
      const key = canonicalMatchupKey(game.teamA.id, game.teamB.id);
      if (predictionCache[key] || inFlight.current.has(key)) continue;

      inFlight.current.add(key);
      setLoadingMatchups((current) => ({ ...current, [key]: true }));

      fetchPrediction(game.teamA.id, game.teamB.id)
        .catch((error: Error) => {
          if (cancelled) return;
          setErrorMatchups((current) => ({ ...current, [key]: error.message }));
        })
        .finally(() => {
          inFlight.current.delete(key);
          if (cancelled) return;
          setLoadingMatchups((current) => ({ ...current, [key]: false }));
        });
    }

    return () => {
      cancelled = true;
    };
  }, [predictionCache, resolvedGames]);

  useEffect(() => {
    if (!gradingActive || modelBracketPicks || modelBracketLoading) return;

    let cancelled = false;
    setModelBracketLoading(true);

    const buildModelBracket = async () => {
      let next: Record<string, number> = {};
      for (const game of orderedGames) {
        const teamA = resolveSource(game.sourceA, next);
        const teamB = resolveSource(game.sourceB, next);
        if (!teamA || !teamB) continue;
        const prediction = await fetchPrediction(teamA.id, teamB.id);
        next = { ...next, [game.id]: prediction.modelWinnerId };
      }
      if (!cancelled) {
        setModelBracketPicks(next);
      }
    };

    buildModelBracket()
      .catch(() => {
        if (!cancelled) {
          setModelBracketPicks({});
        }
      })
      .finally(() => {
        if (!cancelled) {
          setModelBracketLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [gradingActive, modelBracketLoading, modelBracketPicks, orderedGames]);

  function handleSelectWinner(gameId: string, teamId: number) {
    disconnectFromSharedUrlIfNeeded();
    setShareStatus(null);
    setSelectedWinners((current) => sanitizeWinnerMap({ ...current, [gameId]: teamId }));
  }

  function handleReset() {
    clearUrlShareState();
    setSelectedWinners({});
    setShareStatus({ tone: "info", text: "Bracket reset. Saved local picks cleared." });
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(storageKey);
    }
  }

  function pickBetterSeed(teamA: BracketTeam, teamB: BracketTeam, prediction?: MatchupPrediction): number {
    if (teamA.seed !== teamB.seed) {
      return teamA.seed < teamB.seed ? teamA.id : teamB.id;
    }
    return prediction?.modelWinnerId ?? teamA.id;
  }

  async function handleAutofill(mode: "seed" | "model") {
    setAutoFillMode(mode);
    try {
      disconnectFromSharedUrlIfNeeded();
      let next = { ...selectedWinners };
      for (const game of orderedGames) {
        const teamA = resolveSource(game.sourceA, next);
        const teamB = resolveSource(game.sourceB, next);
        if (!teamA || !teamB) continue;
        const prediction = await fetchPrediction(teamA.id, teamB.id);
        const winnerId = mode === "model" ? prediction.modelWinnerId : pickBetterSeed(teamA, teamB, prediction);
        next = sanitizeWinnerMap({ ...next, [game.id]: winnerId });
      }

      setSelectedWinners(next);
      setShareStatus({
        tone: "success",
        text: mode === "model" ? "Bracket filled with model picks." : "Bracket filled with better-seed picks.",
      });
      if (typeof window !== "undefined") {
        window.localStorage.setItem(storageKey, JSON.stringify(next));
      }
    } finally {
      setAutoFillMode(null);
    }
  }

  async function handleCopyShareLink() {
    if (typeof window === "undefined") return;
    const shareUrl = buildShareUrl(window.location.href, field.season, gameIds, selectedWinners);
    try {
      await navigator.clipboard.writeText(shareUrl);
      setShareStatus({ tone: "success", text: "Share link copied to clipboard." });
    } catch {
      window.prompt("Copy this bracket link", shareUrl);
      setShareStatus({ tone: "info", text: "Share link ready to copy." });
    }
  }

  function handleExportJson() {
    if (typeof window === "undefined") return;
    const json = serializeBracketExport(field.season, gameIds, selectedWinners);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `hoops-edge-ncaa-bracket-${field.season}.json`;
    link.click();
    URL.revokeObjectURL(url);
    setShareStatus({ tone: "success", text: "Bracket JSON exported." });
  }

  async function handleImportJson(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const parsed = deserializeBracketImport(text, field.season, gameIds);
      if (!parsed.picks) {
        setShareStatus({ tone: "error", text: parsed.error ?? "Imported bracket is invalid." });
        return;
      }

      disconnectFromSharedUrlIfNeeded();
      if (!applyValidatedPicks(parsed.picks)) {
        setShareStatus({
          tone: "error",
          text: "Imported bracket has invalid downstream winners for the current field.",
        });
        return;
      }

      setShareStatus({ tone: "success", text: `Imported ${Object.keys(parsed.picks).length} bracket picks.` });
    } catch {
      setShareStatus({ tone: "error", text: "Failed to import bracket file." });
    } finally {
      event.target.value = "";
    }
  }

  const roundGames = useMemo(() => {
    const grouped = new Map<string, ResolvedBracketGame[]>();
    for (const roundId of getRoundOrder()) grouped.set(roundId, []);
    for (const game of resolvedGames) {
      const current = grouped.get(game.roundId) ?? [];
      current.push(game);
      grouped.set(game.roundId, current);
    }
    return grouped;
  }, [resolvedGames]);
  const regionRoundGames = useMemo(() => {
    const grouped = new Map<string, Map<BracketRoundId, ResolvedBracketGame[]>>();
    for (const region of field.regions) {
      const roundMap = new Map<BracketRoundId, ResolvedBracketGame[]>();
      for (const roundId of getRoundOrder()) {
        roundMap.set(roundId, []);
      }
      grouped.set(region.name, roundMap);
    }

    for (const game of resolvedGames) {
      if (!game.region) continue;
      const roundMap = grouped.get(game.region);
      if (!roundMap) continue;
      roundMap.set(game.roundId, [...(roundMap.get(game.roundId) ?? []), game]);
    }

    return grouped;
  }, [field.regions, resolvedGames]);
  const laneRegions = useMemo(
    () => ({
      topLeft: canonicalRegionName(field, "east", 0),
      bottomLeft: canonicalRegionName(field, "west", 1),
      topRight: canonicalRegionName(field, "south", 2),
      bottomRight: canonicalRegionName(field, "midwest", 3),
    }),
    [field],
  );

  function buildRoundSectionState(roundList: ResolvedBracketGame[]) {
    const predictions = Object.fromEntries(roundList.map((game) => [game.id, predictionsByGame[game.id]])) as Record<
      string,
      MatchupPrediction | undefined
    >;
    const comparisons = Object.fromEntries(roundList.map((game) => [game.id, comparisonsByGame[game.id]]));
    const grading = Object.fromEntries(roundList.map((game) => [game.id, userGrade?.byGame[game.id]]));
    const loadingGames = Object.fromEntries(
      roundList.map((game) => {
        const key = game.teamA && game.teamB ? canonicalMatchupKey(game.teamA.id, game.teamB.id) : "";
        return [game.id, key ? loadingMatchups[key] : false];
      }),
    ) as Record<string, boolean | undefined>;
    const errorGames = Object.fromEntries(
      roundList.map((game) => {
        const key = game.teamA && game.teamB ? canonicalMatchupKey(game.teamA.id, game.teamB.id) : "";
        return [game.id, key ? errorMatchups[key] : undefined];
      }),
    ) as Record<string, string | undefined>;

    return { predictions, comparisons, grading, loadingGames, errorGames };
  }

  function renderRoundSection(section: BracketRoundSection, minWidth: number) {
    if (!section.games.length) return null;
    const roundState = buildRoundSectionState(section.games);
    return (
      <BracketRound
        key={`${section.label}-${section.games[0]?.id ?? "empty"}`}
        label={section.label}
        games={section.games}
        predictions={roundState.predictions}
        comparisons={roundState.comparisons}
        grading={roundState.grading}
        loadingGames={roundState.loadingGames}
        errorGames={roundState.errorGames}
        onSelectWinner={handleSelectWinner}
        compact={isCompactLayout}
        stickyTitle={false}
        dense
        minWidth={minWidth}
      />
    );
  }

  function renderRegionLane(regionName: string, side: "left" | "right") {
    const roundMap = regionRoundGames.get(regionName);
    if (!roundMap) return null;

    const sectionsByColumn: BracketRoundSection[][] =
      side === "left"
        ? [
            [
              { label: "First Four", games: roundMap.get("first-four") ?? [] },
              { label: "Round of 64", games: roundMap.get("round-of-64") ?? [] },
            ],
            [{ label: "Round of 32", games: roundMap.get("round-of-32") ?? [] }],
            [{ label: "Sweet 16", games: roundMap.get("sweet-16") ?? [] }],
            [{ label: "Elite 8", games: roundMap.get("elite-8") ?? [] }],
          ]
        : [
            [{ label: "Elite 8", games: roundMap.get("elite-8") ?? [] }],
            [{ label: "Sweet 16", games: roundMap.get("sweet-16") ?? [] }],
            [{ label: "Round of 32", games: roundMap.get("round-of-32") ?? [] }],
            [
              { label: "Round of 64", games: roundMap.get("round-of-64") ?? [] },
              { label: "First Four", games: roundMap.get("first-four") ?? [] },
            ],
          ];

    return (
      <section
        style={{
          borderRadius: 12,
          border: "1px solid #e2e8f0",
          background: "#f8fafc",
          padding: 10,
        }}
      >
        <div
          style={{
            marginBottom: 8,
            display: "flex",
            justifyContent: "space-between",
            gap: 8,
            alignItems: "baseline",
          }}
        >
          <h2
            style={{
              fontSize: 14,
              fontWeight: 700,
              letterSpacing: "-0.02em",
              margin: 0,
              color: "#0f172a",
            }}
          >
            {regionName} Region
          </h2>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: isCompactLayout
              ? "repeat(2, minmax(0, 1fr))"
              : side === "left"
                ? "minmax(138px,1.18fr) repeat(2, minmax(128px,1fr)) minmax(134px,0.94fr)"
                : "minmax(134px,0.94fr) repeat(2, minmax(128px,1fr)) minmax(138px,1.18fr)",
            gap: 8,
            alignItems: "start",
          }}
        >
          {sectionsByColumn.map((sections, index) => (
            <div key={`${regionName}-${side}-${index}`} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {sections.map((section) => renderRoundSection(section, 124))}
            </div>
          ))}
        </div>
      </section>
    );
  }

  if (validation && !validation.valid) {
    return (
      <section
        style={{
          padding: 20,
          borderRadius: 12,
          border: "1px solid #fecaca",
          background: "#fef2f2",
          color: "#991b1b",
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 8 }}>NCAA builder data failed validation</div>
        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, lineHeight: 1.6 }}>
          {validation.errors.slice(0, 8).join(" | ")}
        </div>
      </section>
    );
  }

  return (
    <section
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 18,
      }}
    >
      <div
        style={{
          background: "#ffffff",
          border: "1px solid #e2e8f0",
          borderRadius: 12,
          padding: 14,
          boxShadow: "0 1px 3px rgba(0, 0, 0, 0.04)",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: 12,
            flexWrap: "wrap",
            marginBottom: 10,
          }}
        >
          <div>
            <h1
              style={{
                fontSize: 22,
                fontWeight: 700,
                letterSpacing: "-0.02em",
                margin: "0 0 4px",
                color: "#0f172a",
              }}
            >
              NCAA Tournament Bracket Builder
            </h1>
            <div
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 11,
                color: "#64748b",
              }}
            >
              Manual picks with live Hoops Edge matchup projections on every resolved game
            </div>
          </div>

          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <button
              type="button"
              onClick={handleCopyShareLink}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#ffffff",
                color: "#0f172a",
                cursor: "pointer",
              }}
            >
              Copy share link
            </button>
            <button
              type="button"
              onClick={handleExportJson}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#ffffff",
                color: "#0f172a",
                cursor: "pointer",
              }}
            >
              Export JSON
            </button>
            <button
              type="button"
              onClick={() => importInputRef.current?.click()}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#ffffff",
                color: "#0f172a",
                cursor: "pointer",
              }}
            >
              Import JSON
            </button>
            <button
              type="button"
              onClick={() => handleAutofill("model")}
              disabled={autoFillMode !== null}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #0f172a",
                background: "#0f172a",
                color: "#ffffff",
                cursor: autoFillMode ? "wait" : "pointer",
              }}
            >
              {autoFillMode === "model" ? "Auto-filling..." : "Auto-fill model picks"}
            </button>
            <button
              type="button"
              onClick={() => handleAutofill("seed")}
              disabled={autoFillMode !== null}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#ffffff",
                color: "#0f172a",
                cursor: autoFillMode ? "wait" : "pointer",
              }}
            >
              {autoFillMode === "seed" ? "Auto-filling..." : "Auto-fill better seeds"}
            </button>
            <button
              type="button"
              onClick={handleReset}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#ffffff",
                color: "#0f172a",
                cursor: "pointer",
              }}
            >
              Reset + clear saved
            </button>
            <input
              ref={importInputRef}
              type="file"
              accept="application/json,.json"
              onChange={handleImportJson}
              style={{ display: "none" }}
            />
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: 10,
            marginBottom: 10,
          }}
        >
          <div
            style={{
              borderRadius: 9,
              border: "1px solid #e2e8f0",
              background: "#f8fafc",
              padding: 10,
            }}
          >
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
              Champion
            </div>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#0f172a" }}>
              {champion ? displayTeam(champion.name) : "Awaiting picks"}
            </div>
            {champion ? (
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569", marginTop: 4 }}>
                ({champion.seed}) {champion.region} | Rank {champion.rank}
              </div>
            ) : null}
          </div>

          <div
            style={{
              borderRadius: 9,
              border: "1px solid #e2e8f0",
              background: "#f8fafc",
              padding: 10,
            }}
          >
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
              Progress
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>
              {picksMade}/{totalPickableGames} picks made
            </div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569", marginTop: 4 }}>
              Picks save locally until reset
            </div>
          </div>

          <div
            style={{
              borderRadius: 9,
              border: "1px solid #e2e8f0",
              background: "#f8fafc",
              padding: 10,
            }}
          >
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
              Details
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>
              Spread and ML% live in each team tile
            </div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569", marginTop: 4 }}>
              Adj Pace / OE / DE / Net live in the info modal
            </div>
          </div>

          {gradingActive && userGrade ? (
            <div
              style={{
                borderRadius: 9,
                border: "1px solid #dcfce7",
                background: "#f0fdf4",
                padding: 10,
              }}
            >
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#166534", marginBottom: 4 }}>
                Live Grading
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>
                {userGrade.correct} correct • {userGrade.incorrect} missed • {userGrade.pending} pending
              </div>
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#166534", marginTop: 4 }}>
                Score {userGrade.score}/{userGrade.possibleScore}
              </div>
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#166534", marginTop: 4 }}>
                {gradingBreakdown(userGrade) || "Awaiting scored rounds"}
              </div>
            </div>
          ) : null}

          {gradingActive ? (
            <div
              style={{
                borderRadius: 9,
                border: "1px solid #e2e8f0",
                background: "#f8fafc",
                padding: 10,
              }}
            >
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
                Vs Model
              </div>
              {modelGrade ? (
                <>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>
                    You {userGrade?.score ?? 0} pts • Model {modelGrade.score} pts
                  </div>
                  <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569", marginTop: 4 }}>
                    You {userGrade?.correct ?? 0} correct • Model {modelGrade.correct} correct
                  </div>
                </>
              ) : (
                <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569" }}>
                  {modelBracketLoading ? "Building model bracket..." : "Model grading unavailable"}
                </div>
              )}
            </div>
          ) : null}

          <div
            style={{
              borderRadius: 9,
              border: "1px solid #e2e8f0",
              background: "#f8fafc",
              padding: 10,
            }}
          >
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
              Legend
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>
              Dark = your pick, Blue = favorite, Amber = upset/fade
            </div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569", marginTop: 4 }}>
              Major upset = pick seeded {MAJOR_UPSET_SEED_GAP}+ lines worse
            </div>
          </div>
        </div>

        <div
          style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 11,
            lineHeight: 1.6,
            color: "#64748b",
            marginTop: -2,
          }}
        >
          {field.note}
        </div>
        {resultsErrors.length ? (
          <div
            style={{
              marginTop: 10,
              padding: "8px 10px",
              borderRadius: 8,
              border: "1px solid #fde68a",
              background: "#fffbeb",
              color: "#92400e",
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: 11,
              lineHeight: 1.5,
            }}
          >
            Results grading unavailable: {resultsErrors[0]}
          </div>
        ) : null}
        {shareStatus ? (
          <div
            style={{
              marginTop: 10,
              padding: "8px 10px",
              borderRadius: 8,
              border: `1px solid ${
                shareStatus.tone === "error" ? "#fecaca" : shareStatus.tone === "success" ? "#bbf7d0" : "#cbd5e1"
              }`,
              background:
                shareStatus.tone === "error" ? "#fef2f2" : shareStatus.tone === "success" ? "#f0fdf4" : "#f8fafc",
              color: shareStatus.tone === "error" ? "#991b1b" : shareStatus.tone === "success" ? "#166534" : "#475569",
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: 11,
              lineHeight: 1.5,
            }}
          >
            {shareStatus.text}
          </div>
        ) : null}
      </div>

      <div
        style={{
          overflowX: "auto",
          paddingBottom: 8,
        }}
      >
        {isCompactLayout ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 14,
            }}
          >
            {renderRegionLane(laneRegions.topLeft, "left")}
            {renderRegionLane(laneRegions.bottomLeft, "left")}
            <section
              style={{
                borderRadius: 12,
                border: "1px solid #e2e8f0",
                background: "#f8fafc",
                padding: 12,
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              {renderRoundSection({ label: "Final Four", games: roundGames.get("final-four") ?? [] }, 150)}
              {renderRoundSection({ label: "National Championship", games: roundGames.get("national-championship") ?? [] }, 150)}
            </section>
            {renderRegionLane(laneRegions.topRight, "right")}
            {renderRegionLane(laneRegions.bottomRight, "right")}
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1fr) minmax(188px, 212px) minmax(0, 1fr)",
              gridTemplateRows: "auto auto",
              gap: 10,
              alignItems: "start",
              minWidth: 1220,
            }}
          >
            <div style={{ gridColumn: 1, gridRow: 1 }}>{renderRegionLane(laneRegions.topLeft, "left")}</div>
            <div style={{ gridColumn: 1, gridRow: 2 }}>{renderRegionLane(laneRegions.bottomLeft, "left")}</div>

            <section
              style={{
                gridColumn: 2,
                gridRow: "1 / span 2",
                alignSelf: "stretch",
                borderRadius: 12,
                border: "1px solid #e2e8f0",
                background: "#f8fafc",
                padding: 10,
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                gap: 10,
              }}
            >
              {renderRoundSection({ label: "Final Four", games: (roundGames.get("final-four") ?? []).slice(0, 1) }, 190)}
              {renderRoundSection({ label: "National Championship", games: roundGames.get("national-championship") ?? [] }, 190)}
              {renderRoundSection({ label: "Final Four", games: (roundGames.get("final-four") ?? []).slice(1) }, 190)}
            </section>

            <div style={{ gridColumn: 3, gridRow: 1 }}>{renderRegionLane(laneRegions.topRight, "right")}</div>
            <div style={{ gridColumn: 3, gridRow: 2 }}>{renderRegionLane(laneRegions.bottomRight, "right")}</div>
          </div>
        )}
      </div>
    </section>
  );
}
