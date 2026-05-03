"use client";
/**
 * components/StockRankingTable.tsx
 * Sortable table of ML-ranked stocks with score bars, SHAP tooltips,
 * sentiment badges, and click-through to stock detail.
 */
import { useState, useMemo } from "react";
import { TrendingUp, TrendingDown, Minus, ChevronUp, ChevronDown, Info } from "lucide-react";
import numeral from "numeral";

interface RankingEntry {
  rank: number;
  ticker: string;
  name: string;
  sector?: string;
  score: number;
  predicted_return_5d?: number;
  sentiment_score?: number;
  top_features?: Record<string, number>;
  current_price?: number;
  change_pct_1d?: number;
}

interface Props {
  data: RankingEntry[];
  isLoading: boolean;
}

type SortKey = "rank" | "score" | "predicted_return_5d" | "change_pct_1d" | "sentiment_score";

export default function StockRankingTable({ data, isLoading }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [hoveredTicker, setHoveredTicker] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  const sorted = useMemo(() => {
    let filtered = data.filter(
      (r) =>
        r.ticker.toLowerCase().includes(search.toLowerCase()) ||
        r.name?.toLowerCase().includes(search.toLowerCase()) ||
        r.sector?.toLowerCase().includes(search.toLowerCase())
    );
    return filtered.sort((a, b) => {
      const av = (a as any)[sortKey] ?? 0;
      const bv = (b as any)[sortKey] ?? 0;
      return sortDir === "asc" ? av - bv : bv - av;
    });
  }, [data, sortKey, sortDir, search]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const SortIcon = ({ col }: { col: SortKey }) =>
    sortKey === col ? (
      sortDir === "desc" ? (
        <ChevronDown size={12} />
      ) : (
        <ChevronUp size={12} />
      )
    ) : (
      <span className="opacity-0 group-hover:opacity-40">
        <ChevronDown size={12} />
      </span>
    );

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">
          Stock Rankings
          <span className="ml-2 text-xs font-normal text-gray-400">
            {sorted.length} stocks · 5-day horizon
          </span>
        </h2>
        <input
          type="text"
          placeholder="Search ticker / sector…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="text-sm bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-gray-300
                     placeholder-gray-600 focus:outline-none focus:border-blue-500 w-48"
        />
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase tracking-wide">
              <Th onClick={() => toggleSort("rank")} label="#" extra={<SortIcon col="rank" />} />
              <th className="px-4 py-3 text-left">Ticker</th>
              <th className="px-4 py-3 text-left hidden md:table-cell">Sector</th>
              <Th onClick={() => toggleSort("score")} label="Score" extra={<SortIcon col="score" />} />
              <Th onClick={() => toggleSort("predicted_return_5d")} label="Pred Return" extra={<SortIcon col="predicted_return_5d" />} />
              <Th onClick={() => toggleSort("change_pct_1d")} label="1d Chg" extra={<SortIcon col="change_pct_1d" />} />
              <Th onClick={() => toggleSort("sentiment_score")} label="Sentiment" extra={<SortIcon col="sentiment_score" />} />
              <th className="px-4 py-3 text-left">Why?</th>
            </tr>
          </thead>
          <tbody>
            {isLoading
              ? Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} />)
              : sorted.map((stock) => (
                  <StockRow
                    key={stock.ticker}
                    stock={stock}
                    isHovered={hoveredTicker === stock.ticker}
                    onHover={setHoveredTicker}
                  />
                ))}
          </tbody>
        </table>
      </div>

      {sorted.length === 0 && !isLoading && (
        <div className="py-12 text-center text-gray-500 text-sm">
          No stocks match your filter
        </div>
      )}
    </div>
  );
}

// ── Subcomponents ─────────────────────────────────────────────────────────
function Th({
  label,
  onClick,
  extra,
}: {
  label: string;
  onClick: () => void;
  extra?: React.ReactNode;
}) {
  return (
    <th
      className="px-4 py-3 text-left cursor-pointer group hover:text-gray-300 transition select-none"
      onClick={onClick}
    >
      <span className="flex items-center gap-1">
        {label} {extra}
      </span>
    </th>
  );
}

function StockRow({
  stock,
  isHovered,
  onHover,
}: {
  stock: RankingEntry;
  isHovered: boolean;
  onHover: (t: string | null) => void;
}) {
  const [showFeatures, setShowFeatures] = useState(false);

  const scoreColor =
    stock.score >= 75
      ? "text-green-400"
      : stock.score >= 55
      ? "text-yellow-400"
      : "text-red-400";

  const retColor =
    (stock.predicted_return_5d ?? 0) > 0
      ? "text-green-400"
      : (stock.predicted_return_5d ?? 0) < 0
      ? "text-red-400"
      : "text-gray-400";

  const chgColor =
    (stock.change_pct_1d ?? 0) > 0
      ? "text-green-400"
      : (stock.change_pct_1d ?? 0) < 0
      ? "text-red-400"
      : "text-gray-400";

  const sentIcon =
    (stock.sentiment_score ?? 0) > 0.1 ? (
      <TrendingUp size={12} className="text-green-400" />
    ) : (stock.sentiment_score ?? 0) < -0.1 ? (
      <TrendingDown size={12} className="text-red-400" />
    ) : (
      <Minus size={12} className="text-gray-500" />
    );

  return (
    <>
      <tr
        className={`border-b border-gray-800/50 hover:bg-gray-800/40 transition cursor-pointer
                    ${isHovered ? "bg-gray-800/30" : ""}`}
        onMouseEnter={() => onHover(stock.ticker)}
        onMouseLeave={() => onHover(null)}
        onClick={() => (window.location.href = `/stocks/${stock.ticker}`)}
      >
        {/* Rank */}
        <td className="px-4 py-3 text-gray-500 text-xs w-10">{stock.rank}</td>

        {/* Ticker */}
        <td className="px-4 py-3">
          <div className="font-mono font-semibold text-white">{stock.ticker}</div>
          <div className="text-xs text-gray-500 truncate max-w-[120px]">{stock.name}</div>
        </td>

        {/* Sector */}
        <td className="px-4 py-3 hidden md:table-cell">
          {stock.sector && (
            <span className="px-2 py-0.5 rounded-full bg-gray-800 text-gray-400 text-xs">
              {stock.sector?.slice(0, 16)}
            </span>
          )}
        </td>

        {/* Score bar */}
        <td className="px-4 py-3 w-32">
          <div className="flex items-center gap-2">
            <div className="flex-1 bg-gray-800 rounded-full h-1.5">
              <div
                className={`h-1.5 rounded-full transition-all ${
                  stock.score >= 75
                    ? "bg-green-500"
                    : stock.score >= 55
                    ? "bg-yellow-500"
                    : "bg-red-500"
                }`}
                style={{ width: `${stock.score}%` }}
              />
            </div>
            <span className={`text-xs font-mono font-semibold ${scoreColor} w-8`}>
              {numeral(stock.score).format("0.0")}
            </span>
          </div>
        </td>

        {/* Predicted return */}
        <td className={`px-4 py-3 font-mono text-sm ${retColor}`}>
          {stock.predicted_return_5d != null
            ? `${(stock.predicted_return_5d * 100).toFixed(2)}%`
            : "—"}
        </td>

        {/* 1d change */}
        <td className={`px-4 py-3 font-mono text-sm ${chgColor}`}>
          {stock.change_pct_1d != null
            ? `${stock.change_pct_1d > 0 ? "+" : ""}${numeral(stock.change_pct_1d).format("0.00")}%`
            : "—"}
        </td>

        {/* Sentiment */}
        <td className="px-4 py-3">
          <div className="flex items-center gap-1">
            {sentIcon}
            <span className="text-xs text-gray-400">
              {stock.sentiment_score != null
                ? numeral(stock.sentiment_score).format("0.00")
                : "—"}
            </span>
          </div>
        </td>

        {/* Explainability trigger */}
        <td className="px-4 py-3">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowFeatures((v) => !v);
            }}
            className="text-gray-500 hover:text-blue-400 transition"
            title="Show model explanations"
          >
            <Info size={14} />
          </button>
        </td>
      </tr>

      {/* SHAP feature importance expandable row */}
      {showFeatures && stock.top_features && (
        <tr className="bg-gray-900/80 border-b border-gray-800/50">
          <td colSpan={8} className="px-6 py-3">
            <div className="text-xs text-gray-400 font-semibold mb-2">
              Top driving features (SHAP values)
            </div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(stock.top_features)
                .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                .slice(0, 6)
                .map(([feature, shap]) => (
                  <div
                    key={feature}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs border ${
                      shap > 0
                        ? "bg-green-950 border-green-800 text-green-300"
                        : "bg-red-950 border-red-800 text-red-300"
                    }`}
                  >
                    <span className="font-mono">{feature}</span>
                    <span className="opacity-60">
                      {shap > 0 ? "+" : ""}
                      {numeral(shap).format("0.0000")}
                    </span>
                  </div>
                ))}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function SkeletonRow() {
  return (
    <tr className="border-b border-gray-800/50">
      {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-4 bg-gray-800 rounded animate-pulse w-16" />
        </td>
      ))}
    </tr>
  );
}
