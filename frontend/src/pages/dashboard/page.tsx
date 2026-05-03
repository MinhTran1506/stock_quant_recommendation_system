"use client";
/**
 * pages/dashboard/page.tsx — Main trading dashboard.
 *
 * Panels:
 *  - Top-ranked stocks (meta-model scores with SHAP explanations)
 *  - VN-Index summary card
 *  - Live price ticker strip (WebSocket)
 *  - Sector heatmap
 *  - Recent news feed
 */
import { useEffect, useState, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  TrendingUp, TrendingDown, Activity, BarChart2,
  Newspaper, AlertTriangle, RefreshCw,
} from "lucide-react";
import numeral from "numeral";

import { predictionsApi, stocksApi, newsApi } from "@/utils/api";
import StockRankingTable from "@/components/StockRankingTable";
import NewsPanel from "@/components/NewsPanel";
import SectorHeatmap from "@/components/SectorHeatmap";
import LiveTickerStrip from "@/components/LiveTickerStrip";
import MetricCard from "@/components/ui/MetricCard";

export default function DashboardPage() {
  // ── Data fetching ──────────────────────────────────────────────────────
  const { data: rankings, isLoading: rankingsLoading, refetch: refetchRankings } = useQuery({
    queryKey: ["rankings", { horizon: 5, top_n: 50 }],
    queryFn: () => predictionsApi.rankings({ horizon: 5, top_n: 50 }).then((r) => r.data),
    refetchInterval: 5 * 60 * 1000, // refresh every 5 min
    staleTime: 4 * 60 * 1000,
  });

  const { data: news } = useQuery({
    queryKey: ["news", { limit: 20 }],
    queryFn: () => newsApi.list({ limit: 20 }).then((r) => r.data),
    refetchInterval: 3 * 60 * 1000,
  });

  // ── Market summary stats ───────────────────────────────────────────────
  const topGainers = rankings?.filter((r: any) => (r.predicted_return_5d ?? 0) > 0).slice(0, 3) ?? [];
  const topLosers  = rankings?.filter((r: any) => (r.predicted_return_5d ?? 0) < 0).slice(-3).reverse() ?? [];
  const avgScore   = rankings?.length
    ? rankings.reduce((a: number, r: any) => a + r.score, 0) / rankings.length
    : 0;
  const bullishPct = rankings?.length
    ? Math.round(rankings.filter((r: any) => r.score > 60).length / rankings.length * 100)
    : 0;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* ── Live ticker strip ────────────────────────────────────────── */}
      <LiveTickerStrip />

      <div className="max-w-screen-2xl mx-auto px-4 py-6 space-y-6">
        {/* ── Page header ──────────────────────────────────────────────── */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Market Intelligence Dashboard
            </h1>
            <p className="text-sm text-gray-400 mt-0.5">
              Vietnam Equity Market · HOSE & HNX · Powered by ML models
            </p>
          </div>
          <button
            onClick={() => refetchRankings()}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm text-gray-300 transition"
          >
            <RefreshCw size={14} />
            Refresh
          </button>
        </div>

        {/* ── Metric cards row ──────────────────────────────────────────── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Stocks Analysed"
            value={rankings?.length ?? 0}
            icon={<BarChart2 size={18} />}
            color="blue"
          />
          <MetricCard
            title="Bullish Signals"
            value={`${bullishPct}%`}
            subtitle={`of universe scored ≥ 60`}
            icon={<TrendingUp size={18} />}
            color={bullishPct > 55 ? "green" : bullishPct < 40 ? "red" : "yellow"}
          />
          <MetricCard
            title="Avg Model Score"
            value={numeral(avgScore).format("0.0")}
            subtitle="Cross-sectional mean"
            icon={<Activity size={18} />}
            color="purple"
          />
          <MetricCard
            title="News Articles"
            value={news?.length ?? 0}
            subtitle="Processed today"
            icon={<Newspaper size={18} />}
            color="cyan"
          />
        </div>

        {/* ── Main content grid ─────────────────────────────────────────── */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Rankings table — takes 2/3 width on xl */}
          <div className="xl:col-span-2">
            <StockRankingTable
              data={rankings ?? []}
              isLoading={rankingsLoading}
            />
          </div>

          {/* Right sidebar */}
          <div className="space-y-6">
            {/* Top movers */}
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                <TrendingUp size={14} className="text-green-400" />
                Top Predicted Gainers (5d)
              </h3>
              {topGainers.map((stock: any) => (
                <StockMoverRow key={stock.ticker} stock={stock} positive />
              ))}
            </div>

            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                <TrendingDown size={14} className="text-red-400" />
                Top Predicted Losers (5d)
              </h3>
              {topLosers.map((stock: any) => (
                <StockMoverRow key={stock.ticker} stock={stock} positive={false} />
              ))}
            </div>

            {/* Sector heatmap */}
            <SectorHeatmap data={rankings ?? []} />
          </div>
        </div>

        {/* ── News feed ─────────────────────────────────────────────────── */}
        <NewsPanel articles={news ?? []} />
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────
function StockMoverRow({
  stock,
  positive,
}: {
  stock: any;
  positive: boolean;
}) {
  const ret = stock.predicted_return_5d ?? 0;
  return (
    <a
      href={`/stocks/${stock.ticker}`}
      className="flex items-center justify-between py-2 hover:bg-gray-800 rounded-lg px-2 -mx-2 transition cursor-pointer"
    >
      <div>
        <span className="font-mono font-semibold text-white text-sm">
          {stock.ticker}
        </span>
        <span className="text-gray-500 text-xs ml-2">{stock.name?.slice(0, 20)}</span>
      </div>
      <div className="text-right">
        <span
          className={`text-sm font-semibold ${positive ? "text-green-400" : "text-red-400"}`}
        >
          {positive ? "+" : ""}
          {numeral(ret * 100).format("0.00")}%
        </span>
        <div className="text-xs text-gray-500">
          Score: {numeral(stock.score).format("0.0")}
        </div>
      </div>
    </a>
  );
}
