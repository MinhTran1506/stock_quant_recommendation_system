"use client";
/**
 * pages/stocks/[ticker]/page.tsx — Individual stock detail page.
 *
 * Sections:
 *  - Price chart (lightweight-charts candlestick with volume)
 *  - Model prediction chart (multi-horizon forecasts with confidence bands)
 *  - Fundamental ratios
 *  - News digest (sentiment-tagged, AI-summarised)
 *  - Order book visualisation
 */
import { useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { createChart, ColorType, CrosshairMode } from "lightweight-charts";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ErrorBar
} from "recharts";
import numeral from "numeral";
import { TrendingUp, TrendingDown, ArrowLeft, ExternalLink } from "lucide-react";

import { stocksApi, predictionsApi, newsApi } from "@/utils/api";

const CHART_THEME = {
  background: "#0f172a",
  textColor: "#94a3b8",
  gridColor: "#1e293b",
  upColor: "#22c55e",
  downColor: "#ef4444",
};

export default function StockDetailPage() {
  const { ticker } = useParams<{ ticker: string }>();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const [priceInterval, setPriceInterval] = useState<"1d" | "1w" | "1m">("1d");

  // ── Data fetching ────────────────────────────────────────────────────
  const { data: stock } = useQuery({
    queryKey: ["stock", ticker],
    queryFn: () => stocksApi.get(ticker).then((r) => r.data),
    enabled: !!ticker,
  });

  const { data: prices } = useQuery({
    queryKey: ["prices", ticker, priceInterval],
    queryFn: () =>
      stocksApi.prices(ticker, { limit: 504, interval: priceInterval }).then((r) => r.data),
    enabled: !!ticker,
  });

  const { data: predictions } = useQuery({
    queryKey: ["predictions", ticker],
    queryFn: () => predictionsApi.ticker(ticker, { limit: 100 }).then((r) => r.data),
    enabled: !!ticker,
  });

  const { data: newsItems } = useQuery({
    queryKey: ["news", ticker],
    queryFn: () => newsApi.ticker(ticker, { limit: 15 }).then((r) => r.data),
    enabled: !!ticker,
  });

  // ── Candlestick chart (lightweight-charts) ────────────────────────────
  useEffect(() => {
    if (!chartContainerRef.current || !prices?.length) return;

    // Destroy previous chart
    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: CHART_THEME.background },
        textColor: CHART_THEME.textColor,
      },
      grid: {
        vertLines: { color: CHART_THEME.gridColor },
        horzLines: { color: CHART_THEME.gridColor },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: CHART_THEME.gridColor },
      timeScale: { borderColor: CHART_THEME.gridColor, timeVisible: true },
      width: chartContainerRef.current.clientWidth,
      height: 400,
    });
    chartRef.current = chart;

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: CHART_THEME.upColor,
      downColor: CHART_THEME.downColor,
      borderDownColor: CHART_THEME.downColor,
      borderUpColor: CHART_THEME.upColor,
      wickDownColor: CHART_THEME.downColor,
      wickUpColor: CHART_THEME.upColor,
    });

    const candleData = prices.map((p: any) => ({
      time: p.date,
      open: p.open,
      high: p.high,
      low: p.low,
      close: p.close,
    }));
    candleSeries.setData(candleData);

    // Volume histogram
    const volumeSeries = chart.addHistogramSeries({
      color: "#3b82f620",
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });
    volumeSeries.setData(
      prices.map((p: any) => ({
        time: p.date,
        value: p.volume,
        color:
          p.close >= p.open
            ? `${CHART_THEME.upColor}50`
            : `${CHART_THEME.downColor}50`,
      }))
    );

    chart.timeScale().fitContent();

    // Responsive resize
    const resizeObserver = new ResizeObserver(() => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    });
    if (chartContainerRef.current) {
      resizeObserver.observe(chartContainerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, [prices]);

  // ── Prediction chart data ─────────────────────────────────────────────
  const predChartData = predictions
    ? (() => {
        const latest = predictions.slice(0, 5);
        return latest.map((p: any) => ({
          horizon: `${p.horizon_days}d`,
          return: p.predicted_return != null ? +(p.predicted_return * 100).toFixed(2) : null,
          lower: p.confidence_lower != null ? +(p.confidence_lower * 100).toFixed(2) : null,
          upper: p.confidence_upper != null ? +(p.confidence_upper * 100).toFixed(2) : null,
          score: p.score,
        }));
      })()
    : [];

  if (!stock) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500" />
      </div>
    );
  }

  const priceChange = stock.change_pct_1d ?? 0;
  const isPositive = priceChange >= 0;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="max-w-screen-xl mx-auto px-4 py-6 space-y-6">
        {/* ── Back + header ─────────────────────────────────────────────── */}
        <div>
          <a
            href="/dashboard"
            className="flex items-center gap-1.5 text-gray-500 hover:text-gray-300 text-sm mb-4 transition"
          >
            <ArrowLeft size={14} /> Back to dashboard
          </a>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-white font-mono">
                {stock.ticker}
              </h1>
              <p className="text-gray-400 mt-0.5">{stock.name}</p>
              <div className="flex items-center gap-2 mt-2">
                <span className="px-2 py-0.5 rounded-full bg-gray-800 text-xs text-gray-400">
                  {stock.exchange}
                </span>
                {stock.sector && (
                  <span className="px-2 py-0.5 rounded-full bg-blue-950 text-xs text-blue-300">
                    {stock.sector}
                  </span>
                )}
              </div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-mono font-bold text-white">
                {stock.latest_price
                  ? numeral(stock.latest_price).format("0,0")
                  : "—"}{" "}
                <span className="text-sm text-gray-500">VND</span>
              </div>
              <div
                className={`flex items-center justify-end gap-1 text-lg font-semibold mt-0.5
                  ${isPositive ? "text-green-400" : "text-red-400"}`}
              >
                {isPositive ? <TrendingUp size={18} /> : <TrendingDown size={18} />}
                {isPositive ? "+" : ""}
                {numeral(priceChange).format("0.00")}%
              </div>
            </div>
          </div>
        </div>

        {/* ── Candlestick chart ─────────────────────────────────────────── */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-gray-300">Price Chart</h2>
            <div className="flex gap-1">
              {(["1d", "1w", "1m"] as const).map((interval) => (
                <button
                  key={interval}
                  onClick={() => setPriceInterval(interval)}
                  className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                    priceInterval === interval
                      ? "bg-blue-600 text-white"
                      : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                  }`}
                >
                  {interval}
                </button>
              ))}
            </div>
          </div>
          <div ref={chartContainerRef} />
        </div>

        {/* ── Two-column: predictions + fundamentals ────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Multi-horizon prediction chart */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
            <h2 className="text-sm font-semibold text-gray-300 mb-4">
              Model Forecasts (multi-horizon)
            </h2>
            {predChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={predChartData}>
                  <CartesianGrid stroke="#1e293b" />
                  <XAxis dataKey="horizon" tick={{ fill: "#64748b", fontSize: 11 }} />
                  <YAxis
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip
                    formatter={(v: any) => [`${v}%`, "Predicted Return"]}
                    contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                  />
                  <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                  <Area
                    type="monotone"
                    dataKey="upper"
                    stroke="transparent"
                    fill="#22c55e15"
                    name="upper"
                  />
                  <Area
                    type="monotone"
                    dataKey="return"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    fill="#3b82f615"
                    dot={{ fill: "#3b82f6", r: 4 }}
                    name="Predicted Return"
                  />
                  <Area
                    type="monotone"
                    dataKey="lower"
                    stroke="transparent"
                    fill="transparent"
                    name="lower"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-56 flex items-center justify-center text-gray-600 text-sm">
                No predictions available
              </div>
            )}
          </div>

          {/* Fundamental ratios */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
            <h2 className="text-sm font-semibold text-gray-300 mb-4">Key Fundamentals</h2>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "Market Cap", value: stock.market_cap ? `${numeral(stock.market_cap / 1e12).format("0.0")}T VND` : "—" },
                { label: "Exchange", value: stock.exchange },
                { label: "Sector", value: stock.sector || "—" },
                { label: "Listed", value: stock.listing_date?.slice(0, 10) || "—" },
              ].map(({ label, value }) => (
                <div key={label} className="bg-gray-800 rounded-lg p-3">
                  <div className="text-xs text-gray-500 mb-0.5">{label}</div>
                  <div className="text-sm font-medium text-white">{value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── News panel ────────────────────────────────────────────────── */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">
            Latest News — AI Summaries
          </h2>
          {newsItems?.length ? (
            <div className="space-y-3">
              {newsItems.map((article: any) => (
                <NewsCard key={article.url || article.title} article={article} />
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-600">No recent news found</p>
          )}
        </div>
      </div>
    </div>
  );
}

function NewsCard({ article }: { article: any }) {
  const sentiment = article.sentiment_score ?? 0;
  const sentColor =
    sentiment > 0.1
      ? "text-green-400 bg-green-950 border-green-800"
      : sentiment < -0.1
      ? "text-red-400 bg-red-950 border-red-800"
      : "text-gray-400 bg-gray-800 border-gray-700";

  return (
    <div className="flex gap-3 p-3 rounded-lg bg-gray-800/50 hover:bg-gray-800 transition">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className={`text-xs px-2 py-0.5 rounded-full border ${sentColor}`}>
            {sentiment > 0.1 ? "Bullish" : sentiment < -0.1 ? "Bearish" : "Neutral"}
          </span>
          {article.event_tags?.slice(0, 2).map((tag: string) => (
            <span
              key={tag}
              className="text-xs px-2 py-0.5 rounded-full bg-blue-950 border border-blue-800 text-blue-300"
            >
              {tag}
            </span>
          ))}
          <span className="text-xs text-gray-600 ml-auto">{article.source}</span>
        </div>
        <h3 className="text-sm font-medium text-white leading-tight">{article.title}</h3>
        {article.summary && (
          <p className="text-xs text-gray-400 mt-1 leading-relaxed line-clamp-2">
            {article.summary}
          </p>
        )}
      </div>
      {article.url && (
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-600 hover:text-blue-400 transition flex-shrink-0 mt-0.5"
          onClick={(e) => e.stopPropagation()}
        >
          <ExternalLink size={14} />
        </a>
      )}
    </div>
  );
}
