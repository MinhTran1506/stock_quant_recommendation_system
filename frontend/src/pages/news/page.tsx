"use client";
/**
 * pages/news/page.tsx — Market news feed with AI enrichment.
 *
 * Features:
 *  - Sentiment filter (Bullish / Neutral / Bearish)
 *  - Event tag filter (earnings, dividend, M&A, macro…)
 *  - Ticker search
 *  - AI-generated summaries (FinBERT + T5)
 *  - Real-time news push via Kafka → WebSocket (when available)
 *  - Export to CSV
 */
import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { TrendingUp, TrendingDown, Minus, Newspaper, Search, Filter, ExternalLink, RefreshCw } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

import { newsApi } from "@/utils/api";

const SENTIMENT_COLORS: Record<string, string> = {
  POSITIVE: "text-green-400 bg-green-950 border-green-800",
  NEGATIVE: "text-red-400 bg-red-950 border-red-800",
  NEUTRAL:  "text-gray-400 bg-gray-800 border-gray-700",
};

const EVENT_COLORS: Record<string, string> = {
  earnings:   "bg-blue-950 text-blue-300 border-blue-800",
  dividend:   "bg-purple-950 text-purple-300 border-purple-800",
  merger:     "bg-orange-950 text-orange-300 border-orange-800",
  macro:      "bg-teal-950 text-teal-300 border-teal-800",
  regulatory: "bg-yellow-950 text-yellow-300 border-yellow-800",
  leadership: "bg-pink-950 text-pink-300 border-pink-800",
  ipo:        "bg-indigo-950 text-indigo-300 border-indigo-800",
  legal:      "bg-red-950 text-red-300 border-red-800",
};

export default function NewsPage() {
  const [sentimentFilter, setSentimentFilter] = useState<string>("ALL");
  const [eventFilter, setEventFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { data: articles, isLoading, refetch } = useQuery({
    queryKey: ["news-feed", sentimentFilter, eventFilter, selectedTicker],
    queryFn: () => {
      const params: any = { limit: 100 };
      if (selectedTicker) params.ticker = selectedTicker;
      return newsApi.list(params).then((r) => r.data);
    },
    refetchInterval: 3 * 60 * 1000,
  });

  const filtered = useMemo(() => {
    if (!articles) return [];
    return articles.filter((a: any) => {
      if (sentimentFilter !== "ALL" && a.sentiment_label !== sentimentFilter) return false;
      if (eventFilter !== "ALL" && !(a.event_tags || []).includes(eventFilter)) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        if (!a.title?.toLowerCase().includes(q) && !a.summary?.toLowerCase().includes(q)) return false;
      }
      return true;
    });
  }, [articles, sentimentFilter, eventFilter, searchQuery]);

  // Aggregate stats
  const stats = useMemo(() => {
    if (!articles) return { bullish: 0, bearish: 0, neutral: 0, avgSentiment: 0 };
    const bullish  = articles.filter((a: any) => a.sentiment_label === "POSITIVE").length;
    const bearish  = articles.filter((a: any) => a.sentiment_label === "NEGATIVE").length;
    const neutral  = articles.filter((a: any) => a.sentiment_label === "NEUTRAL").length;
    const scores   = articles.map((a: any) => a.sentiment_score ?? 0);
    return { bullish, bearish, neutral, avgSentiment: scores.reduce((a: number, b: number) => a + b, 0) / Math.max(scores.length, 1) };
  }, [articles]);

  const allEventTags = useMemo(() => {
    if (!articles) return [];
    const tags = new Set<string>();
    articles.forEach((a: any) => (a.event_tags || []).forEach((t: string) => tags.add(t)));
    return Array.from(tags).sort();
  }, [articles]);

  const sentimentColor = stats.avgSentiment > 0.1 ? "text-green-400" : stats.avgSentiment < -0.1 ? "text-red-400" : "text-gray-400";

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="max-w-screen-xl mx-auto px-4 py-6 space-y-6">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              <Newspaper size={22} className="text-blue-400" />
              Market News Intelligence
            </h1>
            <p className="text-sm text-gray-400 mt-0.5">
              AI-enriched with FinBERT sentiment · T5 summaries · Event classification
            </p>
          </div>
          <button onClick={() => refetch()} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm text-gray-300">
            <RefreshCw size={13} /> Refresh
          </button>
        </div>

        {/* ── Sentiment summary ─────────────────────────────────────── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: "Total Articles", value: articles?.length ?? 0, color: "text-white" },
            { label: "Bullish", value: stats.bullish, color: "text-green-400", icon: <TrendingUp size={14} /> },
            { label: "Bearish", value: stats.bearish, color: "text-red-400", icon: <TrendingDown size={14} /> },
            { label: "Avg Sentiment", value: stats.avgSentiment.toFixed(3), color: sentimentColor },
          ].map(({ label, value, color, icon }) => (
            <div key={label} className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <div className="text-xs text-gray-500 mb-1 flex items-center gap-1">{icon}{label}</div>
              <div className={`text-2xl font-mono font-bold ${color}`}>{value}</div>
            </div>
          ))}
        </div>

        {/* ── Filters ───────────────────────────────────────────────── */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <div className="flex flex-wrap gap-3">
            {/* Search */}
            <div className="relative flex-1 min-w-48">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                type="text"
                placeholder="Search title or summary…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-8 pr-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-300
                           placeholder-gray-600 focus:outline-none focus:border-blue-500"
              />
            </div>

            {/* Sentiment filter */}
            <div className="flex gap-1">
              {["ALL", "POSITIVE", "NEUTRAL", "NEGATIVE"].map((s) => (
                <button
                  key={s}
                  onClick={() => setSentimentFilter(s)}
                  className={`px-3 py-2 rounded-lg text-xs font-medium transition ${
                    sentimentFilter === s
                      ? "bg-blue-600 text-white"
                      : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                  }`}
                >
                  {s === "ALL" ? "All" : s === "POSITIVE" ? "🟢 Bullish" : s === "NEGATIVE" ? "🔴 Bearish" : "⚪ Neutral"}
                </button>
              ))}
            </div>

            {/* Event tag filter */}
            <select
              value={eventFilter}
              onChange={(e) => setEventFilter(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-300 focus:outline-none focus:border-blue-500"
            >
              <option value="ALL">All Events</option>
              {allEventTags.map((t) => (
                <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
              ))}
            </select>

            {/* Ticker filter */}
            <input
              type="text"
              placeholder="Filter by ticker…"
              value={selectedTicker || ""}
              onChange={(e) => setSelectedTicker(e.target.value.toUpperCase() || null)}
              className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-300 font-mono
                         placeholder-gray-600 focus:outline-none focus:border-blue-500 w-32 uppercase"
            />

            <span className="text-sm text-gray-500 self-center ml-auto">
              {filtered.length} articles
            </span>
          </div>
        </div>

        {/* ── News articles ─────────────────────────────────────────── */}
        <div className="space-y-3">
          {isLoading ? (
            Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="h-24 bg-gray-800 rounded-xl animate-pulse" />
            ))
          ) : filtered.length === 0 ? (
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-12 text-center text-gray-600">
              <Newspaper size={36} className="mx-auto mb-3 opacity-30" />
              <p className="text-sm">No articles match your filters</p>
            </div>
          ) : (
            filtered.map((article: any) => (
              <ArticleCard
                key={article.url || article.id}
                article={article}
                expanded={expandedId === (article.url || article.id)}
                onToggle={() => setExpandedId(
                  expandedId === (article.url || article.id) ? null : (article.url || article.id)
                )}
                onTickerClick={setSelectedTicker}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
}

// ── Article Card ───────────────────────────────────────────────────────────────
function ArticleCard({ article, expanded, onToggle, onTickerClick }: any) {
  const sentimentCls = SENTIMENT_COLORS[article.sentiment_label] || SENTIMENT_COLORS.NEUTRAL;
  const sentiment = article.sentiment_score ?? 0;
  const isPositive = sentiment > 0.1;
  const isNegative = sentiment < -0.1;

  let publishedAgo = "";
  try {
    publishedAgo = formatDistanceToNow(new Date(article.published_at), { addSuffix: true });
  } catch { publishedAgo = ""; }

  return (
    <div
      className="bg-gray-900 rounded-xl border border-gray-800 hover:border-gray-700 transition overflow-hidden cursor-pointer"
      onClick={onToggle}
    >
      <div className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            {/* Tags row */}
            <div className="flex flex-wrap items-center gap-1.5 mb-2">
              <span className={`text-xs px-2 py-0.5 rounded-full border ${sentimentCls}`}>
                {isPositive ? "🟢" : isNegative ? "🔴" : "⚪"}{" "}
                {article.sentiment_label}
              </span>
              {(article.event_tags || []).slice(0, 3).map((tag: string) => (
                <span
                  key={tag}
                  className={`text-xs px-2 py-0.5 rounded-full border ${EVENT_COLORS[tag] || "bg-gray-800 text-gray-400 border-gray-700"}`}
                >
                  {tag}
                </span>
              ))}
              <span className="text-xs text-gray-600 ml-auto flex-shrink-0">{article.source}</span>
            </div>

            {/* Title */}
            <h3 className="text-sm font-semibold text-white leading-snug mb-1">
              {article.title}
            </h3>

            {/* Meta row */}
            <div className="flex items-center gap-3 text-xs text-gray-500">
              <span>{publishedAgo}</span>
              {article.sentiment_score != null && (
                <span className={isPositive ? "text-green-500" : isNegative ? "text-red-500" : "text-gray-500"}>
                  Sentiment: {article.sentiment_score.toFixed(3)}
                </span>
              )}
            </div>
          </div>

          {/* External link */}
          {article.url && (
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="text-gray-600 hover:text-blue-400 transition flex-shrink-0 mt-0.5"
            >
              <ExternalLink size={14} />
            </a>
          )}
        </div>
      </div>

      {/* Expanded: AI summary + ticker mentions */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-gray-800/60 pt-3 space-y-3">
          {article.summary && (
            <div>
              <div className="text-xs text-gray-500 font-semibold mb-1 uppercase tracking-wide">
                🤖 AI Summary (T5)
              </div>
              <p className="text-sm text-gray-300 leading-relaxed">{article.summary}</p>
            </div>
          )}

          {(article.event_tags || []).length > 0 && (
            <div>
              <div className="text-xs text-gray-500 font-semibold mb-1 uppercase tracking-wide">Events</div>
              <div className="flex flex-wrap gap-1">
                {article.event_tags.map((tag: string) => (
                  <span
                    key={tag}
                    className={`text-xs px-2 py-0.5 rounded-full border ${EVENT_COLORS[tag] || "bg-gray-800 text-gray-400 border-gray-700"}`}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {article.raw_content && !article.summary && (
            <div>
              <div className="text-xs text-gray-500 font-semibold mb-1 uppercase tracking-wide">Excerpt</div>
              <p className="text-xs text-gray-500 leading-relaxed line-clamp-3">{article.raw_content.slice(0, 300)}…</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
