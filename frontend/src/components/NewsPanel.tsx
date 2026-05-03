"use client";
/**
 * components/NewsPanel.tsx — News feed with sentiment and event tags.
 */
import { useState } from "react";
import { ExternalLink, TrendingUp, TrendingDown, Minus, ChevronDown, ChevronUp } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

interface Article {
  id: string;
  title: string;
  source?: string;
  url?: string;
  published_at: string;
  summary?: string;
  sentiment_score?: number;
  sentiment_label?: string;
  event_tags?: string[];
  ticker?: string;
}

interface Props {
  articles: Article[];
  maxVisible?: number;
}

export default function NewsPanel({ articles, maxVisible = 6 }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [filter, setFilter] = useState<string>("ALL");

  const EVENT_FILTERS = ["ALL", "earnings", "dividend", "merger", "macro", "regulatory", "ipo"];

  const filtered = articles.filter(
    (a) => filter === "ALL" || (a.event_tags || []).includes(filter)
  );
  const visible = expanded ? filtered : filtered.slice(0, maxVisible);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h2 className="text-sm font-semibold text-gray-300">
          Market News
          <span className="ml-2 text-xs font-normal text-gray-500">
            AI-summarised · sentiment-tagged
          </span>
        </h2>

        {/* Event filter chips */}
        <div className="flex flex-wrap gap-1.5">
          {EVENT_FILTERS.map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2.5 py-0.5 rounded-full text-xs font-medium transition ${
                filter === f
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {visible.length === 0 ? (
        <p className="text-sm text-gray-600 py-4">No news matching filter</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {visible.map((article) => (
            <ArticleCard key={article.id || article.url} article={article} />
          ))}
        </div>
      )}

      {filtered.length > maxVisible && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="mt-3 w-full flex items-center justify-center gap-1.5 text-xs text-gray-500
                     hover:text-gray-300 transition py-2 border-t border-gray-800"
        >
          {expanded ? (
            <>
              <ChevronUp size={12} /> Show less
            </>
          ) : (
            <>
              <ChevronDown size={12} /> Show {filtered.length - maxVisible} more
            </>
          )}
        </button>
      )}
    </div>
  );
}

function ArticleCard({ article }: { article: Article }) {
  const [showSummary, setShowSummary] = useState(false);

  const sentiment = article.sentiment_score ?? 0;
  const sentimentConfig =
    sentiment > 0.1
      ? { label: "Bullish", color: "text-green-400 bg-green-950 border-green-800", icon: <TrendingUp size={10} /> }
      : sentiment < -0.1
      ? { label: "Bearish", color: "text-red-400 bg-red-950 border-red-800", icon: <TrendingDown size={10} /> }
      : { label: "Neutral", color: "text-gray-400 bg-gray-800 border-gray-700", icon: <Minus size={10} /> };

  let timeAgo = "";
  try {
    timeAgo = formatDistanceToNow(new Date(article.published_at), { addSuffix: true });
  } catch {}

  return (
    <div className="bg-gray-800/50 rounded-xl border border-gray-700/50 p-3 hover:border-gray-600 transition">
      {/* Tags row */}
      <div className="flex flex-wrap items-center gap-1.5 mb-2">
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs ${sentimentConfig.color}`}>
          {sentimentConfig.icon}
          {sentimentConfig.label}
        </span>
        {article.ticker && (
          <a
            href={`/stocks/${article.ticker}`}
            className="px-2 py-0.5 rounded-full bg-blue-950 border border-blue-800 text-xs text-blue-300 hover:text-blue-200 font-mono"
            onClick={(e) => e.stopPropagation()}
          >
            {article.ticker}
          </a>
        )}
        {(article.event_tags || []).slice(0, 2).map((tag) => (
          <span key={tag} className="px-2 py-0.5 rounded-full bg-gray-700 text-xs text-gray-400">
            {tag}
          </span>
        ))}
        <span className="text-xs text-gray-600 ml-auto">{timeAgo}</span>
      </div>

      {/* Title */}
      <h3 className="text-sm font-medium text-white leading-snug mb-1 line-clamp-2">
        {article.title}
      </h3>

      {/* AI Summary (expandable) */}
      {article.summary && (
        <div>
          <button
            onClick={() => setShowSummary((v) => !v)}
            className="text-xs text-blue-400 hover:text-blue-300 transition mb-1"
          >
            {showSummary ? "Hide summary" : "AI summary"}
          </button>
          {showSummary && (
            <p className="text-xs text-gray-400 leading-relaxed border-l-2 border-blue-800 pl-2">
              {article.summary}
            </p>
          )}
        </div>
      )}

      {/* Source + link */}
      <div className="flex items-center justify-between mt-2">
        <span className="text-xs text-gray-600">{article.source}</span>
        {article.url && (
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-blue-400 transition"
          >
            <ExternalLink size={12} />
          </a>
        )}
      </div>
    </div>
  );
}
