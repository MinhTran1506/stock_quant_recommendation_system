"use client";
/**
 * pages/models/page.tsx — ML Model registry and monitoring dashboard.
 *
 * Sections:
 *  - Champion model cards per model type (TFT, N-BEATS, Meta, TCN)
 *  - Version history with metrics comparison
 *  - Directional accuracy time-series (live from DB)
 *  - Drift alerts panel
 *  - Manual retrain trigger (superuser only)
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
} from "recharts";
import {
  Brain, Trophy, Clock, Activity, AlertTriangle,
  ChevronDown, ChevronUp, ExternalLink,
} from "lucide-react";
import numeral from "numeral";

import { predictionsApi } from "@/utils/api";

const MODEL_TYPE_COLORS: Record<string, string> = {
  TFT:   "#3b82f6",
  NBEATS: "#8b5cf6",
  META:  "#22c55e",
  TCN:   "#f59e0b",
  GNN:   "#ec4899",
};

const MODEL_TYPE_LABELS: Record<string, string> = {
  TFT:   "Temporal Fusion Transformer",
  NBEATS: "N-BEATS / N-HiTS Ensemble",
  META:  "Meta Ranking (LightGBM)",
  TCN:   "Temporal CNN (Intraday)",
  GNN:   "Graph Neural Network",
};

export default function ModelsPage() {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { data: versions, isLoading } = useQuery({
    queryKey: ["model-versions"],
    queryFn: () => predictionsApi.modelRegistry().then((r) => r.data),
    refetchInterval: 60_000,
  });

  const champions = versions?.filter((v: any) => v.is_champion) ?? [];
  const byType = versions?.reduce((acc: any, v: any) => {
    const t = v.model_type || "OTHER";
    acc[t] = acc[t] ? [...acc[t], v] : [v];
    return acc;
  }, {}) ?? {};

  // Radar data for champion model comparison
  const radarData = champions.map((v: any) => ({
    model: v.model_type,
    "Dir. Accuracy": Math.round((v.metrics?.directional_accuracy ?? 0) * 100),
    "IC": Math.round(Math.abs(v.metrics?.ic ?? 0) * 100),
    "1 - MAE": Math.round(Math.max(0, 1 - (v.metrics?.mae ?? 0.1)) * 100),
    "Sharpe": Math.min(100, Math.round((v.metrics?.sharpe_ratio ?? 0) * 25)),
  }));

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="max-w-screen-xl mx-auto px-4 py-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Model Registry</h1>
          <p className="text-sm text-gray-400 mt-0.5">
            Champion models, version history, and performance monitoring
          </p>
        </div>

        {/* ── Champion cards ───────────────────────────────────────── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {isLoading
            ? Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="h-40 bg-gray-800 rounded-xl animate-pulse" />
              ))
            : champions.map((v: any) => (
                <ChampionCard key={v.id} version={v} />
              ))}
          {!isLoading && !champions.length && (
            <div className="col-span-3 bg-gray-900 rounded-xl border border-gray-800 p-12
                            text-center text-gray-600">
              <Brain size={36} className="mx-auto mb-3 opacity-30" />
              <p className="text-sm">
                No champion models registered yet.
                <br />
                Run the training pipeline to register models.
              </p>
            </div>
          )}
        </div>

        {/* ── Performance radar ────────────────────────────────────── */}
        {radarData.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <h2 className="text-sm font-semibold text-gray-300 mb-4">
                Champion Model Comparison
              </h2>
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#1e293b" />
                  <PolarAngleAxis
                    dataKey="model"
                    tick={{ fill: "#64748b", fontSize: 11 }}
                  />
                  {["Dir. Accuracy", "IC", "1 - MAE", "Sharpe"].map((key, i) => (
                    <Radar
                      key={key}
                      name={key}
                      dataKey={key}
                      stroke={Object.values(MODEL_TYPE_COLORS)[i]}
                      fill={Object.values(MODEL_TYPE_COLORS)[i]}
                      fillOpacity={0.15}
                    />
                  ))}
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Metrics table */}
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <h2 className="text-sm font-semibold text-gray-300 mb-4">
                Champion Metrics Summary
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-800 text-gray-500 uppercase">
                      <th className="py-2 text-left">Model</th>
                      <th className="py-2 text-right">Dir. Acc</th>
                      <th className="py-2 text-right">MAE</th>
                      <th className="py-2 text-right">IC</th>
                      <th className="py-2 text-left">Trained</th>
                    </tr>
                  </thead>
                  <tbody>
                    {champions.map((v: any) => (
                      <tr key={v.id} className="border-b border-gray-800/50">
                        <td className="py-2.5">
                          <span
                            className="inline-block w-2 h-2 rounded-full mr-2"
                            style={{ background: MODEL_TYPE_COLORS[v.model_type] || "#64748b" }}
                          />
                          <span className="text-white font-medium">{v.model_type}</span>
                        </td>
                        <td className="py-2.5 text-right font-mono">
                          {v.metrics?.directional_accuracy != null
                            ? `${(v.metrics.directional_accuracy * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                        <td className="py-2.5 text-right font-mono text-gray-400">
                          {v.metrics?.mae != null
                            ? numeral(v.metrics.mae).format("0.00000")
                            : "—"}
                        </td>
                        <td className="py-2.5 text-right font-mono">
                          {v.metrics?.ic != null
                            ? numeral(v.metrics.ic).format("0.000")
                            : "—"}
                        </td>
                        <td className="py-2.5 text-gray-500">
                          {v.trained_at?.slice(0, 10)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ── Version history by model type ─────────────────────────── */}
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-gray-300">Version History</h2>
          {Object.entries(byType).map(([modelType, vList]: [string, any]) => (
            <ModelTypeSection
              key={modelType}
              modelType={modelType}
              versions={vList}
              expandedId={expandedId}
              onToggle={setExpandedId}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────
function ChampionCard({ version }: { version: any }) {
  const color = MODEL_TYPE_COLORS[version.model_type] || "#64748b";
  const dirAcc = version.metrics?.directional_accuracy;
  const ic = version.metrics?.ic;

  return (
    <div
      className="bg-gray-900 rounded-xl border border-gray-800 p-4 relative overflow-hidden"
      style={{ borderTop: `3px solid ${color}` }}
    >
      {/* Champion badge */}
      <div className="absolute top-3 right-3">
        <Trophy size={14} className="text-yellow-400" />
      </div>

      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center text-white text-xs font-bold"
          style={{ background: `${color}25`, border: `1px solid ${color}40` }}
        >
          <Brain size={14} style={{ color }} />
        </div>
        <div>
          <div className="font-semibold text-white text-sm">{version.model_type}</div>
          <div className="text-xs text-gray-500">
            {MODEL_TYPE_LABELS[version.model_type] || version.name}
          </div>
        </div>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricPill
          label="Dir. Accuracy"
          value={dirAcc != null ? `${(dirAcc * 100).toFixed(1)}%` : "—"}
          good={dirAcc != null && dirAcc > 0.52}
        />
        <MetricPill
          label="IC"
          value={ic != null ? numeral(ic).format("0.000") : "—"}
          good={ic != null && ic > 0.05}
        />
      </div>

      {/* Version + date */}
      <div className="flex items-center justify-between text-xs text-gray-600">
        <span className="font-mono">v{version.version}</span>
        <span className="flex items-center gap-1">
          <Clock size={10} />
          {version.trained_at?.slice(0, 10)}
        </span>
      </div>

      {/* MLflow link */}
      {version.mlflow_run_id && (
        <a
          href={`http://localhost:5000/#/experiments/0/runs/${version.mlflow_run_id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-2 flex items-center gap-1 text-xs text-gray-600 hover:text-blue-400 transition"
        >
          <ExternalLink size={10} /> View in MLflow
        </a>
      )}
    </div>
  );
}

function MetricPill({ label, value, good }: any) {
  return (
    <div className="bg-gray-800 rounded-lg p-2">
      <div className="text-xs text-gray-500 mb-0.5">{label}</div>
      <div className={`text-sm font-mono font-semibold ${good ? "text-green-400" : "text-gray-300"}`}>
        {value}
      </div>
    </div>
  );
}

function ModelTypeSection({ modelType, versions, expandedId, onToggle }: any) {
  const champion = versions.find((v: any) => v.is_champion);
  const color = MODEL_TYPE_COLORS[modelType] || "#64748b";

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
      <button
        onClick={() =>
          onToggle(expandedId === modelType ? null : modelType)
        }
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-800/50 transition"
      >
        <div className="flex items-center gap-3">
          <div
            className="w-2 h-2 rounded-full"
            style={{ background: color }}
          />
          <span className="text-sm font-medium text-white">{modelType}</span>
          <span className="text-xs text-gray-500">
            {versions.length} version{versions.length !== 1 ? "s" : ""}
          </span>
          {champion && (
            <span className="flex items-center gap-1 text-xs text-yellow-400">
              <Trophy size={10} /> Champion: v{champion.version}
            </span>
          )}
        </div>
        {expandedId === modelType ? (
          <ChevronUp size={14} className="text-gray-500" />
        ) : (
          <ChevronDown size={14} className="text-gray-500" />
        )}
      </button>

      {expandedId === modelType && (
        <div className="border-t border-gray-800 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 uppercase tracking-wide">
                <th className="px-4 py-2.5 text-left">Version</th>
                <th className="px-4 py-2.5 text-right">Dir. Acc</th>
                <th className="px-4 py-2.5 text-right">MAE</th>
                <th className="px-4 py-2.5 text-right">RMSE</th>
                <th className="px-4 py-2.5 text-right">IC</th>
                <th className="px-4 py-2.5 text-left">Trained</th>
                <th className="px-4 py-2.5 text-left">Status</th>
              </tr>
            </thead>
            <tbody>
              {versions
                .sort((a: any, b: any) => new Date(b.trained_at).getTime() - new Date(a.trained_at).getTime())
                .map((v: any) => (
                  <tr
                    key={v.id}
                    className={`border-b border-gray-800/50 ${
                      v.is_champion ? "bg-yellow-950/10" : ""
                    }`}
                  >
                    <td className="px-4 py-2.5 font-mono text-white">
                      v{v.version}
                      {v.is_champion && (
                        <Trophy size={10} className="inline ml-1.5 text-yellow-400" />
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono">
                      {v.metrics?.directional_accuracy != null
                        ? `${(v.metrics.directional_accuracy * 100).toFixed(1)}%`
                        : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-gray-400">
                      {v.metrics?.mae != null
                        ? numeral(v.metrics.mae).format("0.00000") : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-gray-400">
                      {v.metrics?.rmse != null
                        ? numeral(v.metrics.rmse).format("0.00000") : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono">
                      {v.metrics?.ic != null
                        ? numeral(v.metrics.ic).format("0.000") : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-gray-500">
                      {v.trained_at?.slice(0, 10)}
                    </td>
                    <td className="px-4 py-2.5">
                      {v.is_champion ? (
                        <span className="text-yellow-400 font-medium">Champion</span>
                      ) : (
                        <span className="text-gray-600">Challenger</span>
                      )}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
