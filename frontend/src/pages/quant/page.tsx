"use client";
/**
 * pages/quant/page.tsx — Quantitative Trading Dashboard
 *
 * Sections:
 *  1. Market Regime Panel (HMM state + probability bars)
 *  2. Strategy Cards (6 strategies with live signal counts)
 *  3. Stat Arb signals table (cointegrated pairs + z-scores)
 *  4. Factor model rankings with breakdown
 *  5. Portfolio optimizer (method selector + weight visualisation)
 *  6. Risk dashboard (VaR, CVaR, drawdown, breach alerts)
 */
import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, PieChart, Pie, Legend, ScatterChart, Scatter, ZAxis,
} from "recharts";
import {
  TrendingUp, TrendingDown, Minus, Activity, AlertTriangle,
  BarChart2, Brain, GitBranch, Shield, FlaskConical,
  RefreshCw, ChevronDown, ChevronUp, Zap,
} from "lucide-react";
import numeral from "numeral";
import toast from "react-hot-toast";

import { api } from "@/utils/api";

const quantApi = {
  strategies:       () => api.get("/quant/strategies").then(r => r.data),
  regime:           () => api.get("/quant/regime").then(r => r.data),
  statArbSignals:   () => api.get("/quant/stat-arb/signals?limit=30").then(r => r.data),
  statArbScan:      (b: any) => api.post("/quant/stat-arb/scan", b).then(r => r.data),
  factorRankings:   () => api.get("/quant/factor-model/rankings?top_n=50").then(r => r.data),
  momentumSignals:  () => api.get("/quant/momentum/signals?top_n=20").then(r => r.data),
  optimizePortfolio:(b: any) => api.post("/quant/portfolio/optimize", b).then(r => r.data),
  riskReport:       () => api.get("/quant/risk/report").then(r => r.data),
};

const REGIME_COLORS: Record<string, string> = {
  BULL: "#22c55e", SIDEWAYS: "#f59e0b", BEAR: "#ef4444",
};

const STRATEGY_ICONS: Record<string, any> = {
  stat_arb: GitBranch, factor_model: BarChart2,
  momentum_regime: TrendingUp, rl_agent: Brain,
  black_litterman: FlaskConical,
};

export default function QuantPage() {
  const [activeTab, setActiveTab] = useState<"overview"|"statarb"|"factors"|"momentum"|"portfolio"|"risk">("overview");
  const [optimizeMethod, setOptimizeMethod] = useState("black_litterman");
  const [selectedTickers, setSelectedTickers] = useState("VNM,VIC,HPG,FPT,TCB,MBB,VPB,STB,BID,VCB");

  const { data: strategies } = useQuery({ queryKey: ["quant-strategies"], queryFn: quantApi.strategies, staleTime: 300_000 });
  const { data: regime, isLoading: regimeLoading, refetch: refetchRegime } = useQuery({ queryKey: ["market-regime"], queryFn: quantApi.regime, refetchInterval: 300_000 });
  const { data: statArbSignals } = useQuery({ queryKey: ["stat-arb-signals"], queryFn: quantApi.statArbSignals, enabled: activeTab === "statarb" || activeTab === "overview", refetchInterval: 60_000 });
  const { data: factorRankings } = useQuery({ queryKey: ["factor-rankings"], queryFn: quantApi.factorRankings, enabled: activeTab === "factors" || activeTab === "overview" });
  const { data: momentumSignals } = useQuery({ queryKey: ["momentum-signals"], queryFn: quantApi.momentumSignals, enabled: activeTab === "momentum" || activeTab === "overview" });
  const { data: riskReport } = useQuery({ queryKey: ["risk-report"], queryFn: quantApi.riskReport, enabled: activeTab === "risk" || activeTab === "overview", refetchInterval: 120_000 });

  const optimizeMutation = useMutation({
    mutationFn: (payload: any) => quantApi.optimizePortfolio(payload),
    onSuccess: () => toast.success("Portfolio optimised"),
    onError: () => toast.error("Optimisation failed"),
  });

  const scanMutation = useMutation({
    mutationFn: (payload: any) => quantApi.statArbScan(payload),
    onSuccess: () => toast.success("Pair scan complete"),
  });

  const regimeColor = REGIME_COLORS[regime?.regime] || "#64748b";
  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "statarb", label: "Stat Arb" },
    { id: "factors", label: "Factors" },
    { id: "momentum", label: "Momentum" },
    { id: "portfolio", label: "Portfolio" },
    { id: "risk", label: "Risk" },
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="max-w-screen-2xl mx-auto px-4 py-6 space-y-6">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              <Zap size={22} className="text-yellow-400" />
              Quantitative Trading Engine
            </h1>
            <p className="text-sm text-gray-400 mt-0.5">
              Hedge-fund grade algorithms: Stat Arb · Factor Models · Momentum · RL · Black-Litterman
            </p>
          </div>
          <button onClick={() => refetchRegime()} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm text-gray-300">
            <RefreshCw size={13} /> Refresh
          </button>
        </div>

        {/* ── Regime banner ──────────────────────────────────────────── */}
        <RegimeBanner regime={regime} loading={regimeLoading} color={regimeColor} />

        {/* ── Tab navigation ─────────────────────────────────────────── */}
        <div className="flex gap-1 bg-gray-900 rounded-xl border border-gray-800 p-1">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition
                          ${activeTab === tab.id
                            ? "bg-blue-600 text-white"
                            : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
                          }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* ── Overview tab ───────────────────────────────────────────── */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            {/* Strategy cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {(strategies || []).map((s: any) => (
                <StrategyCard key={s.id} strategy={s} />
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Top stat arb pairs */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
                <h2 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                  <GitBranch size={14} /> Top Stat Arb Signals
                </h2>
                {statArbSignals?.length ? (
                  <div className="space-y-1.5 max-h-64 overflow-y-auto">
                    {statArbSignals.slice(0, 8).map((s: any) => (
                      <StatArbRow key={`${s.ticker_a}-${s.ticker_b}`} signal={s} compact />
                    ))}
                  </div>
                ) : <EmptyState text="Run stat arb scan to see signals" />}
              </div>

              {/* Top factor stocks */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
                <h2 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                  <BarChart2 size={14} /> Top Factor Stocks
                </h2>
                {factorRankings?.length ? (
                  <div className="space-y-1.5 max-h-64 overflow-y-auto">
                    {factorRankings.slice(0, 8).map((r: any) => (
                      <FactorRow key={r.ticker} ranking={r} compact />
                    ))}
                  </div>
                ) : <EmptyState text="Factor scores loading…" />}
              </div>
            </div>

            {/* Risk summary */}
            {riskReport && <RiskSummaryBar report={riskReport} />}
          </div>
        )}

        {/* ── Stat Arb tab ────────────────────────────────────────────── */}
        {activeTab === "statarb" && (
          <StatArbTab
            signals={statArbSignals || []}
            onScan={(params) => scanMutation.mutate(params)}
            scanResult={scanMutation.data}
            scanning={scanMutation.isPending}
          />
        )}

        {/* ── Factors tab ─────────────────────────────────────────────── */}
        {activeTab === "factors" && (
          <FactorsTab rankings={factorRankings || []} />
        )}

        {/* ── Momentum tab ────────────────────────────────────────────── */}
        {activeTab === "momentum" && (
          <MomentumTab data={momentumSignals} />
        )}

        {/* ── Portfolio tab ───────────────────────────────────────────── */}
        {activeTab === "portfolio" && (
          <PortfolioTab
            method={optimizeMethod}
            onMethodChange={setOptimizeMethod}
            tickers={selectedTickers}
            onTickersChange={setSelectedTickers}
            onOptimize={() => optimizeMutation.mutate({
              tickers: selectedTickers.split(",").map(t => t.trim()).filter(Boolean),
              method: optimizeMethod,
              max_weight: 0.15,
            })}
            result={optimizeMutation.data}
            loading={optimizeMutation.isPending}
          />
        )}

        {/* ── Risk tab ────────────────────────────────────────────────── */}
        {activeTab === "risk" && <RiskTab report={riskReport} />}
      </div>
    </div>
  );
}

// ── Subcomponents ──────────────────────────────────────────────────────────────
function RegimeBanner({ regime, loading, color }: any) {
  if (loading) return <div className="h-20 bg-gray-800 rounded-xl animate-pulse" />;
  if (!regime) return null;
  const Icon = regime?.regime === "BULL" ? TrendingUp : regime?.regime === "BEAR" ? TrendingDown : Minus;
  return (
    <div className="rounded-xl border p-4 flex flex-wrap items-center justify-between gap-4"
         style={{ borderColor: `${color}40`, background: `${color}10` }}>
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ background: `${color}20` }}>
          <Icon size={20} style={{ color }} />
        </div>
        <div>
          <div className="font-bold text-white text-lg">{regime.regime} REGIME</div>
          <div className="text-xs text-gray-400">{regime.description}</div>
        </div>
      </div>
      <div className="flex gap-4 text-sm">
        {[
          { label: "Bull", value: regime.bull_probability, color: "#22c55e" },
          { label: "Sideways", value: regime.sideways_probability, color: "#f59e0b" },
          { label: "Bear", value: regime.bear_probability, color: "#ef4444" },
        ].map(({ label, value, color: c }) => (
          <div key={label} className="text-center">
            <div className="text-xs text-gray-500">{label}</div>
            <div className="font-mono font-bold" style={{ color: c }}>
              {(value * 100).toFixed(0)}%
            </div>
          </div>
        ))}
        <div className="text-center">
          <div className="text-xs text-gray-500">Momentum Scale</div>
          <div className="font-mono font-bold text-blue-400">
            {(regime.momentum_scalar * 100).toFixed(0)}%
          </div>
        </div>
      </div>
    </div>
  );
}

function StrategyCard({ strategy }: { strategy: any }) {
  const Icon = STRATEGY_ICONS[strategy.id] || Activity;
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 hover:border-gray-700 transition">
      <div className="flex items-start justify-between mb-3">
        <div className="w-9 h-9 rounded-lg bg-blue-950 border border-blue-800 flex items-center justify-center">
          <Icon size={16} className="text-blue-400" />
        </div>
        <span className="text-xs text-gray-600 bg-gray-800 px-2 py-0.5 rounded-full">
          {strategy.horizon}
        </span>
      </div>
      <div className="font-semibold text-white text-sm mb-1">{strategy.name}</div>
      <p className="text-xs text-gray-500 mb-3 leading-relaxed line-clamp-2">{strategy.description}</p>
      <div className="flex flex-wrap gap-1">
        {strategy.papers.slice(0, 2).map((p: string) => (
          <span key={p} className="text-xs px-1.5 py-0.5 rounded bg-gray-800 text-gray-500">{p}</span>
        ))}
      </div>
    </div>
  );
}

function StatArbRow({ signal, compact = false }: { signal: any; compact?: boolean }) {
  const absZ = Math.abs(signal.z_score);
  const zColor = absZ > 3 ? "text-red-400" : absZ > 2 ? "text-yellow-400" : "text-green-400";
  const sigText = signal.signal === 1 ? "LONG A" : signal.signal === -1 ? "SHORT A" : "FLAT";
  const sigColor = signal.signal === 1 ? "text-green-400" : signal.signal === -1 ? "text-red-400" : "text-gray-500";
  return (
    <div className="flex items-center justify-between px-3 py-2 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition text-xs">
      <div className="flex items-center gap-2">
        <span className="font-mono font-bold text-white">{signal.ticker_a}</span>
        <span className="text-gray-600">↔</span>
        <span className="font-mono text-gray-300">{signal.ticker_b}</span>
      </div>
      <div className="flex items-center gap-3">
        <span className={`font-mono font-bold ${zColor}`}>z={signal.z_score.toFixed(2)}</span>
        <span className="text-gray-500">HL={signal.half_life?.toFixed(1)}d</span>
        {!compact && <span className={`font-medium ${sigColor}`}>{sigText}</span>}
      </div>
    </div>
  );
}

function FactorRow({ ranking, compact = false }: { ranking: any; compact?: boolean }) {
  const scoreColor = ranking.score >= 70 ? "text-green-400" : ranking.score >= 50 ? "text-yellow-400" : "text-red-400";
  return (
    <div className="flex items-center justify-between px-3 py-2 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition text-xs">
      <div className="flex items-center gap-2">
        <span className="text-gray-500 w-5 text-right">#{ranking.rank}</span>
        <span className="font-mono font-bold text-white">{ranking.ticker}</span>
      </div>
      <div className="flex items-center gap-3">
        <div className="w-20 bg-gray-700 rounded-full h-1.5">
          <div className="h-1.5 rounded-full bg-blue-500" style={{ width: `${ranking.score}%` }} />
        </div>
        <span className={`font-mono font-bold ${scoreColor}`}>{ranking.score?.toFixed(1)}</span>
      </div>
    </div>
  );
}

function RiskSummaryBar({ report }: { report: any }) {
  const hasBreaches = report.breaches?.length > 0;
  const action = report.action_required;
  const actionColor = action === "HALT" ? "bg-red-950 border-red-700" : action === "REDUCE" ? "bg-amber-950 border-amber-700" : "bg-green-950 border-green-700";
  const actionText = action === "HALT" ? "🛑 HALT — exceed hard stop" : action === "REDUCE" ? "⚠️ REDUCE exposure" : "✅ All limits OK";
  return (
    <div className={`rounded-xl border p-4 ${actionColor}`}>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="font-semibold text-sm text-white">{actionText}</div>
        <div className="flex gap-4 text-xs">
          {[
            { label: "VaR 95", value: `${(report.var_95_1d * 100).toFixed(2)}%` },
            { label: "CVaR 95", value: `${(report.cvar_95_1d * 100).toFixed(2)}%` },
            { label: "Drawdown", value: `${(report.current_drawdown * 100).toFixed(2)}%` },
            { label: "Sharpe", value: report.sharpe_ratio?.toFixed(2) },
          ].map(({ label, value }) => (
            <div key={label} className="text-center">
              <div className="text-gray-500">{label}</div>
              <div className="font-mono font-bold text-white">{value}</div>
            </div>
          ))}
        </div>
      </div>
      {hasBreaches && (
        <div className="mt-3 flex flex-wrap gap-1">
          {report.breaches.map((b: string, i: number) => (
            <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-red-900 text-red-300">{b}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function StatArbTab({ signals, onScan, scanResult, scanning }: any) {
  const [minCorr, setMinCorr] = useState(0.65);
  const [maxHL, setMaxHL] = useState(63);
  return (
    <div className="space-y-4">
      {/* Scan controls */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 flex flex-wrap items-end gap-4">
        <div>
          <label className="text-xs text-gray-500 block mb-1">Min Correlation</label>
          <input type="number" step={0.05} min={0.5} max={0.95} value={minCorr}
                 onChange={e => setMinCorr(+e.target.value)}
                 className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500 w-28" />
        </div>
        <div>
          <label className="text-xs text-gray-500 block mb-1">Max Half-Life (days)</label>
          <input type="number" step={5} min={5} max={120} value={maxHL}
                 onChange={e => setMaxHL(+e.target.value)}
                 className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500 w-28" />
        </div>
        <button onClick={() => onScan({ min_corr: minCorr, max_half_life_days: maxHL, lookback_days: 252 })}
                disabled={scanning}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg px-4 py-2 text-sm font-medium">
          {scanning ? <div className="h-4 w-4 rounded-full border-2 border-white border-t-transparent animate-spin" /> : <Activity size={14} />}
          Scan Universe
        </button>
        {scanResult && (
          <span className="text-sm text-green-400">Found {scanResult.n_pairs_found} cointegrated pairs</span>
        )}
      </div>

      {/* Signals table */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800">
          <h2 className="text-sm font-semibold text-gray-300">Active Spread Signals</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 uppercase tracking-wide">
                <th className="px-4 py-3 text-left">Pair</th>
                <th className="px-4 py-3 text-right">Spread</th>
                <th className="px-4 py-3 text-right">Z-Score</th>
                <th className="px-4 py-3 text-right">Hedge β</th>
                <th className="px-4 py-3 text-right">Half-Life</th>
                <th className="px-4 py-3 text-left">Signal</th>
              </tr>
            </thead>
            <tbody>
              {signals.length ? signals.map((s: any) => {
                const absZ = Math.abs(s.z_score);
                const zColor = absZ > 3 ? "text-red-400" : absZ > 2 ? "text-yellow-400" : "text-gray-300";
                const sig = s.signal === 1 ? { label: "LONG A", cls: "text-green-400 bg-green-950" }
                           : s.signal === -1 ? { label: "SHORT A", cls: "text-red-400 bg-red-950" }
                           : { label: "FLAT", cls: "text-gray-500 bg-gray-800" };
                return (
                  <tr key={`${s.ticker_a}-${s.ticker_b}`} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="px-4 py-2.5">
                      <span className="font-mono font-bold text-white">{s.ticker_a}</span>
                      <span className="text-gray-500 mx-1">↔</span>
                      <span className="font-mono text-gray-300">{s.ticker_b}</span>
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-gray-400">{s.spread.toFixed(5)}</td>
                    <td className={`px-4 py-2.5 text-right font-mono font-bold ${zColor}`}>{s.z_score.toFixed(3)}</td>
                    <td className="px-4 py-2.5 text-right font-mono text-gray-400">{s.hedge_ratio.toFixed(4)}</td>
                    <td className="px-4 py-2.5 text-right text-gray-400">{s.half_life.toFixed(1)}d</td>
                    <td className="px-4 py-2.5">
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${sig.cls}`}>{sig.label}</span>
                    </td>
                  </tr>
                );
              }) : (
                <tr><td colSpan={6} className="px-4 py-8 text-center text-gray-600">No signals — click "Scan Universe" first</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function FactorsTab({ rankings }: { rankings: any[] }) {
  const factorKeys = ["factor_mom", "factor_value", "factor_quality", "factor_low_vol", "factor_growth"];
  const factorLabels: Record<string, string> = { factor_mom: "Momentum", factor_value: "Value", factor_quality: "Quality", factor_low_vol: "Low Vol", factor_growth: "Growth" };

  // Radar data for top-5 stocks
  const radarData = rankings.slice(0, 5).map((r: any) => ({
    ticker: r.ticker,
    ...Object.fromEntries(factorKeys.map(k => [factorLabels[k], r[k] ?? 50])),
  }));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Factor radar */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">Top-5 Factor Comparison</h2>
        {radarData.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#1e293b" />
              <PolarAngleAxis dataKey="ticker" tick={{ fill: "#64748b", fontSize: 10 }} />
              {factorKeys.slice(0, 3).map((k, i) => (
                <Radar key={k} name={factorLabels[k]} dataKey={factorLabels[k]}
                       stroke={["#3b82f6", "#22c55e", "#f59e0b"][i]} fill={["#3b82f6", "#22c55e", "#f59e0b"][i]} fillOpacity={0.1} />
              ))}
            </RadarChart>
          </ResponsiveContainer>
        ) : <EmptyState text="Loading factor scores…" />}
      </div>

      {/* Rankings table */}
      <div className="lg:col-span-2 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800">
          <h2 className="text-sm font-semibold text-gray-300">Factor Rankings — All Stocks</h2>
        </div>
        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-gray-900">
              <tr className="border-b border-gray-800 text-gray-500 uppercase tracking-wide">
                <th className="px-4 py-3 text-left">#</th>
                <th className="px-4 py-3 text-left">Ticker</th>
                <th className="px-4 py-3 text-right">Score</th>
                {factorKeys.map(k => (
                  <th key={k} className="px-3 py-3 text-right hidden md:table-cell">{factorLabels[k].slice(0,3)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rankings.map((r: any) => (
                <tr key={r.ticker} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td className="px-4 py-2 text-gray-600">{r.rank}</td>
                  <td className="px-4 py-2 font-mono font-bold text-white">{r.ticker}</td>
                  <td className="px-4 py-2 text-right">
                    <div className="flex items-center justify-end gap-1.5">
                      <div className="w-12 h-1.5 bg-gray-800 rounded-full">
                        <div className="h-1.5 rounded-full bg-blue-500" style={{ width: `${r.score}%` }} />
                      </div>
                      <span className="font-mono text-white">{r.score?.toFixed(1)}</span>
                    </div>
                  </td>
                  {factorKeys.map(k => (
                    <td key={k} className="px-3 py-2 text-right font-mono text-gray-400 hidden md:table-cell">
                      {r[k]?.toFixed(0) ?? "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function MomentumTab({ data }: { data: any }) {
  if (!data) return <EmptyState text="Loading momentum signals…" />;
  const signals = data.signals || [];
  const regimeColor = REGIME_COLORS[data.regime] || "#64748b";
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 p-4 rounded-xl border"
           style={{ borderColor: `${regimeColor}40`, background: `${regimeColor}10` }}>
        <div className="font-bold text-white">Regime: {data.regime}</div>
        <div className="text-sm text-gray-400">Momentum scalar: <span className="font-mono font-bold text-blue-400">{(data.momentum_scalar * 100).toFixed(0)}%</span></div>
        <div className="text-sm text-gray-400">{data.n_signals} active signals</div>
      </div>
      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase">
              <th className="px-4 py-3 text-left">Ticker</th>
              <th className="px-4 py-3 text-right">Weight</th>
              <th className="px-4 py-3 text-center">CS Signal</th>
              <th className="px-4 py-3 text-center">TS Signal</th>
              <th className="px-4 py-3 text-left">Regime</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((s: any) => (
              <tr key={s.ticker} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="px-4 py-2.5 font-mono font-bold text-white">{s.ticker}</td>
                <td className="px-4 py-2.5 text-right font-mono text-blue-400">{(s.weight * 100).toFixed(2)}%</td>
                <td className="px-4 py-2.5 text-center">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${s.cs_signal === 1 ? "bg-green-950 text-green-400" : "bg-gray-800 text-gray-500"}`}>
                    {s.cs_signal === 1 ? "▲" : "—"}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-center">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${s.ts_signal === 1 ? "bg-blue-950 text-blue-400" : "bg-gray-800 text-gray-500"}`}>
                    {s.ts_signal === 1 ? "▲" : "—"}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-xs text-gray-500">{s.regime}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PortfolioTab({ method, onMethodChange, tickers, onTickersChange, onOptimize, result, loading }: any) {
  const methods = [
    { id: "black_litterman", label: "Black-Litterman", desc: "Market equilibrium + ML views" },
    { id: "risk_parity", label: "Risk Parity", desc: "Equal risk contribution (ERC)" },
    { id: "mean_variance", label: "Mean-Variance", desc: "Max Sharpe (Markowitz)" },
    { id: "equal_weight", label: "Equal Weight", desc: "1/N baseline" },
  ];

  const pieData = result
    ? Object.entries(result.weights)
        .filter(([, w]: any) => w > 0.001)
        .map(([t, w]: any) => ({ name: t, value: +(w * 100).toFixed(2) }))
    : [];

  const COLORS = ["#3b82f6","#22c55e","#f59e0b","#ef4444","#8b5cf6","#ec4899","#14b8a6","#f97316"];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Controls */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 space-y-4">
        <h2 className="text-sm font-semibold text-gray-300">Portfolio Optimizer</h2>
        <div>
          <label className="text-xs text-gray-500 block mb-1">Tickers (comma-separated)</label>
          <textarea value={tickers} onChange={e => onTickersChange(e.target.value)} rows={3}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-300 font-mono focus:outline-none focus:border-blue-500 resize-none" />
        </div>
        <div className="space-y-2">
          <label className="text-xs text-gray-500 block">Method</label>
          {methods.map(m => (
            <button key={m.id} onClick={() => onMethodChange(m.id)}
                    className={`w-full text-left px-3 py-2 rounded-lg transition text-xs ${method === m.id ? "bg-blue-950 border border-blue-700 text-blue-300" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}>
              <div className="font-medium">{m.label}</div>
              <div className="text-gray-500">{m.desc}</div>
            </button>
          ))}
        </div>
        <button onClick={onOptimize} disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg py-2.5 text-sm font-semibold">
          {loading ? "Optimising…" : "Run Optimization"}
        </button>
      </div>

      {/* Results */}
      <div className="lg:col-span-2 space-y-4">
        {result ? (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: "Ann. Return", value: `${(result.metrics.annualised_return * 100).toFixed(2)}%` },
                { label: "Ann. Vol", value: `${(result.metrics.annualised_vol * 100).toFixed(2)}%` },
                { label: "Sharpe", value: result.metrics.sharpe_ratio?.toFixed(3) },
                { label: "Max DD", value: `${(result.metrics.max_drawdown * 100).toFixed(2)}%` },
              ].map(({ label, value }) => (
                <div key={label} className="bg-gray-900 rounded-xl border border-gray-800 p-3">
                  <div className="text-xs text-gray-500">{label}</div>
                  <div className="text-lg font-mono font-bold text-white">{value}</div>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Pie chart */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">Weight Allocation</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={80} label={e => e.name}>
                      {pieData.map((_: any, i: number) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip formatter={(v: any) => [`${v}%`, "Weight"]} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              {/* Weights table */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 overflow-y-auto max-h-64">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">Weights</h3>
                <table className="w-full text-xs">
                  <tbody>
                    {Object.entries(result.weights)
                      .filter(([, w]: any) => w > 0.001)
                      .sort(([, a]: any, [, b]: any) => b - a)
                      .map(([ticker, w]: any, i) => (
                        <tr key={ticker} className="border-b border-gray-800/50">
                          <td className="py-1.5 font-mono font-bold text-white">{ticker}</td>
                          <td className="py-1.5 text-right">
                            <div className="flex items-center justify-end gap-2">
                              <div className="w-16 h-1.5 bg-gray-800 rounded-full">
                                <div className="h-1.5 rounded-full" style={{ width: `${w * 100 / 0.15}%`, background: COLORS[i % COLORS.length] }} />
                              </div>
                              <span className="font-mono text-gray-300">{(w * 100).toFixed(1)}%</span>
                            </div>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        ) : (
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-16 flex flex-col items-center justify-center text-gray-600">
            <FlaskConical size={36} className="mb-3 opacity-30" />
            <p className="text-sm">Configure and run the optimizer to see results</p>
          </div>
        )}
      </div>
    </div>
  );
}

function RiskTab({ report }: { report: any }) {
  if (!report) return <EmptyState text="Loading risk metrics…" />;
  return (
    <div className="space-y-4">
      <RiskSummaryBar report={report} />
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: "VaR 95% (1-day)", value: `${(report.var_95_1d * 100).toFixed(3)}%`, desc: "Max expected loss (95% conf.)" },
          { label: "CVaR 95% (1-day)", value: `${(report.cvar_95_1d * 100).toFixed(3)}%`, desc: "Expected loss beyond VaR" },
          { label: "Annualised Vol", value: `${(report.annualised_vol * 100).toFixed(2)}%`, desc: "Portfolio std deviation" },
          { label: "Beta", value: report.beta?.toFixed(3), desc: "Market sensitivity" },
          { label: "Sharpe Ratio", value: report.sharpe_ratio?.toFixed(3), desc: "Risk-adjusted return" },
          { label: "Sortino Ratio", value: report.sortino_ratio?.toFixed(3), desc: "Downside-adj. return" },
          { label: "Current Drawdown", value: `${(report.current_drawdown * 100).toFixed(2)}%`, desc: "From peak" },
          { label: "Max Drawdown", value: `${(report.max_drawdown * 100).toFixed(2)}%`, desc: "Worst peak-to-trough" },
        ].map(({ label, value, desc }) => (
          <div key={label} className="bg-gray-900 rounded-xl border border-gray-800 p-4">
            <div className="text-xs text-gray-500 mb-0.5">{label}</div>
            <div className="text-xl font-mono font-bold text-white">{value}</div>
            <div className="text-xs text-gray-600 mt-0.5">{desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="py-10 flex flex-col items-center justify-center text-gray-600 text-sm">
      <Activity size={28} className="mb-2 opacity-30" />
      {text}
    </div>
  );
}
