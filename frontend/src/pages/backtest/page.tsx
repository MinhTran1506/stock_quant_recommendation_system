"use client";
/**
 * pages/backtest/page.tsx — Backtest submission and results viewer.
 */
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell
} from "recharts";
import { Play, Trash2, ChevronDown, RefreshCw, TrendingUp, Shield, Activity } from "lucide-react";
import numeral from "numeral";
import toast from "react-hot-toast";

import { backtestApi } from "@/utils/api";

const DEFAULT_CONFIG = {
  name: "Test Strategy",
  start_date: "2020-01-01",
  end_date: "2024-01-01",
  initial_capital: 1_000_000_000,
  commission_pct: 0.0015,
  slippage_pct: 0.001,
  stop_loss_pct: 0.07,
  max_position_pct: 0.10,
  max_positions: 20,
  engine: "vectorbt",
  min_score: 60,
  horizon_days: 5,
};

export default function BacktestPage() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // ── Fetch backtest list ───────────────────────────────────────────────
  const { data: runs, isLoading: runsLoading, refetch } = useQuery({
    queryKey: ["backtests"],
    queryFn: () => backtestApi.list().then((r) => r.data),
    refetchInterval: 5000,
  });

  // ── Fetch selected run detail ─────────────────────────────────────────
  const { data: runDetail, isLoading: detailLoading } = useQuery({
    queryKey: ["backtest", selectedRun],
    queryFn: () => backtestApi.get(selectedRun!).then((r) => r.data),
    enabled: !!selectedRun,
    refetchInterval: (data) => (data?.status === "RUNNING" ? 3000 : false),
  });

  // ── Submit mutation ───────────────────────────────────────────────────
  const submitMutation = useMutation({
    mutationFn: (payload: typeof DEFAULT_CONFIG) => backtestApi.submit(payload),
    onSuccess: (res) => {
      toast.success(`Backtest "${config.name}" submitted!`);
      setSelectedRun(res.data.id);
      queryClient.invalidateQueries({ queryKey: ["backtests"] });
    },
    onError: () => toast.error("Failed to submit backtest"),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => backtestApi.delete(id),
    onSuccess: () => {
      toast.success("Deleted");
      setSelectedRun(null);
      queryClient.invalidateQueries({ queryKey: ["backtests"] });
    },
  });

  const metrics = runDetail?.summary_metrics;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="max-w-screen-2xl mx-auto px-4 py-6">
        <h1 className="text-2xl font-bold text-white mb-6">
          Backtest Sandbox
        </h1>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* ── Config panel ──────────────────────────────────────────── */}
          <div className="xl:col-span-1 space-y-4">
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <h2 className="text-sm font-semibold text-gray-300 mb-4">
                Strategy Config
              </h2>
              <ConfigForm
                config={config}
                onChange={setConfig}
                onSubmit={() => submitMutation.mutate(config)}
                isLoading={submitMutation.isPending}
              />
            </div>

            {/* Run history */}
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-gray-300">Run History</h2>
                <button onClick={() => refetch()} className="text-gray-600 hover:text-gray-400">
                  <RefreshCw size={13} />
                </button>
              </div>
              {runsLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-12 bg-gray-800 rounded-lg animate-pulse" />
                  ))}
                </div>
              ) : (
                <div className="space-y-1 max-h-64 overflow-y-auto">
                  {runs?.map((run: any) => (
                    <RunListItem
                      key={run.id}
                      run={run}
                      selected={selectedRun === run.id}
                      onClick={() => setSelectedRun(run.id)}
                      onDelete={() => deleteMutation.mutate(run.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* ── Results panel ─────────────────────────────────────────── */}
          <div className="xl:col-span-3 space-y-4">
            {!selectedRun && (
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-16 flex flex-col items-center justify-center text-gray-600">
                <Activity size={40} className="mb-3 opacity-30" />
                <p className="text-sm">Submit a backtest or select a run from history</p>
              </div>
            )}

            {selectedRun && (
              <>
                {/* Status banner */}
                {runDetail?.status === "RUNNING" && (
                  <div className="bg-blue-950 border border-blue-800 rounded-xl p-4 flex items-center gap-3">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400" />
                    <span className="text-sm text-blue-300">Backtest running…</span>
                  </div>
                )}
                {runDetail?.status === "FAILED" && (
                  <div className="bg-red-950 border border-red-800 rounded-xl p-4 text-sm text-red-300">
                    Backtest failed: {runDetail.error_message}
                  </div>
                )}

                {/* Metric cards */}
                {metrics && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <MetricTile
                      label="Total Return"
                      value={`${metrics.total_return_pct > 0 ? "+" : ""}${numeral(metrics.total_return_pct).format("0.00")}%`}
                      positive={metrics.total_return_pct > 0}
                      icon={<TrendingUp size={14} />}
                    />
                    <MetricTile label="Sharpe Ratio" value={numeral(metrics.sharpe_ratio).format("0.000")} />
                    <MetricTile
                      label="Max Drawdown"
                      value={`${numeral(metrics.max_drawdown_pct).format("0.00")}%`}
                      positive={false}
                      icon={<Shield size={14} />}
                    />
                    <MetricTile label="Win Rate" value={`${numeral(metrics.win_rate).format("0.0")}%`} />
                    <MetricTile label="Ann. Return" value={`${numeral(metrics.annualised_return_pct).format("0.00")}%`} positive={metrics.annualised_return_pct > 0} />
                    <MetricTile label="Sortino" value={numeral(metrics.sortino_ratio).format("0.000")} />
                    <MetricTile label="Total Trades" value={numeral(metrics.total_trades).format("0")} />
                    <MetricTile label="Profit Factor" value={numeral(metrics.profit_factor).format("0.000")} positive={metrics.profit_factor > 1} />
                  </div>
                )}

                {/* Equity curve */}
                {runDetail?.equity_curve?.length > 0 && (
                  <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
                    <h3 className="text-sm font-semibold text-gray-300 mb-3">Equity Curve</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={runDetail.equity_curve}>
                        <CartesianGrid stroke="#1e293b" />
                        <XAxis
                          dataKey="date"
                          tick={{ fill: "#64748b", fontSize: 10 }}
                          interval={Math.floor(runDetail.equity_curve.length / 6)}
                        />
                        <YAxis
                          tick={{ fill: "#64748b", fontSize: 10 }}
                          tickFormatter={(v) => numeral(v / 1e9).format("0.0") + "B"}
                        />
                        <Tooltip
                          formatter={(v: any) => [numeral(v).format("0,0") + " VND", "Portfolio Value"]}
                          contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Monthly returns bar chart */}
                {runDetail?.summary_metrics?.monthly_returns?.length > 0 && (
                  <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
                    <h3 className="text-sm font-semibold text-gray-300 mb-3">Monthly Returns</h3>
                    <ResponsiveContainer width="100%" height={160}>
                      <BarChart data={runDetail.summary_metrics.monthly_returns}>
                        <XAxis dataKey="month" tick={{ fill: "#64748b", fontSize: 9 }} />
                        <YAxis tickFormatter={(v) => `${v}%`} tick={{ fill: "#64748b", fontSize: 9 }} />
                        <Tooltip formatter={(v: any) => [`${v}%`, "Return"]} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
                        <Bar dataKey="return" radius={[3, 3, 0, 0]}>
                          {(runDetail.summary_metrics.monthly_returns ?? []).map((entry: any, i: number) => (
                            <Cell key={i} fill={entry.return >= 0 ? "#22c55e" : "#ef4444"} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────
function ConfigForm({ config, onChange, onSubmit, isLoading }: any) {
  const field = (key: string, label: string, type = "text", step?: number) => (
    <div>
      <label className="text-xs text-gray-500 block mb-1">{label}</label>
      <input
        type={type}
        step={step}
        value={(config as any)[key]}
        onChange={(e) =>
          onChange({ ...config, [key]: type === "number" ? +e.target.value : e.target.value })
        }
        className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5
                   text-sm text-gray-300 focus:outline-none focus:border-blue-500"
      />
    </div>
  );

  return (
    <div className="space-y-3">
      {field("name", "Run Name")}
      {field("start_date", "Start Date", "date")}
      {field("end_date", "End Date", "date")}
      {field("initial_capital", "Initial Capital (VND)", "number", 1e6)}
      {field("min_score", "Min Score Filter", "number", 1)}
      {field("stop_loss_pct", "Stop Loss %", "number", 0.01)}
      {field("max_positions", "Max Positions", "number", 1)}

      <div>
        <label className="text-xs text-gray-500 block mb-1">Engine</label>
        <select
          value={config.engine}
          onChange={(e) => onChange({ ...config, engine: e.target.value })}
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5
                     text-sm text-gray-300 focus:outline-none focus:border-blue-500"
        >
          <option value="vectorbt">vectorbt (fast)</option>
          <option value="backtrader">backtrader (realistic)</option>
        </select>
      </div>

      <button
        onClick={onSubmit}
        disabled={isLoading}
        className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500
                   disabled:opacity-50 text-white rounded-lg py-2.5 text-sm font-semibold transition"
      >
        {isLoading ? (
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
        ) : (
          <Play size={14} />
        )}
        Run Backtest
      </button>
    </div>
  );
}

function RunListItem({ run, selected, onClick, onDelete }: any) {
  const statusColor =
    run.status === "DONE"
      ? "text-green-400"
      : run.status === "RUNNING"
      ? "text-blue-400"
      : run.status === "FAILED"
      ? "text-red-400"
      : "text-gray-500";

  return (
    <div
      onClick={onClick}
      className={`flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition
                  ${selected ? "bg-blue-950 border border-blue-800" : "hover:bg-gray-800"}`}
    >
      <div className="min-w-0">
        <div className="text-xs font-medium text-white truncate">{run.name}</div>
        <div className={`text-xs ${statusColor}`}>
          {run.status}
          {run.sharpe_ratio != null && ` · SR ${numeral(run.sharpe_ratio).format("0.00")}`}
        </div>
      </div>
      <button
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        className="text-gray-700 hover:text-red-400 transition ml-2 flex-shrink-0"
      >
        <Trash2 size={13} />
      </button>
    </div>
  );
}

function MetricTile({ label, value, positive, icon }: any) {
  const color =
    positive === undefined
      ? "text-white"
      : positive
      ? "text-green-400"
      : "text-red-400";
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-3">
      <div className="text-xs text-gray-500 flex items-center gap-1.5 mb-1">
        {icon} {label}
      </div>
      <div className={`text-lg font-bold font-mono ${color}`}>{value}</div>
    </div>
  );
}
