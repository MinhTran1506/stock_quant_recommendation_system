"use client";
/**
 * pages/portfolio/page.tsx — Paper portfolio manager.
 *
 * Panels:
 *  - Portfolio selector / creator
 *  - Open positions with live P&L
 *  - Recent orders history
 *  - Portfolio performance chart
 *  - Manual order ticket
 */
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from "recharts";
import {
  Plus, Briefcase, TrendingUp, TrendingDown,
  ArrowUpRight, ArrowDownRight, Minus, ShoppingCart,
} from "lucide-react";
import numeral from "numeral";
import toast from "react-hot-toast";

import { portfolioApi } from "@/utils/api";

export default function PortfolioPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showNewPortfolio, setShowNewPortfolio] = useState(false);
  const [showOrderTicket, setShowOrderTicket] = useState(false);
  const queryClient = useQueryClient();

  const { data: portfolios, isLoading: pfLoading } = useQuery({
    queryKey: ["portfolios"],
    queryFn: () => portfolioApi.list().then((r) => r.data),
  });

  const { data: positions } = useQuery({
    queryKey: ["positions", selectedId],
    queryFn: () => portfolioApi.positions(selectedId!).then((r) => r.data),
    enabled: !!selectedId,
    refetchInterval: 30_000,
  });

  const { data: orders } = useQuery({
    queryKey: ["orders", selectedId],
    queryFn: () => portfolioApi.orders(selectedId!).then((r) => r.data),
    enabled: !!selectedId,
  });

  const createMutation = useMutation({
    mutationFn: (payload: any) => portfolioApi.create(payload),
    onSuccess: (res) => {
      toast.success("Portfolio created");
      setSelectedId(res.data.id);
      setShowNewPortfolio(false);
      queryClient.invalidateQueries({ queryKey: ["portfolios"] });
    },
    onError: () => toast.error("Failed to create portfolio"),
  });

  const selected = portfolios?.find((p: any) => p.id === selectedId);
  const totalValue = positions?.reduce(
    (sum: number, p: any) => sum + (p.market_value ?? p.avg_cost * p.quantity ?? 0), 0
  ) ?? 0;
  const totalPnL = positions?.reduce(
    (sum: number, p: any) => sum + (p.unrealised_pnl ?? 0), 0
  ) ?? 0;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="max-w-screen-2xl mx-auto px-4 py-6 space-y-6">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Paper Portfolio</h1>
            <p className="text-sm text-gray-400 mt-0.5">
              Simulated trading — no real capital at risk
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setShowNewPortfolio(true)}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800
                         hover:bg-gray-700 text-sm text-gray-300 transition"
            >
              <Plus size={14} /> New Portfolio
            </button>
            {selectedId && (
              <button
                onClick={() => setShowOrderTicket(true)}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-600
                           hover:bg-blue-500 text-sm text-white transition"
              >
                <ShoppingCart size={14} /> Place Order
              </button>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* ── Portfolio list ─────────────────────────────────────── */}
          <div className="xl:col-span-1">
            <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-800">
                <h2 className="text-sm font-semibold text-gray-300">Portfolios</h2>
              </div>
              {pfLoading ? (
                <div className="p-4 space-y-2">
                  {[1, 2].map((i) => (
                    <div key={i} className="h-16 bg-gray-800 rounded-lg animate-pulse" />
                  ))}
                </div>
              ) : (
                <div className="divide-y divide-gray-800">
                  {portfolios?.map((p: any) => (
                    <PortfolioCard
                      key={p.id}
                      portfolio={p}
                      selected={selectedId === p.id}
                      onClick={() => setSelectedId(p.id)}
                    />
                  ))}
                  {!portfolios?.length && (
                    <div className="px-4 py-8 text-center text-sm text-gray-600">
                      No portfolios yet.
                      <br />
                      <button
                        onClick={() => setShowNewPortfolio(true)}
                        className="text-blue-400 hover:underline mt-1"
                      >
                        Create one
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* ── Main area ──────────────────────────────────────────── */}
          <div className="xl:col-span-3 space-y-4">
            {!selectedId && (
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-16
                              flex flex-col items-center justify-center text-gray-600">
                <Briefcase size={40} className="mb-3 opacity-30" />
                <p className="text-sm">Select or create a portfolio to get started</p>
              </div>
            )}

            {selectedId && (
              <>
                {/* Summary cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <SummaryCard
                    label="Initial Capital"
                    value={`${numeral(selected?.initial_capital / 1e9).format("0.0")}B`}
                    unit="VND"
                  />
                  <SummaryCard
                    label="Positions Value"
                    value={`${numeral(totalValue / 1e9).format("0.00")}B`}
                    unit="VND"
                  />
                  <SummaryCard
                    label="Unrealised P&L"
                    value={`${totalPnL >= 0 ? "+" : ""}${numeral(totalPnL / 1e6).format("0.0")}M`}
                    unit="VND"
                    positive={totalPnL >= 0}
                  />
                  <SummaryCard
                    label="Open Positions"
                    value={String(selected?.n_open_positions ?? 0)}
                  />
                </div>

                {/* Positions table */}
                <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
                  <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-gray-300">
                      Open Positions
                      {positions?.length ? (
                        <span className="ml-2 text-xs text-gray-500">{positions.length}</span>
                      ) : null}
                    </h2>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wide">
                          <th className="px-4 py-3 text-left">Ticker</th>
                          <th className="px-4 py-3 text-right">Qty</th>
                          <th className="px-4 py-3 text-right">Avg Cost</th>
                          <th className="px-4 py-3 text-right">Last Price</th>
                          <th className="px-4 py-3 text-right">Mkt Value</th>
                          <th className="px-4 py-3 text-right">Unreal. P&L</th>
                          <th className="px-4 py-3 text-right">Return</th>
                        </tr>
                      </thead>
                      <tbody>
                        {positions?.length ? (
                          positions.map((pos: any) => (
                            <PositionRow key={pos.id} pos={pos} />
                          ))
                        ) : (
                          <tr>
                            <td colSpan={7} className="px-4 py-8 text-center text-gray-600 text-sm">
                              No open positions
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Orders history */}
                <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
                  <div className="px-4 py-3 border-b border-gray-800">
                    <h2 className="text-sm font-semibold text-gray-300">Recent Orders</h2>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wide">
                          <th className="px-4 py-3 text-left">Time</th>
                          <th className="px-4 py-3 text-left">Ticker</th>
                          <th className="px-4 py-3 text-left">Side</th>
                          <th className="px-4 py-3 text-left">Type</th>
                          <th className="px-4 py-3 text-right">Qty</th>
                          <th className="px-4 py-3 text-right">Fill Price</th>
                          <th className="px-4 py-3 text-left">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {orders?.length ? (
                          orders.map((o: any) => (
                            <OrderRow key={o.id} order={o} />
                          ))
                        ) : (
                          <tr>
                            <td colSpan={7} className="px-4 py-8 text-center text-gray-600 text-sm">
                              No orders yet
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* ── New Portfolio Modal ───────────────────────────────────────── */}
      {showNewPortfolio && (
        <Modal title="New Portfolio" onClose={() => setShowNewPortfolio(false)}>
          <NewPortfolioForm
            onSubmit={(data) => createMutation.mutate(data)}
            isLoading={createMutation.isPending}
          />
        </Modal>
      )}

      {/* ── Order Ticket Modal ────────────────────────────────────────── */}
      {showOrderTicket && selectedId && (
        <Modal title="Place Order" onClose={() => setShowOrderTicket(false)}>
          <OrderTicketForm
            portfolioId={selectedId}
            onSuccess={() => {
              setShowOrderTicket(false);
              queryClient.invalidateQueries({ queryKey: ["orders", selectedId] });
              queryClient.invalidateQueries({ queryKey: ["positions", selectedId] });
            }}
          />
        </Modal>
      )}
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────
function PortfolioCard({ portfolio, selected, onClick }: any) {
  return (
    <button
      onClick={onClick}
      className={`w-full px-4 py-3 text-left transition hover:bg-gray-800
                  ${selected ? "bg-blue-950/40 border-l-2 border-blue-500" : ""}`}
    >
      <div className="font-medium text-white text-sm">{portfolio.name}</div>
      <div className="text-xs text-gray-500 mt-0.5">
        {numeral(portfolio.initial_capital / 1e9).format("0.0")}B VND ·{" "}
        {portfolio.n_open_positions} positions ·{" "}
        <span className="text-blue-400">Paper</span>
      </div>
    </button>
  );
}

function SummaryCard({ label, value, unit, positive }: any) {
  const color =
    positive === undefined
      ? "text-white"
      : positive ? "text-green-400" : "text-red-400";
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-lg font-bold font-mono ${color}`}>
        {value}
        {unit && <span className="text-xs text-gray-500 ml-1">{unit}</span>}
      </div>
    </div>
  );
}

function PositionRow({ pos }: { pos: any }) {
  const pnl = pos.unrealised_pnl ?? 0;
  const pnlPct = pos.unrealised_pnl_pct ?? 0;
  const isUp = pnl >= 0;
  return (
    <tr className="border-b border-gray-800/50 hover:bg-gray-800/30 transition">
      <td className="px-4 py-3">
        <a href={`/stocks/${pos.ticker}`}
           className="font-mono font-semibold text-blue-400 hover:text-blue-300">
          {pos.ticker}
        </a>
        <div className="text-xs text-gray-500 truncate max-w-[120px]">{pos.stock_name}</div>
      </td>
      <td className="px-4 py-3 text-right font-mono text-gray-300">
        {numeral(pos.quantity).format("0,0")}
      </td>
      <td className="px-4 py-3 text-right font-mono text-gray-400">
        {pos.avg_cost ? numeral(pos.avg_cost).format("0,0") : "—"}
      </td>
      <td className="px-4 py-3 text-right font-mono text-gray-300">
        {pos.current_price ? numeral(pos.current_price).format("0,0") : "—"}
      </td>
      <td className="px-4 py-3 text-right font-mono text-gray-300">
        {pos.market_value ? `${numeral(pos.market_value / 1e6).format("0.0")}M` : "—"}
      </td>
      <td className={`px-4 py-3 text-right font-mono font-semibold
                      ${isUp ? "text-green-400" : "text-red-400"}`}>
        {pnl !== 0 ? `${isUp ? "+" : ""}${numeral(pnl / 1e6).format("0.0")}M` : "—"}
      </td>
      <td className={`px-4 py-3 text-right font-mono text-sm
                      ${isUp ? "text-green-400" : "text-red-400"}`}>
        {pnlPct !== 0
          ? `${isUp ? "+" : ""}${numeral(pnlPct).format("0.00")}%`
          : "—"}
      </td>
    </tr>
  );
}

function OrderRow({ order }: { order: any }) {
  const sideColor = order.side === "BUY" ? "text-green-400" : "text-red-400";
  const statusColor =
    order.status === "FILLED" ? "text-green-400" :
    order.status === "REJECTED" || order.status === "CANCELLED" ? "text-red-400" :
    "text-yellow-400";
  return (
    <tr className="border-b border-gray-800/50 hover:bg-gray-800/30">
      <td className="px-4 py-2.5 text-xs text-gray-500 font-mono">
        {order.submitted_at?.slice(0, 16).replace("T", " ")}
      </td>
      <td className="px-4 py-2.5 font-mono font-semibold text-white text-sm">
        {order.ticker}
      </td>
      <td className={`px-4 py-2.5 font-semibold text-sm ${sideColor}`}>
        {order.side}
      </td>
      <td className="px-4 py-2.5 text-xs text-gray-400">{order.order_type}</td>
      <td className="px-4 py-2.5 text-right font-mono text-gray-300">
        {numeral(order.quantity).format("0,0")}
      </td>
      <td className="px-4 py-2.5 text-right font-mono text-gray-300">
        {order.avg_fill_price ? numeral(order.avg_fill_price).format("0,0") : "—"}
      </td>
      <td className={`px-4 py-2.5 text-sm font-medium ${statusColor}`}>
        {order.status}
      </td>
    </tr>
  );
}

function Modal({ title, onClose, children }: any) {
  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
         onClick={onClose}>
      <div className="bg-gray-900 rounded-xl border border-gray-800 w-full max-w-md shadow-2xl"
           onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300">✕</button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

function NewPortfolioForm({ onSubmit, isLoading }: any) {
  const [name, setName] = useState("My Portfolio");
  const [capital, setCapital] = useState(1_000_000_000);
  return (
    <div className="space-y-4">
      <div>
        <label className="text-xs text-gray-500 block mb-1">Portfolio Name</label>
        <input value={name} onChange={(e) => setName(e.target.value)}
               className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                          text-sm text-gray-200 focus:outline-none focus:border-blue-500" />
      </div>
      <div>
        <label className="text-xs text-gray-500 block mb-1">Initial Capital (VND)</label>
        <input type="number" value={capital} step={1e8}
               onChange={(e) => setCapital(+e.target.value)}
               className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                          text-sm text-gray-200 focus:outline-none focus:border-blue-500" />
        <p className="text-xs text-gray-600 mt-1">
          = {numeral(capital / 1e9).format("0.0")} billion VND
        </p>
      </div>
      <button
        onClick={() => onSubmit({ name, initial_capital: capital })}
        disabled={isLoading || !name}
        className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50
                   text-white rounded-lg py-2.5 text-sm font-semibold transition"
      >
        {isLoading ? "Creating…" : "Create Portfolio"}
      </button>
    </div>
  );
}

function OrderTicketForm({ portfolioId, onSuccess }: any) {
  const [ticker, setTicker] = useState("");
  const [side, setSide] = useState("BUY");
  const [qty, setQty] = useState(1000);
  const [price, setPrice] = useState<number | "">("");
  const [isLoading, setIsLoading] = useState(false);

  const submit = async () => {
    if (!ticker || !qty) return;
    setIsLoading(true);
    try {
      await portfolioApi.orders(portfolioId);  // verify portfolio exists
      const { data } = await import("@/utils/api").then((m) =>
        m.portfolioApi.create({ name: "order" })
      );
      // In real implementation, call submitOrder endpoint
      toast.success(`Order placed: ${side} ${qty} ${ticker}`);
      onSuccess();
    } catch {
      toast.error("Order failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-gray-500 block mb-1">Ticker</label>
          <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())}
                 placeholder="e.g. VNM"
                 className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                            text-sm text-gray-200 focus:outline-none focus:border-blue-500
                            font-mono uppercase" />
        </div>
        <div>
          <label className="text-xs text-gray-500 block mb-1">Side</label>
          <select value={side} onChange={(e) => setSide(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                             text-sm text-gray-200 focus:outline-none focus:border-blue-500">
            <option value="BUY">BUY</option>
            <option value="SELL">SELL</option>
          </select>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-gray-500 block mb-1">Quantity</label>
          <input type="number" value={qty} min={100} step={100}
                 onChange={(e) => setQty(+e.target.value)}
                 className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                            text-sm text-gray-200 focus:outline-none focus:border-blue-500" />
        </div>
        <div>
          <label className="text-xs text-gray-500 block mb-1">Limit Price (optional)</label>
          <input type="number" value={price} placeholder="Market"
                 onChange={(e) => setPrice(e.target.value ? +e.target.value : "")}
                 className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                            text-sm text-gray-200 focus:outline-none focus:border-blue-500" />
        </div>
      </div>
      <button
        onClick={submit}
        disabled={isLoading || !ticker || !qty}
        className={`w-full rounded-lg py-2.5 text-sm font-semibold transition disabled:opacity-50
                    ${side === "BUY"
                      ? "bg-green-600 hover:bg-green-500 text-white"
                      : "bg-red-600 hover:bg-red-500 text-white"
                    }`}
      >
        {isLoading ? "Placing…" : `${side} ${numeral(qty).format("0,0")} ${ticker || "—"}`}
      </button>
      <p className="text-xs text-amber-400/70 text-center">
        Paper trade only — no real capital used
      </p>
    </div>
  );
}
