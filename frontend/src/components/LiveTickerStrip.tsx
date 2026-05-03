"use client";
/**
 * components/LiveTickerStrip.tsx — Scrolling live price ticker.
 * Connects to WebSocket for each watchlist ticker and displays real-time prices.
 */
import { useEffect, useRef, useState } from "react";
import { TrendingUp, TrendingDown, Wifi, WifiOff } from "lucide-react";
import numeral from "numeral";

const WATCHLIST = ["VNM", "VIC", "HPG", "FPT", "TCB", "VPB", "MWG", "MSN", "GAS", "REE"];
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

interface TickData {
  ticker: string;
  price: number;
  change: number;
  changePct: number;
  volume: number;
  ts: string;
}

export default function LiveTickerStrip() {
  const [ticks, setTicks] = useState<Record<string, TickData>>({});
  const [connected, setConnected] = useState(false);
  const socketsRef = useRef<Record<string, WebSocket>>({});

  useEffect(() => {
    // Connect to one WebSocket per ticker
    WATCHLIST.forEach((ticker) => {
      const connect = () => {
        try {
          const ws = new WebSocket(`${WS_BASE}/ws/prices/${ticker}`);
          socketsRef.current[ticker] = ws;

          ws.onopen = () => setConnected(true);

          ws.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              setTicks((prev) => ({
                ...prev,
                [ticker]: {
                  ticker,
                  price: data.close || data.price || 0,
                  change: data.change || 0,
                  changePct: data.change_pct || 0,
                  volume: data.volume || 0,
                  ts: data.ts || new Date().toISOString(),
                },
              }));
            } catch {}
          };

          ws.onclose = () => {
            // Reconnect after 3 seconds
            setTimeout(connect, 3000);
          };

          ws.onerror = () => ws.close();
        } catch {}
      };
      connect();
    });

    return () => {
      Object.values(socketsRef.current).forEach((ws) => ws.close());
    };
  }, []);

  // Duplicate list for seamless scroll loop
  const displayList = [...WATCHLIST, ...WATCHLIST];

  return (
    <div className="bg-gray-900/80 border-b border-gray-800 backdrop-blur-sm">
      <div className="flex items-center">
        {/* Status indicator */}
        <div className="flex-shrink-0 flex items-center gap-1.5 px-4 border-r border-gray-800 py-2">
          {connected ? (
            <Wifi size={12} className="text-green-400" />
          ) : (
            <WifiOff size={12} className="text-gray-600" />
          )}
          <span className="text-xs text-gray-500 hidden sm:block">Live</span>
        </div>

        {/* Scrolling ticker */}
        <div className="flex-1 overflow-hidden relative">
          <div
            className="flex gap-6 px-4 py-2 animate-ticker"
            style={{ whiteSpace: "nowrap" }}
          >
            {displayList.map((ticker, i) => {
              const tick = ticks[ticker];
              const pct = tick?.changePct ?? 0;
              const isPos = pct >= 0;
              return (
                <a
                  key={`${ticker}-${i}`}
                  href={`/stocks/${ticker}`}
                  className="inline-flex items-center gap-2 hover:opacity-80 transition"
                >
                  <span className="font-mono font-semibold text-white text-xs">
                    {ticker}
                  </span>
                  {tick ? (
                    <>
                      <span className="text-xs text-gray-300">
                        {numeral(tick.price).format("0,0")}
                      </span>
                      <span className={`text-xs flex items-center gap-0.5 ${isPos ? "text-green-400" : "text-red-400"}`}>
                        {isPos ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
                        {isPos ? "+" : ""}{numeral(pct).format("0.00")}%
                      </span>
                    </>
                  ) : (
                    <span className="text-xs text-gray-600">—</span>
                  )}
                </a>
              );
            })}
          </div>
        </div>
      </div>

      {/* Ticker animation CSS */}
      <style jsx>{`
        @keyframes ticker {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        .animate-ticker {
          animation: ticker 40s linear infinite;
        }
        .animate-ticker:hover {
          animation-play-state: paused;
        }
      `}</style>
    </div>
  );
}
