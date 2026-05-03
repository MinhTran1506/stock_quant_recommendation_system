"use client";
/**
 * components/SectorHeatmap.tsx — Sector allocation heatmap.
 * Colors cells by average model score: green = bullish, red = bearish.
 */
import { useMemo } from "react";
import { Treemap, ResponsiveContainer, Tooltip } from "recharts";
import numeral from "numeral";

interface RankingEntry {
  ticker: string;
  sector?: string;
  score: number;
  predicted_return_5d?: number;
}

interface Props {
  data: RankingEntry[];
}

export default function SectorHeatmap({ data }: Props) {
  const treemapData = useMemo(() => {
    if (!data.length) return [];

    // Group by sector
    const sectors: Record<string, { scores: number[]; count: number }> = {};
    for (const stock of data) {
      const s = stock.sector || "Other";
      if (!sectors[s]) sectors[s] = { scores: [], count: 0 };
      sectors[s].scores.push(stock.score);
      sectors[s].count++;
    }

    return Object.entries(sectors).map(([name, { scores, count }]) => {
      const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
      return {
        name,
        size: count,
        avgScore: Math.round(avgScore * 10) / 10,
      };
    });
  }, [data]);

  const CustomContent = (props: any) => {
    const { x, y, width, height, name, avgScore } = props;
    if (width < 30 || height < 20) return null;

    const score = avgScore || 50;
    // Score → colour: 0=red, 50=neutral gray, 100=green
    const r = score < 50 ? 239 : Math.round(239 - (score - 50) * 4.26);
    const g = score > 50 ? 34  : Math.round(34 + (score) * 0.68);
    const b = 50;
    const bg = `rgba(${r},${g},${b},0.25)`;
    const border = `rgba(${r},${g},${b},0.6)`;

    return (
      <g>
        <rect x={x} y={y} width={width} height={height}
              fill={bg} stroke={border} strokeWidth={1} rx={4} />
        {width > 60 && (
          <>
            <text x={x + width / 2} y={y + height / 2 - 6}
                  textAnchor="middle" fill="#e2e8f0" fontSize={11} fontWeight={600}>
              {name.length > 12 ? name.slice(0, 11) + "…" : name}
            </text>
            <text x={x + width / 2} y={y + height / 2 + 10}
                  textAnchor="middle" fill="#94a3b8" fontSize={10}>
              {score}
            </text>
          </>
        )}
      </g>
    );
  };

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">Sector Heatmap</h3>
      {treemapData.length === 0 ? (
        <div className="h-40 flex items-center justify-center text-gray-600 text-sm">
          No data
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <Treemap
            data={treemapData}
            dataKey="size"
            content={<CustomContent />}
          >
            <Tooltip
              content={({ payload }) => {
                if (!payload?.length) return null;
                const d = payload[0]?.payload;
                return (
                  <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs">
                    <div className="font-semibold text-white">{d?.name}</div>
                    <div className="text-gray-400">Stocks: {d?.size}</div>
                    <div className="text-gray-400">Avg Score: {d?.avgScore}</div>
                  </div>
                );
              }}
            />
          </Treemap>
        </ResponsiveContainer>
      )}
      <div className="flex items-center justify-between mt-2 text-xs text-gray-600">
        <span className="text-red-500">◀ Bearish</span>
        <span>Score-weighted</span>
        <span className="text-green-500">Bullish ▶</span>
      </div>
    </div>
  );
}
