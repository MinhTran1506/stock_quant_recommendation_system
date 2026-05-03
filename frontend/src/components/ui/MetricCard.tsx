"use client";
/**
 * components/ui/MetricCard.tsx — KPI metric display card.
 */
import { ReactNode } from "react";
import { clsx } from "clsx";

const COLOR_MAP = {
  blue:   "bg-blue-950/50 border-blue-800/50 text-blue-400",
  green:  "bg-green-950/50 border-green-800/50 text-green-400",
  red:    "bg-red-950/50 border-red-800/50 text-red-400",
  yellow: "bg-yellow-950/50 border-yellow-800/50 text-yellow-400",
  purple: "bg-purple-950/50 border-purple-800/50 text-purple-400",
  cyan:   "bg-cyan-950/50 border-cyan-800/50 text-cyan-400",
};

interface Props {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  color?: keyof typeof COLOR_MAP;
  loading?: boolean;
}

export default function MetricCard({ title, value, subtitle, icon, color = "blue", loading }: Props) {
  const colorClass = COLOR_MAP[color];
  return (
    <div className={clsx("rounded-xl border p-4", colorClass)}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-gray-400">{title}</span>
        {icon && <span className="opacity-60">{icon}</span>}
      </div>
      {loading ? (
        <div className="h-7 w-24 bg-gray-700 rounded animate-pulse" />
      ) : (
        <div className="text-2xl font-bold text-white font-mono tracking-tight">
          {value}
        </div>
      )}
      {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
    </div>
  );
}
