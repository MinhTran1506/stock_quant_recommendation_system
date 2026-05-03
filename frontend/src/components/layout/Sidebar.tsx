"use client";
/**
 * components/layout/Sidebar.tsx — Fixed sidebar navigation.
 */
import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  LayoutDashboard, TrendingUp, FlaskConical,
  Briefcase, Newspaper, Settings, LogOut,
  Activity, ChevronRight, Zap,
} from "lucide-react";
import { clsx } from "clsx";
import { useAuth } from "@/store";

const NAV_ITEMS = [
  { href: "/dashboard",  label: "Dashboard",    icon: LayoutDashboard },
  { href: "/stocks",     label: "Stock Search",  icon: TrendingUp },
  { href: "/backtest",   label: "Backtest",      icon: FlaskConical },
  { href: "/portfolio",  label: "Portfolio",     icon: Briefcase },
  { href: "/news",       label: "News",          icon: Newspaper },
  { href: "/strategies", label: "Strategies",   icon: Activity },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  return (
    <aside className="fixed left-0 top-0 bottom-0 w-64 bg-gray-900 border-r border-gray-800
                      flex flex-col z-40 hidden md:flex">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-gray-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
            <Zap size={16} className="text-white" />
          </div>
          <div>
            <div className="text-sm font-bold text-white tracking-tight">HFT Platform</div>
            <div className="text-xs text-gray-500">Vietnam Equity</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname?.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-all group",
                active
                  ? "bg-blue-600/15 text-blue-400 border border-blue-600/20"
                  : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
              )}
            >
              <div className="flex items-center gap-3">
                <Icon size={16} className={active ? "text-blue-400" : "text-gray-500 group-hover:text-gray-300"} />
                {label}
              </div>
              {active && <ChevronRight size={12} className="text-blue-400" />}
            </Link>
          );
        })}
      </nav>

      {/* Regulatory notice */}
      <div className="mx-3 mb-3 px-3 py-2.5 rounded-lg bg-amber-950/40 border border-amber-800/40">
        <div className="text-xs text-amber-400 font-medium mb-0.5">Paper Trading Mode</div>
        <div className="text-xs text-amber-600 leading-relaxed">
          Live trading disabled pending regulatory clearance.
        </div>
      </div>

      {/* User */}
      <div className="px-3 pb-4 border-t border-gray-800 pt-3">
        <div className="flex items-center gap-3 px-2 mb-2">
          <div className="w-7 h-7 rounded-full bg-gray-700 flex items-center justify-center text-xs font-bold text-gray-300">
            {user?.email?.[0]?.toUpperCase() || "?"}
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-xs font-medium text-gray-300 truncate">
              {user?.fullName || user?.email}
            </div>
            {user?.isSuperuser && (
              <div className="text-xs text-blue-400">Superuser</div>
            )}
          </div>
        </div>
        <button
          onClick={logout}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-500
                     hover:text-red-400 hover:bg-red-950/20 transition"
        >
          <LogOut size={14} />
          Sign out
        </button>
      </div>
    </aside>
  );
}
