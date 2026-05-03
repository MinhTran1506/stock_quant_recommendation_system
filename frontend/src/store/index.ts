/**
 * store/index.ts — Zustand global state store.
 *
 * Slices:
 *  - authSlice: user session, JWT tokens
 *  - portfolioSlice: active portfolio selection
 *  - settingsSlice: UI preferences
 */
import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { persist } from "zustand/middleware";

// ─── Types ────────────────────────────────────────────────────────────────────
interface User {
  id: string;
  email: string;
  fullName?: string;
  isSuperuser: boolean;
}

interface AuthSlice {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  login: (user: User, accessToken: string, refreshToken: string) => void;
  logout: () => void;
}

interface PortfolioSlice {
  activePortfolioId: string | null;
  setActivePortfolio: (id: string | null) => void;
}

interface SettingsSlice {
  theme: "dark" | "light";
  defaultHorizon: number;
  defaultExchange: string;
  showSentiment: boolean;
  setHorizon: (h: number) => void;
  setExchange: (e: string) => void;
  toggleSentiment: () => void;
}

type Store = AuthSlice & PortfolioSlice & SettingsSlice;

// ─── Store ────────────────────────────────────────────────────────────────────
export const useStore = create<Store>()(
  persist(
    immer((set) => ({
      // ── Auth ──────────────────────────────────────────────────────────
      user: null,
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,

      login: (user, accessToken, refreshToken) =>
        set((state) => {
          state.user = user;
          state.accessToken = accessToken;
          state.refreshToken = refreshToken;
          state.isAuthenticated = true;
          // Also persist to localStorage for axios interceptor
          if (typeof window !== "undefined") {
            localStorage.setItem("access_token", accessToken);
            localStorage.setItem("refresh_token", refreshToken);
          }
        }),

      logout: () =>
        set((state) => {
          state.user = null;
          state.accessToken = null;
          state.refreshToken = null;
          state.isAuthenticated = false;
          if (typeof window !== "undefined") {
            localStorage.removeItem("access_token");
            localStorage.removeItem("refresh_token");
          }
        }),

      // ── Portfolio ─────────────────────────────────────────────────────
      activePortfolioId: null,
      setActivePortfolio: (id) =>
        set((state) => {
          state.activePortfolioId = id;
        }),

      // ── Settings ──────────────────────────────────────────────────────
      theme: "dark",
      defaultHorizon: 5,
      defaultExchange: "ALL",
      showSentiment: true,

      setHorizon: (h) =>
        set((state) => {
          state.defaultHorizon = h;
        }),

      setExchange: (e) =>
        set((state) => {
          state.defaultExchange = e;
        }),

      toggleSentiment: () =>
        set((state) => {
          state.showSentiment = !state.showSentiment;
        }),
    })),
    {
      name: "hft-platform-store",
      // Only persist non-sensitive state
      partialize: (state) => ({
        activePortfolioId: state.activePortfolioId,
        defaultHorizon: state.defaultHorizon,
        defaultExchange: state.defaultExchange,
        showSentiment: state.showSentiment,
      }),
    }
  )
);

// ─── Selector hooks ───────────────────────────────────────────────────────────
export const useAuth = () =>
  useStore((s) => ({
    user: s.user,
    isAuthenticated: s.isAuthenticated,
    login: s.login,
    logout: s.logout,
  }));

export const usePortfolioStore = () =>
  useStore((s) => ({
    activePortfolioId: s.activePortfolioId,
    setActivePortfolio: s.setActivePortfolio,
  }));

export const useSettings = () =>
  useStore((s) => ({
    defaultHorizon: s.defaultHorizon,
    defaultExchange: s.defaultExchange,
    showSentiment: s.showSentiment,
    setHorizon: s.setHorizon,
    setExchange: s.setExchange,
    toggleSentiment: s.toggleSentiment,
  }));
