/**
 * src/utils/api.ts — Axios client with auto-auth, refresh, and error handling.
 */
import axios, { AxiosError, AxiosInstance } from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_PREFIX = "/api/v1";

// ── Axios instance ─────────────────────────────────────────────────────────
export const api: AxiosInstance = axios.create({
  baseURL: `${API_BASE}${API_PREFIX}`,
  headers: { "Content-Type": "application/json" },
  timeout: 30_000,
});

// ── Auth token injection ───────────────────────────────────────────────────
api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("access_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Token refresh on 401 ──────────────────────────────────────────────────
let isRefreshing = false;
let failedQueue: Array<{ resolve: Function; reject: Function }> = [];

const processQueue = (error: AxiosError | null, token: string | null = null) => {
  failedQueue.forEach(({ resolve, reject }) =>
    error ? reject(error) : resolve(token)
  );
  failedQueue = [];
};

api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as any;

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then((token) => {
          originalRequest.headers.Authorization = `Bearer ${token}`;
          return api(originalRequest);
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const refreshToken = localStorage.getItem("refresh_token");
        if (!refreshToken) throw new Error("No refresh token");

        const { data } = await axios.post(`${API_BASE}${API_PREFIX}/auth/refresh`, {
          refresh_token: refreshToken,
        });
        localStorage.setItem("access_token", data.access_token);
        localStorage.setItem("refresh_token", data.refresh_token);
        processQueue(null, data.access_token);
        originalRequest.headers.Authorization = `Bearer ${data.access_token}`;
        return api(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError as AxiosError, null);
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        window.location.href = "/login";
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }
    return Promise.reject(error);
  }
);

// ── Typed API calls ────────────────────────────────────────────────────────
export const authApi = {
  login: (email: string, password: string) =>
    api.post("/auth/login", new URLSearchParams({ username: email, password }), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    }),
  register: (email: string, password: string, fullName?: string) =>
    api.post("/auth/register", { email, password, full_name: fullName }),
  me: () => api.get("/auth/me"),
};

export const stocksApi = {
  list: (params?: Record<string, any>) => api.get("/stocks", { params }),
  get: (ticker: string) => api.get(`/stocks/${ticker}`),
  prices: (ticker: string, params?: Record<string, any>) =>
    api.get(`/stocks/${ticker}/prices`, { params }),
};

export const predictionsApi = {
  rankings: (params?: Record<string, any>) =>
    api.get("/predictions/rankings", { params }),
  ticker: (ticker: string, params?: Record<string, any>) =>
    api.get(`/predictions/${ticker}`, { params }),
  modelRegistry: (params?: Record<string, any>) =>
    api.get("/predictions/models/registry", { params }),
};

export const backtestApi = {
  submit: (payload: Record<string, any>) => api.post("/backtest", payload),
  list: (params?: Record<string, any>) => api.get("/backtest", { params }),
  get: (id: string) => api.get(`/backtest/${id}`),
  delete: (id: string) => api.delete(`/backtest/${id}`),
};

export const newsApi = {
  list: (params?: Record<string, any>) => api.get("/news", { params }),
  ticker: (ticker: string, params?: Record<string, any>) =>
    api.get(`/news/${ticker}`, { params }),
};

export const portfolioApi = {
  list: () => api.get("/portfolio"),
  create: (payload: Record<string, any>) => api.post("/portfolio", payload),
  get: (id: string) => api.get(`/portfolio/${id}`),
  positions: (id: string) => api.get(`/portfolio/${id}/positions`),
  orders: (id: string, params?: Record<string, any>) =>
    api.get(`/portfolio/${id}/orders`, { params }),
};
