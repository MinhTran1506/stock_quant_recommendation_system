"use client";
/**
 * pages/login/page.tsx — Authentication page.
 */
import { useState } from "react";
import { Zap, Eye, EyeOff } from "lucide-react";
import toast from "react-hot-toast";
import { authApi } from "@/utils/api";
import { useAuth } from "@/store";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<"login" | "register">("login");
  const { login } = useAuth();

  const handleSubmit = async () => {
    if (!email || !password) {
      toast.error("Please fill in all fields");
      return;
    }
    setIsLoading(true);
    try {
      if (mode === "login") {
        const tokenRes = await authApi.login(email, password);
        const { access_token, refresh_token } = tokenRes.data;
        const meRes = await authApi.me();
        const user = meRes.data;
        login(
          { id: user.id, email: user.email, fullName: user.full_name, isSuperuser: user.is_superuser },
          access_token,
          refresh_token,
        );
        toast.success("Welcome back!");
        window.location.href = "/dashboard";
      } else {
        await authApi.register(email, password);
        toast.success("Account created. Please log in.");
        setMode("login");
      }
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || "Authentication failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-blue-600 mb-4 shadow-lg shadow-blue-600/30">
            <Zap size={28} className="text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">HFT Platform</h1>
          <p className="text-gray-400 text-sm mt-1">Vietnam Equity Intelligence</p>
        </div>

        {/* Card */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 shadow-2xl">
          <h2 className="text-lg font-semibold text-white mb-6">
            {mode === "login" ? "Sign in to your account" : "Create account"}
          </h2>

          <div className="space-y-4">
            <div>
              <label className="text-xs text-gray-400 block mb-1.5">Email address</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3
                           text-sm text-gray-200 placeholder-gray-600
                           focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/20
                           transition"
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              />
            </div>

            <div>
              <label className="text-xs text-gray-400 block mb-1.5">Password</label>
              <div className="relative">
                <input
                  type={showPw ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3
                             text-sm text-gray-200 placeholder-gray-600 pr-10
                             focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/20
                             transition"
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                />
                <button
                  type="button"
                  onClick={() => setShowPw((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                >
                  {showPw ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>
          </div>

          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="mt-6 w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50
                       text-white font-semibold py-3 rounded-xl transition
                       shadow-lg shadow-blue-600/20 text-sm"
          >
            {isLoading ? (
              <div className="flex items-center justify-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                {mode === "login" ? "Signing in…" : "Creating account…"}
              </div>
            ) : mode === "login" ? (
              "Sign in"
            ) : (
              "Create account"
            )}
          </button>

          <div className="mt-4 text-center">
            <button
              onClick={() => setMode((m) => (m === "login" ? "register" : "login"))}
              className="text-sm text-gray-500 hover:text-gray-300 transition"
            >
              {mode === "login"
                ? "Don't have an account? Register"
                : "Already have an account? Sign in"}
            </button>
          </div>
        </div>

        {/* Disclaimer */}
        <p className="text-center text-xs text-gray-600 mt-4">
          Paper trading only · Live trading disabled pending regulatory clearance
        </p>
      </div>
    </div>
  );
}
