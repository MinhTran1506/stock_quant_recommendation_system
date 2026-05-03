/**
 * src/app/layout.tsx — Root Next.js layout with providers.
 */
"use client";
import { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "react-hot-toast";
import Sidebar from "@/components/layout/Sidebar";
import { useAuth } from "@/store";
import "./globals.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <title>Vietnam HFT Platform</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="bg-gray-950 text-gray-100 antialiased">
        <QueryClientProvider client={queryClient}>
          <AppShell>{children}</AppShell>
          <Toaster
            position="top-right"
            toastOptions={{
              style: {
                background: "#1e293b",
                color: "#f1f5f9",
                border: "1px solid #334155",
                borderRadius: "12px",
                fontSize: "13px",
              },
            }}
          />
        </QueryClientProvider>
      </body>
    </html>
  );
}

function AppShell({ children }: { children: ReactNode }) {
  const { isAuthenticated } = useAuth();

  // Public routes that don't need the shell
  if (typeof window !== "undefined") {
    const path = window.location.pathname;
    if (path === "/login" || path === "/register") {
      return <>{children}</>;
    }
  }

  if (!isAuthenticated) {
    // Redirect to login — handled client-side
    if (typeof window !== "undefined") {
      window.location.href = "/login";
    }
    return null;
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 ml-0 md:ml-64 min-h-screen">
        {children}
      </main>
    </div>
  );
}
