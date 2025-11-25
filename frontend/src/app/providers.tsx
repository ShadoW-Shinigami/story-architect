"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, useEffect } from "react";
import { useGlobalWebSocket } from "@/hooks/useWebSocket";

// Component to initialize WebSocket connection on app mount
function WebSocketInitializer({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    console.log("[WebSocketInitializer] Component mounted");
  }, []);

  useGlobalWebSocket();
  return <>{children}</>;
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000, // 1 minute
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <WebSocketInitializer>{children}</WebSocketInitializer>
    </QueryClientProvider>
  );
}
