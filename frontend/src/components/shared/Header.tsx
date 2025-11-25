"use client";

import { useConnectionStore } from "@/stores/connectionStore";
import { cn } from "@/lib/utils";
import { Wifi, WifiOff, Loader2 } from "lucide-react";

interface HeaderProps {
  title: string;
}

export function Header({ title }: HeaderProps) {
  const connectionStatus = useConnectionStore((s) => s.status);

  return (
    <header className="h-14 border-b bg-background flex items-center justify-between px-6">
      <h1 className="text-lg font-semibold">{title}</h1>

      <div className="flex items-center gap-4">
        {/* Connection status indicator */}
        <div
          className={cn(
            "flex items-center gap-2 text-sm",
            connectionStatus === "connected" && "text-green-600",
            connectionStatus === "connecting" && "text-yellow-600",
            connectionStatus === "disconnected" && "text-gray-400",
            connectionStatus === "error" && "text-red-600"
          )}
        >
          {connectionStatus === "connected" && (
            <>
              <Wifi className="h-4 w-4" />
              <span>Connected</span>
            </>
          )}
          {connectionStatus === "connecting" && (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Connecting...</span>
            </>
          )}
          {connectionStatus === "disconnected" && (
            <>
              <WifiOff className="h-4 w-4" />
              <span>Disconnected</span>
            </>
          )}
          {connectionStatus === "error" && (
            <>
              <WifiOff className="h-4 w-4" />
              <span>Connection Error</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
