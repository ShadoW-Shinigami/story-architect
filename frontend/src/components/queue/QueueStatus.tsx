"use client";

import Link from "next/link";
import { useQueueStatus } from "@/hooks/useQueue";
import { Badge } from "@/components/ui/badge";
import { Loader2, Clock, CheckCircle } from "lucide-react";

interface QueueStatusProps {
  compact?: boolean;
}

export function QueueStatus({ compact = false }: QueueStatusProps) {
  const { data: status, isLoading } = useQueueStatus();

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>Loading...</span>
      </div>
    );
  }

  if (!status) {
    return null;
  }

  if (compact) {
    return (
      <div className="space-y-2">
        {status.is_processing ? (
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
            <span className="text-sm">Processing</span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <CheckCircle className="h-4 w-4 text-green-500" />
            <span className="text-sm">Idle</span>
          </div>
        )}
        {status.pending_count > 0 && (
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-yellow-500" />
            <span className="text-sm">{status.pending_count} in queue</span>
          </div>
        )}
        {status.current_task && (
          <Link
            href={`/projects/${status.current_task.session_id}`}
            className="text-xs text-primary hover:underline"
          >
            View current task
          </Link>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Status</span>
        <Badge variant={status.is_processing ? "default" : "secondary"}>
          {status.is_processing ? (
            <>
              <Loader2 className="mr-1 h-3 w-3 animate-spin" />
              Processing
            </>
          ) : (
            "Idle"
          )}
        </Badge>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Pending Tasks</span>
        <Badge variant="outline">{status.pending_count}</Badge>
      </div>

      {status.current_task && (
        <div className="pt-2 border-t">
          <span className="text-sm text-muted-foreground">Current Task</span>
          <Link
            href={`/projects/${status.current_task.session_id}`}
            className="block mt-1 text-sm text-primary hover:underline truncate"
          >
            {status.current_task.session_id}
          </Link>
        </div>
      )}
    </div>
  );
}
