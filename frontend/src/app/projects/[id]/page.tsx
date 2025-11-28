"use client";

import { useEffect, useState, useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
import { useSession, useStartPipeline, useResumePipeline, useCancelPipeline } from "@/hooks/useSessions";
import { useAgentOutputs } from "@/hooks/useAgentOutputs";
import { useWebSocket } from "@/hooks/useWebSocket";
import { Header } from "@/components/shared/Header";
import { Sidebar } from "@/components/shared/Sidebar";
import { PipelineProgress } from "@/components/pipeline/PipelineProgress";
import { OutputViewer } from "@/components/output/OutputViewer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ArrowLeft,
  Play,
  RotateCcw,
  Loader2,
  Calendar,
  Clock,
  StopCircle,
} from "lucide-react";
import { SessionStatus } from "@/types/session";

const statusColors: Record<SessionStatus, string> = {
  pending: "bg-gray-500",
  queued: "bg-yellow-500",
  in_progress: "bg-blue-500",
  completed: "bg-green-500",
  failed: "bg-red-500",
  cancelled: "bg-gray-400",
};

const statusLabels: Record<SessionStatus, string> = {
  pending: "Pending",
  queued: "Queued",
  in_progress: "Processing",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

export default function ProjectDetailPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.id as string;

  const { data: session, isLoading: sessionLoading } = useSession(sessionId);
  const { data: outputs, isLoading: outputsLoading } = useAgentOutputs(sessionId);
  const startPipeline = useStartPipeline();
  const resumePipeline = useResumePipeline();
  const cancelPipeline = useCancelPipeline();

  // State for selected agent in sidebar
  const [selectedAgent, setSelectedAgent] = useState<string | undefined>();

  // Compute set of agents with available outputs
  const availableOutputs = useMemo(() => {
    if (!outputs) return new Set<string>();
    return new Set(
      outputs
        .filter((o) => o.status === "completed" && o.output_data)
        .map((o) => o.agent_name)
    );
  }, [outputs]);

  // Get the selected output data
  const selectedOutput = useMemo(() => {
    if (!outputs || !selectedAgent) return null;
    return outputs.find((o) => o.agent_name === selectedAgent);
  }, [outputs, selectedAgent]);

  // Auto-select first available output when outputs change
  useEffect(() => {
    if (availableOutputs.size > 0 && !selectedAgent) {
      // Select the first completed agent
      const firstAgent = Array.from(availableOutputs)[0];
      setSelectedAgent(firstAgent);
    }
  }, [availableOutputs, selectedAgent]);

  // Connect to WebSocket for real-time updates
  useWebSocket(sessionId);

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const handleStart = async () => {
    try {
      await startPipeline.mutateAsync({
        sessionId,
        startAgent: session?.start_agent || "agent_1",
      });
    } catch (error) {
      console.error("Failed to start pipeline:", error);
    }
  };

  const handleResume = async (fromAgent: string) => {
    try {
      await startPipeline.mutateAsync({
        sessionId,
        startAgent: fromAgent,
      });
    } catch (error) {
      console.error("Failed to resume pipeline:", error);
    }
  };

  const handleRetryFromAgent = async (fromAgent: string) => {
    try {
      await resumePipeline.mutateAsync({
        sessionId,
        fromAgent,
      });
    } catch (error) {
      console.error("Failed to retry from agent:", error);
    }
  };

  const handleCancel = async () => {
    try {
      await cancelPipeline.mutateAsync(sessionId);
    } catch (error) {
      console.error("Failed to cancel pipeline:", error);
    }
  };

  // Can retry when session is not actively running
  const canRetry = session ? ["completed", "failed", "pending", "cancelled"].includes(session.status) : false;

  if (sessionLoading) {
    return (
      <div className="flex min-h-screen bg-background">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <Header title="Loading..." />
          <main className="flex-1 flex items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </main>
        </div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="flex min-h-screen bg-background">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <Header title="Project Not Found" />
          <main className="flex-1 p-6">
            <Card className="p-8 text-center">
              <p className="text-muted-foreground mb-4">
                This project could not be found
              </p>
              <Button onClick={() => router.push("/projects")}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Projects
              </Button>
            </Card>
          </main>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header title={session.name || `Project ${session.id.slice(0, 8)}`} />
        <main className="flex-1 p-6">
          {/* Back button */}
          <Button
            variant="ghost"
            className="mb-4"
            onClick={() => router.push("/projects")}
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Projects
          </Button>

          {/* Project header */}
          <div className="flex items-start justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold">
                {session.name || `Project ${session.id.slice(0, 8)}`}
              </h2>
              <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Calendar className="h-4 w-4" />
                  <span>Created {formatDate(session.created_at)}</span>
                </div>
                {session.updated_at && (
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    <span>Updated {formatDate(session.updated_at)}</span>
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge className={statusColors[session.status]}>
                {statusLabels[session.status]}
              </Badge>
              {session.status === "pending" && (
                <Button onClick={handleStart} disabled={startPipeline.isPending}>
                  {startPipeline.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Start Pipeline
                </Button>
              )}
              {session.status === "in_progress" && (
                <Button
                  variant="destructive"
                  onClick={handleCancel}
                  disabled={cancelPipeline.isPending}
                >
                  {cancelPipeline.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <StopCircle className="h-4 w-4 mr-2" />
                  )}
                  Stop Pipeline
                </Button>
              )}
              {session.status === "failed" && session.current_agent && (
                <Button
                  onClick={() => handleResume(session.current_agent!)}
                  disabled={startPipeline.isPending}
                >
                  {startPipeline.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <RotateCcw className="h-4 w-4 mr-2" />
                  )}
                  Retry
                </Button>
              )}
            </div>
          </div>

          {/* Main content */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Progress panel */}
            <div className="lg:col-span-1">
              <Card>
                <CardHeader>
                  <CardTitle>Progress</CardTitle>
                </CardHeader>
                <CardContent>
                  <PipelineProgress
                    sessionId={sessionId}
                    currentAgent={session.current_agent || undefined}
                    sessionStatus={session.status}
                    availableOutputs={availableOutputs}
                    selectedAgent={selectedAgent}
                    onAgentClick={setSelectedAgent}
                    onRetryFromAgent={handleRetryFromAgent}
                    canRetry={canRetry}
                  />
                </CardContent>
              </Card>
            </div>

            {/* Output panel */}
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>
                    {selectedAgent
                      ? `Output: ${selectedAgent.replace("_", " ").toUpperCase()}`
                      : "Outputs"}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {outputsLoading ? (
                    <div className="flex items-center justify-center h-64">
                      <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                    </div>
                  ) : availableOutputs.size === 0 ? (
                    <div className="text-center py-12 text-muted-foreground">
                      <p>No outputs yet</p>
                      <p className="text-sm mt-1">
                        Start the pipeline to generate outputs
                      </p>
                    </div>
                  ) : selectedOutput && selectedOutput.output_data ? (
                    <OutputViewer
                      agentName={selectedOutput.agent_name}
                      output={selectedOutput.output_data}
                      sessionId={sessionId}
                    />
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">
                      <p>Select an agent from the progress panel to view its output</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Input preview */}
              <Card className="mt-6">
                <CardHeader>
                  <CardTitle>Input</CardTitle>
                </CardHeader>
                <CardContent>
                  <pre className="whitespace-pre-wrap text-sm bg-muted p-4 rounded-lg max-h-64 overflow-auto">
                    {session.input_data}
                  </pre>
                </CardContent>
              </Card>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
