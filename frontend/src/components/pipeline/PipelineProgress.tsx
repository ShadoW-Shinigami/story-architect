"use client";

import { useProgressStore } from "@/stores/progressStore";
import { cn } from "@/lib/utils";
import {
  CheckCircle,
  Circle,
  Loader2,
  XCircle,
  AlertCircle,
  RotateCcw,
} from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";

interface AgentInfo {
  id: string;
  name: string;
  description: string;
  phase: number;
}

const AGENTS: AgentInfo[] = [
  { id: "agent_1", name: "Screenplay", description: "Generate screenplay", phase: 1 },
  { id: "agent_2", name: "Scene Breakdown", description: "Break into scenes", phase: 1 },
  { id: "agent_3", name: "Shot Breakdown", description: "Define shots", phase: 1 },
  { id: "agent_4", name: "Grouping", description: "Group shots", phase: 1 },
  { id: "agent_5", name: "Characters", description: "Create characters", phase: 2 },
  { id: "agent_6", name: "Parent Images", description: "Generate parent shots", phase: 2 },
  { id: "agent_7", name: "Parent Verify", description: "Verify parent images", phase: 2 },
  { id: "agent_8", name: "Child Images", description: "Generate child shots", phase: 2 },
  { id: "agent_9", name: "Child Verify", description: "Verify child images", phase: 2 },
  { id: "agent_10", name: "Video Gen", description: "Generate videos", phase: 3 },
  { id: "agent_11", name: "Editor", description: "Edit final video", phase: 3 },
];

interface AgentStepProps {
  agent: AgentInfo;
  status: "pending" | "in_progress" | "completed" | "failed" | "retrying";
  progress?: number;
  message?: string;
  current?: number;
  total?: number;
  isActive: boolean;
  hasOutput: boolean;
  isSelected: boolean;
  onClick?: () => void;
  onRetryFromAgent?: (agentId: string) => void;
  canRetry?: boolean;
}

function AgentStep({
  agent,
  status,
  progress,
  message,
  current,
  total,
  isActive,
  hasOutput,
  isSelected,
  onClick,
  onRetryFromAgent,
  canRetry,
}: AgentStepProps) {
  const getIcon = () => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "in_progress":
        return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
      case "failed":
        return <XCircle className="h-5 w-5 text-red-500" />;
      case "retrying":
        return <AlertCircle className="h-5 w-5 text-yellow-500 animate-pulse" />;
      default:
        return <Circle className="h-5 w-5 text-muted-foreground" />;
    }
  };

  const isClickable = hasOutput && onClick;

  return (
    <div
      onClick={isClickable ? onClick : undefined}
      className={cn(
        "flex items-start gap-3 p-3 rounded-lg transition-colors",
        isActive && "bg-muted/50",
        status === "failed" && "bg-red-50 dark:bg-red-950/20",
        isSelected && "bg-blue-50 dark:bg-blue-950/30 border-l-2 border-blue-500",
        isClickable && "cursor-pointer hover:bg-muted/70"
      )}
    >
      <div className="pt-0.5">{getIcon()}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <h4
            className={cn(
              "font-medium",
              status === "pending" && "text-muted-foreground"
            )}
          >
            {agent.name}
          </h4>
          <div className="flex items-center gap-2">
            {status === "in_progress" && progress !== undefined && (
              <span className="text-sm text-muted-foreground">
                {Math.round(progress * 100)}%
              </span>
            )}
            {canRetry && onRetryFromAgent && (
              <Button
                size="sm"
                variant="ghost"
                className="h-6 w-6 p-0"
                title={`Run from ${agent.name}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onRetryFromAgent(agent.id);
                }}
              >
                <RotateCcw className="h-3 w-3" />
              </Button>
            )}
          </div>
        </div>
        <p className="text-sm text-muted-foreground">{agent.description}</p>
        {status === "in_progress" && (
          <div className="mt-2 space-y-1">
            {progress !== undefined && (
              <Progress value={progress * 100} className="h-1" />
            )}
            {message && (
              <p className="text-xs text-blue-600 dark:text-blue-400">
                {message}
              </p>
            )}
            {current !== undefined && total !== undefined && (
              <p className="text-xs text-muted-foreground">
                {current} / {total}
              </p>
            )}
          </div>
        )}
        {status === "failed" && (
          <p className="text-xs text-red-600 dark:text-red-400 mt-1">
            Agent failed - check logs for details
          </p>
        )}
      </div>
    </div>
  );
}

interface PipelineProgressProps {
  sessionId: string;
  currentAgent?: string;
  sessionStatus?: string;
  availableOutputs?: Set<string>;
  selectedAgent?: string;
  onAgentClick?: (agentId: string) => void;
  onRetryFromAgent?: (agentId: string) => void;
  canRetry?: boolean;
}

export function PipelineProgress({
  sessionId,
  currentAgent,
  sessionStatus,
  availableOutputs = new Set(),
  selectedAgent,
  onAgentClick,
  onRetryFromAgent,
  canRetry = false,
}: PipelineProgressProps) {
  const { agents, overallProgress, currentMessage } = useProgressStore();

  const getAgentStatus = (agentId: string): AgentStepProps["status"] => {
    // Check store first for real-time updates
    const agentProgress = agents[agentId];
    if (agentProgress) {
      if (agentProgress.progress >= 1) return "completed";
      if (agentProgress.progress > 0) return "in_progress";
    }

    // Fall back to session-based status
    if (!currentAgent) return "pending";

    const currentIdx = AGENTS.findIndex((a) => a.id === currentAgent);
    const agentIdx = AGENTS.findIndex((a) => a.id === agentId);

    if (agentIdx < currentIdx) return "completed";
    if (agentIdx === currentIdx) {
      if (sessionStatus === "failed") return "failed";
      if (sessionStatus === "completed") return "completed";
      return "in_progress";
    }
    return "pending";
  };

  const getAgentProgress = (agentId: string) => {
    return agents[agentId];
  };

  // Group agents by phase
  const phases = [
    { num: 1, label: "Script Analysis", agents: AGENTS.filter((a) => a.phase === 1) },
    { num: 2, label: "Image Generation", agents: AGENTS.filter((a) => a.phase === 2) },
    { num: 3, label: "Video Production", agents: AGENTS.filter((a) => a.phase === 3) },
  ];

  return (
    <div className="space-y-6">
      {/* Overall progress */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">Pipeline Progress</h3>
          <span className="text-sm text-muted-foreground">
            {Math.round(overallProgress * 100)}%
          </span>
        </div>
        <Progress value={overallProgress * 100} className="h-2" />
        {currentMessage && (
          <p className="text-sm text-muted-foreground">{currentMessage}</p>
        )}
      </div>

      {/* Phase-based agent list */}
      {phases.map((phase) => (
        <div key={phase.num} className="space-y-2">
          <h4 className="text-sm font-medium text-muted-foreground">
            Phase {phase.num}: {phase.label}
          </h4>
          <div className="border rounded-lg divide-y">
            {phase.agents.map((agent) => {
              const status = getAgentStatus(agent.id);
              const progress = getAgentProgress(agent.id);
              const hasOutput = availableOutputs.has(agent.id);
              return (
                <AgentStep
                  key={agent.id}
                  agent={agent}
                  status={status}
                  progress={progress?.progress}
                  message={progress?.message}
                  current={progress?.current}
                  total={progress?.total}
                  isActive={agent.id === currentAgent}
                  hasOutput={hasOutput}
                  isSelected={agent.id === selectedAgent}
                  onClick={onAgentClick ? () => onAgentClick(agent.id) : undefined}
                  onRetryFromAgent={onRetryFromAgent}
                  canRetry={canRetry}
                />
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
